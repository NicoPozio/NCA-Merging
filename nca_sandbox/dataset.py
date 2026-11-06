"""Dataset generati al volo su GPU, più validazione e test salvati con seed fissi.

Dataset (cambia solo quali regole sono attive, scene ed eventi hanno sempre la stessa distribuzione):
    base            solo base
    T1 ... T8       base + una regola
    joint_single    per ogni scena base + una regola a caso
    joint_full      base + tutte le regole
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .config import N_RULES, RULE_IDS, SimParams
from .events import EventParams, batch_event_masks, sample_events
from .scenes import SceneParams, sample_scenes
from .simulator import Simulator

DATASETS = ("base",) + tuple(f"T{i + 1}" for i in range(N_RULES)) + ("joint_single", "joint_full")


@dataclass
class Batch:
    initial: torch.Tensor        # (B, 20, H, W)
    trajectory: torch.Tensor     # (B, k + 1, 20, H, W), trajectory[:, 0] == initial
    event_masks: torch.Tensor    # (B, k, 3, H, W) bool, da iniettare all'inizio del tick t
    events: list                 # B liste di (tick, tipo, (riga, colonna), parametri)
    active: torch.Tensor         # (B, 8) bool
    support: torch.Tensor = None     # (B, k, 8, H, W) bool
    conflicts: torch.Tensor = None   # (B, k, 2, H, W) bool
    triggers: torch.Tensor = None    # (B, 8) bool, la regola è scattata almeno una volta

    def __len__(self):
        return self.initial.shape[0]

    def select(self, idx):
        idx_list = idx.tolist() if torch.is_tensor(idx) else list(idx)
        pick = lambda x: None if x is None else x[idx]
        return Batch(pick(self.initial), pick(self.trajectory), pick(self.event_masks),
                     [self.events[i] for i in idx_list], pick(self.active), pick(self.support),
                     pick(self.conflicts), pick(self.triggers))

    @staticmethod
    def cat(batches):
        cat = lambda xs: None if xs[0] is None else torch.cat(xs, 0)
        return Batch(*(cat([getattr(b, f) for b in batches]) if f != "events"
                       else sum((b.events for b in batches), [])
                       for f in ("initial", "trajectory", "event_masks", "events", "active",
                                 "support", "conflicts", "triggers")))


def active_rules(name, n, gen, device):
    """Righe (n, 8) di regole attive per un dataset."""
    act = torch.zeros(n, N_RULES, dtype=torch.bool, device=device)
    if name == "base":
        return act
    if name == "joint_full":
        return ~act
    if name == "joint_single":
        idx = torch.randint(N_RULES, (n,), generator=gen, device=device)
        act[torch.arange(n, device=device), idx] = True
        return act
    if name in RULE_IDS:
        act[:, RULE_IDS[name]] = True
        return act
    names = name.split("+")
    for nm in names:
        act[:, RULE_IDS[nm]] = True
    return act


@dataclass
class GeneratorConfig:
    size: int = 32
    min_trigger_frac: float = 0.5
    min_trigger_cells: int = 8
    max_resample_rounds: int = 4
    sim: SimParams = field(default_factory=SimParams)
    scenes: SceneParams = field(default_factory=SceneParams)
    events: EventParams = field(default_factory=EventParams)


class SandboxGenerator:
    """Genera batch di traiettorie del simulatore, pronti per l'addestramento.

    Esempio:
        gen = SandboxGenerator(device="cuda", seed=0)
        batch = gen.sample(64, "T1", steps=32)
    """

    def __init__(self, config: GeneratorConfig = None, device="cpu", seed=None):
        self.cfg = config or GeneratorConfig()
        self.device = torch.device(device)
        self.sim = Simulator(self.cfg.sim, self.device)
        self.gen = torch.Generator(device=self.device)
        self.cpu_gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)
            self.cpu_gen.manual_seed(seed + 1)
        else:
            self.gen.seed()
            self.cpu_gen.seed()

    def _simulate(self, initial, events, active, steps, keep_info):
        h, w = initial.shape[-2:]
        masks = batch_event_masks(events, steps, h, w, self.device)
        traj, sup, conf, trig = self.sim.rollout(initial, active, steps, masks, keep_info=keep_info)
        return Batch(initial, traj, masks, events, active, sup, conf, trig)

    def _fresh(self, active, steps, keep_info, size):
        n = active.shape[0]
        initial = sample_scenes(n, size, size, self.gen, self.device, self.cfg.scenes, self.cfg.sim)
        cpu_init = initial.cpu()
        events = [sample_events(cpu_init[i], steps, self.cpu_gen, self.cfg.events) for i in range(n)]
        return self._simulate(initial, events, active, steps, keep_info)

    def activated(self, batch):
        """A window counts as activating if its active rules were triggered in at least min_trigger_cells cells."""
        return (batch.triggers * batch.active).sum(1) >= self.cfg.min_trigger_cells

    @torch.no_grad()
    def sample(self, n, rules="base", steps=32, keep_info=True, size=None, balance=True):
        """n scene del dataset `rules` (vedi DATASETS, o una combinazione come "T1+T2"), k = steps tick."""
        size = size or self.cfg.size
        active = active_rules(rules, n, self.gen, self.device)
        batch = self._fresh(active, steps, keep_info, size)
        if not balance or not active.any():
            return batch

        target = int(round(self.cfg.min_trigger_frac * n))
        for _ in range(self.cfg.max_resample_rounds):
            ok = self.activated(batch)
            missing = target - int(ok.sum())
            if missing <= 0:
                break
            # ricampioniamo solo le scene che non attivano, con le stesse regole, e teniamo le nuove buone
            bad = torch.nonzero(~ok).flatten()
            cand = self._fresh(batch.active[bad], steps, keep_info, size)
            good = torch.nonzero(self.activated(cand)).flatten()[:missing]
            if len(good) == 0:
                continue
            keep = torch.ones(n, dtype=torch.bool, device=self.device)
            keep[bad[:len(good)]] = False
            batch = Batch.cat([batch.select(torch.nonzero(keep).flatten()), cand.select(good)])
        perm = torch.randperm(n, generator=self.gen, device=self.device)
        return batch.select(perm)


def _checksum(traj_final):
    return traj_final.double().flatten(1).sum(1).cpu()


@torch.no_grad()
def save_split(path, rules, n=500, steps=256, seed=0, size=32, device="cpu", chunk=50, config=None):
    """Salva una split di validazione o test con seed fisso.

    Le traiettorie complete (500 scene x 257 tick a 32x32 sono circa 10 GB) non si salvano:
    si salvano stato iniziale, eventi e regole attive, e la traiettoria si rigenera al volo,
    perché il simulatore è deterministico. Il checksum dello stato finale lo verifica.
    """
    cfg = config or GeneratorConfig(size=size)
    g = SandboxGenerator(cfg, device=device, seed=seed)
    #every chunk keeps only what gets saved: concatenating the full trajectories would need ~10 GB
    initial, events, active, triggers, last = [], [], [], [], []
    for start in range(0, n, chunk):
        b = g.sample(min(chunk, n - start), rules, steps, keep_info=False, size=size)
        initial.append(b.initial.cpu())
        events += b.events
        active.append(b.active.cpu())
        triggers.append(b.triggers.cpu())
        last.append(b.trajectory[:, -1].cpu())
        del b
    data = {
        "rules": rules, "steps": steps, "size": size, "seed": seed,
        "initial": torch.cat(initial), "events": events, "active": torch.cat(active),
        "triggers": torch.cat(triggers), "checksum": _checksum(torch.cat(last)),
        "sim_params": cfg.sim.to_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


class Split:
    """Una split salvata. Rigenera traiettorie, supporti e conflitti a richiesta."""

    def __init__(self, path, device="cpu", sim_params: SimParams = None):
        self.data = torch.load(path, weights_only=False)
        self.device = torch.device(device)
        self.sim = Simulator(sim_params or SimParams(), self.device)

    def __len__(self):
        return self.data["initial"].shape[0]

    @torch.no_grad()
    def batch(self, idx, steps=None, keep_info=True):
        steps = steps or self.data["steps"]
        idx = list(idx)
        initial = self.data["initial"][idx].to(self.device)
        events = [self.data["events"][i] for i in idx]
        active = self.data["active"][idx].to(self.device)
        h, w = initial.shape[-2:]
        masks = batch_event_masks(events, steps, h, w, self.device)
        traj, sup, conf, trig = self.sim.rollout(initial, active, steps, masks, keep_info=keep_info)
        return Batch(initial, traj, masks, events, active, sup, conf, trig)

    def verify(self, idx=None, atol=1e-3):
        """Controlla che le traiettorie rigenerate coincidano con quelle salvate."""
        idx = list(range(len(self))) if idx is None else list(idx)
        b = self.batch(idx, keep_info=False)
        return torch.allclose(_checksum(b.trajectory[:, -1]), self.data["checksum"][idx], atol=atol)


def main():
    ap = argparse.ArgumentParser(description="Genera le split di validazione e test del sandbox.")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-seed", type=int, default=1000)
    ap.add_argument("--test-seed", type=int, default=2000)
    args = ap.parse_args()
    for name in args.datasets:
        for split, seed in (("val", args.val_seed), ("test", args.test_seed)):
            path = Path(args.out) / f"{name}_{split}_{args.size}.pt"
            data = save_split(path, name, args.n, args.steps, seed, args.size, args.device)
            min_cells = GeneratorConfig().min_trigger_cells
            frac = ((data["triggers"] * data["active"]).sum(1) >= min_cells).float().mean().item()
            print(f"{path}: {args.n} scene, {args.steps} tick, finestre attivanti {frac:.0%}")


if __name__ == "__main__":
    main()