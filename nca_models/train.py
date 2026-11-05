"""Training loop shared by every model of the thesis.

Loss: squared error on all channels against the simulator at every tick of the window, with the
gradient flowing through the whole window. Curriculum on the window length (8 -> 64 ticks),
noise on the starting state, and transition training: part of the windows start from the NCA's
own rollout, so the network learns to correct its errors (Richardson et al.).

Long runs can be split across sessions: every checkpoint_every iterations the full training state
goes to out_resume.pt, and calling train again with the same settings continues from there.
"""

import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from nca_sandbox.config import WX
from nca_sandbox.dataset import GeneratorConfig, SandboxGenerator

from .data import WindowPool
from .model import NCA, NCAConfig, load_nca, save_nca

CONTINUOUS_SLICE = slice(8, WX)     # W .. S: noise goes only on continuous channels, never on materials


@dataclass
class TrainConfig:
    rules: str = "base"                 # base, T1..T8, joint_single, joint_full
    iters: int = 5000
    batch: int = 16
    lr: float = 1e-3
    lr_min: float = 1e-4
    grad_clip: float = 1.0
    k_min: int = 8
    k_max: int = 64
    curriculum_frac: float = 0.6        # share of iterations spent growing k from k_min to k_max
    noise: float = 0.01
    self_start_prob: float = 0.5        # windows that start from the NCA's own rollout
    self_start_lead: int = 8            # ticks of NCA rollout before those windows
    self_start_after: float = 0.1       # only after this share of training, when the NCA is not random
    pool_scenes: int = 128
    pool_ticks: int = 96
    pool_refresh: int = 16              # scenes replaced every refresh_every iterations
    refresh_every: int = 25
    ambiguity_mask: bool = True         # joint_single only, see WindowPool
    support_share: float = 0.3          # share of the loss on cells where an active rule acts (0 = plain MSE)
    support_max_weight: float = 200.0
    support_sampling: float = 0.5       # share of windows drawn where the rules act the most
    size: int = 32
    val_scenes: int = 64
    val_ticks: int = 64
    eval_every: int = 250
    checkpoint_every: int = 250         # how often out_resume.pt is written
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    model: NCA
    history: list = field(default_factory=list)
    best_val: float = math.inf
    finished: bool = True               # False when the run stopped at the deadline and can be resumed
    iteration: int = 0


def window_length(it, cfg: TrainConfig):
    """Curriculum: k grows linearly and in multiples of 8 from k_min to k_max."""
    frac = min(1.0, it / max(1, cfg.curriculum_frac * cfg.iters))
    k = cfg.k_min + frac * (cfg.k_max - cfg.k_min)
    return int(max(cfg.k_min, min(cfg.k_max, 8 * round(k / 8))))


def masked_mse(pred, target, weights=None):
    err = (pred[:, :, :WX] - target[:, :, :WX]) ** 2
    if weights is None:
        return err.mean()
    return (err * weights).sum() / (weights.sum() * err.shape[2]).clamp(min=1.0)


@torch.no_grad()
def make_val_set(cfg: TrainConfig, gen_cfg: GeneratorConfig):
    gen = SandboxGenerator(gen_cfg, device=cfg.device, seed=10_000 + cfg.seed)
    return gen.sample(cfg.val_scenes, cfg.rules, steps=cfg.val_ticks, keep_info=False, size=cfg.size)


@torch.no_grad()
def evaluate(model, val):
    """Free rollout of the NCA over the whole validation horizon, compared with the simulator."""
    model.eval()
    pred = model.rollout(val.initial, val.trajectory.shape[1] - 1, val.event_masks)
    err = ((pred[:, 1:, :WX] - val.trajectory[:, 1:, :WX]) ** 2).mean((0, 2, 3, 4))
    model.train()
    return {"val_mse": float(err.mean()), "val_mse_t8": float(err[:8].mean()), "val_mse_last": float(err[-1])}


def build_model(init, model_cfg: NCAConfig, sim_params, device, seed):
    """init: 'random' or the path of a checkpoint to fine-tune from."""
    if init == "random":
        torch.manual_seed(seed)
        return NCA(model_cfg, sim_params).to(device)
    model, _ = load_nca(init, device)
    return model


def resume_path(out):
    return Path(f"{out}_resume.pt")


def _atomic_save(obj, path):
    """Write to a temporary file and rename it, so a session killed mid-write never leaves a broken file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _comparable(cfg: TrainConfig):
    # the device may change between sessions without changing the run
    d = asdict(cfg)
    d.pop("device")
    return d


def train(cfg: TrainConfig, init="random", model_cfg: NCAConfig = None, gen_cfg: GeneratorConfig = None,
          trainable="all", out=None, log=print, resume=True, deadline=None):
    """Train one NCA. Returns the model and its history; with out also saves out.pt (last) and
    out_best.pt (best validation).

    resume: if out_resume.pt exists, continue from it (the settings must be the same).
    deadline: a time.time() value; once passed, the state is saved to out_resume.pt and the run stops
    with result.finished False. The pool is regenerated on resume rather than saved (it would weigh
    hundreds of MB), so a resumed run is statistically, not bitwise, equal to an uninterrupted one."""
    gen_cfg = gen_cfg or GeneratorConfig(size=cfg.size)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    model = build_model(init, model_cfg or NCAConfig(), gen_cfg.sim, cfg.device, cfg.seed)
    model.set_trainable(trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters, eta_min=cfg.lr_min)

    gen = SandboxGenerator(gen_cfg, device=cfg.device, seed=cfg.seed)
    pool = WindowPool(gen, cfg.rules, cfg.pool_scenes, cfg.pool_ticks, cfg.size,
                      ambiguity_mask=cfg.ambiguity_mask and cfg.rules == "joint_single",
                      support_share=cfg.support_share, support_max_weight=cfg.support_max_weight,
                      support_sampling=cfg.support_sampling)
    val = make_val_set(cfg, gen_cfg)
    g = torch.Generator(device=cfg.device)
    g.manual_seed(cfg.seed + 7)

    result = TrainResult(model)
    meta = {"rules": cfg.rules, "init": str(init), "trainable": trainable, "train_cfg": asdict(cfg)}
    rpath = resume_path(out) if out is not None else None

    def save_state(it):
        _atomic_save({"iter": it, "model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "history": result.history, "best_val": result.best_val,
                      "rng": {"g": g.get_state(), "gen": gen.gen.get_state(), "cpu_gen": gen.cpu_gen.get_state()},
                      "train_cfg": _comparable(cfg), "init": str(init), "trainable": trainable,
                      "model_cfg": model.cfg.to_dict()}, rpath)

    start_it = 1
    if resume and rpath is not None and rpath.exists():
        ck = torch.load(rpath, map_location=cfg.device, weights_only=False)
        same = (ck["train_cfg"] == _comparable(cfg) and ck["init"] == str(init)
                and ck["trainable"] == trainable and ck["model_cfg"] == model.cfg.to_dict())
        if not same:
            raise ValueError(f"{rpath} was written with different settings: delete it to start over")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        result.history, result.best_val = ck["history"], ck["best_val"]
        g.set_state(ck["rng"]["g"])
        gen.gen.set_state(ck["rng"]["gen"])
        gen.cpu_gen.set_state(ck["rng"]["cpu_gen"])
        start_it = ck["iter"] + 1
        log(f"resumed from iteration {ck['iter']}")

    t_start = time.time()
    for it in range(start_it, cfg.iters + 1):
        k = window_length(it, cfg)
        self_start = (it > cfg.self_start_after * cfg.iters
                      and torch.rand(1, generator=g, device=cfg.device).item() < cfg.self_start_prob)
        lead = cfg.self_start_lead if self_start else 0
        start, target, masks, weights, lead_in = pool.sample(cfg.batch, k, lead=lead)

        if lead_in is not None:
            with torch.no_grad():
                x, lead_masks = lead_in
                for t in range(lead_masks.shape[1]):
                    x = model(x, lead_masks[:, t])
                start = x
        if cfg.noise > 0:
            noisy = start[:, CONTINUOUS_SLICE] + cfg.noise * torch.randn(
                start[:, CONTINUOUS_SLICE].shape, generator=g, device=cfg.device)
            start = torch.cat([start[:, :8], noisy.clamp(0, 1), start[:, WX:]], 1)

        pred = model.rollout(start, k, masks)[:, 1:]
        loss = masked_mse(pred, target, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()

        if it % cfg.refresh_every == 0:
            pool.refresh(cfg.pool_refresh)
        if it % cfg.eval_every == 0 or it == cfg.iters:
            metrics = evaluate(model, val)
            row = {"iter": it, "k": k, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                   "time": round(time.time() - t_start, 1), **metrics}
            result.history.append(row)
            log(" ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}" for a, b in row.items()))
            if out is not None and metrics["val_mse"] < result.best_val:
                result.best_val = metrics["val_mse"]
                save_nca(Path(f"{out}_best.pt"), model, history=result.history, **meta)
        result.iteration = it
        if rpath is not None and it < cfg.iters:
            if deadline is not None and time.time() >= deadline:
                save_state(it)
                result.finished = False
                log(f"deadline reached at iteration {it}, state saved to {rpath}")
                return result
            if it % cfg.checkpoint_every == 0:
                save_state(it)
    if out is not None:
        save_nca(Path(f"{out}.pt"), model, history=result.history, **meta)
        if rpath.exists():
            rpath.unlink()
    return result