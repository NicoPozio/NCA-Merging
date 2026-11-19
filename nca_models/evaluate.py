"""Long-horizon evaluation of a checkpoint against the simulator, on the saved splits.

The NCA and the simulator run side by side, one tick at a time, so 256 ticks on hundreds of scenes
never need the full trajectories in memory. For every tick we keep:
- mse_all: squared error on every cell and channel (W .. S and the materials);
- rule_sum, rule_cnt: the same error only on the cells where an active rule acts at that tick,
  kept as sum and count so windows can be averaged correctly later;
- mat_err: share of cells whose most likely material is wrong;
- per channel: error on every cell, error on the rule cells, mean value of the NCA and of the
  simulator. The two means show the slow drifts (fire that never goes out, heat that piles up)
  that degenerate long rollouts.

A record is a dict of plain values and tensors, saved with torch.save, one file per
(checkpoint, dataset, split, steps, n).
"""

import hashlib
import time
from pathlib import Path

import torch

from nca_sandbox.config import N_MATERIALS, WX
from nca_sandbox.dataset import Split, save_split
from nca_sandbox.events import batch_event_masks
from nca_sandbox.simulator import Simulator

from .model import load_nca, sim_params_from_dict

SPLIT_SEEDS = {"val": 1000, "test": 2000}       # the defaults of python -m nca_sandbox.dataset
WINDOWS = ((1, 64), (65, 128), (129, 256))       # 1 .. 64 is the training horizon


def split_path(splits_dir, dataset, split, size=32):
    return Path(splits_dir) / f"{dataset}_{split}_{size}.pt"


def ensure_split(splits_dir, dataset, split, size=32, n=500, steps=256, device="cpu"):
    """Path of the split, generated with the standard seeds if it does not exist yet."""
    path = split_path(splits_dir, dataset, split, size)
    if not path.exists():
        print(f"generating {path} ({n} scenes, {steps} ticks)")
        save_split(path, dataset, n, steps, SPLIT_SEEDS[split], size, device)
    return path


def open_split(path, device="cpu"):
    """A saved split whose simulator uses the parameters the split was generated with."""
    sp = Split(path, device)
    sp.sim = Simulator(sim_params_from_dict(sp.data["sim_params"]), sp.device)
    return sp


def file_hash(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def describe(path):
    """Run, group and name of a checkpoint from its path: <run>/<group>/<name>.pt."""
    path = Path(path)
    return {"run": path.parent.parent.name, "group": path.parent.name, "name": path.stem,
            "label": f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"}


def _support(info_support, active):
    # (B, 8, H, W) & (B, 8) -> (B, H, W): cells where some active rule acts
    return (info_support & active[:, :, None, None]).any(1)


@torch.no_grad()
def evaluate_split(model, split: Split, steps=256, n=None, chunk=64, keep_frames=None):
    """Runs the model and the simulator together on the first n scenes of the split.

    keep_frames: optional (scene_index, ticks) to also return the two states at those ticks,
    used by the frame figures. Returns a dict of per-tick tensors."""
    n = min(n or len(split), len(split))
    if steps > split.data["steps"]:
        raise ValueError(f"the split has {split.data['steps']} ticks, asked for {steps}")
    device = split.device
    model = model.to(device).eval()
    sim = split.sim
    acc = {
        "mse_all": torch.zeros(steps, dtype=torch.float64),
        "rule_sum": torch.zeros(steps, dtype=torch.float64),
        "rule_cnt": torch.zeros(steps, dtype=torch.float64),
        "mat_err": torch.zeros(steps, dtype=torch.float64),
        "ch_all": torch.zeros(steps, WX, dtype=torch.float64),
        "ch_rule": torch.zeros(steps, WX, dtype=torch.float64),
        "mean_pred": torch.zeros(steps, WX, dtype=torch.float64),
        "mean_true": torch.zeros(steps, WX, dtype=torch.float64),
    }
    frames = None
    if keep_frames is not None:
        scene, ticks = keep_frames
        frames = {"ticks": list(ticks), "pred": {}, "true": {}}

    for start in range(0, n, chunk):
        idx = list(range(start, min(n, start + chunk)))
        x_true = split.data["initial"][idx].to(device)
        active = split.data["active"][idx].to(device)
        events = [split.data["events"][i] for i in idx]
        h, w = x_true.shape[-2:]
        masks = batch_event_masks(events, steps, h, w, device)
        x_pred = x_true.clone()
        cells = len(idx) * h * w
        local = frames is not None and scene in idx
        if local:
            j = idx.index(scene)
            if 0 in frames["ticks"]:
                frames["true"][0] = x_true[j].cpu()
                frames["pred"][0] = x_pred[j].cpu()

        for t in range(steps):
            m = masks[:, t]
            x_true, info = sim.step(x_true, active, m, diagnostics=True)
            x_pred = model(x_pred, m)
            sq = (x_pred[:, :WX] - x_true[:, :WX]) ** 2                 # (b, 18, H, W)
            sup = _support(info.support, active).unsqueeze(1)          # (b, 1, H, W)
            acc["ch_all"][t] += sq.sum((0, 2, 3)).double().cpu()
            acc["ch_rule"][t] += (sq * sup).sum((0, 2, 3)).double().cpu()
            acc["rule_cnt"][t] += float(sup.sum())
            wrong = x_pred[:, :N_MATERIALS].argmax(1) != x_true[:, :N_MATERIALS].argmax(1)
            acc["mat_err"][t] += float(wrong.sum())
            acc["mean_pred"][t] += x_pred[:, :WX].sum((0, 2, 3)).double().cpu()
            acc["mean_true"][t] += x_true[:, :WX].sum((0, 2, 3)).double().cpu()
            if local and t + 1 in frames["ticks"]:
                frames["true"][t + 1] = x_true[j].cpu()
                frames["pred"][t + 1] = x_pred[j].cpu()
        acc["_cells"] = acc.get("_cells", 0) + cells

    cells = acc.pop("_cells")
    out = {
        "mse_all": (acc["ch_all"].sum(1) / (cells * WX)).float(),
        "rule_sum": (acc["ch_rule"].sum(1) / WX).float(),
        "rule_cnt": acc["rule_cnt"].float(),
        "mat_err": (acc["mat_err"] / cells).float(),
        "ch_all": (acc["ch_all"] / cells).float(),
        "ch_rule": (acc["ch_rule"] / acc["rule_cnt"].clamp(min=1)[:, None]).float(),
        "mean_pred": (acc["mean_pred"] / cells).float(),
        "mean_true": (acc["mean_true"] / cells).float(),
        "n": n,
        "steps": steps,
    }
    if frames is not None:
        out["frames"] = frames
    return out


def window_mean(rec, key, lo, hi):
    """Mean of a per-tick metric over ticks lo..hi (1-based, inclusive). For the rule error the
    ticks are weighted by how many rule cells they have; nan if the rules never act."""
    lo, hi = max(1, lo), min(hi, rec["steps"])
    if lo > hi:
        return float("nan")
    s = slice(lo - 1, hi)
    if key == "mse_rule":
        cnt = float(rec["rule_cnt"][s].sum())
        return float(rec["rule_sum"][s].sum()) / cnt if cnt > 0 else float("nan")
    return float(rec[key][s].mean())


def rule_curve(rec):
    """Per-tick error on the rule cells (nan where no rule acts)."""
    cnt = rec["rule_cnt"]
    return torch.where(cnt > 0, rec["rule_sum"] / cnt.clamp(min=1), torch.full_like(cnt, float("nan")))


def record_path(records_dir, ckpt, dataset, split, steps, n):
    d = describe(ckpt)
    return Path(records_dir) / d["run"] / d["group"] / f"{d['name']}__{dataset}_{split}_{steps}t_{n}n.pt"


def evaluate_checkpoint(ckpt, dataset, split="val", steps=256, n=256, splits_dir="data/splits",
                        records_dir="results/records", device="cpu", force=False, chunk=64):
    """Evaluates one checkpoint on one dataset and saves the record. A record already on disk is
    reused if the checkpoint file has not changed since."""
    model, meta = load_nca(ckpt, device)
    size = meta["train_cfg"]["size"]
    out_path = record_path(records_dir, ckpt, dataset, split, steps, n)
    h = file_hash(ckpt)
    if out_path.exists() and not force:
        rec = torch.load(out_path, weights_only=False)
        if rec["ckpt_hash"] == h:
            return rec

    path = ensure_split(splits_dir, dataset, split, size, steps=max(steps, 256), device=device)
    sp = open_split(path, device)
    if sp.sim.p.to_dict() != model.p.to_dict():
        print(f"warning: {ckpt} was trained with different simulator parameters than {path}")
    t0 = time.time()
    res = evaluate_split(model, sp, steps, n, chunk)
    train_cfg = meta.get("train_cfg", {})
    rec = {**describe(ckpt), **res, "ckpt": str(ckpt), "ckpt_hash": h, "dataset": dataset, "split": split,
           "rules": meta.get("rules"), "seed": train_cfg.get("seed"), "lr": train_cfg.get("lr"),
           "iters": train_cfg.get("iters"), "init": meta.get("init"), "seconds": round(time.time() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, out_path)
    return rec


def load_records(records_dir="results/records"):
    return [torch.load(p, weights_only=False) for p in sorted(Path(records_dir).rglob("*.pt"))]


def default_datasets(rules):
    """What a model is evaluated on: its own rule plus the base (to see what the rule costs the base
    dynamics); the joint models and the base on everything."""
    if rules in ("joint_single", "joint_full", "base"):
        return ["base"] + [f"T{i}" for i in range(1, 9)] + ["joint_full"]
    return [rules, "base"]


def frames_for(ckpt, dataset, scene, ticks, split="val", splits_dir="data/splits", device="cpu"):
    """States of the NCA and of the simulator at the given ticks of one scene."""
    model, meta = load_nca(ckpt, device)
    path = ensure_split(splits_dir, dataset, split, meta["train_cfg"]["size"], device=device)
    sp = open_split(path, device)
    res = evaluate_split(model, sp, max(ticks), n=scene + 1, chunk=scene + 1, keep_frames=(scene, ticks))
    return res["frames"]