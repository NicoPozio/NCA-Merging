"""Where a trained NCA fails on the cells where its rules act.

For every checkpoint it separates three things:
- one-step error: the NCA starts from the true state at every tick, so it only measures whether
  it learned the transition;
- rollout error: free rollout, so it adds the errors that pile up over the ticks;
- error per channel, to see which part of the rule is missed (charge, materials, fuel, ...).

Usage:
    python -m tools.diagnose runs_v2/specialists/*_s0.pt runs/specialists/*_s0.pt
"""

import argparse
from pathlib import Path

import torch

from nca_models.model import load_nca
from nca_sandbox.config import CHANNEL_NAMES, N_MATERIALS, WX
from nca_sandbox.dataset import GeneratorConfig, SandboxGenerator

GROUPS = {"mat": list(range(N_MATERIALS)), **{CHANNEL_NAMES[c]: [c] for c in range(N_MATERIALS, WX)}}


def _mean(sq, sup):
    # sq (B, k, 18, H, W), sup (B, k, 1, H, W) bool
    return float((sq * sup).sum() / (sup.sum() * sq.shape[2]).clamp(min=1))


@torch.no_grad()
def diagnose(path, n=32, steps=64, seed=12_345, device="cpu"):
    model, ck = load_nca(path, device)
    model.eval()
    rules = ck["rules"]
    gen = SandboxGenerator(GeneratorConfig(size=ck["train_cfg"]["size"], sim=model.p), device=device, seed=seed)
    b = gen.sample(n, rules, steps=steps, keep_info=True)
    true = b.trajectory[:, 1:, :WX]
    if rules == "base":
        sup = torch.ones_like(true[:, :, :1], dtype=torch.bool)
    else:
        sup = (b.support & b.active[:, None, :, None, None]).any(2).unsqueeze(2)

    roll = model.rollout(b.initial, steps, b.event_masks)[:, 1:, :WX]
    one = torch.stack([model(b.trajectory[:, t], b.event_masks[:, t]) for t in range(steps)], 1)[:, :, :WX]
    sq_roll, sq_one = (roll - true) ** 2, (one - true) ** 2

    per_tick = [_mean(sq_roll[:, t:t + 1], sup[:, t:t + 1]) for t in range(steps)]
    out = {"name": f"{Path(path).parent.name}/{Path(path).stem}", "support": float(sup.float().mean()),
           "one_step": _mean(sq_one, sup), "rollout": _mean(sq_roll, sup),
           "tick8": sum(per_tick[:8]) / 8, "last8": sum(per_tick[-8:]) / 8}
    out["channels"] = {g: _mean(sq_roll[:, :, c], sup) for g, c in GROUPS.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    rows = [diagnose(p, args.n, args.steps, device=args.device) for p in args.checkpoints]
    print(f"{'model':28s} {'support':>8s} {'one-step':>10s} {'rollout':>10s} {'ratio':>7s} "
          f"{'ticks1-8':>10s} {'last 8':>10s}  worst channels on the support")
    for r in rows:
        worst = sorted(r["channels"].items(), key=lambda kv: -kv[1])[:3]
        print(f"{r['name']:28s} {100 * r['support']:7.2f}% {r['one_step']:10.3g} {r['rollout']:10.3g} "
              f"{r['rollout'] / max(r['one_step'], 1e-12):7.1f} {r['tick8']:10.3g} {r['last8']:10.3g}  "
              + "  ".join(f"{g} {v:.2g}" for g, v in worst))
    return rows


if __name__ == "__main__":
    main()