"""Mergeability metrics of every pair of specialists, before any merge.

    python -m tools.mergeability --base runs/base/base_s0.pt --specialists runs2/specialists

Writes <out>/<tag>_pairs.csv (one row per pair, 28 for 8 rules) and <tag>_rules.csv (one row
per rule), then prints both and the rank correlation between the metrics over the pairs, to see
which ones say the same thing. The metrics are described in nca_models/mergeability.py.
Gradients use the split of each rule, the simulator metrics the joint_full split (every rule
active); both the validation split by default, so the test split stays untouched.
"""

import argparse
import csv
import math
from pathlib import Path

import torch

from nca_models.evaluate import ensure_split, open_split
from nca_models.mergeability import RULES, pair_table, rule_gradient, rule_table, simulator_stats, task_vector
from nca_models.model import load_nca

PAIR_METRICS = ("cos", "cos_W1", "cos_W2", "top_overlap", "sign_conflict", "sti_W1", "sti_W2", "grad_cos",
                "cross_leak", "support_jaccard", "support_overlap", "channel_cos", "cowrite")


def spearman(a, b):
    """Rank correlation over the pairs where both values are defined."""
    keep = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(keep) < 3:
        return float("nan")
    xs, ys = torch.tensor([k[0] for k in keep]), torch.tensor([k[1] for k in keep])
    rx, ry = xs.argsort().argsort().double(), ys.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = rx.norm() * ry.norm()
    return float(rx @ ry / den) if den > 0 else float("nan")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_table(rows, key, cols):
    cols = [c for c in cols if c in rows[0]]
    print(f"{key:8s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
    for r in rows:
        print(f"{r[key]:8s} " + " ".join(f"{r[c]:11.3g}" if isinstance(r[c], float) else f"{r[c]:>11}" for c in cols))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mergeability metrics of every pair of specialists.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--specialists", required=True, help="folder with <rule>_s<seed>.pt")
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", default="last", choices=("last", "best"))
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--n", type=int, default=64, help="scenes per split")
    ap.add_argument("--ticks", type=int, default=64)
    ap.add_argument("--top", type=float, default=0.2, help="share of weights kept for the sign metrics")
    ap.add_argument("--energy", type=float, default=0.95, help="share of the norm kept by the SVD")
    ap.add_argument("--out", default="results/mergeability")
    ap.add_argument("--tag", default=None, help="default: the run folder of the specialists")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    spec_dir = Path(args.specialists)
    tag = args.tag or spec_dir.parent.name
    base, base_meta = load_nca(args.base, args.device)
    specs = {}
    for r in args.rules:
        path = spec_dir / f"{r}_s{args.seed}{'_best' if args.which == 'best' else ''}.pt"
        model, meta = load_nca(path, args.device)
        if Path(str(meta.get("init"))).name != Path(args.base).name:
            print(f"warning: {path} was trained from {meta.get('init')}, not from {args.base}")
        specs[r] = model
    size = base_meta["train_cfg"]["size"]
    tvs = {r: task_vector(m, base) for r, m in specs.items()}

    grads = {}
    for r in args.rules:
        sp = open_split(ensure_split(args.splits_dir, r, args.split, size, device=args.device), args.device)
        grads[r] = rule_gradient(base, sp, r, args.n, args.ticks, device=args.device)
        print(f"gradient {r}: norm {float(torch.cat([g.flatten() for g in grads[r].values()]).norm()):.3g}")

    joint = open_split(ensure_split(args.splits_dir, "joint_full", args.split, size, device=args.device), args.device)
    stats = simulator_stats(joint, specs, base, args.n, args.ticks, device=args.device)

    pairs = pair_table(tvs, grads, stats, args.top, args.energy)
    rules = rule_table(tvs, grads, stats, args.energy)
    out = Path(args.out)
    write_csv(out / f"{tag}_pairs.csv", pairs)
    write_csv(out / f"{tag}_rules.csv", rules)

    print("\nrules")
    print_table(rules, "rule", [k for k in rules[0] if k != "rule"])
    print("\npairs")
    print_table(pairs, "pair", PAIR_METRICS)
    cols = [c for c in PAIR_METRICS if c in pairs[0]]
    print("\nrank correlation between metrics over the pairs (|rho| near 1 = they say the same thing)")
    print(f"{'':16s}" + " ".join(f"{c[:7]:>7s}" for c in cols))
    for a in cols:
        va = [p[a] for p in pairs]
        print(f"{a:16s}" + " ".join(f"{spearman(va, [p[b] for p in pairs]):7.2f}" for b in cols))
    print(f"\nsaved {out / f'{tag}_pairs.csv'} and {out / f'{tag}_rules.csv'}")
    return pairs, rules


if __name__ == "__main__":
    main()