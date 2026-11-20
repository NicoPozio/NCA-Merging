"""Merge specialists and evaluate the merges against the base, the specialists and joint_full.

    python -m tools.merge --base runs/base/base_s0.pt --specialists runs2/specialists \
        --sets T1+T6 all --methods average ta --alphas 0.3 0.5 0.7 1.0

--sets takes rule sets as T1+T6, plus "all" (the 8 rules) and "pairs" (the 28 pairs).
For every set and every setting of every method the merge is saved as a checkpoint in
<out>/merges/<setting>__<dataset>.pt and evaluated (records shared with tools.results) on:
- the dataset where exactly those rules are active (T1+T6, or joint_full for all 8);
- the dataset of each rule alone.

Per merge the summary keeps:
- comb_rule, comb_all: error on the rule cells and on every cell on the combined dataset, and
  the same for the base (base_rule, base_all);
- ret_Ti: retention of rule i, (base - merge) / (base - specialist) on the rule cells of the
  dataset of rule i alone: 1 = as good as the specialist, 0 = as the base, below 0 = worse;
- ret_mean, ret_min over the rules of the set;
- gap_closed: (base - merge) / (base - joint_full) on the combined dataset, only for all 8
  rules and when --joint is given.

The settings are chosen on the validation split (the default) with short rollouts; the final
numbers go on the test split with --split test --steps 256 --n 256 and the chosen settings only.
Everything is collected in <out>/merge_summary.csv (one row per merge, split, steps and n).
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import torch

from nca_models.evaluate import evaluate_checkpoint, window_mean
from nca_models.merging import dataset_for, hp_tag, load_specialists, merge, save_merge, task_vectors
from nca_models.model import load_nca

RULES = tuple(f"T{i}" for i in range(1, 9))
KEY = ("set", "setting", "split", "steps", "n")


def parse_sets(items):
    sets = []
    for it in items:
        if it == "all":
            sets.append(list(RULES))
        elif it == "pairs":
            sets += [list(p) for p in itertools.combinations(RULES, 2)]
        else:
            sets.append(it.split("+"))
    return sets


def settings(methods, alphas):
    out = []
    for m in methods:
        if m == "ta":
            out += [("ta", {"alpha": a}) for a in alphas]
        else:
            out.append((m, {}))
    return out


def retention(base_err, merge_err, spec_err):
    gap = base_err - spec_err
    if not all(map(math.isfinite, (base_err, merge_err, spec_err))) or gap <= 0:
        return float("nan")
    return (base_err - merge_err) / gap


def read_summary(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {tuple(str(r[k]) for k in KEY): r for r in csv.DictReader(f)}


def write_summary(path, rows):
    rows = list(rows.values())
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _fmt(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return f"{'':>9s}"
    return f"{x:9.3g}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge specialists and evaluate the merges.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--specialists", required=True, help="folder with <rule>_s<seed>[_best].pt")
    ap.add_argument("--which", default="best", choices=("last", "best"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint", default=None, help="joint_full checkpoint, reference for the merge of all rules")
    ap.add_argument("--sets", nargs="+", default=["all"])
    ap.add_argument("--methods", nargs="+", default=["average", "ta"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--out", default="runs_merges")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--records-dir", default="results/records")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    sets = parse_sets(args.sets)
    needed = sorted({r for s in sets for r in s}, key=lambda r: int(r[1:]))
    base, base_meta = load_nca(args.base, args.device)
    specs = load_specialists(Path(args.specialists), needed, args.seed, args.which, args.device)
    taus = task_vectors(base, {r: m for r, (m, _) in specs.items()})
    kw = dict(split=args.split, steps=args.steps, n=args.n, splits_dir=args.splits_dir,
              records_dir=args.records_dir, device=args.device)
    rule_err = lambda rec: window_mean(rec, "mse_rule", 1, args.steps)
    all_err = lambda rec: window_mean(rec, "mse_all", 1, args.steps)

    out = Path(args.out)
    summary_path = out / "merge_summary.csv"
    summary = read_summary(summary_path)
    new_rows = []
    for rules in sets:
        ds = dataset_for(rules)
        singles = [] if len(rules) == 1 else list(rules)
        base_ref = {d: evaluate_checkpoint(args.base, d, **kw) for d in [ds] + singles}
        spec_ref = {r: rule_err(evaluate_checkpoint(specs[r][1], r, **kw)) for r in singles}
        joint_err = None
        if ds == "joint_full" and args.joint:
            joint_err = rule_err(evaluate_checkpoint(args.joint, ds, **kw))

        for method, hp in settings(args.methods, args.alphas):
            tag = hp_tag(method, hp)
            model = merge(base, {r: taus[r] for r in rules}, method, **hp)
            path = out / "merges" / f"{tag}__{ds}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_merge(path, model, args.base, base_meta, rules, method, hp, {r: specs[r][1] for r in rules})

            comb = evaluate_checkpoint(path, ds, **kw)
            row = {"set": ds, "setting": tag, "method": method, "split": args.split, "steps": args.steps,
                   "n": args.n, "comb_rule": rule_err(comb), "comb_all": all_err(comb),
                   "base_rule": rule_err(base_ref[ds]), "base_all": all_err(base_ref[ds])}
            rets = []
            for r in singles:
                ret = retention(rule_err(base_ref[r]), rule_err(evaluate_checkpoint(path, r, **kw)), spec_ref[r])
                row[f"ret_{r}"] = ret
                if math.isfinite(ret):
                    rets.append(ret)
            row["ret_mean"] = sum(rets) / len(rets) if rets else float("nan")
            row["ret_min"] = min(rets) if rets else float("nan")
            if joint_err is not None:
                row["joint_rule"] = joint_err
                row["gap_closed"] = retention(row["base_rule"], row["comb_rule"], joint_err)
            summary[tuple(str(row[k]) for k in KEY)] = row
            new_rows.append(row)
            print(f"{ds:12s} {tag:14s} comb {row['comb_rule']:.3g} (base {row['base_rule']:.3g})  "
                  f"retention mean {row['ret_mean']:.2f} min {row['ret_min']:.2f}"
                  + (f"  gap closed {row['gap_closed']:.2f}" if "gap_closed" in row else ""))

    write_summary(summary_path, summary)

    print(f"\n{args.split}, ticks 1-{args.steps}, {args.n} scenes")
    cols = ["comb_rule", "base_rule", "ret_mean", "ret_min", "gap_closed"]
    print(f"{'set':12s} {'setting':14s} " + " ".join(f"{c:>9s}" for c in cols))
    for r in new_rows:
        print(f"{r['set']:12s} {r['setting']:14s} " + " ".join(_fmt(r.get(c)) for c in cols))

    print("\nbest setting per set and method (highest mean retention)")
    for (ds, method), group in itertools.groupby(sorted(new_rows, key=lambda r: (r["set"], r["method"])),
                                                 key=lambda r: (r["set"], r["method"])):
        group = [g for g in group if math.isfinite(g["ret_mean"])]
        if group:
            best = max(group, key=lambda g: g["ret_mean"])
            print(f"{ds:12s} {method:8s} {best['setting']:14s} retention {best['ret_mean']:.2f}")
    print(f"\nsaved {summary_path}")
    return new_rows


if __name__ == "__main__":
    main()