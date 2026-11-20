"""Point 2: every result in one place, tables and figures generated from the records, never by hand.

    python -m tools.results eval runs_v2/*/*.pt                 evaluate (skips _p1, _resume, _best)
    python -m tools.results table --window 129-256              summary.csv + pivot table in markdown
    python -m tools.results plot heatmap --window 1-64
    python -m tools.results plot curves --dataset T1
    python -m tools.results plot drift --dataset T1 --channels F T
    python -m tools.results plot frames --dataset T1 --scene 3 runs_v2/specialists/T1_s0.pt ...

Every evaluation is a record in results/records/<run>/<group>/; a record is recomputed only if
its checkpoint changed. The comparison rows are always the same: base (lower bound), the
specialist on its own rule, scratch on its own rule, joint_single and joint_full (upper bounds).
A merge will be one more row in the same tables.
"""

import argparse
import csv
from pathlib import Path

import torch

from nca_models.evaluate import (
    WINDOWS, default_datasets, evaluate_checkpoint, frames_for, load_records, window_mean,
)
from nca_models.model import load_nca

METRICS = ("mse_rule", "mse_all", "mat_err")
GROUP_ORDER = ("base", "specialists", "scratch", "joint_single", "joint_full")
DATASET_ORDER = ["base"] + [f"T{i}" for i in range(1, 9)] + ["joint_full"]
OWN_RULE = ("specialists", "scratch")


def parse_window(text):
    lo, hi = text.split("-")
    return int(lo), int(hi)


def pick_checkpoints(paths, which):
    """Final checkpoints only: phase 1, resume files and (unless asked) best are left out."""
    out = []
    for p in map(Path, paths):
        stem = p.stem
        if stem.endswith("_p1") or stem.endswith("_resume") or stem.endswith("_p1_best"):
            continue
        if stem.endswith("_best") != (which == "best"):
            continue
        out.append(p)
    return sorted(out)


def row_key(rec):
    """The row a record goes in. Specialists and scratch collapse to one row per run, each column
    holding the model of that rule; every other model is its own row."""
    if rec["group"] in OWN_RULE:
        return f"{rec['run']}/{rec['group']}" + (" (best)" if rec["name"].endswith("_best") else "")
    return rec["label"]


def keep(rec, args):
    if rec["split"] != args.split or rec["steps"] != args.steps:
        return False
    return not args.runs or rec["run"] in args.runs


def own_rule_only(recs):
    """In the collapsed rows a specialist (or scratch) counts only on its own rule."""
    return [r for r in recs if r["group"] not in OWN_RULE or r["dataset"] == r["rules"]]


def summary_rows(recs):
    rows = []
    for r in recs:
        for lo, hi in WINDOWS + ((1, r["steps"]),):
            if lo > r["steps"]:
                continue
            row = {"run": r["run"], "group": r["group"], "name": r["name"], "rules": r["rules"],
                   "seed": r["seed"], "lr": r["lr"], "dataset": r["dataset"], "split": r["split"],
                   "n": r["n"], "window": f"{lo}-{min(hi, r['steps'])}"}
            row.update({m: window_mean(r, m, lo, hi) for m in METRICS})
            rows.append(row)
    return rows


def pivot(recs, metric, window):
    """Values (row, dataset) -> metric in the window. Rows are sorted by run, then base, specialists,
    scratch, joint_single, joint_full, then anything else (the merges)."""
    lo, hi = window
    values, order, cols = {}, {}, []
    for r in recs:
        k, d = row_key(r), r["dataset"]
        values[(k, d)] = window_mean(r, metric, lo, hi)
        g = GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else len(GROUP_ORDER)
        order[k] = (r["run"], g, k)
        if d not in cols:
            cols.append(d)
    rows = sorted(order, key=order.get)
    cols = [d for d in DATASET_ORDER if d in cols] + [d for d in cols if d not in DATASET_ORDER]
    return values, rows, cols


def markdown(values, rows, cols):
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in rows:
        cells = [("n/a" if values[(r, c)] != values[(r, c)] else f"{values[(r, c)]:.3g}")
                 if (r, c) in values else "" for c in cols]
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_eval(args):
    ckpts = pick_checkpoints(args.checkpoints, args.which)
    for i, ck in enumerate(ckpts, 1):
        rules = load_nca(ck, "cpu")[1]["rules"]
        for d in args.datasets or default_datasets(rules):
            rec = evaluate_checkpoint(ck, d, args.split, args.steps, args.n, args.splits_dir, args.records_dir,
                                      args.device, args.force)
            print(f"[{i}/{len(ckpts)}] {rec['label']:45s} on {d:10s} "
                  f"rule {window_mean(rec, 'mse_rule', 1, args.steps):.3g}  "
                  f"all {window_mean(rec, 'mse_all', 1, args.steps):.3g}  ({rec['seconds']}s)")


def cmd_table(args):
    recs = [r for r in load_records(args.records_dir) if keep(r, args)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(recs)
    if rows:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    window = parse_window(args.window)
    values, rws, cols = pivot(own_rule_only(recs), args.metric, window)
    text = f"{args.metric}, ticks {args.window}, {args.split}\n\n" + markdown(values, rws, cols)
    (out / f"table_{args.metric}_{args.window}.md").write_text(text + "\n")
    print(text)


def cmd_plot(args):
    from tools.figures import plot_curves, plot_drift, plot_frames, plot_heatmap

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load_records(args.records_dir) if keep(r, args)]
    if args.kind == "heatmap":
        window = parse_window(args.window)
        values, rows, cols = pivot(own_rule_only(recs), args.metric, window)
        path = out / f"heatmap_{args.metric}_{args.window}.png"
        plot_heatmap(values, rows, cols, path, f"{args.metric}, ticks {args.window} ({args.split})")
    elif args.kind in ("curves", "drift"):
        sel = [r for r in recs if r["dataset"] == args.dataset and r["n"] == args.n]
        if args.labels:
            sel = [r for r in sel if any(s in r["label"] for s in args.labels)]
        else:
            # default view: final checkpoints, specialists and scratch only on their own rule
            sel = [r for r in own_rule_only(sel) if not r["name"].endswith("_best")]
        if not sel:
            raise SystemExit(f"no records on {args.dataset} with n={args.n}: run eval first")
        if args.kind == "curves":
            metric = args.metric
            if metric == "mse_rule" and all(float(r["rule_cnt"].sum()) == 0 for r in sel):
                # no rule acts on this dataset (the base): show the error on every cell instead
                metric = "mse_all"
            path = out / f"curves_{args.dataset}_{metric}.png"
            plot_curves(sel, path, metric, f"{args.dataset}: {metric} vs tick ({args.split})")
        else:
            path = out / f"drift_{args.dataset}.png"
            plot_drift(sel, args.channels, path, f"{args.dataset}: mean value, NCA vs simulator")
    else:
        frames = {}
        for ck in map(Path, args.checkpoints):
            label = f"{ck.parent.parent.name}/{ck.parent.name}/{ck.stem}"
            frames[label] = frames_for(ck, args.dataset, args.scene, args.ticks, args.split, args.splits_dir,
                                       args.device)
        path = out / f"frames_{args.dataset}_scene{args.scene}.png"
        plot_frames(frames, path, f"{args.dataset}, scene {args.scene} ({args.split})")
    print(f"saved {path}")


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--records-dir", default="results/records")
    common.add_argument("--splits-dir", default="data/splits")
    common.add_argument("--split", default="val", choices=("val", "test"))
    common.add_argument("--steps", type=int, default=256)
    common.add_argument("--n", type=int, default=256)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap = argparse.ArgumentParser(description="Evaluation records, tables and figures.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", parents=[common])
    e.add_argument("checkpoints", nargs="+")
    e.add_argument("--which", default="last", choices=("last", "best"))
    e.add_argument("--datasets", nargs="*", default=None, help="default: own rule + base, everything for joint")
    e.add_argument("--force", action="store_true")

    for name in ("table", "plot"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--metric", default="mse_rule", choices=METRICS)
        p.add_argument("--window", default="129-256")
        p.add_argument("--runs", nargs="*", default=None, help="only these runs (folder names)")
        p.add_argument("--out", default="results")
        if name == "plot":
            p.add_argument("kind", choices=("heatmap", "curves", "drift", "frames"))
            p.add_argument("checkpoints", nargs="*", help="frames only")
            p.add_argument("--dataset", default="T1")
            p.add_argument("--labels", nargs="*", default=None, help="curves/drift: keep labels containing these")
            p.add_argument("--channels", nargs="+", default=["F", "T", "H", "W"])
            p.add_argument("--scene", type=int, default=0)
            p.add_argument("--ticks", type=int, nargs="+", default=[0, 32, 64, 128, 256])

    args = ap.parse_args(argv)
    {"eval": cmd_eval, "table": cmd_table, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()