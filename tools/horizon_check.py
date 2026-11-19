"""Point 1: does phase 2 at a higher lr fix the 256-tick degeneration without losing the rare rules?

Compares the specialists of two runs that differ only in the phase 2 lr, on the validation split
(the test split stays untouched until the final numbers), with rollouts of 256 ticks.

    python -m tools.horizon_check --old runs_v2 --new runs_p2lr3e-4 --rules T1 T4 T5 --device cpu

For every rule and checkpoint (last and best) it prints the error on the rule cells and on every
cell in the windows 1-64 (training horizon), 65-128 and 129-256, the ratio new / old, and the
channels that drift the most at the end. The verdict uses two criteria:
- hold: on the --hold rules the new error is never worse than the old by more than --tol;
- recover: on the --recover rules the new error in 129-256 is lower by at least --tol.
"""

import argparse
from pathlib import Path

import torch

from nca_models.evaluate import WINDOWS, evaluate_checkpoint, window_mean
from nca_sandbox.config import CHANNEL_NAMES


def ckpt_path(run, group, rule, seed, which):
    return Path(run) / group / f"{rule}_s{seed}{'_best' if which == 'best' else ''}.pt"


def config_diff(a, b):
    """train_cfg keys that differ between two checkpoints, device excluded."""
    ca = torch.load(a, map_location="cpu", weights_only=False)["train_cfg"]
    cb = torch.load(b, map_location="cpu", weights_only=False)["train_cfg"]
    return {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb))
            if k != "device" and ca.get(k) != cb.get(k)}


def drift(rec, lo, hi, top=3):
    """Channels whose mean value is furthest from the simulator in ticks lo..hi."""
    s = slice(lo - 1, min(hi, rec["steps"]))
    gap = (rec["mean_pred"][s] - rec["mean_true"][s]).mean(0)
    order = gap.abs().argsort(descending=True)[:top]
    return "  ".join(f"{CHANNEL_NAMES[c]} {float(gap[c]):+.3f}" for c in order)


def ratio(new, old):
    return new / old if old > 0 else float("nan")


def compare(old, new, windows):
    """Rows (metric, window, old, new, ratio)."""
    rows = []
    for key in ("mse_rule", "mse_all", "mat_err"):
        for lo, hi in windows:
            a, b = window_mean(old, key, lo, hi), window_mean(new, key, lo, hi)
            rows.append((key, f"{lo}-{hi}", a, b, ratio(b, a)))
    return rows


def verdict(results, hold, recover, tol, last_window):
    """results: (rule, which) -> rows. Returns the lines of the verdict and, for every checkpoint
    kind (last, best), whether all its criteria pass."""
    lines, ok = [], {}
    for (rule, which), rows in sorted(results.items()):
        ok.setdefault(which, True)
        rule_rows = [r for r in rows if r[0] == "mse_rule"]
        if rule in hold:
            worst = max(r[4] for r in rule_rows)
            passed = worst <= 1 + tol
            lines.append(f"hold     {rule} {which:4s}: worst new/old {worst:.2f} (max {1 + tol:.2f}) "
                         f"{'ok' if passed else 'FAIL'}")
            ok[which] &= passed
        if rule in recover:
            end = next(r for r in rule_rows if r[1] == last_window)[4]
            passed = end <= 1 - tol
            lines.append(f"recover  {rule} {which:4s}: new/old in {last_window} {end:.2f} (max {1 - tol:.2f}) "
                         f"{'ok' if passed else 'FAIL'}")
            ok[which] &= passed
    return lines, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True, help="--out of the run with the current phase 2")
    ap.add_argument("--new", required=True, help="--out of the run with the new phase 2")
    ap.add_argument("--group", default="specialists")
    ap.add_argument("--rules", nargs="+", default=["T1", "T4", "T5"])
    ap.add_argument("--hold", nargs="*", default=["T5"])
    ap.add_argument("--recover", nargs="*", default=["T1"])
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--which", nargs="+", default=["last", "best"], choices=("last", "best"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--records-dir", default="results/records")
    ap.add_argument("--plots", default=None, help="folder for the error vs tick figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    windows = [(lo, min(hi, args.steps)) for lo, hi in WINDOWS if lo <= args.steps]
    last_window = f"{windows[-1][0]}-{windows[-1][1]}"
    results, records = {}, {}
    for rule in args.rules:
        for which in args.which:
            a = ckpt_path(args.old, args.group, rule, args.seed, which)
            b = ckpt_path(args.new, args.group, rule, args.seed, which)
            if which == "last":
                diff = config_diff(a, b)
                print(f"\n{rule}: training settings that differ (old, new): {diff or 'none'}")
            kw = dict(dataset=rule, split=args.split, steps=args.steps, n=args.n, splits_dir=args.splits_dir,
                      records_dir=args.records_dir, device=args.device)
            old, new = evaluate_checkpoint(a, **kw), evaluate_checkpoint(b, **kw)
            records[(rule, which)] = (old, new)
            rows = compare(old, new, windows)
            results[(rule, which)] = rows
            print(f"{rule} {which} ({args.split}, {old['n']} scenes, {args.steps} ticks)")
            print(f"  {'metric':9s} {'ticks':>8s} {'old':>10s} {'new':>10s} {'new/old':>8s}")
            for key, win, x, y, r in rows:
                print(f"  {key:9s} {win:>8s} {x:10.3g} {y:10.3g} {r:8.2f}")
            print(f"  drift {last_window}  old: {drift(old, *windows[-1])}")
            print(f"  drift {last_window}  new: {drift(new, *windows[-1])}")

    lines, ok_by = verdict(results, set(args.hold), set(args.recover), args.tol, last_window)
    passing = [w for w, good in ok_by.items() if good]
    print("\nverdict")
    for line in lines:
        print("  " + line)
    if passing:
        print(f"  -> retrain specialists, scratch and joint with the new phase 2 (passes with: {', '.join(passing)})")
    else:
        print("  -> keep the current phase 2")
    ok = bool(passing)

    if args.plots:
        from tools.figures import plot_curves, plot_drift

        out = Path(args.plots)
        out.mkdir(parents=True, exist_ok=True)
        for (rule, which), (old, new) in records.items():
            plot_curves([old, new], out / f"{rule}_{which}_rule.png", "mse_rule",
                        f"{rule} ({which}): error on the rule cells")
            plot_curves([old, new], out / f"{rule}_{which}_all.png", "mse_all",
                        f"{rule} ({which}): error on every cell")
            plot_drift([old, new], ["F", "T", "H", "W"], out / f"{rule}_{which}_drift.png",
                       f"{rule} ({which}): mean value, NCA vs simulator")
        print(f"figures in {out}")
    return ok


if __name__ == "__main__":
    main()