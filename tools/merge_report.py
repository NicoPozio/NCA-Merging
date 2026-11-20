"""Figures and tables of the merges, from merge_summary.csv only (no merge is redone).

    python -m tools.merge_report --summary runs_merges/merge_summary.csv \
        --pairs results/mergeability/runs2_best_pairs.csv --out results/merges

Writes to --out:
- best_settings.csv / .md: the best setting of every method for every set, by mean log retention;
- scale_all.png: the merge of all 8 rules against the scaling coefficient (alpha, or lam for
  ties), gap closed towards joint_full and mean log retention, one line per method;
- scale_pairs.png: the same for the 28 pairs, mean log retention, one panel per pair;
- matrix_<method>.png: 8 x 8, cell (i, j) = log retention of rule i (row) when merged with rule j
  (column), in the best setting of that pair: a row full of red is a rule that dies in merges;
  matrix_any.png takes for every pair the best setting of any method;
- scatter_<method>.png and predictors_<method>.csv (with --pairs): the mergeability metrics of
  every pair against its best mean log retention, with the rank correlation over the 28 pairs.

Average has no coefficient: in the scale plots it is drawn as a point at 1 / number of rules,
where task arithmetic gives the same merge.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tools.figures import INK, INK_2, SERIES, _style
from tools.mergeability import spearman

RULES = tuple(f"T{i}" for i in range(1, 9))
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#b3261e", "#e34948", "#f4a7a1", "#f2f1ee", "#9ec5f4", "#3987e5", "#184f95"])
PREDICTORS = ("cos", "sub_left_W2", "sti_paper_W2", "grad_cos", "igrad_cos", "grad_cos_base", "cross_leak",
              "support_overlap", "channel_cos", "cowrite")
METHOD_ORDER = ("average", "ta", "ties", "dare", "tsv")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_summary(path, split, steps, n):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r["split"], int(r["steps"]), int(r["n"])) != (split, steps, n):
                continue
            r = dict(r)
            r["hp"] = json.loads(r["hp"]) if r.get("hp") else {}
            rules = RULES if r["set"] == "joint_full" else tuple(r["set"].split("+"))
            r["rules"] = rules
            for k, v in list(r.items()):
                if k.startswith(("err_", "base_", "spec_", "ret_", "lret_", "comb_", "gap_", "joint_")) or k == "scale":
                    r[k] = _num(v)
            if r["method"] == "average":
                r["scale"] = 1 / len(rules)
            rows.append(r)
    return rows


def series_of(r):
    """Line a row belongs to in the scale plots: the method plus its knobs other than the scale."""
    extra = [f"{k} {v:g}" for k, v in sorted(r["hp"].items()) if k not in ("alpha", "lam")]
    return " ".join([r["method"]] + extra)


# the categorical palette with the green moved last, so with six series no two greens meet
PALETTE = tuple(SERIES[i] for i in (0, 1, 2, 3, 4, 6, 7, 5))


def _series_key(name):
    m = name.split()[0]
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, name)


def series_colors(rows):
    names = sorted({series_of(r) for r in rows}, key=_series_key)
    if len(names) > len(PALETTE):
        raise ValueError(f"{len(names)} series do not fit one plot: run with fewer grid points")
    return dict(zip(names, PALETTE))


def best_per(rows, key):
    """Best row (highest mean log retention) for every value of key(row)."""
    out = {}
    for r in rows:
        if not math.isfinite(r.get("lret_mean", float("nan"))):
            continue
        k = key(r)
        if k not in out or r["lret_mean"] > out[k]["lret_mean"]:
            out[k] = r
    return out


def write_best_settings(rows, out):
    best = best_per(rows, lambda r: (r["set"], r["method"]))
    cols = ["set", "method", "setting", "lret_mean", "lret_min", "ret_mean", "comb_rule", "base_rule", "gap_closed"]
    table = [{c: best[k].get(c, "") for c in cols} for k in sorted(best, key=lambda k: (k[0] != "joint_full", k))]
    with open(out / "best_settings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    fmt = lambda v: f"{v:.3g}" if isinstance(v, float) and math.isfinite(v) else ("" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    lines += ["| " + " | ".join(fmt(t[c]) for c in cols) + " |" for t in table]
    (out / "best_settings.md").write_text("\n".join(lines) + "\n")
    return best


def _scale_lines(ax, rows, metric, colors):
    for name, group in itertools.groupby(sorted(rows, key=lambda r: _series_key(series_of(r))), key=series_of):
        g = sorted(group, key=lambda r: r["scale"])
        x = [r["scale"] for r in g]
        y = [r.get(metric, float("nan")) for r in g]
        style = dict(color=colors[name], linewidth=1.5)
        if len(g) == 1:
            ax.plot(x, y, marker="o", markersize=5, linestyle="none", **{k: v for k, v in style.items() if k != "linewidth"},
                    label=name)
        else:
            ax.plot(x, y, marker="o", markersize=3, label=name, **style)


def plot_scale_all(rows, out):
    sel = [r for r in rows if r["set"] == "joint_full"]
    if not sel:
        return None
    colors = series_colors(sel)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, label in ((axes[0], "gap_closed", "gap closed towards joint_full"),
                              (axes[1], "lret_mean", "mean log retention")):
        _scale_lines(ax, sel, metric, colors)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.axhline(1, color=INK_2, linewidth=0.8, linestyle="--")
        ax.set_ylim(-1.5, 1.2)
        ax.set_xlabel("scaling coefficient (alpha, lam)", fontsize=8, color=INK_2)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        _style(ax)
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("merge of all 8 rules (values below -1.5 are cut)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = out / "scale_all.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_scale_pairs(rows, out):
    sel = [r for r in rows if r["set"] != "joint_full" and len(r["rules"]) == 2]
    if not sel:
        return None
    colors = series_colors(sel)
    sets = sorted({r["set"] for r in sel}, key=lambda s: [int(x[1:]) for x in s.split("+")])
    cols = 7
    nrows = math.ceil(len(sets) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(2.3 * cols, 2.0 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(sets):]:
        ax.set_visible(False)
    for ax, s in zip(axes.flat, sets):
        _scale_lines(ax, [r for r in sel if r["set"] == s], "lret_mean", colors)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.axhline(1, color=INK_2, linewidth=0.8, linestyle="--")
        ax.set_ylim(-1.5, 1.2)
        ax.set_title(s, fontsize=8, color=INK, loc="left")
        _style(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower right", ncol=len(labels))
    fig.suptitle("pairs: mean log retention against the scaling coefficient (cut at -1.5)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "scale_pairs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_matrix(rows, out, method=None):
    sel = [r for r in rows if len(r["rules"]) == 2 and (method is None or r["method"] == method)]
    best = best_per(sel, lambda r: r["set"])
    if not best:
        return None
    m = np.full((8, 8), np.nan)
    for r in best.values():
        a, b = r["rules"]
        i, j = RULES.index(a), RULES.index(b)
        m[i, j] = r.get(f"lret_{a}", np.nan)
        m[j, i] = r.get(f"lret_{b}", np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(np.clip(m, -1, 1), cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            v = m[i, j]
            text = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            dark = np.isfinite(v) and abs(v) > 0.6
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if dark else INK)
    ax.set_xticks(range(8), RULES, fontsize=8, color=INK_2)
    ax.set_yticks(range(8), RULES, fontsize=8, color=INK_2)
    ax.set_xlabel("merged with", fontsize=8, color=INK_2)
    ax.set_ylabel("log retention of", fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, 8), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.set_label("1 = as the specialist, 0 = as the base (colors cut at +-1)", fontsize=7, color=INK_2)
    name = method or "any"
    fig.suptitle(f"pairs, best setting of {'each pair' if method is None else method}: "
                 f"how much of each rule survives", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = out / f"matrix_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_pairs(path):
    with open(path, newline="") as f:
        return {r["pair"].replace("-", "+"): {k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)}


def plot_scatter(rows, pairs, out, method=None):
    sel = [r for r in rows if len(r["rules"]) == 2 and (method is None or r["method"] == method)]
    best = best_per(sel, lambda r: r["set"])
    sets = [s for s in sorted(best) if s in pairs]
    if len(sets) < 3:
        return None, []
    y = [best[s]["lret_mean"] for s in sets]
    metrics = [m for m in PREDICTORS if m in next(iter(pairs.values()))]
    fig, axes = plt.subplots(2, math.ceil(len(metrics) / 2), figsize=(3.0 * math.ceil(len(metrics) / 2), 5.6),
                             squeeze=False)
    corr = []
    for ax, m in zip(axes.flat, metrics):
        x = [pairs[s][m] for s in sets]
        rho = spearman(x, y)
        corr.append({"metric": m, "spearman": rho, "pairs": len(sets)})
        ax.scatter(x, np.clip(y, -1.5, 1.2), s=14, color=SERIES[0], edgecolors="white", linewidths=0.5)
        ax.set_title(f"{m}  (rho {rho:+.2f})", fontsize=8, color=INK, loc="left")
        _style(ax)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("best mean log retention", fontsize=7, color=INK_2)
    name = method or "any"
    fig.suptitle(f"mergeability metrics against the merge result ({name}, {len(sets)} pairs, y cut at -1.5)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    path = out / f"scatter_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    with open(out / f"predictors_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "spearman", "pairs"])
        w.writeheader()
        w.writerows(corr)
    return path, corr


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures and tables of the merges.")
    ap.add_argument("--summary", required=True, help="merge_summary.csv written by tools.merge")
    ap.add_argument("--pairs", default=None, help="<tag>_pairs.csv written by tools.mergeability")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="results/merges")
    args = ap.parse_args(argv)

    rows = load_summary(args.summary, args.split, args.steps, args.n)
    if not rows:
        raise SystemExit(f"no merges with split={args.split}, steps={args.steps}, n={args.n} in {args.summary}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = write_best_settings(rows, out)
    methods = sorted({m for _, m in best}, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    sets = sorted({s for s, _ in best}, key=lambda s: (s != "joint_full", s))
    print(f"best mean log retention per set and method ({args.split}, ticks 1-{args.steps}, {args.n} scenes)")
    print(f"{'set':12s} " + " ".join(f"{m:>8s}" for m in methods))
    for st in sets:
        print(f"{st:12s} " + " ".join(f"{best[(st, m)]['lret_mean']:8.2f}" if (st, m) in best else f"{'':>8s}"
                                      for m in methods))
    print()

    saved = [plot_scale_all(rows, out), plot_scale_pairs(rows, out)]
    for method in [None] + methods:
        saved.append(plot_matrix(rows, out, method))
    if args.pairs:
        pairs = load_pairs(args.pairs)
        for method in [None] + methods:
            path, corr = plot_scatter(rows, pairs, out, method)
            saved.append(path)
            if corr:
                print(f"rank correlation with the best mean log retention ({method or 'any method'}):  "
                      + "  ".join(f"{c['metric']} {c['spearman']:+.2f}" for c in corr))
    for p in saved:
        if p:
            print(f"saved {p}")
    return best


if __name__ == "__main__":
    main()