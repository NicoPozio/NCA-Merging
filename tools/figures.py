"""Figures for the results: error vs tick, rule x model heatmap, frame grids, channel drift.

Colors follow a fixed categorical order, so a model keeps its color in every figure; magnitudes
use a single blue ramp, light to dark.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from nca_models.evaluate import rule_curve
from nca_sandbox.config import CHANNEL_NAMES, E, F, N_MATERIALS, O, S, V, W

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = "#52514e"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
GROUP_ORDER = ("base", "specialists", "scratch", "joint_single", "joint_full")

# material colors, then the overlays painted on top in this order
MATERIAL_RGB = np.array([
    (150, 120, 80), (110, 170, 70), (135, 135, 135), (60, 60, 60),
    (30, 100, 40), (185, 195, 205), (185, 230, 250), (230, 90, 20)]) / 255
OVERLAYS = ((O, (70, 45, 20), 0.8), (W, (40, 90, 200), 1.0), (V, (245, 245, 245), 0.6),
            (S, (90, 90, 90), 0.6), (F, (255, 140, 0), 1.0), (E, (255, 240, 60), 1.0))


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def colors_for(labels):
    """One fixed color per label, in order; more than 8 series do not fit one chart."""
    if len(labels) > len(SERIES):
        raise ValueError(f"{len(labels)} series: split them over more figures (max {len(SERIES)})")
    return dict(zip(labels, SERIES))


def sort_records(recs):
    def key(r):
        g = GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else len(GROUP_ORDER)
        return (r["run"], g, r["name"])
    return sorted(recs, key=key)


def plot_curves(recs, out, metric="mse_rule", title=None, train_horizon=64, smooth=1):
    """Error vs tick, one line per record (log scale). metric: mse_rule, mse_all or mat_err."""
    recs = sort_records(recs)
    labels = [r["label"] for r in recs]
    col = colors_for(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in recs:
        y = rule_curve(r) if metric == "mse_rule" else r[metric]
        y = y.numpy().astype(float)
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            y = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), kernel, mode="same")
        ax.plot(np.arange(1, len(y) + 1), y, color=col[r["label"]], linewidth=1.5, label=r["label"])
    ax.axvline(train_horizon, color=INK_2, linewidth=0.8, linestyle="--")
    ax.text(train_horizon, 1.0, " training horizon", transform=ax.get_xaxis_transform(),
            color=INK_2, fontsize=7, va="top")
    if metric != "mat_err":
        ax.set_yscale("log")
    ax.set_xlabel("tick", color=INK_2, fontsize=9)
    ax.set_ylabel(metric, color=INK_2, fontsize=9)
    ax.set_title(title or metric, color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="best")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_heatmap(values, rows, cols, out, title=""):
    """values: dict (row, col) -> float. Log color scale, number written in every cell."""
    m = np.array([[values.get((r, c), np.nan) for c in cols] for r in rows], dtype=float)
    finite = m[np.isfinite(m) & (m > 0)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(cols), 0.9 + 0.45 * len(rows)))
    if finite.size:
        norm = LogNorm(vmin=finite.min(), vmax=max(finite.max(), finite.min() * 1.01))
        ax.imshow(np.where(np.isfinite(m) & (m > 0), m, np.nan), cmap=BLUES, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = m[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=INK_2)
                continue
            dark = finite.size and math.log(v / finite.min() + 1e-12) > 0.55 * math.log(finite.max() / finite.min() + 1e-12)
            ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7, color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK_2)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def render_rgb(state):
    """(20, H, W) state -> (H, W, 3) image: material color, then water, oil, vapor, smoke, fire and
    charge painted on top with their intensity."""
    x = state.detach().float().cpu().clamp(0, 1).numpy()
    mats = x[:N_MATERIALS]
    rgb = np.einsum("mhw,mc->hwc", mats, MATERIAL_RGB) / np.maximum(mats.sum(0), 1e-6)[..., None]
    for ch, color, strength in OVERLAYS:
        a = (strength * x[ch])[..., None]
        rgb = (1 - a) * rgb + a * (np.array(color) / 255)
    return rgb.clip(0, 1)


def plot_frames(frames_by_label, out, title=""):
    """frames_by_label: label -> frames from evaluate.frames_for. First row is the simulator."""
    labels = list(frames_by_label)
    first = frames_by_label[labels[0]]
    ticks = first["ticks"]
    rows = [("simulator", first["true"])] + [(lb, frames_by_label[lb]["pred"]) for lb in labels]
    fig, axes = plt.subplots(len(rows), len(ticks), figsize=(1.7 * len(ticks), 1.8 * len(rows)), squeeze=False)
    for i, (name, states) in enumerate(rows):
        for j, t in enumerate(ticks):
            ax = axes[i, j]
            ax.imshow(render_rgb(states[t]), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"t = {t}", fontsize=8, color=INK_2)
            if j == 0:
                ax.set_ylabel(name, fontsize=7, color=INK_2, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_drift(recs, channels, out, title=""):
    """Mean value of some channels over the ticks, NCA (solid) against the simulator (dashed),
    one panel per channel. This is where a degenerating rollout shows its slow drift."""
    recs = sort_records(recs)
    if len({(r["dataset"], r["split"], r["n"]) for r in recs}) > 1:
        raise ValueError("drift plots compare models on the same scenes: one dataset, split and n")
    col = colors_for([r["label"] for r in recs])
    idx = [CHANNEL_NAMES.index(c) for c in channels]
    fig, axes = plt.subplots(1, len(idx), figsize=(3.4 * len(idx), 3.0), squeeze=False)
    for ax, c, name in zip(axes[0], idx, channels):
        truth = recs[0]["mean_true"][:, c].numpy()
        ax.plot(np.arange(1, len(truth) + 1), truth, color=NEUTRAL, linewidth=1.5, linestyle="--", label="simulator")
        for r in recs:
            y = r["mean_pred"][:, c].numpy()
            ax.plot(np.arange(1, len(y) + 1), y, color=col[r["label"]], linewidth=1.5, label=r["label"])
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("tick", fontsize=8, color=INK_2)
        _style(ax)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)