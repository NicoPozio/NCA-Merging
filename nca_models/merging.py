"""Model merging of specialists fine-tuned from the same base.

Every method works on task vectors, tau_i = specialist_i - base, taken over the whole state dict
(tensors that the fine-tuning never changes have tau = 0, so they stay as in the base), and
returns the merged weights as base + something built from the taus:

    average   base + sum_i w_i tau_i with sum_i w_i = 1 (uniform by default): the plain mean of
              the specialists, the gentlest method on emergent capabilities (paper 25)
    ta        base + alpha * sum_i tau_i, or one alpha per rule (task arithmetic, paper 15)

A new method is a function (base_sd, taus, **hp) -> state dict registered in MERGERS, so the
merging script, the checkpoints and the evaluation do not change when TIES, DARE or TSV arrive.
"""

import copy

import torch

from .model import load_nca, save_nca


def state(model):
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}


def task_vectors(base, specialists):
    """{rule: {key: tau}} for {rule: model} fine-tuned from `base` (models or state dicts)."""
    b = base if isinstance(base, dict) else state(base)
    out = {}
    for rule, spec in specialists.items():
        s = spec if isinstance(spec, dict) else state(spec)
        if s.keys() != b.keys():
            raise ValueError(f"{rule}: the specialist and the base have different parameters")
        out[rule] = {k: s[k] - b[k] for k in b}
    return out


def _combine(base_sd, taus, coef):
    """base + sum_i coef[i] * tau_i."""
    out = {k: v.clone() for k, v in base_sd.items()}
    for rule, tv in taus.items():
        for k, t in tv.items():
            if out[k].is_floating_point():
                out[k] += coef[rule] * t
    return out


def merge_average(base_sd, taus, weights=None):
    """Weighted mean of the specialists: weights are normalized to sum to 1."""
    w = {r: 1.0 for r in taus} if weights is None else {r: float(weights[r]) for r in taus}
    total = sum(w.values())
    return _combine(base_sd, taus, {r: v / total for r, v in w.items()})


def merge_ta(base_sd, taus, alpha=1.0):
    """Task arithmetic: one alpha for every rule, or a dict {rule: alpha}."""
    coef = {r: float(alpha[r]) for r in taus} if isinstance(alpha, dict) else {r: float(alpha) for r in taus}
    return _combine(base_sd, taus, coef)


MERGERS = {"average": merge_average, "ta": merge_ta}


def merge(base_model, taus, method, **hp):
    """A new model with the merged weights (the base model is not touched)."""
    if method not in MERGERS:
        raise ValueError(f"unknown method {method}: choose from {sorted(MERGERS)}")
    merged = copy.deepcopy(base_model)
    merged.load_state_dict(MERGERS[method](state(base_model), taus, **hp))
    return merged


def dataset_for(rules, n_rules=8):
    """Name of the dataset where exactly these rules are active: T1, T1+T6, ..., joint_full."""
    order = sorted(rules, key=lambda r: int(r[1:]))
    if len(order) == n_rules:
        return "joint_full"
    return "+".join(order)


def hp_tag(method, hp):
    """Short, file-safe name of a merge setting, e.g. ta_a0.5."""
    if not hp:
        return method
    parts = []
    for k, v in sorted(hp.items()):
        if isinstance(v, dict):
            v = "-".join(f"{r}{x:g}" for r, x in sorted(v.items()))
        elif isinstance(v, float):
            v = f"{v:g}"
        parts.append(f"{k[0]}{v}")
    return f"{method}_{'_'.join(parts)}"


def save_merge(path, model, base_path, base_meta, rules, method, hp, specialist_paths):
    """Saved like a trained checkpoint, so tools.results evaluates it as one more model.
    `rules` is the dataset the merge is meant for, `merge` records how it was built."""
    save_nca(path, model, rules=dataset_for(rules), init=str(base_path), trainable="all",
             train_cfg=base_meta.get("train_cfg", {}),
             merge={"method": method, "hp": hp, "rules": list(rules),
                    "specialists": {r: str(p) for r, p in specialist_paths.items()}})


def load_specialists(spec_dir, rules, seed=0, which="best", device="cpu"):
    """{rule: (model, path)} from a folder of specialists."""
    out = {}
    for r in rules:
        path = spec_dir / f"{r}_s{seed}{'_best' if which == 'best' else ''}.pt"
        out[r] = (load_nca(path, device)[0], path)
    return out