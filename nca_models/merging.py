"""Model merging of specialists fine-tuned from the same base.

Every method works on task vectors, tau_i = specialist_i - base, taken over the whole state dict
(tensors that the fine-tuning never changes have tau = 0, so they stay as in the base), and
returns the merged weights as base + something built from the taus:

    average   base + sum_i w_i tau_i with sum_i w_i = 1 (uniform by default): the plain mean of
              the specialists, the gentlest method on emergent capabilities (paper 25)
    ta        base + alpha * sum_i tau_i, or one alpha per rule (task arithmetic, paper 15)
    ties      trim every tau to its top k share by magnitude, elect the sign of every weight by
              the summed mass, average only the values that agree with it (disjoint mean), then
              base + lam * merged (TIES, paper 17)
    dare      drop a random share p of every tau and rescale the rest by 1 / (1 - p), then task
              arithmetic with alpha (DARE + TA, paper 18); the drop is seeded, so it is repeatable
    tsv       for every weight matrix: SVD of every tau, keep the top 1/T of the singular vectors
              of each (T = number of rules), make all the kept left and right vectors orthogonal
              with the closest orthogonal matrix (Procrustes), rebuild and add alpha times it to
              the base; vectors (biases) take the mean of the taus times alpha (TSV-M, paper 24,
              which uses alpha = 1)

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


def _flat_keys(base_sd):
    return [k for k, v in base_sd.items() if v.is_floating_point()]


def merge_ties(base_sd, taus, k=0.2, lam=1.0):
    keys = _flat_keys(base_sd)
    shapes = [base_sd[key].shape for key in keys]
    flat = torch.stack([torch.cat([tv[key].flatten() for key in keys]) for tv in taus.values()])   # (T, P)
    trimmed = torch.zeros_like(flat)
    keep = max(1, int(round(k * flat.shape[1])))
    for i in range(flat.shape[0]):
        idx = flat[i].abs().topk(keep).indices
        trimmed[i, idx] = flat[i, idx]
    sign = torch.sign(trimmed.sum(0))
    agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
    merged = (trimmed * agree).sum(0) / agree.sum(0).clamp(min=1)
    out = {key: v.clone() for key, v in base_sd.items()}
    start = 0
    for key, shape in zip(keys, shapes):
        n = shape.numel()
        out[key] += lam * merged[start:start + n].view(shape)
        start += n
    return out


def merge_dare(base_sd, taus, p=0.3, alpha=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dropped = {}
    for rule, tv in taus.items():
        dropped[rule] = {}
        for key, t in tv.items():
            mask = (torch.rand(t.shape, generator=gen) >= p).to(t.dtype)
            dropped[rule][key] = t * mask / (1 - p)
    return merge_ta(base_sd, dropped, alpha)


def _closest_orthogonal(m):
    """Orthogonal matrix closest to m (Procrustes): P Q^T from the SVD m = P S Q^T."""
    p_, _, qh = torch.linalg.svd(m, full_matrices=False)
    return p_ @ qh


def merge_tsv(base_sd, taus, alpha=1.0, keep=None):
    t = len(taus)
    keep = keep or 1 / t
    out = {key: v.clone() for key, v in base_sd.items()}
    for key in _flat_keys(base_sd):
        mats = [tv[key].double() for tv in taus.values()]
        if mats[0].dim() != 2:
            out[key] += alpha * (sum(mats) / t).to(out[key].dtype)
            continue
        r = max(1, int(min(mats[0].shape) * keep))
        us, ss, vs = [], [], []
        for m in mats:
            u, s, vh = torch.linalg.svd(m, full_matrices=False)
            us.append(u[:, :r])
            ss.append(s[:r])
            vs.append(vh[:r].T)
        u, s, v = torch.cat(us, 1), torch.cat(ss), torch.cat(vs, 1)
        rebuilt = _closest_orthogonal(u) @ torch.diag(s) @ _closest_orthogonal(v).T
        out[key] += alpha * rebuilt.to(out[key].dtype)
    return out


MERGERS = {"average": merge_average, "ta": merge_ta, "ties": merge_ties, "dare": merge_dare, "tsv": merge_tsv}


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