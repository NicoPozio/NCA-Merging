"""Mergeability metrics: can we tell, before merging, which pairs of rules will merge well?

All metrics are computed on the specialists and the base only, never on a merge, so they can be
used later to predict the merge results (L1 regression with leave-one-task-out, paper 23).
They look at a pair of rules at three levels:

weights (task vector tau = specialist - base)
- cos, cos_W1, cos_W2: cosine between the task vectors, whole and per layer (one of the 28
  metrics of paper 23; paper 16 says orthogonality is not the right condition, so we expect it
  to predict little and keep it as a reference);
- top_overlap, sign_conflict: share of weights in the top 20% by |tau| of both rules, and among
  those the share with opposite sign (the trim and elect steps of TIES, paper 17);
- sti_W1, sti_W2: singular task interference of the pair (paper 24) with the singular vectors
  holding 95% of each norm, relative to the sum of the singular values kept, so pairs with
  different norms can be compared (our variant);
- sti_paper_W1, sti_paper_W2: the same as in paper 24, keeping a fixed share 1/T of the singular
  vectors of every task as TSV-M does (T = number of rules) and with no normalization;
- sub_left_*, sub_right_*: overlap of the top-k left and right singular subspaces of the two
  dW, ||U_i^T U_j||_F^2 / k, in [0, 1] (subspace overlap of paper 23, same k as sti_paper);
  per rule also the effective rank of dW (how many directions the rule really uses).

function (gradients)
- grad_cos, grad_l2, grad_dot: gradients of the one-tick loss with respect to the weights, each
  specialist at its own weights, on one calibration set shared by the two (states of scenes with
  every rule active): cosine, L2 distance and dot product (encoder gradients of paper 23);
- igrad_cos, igrad_l2, igrad_dot: the same with the gradients with respect to the input state
  (input gradients of paper 23);
- grad_cos_base: cosine between the gradients of the one-tick loss on the cells where rule i acts
  and on the cells where rule j acts, both at the base weights: do the two rules ask the base to
  move in the same direction (our variant, closer to gradient conflict in multi-task learning);
- cross_leak: how much specialist j changes the update of the base on the cells where only rule
  i acts, relative to how much it changes it where its own rule acts, averaged over the two
  directions (weight disentanglement of paper 16, measured with the support mask);
  per rule also locality: the same ratio for a specialist outside its own support.

simulator (exact, on scenes where every rule is active)
- support_jaccard, support_overlap: overlap of the (scene, tick, cell) sets where the two rules
  act, as Jaccard and as share of the smaller set (the rare rules are tiny);
- channel_cos: cosine between the per-channel write profiles of the two rules;
- cowrite: share of the cells where at least one of the two acts in which both write the same
  channel in the same tick.
"""

import itertools
from dataclasses import dataclass

import torch

from nca_sandbox.config import C, F, N_CHANNELS, N_MATERIALS, N_RULES, O, RULE_IDS, W, WX
from nca_sandbox.events import batch_event_masks
from nca_sandbox.simulator import border_mask, inject_events

LAYERS = ("W1", "b1", "W2", "b2")
MATRICES = ("W1", "W2")
RULES = tuple(f"T{i + 1}" for i in range(N_RULES))


def task_vector(spec, base):
    return {k: (getattr(spec, k) - getattr(base, k)).detach().float().cpu() for k in LAYERS if hasattr(spec, k)}


def _flat(tv, keys=None):
    return torch.cat([tv[k].flatten() for k in (keys or tv)])


def _cos(a, b):
    den = a.norm() * b.norm()
    return float(a @ b / den) if den > 0 else float("nan")


def top_mask(v, share):
    """Weights in the top `share` by absolute value (exactly that many, ties broken by index)."""
    k = max(1, int(round(share * v.numel())))
    mask = torch.zeros_like(v, dtype=torch.bool)
    mask[v.abs().topk(k).indices] = True
    return mask


def effective_rank(m, energy=0.95):
    """Smallest number of singular values holding `energy` of the squared Frobenius norm."""
    s = torch.linalg.svdvals(m.double())
    if s.sum() == 0:
        return 0
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((e < energy).sum()) + 1


def _svd_kept(m, energy):
    u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
    r = max(1, effective_rank(m, energy))
    return u[:, :r], s[:r], vh[:r].T


def sti(mats, energy=0.95):
    """Singular task interference (paper 24) of a list of dW of the same layer, relative to the
    sum of the singular values kept: ||(U^T U - I) S (V^T V - I)||_1 / sum(S).
    Every task keeps the singular vectors that hold `energy` of its norm."""
    parts = [_svd_kept(m, energy) for m in mats]
    u = torch.cat([p[0] for p in parts], 1)
    s = torch.cat([p[1] for p in parts])
    v = torch.cat([p[2] for p in parts], 1)
    eye = torch.eye(len(s), dtype=u.dtype)
    val = ((u.T @ u - eye) @ torch.diag(s) @ (v.T @ v - eye)).abs().sum()
    return float(val / s.sum()) if s.sum() > 0 else float("nan")


def kept_rank(m, keep):
    """Number of singular vectors a share `keep` of the smaller side gives (at least 1)."""
    return max(1, int(min(m.shape) * keep))


def sti_paper(mats, keep):
    """STI as in paper 24: every task keeps the top share `keep` of its singular vectors (1/T in
    TSV-M), no normalization: ||(U^T U - I) S (V^T V - I)||_1."""
    us, ss, vs = [], [], []
    for m in mats:
        u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
        k = kept_rank(m, keep)
        us.append(u[:, :k])
        ss.append(s[:k])
        vs.append(vh[:k].T)
    u, s, v = torch.cat(us, 1), torch.cat(ss), torch.cat(vs, 1)
    eye = torch.eye(len(s), dtype=u.dtype)
    return float(((u.T @ u - eye) @ torch.diag(s) @ (v.T @ v - eye)).abs().sum())


def subspace_overlap(a, b, k):
    """Overlap of the top-k left and right singular subspaces of two matrices, in [0, 1]."""
    ua, _, vha = torch.linalg.svd(a.double(), full_matrices=False)
    ub, _, vhb = torch.linalg.svd(b.double(), full_matrices=False)
    left = float((ua[:, :k].T @ ub[:, :k]).pow(2).sum() / k)
    right = float((vha[:k] @ vhb[:k].T).pow(2).sum() / k)
    return left, right


def weight_metrics(ti, tj, top=0.2, energy=0.95, keep=1 / N_RULES):
    a, b = _flat(ti), _flat(tj)
    ma, mb = top_mask(a, top), top_mask(b, top)
    both = ma & mb
    out = {"cos": _cos(a, b), "top_overlap": float(both.sum() / (ma | mb).sum()),
           "sign_conflict": float((torch.sign(a[both]) != torch.sign(b[both])).float().mean()) if both.any()
           else float("nan")}
    for k in MATRICES:
        if k in ti:
            out[f"cos_{k}"] = _cos(ti[k].flatten(), tj[k].flatten())
            out[f"sti_{k}"] = sti([ti[k], tj[k]], energy)
            out[f"sti_paper_{k}"] = sti_paper([ti[k], tj[k]], keep)
            out[f"sub_left_{k}"], out[f"sub_right_{k}"] = subspace_overlap(ti[k], tj[k], kept_rank(ti[k], keep))
    return out


def _states(split, n, ticks, chunk, device):
    """Yields (state before the tick, event masks of the tick, active rules, simulator info,
    state after) for the first n scenes of a split, chunk by chunk and tick by tick."""
    sim = split.sim
    n = min(n, len(split.data["initial"]))
    for start in range(0, n, chunk):
        idx = list(range(start, min(n, start + chunk)))
        x = split.data["initial"][idx].to(device)
        active = split.data["active"][idx].to(device)
        events = [split.data["events"][i] for i in idx]
        masks = batch_event_masks(events, ticks, *x.shape[-2:], device)
        for t in range(ticks):
            m = masks[:, t]
            nxt, info = sim.step(x, active, m, diagnostics=True)
            yield x, m, active, info, nxt
            x = nxt


def rule_gradient(model, split, rule, n=64, ticks=64, chunk=32, device="cpu"):
    """Gradient of the one-tick squared error on the cells where `rule` acts, at the weights of
    `model`, averaged over those cells. Returns {layer: tensor}."""
    model = model.to(device).train()
    params = {k: getattr(model, k) for k in LAYERS if hasattr(model, k)}
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    r = RULE_IDS[rule]
    cnt = 0.0
    for x, m, active, info, nxt in _states(split, n, ticks, chunk, device):
        sup = info.support[:, r].unsqueeze(1)                                   # (b, 1, H, W)
        if not sup.any():
            continue
        pred = model(x, m)
        loss = (((pred[:, :WX] - nxt[:, :WX]) ** 2) * sup).sum() / WX
        loss.backward()
        cnt += float(sup.sum())
    grads = {k: (p.grad.detach().float().cpu() / max(cnt, 1.0)) if p.grad is not None else torch.zeros_like(p).cpu()
             for k, p in params.items()}
    model.zero_grad(set_to_none=True)
    return grads


@torch.no_grad()
def calibration_set(split, n=32, ticks=64, every=8, device="cpu"):
    """States shared by every model for the gradient metrics: the first n scenes of the split
    (the joint_full one, every rule active), one state every `every` ticks.
    Returns (state, event masks, next state of the simulator), each stacked over the states."""
    xs, ms, ys = [], [], []
    t = 0
    for x, m, _, _, nxt in _states(split, n, ticks, max(1, n), device):
        if t % every == 0:
            xs.append(x.cpu())
            ms.append(m.cpu())
            ys.append(nxt.cpu())
        t = (t + 1) % ticks
    return torch.cat(xs), torch.cat(ms), torch.cat(ys)


def model_gradients(model, cal, chunk=64, device="cpu"):
    """Gradients of the one-tick squared error on the calibration set (mean over cells and
    channels W .. S) with respect to the weights and to the input state, at the weights of
    `model`. Returns ({layer: tensor}, input gradient (N, 20, H, W))."""
    model = model.to(device).train()
    params = {k: getattr(model, k) for k in LAYERS if hasattr(model, k)}
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    xs, ms, ys = cal
    total = xs.shape[0] * xs.shape[-1] * xs.shape[-2] * WX
    igrads = []
    for start in range(0, xs.shape[0], chunk):
        x = xs[start:start + chunk].to(device).requires_grad_(True)
        pred = model(x, ms[start:start + chunk].to(device))
        loss = ((pred[:, :WX] - ys[start:start + chunk].to(device)[:, :WX]) ** 2).sum() / total
        loss.backward()
        igrads.append(x.grad.detach().cpu())
    grads = {k: p.grad.detach().float().cpu() for k, p in params.items()}
    model.zero_grad(set_to_none=True)
    return grads, torch.cat(igrads)


def gradient_metrics(gi, gj, prefix):
    a, b = gi.flatten().double(), gj.flatten().double()
    return {f"{prefix}_cos": _cos(a, b), f"{prefix}_l2": float((a - b).norm()), f"{prefix}_dot": float(a @ b)}


def _write_masks(sim, x, m, active):
    """(b, 8, 20, H, W) bool: which channel every rule writes in every cell in this tick.
    A discrete change counts on the channels it rewrites: fire for ignition and extinction,
    materials, C, W and O for a change of material."""
    s = inject_events(x, m, sim.p)
    _, outs = sim.rule_outputs(s, active.to(torch.bool))
    eps = sim.p.eps
    masks = []
    for o in outs:
        w = o.delta.abs() > eps
        if o.ignite is not None or o.extinguish is not None:
            fire = torch.zeros_like(w[:, 0])
            if o.ignite is not None:
                fire |= o.ignite
            if o.extinguish is not None:
                fire |= o.extinguish
            w[:, F] |= fire
        if o.mat_target is not None:
            changed = o.mat_target >= 0
            w[:, :N_MATERIALS] |= changed.unsqueeze(1)
            for ch in (C, W, O):
                w[:, ch] |= changed
        masks.append(w)
    b = border_mask(*x.shape[-2:], x.device)
    return torch.stack(masks, 1) & ~b


@dataclass
class SimStats:
    support: torch.Tensor        # (8,) cells where each rule acts
    inter: torch.Tensor          # (8, 8) cells where both act
    cowrite: torch.Tensor        # (8, 8) cells where both write the same channel
    profile: torch.Tensor        # (8, 20) how often each rule writes each channel
    change_on: torch.Tensor      # (8,) mean |delta_spec - delta_base| of specialist k on its own support
    change_off: torch.Tensor     # (8,) the same outside its support
    change_cross: torch.Tensor   # (8, 8) [i, j]: specialist j on the cells where only rule i acts


@torch.no_grad()
def simulator_stats(split, specialists, base, n=64, ticks=64, chunk=16, device="cpu"):
    """One pass on scenes where every rule is active (the joint_full split).
    specialists: {rule: model}, the rules missing from it are left at zero in the change terms."""
    sim = split.sim
    z8 = lambda *s: torch.zeros(*s, dtype=torch.float64)
    st = SimStats(z8(8), z8(8, 8), z8(8, 8), z8(8, N_CHANNELS), z8(8), z8(8), z8(8, 8))
    on_cnt, off_cnt, cross_cnt = z8(8), z8(8), z8(8, 8)
    base = base.to(device).eval()
    specs = {RULE_IDS[r]: mdl.to(device).eval() for r, mdl in specialists.items()}
    for x, m, active, info, _ in _states(split, n, ticks, chunk, device):
        sup = info.support                                                       # (b, 8, H, W)
        wm = _write_masks(sim, x, m, active)                                     # (b, 8, 20, H, W)
        supf = sup.double()
        st.support += supf.sum((0, 2, 3)).cpu()
        st.inter += torch.einsum("bihw,bjhw->ij", supf, supf).cpu()
        st.profile += wm.double().sum((0, 3, 4)).cpu()
        wf = wm.double()
        st.cowrite += (torch.einsum("bichw,bjchw->bijhw", wf, wf) > 0).double().sum((0, 3, 4)).cpu()

        s = inject_events(x, m, sim.p)
        d0 = base.delta(s)[:, :WX]
        for k, spec in specs.items():
            change = (spec.delta(s)[:, :WX] - d0).abs().mean(1)                  # (b, H, W)
            own = sup[:, k]
            st.change_on[k] += float((change * own).sum())
            on_cnt[k] += float(own.sum())
            st.change_off[k] += float((change * ~own).sum())
            off_cnt[k] += float((~own).sum())
            only = sup & ~own.unsqueeze(1)                                       # (b, 8, H, W): rule i, not k
            st.change_cross[:, k] += torch.einsum("bihw,bhw->i", only.double(), change.double()).cpu()
            cross_cnt[:, k] += only.double().sum((0, 2, 3)).cpu()
    st.change_on /= on_cnt.clamp(min=1)
    st.change_off /= off_cnt.clamp(min=1)
    st.change_cross /= cross_cnt.clamp(min=1)
    st.profile /= st.support.clamp(min=1)[:, None]
    return st


def pair_table(task_vectors, base_grads, spec_grads, stats: SimStats, top=0.2, energy=0.95, keep=1 / N_RULES):
    """One row per pair of rules (28 for 8 rules) with every metric.
    base_grads: {rule: gradients at the base on the rule's cells}; spec_grads: {rule: (weight
    gradients, input gradients) of the specialist on the shared calibration set}."""
    rows = []
    rules = [r for r in RULES if r in task_vectors]
    for ri, rj in itertools.combinations(rules, 2):
        i, j = RULE_IDS[ri], RULE_IDS[rj]
        row = {"pair": f"{ri}-{rj}", "rule_i": ri, "rule_j": rj}
        row.update(weight_metrics(task_vectors[ri], task_vectors[rj], top, energy, keep))
        row.update(gradient_metrics(_flat(spec_grads[ri][0]), _flat(spec_grads[rj][0]), "grad"))
        row.update(gradient_metrics(spec_grads[ri][1], spec_grads[rj][1], "igrad"))
        row["grad_cos_base"] = _cos(_flat(base_grads[ri]), _flat(base_grads[rj]))
        leak = []
        for a, b in ((i, j), (j, i)):
            if stats.change_on[b] > 0:
                leak.append(float(stats.change_cross[a, b] / stats.change_on[b]))
        row["cross_leak"] = sum(leak) / len(leak) if leak else float("nan")
        inter, si, sj = float(stats.inter[i, j]), float(stats.support[i]), float(stats.support[j])
        union = si + sj - inter
        row["support_jaccard"] = inter / union if union > 0 else float("nan")
        row["support_overlap"] = inter / min(si, sj) if min(si, sj) > 0 else float("nan")
        row["channel_cos"] = _cos(stats.profile[i], stats.profile[j])
        row["cowrite"] = float(stats.cowrite[i, j]) / union if union > 0 else float("nan")
        rows.append(row)
    return rows


def rule_table(task_vectors, base_grads, spec_grads, stats: SimStats, energy=0.95):
    """One row per rule: norms, effective ranks, locality, how rare the rule is."""
    rows = []
    total = float(stats.support.sum())
    for r, tv in task_vectors.items():
        k = RULE_IDS[r]
        row = {"rule": r, "tau_norm": float(_flat(tv).norm()), "grad_base_norm": float(_flat(base_grads[r]).norm()),
               "grad_norm": float(_flat(spec_grads[r][0]).norm())}
        for m in MATRICES:
            if m in tv:
                row[f"rank_{m}"] = effective_rank(tv[m], energy)
        row["locality"] = float(stats.change_off[k] / stats.change_on[k]) if stats.change_on[k] > 0 else float("nan")
        row["support_share"] = float(stats.support[k]) / total if total > 0 else float("nan")
        rows.append(row)
    return rows