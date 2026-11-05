"""A pool of simulator trajectories from which training windows are cut.

Generating a fresh trajectory for every step would make the simulator the bottleneck, so we keep
a pool of long trajectories, cut windows of k ticks starting anywhere along them (also mid-way,
as NoiseNCA recommends), and replace a few scenes every so often.

Rare rules (ice, lava, fuel) act on a tiny share of the cells, so the loss would be dominated by
the base and the specialist would learn almost nothing. Two levers fix that: cells where an active
rule acts get a weight chosen so that they make up a fixed share of the loss, whatever the rule,
and part of the windows are drawn where the rules act the most.
"""

import torch

from nca_sandbox.dataset import SandboxGenerator


class WindowPool:
    def __init__(self, gen: SandboxGenerator, rules, n_scenes=128, ticks=96, size=None,
                 ambiguity_mask=False, support_share=0.0, support_max_weight=200.0, support_sampling=0.0,
                 storage_dtype=torch.float16):
        """support_share: share of the loss that goes to the cells where an active rule acts
        (0 = plain MSE); their weight is computed from how rare they are, capped at support_max_weight.
        support_sampling: share of windows drawn in proportion to how much the rules act in them.
        ambiguity_mask (only meaningful for joint_single): for every tick, mark the cells where a
        rule that is off in that scene would have acted. With one rule on per scene and no rule label
        those cells have contradictory targets, so the loss ignores them."""
        self.gen, self.rules, self.n, self.ticks = gen, rules, n_scenes, ticks
        self.size = size or gen.cfg.size
        self.ambiguity_mask = ambiguity_mask
        self.support_share, self.support_max_weight = support_share, support_max_weight
        self.support_sampling = support_sampling
        self.use_support = rules != "base" and (support_share > 0 or support_sampling > 0)
        self.dtype = storage_dtype
        self.traj = self.masks = self.sup = self.keep = self.counts = None
        self._fill(torch.arange(n_scenes))

    @torch.no_grad()
    def _ambiguous(self, batch):
        """(B, ticks, H, W) bool: some inactive rule would act on the cell at that tick."""
        sim = self.gen.sim
        all_on = torch.ones_like(batch.active)
        out = []
        for t in range(self.ticks):
            _, info = sim.step(batch.trajectory[:, t], all_on, batch.event_masks[:, t], diagnostics=True)
            others = info.support & ~batch.active[:, :, None, None]
            out.append(others.any(1))
        return torch.stack(out, 1)

    @torch.no_grad()
    def _fill(self, idx):
        b = self.gen.sample(len(idx), self.rules, steps=self.ticks, keep_info=self.use_support, size=self.size)
        traj = b.trajectory.to(self.dtype)
        sup = counts = keep = None
        if self.use_support:
            sup = (b.support & b.active[:, None, :, None, None]).any(2)     # (B, ticks, H, W)
            counts = sup.flatten(2).sum(2).float()                           # (B, ticks)
        if self.ambiguity_mask:
            keep = ~self._ambiguous(b)
        if self.traj is None:
            self.traj, self.masks, self.active = traj, b.event_masks, b.active
            self.sup, self.counts, self.keep = sup, counts, keep
        else:
            self.traj[idx], self.masks[idx], self.active[idx] = traj, b.event_masks, b.active
            for name, val in (("sup", sup), ("counts", counts), ("keep", keep)):
                if val is not None:
                    getattr(self, name)[idx] = val

    def support_weight(self, sup):
        """Weight of a support cell such that support cells make up support_share of the loss:
        share = w f / (w f + 1 - f), with f the share of support cells in `sup`. Rules that already
        act on more than support_share of the cells keep weight 1."""
        f = float(sup.float().mean())
        if self.support_share <= 0 or f <= 0:
            return 1.0
        s = self.support_share
        return min(self.support_max_weight, max(1.0, s * (1 - f) / ((1 - s) * f)))

    def _weights(self, sup, keep):
        w = None
        if sup is not None and self.support_share > 0:
            w = 1.0 + (self.support_weight(sup) - 1.0) * sup.float()
        if keep is not None:
            w = keep.float() if w is None else w * keep.float()
        return w

    @property
    def weights(self):
        """(n, ticks, H, W) loss weights of the whole pool, or None for plain MSE."""
        return self._weights(self.sup, self.keep)

    def refresh(self, k):
        idx = torch.randperm(self.n, generator=self.gen.gen, device=self.gen.device)[:k]
        self._fill(idx)

    def _window_scores(self, k, lead):
        """(n, ticks - k - lead + 1): how many cell-ticks of rule action fall in each window."""
        cs = torch.cat([torch.zeros_like(self.counts[:, :1]), self.counts.cumsum(1)], 1)
        starts = torch.arange(lead, self.ticks - k + 1, device=self.counts.device)
        return cs[:, starts + k] - cs[:, starts]

    def _draw(self, batch, k, lead):
        dev = self.traj.device
        g = self.gen.gen
        idx = torch.randint(self.n, (batch,), generator=g, device=dev)
        t0 = torch.randint(lead, self.ticks - k + 1, (batch,), generator=g, device=dev)
        if self.counts is None or self.support_sampling <= 0:
            return idx, t0
        scores = self._window_scores(k, lead)
        if scores.sum() <= 0:
            return idx, t0
        flat = torch.multinomial(scores.flatten(), batch, replacement=True, generator=g)
        s_idx, s_t0 = flat // scores.shape[1], flat % scores.shape[1] + lead
        pick = torch.rand(batch, generator=g, device=dev) < self.support_sampling
        return torch.where(pick, s_idx, idx), torch.where(pick, s_t0, t0)

    def sample(self, batch, k, lead=0):
        """batch windows of k ticks. lead > 0 also returns the `lead` ticks before each window,
        used to start from the NCA's own rollout instead of the true state.

        Returns start (B, 20, H, W), targets (B, k, 20, H, W), event masks (B, k, 3, H, W),
        loss weights (B, k, 1, H, W) or None, and the lead-in (states, masks) or None.
        """
        if k > self.ticks:
            raise ValueError(f"window of {k} ticks but the pool keeps {self.ticks}")
        lead = min(lead, self.ticks - k)
        dev = self.traj.device
        idx, t0 = self._draw(batch, k, lead)
        tt = t0[:, None] + torch.arange(k + 1, device=dev)[None]          # (B, k + 1)
        win = self.traj[idx[:, None], tt].float()
        masks = self.masks[idx[:, None], tt[:, :-1]]
        rows, cols = idx[:, None], tt[:, :-1]
        sup = None if self.sup is None else self.sup[rows, cols]
        keep = None if self.keep is None else self.keep[rows, cols]
        weights = self._weights(sup, keep)
        if weights is not None:
            weights = weights.unsqueeze(2)
        lead_in = None
        if lead > 0:
            lt = t0[:, None] - lead + torch.arange(lead, device=dev)[None]
            lead_in = (self.traj[idx, lt[:, 0]].float(), self.masks[idx[:, None], lt])
        return win[:, 0], win[:, 1:], masks, weights, lead_in