"""The NCA: fixed perception, a per-cell MLP, purely additive update.

The trainable parameters are exactly W1 (hidden, 80), b1, W2 (20, hidden), b2, so task vectors
and merges can work directly on the state_dict.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from nca_sandbox.config import CLIPPED, E, F, H, N_CHANNELS, T, WX, MaterialProps, SimParams
from nca_sandbox.simulator import apply_border

N_FILTERS = 4


def perception_kernels():
    """Identity, Sobel x, Sobel y and the 8-neighbour Laplacian (mean of neighbours minus the cell,
    the same operator the simulator uses for diffusion)."""
    ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    sobel_y = sobel_x.T
    lap = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]) / 8
    return torch.stack([ident, sobel_x, sobel_y, lap])


def inject_events(state, masks, p: SimParams):
    """Same result as the simulator's inject_events, written without in-place writes so that
    gradients can flow through it."""
    if masks is None:
        return state
    rain, bolt, ignite = masks[:, 0], masks[:, 1], masks[:, 2]
    one = torch.ones_like(state[:, 0])
    h = torch.where(rain, (state[:, H] + p.rain_h).clamp(max=1.0), state[:, H])
    e = torch.where(bolt, one, state[:, E])
    t = torch.where(bolt, one, state[:, T])
    f = torch.where(ignite, one, state[:, F])
    t = torch.where(ignite, t.clamp(min=p.ignite_t), t)
    new = {H: h, E: e, T: t, F: f}
    s = torch.stack([new.get(c, state[:, c]) for c in range(state.shape[1])], 1)
    return apply_border(s, p.t_amb)


@dataclass
class NCAConfig:
    channels: int = N_CHANNELS
    hidden: int = 128          # 0 means no hidden layer: a single linear map 80 -> 20
    act: str = "relu"          # relu, abs or none

    def to_dict(self):
        return asdict(self)


class NCA(nn.Module):
    def __init__(self, cfg: NCAConfig = None, sim_params: SimParams = None):
        super().__init__()
        self.cfg = cfg or NCAConfig()
        self.p = sim_params or SimParams()
        c, h = self.cfg.channels, self.cfg.hidden
        n_in = c * N_FILTERS
        k = perception_kernels()
        # depthwise weights (C * 4, 1, 3, 3): output channel c * 4 + f is filter f applied to channel c
        self.register_buffer("kernels", k.repeat(c, 1, 1).unsqueeze(1), persistent=False)

        if h > 0:
            self.W1 = nn.Parameter(torch.empty(h, n_in))
            self.b1 = nn.Parameter(torch.zeros(h))
            nn.init.kaiming_uniform_(self.W1, a=5 ** 0.5)
            self.W2 = nn.Parameter(torch.zeros(c, h))
        else:
            self.W2 = nn.Parameter(torch.zeros(c, n_in))
        # last layer starts at zero, so a fresh NCA does nothing and grows its update from there
        self.b2 = nn.Parameter(torch.zeros(c))

    def perceive(self, x):
        p = Fn.pad(x, (1, 1, 1, 1), mode="replicate")
        return Fn.conv2d(p, self.kernels, groups=self.cfg.channels)

    def _act(self, z):
        if self.cfg.act == "relu":
            return torch.relu(z)
        if self.cfg.act == "abs":
            return z.abs()
        return z

    def delta(self, x):
        z = self.perceive(x).permute(0, 2, 3, 1)                       # (B, H, W, 80)
        if self.cfg.hidden > 0:
            z = self._act(Fn.linear(z, self.W1, self.b1))
        out = Fn.linear(z, self.W2, self.b2)
        return out.permute(0, 3, 1, 2)

    def forward(self, x, event_masks=None):
        """One tick: events injected exactly as in the simulator, then state + delta, clipping,
        wind rewritten unchanged and the fixed border."""
        s = inject_events(x, event_masks, self.p)
        nxt = s + self.delta(s)
        nxt = torch.cat([nxt[:, :WX].clamp(0.0, 1.0), s[:, WX:]], 1)
        return apply_border(nxt, self.p.t_amb)

    def rollout(self, x, steps, event_masks=None):
        """(B, 20, H, W) -> (B, steps + 1, 20, H, W); event_masks (B, steps, 3, H, W) or None."""
        traj = [x]
        for t in range(steps):
            x = self(x, None if event_masks is None else event_masks[:, t])
            traj.append(x)
        return torch.stack(traj, 1)

    def set_trainable(self, which="all"):
        """'all' trains every parameter, 'last' only W2 and b2 (the case where merge = oracle)."""
        for name, prm in self.named_parameters():
            prm.requires_grad_(which == "all" or name in ("W2", "b2"))


def save_nca(path, model, **meta):
    torch.save({"state_dict": model.state_dict(), "model_cfg": model.cfg.to_dict(),
                "sim_params": model.p.to_dict(), **meta}, path)


def sim_params_from_dict(d):
    d = dict(d)
    return SimParams(**{**d, "materials": MaterialProps(**d["materials"])})


def load_nca(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NCA(NCAConfig(**ckpt["model_cfg"]), sim_params_from_dict(ckpt["sim_params"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


assert set(CLIPPED) == set(range(WX)), "clipping in forward assumes every channel before wind is clipped"