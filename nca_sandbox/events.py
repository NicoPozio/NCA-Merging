from dataclasses import dataclass

import torch

from .config import TREE, O, GRASS, METAL, C

RAIN, BOLT, IGNITE = 0, 1, 2
EVENT_TYPES = ("pioggia", "fulmine", "accensione")


@dataclass(frozen=True)
class EventParams:
    p_storm: float = 0.35
    storm_duration: tuple = (5, 25)
    storm_radius: tuple = (3, 10)
    p_bolt_per_tick: float = 0.3    # durante un temporale
    metal_bolt_weight: float = 10.0
    p_ignite: float = 0.5
    n_ignite: tuple = (1, 3)
    ignite_window: float = 0.5        # le accensioni cadono nella prima metà della finestra


def _randint(lo, hi, gen, device):
    return int(torch.randint(lo, hi + 1, (1,), generator=gen, device=device).item())


def _rand(gen, device):
    return float(torch.rand(1, generator=gen, device=device).item())


def _interior_cells(h, w, device):
    rr, cc = torch.meshgrid(torch.arange(1, h - 1, device=device), torch.arange(1, w - 1, device=device), indexing="ij")
    return rr.flatten(), cc.flatten()


def sample_events(state, steps, gen, ep: EventParams = None):
    """Campiona la lista di eventi di una scena. state: (20, H, W) stato iniziale."""
    ep = ep or EventParams()
    device = state.device
    h, w = state.shape[-2:]
    rr, cc = _interior_cells(h, w, device)
    events = []

    if _rand(gen, device) < ep.p_storm:
        start = _randint(0, steps - 1, gen, device)
        dur = _randint(*ep.storm_duration, gen, device)
        radius = _randint(*ep.storm_radius, gen, device)
        k = int(torch.randint(len(rr), (1,), generator=gen, device=device).item())
        center = (int(rr[k]), int(cc[k]))
        events.append((start, "pioggia", center, {"radius": radius, "duration": dur}))

        # fulmini durante il temporale, ovunque nella mappa; il metallo è colpito 10 volte più spesso
        metal = state[METAL, rr, cc] > 0.5
        weights = torch.where(metal, ep.metal_bolt_weight, 1.0)
        for t in range(start, min(start + dur, steps)):
            if _rand(gen, device) < ep.p_bolt_per_tick:
                j = int(torch.multinomial(weights, 1, generator=gen).item())
                events.append((t, "fulmine", (int(rr[j]), int(cc[j])), {}))

    if _rand(gen, device) < ep.p_ignite:
        flam = ((state[GRASS, rr, cc] + state[TREE, rr, cc]) > 0.5) & (state[C, rr, cc] > 0)
        fuel = state[O, rr, cc] >= 0.1
        cand = torch.nonzero(flam | fuel).flatten()
        if len(cand) > 0:
            last = max(0, int(steps * ep.ignite_window) - 1)
            for _ in range(_randint(*ep.n_ignite, gen, device)):
                j = cand[int(torch.randint(len(cand), (1,), generator=gen, device=device).item())]
                t = _randint(0, last, gen, device)
                events.append((t, "accensione", (int(rr[j]), int(cc[j])), {}))

    events.sort(key=lambda e: (e[0], e[1]))
    return events


def events_to_masks(events, steps, h, w, device="cpu"):
    """Lista di eventi -> maschere (steps, 3, H, W) bool: [pioggia, fulmine, accensione]."""
    masks = torch.zeros(steps, 3, h, w, dtype=torch.bool, device=device)
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    for tick, kind, (r, c), params in events:
        if kind == "pioggia":
            disk = (yy - r) ** 2 + (xx - c) ** 2 <= params["radius"] ** 2
            for t in range(tick, min(tick + params["duration"], steps)):
                masks[t, RAIN] |= disk
        elif tick < steps:
            masks[tick, BOLT if kind == "fulmine" else IGNITE, r, c] = True
    return masks


def batch_event_masks(event_lists, steps, h, w, device="cpu"):
    return torch.stack([events_to_masks(ev, steps, h, w, device) for ev in event_lists])