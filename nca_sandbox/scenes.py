from dataclasses import dataclass

import torch
import torch.nn.functional as Fn

from .config import (
    TREE, W, O, E, GRASS, H, LAVA, ICE, METAL, N_CHANNELS, N_MATERIALS, STONE, S, SOIL, T, GROUND,
    C, V, WX, WY, SimParams,
)
from .simulator import apply_border


@dataclass(frozen=True)
class SceneParams:
    # mappe procedurali
    grass_level: float = 0.35
    forest_level: float = 0.72
    rock_level: float = 0.78
    p_river: float = 0.5
    river_width: float = 0.035
    p_lake: float = 0.5
    lake_level: float = 0.82
    p_fence: float = 0.5
    p_fuel: float = 0.6
    p_lava: float = 0.3
    p_cold_zone: float = 0.5
    p_wind: float = 0.5
    t_noise: float = 0.02
    h_dry: tuple = (0.05, 0.35)
    # micro-scene
    n_shapes: tuple = (3, 8)
    n_patches: tuple = (1, 4)
    p_charge_seed: float = 0.5
    p_cold_ice: float = 0.6


def _u(shape, gen, device, lo=0.0, hi=1.0):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def _ri(shape, lo, hi, gen, device):
    return torch.randint(lo, hi + 1, shape, generator=gen, device=device)


def smooth_noise(b, h, w, cell, gen, device):
    """Rumore a bassa frequenza in [0, 1], normalizzato scena per scena."""
    gh, gw = h // cell + 2, w // cell + 2
    x = torch.rand(b, 1, gh, gw, generator=gen, device=device)
    x = Fn.interpolate(x, size=(h, w), mode="bicubic", align_corners=True)[:, 0]
    lo = x.flatten(1).min(1).values.view(-1, 1, 1)
    hi = x.flatten(1).max(1).values.view(-1, 1, 1)
    return (x - lo) / (hi - lo).clamp(min=1e-6)


def _grid(h, w, device):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return yy.float(), xx.float()


def _disks(b, h, w, n, rmin, rmax, gen, device):
    """n dischi casuali per scena -> (B, n, H, W) bool."""
    yy, xx = _grid(h, w, device)
    cy = _u((b, n, 1, 1), gen, device, 1, h - 2)
    cx = _u((b, n, 1, 1), gen, device, 1, w - 2)
    r = _u((b, n, 1, 1), gen, device, rmin, rmax)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _segments(b, h, w, n, gen, device, axis_aligned=True):
    """n segmenti spessi una cella per scena -> (B, n, H, W) bool."""
    yy, xx = _grid(h, w, device)
    y0 = _u((b, n, 1, 1), gen, device, 1, h - 2)
    x0 = _u((b, n, 1, 1), gen, device, 1, w - 2)
    if axis_aligned:
        horiz = torch.rand((b, n, 1, 1), generator=gen, device=device) < 0.5
        length = _u((b, n, 1, 1), gen, device, h * 0.3, h * 0.9)
        y1 = torch.where(horiz, y0, y0 + length)
        x1 = torch.where(horiz, x0 + length, x0)
    else:
        y1 = _u((b, n, 1, 1), gen, device, 1, h - 2)
        x1 = _u((b, n, 1, 1), gen, device, 1, w - 2)
    # distanza punto-segmento
    dy, dx = y1 - y0, x1 - x0
    t = (((yy - y0) * dy + (xx - x0) * dx) / (dy ** 2 + dx ** 2).clamp(min=1e-6)).clamp(0, 1)
    dist2 = (yy - (y0 + t * dy)) ** 2 + (xx - (x0 + t * dx)) ** 2
    return dist2 <= 0.5 ** 2 + 1e-6


def _pick(masks, enabled):
    """OR dei soli elementi abilitati: masks (B, n, H, W), enabled (B, n) bool."""
    return (masks & enabled[:, :, None, None]).any(1)


def _blur(x, radius=2):
    k = 2 * radius + 1
    return Fn.avg_pool2d(x.unsqueeze(1), k, stride=1, padding=radius, count_include_pad=False)[:, 0]


def assemble(mat_idx, fields, wind, p: SimParams):
    """Indice di materiale + campi -> stato (B, 20, H, W) coerente con la specifica.

    fields: dict canale -> (B, H, W). Gli strati vengono azzerati sulle strutture,
    U viene dal materiale, la lava parte rovente, poi si scrive il bordo.
    """
    b, h, w = mat_idx.shape
    device = mat_idx.device
    s = torch.zeros(b, N_CHANNELS, h, w, device=device)
    s[:, :N_MATERIALS] = Fn.one_hot(mat_idx, N_MATERIALS).permute(0, 3, 1, 2).float()
    for ch, val in fields.items():
        s[:, ch] = val
    soil = torch.zeros(N_MATERIALS, device=device)
    soil[list(SOIL)] = 1.0
    is_soil = soil[mat_idx] > 0.5
    s[:, W] = torch.where(is_soil, s[:, W], torch.zeros_like(s[:, W]))
    s[:, O] = torch.where(is_soil, s[:, O], torch.zeros_like(s[:, O]))
    starting_combustibility = torch.tensor(p.materials.starting_combustibility, device=device)
    s[:, C] = starting_combustibility[mat_idx]
    s[:, T] = torch.where(mat_idx == LAVA, s[:, T].clamp(min=0.95), s[:, T])
    s[:, :WX] = s[:, :WX].clamp(0, 1)
    s[:, WX] = wind[:, 0, None, None]
    s[:, WY] = wind[:, 1, None, None]
    return apply_border(s, p.t_amb)


def sample_wind(b, prob, gen, device):
    on = torch.rand(b, generator=gen, device=device) < prob
    mag = _u((b,), gen, device, 0.3, 1.0)
    ang = _u((b,), gen, device, 0, 2 * torch.pi)
    wind = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], 1)
    return wind * on[:, None]


def procedural_maps(b, h, w, gen, device, sp: SceneParams = None, p: SimParams = None):
    """Prati, boschi, pietra, fiumi e laghi, recinzioni, pozze di carburante, rare colate di lava."""
    sp, p = sp or SceneParams(), p or SimParams()
    coin = lambda q: torch.rand(b, generator=gen, device=device) < q

    veg = smooth_noise(b, h, w, 8, gen, device)
    rock = smooth_noise(b, h, w, 6, gen, device)
    mat = torch.full((b, h, w), GROUND, dtype=torch.long, device=device)
    mat = torch.where(veg > sp.grass_level, torch.full_like(mat, GRASS), mat)
    mat = torch.where(veg > sp.forest_level, torch.full_like(mat, TREE), mat)
    mat = torch.where(rock > sp.rock_level, torch.full_like(mat, STONE), mat)

    # acqua: fiumi come creste di rumore, laghi come picchi; sotto c'è terra
    river = (smooth_noise(b, h, w, 10, gen, device) - 0.5).abs() < sp.river_width * 2
    river &= coin(sp.p_river)[:, None, None]
    lake = (smooth_noise(b, h, w, 8, gen, device) > sp.lake_level) & coin(sp.p_lake)[:, None, None]
    water = river | lake
    mat = torch.where(water, torch.full_like(mat, GROUND), mat)
    aq = water.float()

    # recinzioni di metallo, linee rette
    fences = _pick(_segments(b, h, w, 2, gen, device), coin(sp.p_fence)[:, None] & (torch.rand(b, 2, generator=gen, device=device) < torch.tensor([1.0, 0.5], device=device)))
    mat = torch.where(fences, torch.full_like(mat, METAL), mat)

    # pozze di carburante sul suolo
    fuel_mask = _pick(_disks(b, h, w, 2, 1.0, 3.0, gen, device), coin(sp.p_fuel)[:, None] & (torch.rand(b, 2, generator=gen, device=device) < 0.6))
    fuel = fuel_mask.float() * _u((b, h, w), gen, device, 0.4, 1.0) * (~water).float()

    # colate di lava, rare
    lava = _pick(_disks(b, h, w, 1, 1.0, 3.0, gen, device), coin(sp.p_lava)[:, None])
    mat = torch.where(lava, torch.full_like(mat, LAVA), mat)

    temp = p.t_amb + sp.t_noise * (2 * smooth_noise(b, h, w, 4, gen, device) - 1)
    cold = _pick(_disks(b, h, w, 1, h * 0.2, h * 0.45, gen, device), coin(sp.p_cold_zone)[:, None])
    cold_t = _u((b, 1, 1), gen, device, 0.03, 0.12)
    temp = torch.where(cold, cold_t.expand_as(temp), temp)

    dry = _u((b, 1, 1), gen, device, *sp.h_dry)
    hum = dry + 0.1 * smooth_noise(b, h, w, 6, gen, device)
    hum = torch.maximum(hum, 0.8 * _blur(aq, 2))

    fields = {W: aq, O: fuel, T: temp, H: hum}
    return assemble(mat, fields, sample_wind(b, sp.p_wind, gen, device), p)


def micro_scenes(b, h, w, gen, device, sp: SceneParams = None, p: SimParams = None):
    """Linee, cerchi e quadrati di materiali e strati casuali, con macchie di caldo e di freddo."""
    sp, p = sp or SceneParams(), p or SimParams()
    yy, xx = _grid(h, w, device)
    mat = torch.full((b, h, w), GROUND, dtype=torch.long, device=device)
    aq = torch.zeros(b, h, w, device=device)
    fuel = torch.zeros(b, h, w, device=device)
    temp = torch.full((b, h, w), p.t_amb, device=device)
    hum = _u((b, 1, 1), gen, device, *sp.h_dry).expand(b, h, w).clone()
    vap = torch.zeros(b, h, w, device=device)
    smoke = torch.zeros(b, h, w, device=device)
    charge = torch.zeros(b, h, w, device=device)

    def shape_masks():
        kind = _ri((b, 1, 1), 0, 2, gen, device)
        cy = _u((b, 1, 1), gen, device, 1, h - 2)
        cx = _u((b, 1, 1), gen, device, 1, w - 2)
        size = _u((b, 1, 1), gen, device, 1.5, h * 0.3)
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= size ** 2
        square = ((yy - cy).abs() <= size) & ((xx - cx).abs() <= size)
        line = _segments(b, h, w, 1, gen, device, axis_aligned=False)[:, 0]
        return torch.where(kind == 0, line, torch.where(kind == 1, circle, square))

    n_max = sp.n_shapes[1]
    n_shapes = _ri((b,), *sp.n_shapes, gen, device)
    for i in range(n_max):
        m = shape_masks() & (i < n_shapes)[:, None, None]
        new_mat = _ri((b, 1, 1), 0, N_MATERIALS - 1, gen, device).expand(b, h, w)
        mat = torch.where(m, new_mat, mat)
        # strato: nessuno, acqua o carburante (tenuto solo sul suolo in assemble)
        layer = _ri((b, 1, 1), 0, 2, gen, device)
        level = _u((b, 1, 1), gen, device, 0.2, 1.0)
        aq = torch.where(m & (layer == 1), level.expand(b, h, w), torch.where(m, torch.zeros_like(aq), aq))
        fuel = torch.where(m & (layer == 2), level.expand(b, h, w), torch.where(m, torch.zeros_like(fuel), fuel))

    n_patch = _ri((b,), *sp.n_patches, gen, device)
    for i in range(sp.n_patches[1]):
        m = shape_masks() & (i < n_patch)[:, None, None]
        kind = _ri((b, 1, 1), 0, 5, gen, device)
        hot = _u((b, 1, 1), gen, device, 0.6, 1.0).expand(b, h, w)
        cold = _u((b, 1, 1), gen, device, 0.0, 0.12).expand(b, h, w)
        lvl = _u((b, 1, 1), gen, device, 0.3, 1.0).expand(b, h, w)
        temp = torch.where(m & (kind == 0), hot, temp)
        #kind 5 is cold water, the only way water can freeze
        temp = torch.where(m & ((kind == 1) | (kind == 5)), cold, temp)
        aq = torch.where(m & (kind == 5), (0.5 + 0.5 * lvl).clamp(max=1.0), aq)
        hum = torch.where(m & (kind == 2), lvl, hum)
        vap = torch.where(m & (kind == 3), lvl, vap)
        smoke = torch.where(m & (kind == 4), lvl, smoke)

    #ice that does not melt at once: from below the freezing point to just above the melting one
    ice_t = _u((b, 1, 1), gen, device, 0.02, 0.3).expand(b, h, w)
    cold_ice = (mat == ICE) & (torch.rand(b, generator=gen, device=device) < sp.p_cold_ice)[:, None, None]
    temp = torch.where(cold_ice, ice_t, temp)

    # piccoli semi di carica sui conduttori, per dare a T4 qualcosa da propagare
    seeds = _disks(b, h, w, 1, 0.5, 1.2, gen, device)[:, 0]
    seeds &= (torch.rand(b, generator=gen, device=device) < sp.p_charge_seed)[:, None, None]
    charge = torch.where(seeds, torch.ones_like(charge), charge)

    fields = {W: aq, O: fuel, T: temp, H: hum, V: vap, S: smoke, E: charge}
    return assemble(mat, fields, sample_wind(b, sp.p_wind, gen, device), p)


def sample_scenes(b, h, w, gen, device, sp: SceneParams = None, p: SimParams = None, frac_procedural=0.5):
    """Metà mappe procedurali e metà micro-scene, mescolate."""
    n_proc = int((torch.rand(b, generator=gen, device=device) < frac_procedural).sum().item())
    parts = []
    if n_proc > 0:
        parts.append(procedural_maps(n_proc, h, w, gen, device, sp, p))
    if b - n_proc > 0:
        parts.append(micro_scenes(b - n_proc, h, w, gen, device, sp, p))
    out = torch.cat(parts, 0)
    perm = torch.randperm(b, generator=gen, device=device)
    return out[perm]