from dataclasses import dataclass, field
from typing import Callable

import torch

from .config import (
    TREE, W, O, CINDER, E, GRASS, F, ICE, H, LAVA, METAL, N_RULES, STONE, RULE_IDS, T,
    GROUND, V, SimParams,
)
from .events import batch_event_masks
from .scenes import assemble
from .simulator import Simulator


@dataclass
class CompositionScene:
    name: str
    kind: str                    # indipendenti, catena, catena lunga, blocco, conflitto
    rules: tuple                 # es. ("T4", "T1")
    state: torch.Tensor          # (20, H, W)
    events: list
    steps: int
    expect: str                  # cosa deve emergere, in parole
    indicators: Callable = field(repr=False, default=None)   # traj (k+1, 20, H, W) -> dict

    def active(self, device="cpu"):
        a = torch.zeros(N_RULES, dtype=torch.bool, device=device)
        for r in self.rules:
            a[RULE_IDS[r]] = True
        return a


class _Canvas:
    """Una griglia da dipingere: materiale e campi, poi assemble()."""

    def __init__(self, size, t=0.3, h=0.15, wind=(0.0, 0.0), p=None):
        self.p = p or SimParams()
        self.size = size
        self.mat = torch.full((1, size, size), GROUND, dtype=torch.long)
        self.f = {ch: torch.zeros(1, size, size) for ch in (W, O, T, H, E, V)}
        self.f[T][:] = t
        self.f[H][:] = h
        self.wind = torch.tensor([wind], dtype=torch.float32)

    def rect(self, r0, r1, c0, c1, mat=None, **fields):
        sl = (0, slice(r0, r1), slice(c0, c1))
        if mat is not None:
            self.mat[sl] = mat
        for name, val in fields.items():
            self.f[{"w": W, "o": O, "t": T, "h": H, "e": E, "v": V}[name]][sl] = val
        return self

    def disk(self, cy, cx, rad, mat=None, **fields):
        yy, xx = torch.meshgrid(torch.arange(self.size), torch.arange(self.size), indexing="ij")
        m = ((yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2).unsqueeze(0)
        if mat is not None:
            self.mat[m] = mat
        for name, val in fields.items():
            self.f[{"w": W, "o": O, "t": T, "h": H, "e": E, "v": V}[name]][m] = val
        return self

    def build(self):
        return assemble(self.mat, self.f, self.wind, self.p)[0]


def _mat(traj, m):
    return traj[:, m] > 0.5


def _burnt(traj, region=None):
    """Celle diventate cenere o in fiamme almeno una volta (dentro region, se data)."""
    hit = (_mat(traj, CINDER) | (traj[:, F] > 0.5)).any(0)
    return int((hit & region).sum()) if region is not None else int(hit.sum())


def _region(size, r0, r1, c0, c1):
    m = torch.zeros(size, size, dtype=torch.bool)
    m[r0:r1, c0:c1] = True
    return m


def independent_fence_forest(size=32, p=None):
    cv = _Canvas(size, p=p).rect(6, 7, 3, 14, METAL).rect(16, 28, 17, 29, TREE)
    events = [(0, "accensione", (22, 23), {}), (2, "fulmine", (6, 4), {})]
    forest = _region(size, 16, 28, 17, 29)
    fence = _region(size, 6, 7, 3, 14)

    def ind(traj):
        return {"bosco_bruciato": _burnt(traj, forest),
                "recinzione_carica": int(((traj[:, E] > 0.5).any(0) & fence).sum()),
                "fuoco_fuori_dal_bosco": _burnt(traj, ~forest)}
    return CompositionScene("recinzione_e_bosco_lontani", "indipendenti", ("T4", "T1"), cv.build(), events, 64,
                            "ognuno fa il suo, nessuna interferenza", ind)


def chain_wind_meadow(size=32, p=None):
    cv = _Canvas(size, h=0.15, wind=(1.0, 0.0), p=p).rect(1, size - 1, 1, size - 1, GRASS)
    c0 = size // 2
    events = [(0, "accensione", (c0, c0), {})]
    east, west = _region(size, 1, size - 1, c0 + 1, size - 1), _region(size, 1, size - 1, 1, c0)

    def ind(traj):
        mid = traj[:31]
        return {"bruciato_sottovento": _burnt(mid, east), "bruciato_sopravento": _burnt(mid, west)}


def chain_bolt_fence_grass(size=32, p=None):
    c0 = size // 2
    cv = _Canvas(size, h=0.1, p=p).rect(1, size - 1, 1, size - 1, GRASS).rect(4, size - 4, c0, c0 + 1, METAL)
    events = [(2, "fulmine", (4, c0), {})]
    grass = cv.mat[0] == GRASS

    def ind(traj):
        return {"erba_bruciata": _burnt(traj, grass),
                "t_max_recinzione": float(traj[:, T, 4:size - 4, c0].max())}
    return CompositionScene("fulmine_su_recinzione_e_erba_secca", "catena", ("T4", "T1"), cv.build(), events, 64,
                            "la scarica scalda e l'erba si accende", ind)


def chain_rain_on_fire(size=32, p=None):
    cv = _Canvas(size, h=0.1, p=p).rect(1, size - 1, 1, size - 1, GRASS).rect(8, 24, 8, 24, TREE)
    c0 = size // 2
    events = [(0, "accensione", (c0, c0), {}), (6, "pioggia", (c0, c0), {"radius": size, "duration": 40})]

    def ind(traj):
        burning = (traj[:, F] > 0.5).flatten(1).sum(1)
        return {"celle_in_fiamme_t6": int(burning[6]), "celle_in_fiamme_fine": int(burning[-1]),
                "tick_spegnimento": int(torch.nonzero(burning[6:] == 0).flatten()[0] + 6) if (burning[6:] == 0).any() else -1}
    return CompositionScene("pioggia_su_incendio", "catena", ("T3", "T1"), cv.build(), events, 64,
                            "l'umidità sale e il fuoco si spegne", ind)


def chain_fuel_stone_meadow(size=32, p=None):
    cv = _Canvas(size, h=0.15, p=p).rect(1, size - 1, 1, size // 2, STONE).rect(1, size - 1, size // 2, size - 1, GRASS)
    cv.disk(size // 2, size // 2 - 4, 4, o=1.0)
    events = [(0, "accensione", (size // 2, size // 2 - 4), {})]
    grass = cv.mat[0] == GRASS

    def ind(traj):
        return {"erba_bruciata": _burnt(traj, grass), "carburante_rimasto": float(traj[-1, O].sum())}
    return CompositionScene("carburante_su_pietra_accanto_al_prato", "catena", ("T6", "T1"), cv.build(), events, 64,
                            "il carburante brucia caldo e accende il prato", ind)


def chain_lava_forest(size=32, p=None):
    cv = _Canvas(size, h=0.15, p=p).rect(4, size - 4, 6, 11, LAVA).rect(4, size - 4, 11, 20, TREE)
    forest = cv.mat[0] == TREE

    def ind(traj):
        return {"bosco_bruciato": _burnt(traj, forest), "lava_rimasta": int(_mat(traj[-1:], LAVA).sum())}
    return CompositionScene("colata_di_lava_accanto_al_bosco", "catena", ("T7", "T1"), cv.build(), [], 64,
                            "il bosco si incendia", ind)


def long_lava_lake_wind(size=48, p=None):
    cv = _Canvas(size, h=0.15, wind=(1.0, 0.0), p=p)
    cv.rect(10, size - 10, 3, 9, LAVA)
    cv.rect(10, size - 10, 9, 20, GROUND, w=1.0, h=0.8)
    cv.rect(1, size - 1, 30, size - 1, GRASS)
    meadow = cv.mat[0] == GRASS
    #the edge of the meadow facing the lake, where the vapor plume arrives
    edge = _region(size, 10, size - 10, 30, 36)

    def ind(traj):
        h_edge = traj[:, H][:, edge].mean(1)
        return {"vapore_max": float(traj[:, V].max()),
                "vapore_max_sul_prato": float(traj[:, V][:, meadow].max()),
                "h_bordo_prato_inizio": float(h_edge[0]),
                "h_bordo_prato_max": float(h_edge.max()),
                "lava_diventata_pietra": int((_mat(traj[:1], LAVA)[0] & _mat(traj[-1:], STONE)[0]).sum())}
    return CompositionScene("lava_incontra_lago_con_vento", "catena lunga", ("T7", "T3", "T2", "T8"), cv.build(), [], 96,
                            "pietra e vapore; il vapore va sottovento e condensa, bagnando un prato lontano", ind)

def _storm_scene(ground, size=32, p=None):
    c0 = size // 2
    cv = _Canvas(size, h=0.2, p=p).rect(1, size - 1, 1, size - 1, ground).rect(4, size - 4, c0, c0 + 1, METAL)
    events = [(0, "pioggia", (c0, c0), {"radius": size, "duration": 60}), (45, "fulmine", (6, c0), {})]
    ground_mask = cv.mat[0] == ground

    def ind(traj):
        return {"pozze_aq_ge_0.2": int(((traj[44, W] >= 0.2) & ground_mask).sum()),
                "suolo_elettrizzato": int(((traj[45:, E] > 0.5).any(0) & ground_mask).sum()),
                "bruciato": _burnt(traj, ground_mask)}
    return cv.build(), events, ind


def long_storm_stone(size=32, p=None):
    state, events, ind = _storm_scene(STONE, size, p)
    return CompositionScene("temporale_lungo_su_pietra", "catena lunga", ("T3", "T4"), state, events, 64,
                            "si forma una pozza; il fulmine sulla recinzione la elettrizza", ind)


def block_storm_meadow(size=32, p=None):
    state, events, ind = _storm_scene(GRASS, size, p)
    return CompositionScene("temporale_lungo_su_prato", "blocco", ("T3", "T4", "T1"), state, events, 64,
                            "l'erba assorbe: niente pozza, niente scossa, niente incendio", ind)


def block_frozen_river(size=32, p=None):
    # tra 0,15 e 0,25 acqua e ghiaccio sono entrambi stabili (isteresi)
        # the ice starts cold, otherwise the ambient temperature (0.3 > 0.25) melts it before the shock
    cv = _Canvas(size, t=0.2, p=p).rect(12, 20, 1, size - 1, GROUND, w=1.0, h=0.8).rect(12, 20, 14, 18, ICE, t=0.05)
    events = [(2, "fulmine", (16, 3), {})]
    right = _region(size, 12, 20, 18, size - 1)

    def ind(traj):
        return {"acqua_carica_oltre_il_ghiaccio": int(((traj[:, E] > 0.5).any(0) & right).sum()),
                "acqua_carica_prima": int(((traj[:, E] > 0.5).any(0) & _region(size, 12, 20, 1, 14)).sum()),
                "ghiaccio_rimasto": int(_mat(traj[-1:], ICE).sum())}
    return CompositionScene("scossa_su_fiume_ghiacciato", "blocco", ("T5", "T4"), cv.build(), events, 32,
                            "la carica si ferma al ghiaccio", ind)


def conflict_fire_ice(size=32, p=None):
    cv = _Canvas(size, h=0.1, p=p).rect(4, size - 4, 4, 16, TREE).rect(4, size - 4, 16, 22, ICE, t=0.05)
    events = [(0, "accensione", (size // 2, 10), {})]
    ice = cv.mat[0] == ICE

    def ind(traj):
        return {"ghiaccio_fuso": int((ice & ~_mat(traj[-1:], ICE)[0]).sum()),
                "acqua_prodotta": float(traj[:, W].sum((1, 2)).max()),
                "bosco_bruciato": _burnt(traj, cv.mat[0] == TREE)}
    # nota: "l'acqua bagna" è T3, che qui non è attiva; teniamo le regole della specifica
    return CompositionScene("fuoco_accanto_al_ghiaccio", "conflitto", ("T1", "T5"), cv.build(), events, 64,
                            "il ghiaccio fonde, l'acqua bagna e frena il fuoco", ind)


def conflict_fuel_rain(size=32, p=None):
    c0 = size // 2
    cv = _Canvas(size, h=0.1, p=p).disk(c0, c0, 6, GROUND, o=1.0)
    events = [(0, "accensione", (c0, c0), {}), (3, "pioggia", (c0, c0), {"radius": size, "duration": 40})]
    pool = cv.f[O][0] > 0

    def ind(traj):
        fire = (traj[:, F] > 0.5) & pool
        h_at_out = traj[:, H][:, pool].mean(1)
        burning = fire.flatten(1).any(1)
        last = int(torch.nonzero(burning).flatten()[-1]) if burning.any() else -1
        return {"ultimo_tick_in_fiamme": last, "h_media_allora": float(h_at_out[max(last, 0)])}
    return CompositionScene("carburante_sotto_la_pioggia", "conflitto", ("T6", "T3"), cv.build(), events, 64,
                            "il carburante brucia lo stesso fino a H = 0,7", ind)


SCENES = (
    independent_fence_forest, chain_wind_meadow, chain_bolt_fence_grass, chain_rain_on_fire,
    chain_fuel_stone_meadow, chain_lava_forest, long_lava_lake_wind, long_storm_stone,
    block_storm_meadow, block_frozen_river, conflict_fire_ice, conflict_fuel_rain,
)


def composition_suite(p: SimParams = None):
    return [make(p=p) for make in SCENES]


@torch.no_grad()
def run_scene(scene: CompositionScene, sim: Simulator = None, device="cpu"):
    """Traiettoria della verità G per una scena: (k + 1, 20, H, W)."""
    sim = sim or Simulator(device=device)
    state = scene.state.to(device).unsqueeze(0)
    h, w = state.shape[-2:]
    masks = batch_event_masks([scene.events], scene.steps, h, w, device)
    traj, _, conf, _ = sim.rollout(state, scene.active(device).unsqueeze(0), scene.steps, masks)
    return traj[0], conf[0]


if __name__ == "__main__":
    sim = Simulator()
    for sc in composition_suite():
        traj, conf = run_scene(sc, sim)
        print(f"[{sc.kind}] {sc.name} {sc.rules}: {sc.expect}")
        print(f"    {sc.indicators(traj)}  celle con conflitto: {int(conf.any(1).sum())}")