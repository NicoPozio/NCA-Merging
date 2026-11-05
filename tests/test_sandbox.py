import tempfile
from pathlib import Path

import torch

from nca_sandbox.composition import composition_suite, run_scene
from nca_sandbox.config import (
    W, E, F, ICE, H, METAL, N_RULES, R, T, T2, T4, T5, GROUND, WX, SimParams,
)
from nca_sandbox.dataset import SandboxGenerator, Split, save_split
from nca_sandbox.events import events_to_masks
from nca_sandbox.scenes import assemble
from nca_sandbox.simulator import Simulator, border_mask

SIZE = 16


def _gen(seed=0):
    return SandboxGenerator(seed=seed)


def _scene(mat, **fields):
    """Scena 1xSIZExSIZE da un materiale uniforme e campi costanti o tensori."""
    idx = torch.full((1, SIZE, SIZE), mat, dtype=torch.long)
    chans = {"aq": W, "t": T, "h": H, "e": E}
    f = {chans[k]: (v if torch.is_tensor(v) else torch.full((1, SIZE, SIZE), float(v))) for k, v in fields.items()}
    f.setdefault(T, torch.full((1, SIZE, SIZE), 0.3))
    return idx, f


def _only(rule):
    a = torch.zeros(1, N_RULES, dtype=torch.bool)
    a[0, rule] = True
    return a


def test_state_invariants():
    b = _gen().sample(6, "joint_full", steps=12, size=SIZE, balance=False)
    traj = b.trajectory
    assert traj[:, :, :18].min() >= 0 and traj[:, :, :18].max() <= 1
    assert torch.allclose(traj[:, :, :8].sum(2), torch.ones(1))
    assert torch.equal(traj[:, :, WX:], traj[:, :1, WX:].expand_as(traj[:, :, WX:]))


def test_border_fixed():
    b = _gen().sample(4, "joint_full", steps=10, size=SIZE, balance=False)
    ring = border_mask(SIZE, SIZE, "cpu")
    edge = b.trajectory[..., ring]
    assert torch.all(edge[:, :, GROUND] == 1)
    assert torch.allclose(edge[:, :, T], torch.tensor(SimParams().t_amb))
    assert not b.support[..., ring].any() and not b.conflicts[..., ring].any()


def test_deterministic():
    b1 = _gen(seed=3).sample(4, "T1", steps=10, size=SIZE, balance=False)
    b2 = _gen(seed=3).sample(4, "T1", steps=10, size=SIZE, balance=False)
    assert torch.equal(b1.trajectory, b2.trajectory)
    assert b1.events == b2.events


def test_base_has_no_reactions():
    b = _gen(seed=1).sample(8, "base", steps=16, size=SIZE, balance=False)
    mats = b.trajectory[:, :, :8].argmax(2)
    assert torch.equal(mats, mats[:, :1].expand_as(mats))
    # senza regole il fuoco può solo calare, a parte le accensioni iniettate
    fire = b.trajectory[:, :, F]
    rise = (fire[:, 1:] > fire[:, :-1] + 1e-6) & ~b.event_masks[:, :, 2]
    assert not rise.any()
    assert not b.support.any()


def test_oracle_matches_outside_conflicts():
    # G deve coincidere con O = y0 + sum_i (y_i - y0) ovunque il registro dei conflitti è vuoto
    b = _gen(seed=2).sample(6, "joint_full", steps=20, size=SIZE, balance=False)
    sim = Simulator()
    for t in range(b.event_masks.shape[1]):
        s, m = b.trajectory[:, t], b.event_masks[:, t]
        full = torch.ones(len(b), N_RULES, dtype=torch.bool)
        g, info = sim.step(s, full, m)
        y0, _ = sim.step(s, torch.zeros_like(full), m)
        o = y0.clone()
        for i in range(N_RULES):
            a = torch.zeros_like(full)
            a[:, i] = True
            o += sim.step(s, a, m)[0] - y0
        free = ~info.conflicts.any(1)
        assert (o - g).abs().amax(1)[free].max() < 1e-5


def test_inactive_rules_have_no_support():
    b = _gen(seed=4).sample(6, "T3", steps=12, size=SIZE, balance=False)
    others = torch.ones(N_RULES, dtype=torch.bool)
    others[2] = False
    assert not b.support[:, :, others].any()
    assert not b.conflicts.any()


def test_ice_hysteresis():
    sim = Simulator()
    idx, f = _scene(GROUND, aq=1.0, t=0.2, h=0.8)
    water = assemble(idx, f, torch.zeros(1, 2), sim.p)
    out, _ = sim.step(water, _only(T5))
    assert out[0, ICE, 2:-2, 2:-2].sum() == 0
    idx, f = _scene(GROUND, aq=1.0, t=0.1, h=0.8)
    out, _ = sim.step(assemble(idx, f, torch.zeros(1, 2), sim.p), _only(T5))
    assert torch.all(out[0, ICE, 2:-2, 2:-2] == 1) and torch.all(out[0, W, 2:-2, 2:-2] == 0)
    idx, f = _scene(ICE, t=0.2)
    ice = assemble(idx, f, torch.zeros(1, 2), sim.p)
    assert sim.step(ice, _only(T5))[0][0, ICE, 2:-2, 2:-2].sum() == (SIZE - 4) ** 2
    idx, f = _scene(ICE, t=0.3)
    out, _ = sim.step(assemble(idx, f, torch.zeros(1, 2), sim.p), _only(T5))
    assert torch.all(out[0, GROUND, 2:-2, 2:-2] == 1) and torch.all(out[0, W, 2:-2, 2:-2] == 1)


def test_charge_wave_has_refractory():
    sim = Simulator()
    idx, f = _scene(GROUND)
    idx[0, SIZE // 2, 1:-1] = METAL
    e = torch.zeros(1, SIZE, SIZE)
    e[0, SIZE // 2, 2] = 1.0
    f[E] = e
    x = assemble(idx, f, torch.zeros(1, 2), sim.p)
    fires = torch.zeros(SIZE)
    for _ in range(SIZE + 4):
        prev = x[0, E, SIZE // 2] >= 0.5
        x, _ = sim.step(x, _only(T4))
        fires += ((x[0, E, SIZE // 2] >= 0.5) & ~prev).float()
    # l'onda va da sinistra a destra e ogni cella si scarica una volta sola
    assert fires[3:-1].eq(1).all()
    assert x[0, R].max() <= 1.0


def test_wind_moves_heat_downwind():
    sim = Simulator()
    t = torch.full((1, SIZE, SIZE), 0.3)
    t[0, SIZE // 2, SIZE // 2] = 1.0
    idx, f = _scene(GROUND, t=t)
    x = assemble(idx, f, torch.tensor([[1.0, 0.0]]), sim.p)
    for _ in range(3):
        x, _ = sim.step(x, _only(T2))
    row = x[0, T, SIZE // 2]
    c = SIZE // 2
    assert row[c + 2] > row[c - 2] + 0.05


def test_event_masks():
    ev = [(2, "pioggia", (8, 8), {"radius": 2, "duration": 3}), (1, "fulmine", (3, 4), {}), (0, "accensione", (5, 6), {})]
    m = events_to_masks(ev, 6, SIZE, SIZE)
    assert m[2:5, 0, 8, 8].all() and not m[5, 0].any() and not m[:2, 0].any()
    assert m[1, 1, 3, 4] and m[1, 1].sum() == 1
    assert m[0, 2, 5, 6] and m[:, 2].sum() == 1


def test_balancing_raises_trigger_rate():
    g = _gen(seed=5)
    raw = g.activated(g.sample(24, "T4", steps=16, size=SIZE, balance=False)).float().mean()
    bal = g.activated(g.sample(24, "T4", steps=16, size=SIZE, balance=True)).float().mean()
    assert bal >= raw


def test_joint_single_one_rule_per_scene():
    b = _gen(seed=6).sample(10, "joint_single", steps=4, size=SIZE, balance=False)
    assert torch.all(b.active.sum(1) == 1)


def test_split_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t1_val.pt"
        save_split(path, "T1", n=6, steps=12, seed=0, size=SIZE, chunk=3)
        split = Split(path)
        assert len(split) == 6 and split.verify()
        b = split.batch(range(3))
        assert b.trajectory.shape == (3, 13, 20, SIZE, SIZE)


def test_composition_scenes_run():
    for sc in composition_suite():
        traj, conf = run_scene(sc)
        assert traj.shape[0] == sc.steps + 1
        assert isinstance(sc.indicators(traj), dict)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} test passati")