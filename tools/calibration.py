"""Calibration report for the simulator: phenomenon timings, composition scenes, trigger rates.

Usage:
    python -m tools.calibrate                 # current SimParams defaults
    python -m tools.calibrate --set heat_rate=0.8 t1_heat=0.4
"""

import argparse
from dataclasses import replace

import torch

from nca_sandbox.composition import composition_suite, run_scene
from nca_sandbox.config import (
    CINDER, F, GRASS, H, N_RULES, RULES, STONE, T, TREE, W, MaterialProps, SimParams,
)
from nca_sandbox.dataset import GeneratorConfig, SandboxGenerator
from nca_sandbox.events import batch_event_masks
from nca_sandbox.scenes import assemble
from nca_sandbox.simulator import Simulator


def parse_overrides(items):
    """'name=value' pairs; material properties are written as 'mat.k=0.5,0.4,...'."""
    sim, mat = {}, {}
    for item in items or []:
        key, val = item.split("=")
        if key.startswith("mat."):
            mat[key[4:]] = tuple(float(v) for v in val.split(","))
        else:
            sim[key] = float(val)
    return sim, mat


def make_params(sim_over, mat_over):
    p = SimParams()
    if mat_over:
        p = replace(p, materials=replace(MaterialProps(), **mat_over))
    return replace(p, **sim_over)


def _active(*rules):
    a = torch.zeros(1, N_RULES, dtype=torch.bool)
    for r in rules:
        a[0, r] = True
    return a


def _uniform(material, size, sim, h):
    mat = torch.full((1, size, size), material, dtype=torch.long)
    fields = {T: torch.full((1, size, size), sim.p.t_amb), H: torch.full((1, size, size), h)}
    return mat, fields


def _run(sim, mat, fields, active, steps, events, size=32, wind=(0.0, 0.0)):
    state = assemble(mat, fields, torch.tensor([wind]), sim.p)
    masks = batch_event_masks([events], steps, size, size)
    traj, _, _, _ = sim.rollout(state, active, steps, masks, keep_info=False)
    return traj[0]


def burn_duration(sim, material):
    """Ticks a single ignited cell of `material` stays on fire inside a patch of the same material."""
    mat, fields = _uniform(material, 32, sim, 0.15)
    traj = _run(sim, mat, fields, _active(0), 80, [(0, "accensione", (16, 16), {})])
    return int((traj[:, F, 16, 16] > 0.5).sum())


def front_time(sim, material, wind=(0.0, 0.0)):
    """Ticks for a fire lit at the left edge to reach the right edge of a 32x32 patch (-1 = never)."""
    size = 32
    mat, fields = _uniform(material, size, sim, 0.15)
    events = [(0, "accensione", (r, 2), {}) for r in range(2, size - 2, 3)]
    rules = (0, 1) if wind != (0.0, 0.0) else (0,)
    traj = _run(sim, mat, fields, _active(*rules), 150, events, wind=wind)
    hit = (traj[:, F] > 0.5) | (traj[:, CINDER] > 0.5)
    right = hit[:, 2:size - 2, size - 3].any(1)
    burnt = int(hit.any(0)[1:-1, 1:-1].sum())
    return (int(torch.nonzero(right).flatten()[0]) if right.any() else -1), burnt


def puddle_time(sim, material):
    """Ticks of steady rain before a puddle (W >= 0.2) appears in the middle of a patch (-1 = never)."""
    mat, fields = _uniform(material, 32, sim, 0.2)
    events = [(0, "pioggia", (16, 16), {"radius": 32, "duration": 200})]
    traj = _run(sim, mat, fields, _active(2), 120, events)
    hit = traj[:, W, 16, 16] >= 0.2
    return int(torch.nonzero(hit).flatten()[0]) if hit.any() else -1


def trigger_rates(p, n=128, steps=64, seed=0):
    """Fraction of unbalanced random scenes in which each rule fires at least once."""
    gen = SandboxGenerator(GeneratorConfig(sim=p), seed=seed)
    b = gen.sample(n, "joint_full", steps=steps, keep_info=False, balance=False)
    return (b.triggers > 0).float().mean(0)


def report(p):
    sim = Simulator(p)
    print("timings (target: tree ~10, front 30-50, puddle 20-30 on stone, slower on grass)")
    print(f"  burn duration   tree {burn_duration(sim, TREE):3d}   grass {burn_duration(sim, GRASS):3d}")
    for name, m in (("tree", TREE), ("grass", GRASS)):
        t, burnt = front_time(sim, m)
        tw, burnt_w = front_time(sim, m, wind=(1.0, 0.0))
        print(f"  front {name:5s}     no wind {t:4d} ({burnt} cells)   wind east {tw:4d} ({burnt_w} cells)")
    print(f"  puddle          stone {puddle_time(sim, STONE):4d}   grass {puddle_time(sim, GRASS):4d}")

    print("composition scenes")
    for sc in composition_suite(p):
        traj, conf = run_scene(sc, sim)
        print(f"  [{sc.kind}] {sc.name} {sc.rules}: {sc.expect}")
        print(f"      {sc.indicators(traj)}  conflict cells {int(conf.any(1).sum())}")

    print("trigger rate per rule (unbalanced scenes, 64 ticks)")
    rates = trigger_rates(p)
    print("  " + "  ".join(f"{RULES[i]} {rates[i]:.2f}" for i in range(N_RULES)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    report(make_params(*parse_overrides(args.set)))


if __name__ == "__main__":
    main()