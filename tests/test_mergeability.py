import copy
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from nca_models.evaluate import open_split
from nca_models.mergeability import (
    calibration_set, effective_rank, model_gradients, rule_gradient, simulator_stats, sti, sti_paper,
    subspace_overlap, task_vector, top_mask, weight_metrics,
)
from tools.mergeability import main as merge_main, spearman, starts_from
from nca_models.model import NCA, save_nca
from nca_models.train import TrainConfig
from nca_sandbox.dataset import save_split

SIZE = 12
STEPS = 16


def _noisy(model, seed, scale=0.01):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * scale)
    return m


def _save(path, model, rules, init="random"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_nca(path, model, rules=rules, init=init, trainable="all",
             train_cfg=asdict(TrainConfig(rules=rules, size=SIZE)))
    return path


def test_effective_rank():
    m = torch.randn(10, 2) @ torch.randn(2, 8)
    assert effective_rank(m, 0.999) == 2
    assert effective_rank(torch.zeros(4, 4)) == 0


def test_sti_orthogonal_vs_shared():
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    p, _ = torch.linalg.qr(torch.randn(12, 12))
    a = 3 * torch.outer(q[:, 0], p[:, 0])
    b = 2 * torch.outer(q[:, 1], p[:, 1])
    assert sti([a, b]) < 1e-9
    assert sti([a, a]) > 0.5


def test_sti_paper_and_subspace_overlap():
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    p, _ = torch.linalg.qr(torch.randn(12, 12))
    a = 3 * torch.outer(q[:, 0], p[:, 0])
    b = 2 * torch.outer(q[:, 1], p[:, 1])
    assert sti_paper([a, b], 1 / 12) < 1e-9
    # the paper STI grows with the norm, ours does not
    assert abs(sti_paper([a, 2 * a], 1 / 12) / sti_paper([a, a], 1 / 12) - 1.5) < 1e-6
    assert abs(sti([a, 2 * a]) - sti([a, a])) < 1e-6
    assert max(subspace_overlap(a, b, 1)) < 1e-12
    left, right = subspace_overlap(a, 5 * a, 1)
    assert abs(left - 1) < 1e-9 and abs(right - 1) < 1e-9


def test_weight_metrics_signs():
    torch.manual_seed(0)
    ti = {"W1": torch.randn(8, 6), "b1": torch.randn(8), "W2": torch.randn(4, 8), "b2": torch.randn(4)}
    opposite = {k: -v for k, v in ti.items()}
    same = weight_metrics(ti, ti)
    neg = weight_metrics(ti, opposite)
    assert abs(same["cos"] - 1) < 1e-6 and same["sign_conflict"] == 0 and same["top_overlap"] == 1
    assert abs(neg["cos"] + 1) < 1e-6 and neg["sign_conflict"] == 1

def test_starts_from_follows_phase1():
    with tempfile.TemporaryDirectory() as tmp:
        base = NCA()
        bp = _save(Path(tmp) / "runs/base/base_s0.pt", base, "base")
        p1 = _save(Path(tmp) / "runs2/specialists/T1_s0_p1.pt", base, "T1", init=str(bp))
        final = _save(Path(tmp) / "runs2/specialists/T1_s0_best.pt", base, "T1", init=str(p1))
        other = _save(Path(tmp) / "runs2/scratch/T1_s0_best.pt", base, "T1")
        assert starts_from(final, bp) is True
        assert starts_from(other, bp) is False


def test_top_mask_share():
    v = torch.arange(100.0) - 50
    assert int(top_mask(v, 0.2).sum()) == 20


def test_rule_gradient_and_simulator_stats():
    with tempfile.TemporaryDirectory() as tmp:
        save_split(Path(tmp) / "T1.pt", "T1", n=4, steps=STEPS, seed=1000, size=SIZE)
        save_split(Path(tmp) / "jf.pt", "joint_full", n=4, steps=STEPS, seed=1000, size=SIZE)
        base = NCA()
        g = rule_gradient(base, open_split(Path(tmp) / "T1.pt"), "T1", n=4, ticks=STEPS)
        assert set(g) == {"W1", "b1", "W2", "b2"}
        assert torch.cat([v.flatten() for v in g.values()]).norm() > 0
        specs = {"T1": copy.deepcopy(base), "T3": _noisy(base, 1)}
        st = simulator_stats(open_split(Path(tmp) / "jf.pt"), specs, base, n=4, ticks=STEPS)
        assert torch.allclose(st.inter, st.inter.T) and torch.allclose(st.inter.diagonal(), st.support)
        assert (st.cowrite <= st.inter + 1e-9).all()
        assert st.change_on[0] == 0 and st.change_off[0] == 0
        assert st.change_off[2] > 0
        cal = calibration_set(open_split(Path(tmp) / "jf.pt"), n=4, ticks=STEPS, every=4)
        assert cal[0].shape[0] == 4 * (STEPS // 4)
        gw, gx = model_gradients(_noisy(base, 2), cal)
        assert set(gw) == {"W1", "b1", "W2", "b2"} and gx.shape == cal[0].shape and gx.abs().sum() > 0


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        splits = Path(tmp) / "splits"
        for d in ("T1", "T2", "T3", "joint_full"):
            save_split(splits / f"{d}_val_{SIZE}.pt", d, n=4, steps=STEPS, seed=1000, size=SIZE)
        base = NCA()
        bp = _save(Path(tmp) / "runs/base/base_s0.pt", base, "base")
        for i, r in enumerate(("T1", "T2", "T3")):
            _save(Path(tmp) / f"runs2/specialists/{r}_s0.pt", _noisy(base, i), r, init=str(bp))
        pairs, rules = merge_main(["--base", str(bp), "--specialists", f"{tmp}/runs2/specialists", "--rules",
                                   "T1", "T2", "T3", "--splits-dir", str(splits), "--n", "4", "--ticks", str(STEPS),
                                   "--out", f"{tmp}/out", "--device", "cpu"])
        assert [p["pair"] for p in pairs] == ["T1-T2", "T1-T3", "T2-T3"] and len(rules) == 3
        for key in ("grad_cos", "grad_l2", "igrad_dot", "grad_cos_base", "sti_paper_W2", "sub_right_W1"):
            assert key in pairs[0]
        assert Path(tmp, "out/runs2_pairs.csv").exists() and Path(tmp, "out/runs2_rules.csv").exists()
        tv = task_vector(_noisy(base, 0), base)
        assert set(tv) == {"W1", "b1", "W2", "b2"}


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")