import tempfile
from pathlib import Path

import torch

from nca_models.data import WindowPool
from nca_models.model import NCA, NCAConfig, inject_events, load_nca, save_nca
from nca_models.train import TrainConfig, train, window_length
from nca_models.zoo import main as zoo_main
from nca_sandbox.config import F, GROUND, T, TREE, WX, SimParams
from nca_sandbox.dataset import GeneratorConfig, SandboxGenerator
from nca_sandbox.events import events_to_masks
from nca_sandbox.scenes import assemble
from nca_sandbox.simulator import Simulator, border_mask
from nca_sandbox.simulator import inject_events as sim_inject

SIZE = 12


def _tiny(**kw):
    base = dict(iters=6, batch=2, k_min=2, k_max=4, pool_scenes=4, pool_ticks=8, pool_refresh=2,
                refresh_every=3, val_scenes=2, val_ticks=4, eval_every=3, size=SIZE, device="cpu")
    return TrainConfig(**{**base, **kw})


def _state(mat=TREE):
    idx = torch.full((1, SIZE, SIZE), mat, dtype=torch.long)
    return assemble(idx, {T: torch.full((1, SIZE, SIZE), 0.3)}, torch.tensor([[0.5, 0.0]]), SimParams())


def test_fresh_nca_is_identity():
    x = _state()
    assert torch.allclose(NCA()(x), x)


def test_param_names():
    assert [n for n, _ in NCA().named_parameters()] == ["W1", "b1", "W2", "b2"]
    assert [n for n, _ in NCA(NCAConfig(hidden=0)).named_parameters()] == ["W2", "b2"]
    m = NCA()
    assert m.W1.shape == (128, 80) and m.W2.shape == (20, 128)


def test_perception_matches_simulator_laplacian():
    x = torch.rand(1, 20, SIZE, SIZE)
    p = NCA().perceive(x)
    lap = p[:, 3::4]
    ref = Simulator._mean_n(x[:, T]) - x[:, T]
    assert torch.allclose(lap[:, T], ref, atol=1e-6)


def test_step_keeps_wind_border_and_range():
    m = NCA()
    with torch.no_grad():
        m.W2.normal_(0, 1.0)
        m.b2.normal_(0, 1.0)
    x = _state()
    y = m(x)
    ring = border_mask(SIZE, SIZE, "cpu")
    assert torch.equal(y[:, WX:], x[:, WX:])
    assert y[:, :WX].min() >= 0 and y[:, :WX].max() <= 1
    assert torch.all(y[0, GROUND][ring] == 1)


def test_events_injected_like_simulator():
    x = _state()
    masks = events_to_masks([(0, "accensione", (5, 5), {})], 1, SIZE, SIZE).unsqueeze(0)
    y = NCA()(x, masks[:, 0])
    assert y[0, F, 5, 5] == 1 and y[0, T, 5, 5] >= SimParams().ignite_t


def test_inject_matches_simulator():
    gen = SandboxGenerator(GeneratorConfig(size=SIZE), seed=3)
    b = gen.sample(4, "joint_full", steps=6, keep_info=False, balance=False)
    masks = torch.rand(4, 3, SIZE, SIZE) < 0.2
    x = b.trajectory[:, 3]
    assert torch.equal(inject_events(x, masks, SimParams()), sim_inject(x, masks, SimParams()))


def test_last_layer_merge_is_exact():
    # with W1, b1 shared, the update is linear in (W2, b2): adding task vectors adds the deltas
    torch.manual_seed(0)
    base = NCA()
    a, b = NCA(), NCA()
    for m in (a, b):
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            m.W2.normal_(0, 0.05)
            m.b2.normal_(0, 0.05)
    merged = NCA()
    merged.load_state_dict({k: base.state_dict()[k] + (a.state_dict()[k] - base.state_dict()[k])
                            + (b.state_dict()[k] - base.state_dict()[k]) for k in base.state_dict()})
    x = torch.rand(2, 20, SIZE, SIZE)
    oracle = base.delta(x) + (a.delta(x) - base.delta(x)) + (b.delta(x) - base.delta(x))
    assert torch.allclose(merged.delta(x), oracle, atol=1e-5)


def test_trainable_last():
    m = NCA()
    m.set_trainable("last")
    assert [n for n, p in m.named_parameters() if p.requires_grad] == ["W2", "b2"]


def test_curriculum():
    cfg = TrainConfig(iters=100, k_min=8, k_max=64, curriculum_frac=0.5)
    ks = [window_length(i, cfg) for i in range(1, 101)]
    assert ks[0] == 8 and ks[-1] == 64 and all(k % 8 == 0 for k in ks) and ks == sorted(ks)


def test_pool_windows_follow_simulator():
    gen = SandboxGenerator(GeneratorConfig(size=SIZE), seed=0)
    pool = WindowPool(gen, "T1", n_scenes=3, ticks=10, storage_dtype=torch.float32)
    start, target, masks, weights, lead = pool.sample(4, 3, lead=2)
    assert start.shape == (4, 20, SIZE, SIZE) and target.shape == (4, 3, 20, SIZE, SIZE)
    assert weights is None and lead[0].shape == start.shape and lead[1].shape == (4, 2, 3, SIZE, SIZE)
    act = torch.zeros(4, 8, dtype=torch.bool)
    act[:, 0] = True
    nxt, _ = gen.sim.step(start, act, masks[:, 0], diagnostics=False)
    assert torch.allclose(nxt, target[:, 0], atol=1e-5)


def test_ambiguity_mask_only_hides_other_rules():
    gen = SandboxGenerator(GeneratorConfig(size=SIZE), seed=1)
    pool = WindowPool(gen, "joint_single", n_scenes=6, ticks=6, ambiguity_mask=True)
    assert pool.weights.shape == (6, 6, SIZE, SIZE)
    assert pool.weights.float().mean() > 0.5


def test_support_weights_and_sampling():
    gen = SandboxGenerator(GeneratorConfig(size=SIZE), seed=2)
    pool = WindowPool(gen, "T1", n_scenes=8, ticks=12, support_share=0.9, support_sampling=1.0)
    w = pool.weights
    on = w > 1
    assert torch.equal(on.flatten(2).sum(2).float(), pool.counts)
    assert abs(float((w * on).sum() / w.sum()) - 0.9) < 0.01
    # with full support sampling every window holds some rule action
    _, _, _, weights, _ = pool.sample(16, 3)
    assert (weights > 1).flatten(1).any(1).all()


def test_base_pool_has_no_weights():
    gen = SandboxGenerator(GeneratorConfig(size=SIZE), seed=2)
    pool = WindowPool(gen, "base", n_scenes=2, ticks=4, support_share=0.3, support_sampling=1.0)
    assert pool.weights is None and pool.counts is None


def test_train_reduces_loss_and_saves():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "base"
        res = train(_tiny(iters=40, eval_every=20, lr=3e-3), out=out, log=lambda *_: None)
        assert res.history[-1]["val_mse"] <= res.history[0]["val_mse"] * 1.05
        model, ckpt = load_nca(f"{out}.pt")
        assert ckpt["rules"] == "base" and Path(f"{out}_best.pt").exists()
        assert torch.equal(model.W2, res.model.W2)


def test_finetune_starts_from_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        m = NCA()
        with torch.no_grad():
            m.W2.normal_(0, 0.01)
        save_nca(Path(tmp) / "b.pt", m, rules="base")
        res = train(_tiny(rules="T1", iters=1, lr=1e-9), init=str(Path(tmp) / "b.pt"), log=lambda *_: None)
        assert torch.allclose(res.model.W2, m.W2, atol=1e-6)


def test_zoo_all_groups():
    with tempfile.TemporaryDirectory() as tmp:
        args = ["--out", tmp, "--size", str(SIZE), "--batch", "2", "--base-iters", "2", "--iters", "1",
                "--rules", "T1", "T2", "--device", "cpu"]
        for group in ("base", "specialists", "scratch", "joint_single", "joint_full"):
            zoo_main([group] + args)
        names = sorted(p.relative_to(tmp).as_posix() for p in Path(tmp).rglob("*.pt") if "_best" not in p.name)
        assert names == ["base/base_s0.pt", "joint_full/joint_full_s0.pt", "joint_single/joint_single_s0.pt",
                         "scratch/T1_s0.pt", "scratch/T2_s0.pt", "specialists/T1_s0.pt", "specialists/T2_s0.pt"]
        _, ck = load_nca(Path(tmp) / "joint_single/joint_single_s0.pt")
        assert ck["train_cfg"]["iters"] == 2 and ck["init"].endswith("base_s0.pt")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")