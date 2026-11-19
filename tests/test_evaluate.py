import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nca_models.evaluate import (
    evaluate_checkpoint, evaluate_split, open_split, record_path, rule_curve, window_mean,
)
from nca_models.model import NCA, load_nca, save_nca
from nca_models.train import TrainConfig
from nca_models.zoo import main as zoo_main
from nca_sandbox.config import WX
from nca_sandbox.dataset import save_split
from tools.figures import render_rgb
from tools.horizon_check import main as horizon_main
from tools.results import main as results_main

SIZE = 12
STEPS = 16


def _noisy_nca(seed):
    torch.manual_seed(seed)
    m = NCA()
    with torch.no_grad():
        m.W2.normal_(0, 0.01)
    return m


def _ckpt(path, rules, seed=0, lr=1e-4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(TrainConfig(rules=rules, size=SIZE, seed=seed, lr=lr))
    save_nca(path, _noisy_nca(seed), rules=rules, init="random", trainable="all", train_cfg=cfg)
    return path


def _splits(root, datasets=("T1", "T5", "base")):
    for d in datasets:
        save_split(Path(root) / f"{d}_val_{SIZE}.pt", d, n=6, steps=STEPS, seed=1000, size=SIZE)
    return Path(root)


def test_streaming_matches_rollout():
    with tempfile.TemporaryDirectory() as tmp:
        sp = open_split(_splits(tmp, ("T1",)) / f"T1_val_{SIZE}.pt")
        model = _noisy_nca(1)
        res = evaluate_split(model, sp, STEPS, chunk=4)
        b = sp.batch(range(len(sp)))
        with torch.no_grad():
            pred = model.rollout(b.initial, STEPS, b.event_masks)[:, 1:, :WX]
        sq = (pred - b.trajectory[:, 1:, :WX]) ** 2
        assert torch.allclose(res["mse_all"], sq.mean((0, 2, 3, 4)), rtol=1e-4, atol=1e-9)
        sup = (b.support & b.active[:, None, :, None, None]).any(2).unsqueeze(2)
        ref = (sq * sup).sum() / (sup.sum() * WX)
        assert abs(window_mean(res, "mse_rule", 1, STEPS) - float(ref)) < 1e-5 * max(1.0, float(ref))


def test_window_mean_weights_ticks_by_rule_cells():
    rec = {"steps": 4, "rule_sum": torch.tensor([1.0, 0.0, 9.0, 0.0]), "rule_cnt": torch.tensor([1.0, 0.0, 3.0, 0.0]),
           "mse_all": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    assert window_mean(rec, "mse_rule", 1, 4) == pytest.approx(10 / 4)
    assert window_mean(rec, "mse_all", 3, 8) == pytest.approx(3.5)
    assert torch.isnan(rule_curve(rec)[1])


def test_record_is_reused_until_checkpoint_changes():
    with tempfile.TemporaryDirectory() as tmp:
        splits = _splits(tmp, ("T1",))
        ck = _ckpt(Path(tmp) / "run_a/specialists/T1_s0.pt", "T1")
        kw = dict(dataset="T1", steps=STEPS, n=4, splits_dir=splits, records_dir=Path(tmp) / "rec")
        first = evaluate_checkpoint(ck, **kw)
        assert record_path(Path(tmp) / "rec", ck, "T1", "val", STEPS, 4).exists()
        assert evaluate_checkpoint(ck, **kw)["ckpt_hash"] == first["ckpt_hash"]
        _ckpt(ck, "T1", seed=5)
        again = evaluate_checkpoint(ck, **kw)
        assert again["ckpt_hash"] != first["ckpt_hash"]
        assert not torch.allclose(again["mse_all"], first["mse_all"])


def test_p1_reuse_and_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        old, new, bad = (str(Path(tmp) / d) for d in ("old", "new", "bad"))
        args = ["--size", str(SIZE), "--batch", "2", "--iters", "2", "--phase1-iters", "1", "--rules", "T1",
                "--device", "cpu"]
        zoo_main(["scratch", "--out", old] + args)
        zoo_main(["scratch", "--out", new, "--p1-from", old, "--phase2-lr", "3e-4"] + args)
        assert not (Path(new) / "scratch/T1_s0_p1.pt").exists()
        _, ck = load_nca(Path(new) / "scratch/T1_s0.pt")
        assert ck["init"] == str(Path(old) / "scratch/T1_s0_p1.pt") and ck["train_cfg"]["lr"] == 3e-4
        with pytest.raises(ValueError, match="lr"):
            zoo_main(["scratch", "--out", bad, "--p1-from", old, "--lr", "5e-4"] + args)


def test_horizon_check_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        splits = _splits(tmp)
        for run, lr in (("old", 1e-4), ("new", 3e-4)):
            for i, rule in enumerate(("T1", "T5")):
                for tag in ("", "_best"):
                    _ckpt(Path(tmp) / run / f"specialists/{rule}_s0{tag}.pt", rule, seed=i + (run == "new"), lr=lr)
        ok = horizon_main(["--old", f"{tmp}/old", "--new", f"{tmp}/new", "--rules", "T1", "T5", "--steps",
                           str(STEPS), "--n", "4", "--splits-dir", str(splits), "--records-dir", f"{tmp}/rec",
                           "--plots", f"{tmp}/fig", "--device", "cpu"])
        assert isinstance(ok, bool)
        assert (Path(tmp) / "fig/T1_last_rule.png").exists() and (Path(tmp) / "fig/T5_best_drift.png").exists()


def test_results_eval_table_plots():
    with tempfile.TemporaryDirectory() as tmp:
        splits = _splits(tmp)
        cks = [_ckpt(Path(tmp) / "run/specialists/T1_s0.pt", "T1"), _ckpt(Path(tmp) / "run/scratch/T1_s0.pt", "T1", 3),
               _ckpt(Path(tmp) / "run/base/base_s0.pt", "base", 4)]
        _ckpt(Path(tmp) / "run/specialists/T1_s0_p1.pt", "T1")
        common = ["--steps", str(STEPS), "--n", "4", "--splits-dir", str(splits), "--records-dir", f"{tmp}/rec",
                  "--device", "cpu"]
        results_main(["eval", *map(str, cks), "--datasets", "T1", "base"] + common)
        assert not any(Path(tmp, "rec").rglob("T1_s0_p1__*"))
        out = ["--out", f"{tmp}/res"]
        results_main(["table", "--window", "1-8"] + common + out)
        table = Path(tmp, "res/table_mse_rule_1-8.md").read_text()
        assert "run/specialists" in table and "run/base/base_s0" in table
        assert Path(tmp, "res/summary.csv").read_text().count("\n") > 3
        results_main(["plot", "heatmap", "--window", "1-8"] + common + out)
        results_main(["plot", "curves", "--dataset", "T1"] + common + out)
        results_main(["plot", "drift", "--dataset", "T1", "--channels", "F", "T"] + common + out)
        results_main(["plot", "frames", str(cks[0]), str(cks[1]), "--dataset", "T1", "--ticks", "0", "4", "16"]
                     + common + out)
        for name in ("heatmap_mse_rule_1-8", "curves_T1_mse_rule", "drift_T1", "frames_T1_scene0"):
            assert Path(tmp, f"res/{name}.png").exists()


def test_render_rgb_range():
    img = render_rgb(torch.rand(20, SIZE, SIZE))
    assert img.shape == (SIZE, SIZE, 3) and img.min() >= 0 and img.max() <= 1


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")