import copy
import math
import csv
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nca_models.evaluate import default_datasets
from nca_models.merging import (
    dataset_for, hp_tag, merge, merge_average, merge_ta, state, task_vectors,
)
from nca_models.model import NCA, load_nca, save_nca
from nca_models.train import TrainConfig
from nca_sandbox.dataset import save_split
from tools.merge import main as merge_main, parse_sets, retention

SIZE = 12
STEPS = 8


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


def _setup():
    torch.manual_seed(0)
    base = NCA()
    specs = {"T1": _noisy(base, 1), "T2": _noisy(base, 2), "T3": _noisy(base, 3)}
    return base, specs, task_vectors(base, specs)


def test_ta_limits():
    base, specs, taus = _setup()
    b = state(base)
    zero = merge_ta(b, taus, alpha=0.0)
    assert all(torch.equal(zero[k], b[k]) for k in b)
    one = merge_ta(b, {"T1": taus["T1"]}, alpha=1.0)
    s1 = state(specs["T1"])
    assert all(torch.allclose(one[k], s1[k], atol=1e-6) for k in b)


def test_average_is_mean_of_specialists():
    base, specs, taus = _setup()
    avg = merge_average(state(base), taus)
    mean = {k: sum(state(m)[k] for m in specs.values()) / 3 for k in state(base)}
    assert all(torch.allclose(avg[k], mean[k], atol=1e-6) for k in avg)
    ta = merge_ta(state(base), taus, alpha=1 / 3)
    assert all(torch.allclose(avg[k], ta[k], atol=1e-6) for k in avg)


def test_weights_and_per_rule_alpha():
    base, specs, taus = _setup()
    w = merge_average(state(base), taus, weights={"T1": 2, "T2": 1, "T3": 1})
    a = merge_ta(state(base), taus, alpha={"T1": 0.5, "T2": 0.25, "T3": 0.25})
    assert all(torch.allclose(w[k], a[k], atol=1e-6) for k in w)


def test_merge_returns_new_model():
    base, _, taus = _setup()
    before = state(base)
    m = merge(base, taus, "ta", alpha=0.5)
    assert all(torch.equal(state(base)[k], before[k]) for k in before)
    assert not torch.equal(state(m)["W2"], before["W2"])
    with pytest.raises(ValueError):
        merge(base, taus, "nope")


def test_names():
    assert dataset_for(["T6", "T1"]) == "T1+T6"
    assert dataset_for([f"T{i}" for i in range(1, 9)]) == "joint_full"
    assert hp_tag("ta", {"alpha": 0.5}) == "ta_a0.5"
    assert hp_tag("average", {}) == "average"
    assert len(parse_sets(["pairs"])) == 28 and parse_sets(["T1+T6"]) == [["T1", "T6"]]
    assert default_datasets("T1+T6") == ["T1+T6", "T1", "T6"]


def test_retention():
    assert retention(1.0, 0.5, 0.0) == 0.5
    assert retention(1.0, 2.0, 0.0) == -1.0
    assert math.isnan(retention(1.0, 0.5, 1.0))


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        splits = Path(tmp) / "splits"
        for d in ("T1", "T2", "T1+T2"):
            save_split(splits / f"{d}_val_{SIZE}.pt", d, n=4, steps=STEPS, seed=1000, size=SIZE)
        base = NCA()
        bp = _save(Path(tmp) / "runs/base/base_s0.pt", base, "base")
        for i, r in enumerate(("T1", "T2")):
            _save(Path(tmp) / f"runs2/specialists/{r}_s0_best.pt", _noisy(base, i), r, init=str(bp))
        rows = merge_main(["--base", str(bp), "--specialists", f"{tmp}/runs2/specialists", "--sets", "T1+T2",
                           "--alphas", "0.5", "1.0", "--out", f"{tmp}/runs_merges", "--steps", str(STEPS), "--n", "4",
                           "--splits-dir", str(splits), "--records-dir", f"{tmp}/rec", "--device", "cpu"])
        assert [r["setting"] for r in rows] == ["average", "ta_a0.5", "ta_a1"]
        assert all("ret_T1" in r and "ret_T2" in r for r in rows)
        ck = Path(tmp, "runs_merges/merges/ta_a0.5__T1+T2.pt")
        _, meta = load_nca(ck)
        assert meta["rules"] == "T1+T2" and meta["merge"]["method"] == "ta" and meta["merge"]["hp"] == {"alpha": 0.5}
        with open(Path(tmp, "runs_merges/merge_summary.csv")) as f:
            assert len(list(csv.DictReader(f))) == 3
        # a second run with the same settings updates the rows instead of duplicating them
        merge_main(["--base", str(bp), "--specialists", f"{tmp}/runs2/specialists", "--sets", "T1+T2",
                    "--methods", "average", "--out", f"{tmp}/runs_merges", "--steps", str(STEPS), "--n", "4",
                    "--splits-dir", str(splits), "--records-dir", f"{tmp}/rec", "--device", "cpu"])
        with open(Path(tmp, "runs_merges/merge_summary.csv")) as f:
            assert len(list(csv.DictReader(f))) == 3


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")