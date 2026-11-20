import copy
import csv
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nca_models.evaluate import default_datasets
from nca_models.merging import (
    dataset_for, hp_tag, merge, merge_average, merge_dare, merge_ta, merge_ties, merge_tsv, state, task_vectors,
)
from nca_models.model import NCA, load_nca, save_nca
from nca_models.train import TrainConfig
from nca_sandbox.dataset import save_split
from tools.merge import log_retention, main as merge_main, parse_sets, retention
from tools.merge_report import main as report_main

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


def _close(a, b):
    return all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)


def test_ties():
    base, specs, taus = _setup()
    b = state(base)
    assert _close(merge_ties(b, {"T1": taus["T1"]}, k=1.0, lam=1.0), state(specs["T1"]))
    opposite = {"A": taus["T1"], "B": {k: -v for k, v in taus["T1"].items()}}
    assert _close(merge_ties(b, opposite, k=1.0, lam=1.0), b)
    twice = {"A": taus["T1"], "B": taus["T1"]}
    assert _close(merge_ties(b, twice, k=1.0, lam=1.0), state(specs["T1"]))
    trimmed = merge_ties(b, {"T1": taus["T1"]}, k=0.2, lam=1.0)
    changed = sum(int((trimmed[k] != b[k]).sum()) for k in b if b[k].is_floating_point())
    total = sum(b[k].numel() for k in b if b[k].is_floating_point())
    assert changed <= round(0.2 * total)


def test_dare():
    base, _, taus = _setup()
    b = state(base)
    assert _close(merge_dare(b, taus, p=0.0, alpha=0.5), merge_ta(b, taus, alpha=0.5))
    assert _close(merge_dare(b, taus, p=0.3, alpha=0.5, seed=1), merge_dare(b, taus, p=0.3, alpha=0.5, seed=1))
    assert not _close(merge_dare(b, taus, p=0.3, alpha=0.5, seed=1), merge_dare(b, taus, p=0.3, alpha=0.5, seed=2))


def test_tsv():
    base, specs, taus = _setup()
    b = state(base)
    # one task, every singular vector kept: the merge is the specialist
    assert _close(merge_tsv(b, {"T1": taus["T1"]}, alpha=1.0, keep=1.0), state(specs["T1"]))
    # two rank-one updates with orthogonal directions: nothing to straighten, the merge is their sum
    q, _ = torch.linalg.qr(torch.randn(b["W2"].shape[0], b["W2"].shape[0]))
    r, _ = torch.linalg.qr(torch.randn(b["W2"].shape[1], b["W2"].shape[1]))
    zero = {k: torch.zeros_like(v) for k, v in b.items()}
    t1 = {**zero, "W2": 2 * torch.outer(q[:, 0], r[:, 0])}
    t2 = {**zero, "W2": 3 * torch.outer(q[:, 1], r[:, 1])}
    out = merge_tsv(b, {"A": t1, "B": t2}, alpha=1.0, keep=1 / b["W2"].shape[0])
    assert torch.allclose(out["W2"], b["W2"] + t1["W2"] + t2["W2"], atol=1e-5)


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
    assert abs(log_retention(1.0, 0.1, 0.01) - 0.5) < 1e-9
    assert abs(log_retention(1.0, 0.01, 0.01) - 1) < 1e-9 and abs(log_retention(1.0, 1.0, 0.01)) < 1e-9
    # close base and specialist: the plain retention blows up, the log one stays small
    assert retention(1.0, 1.1, 0.99) < -9 and log_retention(1.0, 1.1, 0.99) > -10


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
                           "--methods", "average", "ta", "--alphas", "0.5", "1.0", "--out", f"{tmp}/runs_merges",
                           "--steps", str(STEPS), "--n", "4", "--splits-dir", str(splits), "--records-dir",
                           f"{tmp}/rec", "--device", "cpu"])
        assert [r["setting"] for r in rows] == ["average", "ta_a0.5", "ta_a1"]
        assert all("lret_T1" in r and "err_T2" in r and "lret_mean" in r for r in rows)
        ck = Path(tmp, "runs_merges/merges/ta_a0.5__T1+T2.pt")
        _, meta = load_nca(ck)
        assert meta["rules"] == "T1+T2" and meta["merge"]["method"] == "ta" and meta["merge"]["hp"] == {"alpha": 0.5}
        with open(Path(tmp, "runs_merges/merge_summary.csv")) as f:
            assert len(list(csv.DictReader(f))) == 3
        # a second run with the same settings updates the rows instead of duplicating them,
        # and leaves the checkpoints untouched so their evaluations are reused
        stamp = ck.stat().st_mtime_ns
        merge_main(["--base", str(bp), "--specialists", f"{tmp}/runs2/specialists", "--sets", "T1+T2",
                    "--methods", "average", "ta", "--alphas", "0.5", "--out", f"{tmp}/runs_merges", "--steps",
                    str(STEPS), "--n", "4", "--splits-dir", str(splits), "--records-dir", f"{tmp}/rec",
                    "--device", "cpu"])
        assert ck.stat().st_mtime_ns == stamp
        with open(Path(tmp, "runs_merges/merge_summary.csv")) as f:
            assert len(list(csv.DictReader(f))) == 3
        # the other methods, then the report
        merge_main(["--base", str(bp), "--specialists", f"{tmp}/runs2/specialists", "--sets", "T1+T2",
                    "--methods", "ties", "dare", "tsv", "--alphas", "0.5", "--lams", "0.5", "--dare-p", "0.3",
                    "--dare-alphas", "0.5", "--out", f"{tmp}/runs_merges", "--steps", str(STEPS), "--n", "4",
                    "--splits-dir", str(splits), "--records-dir", f"{tmp}/rec", "--device", "cpu"])
        with open(Path(tmp, "runs_merges/merge_summary.csv")) as f:
            assert {r["method"] for r in csv.DictReader(f)} == {"average", "ta", "ties", "dare", "tsv"}


def _summary_row(ds, method, hp, lrets, gap=None):
    rules = list(lrets)
    row = {"set": ds, "setting": hp_tag(method, hp), "method": method, "hp": json.dumps(hp, sort_keys=True),
           "scale": hp.get("alpha", hp.get("lam", "")), "split": "val", "steps": 64, "n": 64,
           "comb_rule": 0.01, "base_rule": 0.02, "lret_mean": sum(lrets.values()) / len(rules),
           "lret_min": min(lrets.values()), "ret_mean": 0.0}
    row.update({f"lret_{r}": v for r, v in lrets.items()})
    if gap is not None:
        row["gap_closed"] = gap
    return row


def test_report_from_summary():
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for ds, (a, b) in (("T1+T2", ("T1", "T2")), ("T1+T3", ("T1", "T3")), ("T2+T3", ("T2", "T3"))):
            for alpha, x in ((0.3, 0.4), (0.5, 0.2)):
                rows.append(_summary_row(ds, "ta", {"alpha": alpha}, {a: x, b: x - 0.5}))
            rows.append(_summary_row(ds, "average", {}, {a: 0.2, b: -0.3}))
            rows.append(_summary_row(ds, "dare", {"p": 0.3, "alpha": 0.3}, {a: 0.1, b: 0.1}))
        all8 = {f"T{i}": 0.1 * i for i in range(1, 9)}
        rows.append(_summary_row("joint_full", "ta", {"alpha": 0.1}, all8, gap=0.2))
        rows.append(_summary_row("joint_full", "tsv", {"alpha": 1.0}, all8, gap=0.5))
        fields = list(dict.fromkeys(k for r in rows for k in r))
        summary = Path(tmp, "merge_summary.csv")
        with open(summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
        pairs = Path(tmp, "pairs.csv")
        pairs.write_text("pair,cos,support_overlap\nT1-T2,0.1,0.3\nT1-T3,0.2,0.1\nT2-T3,0.3,0.2\n")
        best = report_main(["--summary", str(summary), "--pairs", str(pairs), "--out", f"{tmp}/report"])
        assert best[("T1+T2", "ta")]["setting"] == "ta_a0.3" and best[("joint_full", "tsv")]["gap_closed"] == 0.5
        for name in ("best_settings.csv", "best_settings.md", "scale_all.png", "scale_pairs.png", "matrix_any.png",
                     "matrix_ta.png", "matrix_dare.png", "scatter_any.png", "predictors_ta.csv"):
            assert Path(tmp, "report", name).exists(), name


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")