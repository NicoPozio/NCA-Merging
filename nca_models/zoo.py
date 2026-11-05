"""The models of the thesis, one command each.

    python -m nca_models.zoo base                          base NCA from random weights
    python -m nca_models.zoo specialists                   T1..T8 fine-tuned from the base
    python -m nca_models.zoo scratch                       T1..T8 from random weights
    python -m nca_models.zoo joint_single                  one NCA, every rule but always one at a time
    python -m nca_models.zoo joint_full                    one NCA, every rule always together
    python -m nca_models.zoo all                           the five groups in order

Checkpoints go to <out>/<group>/<name>.pt (last) and <name>_best.pt (best validation).
With --time-limit the run stops cleanly before the limit; launching the same command again skips
the finished models and resumes the interrupted one from <name>_resume.pt.
Budgets: a specialist gets --iters steps; the joint models get one specialist budget per rule,
so they see as many updates as all the specialists together.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from nca_sandbox.config import N_RULES

from .model import NCAConfig
from .train import TrainConfig, resume_path, train

RULE_NAMES = tuple(f"T{i + 1}" for i in range(N_RULES))
GROUPS = ("base", "specialists", "scratch", "joint_single", "joint_full")


class OutOfTime(Exception):
    pass


DEADLINE = None


def _run(name, out_dir, cfg, init, model_cfg, trainable="all"):
    out = Path(out_dir) / name
    final = Path(f"{out}.pt")
    if final.exists() and not resume_path(out).exists():
        print(f"== {name}: already trained, skipped")
        return final
    print(f"\n== {name}: rules={cfg.rules} init={init} iters={cfg.iters} trainable={trainable}")
    res = train(cfg, init=init, model_cfg=model_cfg, trainable=trainable, out=out, deadline=DEADLINE)
    (out.parent / f"{out.name}_history.json").write_text(json.dumps(res.history, indent=1))
    if not res.finished:
        raise OutOfTime(name)
    return final


def _arch(args):
    """Name tag for a non-default network, so the width and activation variants never skip or
    overwrite each other's checkpoints."""
    return "" if (args.hidden, args.act) == (128, "relu") else f"_h{args.hidden}_{args.act}"


def base_path(args):
    return Path(args.base) if args.base else Path(args.out) / "base" / f"base{_arch(args)}_s{args.seed}.pt"


def run_group(group, args):
    model_cfg = NCAConfig(hidden=args.hidden, act=args.act)
    common = TrainConfig(batch=args.batch, seed=args.seed, size=args.size,
                         **({"device": args.device} if args.device else {}))
    suffix = f"{_arch(args)}_s{args.seed}"

    if group == "base":
        cfg = replace(common, rules="base", iters=args.base_iters)
        return [_run(f"base/base{suffix}", args.out, cfg, "random", model_cfg)]

    if group in ("specialists", "scratch"):
        paths = []
        for rule in args.rules:
            cfg = replace(common, rules=rule, iters=args.iters)
            if group == "specialists":
                init = base_path(args)
                if not init.exists():
                    raise FileNotFoundError(f"{init} not found: train the base first or pass --base")
                tag = "_last" if args.trainable == "last" else ""
                paths.append(_run(f"specialists/{rule}{tag}{suffix}", args.out, cfg, str(init), model_cfg,
                                  args.trainable))
            else:
                paths.append(_run(f"scratch/{rule}{suffix}", args.out, cfg, "random", model_cfg))
        return paths

    if group in ("joint_single", "joint_full"):
        iters = args.joint_iters or args.iters * len(args.rules)
        cfg = replace(common, rules=group, iters=iters, ambiguity_mask=not args.no_ambiguity_mask)
        init = "random" if args.joint_init == "random" else str(base_path(args))
        tag = "_nomask" if group == "joint_single" and args.no_ambiguity_mask else ""
        tag += "_rand" if args.joint_init == "random" else ""
        return [_run(f"{group}/{group}{tag}{suffix}", args.out, cfg, init, model_cfg)]

    raise ValueError(group)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the NCAs of the merging thesis.")
    ap.add_argument("group", choices=GROUPS + ("all",))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128, help="0 = no hidden layer (linear NCA)")
    ap.add_argument("--act", default="relu", choices=("relu", "abs", "none"))
    ap.add_argument("--base-iters", type=int, default=20000)
    ap.add_argument("--iters", type=int, default=5000, help="steps per specialist")
    ap.add_argument("--joint-iters", type=int, default=None, help="default: --iters x number of rules")
    ap.add_argument("--rules", nargs="+", default=list(RULE_NAMES), choices=RULE_NAMES)
    ap.add_argument("--base", default=None, help="base checkpoint (default: <out>/base/base_s<seed>.pt)")
    ap.add_argument("--trainable", default="all", choices=("all", "last"),
                    help="specialists only: 'last' trains W2, b2 (experiment E1)")
    ap.add_argument("--joint-init", default="base", choices=("base", "random"))
    ap.add_argument("--no-ambiguity-mask", action="store_true",
                    help="joint_single: keep the loss on cells where an inactive rule would act")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="hours after which the current model is saved and the run stops")
    args = ap.parse_args(argv)

    global DEADLINE
    DEADLINE = None if args.time_limit is None else time.time() + args.time_limit * 3600
    try:
        for group in (GROUPS if args.group == "all" else (args.group,)):
            run_group(group, args)
    except OutOfTime as stop:
        print(f"\nTime limit reached during {stop}: launch the same command again to resume.")
        return False
    return True


if __name__ == "__main__":
    main()