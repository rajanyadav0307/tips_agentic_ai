"""Command-line interface: ``agent-tips <step>``.

Each subcommand runs one step of the locked data science procedure;
``run-all`` runs them all in order.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_tips import __version__
from agent_tips.config import load_config
from agent_tips.utils.device import select_device, torch_available


def _env() -> int:
    cfg = load_config()
    print(f"agent-tips {__version__}")
    print(f"python     {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"platform   {platform.system()} {platform.machine()}")
    print(f"root       {cfg.root}")
    print(f"torch      {'installed' if torch_available() else 'not installed (CPU only)'}")
    print(f"device     {select_device(cfg.raw.get('compute', {}).get('device', 'auto'))}")
    return 0


def _download(args: argparse.Namespace) -> int:
    from agent_tips.data.download import download

    paths = download(load_config(), only=args.only, verify=args.verify, dry_run=args.dry_run)
    print(f"done: {len(paths)} files")
    return 0


def _clean(_: argparse.Namespace) -> int:
    from agent_tips.cleaning import pipeline, report

    cfg = load_config()
    pipeline.run(cfg)
    print(f"report: {report.build(cfg)}")
    return 0


def _eda(_: argparse.Namespace) -> int:
    from agent_tips.eda import charts

    charts.run(load_config())
    return 0


def _features(_: argparse.Namespace) -> int:
    from agent_tips.features import build, report

    cfg = load_config()
    build.run(cfg)
    print(f"report: {report.build(cfg)}")
    return 0


def _model(_: argparse.Namespace) -> int:
    from agent_tips.models import analysis, report

    cfg = load_config()
    analysis.run(cfg)
    print(f"report: {report.build(cfg)}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from agent_tips.evaluation import report, robustness

    cfg = load_config()
    robustness.run(cfg, repro=not args.no_repro)
    print(f"report: {report.build(cfg)}")
    return 0


def _figures(_: argparse.Namespace) -> int:
    from agent_tips.viz import figures, tables

    cfg = load_config()
    figures.run(cfg)
    tables.run(cfg)
    return 0


def _run_all(args: argparse.Namespace) -> int:
    """Run Steps 2-8 in order; stops at the first step that fails."""
    steps = [
        ("download", _download),
        ("clean", _clean),
        ("eda", _eda),
        ("features", _features),
        ("model", _model),
        ("evaluate", _evaluate),
        ("figures", _figures),
    ]
    for name, fn in steps:
        print(f"== {name}")
        code = fn(args)
        if code:
            return code
    return 0


def _onetry(args: argparse.Namespace) -> int:
    """Paper 2 (one-try study): run one step of its procedure."""
    cfg = load_config()
    if args.step == "collect":
        from agent_tips.onetry import collect

        print(f"report: {collect.run(cfg, rehash=not args.quick)}")
    elif args.step == "clean":
        from agent_tips.onetry import clean

        print(f"report: {clean.run(cfg)}")
    elif args.step == "eda":
        from agent_tips.onetry import eda

        print(f"report: {eda.run(cfg)}")
    elif args.step == "features":
        from agent_tips.onetry import features

        print(f"report: {features.run(cfg, with_embeddings=args.embeddings)}")
    elif args.step == "model":
        from agent_tips.onetry import model

        print(f"report: {model.run(cfg)}")
    elif args.step == "evaluate":
        from agent_tips.onetry import evaluate

        print(f"report: {evaluate.run(cfg, fresh=args.fresh)}")
    else:
        return _onetry_share(args, cfg)
    return 0


def _onetry_share(args: argparse.Namespace, cfg: Any) -> int:
    """Paper 2, Step 7 shared across computers (amendment A9)."""
    from agent_tips.onetry import share

    if args.step in ("run", "bundle", "pack") and not args.machine:
        print(f"onetry {args.step} needs --machine (one of {sorted(share.machines(cfg))})")
        return 2
    if args.step == "plan":
        share.plan(cfg)
    elif args.step == "run":
        share.run(cfg, args.machine, jobs=args.jobs)
        left = share.missing(cfg, share.assigned(cfg, args.machine))
        print("done: every task finished" if not left else f"{len(left)} tasks left; run again")
    elif args.step == "bundle":
        share.bundle(cfg, args.machine)
    elif args.step == "pack":
        share.pack(cfg, args.machine)
    elif args.step == "merge":
        if not args.file:
            print("onetry merge needs the results zip: agent-tips onetry merge results-NAME.zip")
            return 2
        share.merge(cfg, Path(args.file))
    elif args.step == "crosscheck":
        out = share.crosscheck(cfg)
        print("crosscheck " + ("passed" if out["passed"] else "FAILED") + f": {out['rows']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="agent-tips", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="show platform, paths and compute device")
    dl = sub.add_parser("download", help="Step 2: download pinned data and verify checksums")
    dl.add_argument("--only", nargs="+", metavar="DATASET", help="dataset keys from the config")
    dl.add_argument("--verify", action="store_true", help="re-hash files already downloaded")
    dl.add_argument("--dry-run", action="store_true", help="only show what would be downloaded")
    sub.add_parser("clean", help="Step 3: build clean tries/steps tables and the cleaning report")
    sub.add_parser("eda", help="Step 4: exploration charts and their numbers")
    sub.add_parser("features", help="Step 5: labels, label validation, issue and try features")
    sub.add_parser("model", help="Step 6: fair comparisons, regressions, survival, forks")
    ev = sub.add_parser("evaluate", help="Step 7: robustness checks and reproducibility")
    ev.add_argument(
        "--no-repro", action="store_true", help="skip the two-run reproducibility check"
    )
    sub.add_parser("figures", help="Step 8: paper figures, tables and number macros")
    ot = sub.add_parser("onetry", help="Paper 2 (one-try study): run one step of its procedure")
    ot.add_argument(
        "step",
        choices=[
            "collect",
            "clean",
            "eda",
            "features",
            "model",
            "evaluate",
            "plan",
            "run",
            "bundle",
            "pack",
            "merge",
            "crosscheck",
        ],
        help="collect: verify sources and extract (Step 2); clean: cleaning rules (Step 3); "
        "eda: exploration and comparison inventory (Step 4); features: Step 5; "
        "model: answer keys and go/no-go (Step 6); evaluate: full run and tests (Step 7). "
        "Step 7 on several computers: plan (split the work), bundle (zip for a partner), "
        "run (this machine's share), pack (zip the results), merge (add a results zip), "
        "crosscheck (re-run a few tasks of the other machines here)",
    )
    ot.add_argument("file", nargs="?", help="merge: the results zip to add")
    ot.add_argument("--machine", help="run/bundle/pack: which computer (see the config)")
    ot.add_argument("--jobs", type=int, help="run: parallel tasks (default from the config)")
    ot.add_argument("--quick", action="store_true", help="collect: check sizes, skip re-hashing")
    ot.add_argument(
        "--embeddings", action="store_true", help="features: also compute rung-3 embeddings (GPU)"
    )
    ot.add_argument("--fresh", action="store_true", help="evaluate: redo comparisons already done")
    ra = sub.add_parser("run-all", help="Steps 2-8 in order (download through figures)")
    ra.add_argument("--no-repro", action="store_true", help="skip the reproducibility check")
    ra.set_defaults(only=None, verify=False, dry_run=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "env":
        return _env()
    if args.command == "download":
        return _download(args)
    if args.command == "clean":
        return _clean(args)
    if args.command == "eda":
        return _eda(args)
    if args.command == "features":
        return _features(args)
    if args.command == "model":
        return _model(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "figures":
        return _figures(args)
    if args.command == "run-all":
        return _run_all(args)
    if args.command == "onetry":
        return _onetry(args)
    raise AssertionError(f"unhandled command {args.command}")  # argparse prevents this


if __name__ == "__main__":
    raise SystemExit(main())
