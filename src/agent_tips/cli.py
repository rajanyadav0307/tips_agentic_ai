"""Command-line interface: ``agent-tips <step>``.

Each subcommand runs one step of the locked data science procedure.
Steps that are not built yet exit with code 2 and say which phase adds them.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

from agent_tips import __version__
from agent_tips.config import load_config
from agent_tips.utils.device import select_device, torch_available

# subcommand -> (procedure step, plan phase that implements it)
STEPS: dict[str, tuple[str, str]] = {
    "download": ("Step 2: collect data", "A"),
    "clean": ("Step 3: clean and prepare", "B"),
    "eda": ("Step 4: explore", "B"),
    "features": ("Step 5: create features", "C"),
    "model": ("Step 6: model and analyze", "D"),
    "evaluate": ("Step 7: evaluate", "D"),
    "figures": ("Step 8: communicate", "E"),
    "run-all": ("Steps 2-8 in order", "E"),
}


def _env() -> int:
    cfg = load_config()
    print(f"agent-tips {__version__}")
    print(f"python     {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"platform   {platform.system()} {platform.machine()}")
    print(f"root       {cfg.root}")
    print(f"torch      {'installed' if torch_available() else 'not installed (CPU only)'}")
    print(f"device     {select_device(cfg.raw.get('compute', {}).get('device', 'auto'))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="agent-tips", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="show platform, paths and compute device")
    for name, (step, _) in STEPS.items():
        sub.add_parser(name, help=step)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "env":
        return _env()
    step, phase = STEPS[args.command]
    print(f"'{args.command}' ({step}) is not implemented yet; it is added in phase {phase}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
