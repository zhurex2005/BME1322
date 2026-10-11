#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py — unified entry point for generalized Gomoku (BME1322 Assignment 1)

    python main.py                      # launch GUI (Tkinter); human vs AI by default
    python main.py --mode eve           # GUI, AI vs AI
    python main.py --cli --mode pvp     # use the CLI (arguments go to main_cli.py)

See gui.py / main_cli.py for the meaning of every GUI and CLI parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

USAGE = """\
Generalized Gomoku BME1322 Assignment 1
    python main.py [--cli] [options...]

Common options (may appear before or after --cli):
    --mode {pvp,pve,eve}   human vs human / human vs AI / AI vs AI (default pve)
    --board N              board side length N (default 15)
    --win K                stones in a row K (default 5)
    --time T               time limit per AI move, seconds (default 5.0)
    --ai PATH              AI file for black (or the AI in human vs AI mode)
    --ai2 PATH             AI file for white in AI vs AI mode

Examples:
    python main.py --board 9 --win 4 --mode pve
    python main.py --cli --mode eve --ai gomoku_ai.py --ai2 random_ai.py --quiet
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--cli" in argv:
        argv.remove("--cli")
        import main_cli
        return main_cli.main(argv)

    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    try:
        import gui
    except ImportError as exc:                 # fall back to the CLI without Tkinter
        print(f"Cannot load the GUI ({exc}), switching to CLI mode.\n", file=sys.stderr)
        import main_cli
        return main_cli.main(["--mode", "pvp"] + argv)
    return gui.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
