#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main_cli.py — generalized Gomoku CLI (human vs human / human vs AI / AI vs AI)

Usage examples
--------
    # human vs human (15x15, five in a row)
    python main_cli.py --mode pvp

    # human vs AI (the player takes black and moves first; the AI has 5 seconds)
    python main_cli.py --mode pve --human-color black --time 5

    # AI vs AI: your own AI against a random AI (9x9, four in a row)
    python main_cli.py --mode eve --board 9 --win 4 --ai gomoku_ai.py --ai2 random_ai.py

Interactive commands (entered when it is a human's turn)
--------------------------
    <row> <col> / <row>,<col>   place a stone (0-based; (0,0) is top-left)
    board                       reprint the board
    undo                        undo (human vs AI: one move for each side)
    help                        show help
    quit                        quit
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:              # allow running via python main_cli.py
    sys.path.insert(0, str(HERE))

from gomoku_game import (  # noqa: E402
    BLACK, EMPTY, WHITE, GomokuGame, PLAYER_NAME, opponent,
)

DEFAULT_AI = "gomoku_ai.py"
DEFAULT_AI2 = "random_ai.py"


# ------------------------------------------------------------------ AI loading

def load_ai(path, player_id: int, board_size: int, win_length: int):
    """Load an AI file using the shared protocol; return (instance, display name)."""
    path = Path(path)
    if not path.is_absolute():
        candidate = HERE / path
        path = candidate if candidate.is_file() else Path.cwd() / path
    if not path.is_file():
        raise FileNotFoundError(f"AI file not found: {path}")
    module_name = f"_gomoku_cli_ai_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "GomokuAI"):
        raise AttributeError(f"GomokuAI class not found in {path}")
    name = getattr(module, "NAME", None) or path.parent.name or path.stem
    return module.GomokuAI(player_id, board_size, win_length), str(name)


# ------------------------------------------------------------------ Display

def print_board(game: GomokuGame, title: str = "") -> None:
    if title:
        print(f"\n--- {title} ---")
    print(game.ascii_board())
    print(f"    {game.result_text()}")


def announce_move(name: str, player: int, move, elapsed=None) -> None:
    global game_move_no
    mark = "X" if player == BLACK else "O"
    extra = f"  ({elapsed:.2f}s)" if elapsed is not None else ""
    print(f"[{game_move_no:>3}] {name}({mark}) -> ({move[0]}, {move[1]}){extra}")
    game_move_no += 1


game_move_no = 0


# ------------------------------------------------------------------ Human input

def parse_move(text: str, n: int):
    """Parse coordinates typed by a human; return (row, col), or None if unparsable."""
    cleaned = text.replace(",", " ").replace("，", " ").strip()
    parts = cleaned.split()
    if len(parts) != 2:
        return None
    try:
        row, col = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= row < n and 0 <= col < n):
        return None
    return row, col


HELP_TEXT = """\
Enter coordinates to place a stone:  <row> <col>   or   <row>,<col>   (0-based)
Other commands:  board redraw | undo undo move | help help | quit quit
"""


def human_turn(game: GomokuGame, name: str, undo_steps: int = 1) -> str:
    """Wait for the human to place a stone. Returns "ok" / "quit"."""
    while True:
        try:
            raw = input(f"{name}({'X' if game.current_player == BLACK else 'O'})"
                        f" place a stone (row col): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "quit"
        low = raw.lower()
        if low in ("quit", "exit", "q"):
            return "quit"
        if low in ("help", "h", "?"):
            print(HELP_TEXT)
            continue
        if low in ("board", "b"):
            print_board(game)
            continue
        if low in ("undo", "u"):
            if game.undo(undo_steps):
                print_board(game, "Move undone")
            else:
                print("There are no moves left to undo.")
            continue
        move = parse_move(raw, game.board_size)
        if move is None:
            print("Unrecognized input; enter two integers from 0 to "
                  f"{game.board_size - 1}, for example  7 7")
            continue
        if not game.is_legal(*move):
            print("Cannot place a stone there (out of bounds or already "
                  "occupied); please try another position.")
            continue
        game.place(*move)
        return "ok"


# ------------------------------------------------------------------ AI moves

def ai_turn(game: GomokuGame, ai, name: str, time_limit: float) -> str:
    """Ask the AI to make one move. Returns "ok" / "failed"."""
    board_copy = [row[:] for row in game.board]
    last_opponent_move = game.last_move if game.history else None
    start = time.perf_counter()
    try:
        move = ai.get_move(board_copy, last_opponent_move, time_limit)
    except Exception as exc:                       # noqa: BLE001
        print(f"!! {name} raised an exception: {exc!r}")
        return "failed"
    elapsed = time.perf_counter() - start
    if elapsed > time_limit:
        print(f"!! {name} timed out ({elapsed:.2f}s > {time_limit:.2f}s)")
        return "failed"
    if not isinstance(move, (tuple, list)) or len(move) != 2 \
            or not all(isinstance(v, int) and not isinstance(v, bool)
                       for v in move):
        print(f"!! {name} returned an illegal move: {move!r}")
        return "failed"
    if not game.is_legal(int(move[0]), int(move[1])):
        print(f"!! {name} returned an illegal move: {tuple(move)}")
        return "failed"
    game.place(int(move[0]), int(move[1]))
    announce_move(name, game.history[-1][2], (int(move[0]), int(move[1])),
                  elapsed)
    return "ok"


# ------------------------------------------------------------------ Match flow

def run_pvp(args) -> None:
    game = GomokuGame(args.board, args.win)
    names = {BLACK: args.name_black or "Player 1", WHITE: args.name_white or "Player 2"}
    print_board(game, f"Human vs human {args.board}x{args.board}, win {args.win}")
    while not game.game_over:
        player = game.current_player
        if human_turn(game, names[player]) == "quit":
            print("Match exited.")
            return
        print_board(game)
    print("\nFinal result:", game.result_text())


def run_pve(args) -> None:
    human_color = WHITE if args.human_color == "white" else BLACK
    ai_color = opponent(human_color)
    game = GomokuGame(args.board, args.win)
    ai, ai_name = load_ai(args.ai, ai_color, args.board, args.win)
    human_name = args.name_black if human_color == BLACK else args.name_white
    human_name = human_name or "Player"
    print_board(game, f"Human vs AI {args.board}x{args.board}, win {args.win} | "
                      "you play "
                      f"{'black (first)' if human_color == BLACK else 'white (second)'}"
                      f", AI = {ai_name}")
    while not game.game_over:
        if game.current_player == human_color:
            if human_turn(game, human_name,
                          undo_steps=2 if game.history else 1) == "quit":
                print("Match exited.")
                return
        else:
            print(f"... {ai_name} is thinking (limit {args.time}s)")
            if ai_turn(game, ai, ai_name, args.time) == "failed":
                print(f"{ai_name} loses; you win.")
                return
        print_board(game)
    print("\nFinal result:", game.result_text())


def run_eve(args) -> None:
    game = GomokuGame(args.board, args.win)
    black_ai, black_name = load_ai(args.ai, BLACK, args.board, args.win)
    white_ai, white_name = load_ai(args.ai2, WHITE, args.board, args.win)
    ais = {BLACK: (black_ai, black_name), WHITE: (white_ai, white_name)}
    print_board(game, f"AI vs AI {args.board}x{args.board}, win {args.win} | "
                      f"{black_name}(X) vs {white_name}(O)")
    while not game.game_over:
        player = game.current_player
        ai, name = ais[player]
        if ai_turn(game, ai, name, args.time) == "failed":
            print(f"{name} loses.")
            return
        if not args.quiet:
            print_board(game)
        if args.delay:
            time.sleep(args.delay)
    print("\nFinal result:", game.result_text())


# ------------------------------------------------------------------ Entry point

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generalized Gomoku command-line game "
                    "(human vs human / human vs AI / AI vs AI)")
    p.add_argument("--mode", choices=("pvp", "pve", "eve"), default="pve",
                   help="game mode: pvp=human vs human, pve=human vs AI, "
                        "eve=AI vs AI (default pve)")
    p.add_argument("--board", type=int, default=15,
                   help="board side length N (default 15)")
    p.add_argument("--win", type=int, default=5, help="stones in a row K (default 5)")
    p.add_argument("--time", type=float, default=5.0,
                   help="time limit per AI move in seconds (default 5.0)")
    p.add_argument("--ai", default=DEFAULT_AI,
                   help=f"AI file (default {DEFAULT_AI})")
    p.add_argument("--ai2", default=DEFAULT_AI2,
                   help=f"second AI used in AI vs AI mode (default {DEFAULT_AI2})")
    p.add_argument("--human-color", choices=("black", "white"), default="black",
                   help="human stone color in human vs AI mode (default black)")
    p.add_argument("--name-black", default=None, help="display name for black")
    p.add_argument("--name-white", default=None, help="display name for white")
    p.add_argument("--delay", type=float, default=0.0,
                   help="pause in seconds between moves in AI vs AI mode")
    p.add_argument("--quiet", action="store_true",
                   help="do not print the board after every move in AI vs AI mode")
    return p


def main(argv=None) -> int:
    global game_move_no
    args = build_parser().parse_args(argv)
    if not (3 <= args.board):
        print("--board must be >= 3", file=sys.stderr)
        return 2
    if not (3 <= args.win <= args.board):
        print("3 <= --win <= --board is required", file=sys.stderr)
        return 2
    if args.time <= 0:
        print("--time must be > 0", file=sys.stderr)
        return 2

    game_move_no = 1
    print(f"Generalized Gomoku BME1322 Assignment 1 -- mode {args.mode}, "
          f"board {args.board}x{args.board}, win {args.win}")
    try:
        if args.mode == "pvp":
            run_pvp(args)
        elif args.mode == "pve":
            run_pve(args)
        else:
            run_eve(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
