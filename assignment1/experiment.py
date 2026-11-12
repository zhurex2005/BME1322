#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""experiment.py — BME1322 Assignment 1 experiment script

Drives two protocol-compliant AIs directly through the built-in match engine
(gomoku_game.GomokuGame), tallying per-game win rate, average thinking time per
move, average search depth/node count, and average game length. Results are
printed to the screen and written to results/*.json for report.md to cite.

Subcommands
-----------
    vs-random    our AI against the random AI (optionally several (N,K) configs)
    selfplay     our AI playing itself (some games as black, some as white)
    depth        compare different search depth caps under a fixed time limit
    arena-check  call the course referee arena.py as a protocol compliance smoke test

Examples
--------
    python experiment.py vs-random --board 15 --win 5 --time 1.0 --games 40
    python experiment.py selfplay  --board 15 --win 5 --time 0.5 --games 8
    python experiment.py depth     --board 9  --win 4 --time 1.0 --games 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import locale
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gomoku_game import BLACK, WHITE, GomokuGame  # noqa: E402

RESULTS_DIR = HERE / "results"


# ------------------------------------------------------------------ helpers

def load_ai_class(path: Path):
    """Load an AI class per the unified protocol.

    Same approach as the course referee arena.py.
    """
    path = Path(path)
    if not path.is_absolute():
        path = HERE / path
    spec = importlib.util.spec_from_file_location(
        f"_exp_ai_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "GomokuAI"):
        raise AttributeError(f"no GomokuAI class in {path}")
    name = getattr(module, "NAME", None) or path.stem
    return module.GomokuAI, str(name)


def make_factory(ai_class, **overrides):
    """Return a (player, n, k) -> AI instance factory.

    overrides serves comparison runs.
    """
    def factory(player, n, k):
        ai = ai_class(player, n, k)
        for key, value in overrides.items():
            setattr(ai, key, value)
        return ai
    return factory


def play_game(black_factory, white_factory, n, k, time_limit):
    """Play one game and return raw statistics from the absolute color viewpoint."""
    game = GomokuGame(n, k)
    ais = {BLACK: black_factory(BLACK, n, k),
           WHITE: white_factory(WHITE, n, k)}
    stats = {BLACK: {"time": [], "nodes": [], "depth": []},
             WHITE: {"time": [], "nodes": [], "depth": []}}
    winner, reason, aborted = 0, "draw", False

    while not game.game_over:
        player = game.current_player
        ai = ais[player]
        board_copy = [row[:] for row in game.board]
        last = game.last_move if game.history else None
        start = time.perf_counter()
        try:
            move = ai.get_move(board_copy, last, time_limit)
        except BaseException as exc:                   # noqa: BLE001
            winner, reason, aborted = 3 - player, f"exception {exc!r}", True
            break
        elapsed = time.perf_counter() - start
        stats[player]["time"].append(elapsed)
        info = getattr(ai, "last_search", None)
        if isinstance(info, dict):
            stats[player]["nodes"].append(info.get("nodes", 0))
            stats[player]["depth"].append(info.get("depth", 0))
        if elapsed > time_limit:
            winner, reason, aborted = 3 - player, f"timeout {elapsed:.2f}s", True
            break
        if not isinstance(move, (tuple, list)) or len(move) != 2 \
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           for v in move):
            winner, reason, aborted = 3 - player, f"illegal {move!r}", True
            break
        if not game.is_legal(int(move[0]), int(move[1])):
            winner, reason, aborted = 3 - player, f"illegal {tuple(move)}", True
            break
        game.place(int(move[0]), int(move[1]))

    if not aborted:
        winner = game.winner
        reason = "draw" if game.draw else f"{k}-in-a-row"
    return {"winner_abs": winner, "reason": reason, "moves": game.move_count,
            "stats": stats}


def to_a_view(raw, a_is_black):
    """Convert an absolute-color result to the "A viewpoint".

    winner: 1 = A, 2 = B, 0 = draw.
    """
    a_color = BLACK if a_is_black else WHITE
    b_color = 3 - a_color
    if raw["winner_abs"] == a_color:
        winner = 1
    elif raw["winner_abs"] == b_color:
        winner = 2
    else:
        winner = 0
    return {"winner": winner, "reason": raw["reason"], "moves": raw["moves"],
            "a_is_black": a_is_black,
            "stats_a": raw["stats"][a_color], "stats_b": raw["stats"][b_color]}


def summarize(label, name_a, name_b, games):
    stat = {"label": label, "name_a": name_a, "name_b": name_b,
            "games": len(games), "win_a": 0, "win_b": 0, "draw": 0,
            "win_a_as_black": 0, "win_a_as_white": 0, "lengths": [],
            "time_a": [], "time_b": [], "nodes_a": [], "nodes_b": [],
            "depth_a": [], "depth_b": [], "reasons": []}
    for g in games:
        stat["win_a"] += 1 if g["winner"] == 1 else 0
        stat["win_b"] += 1 if g["winner"] == 2 else 0
        stat["draw"] += 1 if g["winner"] == 0 else 0
        if g["winner"] == 1:
            stat["win_a_as_black" if g["a_is_black"] else "win_a_as_white"] += 1
        stat["lengths"].append(g["moves"])
        stat["time_a"] += g["stats_a"]["time"]
        stat["time_b"] += g["stats_b"]["time"]
        stat["nodes_a"] += g["stats_a"]["nodes"]
        stat["nodes_b"] += g["stats_b"]["nodes"]
        stat["depth_a"] += g["stats_a"]["depth"]
        stat["depth_b"] += g["stats_b"]["depth"]
        stat["reasons"].append(g["reason"])
    return stat


def fmt(stat) -> str:
    games = max(1, stat["games"])
    out = [f"[{stat['label']}] {stat['name_a']} vs {stat['name_b']}: "
           f"{stat['games']} games -> won {stat['win_a']} lost {stat['win_b']} "
           f"drew {stat['draw']}, win rate {stat['win_a'] / games:.1%}"
           f" (wins as black {stat['win_a_as_black']}, "
           f"wins as white {stat['win_a_as_white']})"]
    for tag, key in (("ours", "a"), ("opponent", "b")):
        times = stat[f"time_{key}"]
        nodes = stat[f"nodes_{key}"]
        depths = stat[f"depth_{key}"]
        if times:
            extra = ""
            if nodes:
                extra += f", avg nodes {statistics.mean(nodes):,.0f}"
            if depths:
                extra += f", avg depth {statistics.mean(depths):.1f}"
            out.append(f"    {tag}: avg per move {statistics.mean(times):.3f}s"
                       f" (max {max(times):.3f}s){extra}, "
                       f"total moves {len(times)}")
    if stat["lengths"]:
        out.append(f"    avg game length {statistics.mean(stat['lengths']):.1f}"
                   f" moves (min {min(stat['lengths'])}, "
                   f"max {max(stat['lengths'])}); "
                   f"end reason distribution: "
                   + ", ".join(f"{r} x{stat['reasons'].count(r)}"
                               for r in sorted(set(stat["reasons"]))))
    return "\n".join(out)


def run_matches(factory_a, factory_b, name_a, name_b, n, k, time_limit,
                games, seed, label):
    """A plays B for games games, swapping colors from game to game."""
    random.seed(seed)
    results = []
    for i in range(games):
        a_is_black = (i % 2 == 0)
        raw = play_game(factory_a, factory_b, n, k, time_limit) if a_is_black \
            else play_game(factory_b, factory_a, n, k, time_limit)
        view = to_a_view(raw, a_is_black)
        results.append(view)
        res = ("A wins" if view["winner"] == 1
               else ("B wins" if view["winner"] == 2 else "draw"))
        print(f"    game {i + 1:>3}/{games}: A as "
              f"{'black' if a_is_black else 'white'} -> {res}"
              f" [{view['reason']}], {view['moves']} moves", flush=True)
    stat = summarize(label, name_a, name_b, results)
    print(fmt(stat), flush=True)
    return stat


def save(record, out_name):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / out_name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"results written to {path}", flush=True)


# ------------------------------------------------------------------ subcommands

def cmd_vs_random(args):
    ai_class, ai_name = load_ai_class(args.ai)
    rnd_class, rnd_name = load_ai_class(args.ai2)
    configs = [(args.board, args.win)] if args.board else [(9, 4), (11, 5), (15, 5)]
    record = {"kind": "vs-random", "time_limit": args.time, "seed": args.seed,
              "games_per_config": args.games, "results": []}
    for n, k in configs:
        print(f"\n=== (N,K) = ({n},{k}), time limit {args.time}s per move, "
              f"{args.games} games (half as black, half as white) ===", flush=True)
        stat = run_matches(make_factory(ai_class), make_factory(rnd_class),
                           ai_name, rnd_name, n, k, args.time, args.games,
                           args.seed, f"vs-random {n}x{n} K={k}")
        stat["config"] = {"N": n, "K": k}
        record["results"].append(stat)
    save(record, f"vs_random_{int(time.time())}.json")
    return record


def cmd_selfplay(args):
    ai_class, ai_name = load_ai_class(args.ai)
    print(f"\n=== selfplay (N,K)=({args.board},{args.win}), time limit "
          f"{args.time}s per move, {args.games} games (half as black, half as "
          f"white); tie-breaking among equal-scoring moves is enabled so that "
          f"the games are not all identical ===", flush=True)
    a = make_factory(ai_class, randomize_ties=True)
    b = make_factory(ai_class, randomize_ties=True)
    stat = run_matches(a, b, f"{ai_name}-A", f"{ai_name}-B", args.board,
                       args.win, args.time, args.games, args.seed,
                       f"selfplay {args.board}x{args.board} K={args.win}")
    record = {"kind": "selfplay", "time_limit": args.time, "seed": args.seed,
              "config": {"N": args.board, "K": args.win}, "result": stat}
    save(record, f"selfplay_{int(time.time())}.json")
    return record


def cmd_depth(args):
    ai_class, ai_name = load_ai_class(args.ai)
    rnd_class, rnd_name = load_ai_class(args.ai2)
    record = {"kind": "depth", "time_limit": args.time, "seed": args.seed,
              "config": {"N": args.board, "K": args.win}, "results": []}
    for cap in args.caps:
        print(f"\n=== depth cap max_depth_cap={cap}, time limit {args.time}s per "
              f"move, {args.games} games ===", flush=True)
        stat = run_matches(make_factory(ai_class, max_depth_cap=cap),
                           make_factory(rnd_class),
                           f"{ai_name}(depth<={cap})", rnd_name,
                           args.board, args.win, args.time, args.games,
                           args.seed, f"depth<={cap} vs random")
        stat["max_depth_cap"] = cap
        record["results"].append(stat)
    save(record, f"depth_{int(time.time())}.json")
    return record


def cmd_timing(args):
    """Time-limit compliance stress test.

    Selfplay under various time_limits, tracking the slowest move.
    """
    ai_class, ai_name = load_ai_class(args.ai)
    a = make_factory(ai_class, randomize_ties=True)
    b = make_factory(ai_class, randomize_ties=True)
    record = {"kind": "timing", "config": {"N": args.board, "K": args.win},
              "results": []}
    for tl in args.times:
        times = []
        violations = 0
        for i in range(args.games):
            random.seed(args.seed + i)
            raw = play_game(a, b, args.board, args.win, tl)
            for color in (BLACK, WHITE):
                times += raw["stats"][color]["time"]
        # violation = a single move's thinking time exceeded the referee's limit
        violations = sum(1 for t in times if t > tl)
        worst = max(times) if times else 0.0
        row = {"time_limit": tl, "moves": len(times),
               "max_time": worst, "mean_time": statistics.mean(times) if times else 0.0,
               "violations": violations,
               "margin": tl - worst}
        record["results"].append(row)
        print(f"time_limit={tl:>5}s  moves {row['moves']:>4}  "
              f"slowest move {worst:.3f}s  mean {row['mean_time']:.3f}s  "
              f"violations {violations}  margin {row['margin']:.3f}s", flush=True)
    save(record, f"timing_{int(time.time())}.json")
    return record


def cmd_duel(args):
    """Two differently configured AIs play head to head.

    For example depth<=1 vs depth<=10, a sharper test than vs-random.
    """
    ai_class, ai_name = load_ai_class(args.ai)
    a = make_factory(ai_class, max_depth_cap=args.cap_a)
    b = make_factory(ai_class, max_depth_cap=args.cap_b)
    print(f"\n=== duel (N,K)=({args.board},{args.win}), time limit {args.time}s, "
          f"{args.games} games (half as black, half as white) ===", flush=True)
    stat = run_matches(a, b, f"{ai_name}(depth<={args.cap_a})",
                       f"{ai_name}(depth<={args.cap_b})", args.board, args.win,
                       args.time, args.games, args.seed,
                       f"duel depth<={args.cap_a} vs depth<={args.cap_b}")
    record = {"kind": "duel", "time_limit": args.time, "seed": args.seed,
              "config": {"N": args.board, "K": args.win},
              "cap_a": args.cap_a, "cap_b": args.cap_b, "result": stat}
    save(record, f"duel_{int(time.time())}.json")
    return record


def cmd_robustness(args):
    """Protocol robustness self-check.

    Read-only input, return type, illegal/extreme time limits, mismatched
    sizes, a nearly full board, and so on.
    """
    ai_class, ai_name = load_ai_class(args.ai)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  -- {detail}" if detail else ""), flush=True)

    n, k = args.board, args.win

    # T1 must not modify the board handed in by the referee
    ai = ai_class(1, n, k)
    game = GomokuGame(n, k)
    random.seed(1)
    for _ in range(6):
        mv = random.choice(game.legal_moves())
        game.place(*mv)
    before = [row[:] for row in game.board]
    mv = ai.get_move(game.board, game.last_move, 0.3)
    check("does not modify the board handed in by the referee",
          game.board == before, f"returned {tuple(mv)}")

    # T2 the return type must be (int, int)
    check("returns an (int, int) tuple",
          isinstance(mv, tuple) and len(mv) == 2
          and all(isinstance(v, int) and not isinstance(v, bool) for v in mv),
          repr(mv))

    # T3 an empty board plus a tiny time limit stays legal and does not time out
    for tl in (0.01, 0.05, 0.2):
        ai2 = ai_class(2, n, k)
        empty = [[0] * n for _ in range(n)]
        t0 = time.perf_counter()
        m = ai2.get_move(empty, None, tl)
        el = time.perf_counter() - t0
        legal = isinstance(m, tuple) and 0 <= m[0] < n and 0 <= m[1] < n \
            and empty[m[0]][m[1]] == 0
        check(f"time limit {tl}s: move on an empty board is legal and on time",
              legal and el <= max(tl, 0.5), f"move={tuple(m)}, {el:.3f}s")

    # T4 when only one empty square remains it must be filled
    ai3 = ai_class(1, 4, 3)
    board = [[1 if (r * 4 + c) % 2 == 0 else 2 for c in range(4)]
             for r in range(4)]
    board[1][1] = 0
    m = ai3.get_move(board, (0, 0), 0.5)
    check("fills the only empty square when one remains", tuple(m) == (1, 1),
          repr(tuple(m)))

    # T5 adapt when the size handed in by the referee differs from the constructor
    ai4 = ai_class(1, 15, 5)
    b9 = [[0] * 9 for _ in range(9)]
    b9[4][4] = 2
    m = ai4.get_move(b9, (4, 4), 0.3)
    check("adapts when the board size differs from the constructor arguments",
          isinstance(m, tuple) and 0 <= m[0] < 9 and 0 <= m[1] < 9,
          f"built with 15, received 9 -> {tuple(m)}")

    # T6 the extreme configuration K = N
    ai5 = ai_class(1, 5, 5)
    b5 = [[0] * 5 for _ in range(5)]
    m = ai5.get_move(b5, None, 0.3)
    check("the extreme configuration K = N is usable",
          isinstance(m, tuple) and 0 <= m[0] < 5 and 0 <= m[1] < 5, repr(tuple(m)))

    # T7 100 consecutive calls: all legal, all returned on time
    ai6 = ai_class(1, n, k)
    game = GomokuGame(n, k)
    random.seed(7)
    ok, worst = True, 0.0
    tl = 0.2
    for _ in range(100):
        if game.game_over:
            break
        t0 = time.perf_counter()
        m = ai6.get_move(game.board, game.last_move, tl)
        el = time.perf_counter() - t0
        worst = max(worst, el)
        if not (isinstance(m, tuple) and len(m) == 2
                and all(isinstance(v, int) for v in m)
                and game.is_legal(int(m[0]), int(m[1]))):
            ok = False
            break
        game.place(int(m[0]), int(m[1]))
    check("100 calls in a row during a game are all legal and on time",
          ok and worst <= tl, f"slowest {worst:.3f}s / limit {tl}s")

    # T8 the winning point: it must win outright when it can line up K stones
    ai7 = ai_class(1, n, k)
    board = [[0] * n for _ in range(n)]
    for c in range(k - 1):
        board[7][3 + c] = 1
    board[8][8] = 2
    m = ai7.get_move(board, (8, 8), 0.5)
    check("wins outright when it can line up K stones",
          tuple(m) in {(7, 2), (7, 3 + k - 1)}, f"move={tuple(m)}")

    passed = sum(1 for c in checks if c["ok"])
    print(f"\nrobustness self-check: {passed}/{len(checks)} checks passed",
          flush=True)
    record = {"kind": "robustness", "ai": ai_name, "passed": passed,
              "total": len(checks), "checks": checks}
    save(record, f"robustness_{int(time.time())}.json")
    return record


def _decode(raw: bytes) -> str:
    """Decode a subprocess' output, tolerating non-UTF-8 consoles.

    arena.py is the course judge and prints Chinese; on a Windows console with
    a legacy code page (e.g. cp936) its stdout is not UTF-8, so trying UTF-8
    first and falling back to the preferred encoding keeps the log readable.
    """
    for enc in ("utf-8", locale.getpreferredencoding(False), "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _safe_print(text: str, stream=None) -> None:
    """Print text even when the console encoding cannot represent it."""
    stream = stream if stream is not None else sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"),
              file=stream)


def cmd_arena_check(args):
    arena = Path(args.arena)
    cmd = [sys.executable, str(arena), str(HERE / "gomoku_ai.py"),
           str(HERE / "random_ai.py"), "--board", str(args.board),
           "--win", str(args.win), "--time", str(args.time),
           "--games", str(args.games), "--quiet"]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(arena.parent), capture_output=True)
    _safe_print(_decode(proc.stdout))
    if proc.returncode != 0:
        _safe_print(_decode(proc.stderr), stream=sys.stderr)
    return proc.returncode


# ------------------------------------------------------------------ entry point

def build_parser():
    p = argparse.ArgumentParser(description="BME1322 Assignment 1 experiment "
                                            "script")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, board_default=None):
        sp.add_argument("--board", type=int, default=board_default,
                        help="board side length N (vs-random runs several "
                             "configurations when omitted)")
        sp.add_argument("--win", type=int, default=5, help="run length K")
        sp.add_argument("--time", type=float, default=1.0,
                        help="time limit per move (seconds)")
        sp.add_argument("--games", type=int, default=20, help="number of games")
        sp.add_argument("--seed", type=int, default=20261011, help="random seed")
        sp.add_argument("--ai", default="gomoku_ai.py", help="our AI file")
        return sp

    sp = common(sub.add_parser("vs-random"))
    sp.add_argument("--ai2", default="random_ai.py", help="opponent AI file")
    sp.set_defaults(func=cmd_vs_random)

    sp = common(sub.add_parser("selfplay"), board_default=15)
    sp.set_defaults(func=cmd_selfplay)

    sp = common(sub.add_parser("depth"), board_default=9)
    sp.add_argument("--ai2", default="random_ai.py")
    sp.add_argument("--caps", type=int, nargs="+", default=[1, 2, 4, 10],
                    help="max_depth_cap values to compare")
    sp.set_defaults(func=cmd_depth)

    sp = common(sub.add_parser("timing"), board_default=15)
    sp.add_argument("--times", type=float, nargs="+",
                    default=[0.1, 0.5, 1.0, 5.0],
                    help="time limits per move (seconds) to test")
    sp.set_defaults(func=cmd_timing)

    sp = common(sub.add_parser("robustness"), board_default=15)
    sp.set_defaults(func=cmd_robustness)

    sp = common(sub.add_parser("duel"), board_default=9)
    sp.add_argument("--cap-a", type=int, default=1, dest="cap_a")
    sp.add_argument("--cap-b", type=int, default=10, dest="cap_b")
    sp.set_defaults(func=cmd_duel)

    sp = sub.add_parser("arena-check")
    sp.add_argument("--arena",
                    default=str(HERE.parent / "assignment-1-code" / "arena.py"))
    sp.add_argument("--board", type=int, default=15)
    sp.add_argument("--win", type=int, default=5)
    sp.add_argument("--time", type=float, default=1.0)
    sp.add_argument("--games", type=int, default=2)
    sp.set_defaults(func=cmd_arena_check)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    board = getattr(args, "board", None)
    if board is not None and args.cmd != "arena-check":
        win = getattr(args, "win", 5)
        if board < 3 or not (3 <= win <= board):
            print("must satisfy 3 <= K <= N", file=sys.stderr)
            return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
