#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gomoku_ai.py — BME1322 Assignment 1: generalized Gomoku AI (Gomoku AI Protocol v1)

Algorithm
---------
* Adversarial search: Negamax (an equivalent form of MiniMax) + Alpha-Beta pruning
* Iterative Deepening: deepen one level at a time from depth 1, and keep the result
  of the last level completed within the time limit
* Evaluation: a "length-K window pattern" score adapted to the parameter K
  (run length / open patterns / blocked patterns)
* Candidate moves: only empty cells near existing stones (Chebyshev distance
  <= 2), sorted by incremental pattern gain
* Transposition Table: Zobrist hashing, caching positions already searched
* Time limit: wall-clock budget = time_limit * safety; on timeout the current
  level is aborted by an exception and the previous level's result is returned

Protocol requirements (lecture notes, section 2)
-----------------------------------------------
    GomokuAI(player_id, board_size, win_length)
    get_move(board, last_opponent_move, time_limit) -> (row, col)
    board: an N x N two-dimensional list, 0 = empty, 1 = black (first player),
    2 = white; coordinates start at 0

This file is kept "fully self-contained" (it imports only the standard library),
because the course referee program arena.py loads this file separately with
importlib, so the directory never enters sys.path and this file therefore cannot
import other modules from the same directory.

N and K come entirely from the constructor arguments; there is no hard-coded
board size or "five in a row" assumption anywhere in the code.
"""

from __future__ import annotations

import random
import time

#: AI name shown in the referee output (optional variable, notes section 2.2)
NAME = "AlphaBeta-Gomoku"

EMPTY, BLACK, WHITE = 0, 1, 2

#: A "certain win" score, far larger than any ordinary pattern score
WIN_SCORE = 10 ** 12
INF = float("inf")

#: Four scan directions (only half of them, to avoid duplicates)
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class _Timeout(Exception):
    """Internal signal: this level exceeded the time limit; abandon it and fall
    back to the previous level's result."""


class GomokuAI:
    """Alpha-Beta Gomoku AI for the unified match protocol v1."""

    # ---- The following are tunable parameters; experiment scripts may
    # ---- override them after construction for comparative experiments ----
    candidate_limit = 12      # maximum candidate moves expanded per level
    time_safety = 0.75        # fraction of the time limit actually used
    max_depth_cap = 10        # depth cap for iterative deepening
    block_weight = 1.0        # weight of "blocking the opponent" in move ordering
    defense_weight = 1.05     # opponent's pattern weight in the evaluation
                              # (slightly defense-oriented)
    neighbor_radius = 2       # radius used for candidate move generation
    randomize_ties = False    # randomly break ties between equally scored moves
                              # (used only in self-play experiments)

    def __init__(self, player_id: int, board_size: int, win_length: int):
        """
        :param player_id: this game's color, 1 = black (first), 2 = white (second)
        :param board_size: board side length N
        :param win_length: number of stones in a row K
        """
        self.player_id = int(player_id)
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        # Tolerate other values the referee might pass in
        if self.board_size < 3:
            self.board_size = 3
        if not (1 <= self.win_length <= self.board_size):
            self.win_length = min(5, self.board_size)

        self.n = self.board_size
        self.n2 = self.n * self.n
        self.k = self.win_length
        self.opponent_id = 3 - self.player_id

        # On small boards / with short runs the branching factor is itself small,
        # so more candidates can be kept
        if self.k <= 3:
            self.candidate_limit = max(self.candidate_limit, 14)
        if self.n <= 8:
            self.candidate_limit = max(self.candidate_limit, 16)

        self._build_tables()
        self._rng = random.Random()
        self._reset_state()

        # Statistics (the report's "average thinking time per move / number of
        # search nodes" is derived from these)
        self.stats = {"moves": 0, "nodes": 0, "time": 0.0, "depth": 0}
        self.last_search = {"depth": 0, "nodes": 0, "time": 0.0, "value": 0}

    # ================================================================ build tables

    def _build_tables(self) -> None:
        """Precompute windows, window-to-cell indices, neighborhoods and so on."""
        n, n2, k = self.n, self.n2, self.k

        # 1) All straight windows of length K (horizontal / vertical / diagonals)
        windows: list[tuple[int, ...]] = []
        for r in range(n):
            for c in range(n):
                for dr, dc in DIRECTIONS:
                    rr, cc = r + (k - 1) * dr, c + (k - 1) * dc
                    if 0 <= rr < n and 0 <= cc < n:
                        windows.append(tuple((r + t * dr) * n + (c + t * dc)
                                             for t in range(k)))
        self._windows = windows

        cell_windows: list[list[int]] = [[] for _ in range(n2)]
        for wi, w in enumerate(windows):
            for idx in w:
                cell_windows[idx].append(wi)
        self._cell_windows = cell_windows

        # 2) Each cell's neighborhood (Chebyshev distance 1..neighbor_radius)
        rad = self.neighbor_radius
        neighbors: list[list[int]] = [[] for _ in range(n2)]
        for r in range(n):
            for c in range(n):
                idx = r * n + c
                box = []
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < n and 0 <= cc < n:
                            box.append(rr * n + cc)
                neighbors[idx] = box
        self._neighbor_cells = neighbors

        # 3) Window score table: what a clean window (stones of only one side)
        #    holding c stones is worth. The closer to victory the higher, and
        #    adapted to K:
        #    c = K      -> K stones already connected (certain win)
        #    c = K-1    -> one stone short (open four / four blocked on one side)
        #    c = K-2    -> open three, and so on
        table = [0] * (k + 1)
        for c in range(1, k):
            gap = k - c
            if gap == 1:
                table[c] = 10 ** 7
            elif gap == 2:
                table[c] = 10 ** 4
            elif gap == 3:
                table[c] = 10 ** 2
            elif gap == 4:
                table[c] = 10
            else:
                table[c] = 1
        table[k] = WIN_SCORE
        self._table = table

        # 4) Zobrist hashing (fixed seed, to guarantee reproducibility)
        rng = random.Random(20260928)
        self._zobrist = [rng.getrandbits(64) for _ in range(2 * n2)]
        self._side_key = rng.getrandbits(64)

    def _reset_state(self) -> None:
        self._board = [EMPTY] * self.n2          # flat board, used by the search
        self._scores = [0, 0, 0]                 # both sides' pattern totals, by color
        self._near = [0] * self.n2               # stones within each cell's radius
        self._undo_stack: list[tuple[int, int, int, int]] = []
        self._hash = 0
        self._tt: dict[int, tuple[int, int, float]] = {}
        self._nodes = 0
        self._deadline = 0.0
        self._safe_board = [EMPTY] * self.n2     # real board copy, exception fallback
        self._time_limit = 5.0

    # ================================================================ sync with referee

    def _resize(self, new_n: int) -> None:
        """Adapt when the referee's board size differs from the constructor
        argument (the protocol requires adapting to N)."""
        new_n = int(new_n)
        if new_n < 3:
            new_n = 3
        self.board_size = self.n = new_n
        self.n2 = new_n * new_n
        if self.k > new_n:
            self.win_length = self.k = new_n
        self._build_tables()
        self._reset_state()

    def _rebuild(self, board) -> None:
        """Sync the referee's two-dimensional board into the internal flat one,
        and recompute scores / neighborhoods / hash."""
        if len(board) != self.n:
            self._resize(len(board))
        n, n2 = self.n, self.n2
        flat = self._board
        for r in range(n):
            row = board[r]
            base = r * n
            for c in range(n):
                v = row[c]
                flat[base + c] = v if v in (BLACK, WHITE) else EMPTY

        near = self._near
        for i in range(n2):
            near[i] = 0
        for i in range(n2):
            if flat[i]:
                for j in self._neighbor_cells[i]:
                    near[j] += 1

        scores = [0, 0, 0]
        table = self._table
        for w in self._windows:
            black = white = 0
            for j in w:
                v = flat[j]
                if v == BLACK:
                    black += 1
                elif v == WHITE:
                    white += 1
            if white == 0:
                scores[BLACK] += table[black]
            elif black == 0:
                scores[WHITE] += table[white]
        self._scores = scores

        h = 0
        zob = self._zobrist
        for i in range(n2):
            v = flat[i]
            if v:
                h ^= zob[(i << 1) | (v - 1)]
        self._hash = h
        self._undo_stack.clear()

    # ================================================================ place / undo

    def _place(self, idx: int, player: int) -> bool:
        """Place a stone of color player at idx, incrementally maintaining score /
        neighborhoods / hash. Returns whether this wins immediately."""
        flat = self._board
        table = self._table
        opp = 3 - player
        k = self.k
        flat[idx] = player
        d_me = 0
        d_opp = 0
        won = False
        for wi in self._cell_windows[idx]:
            w = self._windows[wi]
            mine = 0
            theirs = 0
            for j in w:
                v = flat[j]
                if v == player:
                    mine += 1
                elif v == opp:
                    theirs += 1
            if theirs == 0:
                d_me += table[mine] - table[mine - 1]
                if mine >= k:
                    won = True
            elif mine == 1:
                # this stone blocked the opponent's previously clean window, so
                # the opponent's score in that window drops to zero
                d_opp -= table[theirs]
        scores = self._scores
        scores[player] += d_me
        scores[opp] += d_opp
        near = self._near
        for j in self._neighbor_cells[idx]:
            near[j] += 1
        self._hash ^= self._zobrist[(idx << 1) | (player - 1)]
        self._undo_stack.append((idx, player, d_me, d_opp))
        return won

    def _remove(self) -> None:
        """Undo the most recent _place."""
        idx, player, d_me, d_opp = self._undo_stack.pop()
        self._board[idx] = EMPTY
        scores = self._scores
        scores[player] -= d_me
        scores[3 - player] -= d_opp
        near = self._near
        for j in self._neighbor_cells[idx]:
            near[j] -= 1
        self._hash ^= self._zobrist[(idx << 1) | (player - 1)]

    # ================================================================ eval / candidates

    def _evaluate(self, player: int) -> float:
        """Evaluation from player's view (opponent patterns weighted a little
        higher, i.e. tilted toward defense)."""
        scores = self._scores
        return scores[player] - scores[3 - player] * self.defense_weight

    def _candidates(self) -> list[int]:
        """Candidate empty cells: empty points with a stone within neighbor_radius."""
        flat = self._board
        near = self._near
        return [i for i in range(self.n2) if flat[i] == EMPTY and near[i] > 0]

    def _move_gain(self, idx: int, player: int):
        """Estimate the gain of playing at idx **without modifying persistent
        state** (used for move ordering).

        Returns (my score gain d_me, blocking gain d_opp, whether it wins now).
        """
        flat = self._board
        table = self._table
        opp = 3 - player
        k = self.k
        flat[idx] = player
        d_me = 0
        d_opp = 0
        won = False
        for wi in self._cell_windows[idx]:
            w = self._windows[wi]
            mine = 0
            theirs = 0
            for j in w:
                v = flat[j]
                if v == player:
                    mine += 1
                elif v == opp:
                    theirs += 1
            if theirs == 0:
                d_me += table[mine] - table[mine - 1]
                if mine >= k:
                    won = True
            elif mine == 1:
                d_opp += table[theirs]
        flat[idx] = EMPTY
        return d_me, d_opp, won

    def _ordered_moves(self, cands: list[int], player: int) -> list[int]:
        """Sort moves by (my gain + block_weight * blocking gain), descending."""
        bw = self.block_weight
        scored = []
        gain = self._move_gain
        rnd = self._rng.random if self.randomize_ties else None
        for idx in cands:
            d_me, d_opp, won = gain(idx, player)
            key = d_me + bw * d_opp
            if won:
                key += WIN_SCORE * 2          # a winning move is always ranked first
            # the third element only breaks ties randomly under randomize_ties
            # (it does not affect the order of moves with different scores)
            scored.append((key, rnd() if rnd else 0.0, idx))
        scored.sort(reverse=True)
        return [idx for _key, _jitter, idx in scored]

    # ================================================================ search

    def _negamax(self, depth: int, alpha: float, beta: float,
                 player: int, ply: int) -> float:
        self._nodes += 1
        # the time limit is checked at every node: a node costs milliseconds
        # while time.perf_counter() costs tens of nanoseconds, so the timeout
        # error stays on the scale of one node (measured < 20 ms), avoiding a
        # wall-clock loss.
        if time.perf_counter() > self._deadline:
            raise _Timeout

        if self._scores[player] >= WIN_SCORE:
            return WIN_SCORE - ply            # this side has K stones connected
        if depth <= 0:
            return self._evaluate(player)

        key = self._hash ^ (self._side_key if player == BLACK else 0)
        entry = self._tt.get(key)
        alpha0 = alpha
        if entry is not None:
            e_depth, e_flag, e_score = entry
            if e_depth >= depth:
                if e_flag == 0:
                    return e_score
                if e_flag == 1 and e_score >= beta:
                    return e_score
                if e_flag == 2 and e_score <= alpha:
                    return e_score

        cands = self._candidates()
        if not cands:
            return 0.0                        # board full, a draw
        ordered = self._ordered_moves(cands, player)
        if len(ordered) > self.candidate_limit:
            ordered = ordered[:self.candidate_limit]

        opp = 3 - player
        best = -INF
        for idx in ordered:
            if self._place(idx, player):
                score = WIN_SCORE - ply
            else:
                score = -self._negamax(depth - 1, -beta, -alpha, opp, ply + 1)
            self._remove()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        # transposition table: cache only non-winning/non-losing scores, to
        # avoid polluting entries with ply-dependent values
        if -WIN_SCORE / 2 < best < WIN_SCORE / 2:
            if len(self._tt) > 400_000:
                self._tt.clear()
            if best >= beta:
                flag = 1                      # lower bound
            elif best <= alpha0:
                flag = 2                      # upper bound
            else:
                flag = 0                      # exact value
            self._tt[key] = (depth, flag, best)
        return best

    def _root_search(self, depth: int, player: int):
        """Root search: returns (best score, best move); the move is None when no
        move is available."""
        cands = self._candidates()
        if not cands:
            return 0.0, None
        ordered = self._ordered_moves(cands, player)
        if len(ordered) > self.candidate_limit:
            ordered = ordered[:self.candidate_limit]

        opp = 3 - player
        alpha, beta = -INF, INF
        best_score = -INF
        best_move = ordered[0]
        for idx in ordered:
            if time.perf_counter() > self._deadline:
                raise _Timeout
            if self._place(idx, player):
                score = WIN_SCORE
            else:
                score = -self._negamax(depth - 1, -beta, -alpha, opp, 1)
            self._remove()
            if score > best_score:
                best_score = score
                best_move = idx
            if best_score > alpha:
                alpha = best_score
        return best_score, best_move

    # ================================================================ public interface

    def get_move(self, board, last_opponent_move=None, time_limit: float = 5.0):
        """Protocol interface: returns (row, col). Any exception degrades into a
        legal move, never a loss."""
        try:
            return self._get_move(board, last_opponent_move, time_limit)
        except BaseException:                 # noqa: BLE001 -- last-resort fallback
            return self._fallback_move(board)

    # ---------------------------------------------------------------- internal impl

    def _get_move(self, board, last_opponent_move, time_limit):
        start = time.perf_counter()
        try:
            tl = float(time_limit)
        except (TypeError, ValueError):
            tl = 5.0
        if tl <= 0:
            tl = 0.05
        self._time_limit = tl
        self._nodes = 0

        self._rebuild(board)
        self._safe_board = list(self._board)
        self._deadline = start + max(0.02, tl * self.time_safety)

        player = self.player_id
        cands = self._candidates()
        if not cands:
            # empty board (or only far-away empty points left): general
            # heuristic -- play near the center
            return self._fallback_move()

        ordered = self._ordered_moves(cands, player)

        # fast path 1: if K stones can be connected at once, win right away
        for idx in ordered:
            if self._move_gain(idx, player)[2]:
                return self._finish(idx, start, depth=0, value=WIN_SCORE)

        # fast path 2: iterative deepening, keeping the last level finished in time
        best_move = ordered[0]
        best_value = 0.0
        depth_done = 0
        cap = self.max_depth_cap
        empties = self.n2 - sum(1 for v in self._board if v)
        cap = max(1, min(cap, empties))
        for depth in range(1, cap + 1):
            if time.perf_counter() > self._deadline:
                break
            try:
                value, move = self._root_search(depth, player)
            except _Timeout:
                break
            if move is not None:
                best_move = move
                best_value = value
                depth_done = depth
            if value >= WIN_SCORE / 2:        # a forced win was already found
                break
            if value <= -WIN_SCORE / 2:       # every move loses, searching is wasted
                break

        elapsed = time.perf_counter() - start
        self.stats["moves"] += 1
        self.stats["nodes"] += self._nodes
        self.stats["time"] += elapsed
        self.stats["depth"] = depth_done
        self.last_search = {"depth": depth_done, "nodes": self._nodes,
                            "time": elapsed, "value": best_value}
        return self._to_rc(best_move)

    def _finish(self, idx: int, start: float, depth: int, value: float):
        """Record statistics and return the move coordinates."""
        elapsed = time.perf_counter() - start
        self.stats["moves"] += 1
        self.stats["nodes"] += self._nodes
        self.stats["time"] += elapsed
        self.stats["depth"] = depth
        self.last_search = {"depth": depth, "nodes": self._nodes,
                            "time": elapsed, "value": value}
        return self._to_rc(idx)

    def _fallback_move(self, board2d=None):
        """Last line of defense: return a certainly legal move (preferring cells
        near the board center)."""
        # first choice: look on the referee's 2-D board directly (this still
        # works even if the internal state can no longer be trusted)
        if board2d is not None:
            try:
                n = len(board2d)
                best = None
                for r in range(n):
                    row = board2d[r]
                    for c in range(n):
                        if row[c] == EMPTY:
                            d = abs(r - n // 2) + abs(c - n // 2)
                            if best is None or d < best[0]:
                                best = (d, int(r), int(c))
                if best is not None:
                    return (best[1], best[2])
            except BaseException:             # noqa: BLE001
                pass

        board = self._safe_board
        n = self.n
        center = (n // 2) * n + (n // 2)
        if board[center] == EMPTY:
            return self._to_rc(center)
        # sweep once, from nearest to the center outwards
        for radius in range(1, n + 1):
            for r in range(n):
                for c in range(n):
                    if abs(r - n // 2) + abs(c - n // 2) == radius \
                            and board[r * n + c] == EMPTY:
                        return (int(r), int(c))
        for i, v in enumerate(board):
            if v == EMPTY:
                return self._to_rc(i)
        return (0, 0)     # board full (the referee will not call get_move then)

    def _to_rc(self, idx: int):
        return (int(idx // self.n), int(idx % self.n))


# ==================================================================== self-test

if __name__ == "__main__":
    import sys

    n, k = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (15, 5)
    ai = GomokuAI(1, n, k)
    empty = [[0] * n for _ in range(n)]
    print("First move on empty board:", ai.get_move(empty, None, 5.0))

    board = [[0] * n for _ in range(n)]
    for i in range(4):                        # white has 4 in a row, black must block
        board[n // 2][i + 1] = WHITE
    board[n // 2 + 1][n // 2 + 1] = BLACK
    print("Blocking move:", ai.get_move(board, (n // 2, 4), 5.0))
    print("Search statistics:", ai.last_search)
