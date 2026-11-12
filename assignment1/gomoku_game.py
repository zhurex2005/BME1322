#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gomoku_game.py — generalized Gomoku (m, n, k)-Game board logic

BME1322 Assignment 1. Stdlib-only, and it centralizes the "game rules":

    * Board: an N x N 2-D list, 0 = empty, 1 = black (first), 2 = white (second)
    * Move legality checks (out of bounds / already occupied)
    * K-in-a-row check (row, column, both diagonals; long runs win, no bans)
    * Draw detection (the board is full with no K-in-a-row)
    * Helper features: undo, copy, ASCII printing

The game logic is shared by the CLI (main_cli.py), the GUI (gui.py), and the
experiment scripts (experiment.py); the tournament AI file gomoku_ai.py is fully
self-contained so the judge can load it alone; it does not import this module.
"""

from __future__ import annotations

EMPTY, BLACK, WHITE = 0, 1, 2

#: The four scan directions: horizontal, vertical, main diagonal, anti-diagonal
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

STONE_CHAR = {EMPTY: ".", BLACK: "X", WHITE: "O"}

PLAYER_NAME = {BLACK: "Black (first player)", WHITE: "White (second player)"}


def opponent(player: int) -> int:
    """Return the opponent's stone id."""
    return 3 - player


class IllegalMove(ValueError):
    """The move is illegal (out of bounds or the position is occupied)."""


class GomokuGame:
    """All state and rules for one game of generalized Gomoku.

    Parameters
    ----------
    board_size : int
        Board side length N (5 <= N <= 15 here, but no upper limit is imposed).
    win_length : int
        Number of stones in a row K (3 <= K <= N).
    first_player : int
        The colour of the first player; defaults to black, 1.
    """

    def __init__(self, board_size: int = 15, win_length: int = 5,
                 first_player: int = BLACK) -> None:
        if board_size < 3:
            raise ValueError("board_size must be >= 3")
        if not (3 <= win_length <= board_size):
            raise ValueError("must satisfy 3 <= win_length <= board_size")
        if first_player not in (BLACK, WHITE):
            raise ValueError("first_player must be 1 (black) or 2 (white)")

        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.first_player = int(first_player)

        self.board = [[EMPTY] * self.board_size for _ in range(self.board_size)]
        self.history: list[tuple[int, int, int]] = []   # (row, col, player)
        self.current_player = self.first_player
        self.winner = 0                # 0 = undecided, 1/2 = winner
        self.win_line: list[tuple[int, int]] = []
        self.last_move: tuple[int, int] | None = None
        self.game_over = False
        self.draw = False
        self.move_count = 0

    # ------------------------------------------------------------ Basic queries

    @property
    def empty_count(self) -> int:
        return self.board_size * self.board_size - self.move_count

    def in_bounds(self, row: int, col: int) -> bool:
        n = self.board_size
        return 0 <= row < n and 0 <= col < n

    def is_legal(self, row: int, col: int) -> bool:
        """Whether (row, col) can be played right now."""
        if self.game_over:
            return False
        if not isinstance(row, int) or not isinstance(col, int):
            return False
        return self.in_bounds(row, col) and self.board[row][col] == EMPTY

    def legal_moves(self) -> list[tuple[int, int]]:
        n = self.board_size
        return [(r, c) for r in range(n) for c in range(n)
                if self.board[r][c] == EMPTY]

    # ------------------------------------------------------------ Placing / undo

    def place(self, row: int, col: int, player: int | None = None) -> bool:
        """Play at (row, col). Raises IllegalMove when the move is illegal.

        Returns True if this move wins the game immediately.
        """
        if self.game_over:
            raise IllegalMove("the game is already over")
        if player is None:
            player = self.current_player
        elif player != self.current_player:
            raise IllegalMove("it is not this side's turn to move")
        if not self.in_bounds(row, col):
            raise IllegalMove(f"coordinates out of bounds: ({row}, {col})")
        if self.board[row][col] != EMPTY:
            raise IllegalMove(f"position already occupied: ({row}, {col})")

        self.board[row][col] = player
        self.history.append((row, col, player))
        self.last_move = (row, col)
        self.move_count += 1

        if self.check_win_at(row, col, player):
            self.winner = player
            self.win_line = self.collect_win_line(row, col, player)
            self.game_over = True
            return True

        if self.empty_count == 0:
            self.game_over = True
            self.draw = True
            return False

        self.current_player = opponent(player)
        return False

    def undo(self, steps: int = 1) -> int:
        """Undo `steps` moves and return how many were actually undone."""
        undone = 0
        for _ in range(steps):
            if not self.history:
                break
            row, col, _player = self.history.pop()
            self.board[row][col] = EMPTY
            self.move_count -= 1
            undone += 1
        if undone:
            self.game_over = False
            self.draw = False
            self.winner = 0
            self.win_line = []
            if self.history:
                # The last move was made by history[-1][2]; its opponent is next
                self.last_move = (self.history[-1][0], self.history[-1][1])
                self.current_player = opponent(self.history[-1][2])
            else:
                self.last_move = None
                self.current_player = self.first_player
        return undone

    # ------------------------------------------------------------ Win detection

    def check_win_at(self, row: int, col: int, player: int) -> bool:
        """Whether player forms K in a row or more by playing (row, col)."""
        n, k = self.board_size, self.win_length
        board = self.board
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                rr, cc = row + sign * dr, col + sign * dc
                while 0 <= rr < n and 0 <= cc < n and board[rr][cc] == player:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= k:
                return True
        return False

    def collect_win_line(self, row: int, col: int, player: int) \
            -> list[tuple[int, int]]:
        """Return the coordinates of the longest winning run (for highlighting)."""
        n = self.board_size
        best: list[tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            line = [(row, col)]
            for sign in (1, -1):
                rr, cc = row + sign * dr, col + sign * dc
                while 0 <= rr < n and 0 <= cc < n \
                        and self.board[rr][cc] == player:
                    line.append((rr, cc))
                    rr += sign * dr
                    cc += sign * dc
            if len(line) >= self.win_length and len(line) > len(best):
                best = sorted(line)
        return best

    @property
    def is_over(self) -> bool:
        return self.game_over

    def result_text(self) -> str:
        if not self.game_over:
            return f"Game in progress, {PLAYER_NAME[self.current_player]} to move"
        if self.draw:
            return "Board full, draw (Draw)"
        return f"{PLAYER_NAME[self.winner]} wins! ({self.win_length} in a row)"

    # ------------------------------------------------------------ Utilities

    def copy(self) -> "GomokuGame":
        new = GomokuGame(self.board_size, self.win_length, self.first_player)
        new.board = [row[:] for row in self.board]
        new.history = list(self.history)
        new.current_player = self.current_player
        new.winner = self.winner
        new.win_line = list(self.win_line)
        new.last_move = self.last_move
        new.game_over = self.game_over
        new.draw = self.draw
        new.move_count = self.move_count
        return new

    def ascii_board(self, indent: str = "    ") -> str:
        """Return the ASCII board string (row/col numbers start at 0, per protocol)."""
        n = self.board_size
        width = len(str(n - 1))
        header = indent + " " * (width + 2) + " ".join(
            f"{c:>{width}}" for c in range(n))
        lines = [header]
        for r in range(n):
            cells = " ".join(f"{STONE_CHAR[v]:>{width}}" for v in self.board[r])
            marker = "*" if self.last_move and self.last_move[0] == r else " "
            lines.append(f"{indent}{r:>{width}} {marker}{cells}")
        return "\n".join(lines)


def build_from_board(board, win_length: int, first_player: int = BLACK) \
        -> GomokuGame:
    """Build a game object from an existing 2-D board (e.g. from the judge).

    Both sides' counts are tallied; as the order cannot be recovered, the
    history is rebuilt as "black then white", for display only.
    """
    n = len(board)
    game = GomokuGame(n, win_length, first_player)
    game.board = [list(row) for row in board]
    black = sum(row.count(BLACK) for row in game.board)
    white = sum(row.count(WHITE) for row in game.board)
    game.move_count = black + white
    order = []
    # Rebuild the history approximately: alternate black then white
    while len(order) < black + white:
        order.append(BLACK if len(order) % 2 == 0 else WHITE)
    for r in range(n):
        for c in range(n):
            if game.board[r][c]:
                game.history.append((r, c, game.board[r][c]))
    game.last_move = game.history[-1][:2] if game.history else None
    if black == white:
        game.current_player = BLACK
    elif black == white + 1:
        game.current_player = WHITE
    else:
        game.current_player = first_player
    return game


if __name__ == "__main__":  # simple self-test
    g = GomokuGame(5, 3)
    for mv in [(2, 2), (0, 0), (2, 3), (0, 1), (2, 4)]:
        g.place(*mv)
    print(g.ascii_board())
    print(g.result_text())
    assert g.winner == BLACK and g.game_over
    g.undo(2)
    assert not g.game_over and g.current_player == WHITE
    print("gomoku_game.py self-test OK")
