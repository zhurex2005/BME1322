#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui.py — Gomoku graphical interface (Tkinter, Python standard library)

Three modes are supported:
    * human vs human: players take turns placing stones with the mouse
    * human vs AI: the human plays Black or White, the other side is moved by
      an AI that follows a unified protocol
    * AI vs AI: two AI files play each other automatically (watchable move by
      move)

Board side N, win length K and the AI time limit are configurable; the AI
thinks in its own thread and the interface collects the result by polling with
after(), so the window stays responsive during the search.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from gomoku_game import BLACK, EMPTY, WHITE, GomokuGame, PLAYER_NAME, opponent
from main_cli import load_ai

HERE = Path(__file__).resolve().parent

BOARD_BG = "#e8b96a"
LINE_COLOR = "#5a3d17"
BLACK_STONE = "#1c1c1c"
WHITE_STONE = "#f7f7f7"
LAST_MARK = "#e23b3b"
WIN_COLOR = "#2f8f2f"


class GomokuGUI:
    """Tkinter Gomoku interface."""

    def __init__(self, root: tk.Tk, board_size: int = 15, win_length: int = 5,
                 mode: str = "pve", time_limit: float = 5.0,
                 ai_black_path: str = "gomoku_ai.py",
                 ai_white_path: str = "random_ai.py") -> None:
        self.root = root
        self.root.title("Gomoku — BME1322 Assignment 1")
        self.root.resizable(False, False)

        self.canvas_size = 620
        self.margin = 32
        self.cell = 30.0

        self.game = GomokuGame(board_size, win_length)
        self.mode = tk.StringVar(value=mode)
        self.size_var = tk.IntVar(value=board_size)
        self.win_var = tk.IntVar(value=win_length)
        self.time_var = tk.DoubleVar(value=time_limit)
        self.human_color_var = tk.StringVar(value="black")
        self.ai_black_var = tk.StringVar(value=ai_black_path)
        self.ai_white_var = tk.StringVar(value=ai_white_path)
        self.status_var = tk.StringVar(value="Ready to start")

        self.ai = {}                 # {color: instance}
        self.ai_name = {}
        self.thinking = False
        self._ai_result = None
        self._ai_error = None

        self._build_controls()
        self.canvas = tk.Canvas(root, width=self.canvas_size,
                                height=self.canvas_size, bg=BOARD_BG,
                                highlightthickness=1,
                                highlightbackground=LINE_COLOR)
        self.canvas.pack(padx=10, pady=(0, 6))
        self.canvas.bind("<Button-1>", self.on_click)

        self.status = ttk.Label(root, textvariable=self.status_var,
                                anchor="center", font=("Microsoft YaHei UI", 11))
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        self.new_game()

    # -------------------------------------------------------- UI construction

    def _build_controls(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=10, pady=8)

        ttk.Label(bar, text="Mode:").grid(row=0, column=0, padx=(0, 4))
        for col, (value, label) in enumerate(
                (("pvp", "Human"), ("pve", "Human vs AI"), ("eve", "AI vs AI"))):
            ttk.Radiobutton(bar, text=label, value=value,
                            variable=self.mode,
                            command=self.on_mode_change)\
                .grid(row=0, column=1 + col, padx=2)

        ttk.Label(bar, text="  Board N:").grid(row=0, column=4)
        ttk.Spinbox(bar, from_=5, to=25, width=4, textvariable=self.size_var)\
            .grid(row=0, column=5, padx=2)
        ttk.Label(bar, text="Win K:").grid(row=0, column=6)
        ttk.Spinbox(bar, from_=3, to=25, width=4, textvariable=self.win_var)\
            .grid(row=0, column=7, padx=2)
        ttk.Label(bar, text="Time limit(s):").grid(row=0, column=8)
        ttk.Spinbox(bar, from_=0.1, to=30, increment=0.5, width=5,
                    textvariable=self.time_var).grid(row=0, column=9, padx=2)

        ttk.Label(bar, text="I play:").grid(row=0, column=10, padx=(8, 2))
        ttk.Radiobutton(bar, text="Black", value="black",
                        variable=self.human_color_var).grid(row=0, column=11)
        ttk.Radiobutton(bar, text="White", value="white",
                        variable=self.human_color_var).grid(row=0, column=12)

        ttk.Button(bar, text="New game", command=self.new_game)\
            .grid(row=0, column=13, padx=(10, 2))
        ttk.Button(bar, text="Undo", command=self.undo)\
            .grid(row=0, column=14, padx=2)

        bar2 = ttk.Frame(self.root)
        bar2.pack(fill="x", padx=10, pady=(0, 4))
        ttk.Label(bar2, text="Black AI:").grid(row=0, column=0)
        ttk.Entry(bar2, textvariable=self.ai_black_var, width=28)\
            .grid(row=0, column=1, padx=2)
        ttk.Button(bar2, text="Browse…",
                   command=lambda: self._pick(self.ai_black_var))\
            .grid(row=0, column=2, padx=(0, 8))
        ttk.Label(bar2, text="White AI:").grid(row=0, column=3)
        ttk.Entry(bar2, textvariable=self.ai_white_var, width=28)\
            .grid(row=0, column=4, padx=2)
        ttk.Button(bar2, text="Browse…",
                   command=lambda: self._pick(self.ai_white_var))\
            .grid(row=0, column=5)

    def _pick(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select AI file", initialdir=str(HERE),
            filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            var.set(path)

    def on_mode_change(self) -> None:
        self.new_game()

    # ------------------------------------------------------- game management

    def new_game(self) -> None:
        n = int(self.size_var.get())
        k = int(self.win_var.get())
        if not (5 <= n <= 25):
            messagebox.showerror("Invalid parameter",
                                 "Board side N must be between 5 and 25")
            return
        if not (3 <= k <= n):
            messagebox.showerror("Invalid parameter", "We must have 3 <= K <= N")
            return
        self.game = GomokuGame(n, k)
        self.cell = (self.canvas_size - 2 * self.margin) / (n - 1)
        self.thinking = False
        self._ai_result = None
        self._load_ais()
        self.redraw()
        self.update_status()
        self.root.after(120, self.maybe_ai_move)

    def _load_ais(self) -> None:
        """Prepare the AI instances for the current mode."""
        n, k = self.game.board_size, self.game.win_length
        self.ai, self.ai_name = {}, {}
        mode = self.mode.get()
        try:
            if mode == "pve":
                human = BLACK if self.human_color_var.get() == "black" \
                    else WHITE
                ai_color = opponent(human)
                ai, name = load_ai(self.ai_black_var.get(), ai_color, n, k)
                self.ai[ai_color] = ai
                self.ai_name[ai_color] = name
            elif mode == "eve":
                b, bname = load_ai(self.ai_black_var.get(), BLACK, n, k)
                w, wname = load_ai(self.ai_white_var.get(), WHITE, n, k)
                self.ai = {BLACK: b, WHITE: w}
                self.ai_name = {BLACK: bname, WHITE: wname}
        except Exception as exc:                       # noqa: BLE001
            self.ai, self.ai_name = {}, {}
            messagebox.showerror("Failed to load AI", str(exc))

    def _is_human_turn(self) -> bool:
        if self.game.game_over:
            return False
        mode = self.mode.get()
        if mode == "pvp":
            return True
        if mode == "pve":
            human = BLACK if self.human_color_var.get() == "black" else WHITE
            return self.game.current_player == human
        return False

    # ------------------------------------------------------- interaction

    def on_click(self, event) -> None:
        if self.thinking or not self._is_human_turn():
            return
        row, col = self.pixel_to_cell(event.x, event.y)
        if row is None or not self.game.is_legal(row, col):
            return
        self.game.place(row, col)
        self.redraw()
        self.update_status()
        self.root.after(120, self.maybe_ai_move)

    def undo(self) -> None:
        if self.thinking:
            return
        steps = 2 if (self.mode.get() == "pve" and len(self.game.history) >= 2
                      and not self.game.game_over) else 1
        if self.game.undo(steps):
            self.redraw()
            self.update_status()
            self.root.after(120, self.maybe_ai_move)

    def maybe_ai_move(self) -> None:
        """If it is the AI's turn, start a background thread to think."""
        if self.game.game_over or self.thinking:
            return
        if self._is_human_turn():
            return
        player = self.game.current_player
        ai = self.ai.get(player)
        name = self.ai_name.get(player, "AI")
        if ai is None:
            self.status_var.set("No AI is available in this mode; check the "
                                "AI file paths")
            return
        self.thinking = True
        self.status_var.set(f"{name} thinking …")
        board_copy = [row[:] for row in self.game.board]
        last = self.game.last_move if self.game.history else None
        limit = float(self.time_var.get())

        def work():
            start = time.perf_counter()
            try:
                move = ai.get_move(board_copy, last, limit)
                self._ai_result = (move, time.perf_counter() - start, limit)
            except Exception as exc:                   # noqa: BLE001
                self._ai_error = repr(exc)
            finally:
                pass

        self._ai_error = None
        threading.Thread(target=work, daemon=True).start()
        self.root.after(60, self._poll_ai)

    def _poll_ai(self) -> None:
        if self._ai_error is not None:
            self.thinking = False
            messagebox.showerror("AI error", self._ai_error)
            self.status_var.set("The AI raised an exception; loss/abort")
            return
        if self._ai_result is None:
            self.root.after(60, self._poll_ai)
            return
        move, elapsed, limit = self._ai_result
        self._ai_result = None
        self.thinking = False
        player = self.game.current_player
        name = self.ai_name.get(player, "AI")
        if elapsed > limit:
            messagebox.showerror("Time limit exceeded",
                                 f"{name} took {elapsed:.2f}s, over the {limit}s "
                                 f"limit")
            self.status_var.set(f"{name} exceeded the time limit and loses")
            return
        if not isinstance(move, (tuple, list)) or len(move) != 2 \
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           for v in move) or not self.game.is_legal(int(move[0]), int(move[1])):
            messagebox.showerror("Illegal move",
                                 f"{name} returned the illegal move {move!r}")
            self.status_var.set(f"{name} played an illegal move and loses")
            return
        self.game.place(int(move[0]), int(move[1]))
        self.redraw()
        self.update_status(f"{name} played ({move[0]}, {move[1]}) "
                           f"in {elapsed:.2f}s")
        self.root.after(150, self.maybe_ai_move)

    def update_status(self, prefix: str = "") -> None:
        head = f"{prefix} " if prefix else ""
        if self.game.game_over:
            self.status_var.set(head + self.game.result_text())
        elif self.thinking:
            self.status_var.set(head + "AI thinking …")
        else:
            who = PLAYER_NAME[self.game.current_player]
            mode = self.mode.get()
            tag = ""
            if mode == "eve":
                tag = " (AI)"
            elif mode == "pve":
                human = BLACK if self.human_color_var.get() == "black" else WHITE
                tag = " (you)" if self.game.current_player == human else " (AI)"
            self.status_var.set(f"{head}Turn: {who}{tag}")

    # ------------------------------------------------------- drawing

    def cell_to_pixel(self, row: int, col: int):
        return (self.margin + col * self.cell, self.margin + row * self.cell)

    def pixel_to_cell(self, x: float, y: float):
        col = round((x - self.margin) / self.cell)
        row = round((y - self.margin) / self.cell)
        n = self.game.board_size
        if 0 <= row < n and 0 <= col < n:
            return int(row), int(col)
        return None, None

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        n = self.game.board_size
        # grid lines
        for i in range(n):
            x0, y0 = self.cell_to_pixel(i, 0)
            x1, y1 = self.cell_to_pixel(i, n - 1)
            c.create_line(x0, y0, x1, y1, fill=LINE_COLOR)
            x0, y0 = self.cell_to_pixel(0, i)
            x1, y1 = self.cell_to_pixel(n - 1, i)
            c.create_line(x0, y0, x1, y1, fill=LINE_COLOR)
        # star points (drawn only when the board is large enough)
        if n >= 9:
            mid = n // 2
            pts = {mid} | ({3, n - 4} if n >= 13 else set())
            for r in pts:
                for cc in pts:
                    x, y = self.cell_to_pixel(r, cc)
                    c.create_oval(x - 3, y - 3, x + 3, y + 3,
                                  fill=LINE_COLOR, outline="")
        # stones
        radius = self.cell * 0.42
        for r in range(n):
            for cc in range(n):
                v = self.game.board[r][cc]
                if v == EMPTY:
                    continue
                x, y = self.cell_to_pixel(r, cc)
                fill = BLACK_STONE if v == BLACK else WHITE_STONE
                c.create_oval(x - radius, y - radius, x + radius, y + radius,
                              fill=fill, outline="#333333")
        # marker for the last move
        if self.game.last_move:
            x, y = self.cell_to_pixel(*self.game.last_move)
            c.create_oval(x - 4, y - 4, x + 4, y + 4,
                          fill=LAST_MARK, outline="")
        # highlight the winning line
        if len(self.game.win_line) >= 2:
            (r0, c0), (r1, c1) = self.game.win_line[0], self.game.win_line[-1]
            x0, y0 = self.cell_to_pixel(r0, c0)
            x1, y1 = self.cell_to_pixel(r1, c1)
            c.create_line(x0, y0, x1, y1, fill=WIN_COLOR, width=4)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Gomoku graphical interface")
    parser.add_argument("--board", type=int, default=15)
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument("--mode", choices=("pvp", "pve", "eve"), default="pve")
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--ai", default="gomoku_ai.py")
    parser.add_argument("--ai2", default="random_ai.py")
    args = parser.parse_args(argv)

    root = tk.Tk()
    GomokuGUI(root, board_size=args.board, win_length=args.win,
              mode=args.mode, time_limit=args.time,
              ai_black_path=args.ai, ai_white_path=args.ai2)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
