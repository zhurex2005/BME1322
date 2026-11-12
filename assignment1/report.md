# BME1322 Assignment 1 Lab Report: Generalized Gomoku and Adversarial Search

> Course content: Chapter 4, Adversarial Search & Games
> Submission directory: `assignment1/`

- Name / Student ID: <!-- fill in your name and student ID here -->
- AI display name: `AlphaBeta-Gomoku`

---

## 0. Abstract

This assignment implements a **generalized Gomoku ((m, n, k)-game)** program together with an AI that conforms to the course **Gomoku AI Protocol v1**:

- The AI uses **Negamax (the equivalent form of MiniMax) + Alpha-Beta pruning**, combined with **iterative deepening**, a **K-adaptive pattern evaluation function**, **neighbourhood candidate-move generation + incremental move ordering**, a **Zobrist transposition table** and **strict time-limit management**;
- The board side length `N` and the win length `K` both come entirely from constructor parameters, and the code contains **no hard-coded size and no "five in a row" assumption**; it adapts to the `5 ≤ N ≤ 15`, `3 ≤ K ≤ N` required by the lecture notes (and also supports larger sizes);
- The game program supports three modes — **human vs human / human vs AI / AI vs AI** — and provides both a **Tkinter GUI** and a **command-line interface**, which share the same game logic;
- Experiments show that: against a random AI, the AI achieves a **100% win rate** over 80 games across three `(N, K)` configurations; under per-move time limits ranging from 0.1 s to 5.0 s it **never once timed out** (the slowest move was exactly 75% of the limit + 2 ms); the protocol robustness self-check passed 10/10; and compliance matches against the course referee `arena.py` were completed at a 5.0 s time limit for several `(N, K)` settings.

---

## 1. Submission Contents and How to Run

### 1.1 File Structure

```
assignment1/
├── gomoku_ai.py       # [for grading] unified-protocol AI: Negamax + Alpha-Beta + iterative deepening
├── gomoku_game.py     # game logic: placing stones, K-in-a-row detection, draw detection, undo
├── main.py            # unified entry point: launches the GUI by default, --cli for the command line
├── main_cli.py        # command-line interface (human / human-vs-AI / AI-vs-AI)
├── gui.py             # Tkinter GUI (human / human-vs-AI / AI-vs-AI)
├── random_ai.py       # the random-AI reference implementation provided by the course (only a NAME variable added)
├── experiment.py      # experiment script: win rate / self-play / depth comparison / timing / robustness / referee compliance
├── results/           # raw experiment output (JSON + text logs), the data source for report.md
├── requirements.txt   # dependency notes (standard library only)
└── report.md          # this report
```

### 1.2 How to Run

```bash
# GUI (human vs AI by default; click to place a stone)
python main.py
python main.py --mode eve --ai gomoku_ai.py --ai2 random_ai.py --board 9 --win 4

# command line: human vs human / human vs AI / AI vs AI
python main_cli.py --mode pvp --board 15 --win 5
python main_cli.py --mode pve --human-color black --time 5
python main_cli.py --mode eve --board 9 --win 4 --ai gomoku_ai.py --ai2 random_ai.py

# course referee compliance test
python arena.py assignment1/gomoku_ai.py random_ai.py --board 15 --win 5 --games 2
python arena.py assignment1/gomoku_ai.py random_ai.py --board 9 --win 4 --time 1.0 --games 1 --quiet

# reproduce every experiment in this report
python experiment.py vs-random --board 15 --win 5 --time 1.0 --games 40
python experiment.py selfplay  --board 15 --win 5 --time 0.5 --games 8
python experiment.py depth     --board 9  --win 4 --time 1.0 --games 20 --caps 1 2 4 10
python experiment.py duel      --board 9  --win 4 --time 1.0 --games 20 --cap-a 1 --cap-b 10
python experiment.py timing    --board 15 --win 5 --games 2 --times 0.1 0.5 1.0
python experiment.py robustness --board 15 --win 5
python experiment.py arena-check --board 15 --win 5 --time 5.0 --games 2
```

---

## 2. Overall Design

### 2.1 Module Breakdown (and One Key Engineering Constraint)

The design separates "game rules" from "AI decision-making" completely:

| Module | Responsibility | Dependencies |
| --- | --- | --- |
| `gomoku_game.py` | Pure game state machine: legality, K-in-a-row detection, draws, undo, ASCII printing | standard library |
| `gomoku_ai.py` | Unified-protocol AI; interacts with the outside world only through `get_move(board, last_opponent_move, time_limit)` | **standard library only, fully self-contained** |
| `main_cli.py` / `gui.py` | The two interfaces, both calling the same `GomokuGame` and the same AI loader | the two above |
| `main.py` | Unified entry point (GUI by default, `--cli` switches to the command line) | the two above |
| `experiment.py` | Experiment script: drives two protocol AIs directly and collects statistics | `gomoku_game` |

**Key constraint: `gomoku_ai.py` must be completely self-contained.** The course referee `arena.py`
uses `importlib.util.spec_from_file_location` to load the AI file directly and **does not add the
file's directory to `sys.path`**, so if `gomoku_ai.py` contained `from gomoku_game import ...` it would
immediately raise `ImportError` inside the referee program and be judged the loser. The AI therefore has
its own independent board representation and rule checking: only the "protocol" is shared,
not the "code". This was a deliberate design decision during implementation.

### 2.2 Implementing the Three Play Modes

All three modes share the same `GomokuGame`; the only difference is "who generates the moves":

| Mode | Move source | Implementation notes |
| --- | --- | --- |
| Human vs human (`pvp`) | Two humans | Alternately parse `row col` input; illegal input produces a prompt and asks for re-entry |
| Human vs AI (`pve`) | A human + the AI | The human may choose black or white; when it is the AI's turn, `GomokuAI.get_move` is called; one undo revokes one move by each side (`undo_steps=2`), guaranteeing that it is the human's turn again |
| AI vs AI (`eve`) | Two AIs | `--ai` / `--ai2` load two protocol AIs respectively (either two copies of your own or another person's AI file) and play a whole game automatically |

The GUI and the CLI differ only in the interaction layer: the GUI uses `after()` to poll a background
thread that retrieves the AI's move (the AI thinks inside a `threading.Thread`), so the window does not
freeze during a 5-second search; the CLI calls it synchronously.
Both interfaces perform the same checks as the referee: **timeouts / illegal moves / raised exceptions**
are all detected and reported.

### 2.3 The Configuration Mechanism for N and K

The configuration chain is "command-line arguments or GUI widgets → constructor parameters → static tables generated at runtime":

- Both the CLI and the GUI expose `board`(N) and `win`(K) and validate `3 ≤ K ≤ N`;
- Every check in `GomokuGame(board_size, win_length)` (including `check_win_at`) takes
  `self.win_length` as its parameter; the literal 5 never appears;
- `GomokuAI(player_id, board_size, win_length)` generates the **evaluation score table from K** and
  the window/neighbourhood index tables from N at construction time; if the board size passed in by the
  referee differs from the constructor parameter, the AI **rebuilds its tables**
  (robustness test T5).

---

## 3. AI Algorithm Description

### 3.1 Problem Formalisation

Gomoku is a **two-player, zero-sum, perfect-information, deterministic, alternating-move** game, and can be
described by `(S, A, Result, Terminal, Utility)`:

- State `s`: the distribution of both players' stones on the N×N board, plus whose turn it is;
- Action `a`: placing a stone on any empty point (the branching factor is at most N², which is exactly why candidate-move pruning is necessary);
- Terminal test: either side connects K stones (an overline also wins; there are no forbidden moves) or the board is full (a draw);
- Utility: a win is `+WIN`, a draw is `0`, a loss is `-WIN`, and **the faster the win the better** (implemented as `WIN - ply`).

I use the Negamax formulation (an equivalent way of writing MiniMax, with shorter code and natural symmetry):

```
value(s) = max over a in A(s) of ( -value(Result(s, a)) )      whoever moves maximises their own score
```

### 3.2 Search Framework

```
function GET_MOVE(board, last_move, time_limit):
    rebuild(board)                       # sync the internal flat board; recompute pattern scores / neighbourhood / hash
    cands = candidates()                 # empty points with a stone within radius 2
    if cands is empty: return center_or_nearest_empty()

    ordered = ordered_moves(cands, me)   # incremental pattern gain + blocking gain ordering
    if some move in ordered wins immediately: return it          # fast path 1

    deadline = now + time_limit * 0.75
    best = ordered[0]                                            # fallback: the top-ranked move
    for depth in 1 .. max_depth_cap:                             # iterative deepening
        if now > deadline: break
        try:  value, move = root_search(depth, me)
        except Timeout: break                                    # abandon the unfinished level
        best = move
        if value >= WIN/2: break                                 # a forced win was found; stop early
    return best

function NEGAMAX(depth, alpha, beta, player, ply):
    if now > deadline: raise Timeout                             # check the time limit at every node
    if score[player] >= WIN: return WIN - ply
    if depth == 0: return evaluate(player)
    if TT hit with sufficient depth: return stored bound
    for move in top_k(ordered_moves(candidates(), player)):
        win = place(move, player)
        value = (WIN - ply) if win else -NEGAMAX(depth-1, -beta, -alpha, other(player), ply+1)
        remove()
        best = max(best, value); alpha = max(alpha, best)
        if alpha >= beta: break                                  # Alpha-Beta pruning
    store(TT, depth, best, flag)
    return best
```

### 3.3 Evaluation Function: K-Adaptive "Window Pattern" Scoring

**Core idea**: threats in Gomoku exist only inside straight-line windows of length K. All of the board's
windows of length K — horizontal, vertical and the two diagonals — are enumerated in advance (572 of them
for 15×15 with K=5), and then:

> For each window, if it contains **stones of both sides at the same time**, the window scores 0 for both sides (a dead window);
> if it contains stones of only one side and their count is c, the window scores `table[c]` for that side.

| Number of own stones in the window, c | Meaning | Score `table[c]` |
| --- | --- | --- |
| `c = K` | K stones already connected (a forced win) | `WIN = 10^12` |
| `c = K-1` | One stone short of K (open four / closed four) | `10^7` |
| `c = K-2` | Open-three level | `10^4` |
| `c = K-3` | Open-two level | `10^2` |
| `c = K-4` | Single-stone potential | `10` |
| Other | Weaker seeds | `1` |

There are three aspects of this design that I am particularly satisfied with:

1. **Naturally K-adaptive**: the score table is generated at construction time from `gap = K - c`, so K=3/5/6/7 need no code changes at all;
2. **No extra branching is needed to distinguish "open" from "closed"**: an "open four" (able to become five
   on either side) is scored `10^7` by **two** windows each, whereas a "closed four" is scored `10^7` by only
   one window; the difference in strength between the two is expressed naturally by the **window multiplicity**;
   by the same reasoning, the combined threat of a "double open three" also receives two lots of `10^4`,
   which is why the evaluation function can recognise compound threats;
3. **O(1) incremental maintenance**: placing a stone only requires updating the "windows containing that point"
   (≤ 4K of them); I use `_place/_remove` to maintain the cumulative scores of both sides, so evaluating a
   leaf node is a single subtraction, with no need to rescan the board.

The final evaluation value (from `player`'s point of view):

```
evaluate(player) = score[player] - 1.05 * score[opponent]
```

The opponent's pattern weight is set to 1.05, slightly greater than 1, which makes the AI **prefer to
eliminate the opponent's threats first** when the scores are close (defence first); this is the setting
I kept after tuning.

### 3.4 Candidate Move Generation and Move Ordering

- **Candidate generation**: only empty points with "a stone within radius 2" are considered (implemented with
  an incrementally maintained neighbourhood count array; placing/removing a stone only updates the surrounding
  24 cells). On an empty board the AI plays the centre point directly (the lecture notes explicitly allow this
  general heuristic). This reduces the branching factor from N² (225) to a few dozen.
- **Move ordering**: for each candidate point a `key = my score gain + 1.0 × opponent score loss` is computed,
  i.e. "both how much stronger I become and how much this move blocks the opponent". This key is a
  **pure computation that does not modify state**, so ordering is cheap; the ranking front-loads
  "win immediately > block the opponent's K > make my own four/three".
- Each level expands at most `candidate_limit = 12` moves (relaxed to 14/16 when K ≤ 3 or N ≤ 8),
  and good ordering plus a sensible K keeps Alpha-Beta's effective branching well below 12.

### 3.5 Time-Limit Management (a Hard Requirement of This Assignment)

The lecture notes stress that "a timeout = an immediate loss", so time-limit management is protected at four levels:

1. **Budget**: `deadline = now + time_limit × 0.75`, using only 75% of the limit and leaving 25% for
   interpreter jitter, GC and machine load;
2. **Check the time limit at every node**: this point is crucial — the initial implementation checked
   "once every 256 nodes", and in the self-play experiment that produced a **timeout of 0.63 s > 0.5 s**
   (each node takes about 1~3 ms, so 256 nodes amount to an error of 0.25~0.8 s). After switching to a
   per-node check, the few tens of nanoseconds of `perf_counter()` overhead became negligible, and the
   measured slowest move = budget + ≤ 2 ms (see Section 4.5);
3. **Iterative deepening**: every depth level is either "completed in full or discarded entirely", so at any
   moment there is a complete move from a finished shallower level in hand, and no half-finished result is
   ever returned;
4. **Fallback**: the top-ranked move is stored at the very beginning; if even the first level is not completed,
   or if any exception occurs along the way (the outer layer of `get_move` wraps `try/except BaseException`),
   the AI degrades to a move that is **guaranteed legal** (preferring points near the centre). The protocol
   demands that it "never lose because of an exception/timeout", and this layer is the final insurance.

### 3.6 Other Optimisations: the Zobrist Transposition Table

A fixed-seed Zobrist hash (one 64-bit random number per `(cell, colour)`, XOR-updated when a stone is placed
or removed, with a side key XORed in for the side to move) is used as the transposition-table key, caching
`(depth, bound flag, score)` with the standard Alpha-Beta exact/lower/upper bound reuse rules. To avoid
polluting the table with ply-dependent forced-win scores, only non-win/non-loss scores are cached.

---

## 4. Experimental Evaluation

Experimental environment: Windows, Python 3.11.4, 32 logical cores; all games were driven by `experiment.py`
using a self-built engine (the same protocol as the referee), with a new AI instance created for each game
(matching the lecture notes' requirement of "a new instance per game").
Unless stated otherwise, the **per-move time limit is 1.0 s** (the official format is 5.0 s; Section 4.5
reports the 5.0 s results separately). The random AI's random seed is fixed at `seed=20261011`, so the
results are reproducible.

### 4.1 Win Rate Against random_ai (15×15, K=5, 40 Games)

| Metric | Value |
| --- | --- |
| Games (half moving first, half moving second) | 40 |
| Record | **40 wins, 0 losses, 0 draws, a 100.0% win rate** (20/20 as black, 20/20 as white) |
| Average thinking time per move | **0.399 s** (slowest 0.751 s, time limit 1.0 s) |
| Average search nodes / depth | 2 040 nodes / 2.7 plies |
| Average game length | 9.8 moves (shortest 9, longest 13) |
| Way the game ended | In every game the AI itself connected 5 stones |

> Note: the random AI never blocks actively, so games are very short; ending after an average of
> 9.8 moves also shows that the AI can construct a five-in-a-row within a very short sequence.

### 4.2 Comparison Across Several (N, K) Configurations (20 Games Each)

| Configuration | Games | Record | Win rate | Average per move | Slowest move | Average nodes | Average depth | Average moves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9×9, K=4 | 20 | 20 wins 0 losses | **100%** | 0.288 s | 0.751 s | 2 570 | 2.7 | 7.6 |
| 11×11, K=5 | 20 | 20 wins 0 losses | **100%** | 0.425 s | 0.751 s | 2 745 | 3.0 | 10.8 |
| 15×15, K=5 | 40 | 40 wins 0 losses | **100%** | 0.399 s | 0.751 s | 2 040 | 2.7 | 9.8 |

All three configurations achieved a 100% win rate with **not a single timeout or illegal move**, showing
that the AI works correctly for different N and K; the larger the board and the larger K, the greater the
average number of moves and the search time, which matches intuition.

### 4.3 Self-Play (15×15, K=5, 8 Games, 0.5 s per Move)

| Metric | Value |
| --- | --- |
| Record | 3 wins 5 losses (A as black 0 wins 4 losses, A as white 3 wins 1 loss) |
| Average thinking time per move | A 0.333 s / B 0.329 s (time limit 0.5 s, slowest 0.377 s) |
| Average search nodes / depth | ≈ 1 000 nodes / 3.6 plies |
| Average game length | **32.4 moves** (shortest 20, longest 50) |
| Way the game ended | All 8 games were decided by five in a row; no draws and no timeouts |

Two observations from self-play:

1. The result is close to even (3:5) as expected — both sides are the same code; so that games are not
   completely identical, the experiment enabled "random tie-breaking among equal-scoring moves"
   (`randomize_ties=True`; it stays off in official grading, so the AI remains deterministic);
2. **Game length stretched from 9.8 moves (against random) to 32.4 moves**: against an opponent that blocks
   actively, the AI must genuinely construct double threats in order to win, which is exactly what
   "chess of real quality" looks like. With only 8 games in the sample, the difference between moving first
   and moving second (here A won 3 of its 4 games as white and none of its 4 as black) is small-sample
   noise, and no conclusion can be drawn from it.

### 4.4 The Effect of Search Depth

**(a) Gradually relaxing the depth cap under a fixed time limit, against random_ai (9×9, K=4, 20 games each)**

| `max_depth_cap` | Win rate | Average per move | Slowest move | Average nodes | Depth actually reached |
| --- | --- | --- | --- | --- | --- |
| 1 | 100% | 0.001 s | 0.001 s | 7 | 0.6 |
| 2 | 100% | 0.003 s | 0.006 s | 36 | 1.3 |
| 4 | 100% | 0.052 s | 0.182 s | 554 | 2.3 |
| 10 (default) | 100% | 0.289 s | 0.750 s | 2 545 | 2.7 |

All four configurations win against the random AI, so **this experiment cannot distinguish playing strength**,
but it clearly demonstrates the cost of "trading time for depth": the looser the depth cap, the deeper the
search goes and the more nodes are visited within the same time limit.

**(b) Direct head-to-head duel: `depth ≤ 1` vs `depth ≤ 10` (9×9, K=4, 20 games)**

| Metric | depth ≤ 1 | depth ≤ 10 |
| --- | --- | --- |
| Record | **0 wins 20 losses** | **20 wins 0 losses** |
| Average per move | 0.001 s | 0.328 s |
| Average nodes / actual depth | 11 / 0.9 | 2 079 / 3.8 |
| Average game length | 10.5 moves (10~11) | same as the left column |

The depth-1 AI only "looks one move ahead + does a static evaluation" and was shut out **20:0** by the full
version, with all 20 games finishing within 10~11 moves — it cannot recognise the opponent's two-move kill.
This comparison shows that the **evaluation function + deeper search** is the source of playing strength,
and that heuristic ordering alone is not enough.

**(c) Head-to-head duel under the official format: `depth ≤ 2` vs `depth ≤ 10` (15×15, K=5, 12 games, 1.0 s per move)**

| Metric | depth ≤ 2 | depth ≤ 10 |
| --- | --- | --- |
| Record | **3 wins 9 losses (a 25.0% win rate)** (2 wins as black, 1 win as white) | **9 wins 3 losses** |
| Average per move | 0.015 s (slowest 0.027 s) | 0.642 s (slowest 0.751 s) |
| Average nodes / actual depth | 49 / 1.9 | 2 076 / **4.1** |
| Average game length | **37.8 moves** (shortest 26, longest 90) | same as the left column |

This is the most convincing group of all: on the genuine 15×15 five-in-a-row configuration, a depth cap of 2
scores only a **25.0% win rate** against a depth cap of 10; the deeper side searches to 4.1 plies on average
and can therefore see the forced sequence "open three → closed four → five in a row" several moves ahead.
At the same time it exposes the problem discussed in Section 5.2: games are stretched to an average of
37.8 moves (longest 90), showing that with K=5 a pure depth search can hardly find a forced win within 4 plies,
and that **threat-space search (VCF/VCT) would be a more cost-effective direction than continuing to deepen**.

### 4.5 Behaviour Under Different (N, K) and Time-Limit Compliance

**(a) Time-limit compliance stress test (15×15, K=5, self-play)**

| Time limit `time_limit` | Moves counted | Slowest move | Average | Timeout count | Margin |
| --- | --- | --- | --- | --- | --- |
| 0.1 s | 18 | 0.076 s | 0.047 s | **0** | 0.024 s |
| 0.5 s | 40 | 0.377 s | 0.305 s | **0** | 0.123 s |
| 1.0 s | 96 | 0.752 s | 0.672 s | **0** | 0.248 s |
| 5.0 s (official format) | 67 | 3.751 s | 3.295 s | **0** | 1.249 s |

Every row lands right around "time limit × 0.75" (0.1→0.075, 0.5→0.375, 1.0→0.75, 5.0→3.75), with the
overshoot ≤ 2 ms, i.e. the "per-node time-limit check" described in Section 3.5 really does compress the
error to the scale of a single node. **Under the official 5.0 s format, the slowest move took 3.751 s,
still leaving a margin of 1.25 s.**

**(b) Compliance matches against the course referee `arena.py` (time limit 5.0 s)**

The command has the form `python arena.py assignment1/gomoku_ai.py random_ai.py --board N --win K --time 5.0`:

| Configuration (N, K) | Games | Result | Way the game ended | Any timeout/illegal move? |
| --- | --- | --- | --- | --- |
| (15, 5) | 2 | AlphaBeta-Gomoku 2.0 : 0.0 Random-AI | 11 / 12 moves, 5 in a row | none |
| (11, 6) | 2 | AlphaBeta-Gomoku 2.0 : 0.0 Random-AI | 17 / 18 moves, 6 in a row | none |
| (9, 4) | 2 | AlphaBeta-Gomoku 2.0 : 0.0 Random-AI | 7 / 8 moves, 4 in a row | none |
| (5, 3) | 2 | AlphaBeta-Gomoku 2.0 : 0.0 Random-AI | 5 / 6 moves, 3 in a row | none |
| (13, 7) | 1 | AlphaBeta-Gomoku 1.0 : 0.0 Random-AI | 17 moves, 7 in a row | none |

**All 9 games were won**, and **not once did a timeout, an exception or an illegal move cause a loss** in the
referee program; this also verifies that the AI loads and adapts correctly for
`K = 3 / 4 / 5 / 6 / 7` and `N = 5 / 9 / 11 / 13 / 15`.
(The raw output is in `results/log_arena_compliance.txt`; the referee's own Chinese banner is reproduced
there verbatim — `experiment.py` decodes the referee's subprocess output with a UTF-8 → locale → cp936
fallback so that a legacy Windows console cannot break the log.)

### 4.6 Protocol Robustness Self-Check (10/10 Passed)

The checks and results of `python experiment.py robustness`:

| # | Check | Result |
| --- | --- | --- |
| T1 | Does not modify the `board` passed in by the referee (lecture notes §2.5: treat it as read-only) | PASS |
| T2 | The return type is strictly a `(int, int)` tuple (not a bool, not a numpy scalar) | PASS |
| T3 | On an empty board with an extremely small time limit (0.01 / 0.05 / 0.2 s) it still returns a legal move without timing out | PASS |
| T4 | When only one empty point remains on the board, it fills that point | PASS |
| T5 | When the size passed in by the referee (9) differs from the constructor parameter (15) it adapts and returns a legal move | PASS |
| T6 | The extreme configuration `K = N` (5×5, K=5) works | PASS |
| T7 | 100 consecutive calls within a single game are all legal and all return on time (slowest 0.151 s / cap 0.2 s) | PASS |
| T8 | It wins immediately when it can connect K stones itself (it never misses a winning move) | PASS |
| — | Total | **10/10** |

### 4.7 Result Files

All raw output is stored in `assignment1/results/`:

| File | Contents |
| --- | --- |
| `vs_random_*.json`, `log_vs_random_*.txt` | The three configurations against the random AI in Sections 4.1 / 4.2 |
| `selfplay_*.json`, `log_selfplay_15x5.txt` | The self-play in Section 4.3 |
| `depth_*.json`, `log_depth_9x4.txt` | Depth caps against the random AI in Section 4.4(a) |
| `duel_*.json`, `log_duel_*.txt` | The depth duels in Section 4.4(b) (9×9 and 15×15) |
| `timing_*.json`, `log_timing*.txt` | Time-limit compliance in Section 4.5(a) |
| `log_arena_compliance.txt` | The compliance matches against the course referee `arena.py` in Section 4.5(b) |
| `robustness_*.json`, `log_robustness.txt` | The protocol robustness self-check in Section 4.6 |

---

## 5. Discussion and Reflection

### 5.1 Difficulties Encountered

1. **The timeout bug (the most important one)**: the initial time-limit check was "once every 256 nodes", and
   the self-play experiment immediately exposed a game with `timeout 0.63s` (time limit 0.5 s) — in official
   grading that is an **immediate loss**. The cause is that one search node takes about 1~3 ms, so a check
   interval of 256 nodes means an error of up to 0.8 s. After switching to a per-node check and keeping a 25%
   time margin plus the ordering fallback, the number of timeouts dropped to 0 at every time limit.
   This taught me that **in a competition timed by the wall clock, "fast on average" is meaningless; only
   the "upper bound" matters**.
2. **The AI file must be self-contained**: because the referee loads `gomoku_ai.py` on its own via `importlib`,
   it cannot `import` the `gomoku_game.py` in the same directory. I therefore gave the AI its own compact rule
   checker and treated "share the protocol, not the code" as a design constraint; the T5 check in
   `experiment.py` also verifies whether the AI adapts when the board size differs from the constructor parameter.
3. **Python's speed bottleneck**: in pure Python the cost of evaluation and move ordering is large, and in the
   15×15 midgame each move can only search 3~5 plies deep. This is the main direction for improvement below.

### 5.2 Shortcomings of the Evaluation Function

- The evaluation is a **linear superposition of window scores**; although "window multiplicity" can indirectly
  reflect the strength of open fours/closed fours and double open threes, it **does not explicitly perform a
  threat-space search (VCF/VCT)**: if a winning line requires 5~7 consecutive forcing threats, a depth-3~4
  search cannot see it, and the AI will either miss a "double threat" combination or fail to defend against one.
- The score table uses hand-chosen magnitudes (10^7 / 10^4 / 10^2 …), which only guarantee that the **ordering
  relations** are sensible; no systematic weight learning through self-play was performed.
- The defensive weight of 1.05 was picked by hand from a small number of games; there is no batch controlled
  experiment like the one in Section 4.3 to confirm the optimal value.
- There is no distinction between "threats on my turn" and "threats on the opponent's turn" (tempo), so in
  exchange positions where both sides hold fours/threes at the same time the evaluation is overly optimistic.

### 5.3 If I Were to Continue Improving

1. **Threat-space search**: add a layer of VCF/VCT search on top of Alpha-Beta, dedicated to finding lines of
   "consecutive forcing threats until K is reached", which is usually more cost-effective than simply deepening
   by one ply;
2. **Bitboards + bit operations**: pack each row/column/diagonal into an integer and determine the pattern of a
   whole line at once with bit operations; this is expected to raise node speed by an order of magnitude and
   thus search 2~3 plies deeper within 5 s;
3. **Transposition-table replacement policy**: the current policy is "clear the table when it is full";
   switching to depth/age-based replacement (e.g. a two-tier table with depth-preferred + always-replace)
   would be more stable;
4. **MCTS / UCT (a bonus direction in the lecture notes)**: Gomoku evaluation functions are hard to get right,
   whereas MCTS only needs random simulations to the end of the game; it could serve as a second AI
   (`mcts_ai.py`) playing head-to-head against the current Alpha-Beta, comparing the two paradigms on
   (15,5) and (9,4);
5. **Threat-first candidate generation**: the current rule is "empty points within radius 2"; this could be
   split further into "forced-win points / forced-block points / others", applying candidate pruning only to
   the last category.

### 5.4 Compliance Self-Check

- [x] Only the Python standard library is used (`random`, `time`, `threading`, `importlib`, `tkinter`, etc.);
      no external Gomoku engine, online service or network resource is called;
- [x] There is no strategy targeting a specific opponent (identifying an opponent's ID/name);
- [x] There is no table-lookup-style embedded fixed response; all core decisions come from runtime search and
      evaluation, and the only "rule-based" move is playing the centre on an empty board (explicitly allowed
      by the lecture notes);
- [x] The file name `gomoku_ai.py`, the class name `GomokuAI` and the interface signature match Section 2.2 of
      the lecture notes, with no changes made;
- [x] `N` and `K` both come from constructor parameters with no hard-coding; legality and the time limit are
      checked before a move is returned;
- [x] The `board` passed in by the referee is not modified (verified by T1).

### 5.5 References

1. Course lecture notes, "Generalized Gomoku Game and Adversarial Search (Assignment - 1)", Chapter 4,
   Adversarial Search & Games;
2. S. Russell & P. Norvig, *Artificial Intelligence: A Modern Approach*, the chapters on
   MiniMax / Alpha-Beta pruning / iterative deepening / transposition tables (referenced for pseudocode style);
3. Zobrist hashing and the Negamax formulation are standard practice in game programs; the code was implemented
   independently for this assignment.
