#!/usr/bin/env python3
"""
BME1322 Assignment 3 -- Blood Glucose Management Agent
======================================================

Reproduces *every* figure and *every* number quoted in ``report.md``:

    python assignment3/main.py     # from the directory that contains assignment3/

Outputs
-------
* ``assignment3/figures/fig01_visit_counts.png`` ... ``fig09_convergence.png``
* ``assignment3/results.json``   (all numbers, for copying into the report)
* a markdown summary of the tables on stdout

Pipeline
--------
1. Part A: count-based model (P^, R^) on the shared 2 000-episode random
   dataset, then synchronous value iteration and greedy policy extraction.
2. Part B: tabular Q-learning over a single ordered pass of the *same* data.
3. Both policies are evaluated on the *same* 500 fresh episodes.
4. Experiments: data ablation (500/1000/2000), online epsilon-greedy
   (fixed vs decayed, 5 seeds), naive baselines, multi-seed robustness and a
   step-size/convergence sweep (Figure 9).

The environment file ``glucose_env.py`` is provided by the course and is
expected one directory above ``assignment3/`` (it is *not* part of the
submission).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------- #
# imports: allow "python main.py" from the parent directory or from here
# ---------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                             # course-provided, not submitted
    from glucose_env import (
        ACTION_COST,
        ACTION_NAMES,
        GAMMA,
        MAX_STEPS,
        STATE_NAMES,
        GlucoseEnv,
        collect_random_episodes,
    )
except ImportError as exc:                       # pragma: no cover
    raise SystemExit(
        "glucose_env.py was not found.\n"
        "It is the simulator provided with the assignment and is deliberately not\n"
        "part of this submission; copy it next to assignment3/ (or set PYTHONPATH),\n"
        f"then run `python assignment3/main.py` again.\n\n({exc})"
    ) from None

try:                                             # display-only constants (Table 1)
    from glucose_env import BIN_EDGES, REWARDS
except ImportError:                              # pragma: no cover
    BIN_EDGES = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    REWARDS = {"B1": -10.0, "B2": -1.0, "B3": +1.0, "B4": 0.0, "B5": -1.0,
               "B6": -2.0, "B7": -4.0, "B8": -6.0, "B9": -8.0}
from model_based import ModelBasedAgent         # noqa: E402
from q_learning import QLearningAgent           # noqa: E402
import plotting as P                            # noqa: E402

# ---------------------------------------------------------------------- #
# configuration (single source of truth for the whole report)
# ---------------------------------------------------------------------- #
FIG_DIR = os.path.join(HERE, "figures")
DATA_SEED = 2026            # seed of the shared random dataset
N_EPISODES = 2000           # data budget from the assignment
EVAL_EPISODES = 500         # evaluation episodes
EVAL_SEED_BASE = 777_000    # evaluation seeds are disjoint from the data
TERMINAL = "B1"             # severe hypoglycaemia: absorbing, R = -10
NON_TERMINAL = [s for s in STATE_NAMES if s != TERMINAL]
THETA = 1e-4                # value-iteration stopping threshold
ALPHA = 0.5                 # Q-learning step size
EPS_FIXED = 0.3
EPS_DECAY_START = 0.3
EPS_DECAY = 0.995
EPS_MIN = 0.05
ONLINE_SEEDS = [1, 2, 3, 4, 5]
WINDOW = 50                 # sliding-window width from the assignment
ROBUST_SEEDS = [DATA_SEED, 7, 123, 999, 31337]
SEED_CURVE_COLOR = "#9FB3C8"   # thin per-seed curves of the online experiment
MEAN_CURVE_COLOR = "#C00000"   # their seed mean
CONV_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
CONV_PASSES = list(range(1, 9))   # passes over the dataset (repeated learn())


# ---------------------------------------------------------------------- #
# evaluation
# ---------------------------------------------------------------------- #
def evaluate_policy(policy_fn: Callable[[str], Optional[str]],
                    n_episodes: int = EVAL_EPISODES,
                    seed_base: int = EVAL_SEED_BASE) -> Dict[str, float]:
    """Run ``n_episodes`` evaluation episodes with ``policy_fn``.

    Episode semantics follow the MDP of the assignment: **B1 is terminal**, so
    the episode ends as soon as B1 is entered and no action is taken from it.
    ``reset()`` occasionally lands in B1 (~4.8 % of episodes); such an episode
    is terminated immediately with the return the environment has paid so far
    (0.0) and is counted as a B1 termination.

    :return: dict with mean/median return, its std, B3 time fraction, B1
             termination rate and the mean episode length.
    """
    returns: List[float] = []
    lengths: List[int] = []
    b3_steps = 0
    all_steps = 0
    b1_terminations = 0
    b3_episodes = 0

    for i in range(n_episodes):
        env = GlucoseEnv(seed=seed_base + i)
        s = env.reset()
        total = 0.0
        steps = 0
        in_b3 = 0
        done = (s == TERMINAL)
        if done:
            b1_terminations += 1
        while not done:
            a = policy_fn(s)
            if a is None:                       # defensive: no action defined
                done = True
                break
            ns, r, done, _info = env.step(a)
            total += r
            steps += 1
            if ns == "B3":
                in_b3 += 1
            if done and ns == TERMINAL:
                b1_terminations += 1
            s = ns
        returns.append(total)
        lengths.append(steps)
        b3_steps += in_b3
        all_steps += steps
        if in_b3 > 0:
            b3_episodes += 1

    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
    return {
        "mean_return": mean_r,
        "std_return": var ** 0.5,
        "b3_fraction": b3_steps / all_steps if all_steps else 0.0,
        "b3_episode_fraction": b3_episodes / n_episodes,
        "b1_termination_rate": b1_terminations / n_episodes,
        "mean_length": sum(lengths) / len(lengths),
    }


def policy_from_table(table: Dict[str, str]) -> Callable[[str], Optional[str]]:
    """Policy function from a {state: action} dict (missing state -> None)."""
    return lambda s: table.get(s)


def agreement(policy_a: Dict[str, str], policy_b: Dict[str, str],
              states: Sequence[str] = NON_TERMINAL) -> Dict[str, object]:
    """Per-state agreement between two policies (non-terminal states only)."""
    per_state = {s: policy_a.get(s) == policy_b.get(s) for s in states}
    n_same = sum(1 for v in per_state.values() if v)
    return {"per_state": per_state, "n_same": n_same,
            "n_states": len(states), "rate": n_same / len(states)}


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _model_rows(model, state: str) -> Dict[str, object]:
    """P_hat(.|state, a) and R_hat(state, a) for the four actions.

    Stored in results.json so that every mechanism quoted in analysis.md
    (e.g. "i1 reaches B3 with probability 0.57") can be checked.
    """
    probs, rewards = {}, {}
    for a in ACTION_NAMES:
        row = model.get_transition(state, a)
        probs[a] = {s: round(p, 3) for s, p in row.items() if p > 1e-9}
        rewards[a] = round(model.get_reward(state, a), 3)
    return {"P": probs, "R": rewards,
            "visits": {a: model.get_visit_counts()[state][a]
                       for a in ACTION_NAMES}}


def _summarise_visits(visit: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    pairs = [(s, a, visit[s][a]) for s in STATE_NAMES for a in ACTION_NAMES]
    counts = [c for _s, _a, c in pairs]
    total = sum(counts)
    unvisited = [f"{s}/{a}" for s, a, c in pairs if c == 0]
    sparse = [f"{s}/{a}" for s, a, c in pairs if 0 < c <= 50]
    per_state = {s: sum(visit[s].values()) for s in STATE_NAMES}
    return {
        "n_pairs": len(pairs),
        "total_transitions": total,
        "min": min(counts),
        "max": max(counts),
        "median": sorted(counts)[len(counts) // 2],
        "unvisited_pairs": unvisited,
        "sparse_pairs_le50": sparse,
        "min_visits_per_pair": {s: min(visit[s].values()) for s in STATE_NAMES},
        "per_state": per_state,
        "counts": {f"{s}/{a}": visit[s][a] for s in STATE_NAMES for a in ACTION_NAMES},
    }


def _nice_bin_labels(counts: List[int]) -> tuple:
    """Log-ish bins for the visit-count histogram."""
    edges = [0, 25, 50, 100, 200, 400, 800, 1600, 3200]
    labels, values = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = sum(1 for c in counts if lo < c <= hi)
        labels.append(f"{lo}-{hi}")
        values.append(float(n))
    return labels, values


def _windowed(values: Sequence[float], w: int) -> List[float]:
    """Centred sliding-window mean of width ``w``."""
    out, n = [], len(values)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _print_table(title: str, header: Sequence[str], rows: Sequence[Sequence[str]]):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


# ---------------------------------------------------------------------- #
# main pipeline
# ---------------------------------------------------------------------- #
def main() -> Dict[str, object]:
    os.makedirs(FIG_DIR, exist_ok=True)
    results: Dict[str, object] = {}
    t_start = time.time()

    print("=" * 72)
    print("BME1322 Assignment 3 -- model-based vs Q-learning")
    print(f"backend for figures: {P.backend_name()}")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 0) the shared dataset
    # ------------------------------------------------------------------ #
    data = collect_random_episodes(N_EPISODES, seed=DATA_SEED)
    n_trans = sum(len(ep) for ep in data)
    n_steps = n_trans
    print(f"\n[data] {len(data)} episodes, {n_trans} transitions "
          f"(seed {DATA_SEED})")
    results["data"] = {"seed": DATA_SEED, "episodes": len(data),
                       "transitions": n_trans,
                       "mean_length": n_trans / len(data),
                       "max_length": max(len(ep) for ep in data)}

    # ------------------------------------------------------------------ #
    # 1) PART A -- model-based agent
    # ------------------------------------------------------------------ #
    model = ModelBasedAgent(states=list(STATE_NAMES), actions=list(ACTION_NAMES),
                            gamma=GAMMA, seed=0)
    model.learn_model(data)
    delta_hist = model.run_value_iteration(theta=THETA, max_iterations=1000)
    V_a = model.get_value_table()
    Q_a = model.get_q_table()
    pol_a = model.get_policy_table()
    visits = model.get_visit_counts()
    visit_stats = _summarise_visits(visits)

    print(f"\n[Part A] value iteration converged in {model.vi_iterations} "
          f"iterations (last max|dV| = {delta_hist[-1]:.2e}, theta={THETA:g})")
    print(f"[Part A] terminal states detected from data: {model.terminal_states}")

    # --- figure 1: visit counts ---------------------------------------- #
    counts = [visits[s][a] for s in STATE_NAMES for a in ACTION_NAMES]
    labels, binned = _nice_bin_labels(counts)
    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig01_visit_counts.png"),
        title="Figure 1: visit counts of the 2 000-episode random dataset",
        figsize=(12.5, 4.4),
        panels=[
            P.Panel(kind="heatmap", title="(a) N(s,a) per state-action pair",
                    matrix=[[visits[s][a] for a in ACTION_NAMES]
                            for s in STATE_NAMES],
                    rows=list(STATE_NAMES), cols=list(ACTION_NAMES),
                    fmt="{:.0f}", xlabel="action", cbar_label="N(s,a)",
                    annotate=True),
            P.Panel(kind="bars", title="(b) distribution over the 36 (s,a) pairs",
                    categories=labels, values=binned, fmt="{:.0f}",
                    xlabel="visit count N(s,a)", ylabel="# of (s,a) pairs",
                    annotate=True, ytick_int=True),
        ]))

    # --- figure 2: convergence + V -------------------------------------- #
    its = list(range(1, len(delta_hist) + 1))
    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig02_value_iteration.png"),
        title="Figure 2: value iteration on the estimated model",
        figsize=(12.5, 4.4),
        panels=[
            P.Panel(kind="lines", title="(a) sup-norm change per sweep",
                    xlabel="iteration k", ylabel="max change in V per sweep",
                    logy=True, legend=False, hlines=[THETA],
                    series=[P.Series("max|dV|", delta_hist, xs=its)]),
            P.Panel(kind="bars", title="(b) converged V*(s)",
                    categories=list(STATE_NAMES), values=[V_a[s] for s in STATE_NAMES],
                    ylabel="V*(s)", annotate=True, fmt="{:.2f}",
                    colors=[("#C0504D" if s == TERMINAL else "#2E5C8A")
                            for s in STATE_NAMES]),
        ]))

    # --- figure 3: Q table of the model -------------------------------- #
    def _greedy_cells(table, states, actions):
        """(row, col) of the greedy action of every non-terminal state."""
        return [(i, actions.index(max(actions, key=lambda a: table[s][a])))
                for i, s in enumerate(states) if s not in (TERMINAL,)]

    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig03_qvalues_model.png"),
        title="Figure 3: Q(s,a) extracted from V* (Part A); box = greedy action",
        figsize=(8.2, 4.6),
        panels=[P.Panel(
            kind="heatmap", title="",
            matrix=[[Q_a[s][a] for a in ACTION_NAMES] for s in STATE_NAMES],
            rows=list(STATE_NAMES), cols=list(ACTION_NAMES), fmt="{:.2f}",
            xlabel="action", cbar_label="Q(s,a)", annotate=True,
            emphasis=_greedy_cells(Q_a, STATE_NAMES, ACTION_NAMES))]))

    # --- evaluation ----------------------------------------------------- #
    met_a = evaluate_policy(policy_from_table(pol_a))
    print(f"[Part A] eval: return {met_a['mean_return']:.3f} "
          f"+- {met_a['std_return']:.2f} | B3 {met_a['b3_fraction']:.4f} | "
          f"B1 term {met_a['b1_termination_rate']:.4f} | "
          f"len {met_a['mean_length']:.1f}")

    # ------------------------------------------------------------------ #
    # 2) PART B -- tabular Q-learning, one ordered pass
    # ------------------------------------------------------------------ #
    ql = QLearningAgent(states=list(STATE_NAMES), actions=list(ACTION_NAMES),
                        gamma=GAMMA, alpha=ALPHA, seed=0)
    ql.learn(data, mode="fixed")
    Q_b = ql.get_q_table()
    pol_b = ql.get_policy_table()
    V_b = {s: ql.get_state_value(s) for s in STATE_NAMES}
    met_b = evaluate_policy(policy_from_table(pol_b))
    agree = agreement(pol_a, pol_b)
    print(f"\n[Part B] {ql.n_updates} updates (alpha={ALPHA})")
    print(f"[Part B] eval: return {met_b['mean_return']:.3f} "
          f"+- {met_b['std_return']:.2f} | B3 {met_b['b3_fraction']:.4f} | "
          f"B1 term {met_b['b1_termination_rate']:.4f} | "
          f"len {met_b['mean_length']:.1f}")
    print(f"[Part B] policy agreement with Part A: {agree['n_same']}/"
          f"{agree['n_states']}")

    # --- figure 8: Q table of Q-learning ------------------------------- #
    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig05_qvalues_qlearning.png"),
        title="Figure 5: Q(s,a) after one ordered pass of Q-learning (Part B); "
              "box = greedy action",
        figsize=(8.2, 4.6),
        panels=[P.Panel(
            kind="heatmap", title="",
            matrix=[[Q_b[s][a] for a in ACTION_NAMES] for s in STATE_NAMES],
            rows=list(STATE_NAMES), cols=list(ACTION_NAMES), fmt="{:.2f}",
            xlabel="action", cbar_label="Q(s,a)", annotate=True,
            emphasis=_greedy_cells(Q_b, STATE_NAMES, ACTION_NAMES))]))

    # ------------------------------------------------------------------ #
    # 3) naive baselines
    # ------------------------------------------------------------------ #
    baselines: Dict[str, Dict[str, float]] = {}
    for a in ACTION_NAMES:
        baselines[f"always-{a}"] = evaluate_policy(lambda _s, a=a: a)
        print(f"[baseline] always-{a:<2s}: return "
              f"{baselines[f'always-{a}']['mean_return']:9.2f} | "
              f"B3 {baselines[f'always-{a}']['b3_fraction']:.3f} | "
              f"B1 {baselines[f'always-{a}']['b1_termination_rate']:.3f}")

    # ------------------------------------------------------------------ #
    # 4) data ablation 500 / 1000 / 2000
    # ------------------------------------------------------------------ #
    ablation = []
    for k in (500, 1000, 2000):
        sub = data[:k]
        m = ModelBasedAgent(states=list(STATE_NAMES),
                            actions=list(ACTION_NAMES), gamma=GAMMA, seed=0)
        m.learn_model(sub)
        m.run_value_iteration(theta=THETA)
        pa = m.get_policy_table()
        ma = evaluate_policy(policy_from_table(pa))

        q = QLearningAgent(states=list(STATE_NAMES),
                           actions=list(ACTION_NAMES), gamma=GAMMA, alpha=ALPHA,
                           seed=0)
        q.learn(sub, mode="fixed")
        pb = q.get_policy_table()
        mb = evaluate_policy(policy_from_table(pb))
        ag = agreement(pa, pb)
        vk = m.get_visit_counts()
        n_lt100 = sum(1 for s in STATE_NAMES for a in ACTION_NAMES
                      if vk[s][a] < 100)
        ablation.append({
            "episodes": k,
            "transitions": sum(len(e) for e in sub),
            "visits_a": _summarise_visits(vk),
            "n_pairs_lt100": n_lt100,
            "min_visits": min(vk[s][a] for s in STATE_NAMES for a in ACTION_NAMES),
            "policy_a": pa, "policy_b": pb,
            "vi_iterations": m.vi_iterations,
            "agreement": ag,
            "eval_a": ma, "eval_b": mb,
        })
        print(f"[ablation] n={k:5d}: A return {ma['mean_return']:8.2f} "
              f"(B3 {ma['b3_fraction']:.3f}) | B return "
              f"{mb['mean_return']:8.2f} (B3 {mb['b3_fraction']:.3f}) | "
              f"agreement {ag['n_same']}/{ag['n_states']}")

    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig04_data_ablation.png"),
        title="Figure 4: effect of the data budget (policies evaluated on "
              "500 fresh episodes)",
        figsize=(13.0, 3.9),
        panels=[
            P.Panel(kind="grouped_bars", title="(a) mean undiscounted return",
                    categories=[str(a["episodes"]) for a in ablation],
                    ylabel="mean return", legend=True,
                    series=[P.Series("Part A (model)",
                                     [a["eval_a"]["mean_return"] for a in ablation]),
                            P.Series("Part B (Q-learning)",
                                     [a["eval_b"]["mean_return"] for a in ablation])]),
            P.Panel(kind="grouped_bars", title="(b) fraction of time in B3",
                    categories=[str(a["episodes"]) for a in ablation],
                    ylabel="P(state = B3)", fmt="{:.3f}",
                    series=[P.Series("Part A", [a["eval_a"]["b3_fraction"] for a in ablation]),
                            P.Series("Part B", [a["eval_b"]["b3_fraction"] for a in ablation])]),
            P.Panel(kind="bars", title="(c) sparsity of the estimated model",
                    categories=[str(a["episodes"]) for a in ablation],
                    values=[float(a["n_pairs_lt100"]) for a in ablation],
                    ylabel="(s,a) pairs with N < 100", annotate=True,
                    fmt="{:.0f}", ytick_int=True,
                    xlabel="episodes of training data"),
        ]))

    # ------------------------------------------------------------------ #
    # 5) online epsilon-greedy experiment (Chapter 9: exploration/regret)
    # ------------------------------------------------------------------ #
    def train_online(seed: int, decay: bool, episodes: int = N_EPISODES):
        """Online Q-learning: the agent picks every action itself."""
        agent = QLearningAgent(states=list(STATE_NAMES),
                               actions=list(ACTION_NAMES), gamma=GAMMA,
                               alpha=ALPHA, epsilon=EPS_DECAY_START if decay
                               else EPS_FIXED,
                               epsilon_decay=EPS_DECAY, epsilon_min=EPS_MIN,
                               seed=seed)
        env = GlucoseEnv(seed=seed * 100_000 + 11)
        rets, eps_hist = [], []
        for ep in range(episodes):
            s = env.reset()
            total, steps, done = 0.0, 0, (s == TERMINAL)
            while not done and steps < MAX_STEPS:
                a = agent.act(s)
                ns, r, done, _ = env.step(a)
                agent.update(s, a, r, ns, done)
                total += r
                steps += 1
                s = ns
            if decay:
                agent.decay_epsilon()
            rets.append(total)
            eps_hist.append(agent.epsilon)
        return rets, eps_hist

    online: Dict[str, Dict[str, object]] = {}
    for name, decay in (("fixed", False), ("decay", True)):
        curves, eps_curves = [], []
        for seed in ONLINE_SEEDS:
            rets, eps = train_online(seed, decay)
            curves.append(rets)
            eps_curves.append(eps)
        n = len(curves[0])
        mean_curve = [sum(c[i] for c in curves) / len(curves) for i in range(n)]
        lo_curve = [min(c[i] for c in curves) for i in range(n)]
        hi_curve = [max(c[i] for c in curves) for i in range(n)]
        tail = [sum(c[-200:]) / 200.0 for c in curves]
        online[name] = {
            "curves": curves, "seeds": ONLINE_SEEDS,
            "mean_curve": mean_curve, "lo": lo_curve, "hi": hi_curve,
            "eps_curve": [sum(e[i] for e in eps_curves) / len(eps_curves)
                          for i in range(n)],
            "final_mean_200": sum(tail) / len(tail),
            "final_per_seed": tail,
            "total_return": [sum(c) for c in curves],
        }
        g_star = met_a["mean_return"]
        regret = [N_EPISODES * g_star - sum(c) for c in curves]
        online[name]["regret"] = regret
        online[name]["regret_mean"] = sum(regret) / len(regret)
        print(f"[online] {name:5s}: last-200 mean {online[name]['final_mean_200']:7.2f} | "
              f"total regret {online[name]['regret_mean']:9.1f}")

    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig06_online_learning.png"),
        title="Figure 6: online epsilon-greedy Q-learning, "
              f"{len(ONLINE_SEEDS)} seeds, {WINDOW}-episode sliding window",
        figsize=(14.2, 4.0),
        panels=[
            P.Panel(kind="lines", title="(a) fixed epsilon = 0.3",
                    xlabel="episode", ylabel="return (windowed mean)",
                    legend=True,
                    series=[P.Series("", _windowed(c, WINDOW),
                                     xs=list(range(len(c))), thin=True,
                                     color=SEED_CURVE_COLOR)
                            for c in online["fixed"]["curves"]] +
                           [P.Series("seed mean", _windowed(online["fixed"]["mean_curve"], WINDOW),
                                     xs=list(range(N_EPISODES)),
                                     color=MEAN_CURVE_COLOR)]),
            P.Panel(kind="lines", title="(b) decayed epsilon 0.3 -> 0.05",
                    xlabel="episode", ylabel="return (windowed mean)",
                    legend=True,
                    series=[P.Series("", _windowed(c, WINDOW),
                                     xs=list(range(len(c))), thin=True,
                                     color=SEED_CURVE_COLOR)
                            for c in online["decay"]["curves"]] +
                           [P.Series("seed mean", _windowed(online["decay"]["mean_curve"], WINDOW),
                                     xs=list(range(N_EPISODES)),
                                     color=MEAN_CURVE_COLOR)]),
            P.Panel(kind="bars", title="(c) mean return of the last 200 episodes",
                    categories=["fixed", "decay"],
                    values=[online["fixed"]["final_mean_200"],
                            online["decay"]["final_mean_200"]],
                    ylabel="return", annotate=True, fmt="{:.2f}"),
        ]))

    # ------------------------------------------------------------------ #
    # 6) Part A vs Part B vs baselines
    # ------------------------------------------------------------------ #
    comparison = {"model (A)": met_a, "Q-learn (B)": met_b}
    comparison.update({"all w": baselines["always-w"], "all g": baselines["always-g"],
                       "all i1": baselines["always-i1"],
                       "all i2": baselines["always-i2"]})

    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig07_ab_comparison.png"),
        title="Figure 7: the two learned policies against naive baselines "
              "(500 fresh episodes each)",
        figsize=(13.0, 3.9),
        panels=[
            P.Panel(kind="bars", title="(a) mean undiscounted return",
                    categories=list(comparison.keys()),
                    values=[m["mean_return"] for m in comparison.values()],
                    ylabel="mean return", annotate=True, fmt="{:.1f}"),
            P.Panel(kind="bars", title="(b) fraction of time in B3",
                    categories=list(comparison.keys()),
                    values=[m["b3_fraction"] for m in comparison.values()],
                    ylabel="P(state = B3)", annotate=True, fmt="{:.3f}"),
            P.Panel(kind="bars", title="(c) B1 termination rate",
                    categories=list(comparison.keys()),
                    values=[m["b1_termination_rate"] for m in comparison.values()],
                    ylabel="P(episode ends in B1)", annotate=True, fmt="{:.3f}"),
        ]))

    # ------------------------------------------------------------------ #
    # 7) robustness across dataset seeds
    # ------------------------------------------------------------------ #
    robust = []
    for sd in ROBUST_SEEDS:
        d = data if sd == DATA_SEED else collect_random_episodes(N_EPISODES, seed=sd)
        m = ModelBasedAgent(states=list(STATE_NAMES),
                            actions=list(ACTION_NAMES), gamma=GAMMA, seed=0)
        m.learn_model(d)
        m.run_value_iteration(theta=THETA)
        pa = m.get_policy_table()
        ma = evaluate_policy(policy_from_table(pa))
        q = QLearningAgent(states=list(STATE_NAMES),
                           actions=list(ACTION_NAMES), gamma=GAMMA, alpha=ALPHA,
                           seed=0)
        q.learn(d, mode="fixed")
        pb = q.get_policy_table()
        mb = evaluate_policy(policy_from_table(pb))
        ag = agreement(pa, pb)
        robust.append({"seed": sd, "policy_a": pa, "policy_b": pb,
                       "eval_a": ma, "eval_b": mb, "agreement": ag,
                       # Q tables so that every number quoted in analysis.md
                       # can be traced back to this file
                       "q_a": {s: {a: round(v, 3) for a, v in row.items()}
                               for s, row in m.get_q_table().items()},
                       "q_b": {s: {a: round(v, 3) for a, v in row.items()}
                               for s, row in q.get_q_table().items()},
                       "v_a": m.get_value_table(),
                       "v_b": {s: round(q.get_state_value(s), 3)
                               for s in STATE_NAMES},
                       # the two states discussed in analysis.md
                       "model_B4": _model_rows(m, "B4"),
                       "model_B8": _model_rows(m, "B8")})
        print(f"[robust] seed {sd:6d}: A {ma['mean_return']:8.2f} | "
              f"B {mb['mean_return']:8.2f} | agreement "
              f"{ag['n_same']}/{ag['n_states']} | "
              f"disagree: {[s for s, v in ag['per_state'].items() if not v]}")

    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig08_robustness.png"),
        title="Figure 8: stability across dataset seeds (identical evaluation "
              "seeds for every run)",
        figsize=(13.0, 4.0),
        panels=[
            P.Panel(kind="grouped_bars", title="(a) mean return per dataset seed",
                    categories=[str(r["seed"]) for r in robust],
                    ylabel="mean return", legend=True,
                    series=[P.Series("Part A", [r["eval_a"]["mean_return"] for r in robust]),
                            P.Series("Part B", [r["eval_b"]["mean_return"] for r in robust])]),
            P.Panel(kind="bars", title="(b) policy agreement with Part A (of 8)",
                    categories=[str(r["seed"]) for r in robust],
                    values=[float(r["agreement"]["n_same"]) for r in robust],
                    ylabel="# identical states", annotate=True, fmt="{:.0f}",
                    ylim=(0, 8)),
            P.Panel(kind="bars", title="(c) B1 termination rate of Part B",
                    categories=[str(r["seed"]) for r in robust],
                    values=[r["eval_b"]["b1_termination_rate"] for r in robust],
                    ylabel="P(episode ends in B1)", annotate=True, fmt="{:.3f}"),
        ]))

    # ------------------------------------------------------------------ #
    # 8) convergence conditions of Chapter 9 (step size)
    # ------------------------------------------------------------------ #
    def q_gap(q: QLearningAgent) -> float:
        """||Q - Q^A||_inf over the non-terminal (s, a) pairs."""
        return max(abs(q.get_q_value(s, a) - Q_a[s][a])
                   for s in NON_TERMINAL for a in ACTION_NAMES)

    # (i) one pass, different constant step sizes
    alpha_sweep = []
    for alpha in CONV_ALPHAS:
        q = QLearningAgent(states=list(STATE_NAMES),
                           actions=list(ACTION_NAMES), gamma=GAMMA,
                           alpha=alpha, seed=0)
        q.learn(data, mode="fixed")
        pq = q.get_policy_table()
        ag = agreement(pol_a, pq)
        metq = evaluate_policy(policy_from_table(pq))
        alpha_sweep.append({"alpha": alpha, "gap": q_gap(q), "agreement": ag,
                            "eval": metq, "policy": pq})
        print(f"[conv] 1 pass, alpha={alpha:<4}: ||Q-Q^A||inf {q_gap(q):7.3f} | "
              f"agreement {ag['n_same']}/8 | return {metq['mean_return']:8.2f}")

    # (ii) repeated passes, constant vs Robbins-Monro step size
    conv_settings = [("const alpha=0.9", 0.9, False),
                     ("const alpha=0.5", 0.5, False),
                     ("decaying 0.5/(1+k)", 0.5, True)]
    conv_curves = {}
    for name, a0, decay in conv_settings:
        q = QLearningAgent(states=list(STATE_NAMES),
                           actions=list(ACTION_NAMES), gamma=GAMMA,
                           alpha=a0, seed=0)
        gaps, rets, agrs = [], [], []
        for k in range(len(CONV_PASSES)):
            q.alpha = a0 / (1 + k) if decay else a0
            q.learn(data, mode="fixed")
            gaps.append(q_gap(q))
            pq = q.get_policy_table()
            agrs.append(float(agreement(pol_a, pq)["n_same"]))
            rets.append(evaluate_policy(policy_from_table(pq))["mean_return"])
        conv_curves[name] = {"gaps": gaps, "returns": rets, "agreements": agrs,
                             "passes": CONV_PASSES}
        print(f"[conv] {name:20s}: gap "
              f"{' -> '.join(f'{g:.2f}' for g in gaps)} | return "
              f"{rets[-1]:.2f} | agreement {int(agrs[-1])}/8")

    P.save(P.Figure(
        out=os.path.join(FIG_DIR, "fig09_convergence.png"),
        title="Figure 9: Chapter 9 convergence conditions -- the step size "
              "controls whether Q-learning approaches Q*",
        figsize=(13.0, 3.9),
        panels=[
            P.Panel(kind="bars", title="(a) one pass: distance to Q*",
                    categories=[f"{a:g}" for a in CONV_ALPHAS],
                    values=[r["gap"] for r in alpha_sweep],
                    ylabel="||Q - Q*||inf", annotate=True, fmt="{:.2f}",
                    xlabel="constant step size alpha"),
            P.Panel(kind="lines", title="(b) repeated passes: distance to Q*",
                    xlabel="pass over the dataset", ylabel="||Q - Q*||inf",
                    legend=True, logy=True,
                    series=[P.Series(name, c["gaps"], xs=[float(k) for k in c["passes"]],
                                     color=P.PALETTE[i % len(P.PALETTE)])
                            for i, (name, c) in enumerate(conv_curves.items())]),
            P.Panel(kind="bars", title="(c) one pass: evaluation return",
                    categories=[f"{a:g}" for a in CONV_ALPHAS],
                    values=[r["eval"]["mean_return"] for r in alpha_sweep],
                    ylabel="mean return", annotate=True, fmt="{:.2f}",
                    xlabel="constant step size alpha",
                    hlines=[met_a["mean_return"]],
                    ylim=(0.0, met_a["mean_return"] * 1.18)),
        ]))
    # ------------------------------------------------------------------ #
    # 9) tables on stdout + results.json
    # ------------------------------------------------------------------ #
    results.update({
        "part_a": {"policy": pol_a, "values": V_a, "vi_iterations": model.vi_iterations,
                   "vi_final_delta": delta_hist[-1], "terminal_states": model.terminal_states,
                   "eval": met_a, "visits": visit_stats, "q": Q_a},
        "part_b": {"policy": pol_b, "q": Q_b, "values": V_b, "eval": met_b,
                   "n_updates": ql.n_updates, "alpha": ALPHA},
        "agreement": agree,
        "baselines": baselines,
        "ablation": ablation,
        "online": {k: {kk: vv for kk, vv in v.items()
                            if kk not in ("curves", "lo", "hi")}
                   for k, v in online.items()},
        "robust": robust,
        "convergence": {"alpha_sweep": alpha_sweep, "curves": conv_curves},
        "setup": {"data_seed": DATA_SEED, "episodes": N_EPISODES,
                  "eval_episodes": EVAL_EPISODES, "eval_seed_base": EVAL_SEED_BASE,
                  "theta": THETA, "alpha": ALPHA, "gamma": GAMMA,
                  "backend": P.backend_name(), "online_seeds": ONLINE_SEEDS,
                  "window": WINDOW},
        "runtime_seconds": time.time() - t_start,
    })

    # ---- markdown tables ---------------------------------------------- #
    rows = []
    for i, s in enumerate(STATE_NAMES):
        rng = f"[{BIN_EDGES[i]:g}, {BIN_EDGES[i + 1]:g})"
        if s == TERMINAL:
            rng += " terminal"
        rows.append([s, rng, f"{REWARDS[s]:+.1f}" if REWARDS[s] else "0.0",
                     f"{V_a[s]:.3f}", pol_a.get(s) if s != TERMINAL else "-",
                     f"{V_b[s]:.3f}", pol_b.get(s) if s != TERMINAL else "-",
                     visits[s]["w"], visits[s]["g"],
                     visits[s]["i1"], visits[s]["i2"]])
    _print_table(
        "Table 1 -- model, values and policies",
        ["state", "range (mmol/L)", "R(s)", "V^A(s)", "pi^A", "V^B(s)", "pi^B",
         "N(s,w)", "N(s,g)", "N(s,i1)", "N(s,i2)"],
        rows)

    _print_table("Table 2 -- evaluation on 500 fresh episodes",
                 ["policy", "mean return", "std", "B3 fraction",
                  "B1 termination", "mean length"],
                 [[k, f"{m['mean_return']:.2f}", f"{m['std_return']:.2f}",
                   f"{m['b3_fraction']:.4f}", f"{m['b1_termination_rate']:.4f}",
                   f"{m['mean_length']:.1f}"] for k, m in comparison.items()])

    _print_table("Table 3 -- data ablation",
                 ["episodes", "transitions", "A return", "B return",
                  "A B3", "B B3", "A B1", "B B1", "agreement"],
                 [[a["episodes"], a["transitions"],
                   f"{a['eval_a']['mean_return']:.2f}", f"{a['eval_b']['mean_return']:.2f}",
                   f"{a['eval_a']['b3_fraction']:.3f}", f"{a['eval_b']['b3_fraction']:.3f}",
                   f"{a['eval_a']['b1_termination_rate']:.3f}",
                   f"{a['eval_b']['b1_termination_rate']:.3f}",
                   f"{a['agreement']['n_same']}/8"] for a in ablation])

    _print_table("Table 4 -- online epsilon-greedy (5 seeds)",
                 ["setting", "last-200 return", "total regret"],
                 [[k, f"{v['final_mean_200']:.2f}", f"{v['regret_mean']:.0f}"]
                  for k, v in online.items()])

    _print_table("Table 5 -- robustness across dataset seeds",
                 ["dataset seed", "A return", "B return", "agreement",
                  "states where B differs"],
                 [[r["seed"], f"{r['eval_a']['mean_return']:.2f}",
                   f"{r['eval_b']['mean_return']:.2f}",
                   f"{r['agreement']['n_same']}/8",
                   ", ".join(s for s, ok in r["agreement"]["per_state"].items()
                             if not ok) or "-"] for r in robust])

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n[done] figures in {FIG_DIR}")
    print(f"[done] results.json written ({results['runtime_seconds']:.1f} s total)")
    return results


if __name__ == "__main__":
    main()
