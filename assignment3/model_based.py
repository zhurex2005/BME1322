"""
BME1322 Assignment 3 - Part A: Model-Based RL
=============================================

Two-step route: (1) estimate the MDP model  P^(s'|s,a)  and  R^(s,a)
from a (fixed) batch of interaction experience with *counting* only, then
(2) solve the estimated MDP with synchronous value iteration and extract a
greedy policy from the Bellman optimality equation (Chapter 8).

Only the Python standard library is used; no RL / planning library is
called (assignment rule).  The transition table and reward table of the
provided simulator are never read: `learn_model` consumes nothing but the
(s, a, r, s', done) tuples it is given.

Public protocol (Assignment 3, "Model-Based RL Protocol v1")
-----------------------------------------------------------
    class ModelBasedAgent:
        __init__(states, actions, gamma=0.9, seed=None)
        learn_model(episodes)
        run_value_iteration(theta=1e-4, max_iterations=1000)
        get_transition(state, action) -> {next_state: prob}
        get_reward(state, action) -> float
        get_value(state) -> float
        get_policy(state) -> action
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Transition = Tuple[str, str, float, str, bool]


class ModelBasedAgent:
    """Model-based agent: count-based model estimation + value iteration."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, states: Sequence[str], actions: Sequence[str],
                 gamma: float = 0.9, seed: Optional[int] = None):
        """
        :param states:  state names, ordered (e.g. ['B1', ..., 'B9']).
        :param actions: action names, ordered.  The order defines the
                        tie-breaking rule of the greedy policy: on equal
                        Q values the *first* action in this list wins.
        :param gamma:   discount factor (0.9, as used in the lecture).
        :param seed:    unused (the agent itself has no stochasticity);
                        kept for interface symmetry with the grader.
        """
        self.states: List[str] = list(states)
        self.actions: List[str] = list(actions)
        self.gamma: float = float(gamma)
        self.seed = seed

        # ---- raw statistics collected in learn_model() ---------------- #
        # N[s][a][s'] = number of observed transitions s --a--> s'
        self._counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        # number of visits per (s, a)
        self._visit: Dict[str, Dict[str, int]] = {}
        # sum of rewards observed on (s, a) and their count
        self._reward_sum: Dict[str, Dict[str, float]] = {}
        self._reward_cnt: Dict[str, Dict[str, int]] = {}

        # ---- estimated model ------------------------------------------ #
        self.P: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.R: Dict[str, Dict[str, float]] = {}

        # ---- solution -------------------------------------------------- #
        self.V: Dict[str, float] = {s: 0.0 for s in self.states}
        self.policy: Dict[str, str] = {}
        self.Q: Dict[str, Dict[str, float]] = {}

        # ---- bookkeeping for the report -------------------------------- #
        self.n_episodes_seen: int = 0
        self.n_transitions_seen: int = 0
        self.n_skipped_terminal: int = 0
        self.terminal_states: List[str] = []
        self.vi_iterations: int = 0
        self.vi_delta: float = float("nan")
        self.vi_converged: bool = False

        self._ensure_tables()
        self.policy = self._initial_policy()

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _ensure_tables(self) -> None:
        """Allocate the nested dictionaries for every (s, a) pair."""
        for s in self.states:
            self._counts.setdefault(s, {})
            self._visit.setdefault(s, {})
            self._reward_sum.setdefault(s, {})
            self._reward_cnt.setdefault(s, {})
            self.P.setdefault(s, {})
            self.R.setdefault(s, {})
            self.Q.setdefault(s, {})
            for a in self.actions:
                self._counts[s].setdefault(a, {})
                self._visit[s].setdefault(a, 0)
                self._reward_sum[s].setdefault(a, 0.0)
                self._reward_cnt[s].setdefault(a, 0)
                self.P[s].setdefault(a, {})
                self.R[s].setdefault(a, 0.0)
                self.Q[s].setdefault(a, 0.0)

    def _initial_policy(self) -> Dict[str, str]:
        """Uniform (first-action) policy, used before any planning happened."""
        return {s: self.actions[0] for s in self.states}

    def _argmax_action(self, values: Dict[str, float]) -> str:
        """Greedy action w.r.t. ``values``; ties resolved by list order."""
        best_a = self.actions[0]
        best_v = values[best_a]
        for a in self.actions[1:]:          # strict '>' => first action wins
            if values[a] > best_v:
                best_v = values[a]
                best_a = a
        return best_a

    # ------------------------------------------------------------------ #
    # 1) model estimation  (Chapter 9: "learning a model from experience")
    # ------------------------------------------------------------------ #
    def learn_model(self, episodes: Iterable[Sequence[Transition]]) -> None:
        """Estimate P^(s'|s,a) and R^(s,a) by counting / averaging.

        P^(s'|s,a) = N(s,a,s') / N(s,a)           (normalised visit counts)
        R^(s,a)    = mean of the rewards observed on (s,a)

        Unvisited (s, a) pairs get a *uniform* transition distribution and
        R^ = 0 (an uninformative, maximally conservative prior; see the
        report for the justification).

        Terminal states are detected from the data: a state is terminal if it
        is entered by at least one done=True transition and never *continues*
        an episode, i.e. it never appears as the ``state`` of a non-terminal
        transition at position > 0 of an episode.  The position filter is
        needed because the simulator may ``reset()`` straight into B1: the
        agent must then still act once, which looks like an outgoing
        "transition" out of a terminal state.  A terminal state keeps V = 0
        and is never bootstrapped from.
        """
        self._ensure_tables()
        episodes = list(episodes)      # two passes below: materialise once

        # reset statistics so that learn_model() can be called repeatedly
        self.n_episodes_seen = 0
        self.n_transitions_seen = 0
        for s in self.states:
            for a in self.actions:
                self._counts[s][a] = {}
                self._visit[s][a] = 0
                self._reward_sum[s][a] = 0.0
                self._reward_cnt[s][a] = 0

        has_outgoing = {s: False for s in self.states}
        terminal_evidence = {s: False for s in self.states}
        n_trans = 0
        n_skipped = 0

        # --- pass 1: which states are terminal? ------------------------ #
        for episode in episodes:
            for t, (s, _a, _r, ns, done) in enumerate(episode):
                if ns not in self.states:
                    continue
                if not done:
                    # a state that *continues* an episode is not terminal;
                    # position 0 is ignored (see the docstring: reset() may
                    # start inside a terminal state).
                    if t > 0 and s in has_outgoing:
                        has_outgoing[s] = True
                else:
                    terminal_evidence[ns] = True

        self.terminal_states = [s for s in self.states
                                if terminal_evidence[s] and not has_outgoing[s]]

        # --- pass 2: count --------------------------------------------- #
        for episode in episodes:
            self.n_episodes_seen += 1
            for (s, a, r, ns, done) in episode:
                # ignore (s, a) that are outside the declared support
                if s not in self._counts or a not in self.actions:
                    continue
                if ns not in self.states:
                    continue
                if s in self.terminal_states:
                    # a terminal state has no actions: these tuples exist only
                    # because reset() may land in B1, and they cannot define a
                    # transition model (s, a) -> s'.
                    n_skipped += 1
                    continue
                n_trans += 1

                row = self._counts[s][a]
                row[ns] = row.get(ns, 0) + 1
                self._visit[s][a] += 1
                self._reward_sum[s][a] += float(r)
                self._reward_cnt[s][a] += 1

        self.n_transitions_seen += n_trans
        self.n_skipped_terminal = n_skipped

        self._normalise()
        # a fresh model invalidates the previous solution
        self.V = {s: 0.0 for s in self.states}
        self.Q = {s: {a: 0.0 for a in self.actions} for s in self.states}
        self.policy = self._initial_policy()
        self.vi_iterations = 0
        self.vi_converged = False

    def _normalise(self) -> None:
        """Turn the counts into probabilities and the reward sums into means."""
        uniform = 1.0 / float(len(self.states))
        for s in self.states:
            for a in self.actions:
                n = self._visit[s][a]
                if n > 0:
                    # count-based estimate, normalised by the visit count
                    self.P[s][a] = {ns: c / float(n)
                                    for ns, c in self._counts[s][a].items()}
                    self.R[s][a] = self._reward_sum[s][a] / float(n)
                else:
                    # never visited: uniform over states, zero reward
                    self.P[s][a] = {ns: uniform for ns in self.states}
                    self.R[s][a] = 0.0

    # ------------------------------------------------------------------ #
    # 2) planning  (Chapter 8: Bellman optimality equation)
    # ------------------------------------------------------------------ #
    def _is_terminal(self, s: str) -> bool:
        return s in self.terminal_states

    def _q_from_v(self, s: str, V: Dict[str, float]) -> Dict[str, float]:
        """Q(s,a) = R^(s,a) + gamma * sum_s' P^(s'|s,a) * V(s')."""
        out: Dict[str, float] = {}
        for a in self.actions:
            expected = 0.0
            for ns, p in self.P[s][a].items():
                v_ns = 0.0 if self._is_terminal(ns) else V.get(ns, 0.0)
                expected += p * v_ns
            out[a] = self.R[s][a] + self.gamma * expected
        return out

    def run_value_iteration(self, theta: float = 1e-4,
                            max_iterations: int = 1000) -> List[float]:
        """Synchronous (Jacobi-style) value iteration on the learned model.

            V_{k+1}(s) = max_a [ R^(s,a) + gamma * sum_s' P^(s'|s,a) V_k(s') ]

        Every state is updated from the *previous* sweep's values (a true
        synchronous sweep, not Gauss-Seidel).  Terminal states keep V = 0.
        Iteration stops when the sup-norm change satisfies
        ``max_s |V_{k+1}(s) - V_k(s)| < theta`` or after ``max_iterations``.

        :return: the history of the sup-norm change per iteration.
        """
        V: Dict[str, float] = {s: 0.0 for s in self.states}
        history: List[float] = []
        converged = False

        for it in range(1, int(max_iterations) + 1):
            V_new: Dict[str, float] = {}
            delta = 0.0
            for s in self.states:
                if self._is_terminal(s):
                    V_new[s] = 0.0          # absorbing: no future return
                    continue
                q = self._q_from_v(s, V)
                v = max(q[a] for a in self.actions)
                V_new[s] = v
                delta = max(delta, abs(v - V.get(s, 0.0)))
            V = V_new
            history.append(delta)
            if delta < theta:
                converged = True
                self.vi_iterations = it
                break

        if not converged:
            self.vi_iterations = int(max_iterations)
        self.vi_delta = history[-1] if history else float("nan")
        self.vi_converged = converged

        self.V = V
        # a terminal state has no actions: its Q row stays at 0
        self.Q = {s: ({a: 0.0 for a in self.actions} if self._is_terminal(s)
                  else self._q_from_v(s, V)) for s in self.states}
        self.policy = {s: self._argmax_action(self.Q[s]) for s in self.states}
        return history

    # ------------------------------------------------------------------ #
    # 3) accessors required by the protocol
    # ------------------------------------------------------------------ #
    def get_transition(self, state: str, action: str) -> Dict[str, float]:
        """Estimated {next_state: probability} for (state, action)."""
        return dict(self.P.get(state, {}).get(action, {}))

    def get_reward(self, state: str, action: str) -> float:
        """Estimated expected immediate reward R^(s,a)."""
        return float(self.R.get(state, {}).get(action, 0.0))

    def get_value(self, state: str) -> float:
        """V^(state) of the current (last) value-iteration solution."""
        return float(self.V.get(state, 0.0))

    def get_policy(self, state: str) -> str:
        """Greedy action argmax_a Q^(s,a); ties -> first action in the list."""
        if state in self.Q and self.Q[state]:
            return self._argmax_action(self.Q[state])
        return self.policy.get(state, self.actions[0])

    # ------------------------------------------------------------------ #
    # extras used by main.py (not part of the grader protocol)
    # ------------------------------------------------------------------ #
    def get_q_table(self) -> Dict[str, Dict[str, float]]:
        """Full Q^ table extracted from V^."""
        return {s: dict(self.Q[s]) for s in self.states}

    def get_visit_counts(self) -> Dict[str, Dict[str, int]]:
        """N(s,a) for every (s, a) -- needed for the sparsity report."""
        return {s: dict(self._visit[s]) for s in self.states}

    def get_policy_table(self) -> Dict[str, str]:
        return {s: self.get_policy(s) for s in self.states}

    def get_value_table(self) -> Dict[str, float]:
        return {s: self.get_value(s) for s in self.states}


if __name__ == "__main__":
    # tiny smoke test on a hand-made episode list
    states = ["B1", "B2", "B3"]
    actions = ["w", "g"]
    episodes = [
        [("B2", "g", 0.5, "B3", False), ("B3", "w", 1.0, "B2", True)],
        [("B3", "w", 1.0, "B3", False), ("B3", "g", 0.5, "B2", True)],
    ]
    agent = ModelBasedAgent(states=states, actions=actions, gamma=0.9)
    agent.learn_model(episodes)
    agent.run_value_iteration()
    print("P(B2,g) =", agent.get_transition("B2", "g"))
    print("R(B2,g) =", agent.get_reward("B2", "g"))
    print("V       =", agent.get_value_table())
    print("policy  =", agent.get_policy_table())
