"""
BME1322 Assignment 3 - Part B: Tabular Q-Learning
=================================================

One-step route: skip the model entirely and learn the action-value function
directly from the (s, a, r, s', done) experience (Chapter 9).

Update rule (identical to the lecture):

    Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') ]

with the bootstrap term dropped when ``done`` is True (terminal transition).

The main task of the assignment is *offline batch* learning: ``learn()``
walks the given episodes in order and calls ``update()`` once per
transition.  No action is sampled there.  epsilon-greedy exploration is
only used by the additional *online* experiment required in the report;
``act()`` / ``decay_epsilon()`` implement it.

Public protocol (Assignment 3, "Q-Learning Protocol v1")
--------------------------------------------------------
    class QLearningAgent:
        __init__(states, actions, gamma=0.9, alpha=0.5, epsilon=0.3,
                 epsilon_decay=0.995, epsilon_min=0.05, seed=None)
        learn(episodes, mode="fixed")
        update(state, action, reward, next_state, done)
        get_q_value(state, action) -> float
        get_policy(state) -> action
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Transition = Tuple[str, str, float, str, bool]


class QLearningAgent:
    """Tabular Q-learning agent (offline batch learning + optional online)."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, states: Sequence[str], actions: Sequence[str],
                 gamma: float = 0.9, alpha: float = 0.5,
                 epsilon: float = 0.3, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05, seed: Optional[int] = None):
        """
        :param states:        state names, ordered.
        :param actions:       action names, ordered.  The order defines the
                              tie-breaking rule: on equal Q values the first
                              action in this list is returned.
        :param gamma:         discount factor.
        :param alpha:         constant step size of the Q-learning update.
        :param epsilon:       exploration rate (online experiment only).
        :param epsilon_decay: multiplicative decay per episode (online only).
        :param epsilon_min:   lower bound of epsilon (online only).
        :param seed:          RNG seed for epsilon-greedy action selection.
        """
        self.states: List[str] = list(states)
        self.actions: List[str] = list(actions)
        self.gamma: float = float(gamma)
        self.alpha: float = float(alpha)

        self.epsilon: float = float(epsilon)
        self.epsilon_decay: float = float(epsilon_decay)
        self.epsilon_min: float = float(epsilon_min)
        self.rng = random.Random(seed)

        # Q(s, a) initialised to 0 for every pair
        self.Q: Dict[str, Dict[str, float]] = {
            s: {a: 0.0 for a in self.actions} for s in self.states
        }

        # bookkeeping (report / debugging)
        self.n_updates: int = 0
        self.last_target: float = float("nan")
        self.last_td_error: float = float("nan")
        self.terminal_states: List[str] = []

    # ------------------------------------------------------------------ #
    # core update
    # ------------------------------------------------------------------ #
    def update(self, state: str, action: str, reward: float,
               next_state: str, done: bool) -> None:
        """One tabular Q-learning update.

            target = r                      if done
                   = r + gamma * max_a' Q(s', a')   otherwise
            Q(s,a) <- (1 - alpha) Q(s,a) + alpha * target
        """
        if state not in self.Q or action not in self.Q[state]:
            raise KeyError(f"unknown (state, action) = ({state!r}, {action!r})")

        if done:
            target = float(reward)
        else:
            q_next = self.Q.get(next_state)
            if q_next is None:                 # unseen successor: bootstrap 0
                bootstrap = 0.0
            else:
                bootstrap = max(q_next[a] for a in self.actions)
            target = float(reward) + self.gamma * bootstrap

        q_old = self.Q[state][action]
        self.Q[state][action] = (1.0 - self.alpha) * q_old + self.alpha * target

        self.n_updates += 1
        self.last_target = target
        self.last_td_error = target - q_old

    # ------------------------------------------------------------------ #
    # offline batch learning (main task)
    # ------------------------------------------------------------------ #
    def learn(self, episodes: Iterable[Sequence[Transition]],
              mode: str = "fixed") -> None:
        """Learn from a fixed batch of episodes, in the given order.

        Every episode is traversed transition by transition and each one is
        fed to :meth:`update`.  No action is sampled and no exploration rate
        is used: this is pure offline (batch) Q-learning on the shared data
        budget, which is what makes the comparison with Part A fair.

        Terminal states are detected from the data (a state that is entered by
        a ``done=True`` transition and never *continues* an episode).  Tuples
        that start in such a state are skipped: a terminal state has no
        actions, so it cannot contribute a Q value.  Such tuples only exist
        because the simulator may ``reset()`` straight into B1; see the
        report.

        :param episodes: list of episodes; an episode is a list of
                         (state, action, reward, next_state, done) tuples.
        :param mode:     kept for interface compatibility.  "fixed" is the
                         offline task; "online" exists only so that the
                         optional online experiment can share this method
                         (it does not sample actions either - see act()).
        """
        _ = mode  # documented above; the update rule is identical either way
        episodes = list(episodes)      # two passes below: materialise once
        self.terminal_states = self.detect_terminal_states(episodes)
        for episode in episodes:
            for (s, a, r, ns, done) in episode:
                if s in self.terminal_states:
                    continue
                self.update(s, a, r, ns, done)

    def detect_terminal_states(self,
                               episodes: Iterable[Sequence[Transition]]
                               ) -> List[str]:
        """States that end an episode and never continue one.

        ``done=True`` marks the *next* state as terminal; a state that appears
        as the ``state`` of a non-terminal transition (at position > 0, since
        ``reset()`` can start inside B1) is explicitly *not* terminal.
        """
        evidence = {s: False for s in self.states}
        continues = {s: False for s in self.states}
        for episode in episodes:
            for t, (s, _a, _r, ns, done) in enumerate(episode):
                if ns in evidence and done:
                    evidence[ns] = True
                if s in continues and not done and t > 0:
                    continues[s] = True
        self.terminal_states = [s for s in self.states
                                if evidence[s] and not continues[s]]
        return self.terminal_states

    # ------------------------------------------------------------------ #
    # epsilon-greedy behaviour (online experiment only)
    # ------------------------------------------------------------------ #
    def act(self, state: str) -> str:
        """epsilon-greedy action: explore uniformly w.p. epsilon, else greedy."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.get_policy(state)

    def decay_epsilon(self) -> float:
        """epsilon <- max(epsilon_min, epsilon * epsilon_decay)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    # ------------------------------------------------------------------ #
    # accessors required by the protocol
    # ------------------------------------------------------------------ #
    def get_q_value(self, state: str, action: str) -> float:
        """Current Q(state, action); 0.0 for an unknown pair."""
        return float(self.Q.get(state, {}).get(action, 0.0))

    def get_policy(self, state: str) -> str:
        """Greedy action of the current Q row (argmax, ties -> list order)."""
        row = self.Q.get(state)
        if not row:
            return self.actions[0]
        best_a = self.actions[0]
        best_v = row[best_a]
        for a in self.actions[1:]:          # strict '>' => first action wins
            if row[a] > best_v:
                best_v = row[a]
                best_a = a
        return best_a

    # ------------------------------------------------------------------ #
    # extras used by main.py
    # ------------------------------------------------------------------ #
    def get_q_table(self) -> Dict[str, Dict[str, float]]:
        return {s: dict(self.Q[s]) for s in self.states}

    def get_policy_table(self) -> Dict[str, str]:
        return {s: self.get_policy(s) for s in self.states}

    def get_state_value(self, state: str) -> float:
        """max_a Q(state, a) -- the implicit state value of the Q table."""
        row = self.Q.get(state)
        if not row:
            return 0.0
        return max(row[a] for a in self.actions)

    def reset(self) -> None:
        """Zero the Q table (useful for the multi-seed online experiment)."""
        self.Q = {s: {a: 0.0 for a in self.actions} for s in self.states}
        self.n_updates = 0
        self.terminal_states = []


if __name__ == "__main__":
    states = ["B1", "B2", "B3"]
    actions = ["w", "g"]
    episodes = [
        [("B2", "g", 0.5, "B3", False), ("B3", "w", 1.0, "B2", True)],
        [("B3", "w", 1.0, "B3", False), ("B3", "g", 0.5, "B2", True)],
    ]
    agent = QLearningAgent(states=states, actions=actions, gamma=0.9, alpha=0.5)
    agent.learn(episodes)
    print("Q table =", agent.get_q_table())
    print("policy  =", agent.get_policy_table())
