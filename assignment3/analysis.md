# Analysis — BME1322 Assignment 3

*(Answers to the two short questions. Numbers refer to `report.md`, Tables 1–5 and
Figures 1–9; all of them are reproducible with `python assignment3/main.py`.)*

---

## Question 1 — Under this data budget, which route is more sensitive to sparsely visited (s, a) pairs: "learn the model, then plan", or "learn Q directly"? Why?

**Direct Q-learning is markedly more sensitive.** On the same 2 000-episode dataset,
Part A returns literally the same policy and the same value function at 500, 1 000 and
2 000 episodes (Table 3), and on five independent dataset draws the model-based policy
never changes and always scores 85.85 (Table 5), whereas the single-pass Q-learner flips
up to four of the eight non-terminal states and its evaluation return swings from 85.85
down to −11.81. The reason is what each route does with the few transitions it has for a
rare pair: the model-based agent averages *all* visits of that pair into one transition
distribution (with a uniform prior for pairs that were never visited) and then lets value
iteration propagate values globally with a contraction factor γ = 0.9, so single-transition
noise is divided by the visit count and smoothed over the whole state space, while
Q-learning touches each transition once with a constant α = 0.5, so the last few visits of
a rare pair dominate its Q value and the max operator immediately copies that noise into
every predecessor. Sparsity concentrates exactly in the rows where this matters — `B8` and
`B9` have 187–242 visits per action in total, and their greedy actions can be near-ties
(e.g. `B8`: i2 = −3.22 vs i1 = −6.12 in Part A, but i1 = −2.29 vs i2 = −2.86 in Part B on
the seed-999 draw).

---

## Question 2 — Find a state where the two routes disagree, and give a possible cause (model estimation error / insufficient sampling / different value-propagation path).

**State `B4` on dataset seed 7: Part A prescribes `i1`, Part B prescribes `i2`.** Part A's
Q row is $Q(B4,i1)=4.49 > Q(B4,i2)=3.88$, and it is right about the mechanism: from `B4`,
`i1` returns to the target band with probability 0.57 (otherwise it stays in `B4`), while
`i2` also drops into `B2` with probability 0.34, and `B2` is a −1 state with a ≈0.5/0.9
risk of falling into the terminal `B1`. The Q-learner reverses the ranking
($Q(i1)=3.60 < Q(i2)=4.45$) not because the data are lacking — `B4` is one of the
best-covered rows (≈1 000 visits per action) — but because of the **value-propagation
path**: its single pass with α = 0.5 leaves the successors systematically inflated
($V(B5)=4.56$ vs 4.02 in Part A, $V(B7)=2.11$ vs 0.17), and `i2` is the action that
bootstraps through those inflated values, so a 0.6-wide gap in favour of `i1` is crossed
after one pass. The same pattern explains the seed-999 disagreement in `B8`, where Part B
prefers `i1` because it lands in `B7` with probability 0.59 and Part B's `V(B7)` is 1.00
instead of 0.12 — a state with only 339 visits, so both **insufficient sampling of the
successor** and a **different value-propagation path** contribute, and the model-estimation
error of Part A is comparatively small because its counts are averaged over all visits.
