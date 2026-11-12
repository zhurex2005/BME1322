# BME1322 Assignment 3 — Blood-Glucose Management: Model-Based Planning vs. Tabular Q-Learning



---

## 1. Overview and how to reproduce

This report compares two ways of turning a fixed batch of 2 000 random interaction
episodes into a blood-glucose controller:

* **Part A — model-based.** Estimate the transition model $\hat P(s'\mid s,a)$ and the
  reward model $\hat R(s,a)$ by counting, then solve the estimated MDP exactly with
  synchronous value iteration and read the greedy policy off the converged value
  function.
* **Part B — model-free.** Run tabular Q-learning *offline* over exactly the same
  batch, in one ordered pass, with no interaction with the simulator.

Both policies are then evaluated on the same 500 fresh episodes that were never used
for learning, and compared with each other, with four naive baselines, with smaller
data budgets, with the online (interactive) variant of Q-learning, and across five
independent dataset draws.

Everything in this report is produced by a single command, run from the directory that
contains `assignment3/`:

```bash
python assignment3/main.py          # writes assignment3/figures/*.png + results.json
python assignment-3/sanity_check.py # interface check provided with the assignment
```

`main.py` prints the five result tables reproduced below and takes ≈15 s on a laptop.
Figures are drawn with matplotlib when it is installed and with a small built-in
Pillow renderer otherwise, so the script has no hard plotting dependency
(`plotting.backend_name()` reports which one was used, and `main.py` prints it first).

`glucose_env.py` and `sanity_check.py` are **not** part of the submission: they are
supplied by the course and are therefore ignored in `.gitignore`.

### 1.1 Overall design — how the two routes share one pipeline

The two routes answer the same question ("what should the agent do in `B1…B9`?") from the
same input (a list of `(s, a, r, s', done)` tuples) but they differ in *where the
expectation lives*:

```text
  one dataset: 2 000 random episodes, 21 825 transitions (21 732 usable)
      |
      +--> Part A: model-based
      |      count N(s,a), N(s,a,s'), sum r
      |      -> P^(s'|s,a) (uniform prior if N(s,a)=0), R^(s,a)
      |      -> synchronous value iteration, theta = 1e-4
      |      -> Q^A(s,a) = sum_s' P^ [R^ + g V(s')]  ->  pi^A
      |
      +--> Part B: Q-learning
      |      one ordered pass, 21 732 updates
      |      Q <- (1-a) Q + a [r + g max_a' Q(s',a')],  a = 0.5, Q0 = 0
      |      -> Q^B(s,a)  ->  pi^B
      |
      +--> evaluate_policy(pi)  on 500 fresh episodes, identical seeds
             -> mean return | P(state = B3) | P(episode ends in B1)
```

* `model_based.py` and `q_learning.py` are independent, self-contained classes with the
  interface required by the handout (`get_transition`, `get_reward`, `get_value`,
  `get_policy`, `run_value_iteration`; `update`, `learn`, `get_q_value`, `get_policy`).
  Neither imports the other, and neither imports anything beyond the standard library.
* `main.py` is the only place where the two meet: it builds the dataset once, fits both
  agents on it, and pushes every policy through the *same* `evaluate_policy` function, so
  that all numbers in this report are computed by identical code. The extra experiments
  (baselines, ablation, online runs, robustness, convergence sweep) are all variations of
  that same loop, which is why the whole report reproduces in ≈15 s.
* `plotting.py` is a thin declarative layer (a `Figure` of `Panel`s, four panel kinds) so
  that the nine figures can be regenerated without repeating layout code.

Reproducibility choices worth flagging: every random draw is seeded (`2026` for the data,
`777000+i` for evaluation, `100000·s+11` for the online environments, fixed seeds for the
robustness sweep), all reported numbers come from `results.json`, and the two agents see
exactly the same list object.

---

## 2. Problem setup

| element | definition |
|---|---|
| state | 9 bins of blood glucose, `B1 = [2,4)`, `B2 = [4,6)`, …, `B9 = [18,20]` mmol/L |
| terminal | **B1** (severe hypoglycaemia): the episode ends on entering it and the agent never has to act there |
| actions | `w` (wait, cost 0.0), `g` (glucose, −0.5), `i1` (1 unit insulin, −0.2), `i2` (2 units insulin, −0.4) |
| reward | $r = R(s') + c(a)$ with $R = (-10, -1, +1, 0, -1, -2, -4, -6, -8)$ for `B1…B9` |
| horizon | 200 steps of 15 min (50 h), $\gamma = 0.9$ per step |
| dynamics | meal with prob. 0.12 (+2.4 mmol/L, decaying by 0.55/step), regression 0.06 toward 6.5 mmol/L, action shift $\{0, +1.2, -1.5, -2.8\}$, action noise 0.25, metabolic noise 0.30 |

**Data.** `collect_random_episodes(2000, seed=2026)` — 2 000 episodes, 21 825 transitions,
uniformly random actions. 93 of those transitions start in `B1` (a fraction of `reset()`
draws land in the hypoglycaemic bin); because `B1` is terminal they are *not* training
targets, which leaves **21 732 usable transitions**. Both parts use the same data and
the same convention:

> `B1` is absorbing/terminal: no action is ever taken from it, and the policy tables
> report `–` for `B1`. The policy-agreement rate is therefore computed over the **8
> non-terminal states**.

**Evaluation protocol** (`evaluate_policy` in `main.py`). 500 episodes with
`GlucoseEnv(seed=777000+i)`, following the policy deterministically from `reset()`;
episodes that start in `B1` count as a termination with return 0.0. Three metrics are
reported: the **undiscounted mean return** of the episode, the **fraction of steps spent
in `B3`** (the target band), and the **fraction of episodes that end in `B1`**
(severe-hypo termination rate).

---

## 3. Part A — model-based agent (`model_based.py`)

### 3.1 Estimating the model

`ModelBasedAgent.learn_model` makes two passes over the data. The first pass detects the
terminal states from the data itself (a state that is entered by a `done=True` transition
and never continues an episode at a later position — the position filter matters because
`reset()` sometimes starts an episode *in* `B1`); the second pass accumulates

$$N(s,a),\qquad N(s,a,s'),\qquad \hat R(s,a)=\frac{1}{N(s,a)}\sum_{t:\,(s_t,a_t)=(s,a)} r_t ,\qquad
\hat P(s'\mid s,a)=\frac{N(s,a,s')}{N(s,a)} .$$

Unvisited pairs (only the four `B1` pairs here) fall back to a uniform distribution over
the nine states and $\hat R = 0$. The reason for this convention is that a pair with no
evidence gives *no* information about its successor, and a uniform $\hat P$ is the
maximum-entropy choice consistent with that ignorance: it does not invent a preference for
any successor, it keeps every row a proper probability distribution (so the Bellman
operator stays well defined and value iteration's contraction argument applies to the
estimated MDP), and it makes the pair's $Q$ value simply the average of the values of all
successors instead of an arbitrary zero. The $\hat R=0$ half says the same thing about the
immediate reward: with no samples, claim no reward rather than a fabricated one. In this
particular dataset the convention is never exercised inside the policy, because the only
unvisited pairs belong to the terminal state `B1`, from which no action is ever taken.

![Figure 1](figures/fig01_visit_counts.png)

**Figure 1 — visit counts.** *(a)* $N(s,a)$ for all 36 pairs; *(b)* their distribution over
six bins. The coverage is strongly non-uniform: the two central bands collect most of the
data (`B3`: 1 233–1 348 visits per action), the high bands are thin (`B8`: 209–242,
`B9`: 187–215), and `B1` has no visits at all because it is terminal. At the full budget
all **32** non-terminal pairs have $N\ge 100$; at 500 episodes 16 of them are below 100
(Figure 4c), which is where the two methods start to disagree.

### 3.2 Value iteration

`run_value_iteration(theta=1e-4)` applies synchronous Bellman backups to **all** states
at once (no in-place updates), starting from $V_0\equiv 0$, with $V(\text{terminal})=0$
kept fixed. It stops when $\max_s |V_{k+1}(s)-V_k(s)|<\theta$ and returns the whole
history of $\Delta_k$.

![Figure 2](figures/fig02_value_iteration.png)

**Figure 2 — value iteration.** *(a)* the sup-norm change per sweep on a log axis: the
first sweep drops by an order of magnitude (the `B1` penalty is discovered), then the
curve decays geometrically, and the run stops after **82 sweeps** at
$\Delta = 9.01\times10^{-5}$ (dotted line = $\theta$). *(b)* the converged values.

### 3.3 Policy, values and the greedy Q table

The policy is extracted from $Q(s,a)=\sum_{s'}\hat P(s'\mid s,a)\left[\hat R(s,a)+\gamma V(s')\right]$
(Figure 3). Values are relative to the terminal state (`V(B1)=0`), so a state is "good"
when its value is large and positive.

![Figure 3](figures/fig03_qvalues_model.png)

**Figure 3 — $Q(s,a)$ of the estimated model.** The framed cell of each row is the greedy
action, i.e. the policy that Part A returns.

### Table 1 — learned model, values and policies

| state | range (mmol/L) | R(s) | V^A(s) | pi^A | V^B(s) | pi^B | N(s,w) | N(s,g) | N(s,i1) | N(s,i2) |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | [2, 4) terminal | -10.0 | 0.000 | - | 0.000 | - | 0 | 0 | 0 | 0 |
| B2 | [4, 6) | -1.0 | 3.967 | g | 6.317 | g | 1086 | 992 | 1052 | 1034 |
| B3 | [6, 8) | +1.0 | 4.768 | w | 7.513 | w | 1332 | 1289 | 1233 | 1348 |
| B4 | [8, 10) | 0.0 | 4.388 | i1 | 6.624 | i1 | 1065 | 1062 | 990 | 1029 |
| B5 | [10, 12) | -1.0 | 3.712 | i2 | 6.738 | i2 | 736 | 789 | 735 | 723 |
| B6 | [12, 14) | -2.0 | 2.225 | i2 | 3.099 | i2 | 555 | 512 | 548 | 530 |
| B7 | [14, 16) | -4.0 | 0.064 | i2 | 0.138 | i2 | 339 | 376 | 348 | 320 |
| B8 | [16, 18) | -6.0 | -3.601 | i2 | -3.299 | i2 | 240 | 220 | 209 | 242 |
| B9 | [18, 20) | -8.0 | -9.961 | i2 | -8.217 | i2 | 187 | 215 | 201 | 195 |

The resulting controller is a bang-bang regulator around the target band:

* **`B2` → `g`.** Below the target the only sensible move is food: `g` shifts glucose by
  +1.2 mmol/L and reaches `B3` with probability 0.64, so $Q(B2,g)=3.97$ against
  $Q(B2,w)=2.82$; insulin from `B2` sends the patient into `B1` with probability 0.53
  (`i1`) or 0.89 (`i2`) and is correctly priced at $-4.15$ / $-9.01$.
* **`B3` → `w`.** Inside the target band every step pays $+1$ and costs nothing, so the
  optimal action is the one that keeps the state there *and* has zero cost;
  $Q(B3,w)=4.77$ beats $i1$ (3.29: costs −0.2 and leaves the band 61 % of the time,
  mostly to `B2`) and $i2$ (−1.40: 31 % chance of `B1` per step).
* **`B4` → `i1`, `B5…B9` → `i2`.** Above the band the dose grows with the distance to the
  target: one unit of insulin from `B4` returns to `B3` with probability 0.56 while the
  larger $i2$ dose costs more (−0.4) and lands in `B2` 34 % of the time; at `B7…B9` the
  per-step penalty (−4…−8) is so large that the aggressive dose is worth the
  hypoglycaemia risk.

### 3.4 Evaluation

### Table 2 — evaluation on 500 fresh episodes (≈97 000 decisions per policy)

| policy | mean return | std | B3 fraction | B1 termination | mean length |
|---|---|---|---|---|---|
| model (A) | 85.85 | 23.18 | 0.6328 | 0.0320 | 193.6 |
| Q-learn (B) | 85.85 | 23.18 | 0.6328 | 0.0320 | 193.6 |
| all w | -809.34 | 252.63 | 0.0376 | 0.0480 | 190.5 |
| all g | -1602.48 | 292.23 | 0.0030 | 0.0320 | 193.6 |
| all i1 | -11.26 | 3.90 | 0.2340 | 1.0000 | 3.9 |
| all i2 | -10.80 | 2.18 | 0.1115 | 1.0000 | 1.8 |

The learned policy spends 63 % of its time in the target band and dies of hypoglycaemia
in 3.2 % of episodes, whereas every naive baseline is catastrophic: `all w`/`all g` drift
upwards and burn −809/−1602 per episode, while a fixed insulin dose kills the patient
within four steps (100 % `B1` termination, return ≈ −11). The 3.2 % `B1` rate of the
learned policy comes entirely from episodes that *start* in `B1` (16 of the 500 evaluation
episodes), not from decisions taken by the agent.

### 3.5 Data ablation (Part A)

![Figure 4](figures/fig04_data_ablation.png)

**Figure 4 — data budget.** *(a)* mean return, *(b)* `B3` fraction of the two policies and
*(c)* the number of the 36 pairs whose count is below 100, as a function of the number of
training episodes.

### Table 3 — data ablation

| episodes | transitions in data | A return | B return | A B3 | B B3 | A B1 | B B1 | agreement |
|---|---|---|---|---|---|---|---|---|
| 500 | 5336 | 85.85 | 85.85 | 0.633 | 0.633 | 0.032 | 0.032 | 7/8 |
| 1000 | 11140 | 85.85 | 85.83 | 0.633 | 0.633 | 0.032 | 0.032 | 7/8 |
| 2000 | 21825 | 85.85 | 85.85 | 0.633 | 0.633 | 0.032 | 0.032 | 8/8 |

(`transitions` counts every tuple of the drawn data; the 93 tuples that start in `B1` are
excluded from *training* in both parts, as explained in Section 2.)

Part A is **completely insensitive** to the budget in this range: the same policy, the
same value function to three decimals, and the same evaluation numbers at 500, 1 000 and
2 000 episodes. The estimated transition probabilities of the *visited* pairs are already
accurate with 500 episodes, and the 16 pairs that are still sparse at 500 episodes are
all in the extreme high bands (`B7`: 81–91 visits, `B8`: 51–63, `B9`: 46–59) or in `B1`,
which is terminal and never used — exactly the rows where the action ordering is already
robust (`i2` wins by a wide margin).

---

## 4. Part B — tabular Q-learning (`q_learning.py`)

### 4.1 The offline update

`QLearningAgent.learn(data, mode="fixed")` makes **one ordered pass** over the 2 000
episodes, in the order in which the transitions appear in the dataset, and applies

$$Q(s,a)\leftarrow(1-\alpha)\,Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')\right],\qquad \alpha=0.5,$$

with $Q_0\equiv0$ and no bootstrap term when `done=True`. There is no action sampling: the
action is taken from the data, only the *successor value* is maximised. That makes the
pass a sequence of 21 732 TD updates — about 680 per non-terminal pair, i.e. a single
sweep of a method that needs many sweeps to converge — which is exactly why the result
below is a *one-pass approximation* of $Q^*$ rather than $Q^*$ itself.

### 4.2 Q table and policy

![Figure 5](figures/fig05_qvalues_qlearning.png)

**Figure 5 — $Q(s,a)$ after the single pass.** Framed cells = greedy action.

The greedy policy is **identical to Part A on all 8 non-terminal states** (Table 1 right
half), and the evaluation is therefore numerically identical (Table 2). The *values*,
however, are not: they are systematically larger in magnitude and inflated by a
state-dependent constant (e.g. $V^B(B3)=7.51$ vs $V^A(B3)=4.77$, $V^B(B7)=0.14$ vs
$V^A(B7)=0.06$, $V^B(B8)=-3.30$ vs $V^A(B8)=-3.60$). With a constant step size the
iterates converge to the fixed point of the *empirically averaged* Bellman operator
visited in this particular data order, and that fixed point is not $Q^*$; the offset then
grows with the bootstrap chain because every update inherits the bias of its successors.
Section 6 turns this observation into a controlled experiment.

### 4.3 Online ε-greedy experiment (interactive regret)

The offline pass is one thing; the assignment also asks what happens when Q-learning is
allowed to *act*. Five independent runs (env seeds `100000·s+11`) are trained for 2 000
episodes with ε-greedy action selection, ε decayed once per episode, and scored with a
50-episode sliding window of the undiscounted return:

* **fixed** ε = 0.3
* **decay** ε ← max(0.05, 0.995·ε), starting at 0.3

![Figure 6](figures/fig06_online_learning.png)

**Figure 6 — online ε-greedy Q-learning.** Thin lines = individual seeds, thick line =
seed mean, 50-episode sliding window. *(a)* fixed ε = 0.3, *(b)* decaying ε, *(c)* mean
return of the last 200 episodes.

### Table 4 — online ε-greedy, 5 seeds

| setting | last-200 return | total regret |
|---|---|---|
| fixed ε = 0.3 | -11.19 | 194 017 |
| decay 0.3 → 0.05 | +32.32 | 115 248 |

The fixed-ε run **collapses**: by episode ~300 the greedy policy has drifted to `i2` in
`B4…B9` (e.g. $Q(B4,i2)=3.35>Q(B4,i1)=2.27$) and the patient is killed in ≈99 % of the
training episodes, so the windowed return sits at −11 for the whole run — the agent
converges to a *confident but lethal* policy instead of exploring its way out. The decayed
run keeps exploring long enough to find the food action (`g`) in the low bands and settles
at +32 per episode, still far below the ≈86 of the offline policies: online learning pays
a large **regret** for having to discover the dynamics while acting. Regret here is the
pseudo-regret of Chapter 9,
$\text{Regret}(T)=\sum_{t=1}^{T}\left(g^*-G_t\right)$ with $g^*=85.85$ — the per-episode
return of the model-based policy, i.e. the best return available from the same environment —
and $G_t$ the return actually collected in episode $t$ (negative terms make it much larger
than a conventional regret): ≈1.15 × 10⁵ reward units for the decayed run, ≈1.94 × 10⁵ for
the fixed one, i.e. the fixed-ε agent gives up more than an entire optimal episode *per
episode*. This is the Chapter 9 trade-off between exploration and exploitation in its most
brutal form — here a bad exploration policy is not merely suboptimal, it is fatal.

---

## 5. Part A vs. Part B

### 5.1 Policy agreement

Agreement is computed per non-terminal state as `pi^A(s) == pi^B(s)`:

| state | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 |
|---|---|---|---|---|---|---|---|---|
| pi^A | g | w | i1 | i2 | i2 | i2 | i2 | i2 |
| pi^B | g | w | i1 | i2 | i2 | i2 | i2 | i2 |
| agree | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Agreement rate: 8/8 = 100 %** on the main dataset, 7/8 at 500 and 1 000 episodes
(Table 3), and 0.80 averaged over five independent dataset draws (Table 5: 8, 7, 7, 4, 6
of 8).

### 5.2 Performance comparison and explanation

![Figure 7](figures/fig07_ab_comparison.png)

**Figure 7 — the two learned policies against the four naive baselines.** *(a)* the two
learned policies turn the −809 / −1602 of `all w` / `all g` and the −11 of a constant
insulin dose into +86; *(b)* they are the only policies that spend most of their time in
the target band; *(c)* they are the only ones that do not kill the patient.

On the main dataset the two policies are indistinguishable (return 85.85, `B3` 0.633,
`B1` 0.032, identical to two decimals). That is not a coincidence: the two methods agree
on every state, and the evaluation is deterministic given the policy, so **the same policy
must give the same numbers**. The interesting question is therefore not "which one scores
higher on this draw" but "which one is stable when the draw changes" — and there the
difference is stark.

![Figure 8](figures/fig08_robustness.png)

**Figure 8 — stability across five dataset seeds** (same evaluation seeds for every run).

### Table 5 — robustness across dataset seeds

| dataset seed | A return | B return | agreement | states where B differs |
|---|---|---|---|---|
| 2026 | 85.85 | 85.85 | 8/8 | - |
| 7 | 85.85 | 79.85 | 7/8 | B4 |
| 123 | 85.85 | 69.06 | 7/8 | B5 |
| 999 | 85.85 | -11.81 | 4/8 | B2, B4, B6, B8 |
| 31337 | 85.85 | 46.85 | 6/8 | B2, B5 |

**Part A returns exactly the same policy — and therefore the same score — on all five
draws**, while Part B's return ranges from −11.81 to 85.85 and its agreement from 4/8 to
8/8. The reason is structural:

* Part A pools *all* visits of a pair before it does anything: $\hat P$ averages thousands
  of noisy transitions, unvisited pairs fall back to a uniform prior, and value iteration
  then propagates the values globally with a contraction factor $\gamma=0.9$, so the noise
  of any single transition is divided by the number of visits and then smoothed over the
  whole state space.
* Part B touches each transition once, with a constant $\alpha=0.5$, so the last few
  visits of a rare pair dominate its Q value, and the max operator immediately propagates
  an inflated value into every predecessor. Rows with a few hundred visits (`B7…B9`) and
  near-ties such as `B4`: $i1$ vs $i2$ (gap 0.6 in Part A) are exactly where a small
  perturbation flips the argmax.

### 5.3 Answer to the two analysis questions

The full answers are in `analysis.md`; in short: *(1)* direct Q-learning is more sensitive
to sparsely visited pairs, because each sparse pair is represented by a handful of
single-sample TD updates dominated by a constant step size, while the model-based route
averages all visits of a pair and then smooths through global value iteration; *(2)* the
seed-7 disagreement is again a `B4` case — Part A gives $Q(i1)=4.49>Q(i2)=3.88$ while Part
B gives $Q(i1)=3.60<Q(i2)=4.45$ — and since that pair is visited ≈1 000 times per action it
is *not* a sampling problem but a value-propagation one: Part B's inflated $V(B5)=4.56$
(vs. 4.02 in Part A) is exactly what `i2` bootstraps from. The remaining disagreements
(worst on seed 999, 4 of 8 states) mix both mechanisms, with the sparse high bands `B7…B9`
contributing the sampling part.

---

## 6. Chapter 9 — the convergence conditions, tested

Chapter 9 states that tabular Q-learning converges to $Q^*$ if **(i)** every state–action
pair is visited *sufficiently often*, **(ii)** the step sizes satisfy
$\sum_k\alpha_k=\infty$ and $\sum_k\alpha_k^2<\infty$ (a Robbins–Monro schedule), and
**(iii)** $\gamma<1$ with bounded rewards, so that the Bellman operator is a contraction.
The experiment below attacks (ii) directly while holding (i) and (iii) fixed: it re-uses
the same 2 000-episode dataset, whose 32 non-terminal pairs all have between 187 and
1 348 visits (Table 1), and the same environment ($\gamma=0.9$, $r\in[-10.5,\,1]$).

Define the distance to the model-based solution
$\lVert Q-Q^A\rVert_\infty=\max_{(s,a)}\lvert Q(s,a)-Q^A(s,a)\rvert$ over the 8
non-terminal states.

![Figure 9](figures/fig09_convergence.png)

**Figure 9 — step size and convergence.** *(a)* one pass over the data, different constant
step sizes; *(b)* repeated passes with a constant step size (flat) versus a Robbins–Monro
step size $\alpha_k=0.5/(1+k)$ (decreasing); *(c)* evaluation return of the corresponding
policies (dotted line = Part A, 85.85).

| step size α (one pass) | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|---|
| ‖Q − Q^A‖∞ | 1.49 | 2.10 | 3.03 | 3.92 | 5.85 | 8.15 | 10.53 |
| agreement | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 7/8 (`B4`) |
| return | 85.85 | 85.85 | 85.85 | 85.85 | 85.85 | 85.85 | 79.85 |

| repeated passes | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ‖Q − Q^A‖∞, α = 0.9 constant | 10.53 | 10.53 | 10.53 | 10.53 | 10.53 | 10.53 | 10.53 | 10.53 |
| ‖Q − Q^A‖∞, α = 0.5 constant | 5.85 | 5.85 | 5.85 | 5.85 | 5.85 | 5.85 | 5.85 | 5.85 |
| ‖Q − Q^A‖∞, α_k = 0.5/(1+k) | 5.85 | 3.47 | 2.73 | 2.28 | 2.10 | 1.91 | 1.72 | 1.57 |

Interpretation, mapped onto the three conditions:

1. **Condition (ii) — the learning rate — is the binding one here.** With a *constant* step
   size the error is a monotone function of α (0.05 → 1.49, 0.9 → 10.53) and repeating the
   pass does **not** reduce it at all: the process converges to the fixed point of the
   averaged empirical Bellman operator, which is *not* $Q^*$. With $\alpha_k=0.5/(1+k)$ —
   which satisfies $\sum_k\alpha_k=\infty$ and $\sum_k\alpha_k^2<\infty$ — the distance
   keeps shrinking (5.85 → 1.57 and still decreasing), exactly as the theorem predicts. The
   price is speed: a $1/k$ schedule needs many passes, which is why the assignment's single
   ordered pass with α = 0.5 is a *one-pass approximation* rather than a converged $Q^*$.
2. **Condition (i) — sufficient visits — is satisfied offline, but not by an online
   agent.** The offline pass visits every non-terminal pair at least 100 times, which is
   why even α = 0.9 still recovers 7/8 of the policy. The online experiment of Figure 6 is
   the opposite case: the ε-greedy policy stops visiting the states it believes are bad
   (the fixed-ε run ends ~99 % of its episodes in `B1`), condition (i) fails in the limit,
   and the run locks onto a lethal policy.
3. **Condition (iii) — γ < 1 with bounded rewards — holds by construction and is visible in
   Figure 2.** With $\gamma=0.9$ and $r\in[-10.5,1]$ the Bellman operator is a
   γ-contraction, so value iteration must converge geometrically at a rate no worse than
   γ: the measured decay of $\Delta_k$ in Figure 2a is a factor ≈0.87 per sweep, just under
   the 0.9 bound, and 82 sweeps bring the error from ≈9 to $9\times10^{-5}$
   ($0.87^{81}\approx 10^{-5}$). Bounded rewards are what make the *scale* of the error
   finite as well: without them the sup-norm $\lVert Q-Q^A\rVert_\infty$ used above would
   not even be defined.
4. **Why the policy is so much more robust than the values.** At α = 0.5 the Q table is off
   by 5.85 in the sup-norm, yet the evaluation return is bit-identical to Part A's. The
   greedy action only depends on the *ordering* within a row, and the margins between the
   best and the second-best action are usually much larger than the α-dependent offset
   (which is nearly common to a whole row); only near-ties such as `B4` (`i1` vs `i2`,
   gap ≈0.6) are close enough for the offset to flip the argmax — which is exactly the state
   where α = 0.9 disagrees.

---

## 7. Discussion and reflection

**Biggest difficulty.** The environment is deceptively small: 9 states, 4 actions, so both
algorithms are a few dozen lines. Almost all of the work went into the *bookkeeping around*
the algorithms, and each item changed a number in this report:

* *Terminal-state semantics.* `reset()` sometimes returns `B1`, so a naive "done=True ⇒
  terminal" reading declares `B1` non-terminal and lets value iteration assign it a
  positive value ($V(B1)=1.76$ in the first version, with a phantom action that was never
  taken). The fix — a state is terminal only if it is entered by a `done=True` transition
  *and* never continues an episode at a later position — removes 93 tuples from training
  and changes both the value function and the policy tables.
* *Evaluation protocol.* The three required metrics are easy to compute but easy to compute
  *wrongly*: the return must be undiscounted, `B3` is a fraction of *steps*, and `B1`
  termination is a fraction of *episodes* that must include the ~3 % of episodes that start
  in `B1`. All three are implemented in one function (`evaluate_policy`) used by every
  experiment so that the comparison is apples-to-apples.
* *Reproducibility of the figures.* No matplotlib was available in the environment where
  this was developed, so the figure layer had to be written twice (a declarative spec object
  plus two renderers); the figures in this report are the Pillow ones.

**When is the model-based route worth it, and when is direct Q-learning safer?**

* Model-based wins when the data are *off-policy and reusable*: a single batch supports
  counterfactual questions ("what would `i2` have done here?") because $\hat P$ is a
  distribution, and the values are solved globally instead of being stitched together
  transition by transition. It also degrades gracefully with little data, as the ablation
  shows, because counting averages and the uniform prior regularise unvisited pairs.
* Direct Q-learning wins when the state space is too large to tabulate a model (every
  `(s,a)` would need its own distribution), when the environment is non-stationary (a
  model estimated once is stale, while Q-learning can keep updating), or when only the
  policy matters and the values are just a means — as here, where a Q table that is wrong
  by 5.85 still gives the optimal policy.
* Direct Q-learning is *not* safer with a small batch and a constant step size: it is the
  route that breaks (Table 5). The safe statement is: **model-based is more sample-efficient
  and more stable per unit of data; model-free is more flexible per unit of computation.**

**What I would do next.**

1. **Function approximation (Chapter 10).** The 9-bin discretisation throws away the
   ordering of the glucose scale: `B4` and `B5` are neighbours but share no statistics. A
   linear or tile-coded $Q(s,a)$ (or a small neural network) over the continuous glucose
   value would let rare high bins borrow from their neighbours and would probably fix the
   `B8`/`B9` fragility of Part B.
2. **Better use of the buffer.** The offline Q-learner visits each transition exactly once,
   so its result depends on the data order. Replaying the buffer with a Robbins–Monro
   schedule (or with a small experience-replay pass per epoch) would approach $Q^*$ without
   new data — Figure 9b shows the gap closing from 5.85 to 1.57 in eight passes.
3. **Safer exploration.** The online experiment shows that ε-greedy with a fixed ε can
   converge to a lethal policy; an optimistic initialisation or a count-based bonus
   (which is exactly the model Part A already estimates) would keep visiting the high bands.
4. **A risk-sensitive objective.** The −10 terminal reward dominates the return; optimising
   a CVaR or a constrained MDP ("keep $P(B1)$ below 1 %") would change the `B7…B9` policy
   from "aggressive dose" to something more conservative, which is what a clinician would
   actually ask for.

---

## 8. Conclusions

1. With a 2 000-episode random dataset both routes recover the same, near-optimal
   bang-bang controller (85.85 mean return, 63 % of the time in the target band, 3.2 %
   severe-hypo terminations, versus −11 and 100 % for any fixed insulin dose).
2. The model-based route is dramatically more **stable**: identical policy on 500/1 000/2 000
   episodes and on all five dataset seeds, whereas the single-pass Q-learner flips up to
   four states and loses up to 97 reward units per episode on an unlucky draw.
3. A single ordered pass of Q-learning is *not* $Q^*$: its values are inflated by the
   constant step size, and only a Robbins–Monro schedule makes the distance to the
   model-based solution decrease (Figure 9). The policy, however, is much less sensitive
   than the values.
4. Online ε-greedy learning from scratch is far worse than either offline route
   (+32 vs 85.85 per episode) and, with a fixed ε, can converge to a lethal policy — the
   clearest illustration in this assignment of the exploration/exploitation trade-off.

---

## Appendix — files

| file | contents |
|---|---|
| `model_based.py` | `ModelBasedAgent`: counting model, terminal detection, synchronous value iteration, greedy policy extraction |
| `q_learning.py` | `QLearningAgent`: ordered offline pass, ε-greedy interaction, ε decay, policy extraction |
| `main.py` | experiment driver: data, both agents, baselines, ablation, online runs, robustness, convergence sweep, figures, tables, `results.json` |
| `plotting.py` | small declarative figure layer (matplotlib backend, dependency-free Pillow fallback) |
| `figures/` | Figures 1–9 of this report |
| `results.json` | every number quoted above, machine-readable |
| `analysis.md` | the two short-answer questions |

**References.** R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
2nd ed., MIT Press, 2018 (Chapter 9 for the convergence conditions and the
exploration/regret discussion); course notes BME1322, Assignment 3 handout.
