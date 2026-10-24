# Handwritten Digit Classification — Assignment 2 Report

> English translation of [report.md](report.md).

## 1. Overall Design

This assignment implements two probabilistic generative models on the MNIST handwritten-digit dataset and compares their classification performance:

1. **Naive Bayes classifier (NB)**: every pixel of the 28×28 image is treated as a discrete random variable, and a generative model is built under the assumption that pixels are conditionally independent given the class.
2. **Bayesian network classifier (BN)**: 12 discrete features with clear physical meaning are extracted from each image, a directed acyclic graph (DAG) rooted at the class `Y` is designed by hand to encode dependencies among features, conditional probability tables (CPTs) are estimated by maximum likelihood with Laplace smoothing, and exact inference is performed.

Both models follow the unified interface required by the assignment:

```python
clf = NaiveBayesClassifier(alpha=1.0)   # or BayesNetClassifier(alpha=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

`bayes_net.py` additionally provides a module-level function `extract_features(X)` that maps raw `(n, 784)` images to an `(n, 12)` discrete feature matrix. All models are implemented from scratch and rely only on NumPy/SciPy for numerical operations and image filtering; no off-the-shelf classifiers or structure-learning libraries such as sklearn, pgmpy, or bnlearn are used.

The data are the official split under `assignment2-data/mnist`: 60,000 training images and 10,000 test images. `main.py` performs training, testing, metric computation, and generation of the confusion matrices and report figures in a single run:

```bash
pip install -r requirements.txt
python main.py
```

Running it produces `metrics.json` (all metrics) and the six figures under `figures/`. All numbers in this report come from this script's output on the full test set.

---

## 2. Naive Bayes Classifier

### 2.1 Model and Formulas

Let the image pixels be $x_1,\dots,x_{784}$ and the class be $y\in\{0,\dots,9\}$. By Bayes' rule,

$$
P(y\mid x_1,\dots,x_{784}) = \frac{P(y)\,P(x_1,\dots,x_{784}\mid y)}{P(x_1,\dots,x_{784})}.
$$

Estimating the 784-dimensional joint likelihood directly would require $10\times 256^{784}$ parameters, which is infeasible. Naive Bayes introduces the **conditional independence assumption**: given the class $y$, the pixels are mutually independent, i.e.

$$
P(x_1,\dots,x_{784}\mid y)=\prod_{j=1}^{784}P(x_j\mid y).
$$

The posterior is therefore proportional to

$$
P(y)\prod_{j=1}^{784}P(x_j\mid y).
$$

For classification we only need to compare this product (the denominator is independent of $y$ and can be ignored). To avoid underflow from multiplying many probabilities, all computations are carried out in log space:

$$
\log P(y\mid x)=\log P(y)+\sum_{j=1}^{784}\log P(x_j\mid y)+\text{const}.
$$

### 2.2 Parameter Estimation and Smoothing

Both the priors and the conditionals are estimated with **Laplace smoothing (add-$\alpha$)**. Let $N_{yjb}$ be the number of samples of class $y$ whose $j$-th pixel falls into bin $b$, let $N_{yj}=\sum_b N_{yjb}$, and let $B$ be the number of bins. Then

$$
\hat P(x_j=b\mid y)=\frac{N_{yjb}+\alpha}{N_{yj}+\alpha B},\qquad
\hat P(y)=\frac{N_y+\alpha}{N+\alpha K},
$$

where $K=10$ is the number of classes and $N$ is the number of training samples. The default is $\alpha=1$ (i.e. Laplace smoothing), which guarantees a non-zero probability for every pixel–class combination unseen in training, so $\log 0$ never occurs.

### 2.3 Pixel Discretization

Raw pixels take values in 0–255. Building a table over 256 grey levels would require many parameters and many levels have very few samples; binarizing alone would discard stroke-intensity information. This experiment quantizes the grey scale into 3 bins:

| Bin | Grey range | Meaning |
|---|---|---|
| 0 | 0 | Background |
| 1 | 1–127 | Light stroke |
| 2 | 128–255 | Dark stroke |

We split the training set into 50,000/10,000 for validation and compared several discretization schemes (validation accuracy):

| Scheme | Validation accuracy |
|---|---|
| Binarization (threshold 128) | 0.8321 |
| 3 bins, $\alpha=1$ | 0.8345 |
| 3 bins, $\alpha=0.1$ | 0.8355 |
| 4 bins | 0.8353 |
| 8 bins | 0.8336 |
| Multinomial model (raw grey levels) | 0.8184 |

Three bins already express "background / light stroke / dark stroke"; further increasing the number of bins brings little gain while increasing the number of parameters to estimate per class. We finally choose 3 bins and $\alpha=1.0$, which matches the interface default and is close to optimal on the validation set.

### 2.4 Implementation Notes

- `fit` uses a single `np.bincount` call to build a count tensor of shape `(10, 784, 3)`, avoiding Python loops.
- `predict` computes `log_prior[k] + Σ_j log_likelihood[k, j, x_j]` for each class $k$ with fancy indexing in one pass; predicting 10,000 test images takes about 0.8 s.
- An additional `predict_proba` applies a softmax to the joint log-likelihood, which helps analyse the confidence of misclassified samples.

---

## 3. Bayesian Network Classifier

### 3.1 Feature Design

The expressive power of a Bayesian network depends on whether its features are both discriminative and compact. The final feature set contains **12** features, each taking 3–5 values, satisfying the requirement of "6–12 features, each with 2–5 values."

**(1) 10 local stroke-orientation features.** The 28×28 image is divided into **5 horizontal bands × 2 left/right halves**, i.e. 10 cells. For each cell:

1. Compute the gradient magnitude $m$ and orientation $\theta=\operatorname{atan2}(g_y,g_x)\bmod\pi$ with the Sobel operator;
2. Quantize the orientation into 4 intervals: horizontal H (near $0^\circ$), rising diagonal D+ ($45^\circ$), vertical V ($90^\circ$), and falling diagonal D− ($135^\circ$);
3. Cast a gradient-magnitude-weighted vote inside the cell and take the orientation with the most votes as the cell's dominant orientation;
4. If the fraction of ink pixels in the cell is below 0.02, encode the cell as "empty."

| Code | Meaning |
|---|---|
| 0 | Empty (almost no ink) |
| 1 | Horizontal stroke H |
| 2 | Rising diagonal D+ |
| 3 | Vertical stroke V |
| 4 | Falling diagonal D− |

**(2) Aspect-ratio feature.** Compute the height-to-width ratio $h/w$ of the digit's bounding box and discretize it into 3 levels using the training-set terciles $1.176$ and $1.538$ (wider than tall, normal, taller than tall). It separates slender digits such as "1" and "7" from wider digits such as "0" and "8."

**(3) Hole-count feature.** Label the background with 4-connectivity and count the connected background components that do not touch the image border, i.e. the number of enclosed regions inside the digit. The values are 0, 1, and 2 (2 means two or more). It directly captures the topology of the digit: 0/6/9 have one hole, 8 has two, and 1/2/3/4/5/7 usually have none.

The figure `figures/feature_grid.png` shows the 5×2 grid and the corresponding feature codes for one sample of each digit 0–9.

![Feature grid illustration](figures/feature_grid.png)

### 3.2 Network Structure

With the class variable $Y$ as root, the DAG is as follows (see `figures/bn_dag.png`):

![Bayesian network structure](figures/bn_dag.png)

- **12 $Y\to$ feature edges**: the class determines which stroke orientation appears in each cell, how wide the digit is, and how many holes it has. These carry the main discriminative information.
- **2 feature→feature edges**:
  - `c00 → c01`: left half of the top band → right half;
  - `c40 → c41`: left half of the bottom band → right half.

  These two edges encode the statistical dependence between the stroke orientations of the two halves of the same band. For example, when writing "0," the top arc extends from the left half to the right half and both sides are horizontal/diagonal; when writing "8," the left and right sides of the two loops are also highly correlated. Forcing left/right independence (i.e. a pure naive Bayes structure) would make the model underestimate this consistency.

Every feature node has at most one parent (satisfying "at most two parents per feature node"); there are no hidden variables; the graph is acyclic, and edge directions follow the top-to-bottom, left-to-right scan order. Note that these edges express **statistical dependence** rather than a strict causal mechanism: image features are deterministic functions of the pixels, and the edge directions are chosen mainly to obtain an interpretable generative factorization whose CPTs can be estimated.

### 3.3 CPT Learning and Inference

The network factorizes as

$$
P(Y,F_1,\dots,F_{12})=P(Y)\prod_{i=1}^{12}P\bigl(F_i\mid \mathrm{Pa}(F_i)\bigr),
$$

where $\mathrm{Pa}(F_i)$ contains $Y$ and, if present, the feature parents. Each CPT is estimated by maximum likelihood with Laplace smoothing:

$$
\hat P(f_i=v\mid y,\mathrm{pa})=\frac{N(y,\mathrm{pa},v)+\alpha}{N(y,\mathrm{pa})+V_i\alpha},
$$

where $V_i$ is the number of values of that feature and $\alpha=1$.

Since all 12 features are observed at prediction time, $Y$ is the only unobserved variable, so exact inference reduces to directly computing the joint probability for the 10 classes:

$$
\hat y=\arg\max_y\; P(y)\prod_{i=1}^{12}P\bigl(f_i\mid y,\mathrm{pa}_i\bigr).
$$

The implementation likewise accumulates in log space. `predict_proba` applies a softmax over the 10 joint log-probabilities. The network is a strict superset of naive Bayes: removing the two feature edges reduces it to naive Bayes with conditionally independent features.

### 3.4 Validation Experiments on Feature Layouts

We compared several feature layouts on the 50,000/10,000 training split (validation accuracy):

| Feature layout | Validation accuracy |
|---|---|
| 3×3 cell orientation codes + aspect ratio + symmetry + holes (12-D) | 0.6891 |
| 6×2 cell orientation codes (12-D, no global features) | 0.7218 |
| 6×2 cell orientation codes + vertical chain edges | 0.7356 |
| **5×2 cell orientation codes + aspect ratio + holes (12-D)** | **0.7690** |
| The above + `c00→c01`, `c40→c41` | **0.7693** |

The 5×2 horizontal bands capture the vertical structure of digits (loops, arcs, bars) better than 3×3 blocks, and adding the two global topological features (aspect ratio and holes) brings a clear improvement. The two feature edges are essentially on par with no edges on the validation set (0.7690 → 0.7693), indicating that the left/right cell dependence is weak for this feature set, but keeping them makes the network structure better reflect the fact that the features are not fully conditionally independent.

---

## 4. Experimental Evaluation

### 4.1 Experimental Setup

- Training set: 60,000 images; test set: 10,000 images (official split, no extra augmentation).
- NB: 3 bins, $\alpha=1.0$; BN: 12 features, $\alpha=1.0$, structure as in Section 3.2.
- Metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, and confusion matrix; macro averages weight the 10 classes equally.
- Environment: Python 3.11, NumPy 2.x, SciPy, Matplotlib; run on CPU.

### 4.2 Overall Metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Naive Bayes | **0.8466** | **0.8478** | **0.8439** | **0.8447** |
| Bayesian network | 0.7756 | 0.7749 | 0.7742 | 0.7743 |

NB is about **7.1 percentage points** higher than BN on the test set. The training and prediction times of the two models are:

| Model | Training time | Prediction time (10,000 images) |
|---|---|---|
| NB | about 1.0 s | about 0.8 s |
| BN | about 12 s | about 4 s |

Most of BN's training time is spent on Sobel filtering, cell-orientation statistics, and hole labelling for the 60,000 images; NB's counting is a single vectorized `bincount`, so it is an order of magnitude faster.

### 4.3 Per-class Metrics

**Naive Bayes:**

| Digit | Precision | Recall | F1 | #samples |
|---|---|---|---|---|
| 0 | 0.910 | 0.907 | 0.909 | 980 |
| 1 | 0.905 | 0.956 | 0.930 | 1135 |
| 2 | 0.887 | 0.828 | 0.857 | 1032 |
| 3 | 0.771 | 0.834 | 0.801 | 1010 |
| 4 | 0.838 | 0.822 | 0.830 | 982 |
| 5 | 0.825 | 0.719 | 0.768 | 892 |
| 6 | 0.892 | 0.891 | 0.892 | 958 |
| 7 | 0.932 | 0.849 | 0.889 | 1028 |
| 8 | 0.761 | 0.786 | 0.773 | 974 |
| 9 | 0.758 | 0.846 | 0.800 | 1009 |

**Bayesian network:**

| Digit | Precision | Recall | F1 | #samples |
|---|---|---|---|---|
| 0 | 0.739 | 0.793 | 0.765 | 980 |
| 1 | 0.848 | 0.864 | 0.856 | 1135 |
| 2 | 0.727 | 0.753 | 0.740 | 1032 |
| 3 | 0.722 | 0.695 | 0.708 | 1010 |
| 4 | 0.672 | 0.650 | 0.661 | 982 |
| 5 | 0.766 | 0.734 | 0.750 | 892 |
| 6 | 0.865 | 0.843 | 0.854 | 958 |
| 7 | 0.785 | 0.809 | 0.797 | 1028 |
| 8 | 0.826 | 0.823 | 0.825 | 974 |
| 9 | 0.799 | 0.777 | 0.788 | 1009 |

NB's strengths are 0, 1, 6, and 7. The stroke of 1 is the simplest (almost only a vertical line), which the pixel model learns easily; the loop structures of 0 and 6 are also very stable in the per-pixel likelihood. Its relative weaknesses are 5, 8, 3, and 9: the stroke combinations of these digits vary a lot (the top bar of 5 can be long or short; the two loops of 8 differ in size), and a model that treats pixels independently struggles to capture the configuration between strokes.

BN's per-class behaviour follows a similar trend but is flatter: F1 is higher for 0, 1, 6, and 8, and lower for 4, 3, and 2. The F1 of 4 is only 0.661 because the "vertical + horizontal + diagonal" structure of 4 is easily compressed into a mixture of "vertical/diagonal" in the coarse 5×2 cells, causing confusion with 9 and 1.

### 4.4 Confusion Matrices

![NB confusion matrix](figures/nb_confusion.png)

![BN confusion matrix](figures/bn_confusion.png)

Most confused digit pairs (true → predicted, count):

| Rank | NB | BN |
|---|---|---|
| 1 | 4→9: 122 | 5→3: 113 |
| 2 | 5→3: 121 | 3→5: 100 |
| 3 | 8→3: 74 | 1→4: 97 |
| 4 | 9→4: 68 | 0→2: 78 |
| 5 | 7→9: 68 | 9→7: 73 |

NB's 4↔9 confusion is typical: if the top of a handwritten 4 is closed (like a triangle), its pixel distribution is very close to the upper part of a 9; if the lower vertical stroke of a 9 is short, it is also misclassified as 4. The 5→3 confusion occurs because the top bar of 5 and the upper arc of 3 look similar at low resolution. The 8→3 confusion mostly happens when the left loop of 8 is written faintly.

BN's confusion pattern is more "structural": 5→3 and 3→5 show that orientation coding alone cannot distinguish "top bar + left vertical with a bottom bend" from "two arcs"; 1→4 occurs because the right vertical stroke of 4 and the vertical stroke of 1 have the same orientation code, while the horizontal stroke of 4 is easily averaged out when it falls on a cell boundary; 0→2 occurs because the top arc of 2 and the upper loop of 0 are both horizontal/diagonal and their aspect ratios are close.

### 4.5 Misclassified Sample Analysis

![NB misclassified samples](figures/nb_misclassified.png)

The figure shows 8 misclassified NB samples (the titles give the true label, predicted label, and predicted probability), covering the main confusion types in Section 4.4. We observe:

1. **4 → 9**: the top of the 4 is closed and the right vertical stroke is long, so the overall outline is very close to a 9; the pixel model does not explicitly model "whether the top is closed" and simply votes on independent pixel evidence.
2. **5 → 3, 5 → 8**: when the top bar of the 5 is short and the lower bend is round, the evidence for the left vertical stroke is insufficient and the model turns to the right-hand double-arc structure of 3; if the lower bend and the top bar form an approximately closed loop, the digit is classified as 8.
3. **8 → 3**: when the left loop of the 8 is faint and wide open, the evidence for the left "loop" is insufficient and the model prefers the right-hand double-arc structure of 3.
4. **9 → 4**: when the lower vertical stroke of the 9 is short and the upper loop is square, it looks very similar to a closed 4; the predicted probability for this sample is only 0.62, indicating that the model itself is not confident.
5. **7 → 9**: at low resolution, the top bar and long diagonal of a 7 overlap with the "loop + vertical" outline of a 9.
6. **3 → 5**: the upper arc of 3 is close in shape to the top bar of 5, a reverse 3/5 confusion.
7. **2 → 7**: when the zigzag and bottom bar of the 2 are straightened, the overall skeleton resembles a 7.

Most of these errors come from **intra-class variation in writing style** rather than obvious noise; this is also the main source of error for pixel-level generative models—they discard the structural information about "which pixels should co-occur." Another notable phenomenon is that most misclassified samples have $P(\text{pred})$ close to 1.00: the conditional independence assumption treats 784 highly correlated pixels as independent evidence and accumulates it repeatedly, systematically inflating the posterior (overconfidence). This does not affect the correctness of the argmax, but it means that the probability values output by NB should not be used directly as confidence.

---

## 5. Discussion and Reflection

### 5.1 Why Does Naive Bayes Clearly Win?

1. **Information content**: NB uses all 784 pixels (each with 3 values), preserving stroke position, thickness, curvature, and all other details; BN has only 12 coarse-grained features, and a 5×2 cell averages multiple strokes within the cell into a single orientation, irreversibly discarding much detail.
2. **Variance of parameter estimation**: each NB CPT has only $10\times3$ numbers, and 60,000 samples are plentiful; BN's CPTs are also small, but the discretization error of the features themselves is larger.
3. **The "wrongness" of the conditional independence assumption is actually beneficial**: pixels are clearly not independent (adjacent pixels are highly correlated), but NB's decision depends only on the relative magnitudes of the posteriors; although duplicated evidence inflates confidence, it often does not change the argmax. This is why NB has long been robust on text and image tasks.
4. **Data fragmentation caused by feature edges**: after adding the parent `c00` to `c01`, the CPT grows from $10\times5$ to $10\times5\times5$, and the samples of each cell are split into 25 conditional cells, increasing estimation variance; the gain of the edges on the validation set (+0.0003) is far smaller than the gain from feature design. This also explains why BN is not suitable for stacking too many edges when features are few.

### 5.2 Design Trade-offs in the Bayesian Network

- **Why 5×2 rather than 3×3?** Validation experiments show that the 12-D 3×3 + global-feature scheme reaches only 0.689, whereas 5×2 bands express the vertical structure of "upper loop / middle bar / lower loop" better and raise validation accuracy to 0.769. Discriminative information in digits is distributed more along the vertical direction (the upper and lower loops of 0, the two loops of 8, the top bar and lower bend of 5), so horizontal bands are more suitable.
- **Why keep the two feature edges?** The assignment requires at least two feature–feature edges. Experiments show that they have almost no effect on accuracy for this feature set (±0.001), but they are structurally more reasonable: the stroke orientations of the two halves of the same band are not conditionally independent. Removing them would reduce the network to naive Bayes with "12 mutually independent features," losing the point of a BN.
- **Feature discretization thresholds**: the aspect-ratio terciles (1.176, 1.538) and the hole truncation (≥2 recorded as 2) are computed on the training set and then fixed, and reused directly at test time to avoid using test information. The cell-orientation thresholds (empty fraction 0.02, four orientation intervals) are fixed geometric definitions and do not depend on the data.
- **Smoothing**: with $\alpha=1$, every value of every CPT receives at least one pseudo-count. On the validation set, changing $\alpha$ from 0.1 to 1 changes accuracy by less than 0.1 percentage point, showing that the result is insensitive to the smoothing strength; choosing 1 also keeps the implementation consistent with the interface default.

### 5.3 Limitations and Future Work

1. **BN features are still too coarse**: there are 10 cells in 5×2 and each cell can take only one dominant orientation; one could switch to 4×4 or 6×3 cells, or encode "dominant orientation + presence of ink" jointly as 6–8 values per cell (but the "2–5 values per feature" constraint then requires splitting it into two features). Another direction is to add shape features such as endpoint/branch counts, stroke width, and Zernike moments.
2. **The structure could be better justified**: the current edges were hand-specified and confirmed on the validation set; where structure learning is allowed, one could use BIC/MDL scoring with hill-climbing search, or conditional mutual-information tests for each pair of features to decide the edges. The assignment forbids automatic structure learning, so this experiment only performed a small-scale manual comparison.
3. **NB's independence assumption**: one could introduce pixel blocks (e.g. local 2×2 or 4×4 blocks) as "super-pixels," keeping the joint distribution within a block and independence between blocks, yielding a model between NB and the full joint; alternatively, TAN (tree-augmented naive Bayes) adds a tree among features, a natural compromise between BN and NB.
4. **Numerics and engineering**: the current `extract_features` of BN runs Sobel filtering and connected-component labelling independently for each image, taking about 10–12 s for 60,000 images, which is acceptable; for larger datasets, the orientation histograms and hole counting could be implemented with batched convolution / union-find.
5. **Confidence calibration**: NB's posterior probabilities are extreme because of the repeated counting under the independence assumption (misclassified samples can still have high $P(\text{pred})$). If probability outputs are needed, temperature scaling or isotonic calibration can be applied, but this does not affect classification accuracy.

### 5.4 Conclusion

This assignment implemented two generative classifiers, naive Bayes and a Bayesian network, from scratch. NB uses 784 discrete pixels as features and reaches **84.66%** accuracy and a macro F1 of **0.8447** on the test set; BN builds a DAG from 12 hand-designed stroke-orientation/aspect-ratio/hole features and reaches **77.56%** accuracy and a macro F1 of **0.7743**. NB's advantage comes from finer-grained information and lower estimation variance; BN's value lies in its interpretable structure, explicit feature semantics, and explicit modelling of feature dependencies. The roughly 7-percentage-point gap illustrates the trade-off between "feature engineering + structural assumptions" and "pixel-level generative models": on image tasks with abundant samples and high feature dimensionality, simple NB often remains a strong baseline, while BN is better suited to scenarios that require interpretability and prior knowledge.
