"""Bayesian network classifier for handwritten digits.

Feature design
--------------
Each image is divided into 5 horizontal bands and 2 halves, giving ten cells.
For every cell the dominant gradient orientation is estimated from a Sobel
filter and coded with one of five values:

    0 - empty cell (almost no ink)
    1 - horizontal stroke (H)
    2 - rising diagonal (D+)
    3 - vertical stroke (V)
    4 - falling diagonal (D-)

Two global features complete the 12-dimensional representation:

    11 - aspect ratio of the digit's bounding box (3 levels)
    12 - number of enclosed background regions, i.e. holes (0, 1, >=2)

Every feature is discrete with 2-5 values, as required by the assignment.

Model
-----
The class label ``Y`` is the root.  Besides the twelve ``Y -> feature`` edges
the graph contains two feature-feature edges (``c00 -> c01`` and
``c40 -> c41``) which capture the dependency between the left and right halves
of the top and bottom bands.  All conditional probability tables are estimated
by maximum likelihood with Laplace smoothing, and inference is exact (all
features are observed, so the joint log-probability can be evaluated directly).

The class implements the interface required by the assignment:

    clf = BayesNetClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

and the module-level ``extract_features(X)`` function maps a batch of flattened
images to the integer feature matrix.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# feature definition
# ---------------------------------------------------------------------------
N_ROWS, N_COLS = 5, 2
CELL_NAMES = [f"c{i}{j}" for i in range(N_ROWS) for j in range(N_COLS)]
FEATURE_NAMES = CELL_NAMES + ["aspect", "holes"]
# Number of possible values of every feature (cells: 0..4, aspect: 0..2,
# holes: 0..2).  Fixing the cardinalities keeps the CPT shapes independent of
# which values happen to occur in a particular training batch.
FEATURE_CARDINALITIES = [5] * len(CELL_NAMES) + [3, 3]

# P(cell is empty) threshold: fraction of ink pixels inside the cell
EMPTY_RATIO = 0.02

# Aspect-ratio thresholds: terciles of the training set (h / w)
ASPECT_EDGES = (1.176, 1.538)

# holes -> {0, 1, 2} (2 means "two or more")
MAX_HOLES = 2

# Feature-feature edges of the DAG (parent -> child).  The remaining edges are
# Y -> every feature.  Children may have at most two parents.
FEATURE_EDGES = (("c00", "c01"), ("c40", "c41"))


def _as_images(X: np.ndarray) -> np.ndarray:
    """Return a float image batch with values in [0, 1] and shape (n, 28, 28)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        if X.shape[1] != 28 * 28:
            raise ValueError("expected flattened 28x28 images")
        X = X.reshape(-1, 28, 28)
    elif X.ndim != 3 or X.shape[1:] != (28, 28):
        raise ValueError("expected an array of shape (n, 784) or (n, 28, 28)")
    if X.size and X.max() > 1.5:  # raw 0..255 data
        X = X / 255.0
    return X


def _count_holes(binary: np.ndarray) -> np.ndarray:
    """Number of enclosed background regions per image, clipped to [0, 2]."""
    background = ~binary
    n = len(binary)
    four = np.zeros((3, 3, 3), dtype=bool)
    four[1, 1, 1] = True
    four[1, 0, 1] = four[1, 2, 1] = True
    four[1, 1, 0] = four[1, 1, 2] = True

    out = np.zeros(n, dtype=np.int8)
    chunk = 8192
    for start in range(0, n, chunk):
        bg = background[start:start + chunk]
        labels, n_labels = ndimage.label(bg, structure=four)
        if n_labels == 0:
            continue
        flat = labels.reshape(len(bg), -1)
        # image index of every component (components never cross images)
        uniq, first = np.unique(flat, return_index=True)
        uniq = uniq[1:]  # drop label 0 (ink)
        first = first[1:]
        comp_image = (first // (28 * 28)).astype(np.int64)
        comps_per_image = np.bincount(comp_image, minlength=len(bg))

        border = np.zeros_like(bg)
        border[:, 0, :] = border[:, -1, :] = True
        border[:, :, 0] = border[:, :, -1] = True
        border_labels = np.unique(labels[border])
        border_labels = border_labels[border_labels > 0]
        if border_labels.size:
            idx = np.searchsorted(uniq, border_labels)
            touching_per_image = np.bincount(comp_image[idx], minlength=len(bg))
        else:
            touching_per_image = np.zeros(len(bg), dtype=np.int64)

        holes = comps_per_image - touching_per_image
        out[start:start + chunk] = np.clip(holes, 0, MAX_HOLES).astype(np.int8)
    return out


def extract_features(X: np.ndarray) -> np.ndarray:
    """Map images to the 12 discrete features used by the Bayesian network.

    Parameters
    ----------
    X : array-like, shape (n, 784) or (n, 28, 28)
        Pixel intensities in 0..255 (or 0..1).

    Returns
    -------
    features : np.ndarray, shape (n, 12), dtype int8
        Columns follow ``FEATURE_NAMES``: ten cell orientation codes followed
        by the aspect ratio and the hole count.
    """
    images = _as_images(X)
    n = len(images)
    binary = images > 0.5

    gx = ndimage.sobel(images, axis=2, mode="constant")
    gy = ndimage.sobel(images, axis=1, mode="constant")
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) % np.pi
    # 0 = horizontal, 1 = 45 deg, 2 = vertical, 3 = 135 deg
    orientation = np.digitize(angle, [np.pi / 8, 3 * np.pi / 8, 5 * np.pi / 8, 7 * np.pi / 8]) % 4

    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.int8)
    row_edges = [int(round(i * 28 / N_ROWS)) for i in range(N_ROWS + 1)]
    col_edges = [0, 14, 28]
    col = 0
    for i in range(N_ROWS):
        r0, r1 = row_edges[i], row_edges[i + 1]
        for j in range(N_COLS):
            c0, c1 = col_edges[j], col_edges[j + 1]
            patch_mag = magnitude[:, r0:r1, c0:c1]
            patch_ori = orientation[:, r0:r1, c0:c1]
            ink = binary[:, r0:r1, c0:c1].mean(axis=(1, 2))
            hist = np.stack(
                [(patch_mag * (patch_ori == k)).sum(axis=(1, 2)) for k in range(4)], axis=1
            )
            dominant = hist.argmax(axis=1) + 1  # 1..4
            features[:, col] = np.where(ink < EMPTY_RATIO, 0, dominant)
            col += 1

    # aspect ratio of the bounding box
    rows_any = binary.any(axis=2)
    cols_any = binary.any(axis=1)
    r0 = rows_any.argmax(axis=1)
    r1 = 27 - rows_any[:, ::-1].argmax(axis=1)
    c0 = cols_any.argmax(axis=1)
    c1 = 27 - cols_any[:, ::-1].argmax(axis=1)
    aspect = (r1 - r0 + 1) / (c1 - c0 + 1)
    features[:, len(CELL_NAMES)] = np.digitize(aspect, ASPECT_EDGES)

    features[:, len(CELL_NAMES) + 1] = _count_holes(binary)
    return features


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------
class BayesNetClassifier:
    """Bayesian network with the structure described in the module docstring.

    Parameters
    ----------
    alpha : float, default=1.0
        Laplace smoothing strength for every conditional probability table.
    n_classes : int, default=10
        Number of digit classes.
    """

    def __init__(self, alpha: float = 1.0, n_classes: int = 10) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.n_classes = int(n_classes)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesNetClassifier":
        features = extract_features(X)
        y = np.asarray(y, dtype=np.int64)
        K = self.n_classes
        if y.min() < 0 or y.max() >= K:
            raise ValueError("labels outside [0, n_classes)")

        names = FEATURE_NAMES
        parents = {name: [] for name in names}
        for parent, child in FEATURE_EDGES:
            parents[child].append(parent)

        self.feature_names_ = list(names)
        self.parents_ = parents
        self.n_values_ = list(FEATURE_CARDINALITIES)
        self.log_cond_ = []

        if features.shape[1] != len(names):
            raise ValueError("unexpected number of extracted features")
        if features.min() < 0 or (features >= np.array(self.n_values_)).any():
            raise ValueError("feature value outside its declared cardinality")

        # P(child | Y, parents)
        for j, name in enumerate(names):
            values = features[:, j]
            n_values = self.n_values_[j]
            par = parents[name]
            combos = 1
            parent_index = np.zeros(len(features), dtype=np.int64)
            for p in par:
                pj = names.index(p)
                n_p = self.n_values_[pj]
                parent_index = parent_index * n_p + features[:, pj]
                combos *= n_p
            flat = (y[:, None] * (combos * n_values) + parent_index[:, None] * n_values + values[:, None]).ravel()
            counts = np.bincount(flat, minlength=K * combos * n_values).reshape(K, combos, n_values)
            counts = counts.astype(np.float64)
            self.log_cond_.append(
                np.log((counts + self.alpha) / (counts.sum(axis=2, keepdims=True) + n_values * self.alpha))
            )

        class_counts = np.bincount(y, minlength=K).astype(np.float64)
        self.log_prior_ = np.log((class_counts + self.alpha) / (len(y) + K * self.alpha))
        self.n_features_in_ = len(names)
        return self

    # ------------------------------------------------------------------
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        features = extract_features(X)
        n = len(features)
        K = self.n_classes
        names = self.feature_names_
        scores = np.tile(self.log_prior_, (n, 1))
        for j, name in enumerate(names):
            values = features[:, j]
            parent_index = np.zeros(n, dtype=np.int64)
            combos = 1
            for p in self.parents_[name]:
                pj = names.index(p)
                n_p = self.n_values_[pj]
                parent_index = parent_index * n_p + features[:, pj]
                combos *= n_p
            table = self.log_cond_[j]  # (K, combos, n_values)
            for k in range(K):
                scores[:, k] += table[k][parent_index, values]
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most probable digit for every sample in ``X``."""
        return self._joint_log_likelihood(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior class probabilities (rows sum to one)."""
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        return proba / proba.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def describe_structure(self) -> str:
        """Human-readable description of the learned graph."""
        lines = ["Y -> " + ", ".join(self.feature_names_)]
        lines += [f"{p} -> {c}" for p, c in FEATURE_EDGES]
        return "\n".join(lines)
