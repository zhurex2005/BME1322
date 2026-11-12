"""Categorical Naive Bayes classifier for 28x28 handwritten digits.

The model treats every pixel as a discrete random variable.  Pixel intensities
are first quantised into a small number of bins (by default three: background,
light ink and strong ink), after which the standard Naive Bayes assumption

    P(y | x_1..x_784) ∝ P(y) * Π_j P(x_j | y)

is used.  All computations are carried out in log space to avoid underflow.
Laplace (add-alpha) smoothing is applied both to the class priors and to every
conditional probability table.

The class implements the interface required by the assignment:

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
"""

from __future__ import annotations

import numpy as np


class NaiveBayesClassifier:
    """Categorical Naive Bayes for flattened 28x28 images (values 0..255).

    Parameters
    ----------
    alpha : float, default=1.0
        Laplace smoothing strength used for the class priors and for every
        pixel/bin conditional probability.
    n_bins : int, default=3
        Number of intensity bins.  ``bin_edges`` must contain ``n_bins - 1``
        strictly increasing thresholds.
    bin_edges : sequence of float, default=(1.0, 128.0)
        Thresholds passed to :func:`numpy.digitize`.  With the defaults the
        bins are ``[0]`` (background), ``[1, 127]`` (light ink) and
        ``[128, 255]`` (strong ink).
    n_classes : int, default=10
        Number of digit classes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_bins: int = 3,
        bin_edges: tuple[float, ...] = (1.0, 128.0),
        n_classes: int = 10,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if len(bin_edges) != n_bins - 1:
            raise ValueError("bin_edges must contain n_bins - 1 thresholds")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.bin_edges = tuple(float(e) for e in bin_edges)
        self.n_classes = int(n_classes)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _discretize(self, X: np.ndarray) -> np.ndarray:
        """Map raw pixel intensities to integer bin indices."""
        X = np.asarray(X, dtype=np.float32)
        X = X.reshape(len(X), -1)
        return np.digitize(X, self.bin_edges).astype(np.int64)

    # ------------------------------------------------------------------
    # training / inference
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        """Estimate priors and per-pixel/bin likelihoods."""
        Xb = self._discretize(X)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = Xb.shape
        K, B = self.n_classes, self.n_bins
        if y.min() < 0 or y.max() >= K:
            raise ValueError("labels outside [0, n_classes)")

        # counts[k, f, b] = #samples of class k whose pixel f falls in bin b
        flat = (y[:, None] * (n_features * B) + np.arange(n_features)[None, :] * B + Xb).ravel()
        counts = np.bincount(flat, minlength=K * n_features * B).reshape(K, n_features, B).astype(np.float64)

        # Laplace-smoothed conditional probabilities P(pixel = b | class)
        self.log_likelihood_ = np.log(
            (counts + self.alpha) / (counts.sum(axis=2, keepdims=True) + B * self.alpha)
        )

        class_counts = np.bincount(y, minlength=K).astype(np.float64)
        self.log_prior_ = np.log((class_counts + self.alpha) / (n_samples + K * self.alpha))
        self.classes_ = np.arange(K)
        self.n_features_in_ = n_features
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        Xb = self._discretize(X)
        if Xb.shape[1] != self.n_features_in_:
            raise ValueError("feature dimension does not match the training data")
        n = len(Xb)
        # scores[:, k] = log P(y=k) + Σ_f log P(x_f | y=k)
        scores = np.empty((n, self.n_classes), dtype=np.float64)
        cols = np.arange(self.n_features_in_)[None, :]
        for k in range(self.n_classes):
            scores[:, k] = self.log_prior_[k] + self.log_likelihood_[k][cols, Xb].sum(axis=1)
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
