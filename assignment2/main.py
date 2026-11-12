"""Assignment 2 driver: train and evaluate both classifiers.

Usage
-----
    python main.py [--data-dir PATH]

The script
  * loads the MNIST training/test splits,
  * trains ``NaiveBayesClassifier`` and ``BayesNetClassifier`` on the full
    training set,
  * evaluates both on the full test set (accuracy, macro precision/recall/F1,
    confusion matrix),
  * writes ``metrics.json`` and the figures used in ``report.md``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bayes_net import (  # noqa: E402
    FEATURE_EDGES,
    FEATURE_NAMES,
    BayesNetClassifier,
    extract_features,
)
from naive_bayes import NaiveBayesClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent


def resolve_data_dir(explicit: Path | None) -> Path:
    """Find the MNIST directory in the layouts used by the course and locally."""
    if explicit is not None:
        return Path(explicit)
    candidates = [
        HERE.parent / "assignment2-data" / "mnist",
        HERE.parent / "assignment2-data" / "assignment2-data" / "mnist",
        HERE / "data" / "mnist",
        Path.cwd() / "assignment2-data" / "mnist",
    ]
    for candidate in candidates:
        if (candidate / "train-images-idx3-ubyte.gz").is_file():
            return candidate
    return candidates[0]


DEFAULT_DATA = resolve_data_dir(None)
FIG_DIR = HERE / "figures"
N_CLASSES = 10


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def load_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four gzip-compressed IDX files from ``data_dir``."""
    data_dir = Path(data_dir)
    X_train = _read_idx(data_dir / "train-images-idx3-ubyte.gz").reshape(-1, 784)
    y_train = _read_idx(data_dir / "train-labels-idx1-ubyte.gz").ravel()
    X_test = _read_idx(data_dir / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    y_test = _read_idx(data_dir / "t10k-labels-idx1-ubyte.gz").ravel()
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> dict:
    """Return accuracy, macro precision/recall/F1, per-class metrics and confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(n_classes, n_classes)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)          # true counts
    predicted = cm.sum(axis=0).astype(np.float64)        # predicted counts
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    present = support > 0
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
        "confusion": cm,
    }


def top_confusions(cm: np.ndarray, k: int = 5) -> list[tuple[int, int, int]]:
    """Most frequent off-diagonal confusion pairs (true, predicted, count)."""
    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    return [(i, j, c) for c, i, j in pairs[:k]]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                       proba: np.ndarray, path: Path, n: int = 8,
                       pairs: tuple[tuple[int, int], ...] = ()) -> None:
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    chosen: list[int] = []
    for true_digit, pred_digit in pairs:
        hits = np.where((y_true == true_digit) & (y_pred == pred_digit))[0]
        if len(hits) and int(hits[0]) not in chosen:
            chosen.append(int(hits[0]))
    rest = [int(i) for i in wrong if int(i) not in chosen]
    chosen += rest[: max(0, n - len(chosen))]
    chosen = chosen[:n]
    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), chosen):
        ax.imshow(images[idx].reshape(28, 28), cmap="gray_r")
        ax.set_title(
            f"true={y_true[idx]}  pred={y_pred[idx]}\nP(pred)={proba[idx, y_pred[idx]]:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Naive Bayes misclassified examples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(metrics_nb: dict, metrics_bn: dict, path: Path) -> None:
    labels = ["Accuracy", "Macro F1"]
    nb = [metrics_nb["accuracy"], metrics_nb["macro_f1"]]
    bn = [metrics_bn["accuracy"], metrics_bn["macro_f1"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(x - 0.18, nb, 0.36, label="Naive Bayes", color="#4C72B0")
    ax.bar(x + 0.18, bn, 0.36, label="Bayesian Network", color="#DD8452")
    for xi, v in zip(x - 0.18, nb):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.18, bn):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test-set comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dag(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    pos = {"Y": (0.5, 0.96)}
    for i in range(5):
        for j in range(2):
            pos[f"c{i}{j}"] = (0.30 + 0.40 * j, 0.80 - 0.13 * i)
    pos["aspect"] = (0.24, 0.08)
    pos["holes"] = (0.76, 0.08)

    for name in FEATURE_NAMES:
        x0, y0 = pos["Y"]
        x1, y1 = pos[name]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#9aa7b4", lw=0.8,
                                    shrinkA=14, shrinkB=18))
    for parent, child in FEATURE_EDGES:
        x0, y0 = pos[parent]
        x1, y1 = pos[child]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=2.2,
                                    shrinkA=20, shrinkB=20,
                                    connectionstyle="arc3,rad=0.25"))
    for name, (x, y) in pos.items():
        ax.text(x, y, name, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#4C72B0"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.axis("off")
    ax.set_title("Bayesian network structure (grey: Y -> feature, red: feature -> feature)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_grid(images: np.ndarray, features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    # pick one example per digit for a small panel
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    for digit, ax in enumerate(axes.ravel()):
        idx = int(np.where(labels == digit)[0][0])
        ax.imshow(images[idx].reshape(28, 28), cmap="gray_r")
        for r in (0, 6, 11, 17, 22, 28):
            ax.axhline(r - 0.5, color="#4C72B0", lw=0.7)
        ax.axvline(13.5, color="#4C72B0", lw=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        codes = " ".join(
            str(features[idx, 2 * i + j]) for i in range(5) for j in range(2)
        )
        ax.set_title(f"label {digit}\ncells {codes}\nA={features[idx, 10]} H={features[idx, 11]}",
                     fontsize=7)
    fig.suptitle("5x2 cell grid overlaid on example digits (cell codes: 0 empty, 1 H, 2 D+, 3 V, 4 D-)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the assignment-2 models")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading MNIST from {args.data_dir} ...")
    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    print(f"train {X_train.shape}, test {X_test.shape}")

    # ---------------- Naive Bayes ----------------
    t0 = time.time()
    nb = NaiveBayesClassifier(alpha=1.0).fit(X_train, y_train)
    t_fit_nb = time.time() - t0
    t0 = time.time()
    pred_nb = nb.predict(X_test)
    t_pred_nb = time.time() - t0
    proba_nb = nb.predict_proba(X_test)
    m_nb = evaluate(y_test, pred_nb)
    print(f"\n[Naive Bayes] fit {t_fit_nb:.1f}s  predict {t_pred_nb:.1f}s")
    print_metrics("Naive Bayes", m_nb)

    # ---------------- Bayesian network ----------------
    t0 = time.time()
    bn = BayesNetClassifier(alpha=1.0).fit(X_train, y_train)
    t_fit_bn = time.time() - t0
    t0 = time.time()
    features_test = extract_features(X_test)
    proba_bn = bn.predict_proba(X_test)
    pred_bn = proba_bn.argmax(axis=1)
    t_pred_bn = time.time() - t0
    m_bn = evaluate(y_test, pred_bn)
    print(f"\n[Bayesian Network] fit {t_fit_bn:.1f}s  predict {t_pred_bn:.1f}s")
    print_metrics("Bayesian Network", m_bn)
    print("\nStructure:\n" + bn.describe_structure())

    # ---------------- figures ----------------
    plot_confusion(m_nb["confusion"], "Naive Bayes confusion matrix (test)", FIG_DIR / "nb_confusion.png")
    plot_confusion(m_bn["confusion"], "Bayesian Network confusion matrix (test)", FIG_DIR / "bn_confusion.png")
    plot_misclassified(
        X_test, y_test, pred_nb, proba_nb, FIG_DIR / "nb_misclassified.png",
        pairs=((4, 9), (5, 3), (8, 3), (9, 4), (7, 9), (5, 8), (3, 5), (2, 7)),
    )
    plot_comparison(m_nb, m_bn, FIG_DIR / "model_comparison.png")
    plot_dag(FIG_DIR / "bn_dag.png")
    plot_feature_grid(X_test, features_test, y_test, FIG_DIR / "feature_grid.png")

    # ---------------- persist metrics ----------------
    def dump(m: dict) -> dict:
        return {
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "per_class": [
                {
                    "digit": i,
                    "precision": float(m["precision"][i]),
                    "recall": float(m["recall"][i]),
                    "f1": float(m["f1"][i]),
                    "support": int(m["support"][i]),
                }
                for i in range(N_CLASSES)
            ],
            "confusion": m["confusion"].tolist(),
            "top_confusions": [
                {"true": i, "predicted": j, "count": c} for i, j, c in top_confusions(m["confusion"])
            ],
        }

    payload = {
        "naive_bayes": dump(m_nb),
        "bayes_net": dump(m_bn),
        "timings": {
            "nb_fit": t_fit_nb,
            "nb_predict": t_pred_nb,
            "bn_fit": t_fit_bn,
            "bn_predict": t_pred_bn,
        },
        "bn_structure": {
            "features": FEATURE_NAMES,
            "edges": [["Y", n] for n in FEATURE_NAMES] + [list(e) for e in FEATURE_EDGES],
        },
    }
    (HERE / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nMost confused pairs (true -> predicted):")
    for name, m in (("NB", m_nb), ("BN", m_bn)):
        pairs = "  ".join(f"{i}->{j}:{c}" for i, j, c in top_confusions(m["confusion"]))
        print(f"  {name}: {pairs}")
    print(f"\nSaved metrics.json and figures to {FIG_DIR}")


def print_metrics(name: str, m: dict) -> None:
    print(f"  accuracy       {m['accuracy']:.4f}")
    print(f"  macro precision {m['macro_precision']:.4f}")
    print(f"  macro recall   {m['macro_recall']:.4f}")
    print(f"  macro F1       {m['macro_f1']:.4f}")
    print("  per class:")
    for i in range(N_CLASSES):
        print(f"    {i}: P={m['precision'][i]:.3f} R={m['recall'][i]:.3f} "
              f"F1={m['f1'][i]:.3f} n={m['support'][i]}")


if __name__ == "__main__":
    main()
