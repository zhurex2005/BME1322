"""Assignment 4 公共工具：数据读取、固定折、Pipeline 与指标体系。

classify.py / regress.py / ablations.py / main.py 全部从这里取数据与评分函数，
以保证报告中的每个数字都出自同一套口径：

* 固定 5 折来自 ``assignment4-data/splits.json``，不重新划分（4.5 第 3 项除外）；
* 所有预处理（中位数填补、标准化）都包在 ``Pipeline`` 内，折内 ``fit``，无泄漏；
* 分类指标先按折计算再取均值；AUC 一律用连续分数（decision_function），
  ``auc_hard`` 仅用于复现讲义 4.4 中"硬标签口径"的对照；
* 回归指标先按折计算再取均值；朴素参考线（训练折均值）同样折内计算。

数据文件不随代码提交（见作业第七节），脚本会依次搜索
``$A4_DATA``、``../assignment4-data``、``./assignment4-data`` 等位置。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
RESULT_DIR = HERE / "results"

SEED = 42

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
LABEL = "num"

FEATURE_CN = {
    "age": "年龄", "sex": "性别", "cp": "胸痛类型", "trestbps": "静息血压",
    "chol": "胆固醇", "fbs": "空腹血糖", "restecg": "静息心电图",
    "thalach": "最大心率", "exang": "运动心绞痛", "oldpeak": "ST 段压低",
    "slope": "ST 段斜率", "ca": "显影血管数", "thal": "灌注结果",
}

# 特征分组（2.1 节）：A 常规问诊/体检，B 负荷试验，C 有创或昂贵检查
FEATURE_GROUPS = {
    "常规": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"],
    "负荷试验": ["thalach", "exang", "oldpeak", "slope"],
    "造影/灌注": ["ca", "thal"],
}

CLASS_METRICS = [
    "accuracy", "precision_pos", "recall_pos", "f1_pos",
    "precision_neg", "recall_neg", "f1_neg", "f1_macro", "auc", "auc_hard",
]


# --------------------------------------------------------------------------- #
# 数据与折
# --------------------------------------------------------------------------- #
def data_dir() -> Path:
    """定位课程数据目录（含 heart_cleveland.csv 与 splits.json）。"""
    candidates = []
    env = os.environ.get("A4_DATA")
    if env:
        candidates.append(Path(env))
    for base in [HERE, *HERE.parents]:
        candidates += [base / "assignment4-data", base / "data" / "assignment4-data"]
    for cand in candidates:
        if (cand / "heart_cleveland.csv").exists() and (cand / "splits.json").exists():
            return cand
    raise FileNotFoundError(
        "找不到 assignment4-data/（需要 heart_cleveland.csv 与 splits.json）。"
        "可用环境变量 A4_DATA 指定目录。"
    )


def load_data(drop: tuple[str, ...] = ()):
    """读取 Cleveland 数据。

    返回 (df, features, X, y_num, y_bin)：
      * ``X`` 为 float 矩阵，空单元格读成 NaN，交给 SimpleImputer；
      * ``y_num`` 为 0–4 的严重程度，``y_bin`` 为 ``num > 0``。
    """
    df = pd.read_csv(data_dir() / "heart_cleveland.csv")
    features = [c for c in FEATURES if c not in drop]
    X = df[features].to_numpy(dtype=float)
    y_num = df[LABEL].to_numpy(dtype=int)
    y_bin = (y_num > 0).astype(int)
    return df, features, X, y_num, y_bin


def load_cv(n: int = 303):
    """读取课程固定 5 折，返回 [(train_idx, test_idx), ...]。"""
    with open(data_dir() / "splits.json", encoding="utf-8") as f:
        folds = json.load(f)["cleveland"]["folds"]
    all_idx = np.arange(n)
    cv = [(np.setdiff1d(all_idx, np.asarray(test)), np.asarray(test)) for test in folds]
    assert sum(len(te) for _, te in cv) == n
    return cv


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def make_pipe(estimator, scale: bool = True, impute: bool = True) -> Pipeline:
    """统一的预处理链：中位数填补 -> 标准化（可选）-> 模型。"""
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _scores(est, X) -> np.ndarray:
    """连续分数：优先 decision_function，其次正类概率。"""
    if hasattr(est, "decision_function"):
        return np.asarray(est.decision_function(X), dtype=float)
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(est.predict(X), dtype=float)


# --------------------------------------------------------------------------- #
# 分类交叉验证
# --------------------------------------------------------------------------- #
def cv_classification(pipe_factory, X, y, cv, return_train: bool = False) -> dict:
    """按固定折评估分类器。

    ``pipe_factory`` 是无参可调用对象，每次返回一条未拟合的 Pipeline。
    返回 ``fold``（每折指标数组）、``mean``、``std``、``oof``（全表分数/预测）、
    ``confusion``（各折混淆矩阵之和）、``train_mean``（可选，训练折指标）。
    """
    y = np.asarray(y, dtype=int)
    fold = {k: [] for k in CLASS_METRICS}
    train = {"accuracy": [], "auc": []}
    oof_score = np.zeros(len(y), dtype=float)
    oof_pred = np.zeros(len(y), dtype=int)
    cm = np.zeros((2, 2), dtype=int)

    for tr, te in cv:
        est = pipe_factory()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        score = _scores(est, X[te])
        oof_score[te] = score
        oof_pred[te] = pred

        fold["accuracy"].append(accuracy_score(y[te], pred))
        fold["precision_pos"].append(precision_score(y[te], pred, pos_label=1, zero_division=0))
        fold["recall_pos"].append(recall_score(y[te], pred, pos_label=1, zero_division=0))
        fold["f1_pos"].append(f1_score(y[te], pred, pos_label=1, zero_division=0))
        fold["precision_neg"].append(precision_score(y[te], pred, pos_label=0, zero_division=0))
        fold["recall_neg"].append(recall_score(y[te], pred, pos_label=0, zero_division=0))
        fold["f1_neg"].append(f1_score(y[te], pred, pos_label=0, zero_division=0))
        fold["f1_macro"].append(f1_score(y[te], pred, average="macro", zero_division=0))
        fold["auc"].append(roc_auc_score(y[te], score))
        fold["auc_hard"].append(roc_auc_score(y[te], pred))
        cm += confusion_matrix(y[te], pred, labels=[0, 1])

        if return_train:
            train_pred = est.predict(X[tr])
            train["accuracy"].append(accuracy_score(y[tr], train_pred))
            train["auc"].append(roc_auc_score(y[tr], _scores(est, X[tr])))

    out = {
        "fold": {k: np.asarray(v) for k, v in fold.items()},
        "mean": {k: float(np.mean(v)) for k, v in fold.items()},
        "std": {k: float(np.std(v)) for k, v in fold.items()},
        "oof_score": oof_score,
        "oof_pred": oof_pred,
        "confusion": cm,
    }
    if return_train:
        out["train_mean"] = {k: float(np.mean(v)) for k, v in train.items()}
    return out


# --------------------------------------------------------------------------- #
# 回归交叉验证
# --------------------------------------------------------------------------- #
REG_METRICS = ["r2", "mae", "rmse"]


def cv_regression(pipe_factory, X, y, cv, structure: bool = False) -> dict:
    """按固定折评估回归器；``structure`` 时额外记录叶节点数与树深。"""
    y = np.asarray(y, dtype=float)
    fold = {k: [] for k in REG_METRICS}
    fold_mse = []
    train_r2 = []
    leaves, depth = [], []
    oof = np.zeros(len(y), dtype=float)
    mean_pred = np.zeros(len(y), dtype=float)

    for tr, te in cv:
        est = pipe_factory()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        oof[te] = pred
        # 朴素参考线：全部预测训练折均值
        mu = float(np.mean(y[tr]))
        mean_pred[te] = mu

        fold["r2"].append(r2_score(y[te], pred))
        fold["mae"].append(mean_absolute_error(y[te], pred))
        fold_mse.append(mean_squared_error(y[te], pred))
        fold["rmse"].append(float(np.sqrt(fold_mse[-1])))
        train_r2.append(r2_score(y[tr], est.predict(X[tr])))
        if structure:
            model = est.named_steps["model"]
            leaves.append(model.get_n_leaves())
            depth.append(model.get_depth())

    means = {k: float(np.mean(v)) for k, v in fold.items()}
    # 与 baselines.py 一致：RMSE = sqrt(五折 MSE 的均值)，而不是五折 RMSE 的均值
    means["rmse"] = float(np.sqrt(np.mean(fold_mse)))
    out = {
        "fold": {k: np.asarray(v) for k, v in fold.items()},
        "mean": means,
        "std": {k: float(np.std(v)) for k, v in fold.items()},
        "train_r2": float(np.mean(train_r2)),
        "oof": oof,
    }
    if structure:
        out["leaves"] = float(np.mean(leaves))
        out["depth"] = float(np.mean(depth))
    return out


def naive_regression(y, cv) -> dict:
    """朴素参考线：每折全部预测训练折均值。"""
    y = np.asarray(y, dtype=float)
    r2, mae, rmse = [], [], []
    for tr, te in cv:
        mu = float(np.mean(y[tr]))
        pred = np.full(len(te), mu)
        r2.append(r2_score(y[te], pred))
        mae.append(mean_absolute_error(y[te], pred))
        rmse.append(float(np.sqrt(mean_squared_error(y[te], pred))))
    return {"r2": float(np.mean(r2)), "mae": float(np.mean(mae)),
            "rmse": float(np.mean(rmse))}


def naive_classification(y, cv) -> dict:
    """朴素参考线：每折全部预测训练折多数类。"""
    y = np.asarray(y, dtype=int)
    acc, f1m, recall_pos, precision_pos = [], [], [], []
    for tr, te in cv:
        majority = int(np.mean(y[tr]) > 0.5)
        pred = np.full(len(te), majority, dtype=int)
        acc.append(accuracy_score(y[te], pred))
        f1m.append(f1_score(y[te], pred, average="macro", zero_division=0))
        recall_pos.append(recall_score(y[te], pred, pos_label=1, zero_division=0))
        precision_pos.append(precision_score(y[te], pred, pos_label=1, zero_division=0))
    return {"accuracy": float(np.mean(acc)), "f1_macro": float(np.mean(f1m)),
            "recall_pos": float(np.mean(recall_pos)),
            "precision_pos": float(np.mean(precision_pos)), "auc": 0.5}


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def fmt(v, nd: int = 3) -> str:
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def md_table(headers, rows, nd: int = 3) -> str:
    """生成 markdown 表格字符串。"""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(fmt(v, nd) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def print_md(text: str) -> None:
    print(text)
    print()


def to_jsonable(obj):
    """把 numpy 标量/数组递归转成可写入 JSON 的对象。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def setup_matplotlib():
    """统一图表风格；返回 pyplot。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def savefig(fig, name: str) -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    return path
