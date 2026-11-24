"""课程参考基线 ``baselines.py --json`` 的数值快照。

本文件只是把课程脚本在**固定 5 折 + 固定超参**下打印的数字抄录下来，供
``classify.py`` / ``regress.py`` 生成「复现核对表」使用。它不参与任何训练。

来源：assignment4-data/baselines.py --json（2026-11-23 在本机运行，
scikit-learn 1.9.1 / numpy 2.4.1 / pandas 2.3.3）。
若换用差异较大的库版本，基线数字本身可能有 ±0.001 量级的变化；核对容差取 0.01
（作业 2.4）。
"""

TOL = 0.01

BASELINE = {
    "cleveland/cls/KNN (k=5)": {
        "accuracy": 0.821803, "precision": 0.824888, "recall": 0.777249,
        "f1": 0.799504, "auc": 0.881985,
    },
    "cleveland/cls/Naive Bayes": {
        "accuracy": 0.854645, "precision": 0.863163, "recall": 0.820106,
        "f1": 0.838457, "auc": 0.902211,
    },
    "cleveland/cls/Logistic Regression": {
        "accuracy": 0.831585, "precision": 0.830438, "recall": 0.798413,
        "f1": 0.812617, "auc": 0.911631,
    },
    "cleveland/cls/SVM (linear)": {
        "accuracy": 0.834863, "precision": 0.836711, "recall": 0.798413,
        "f1": 0.815262, "auc": 0.910463,
    },
    "cleveland/cls/SVM (RBF)": {
        "accuracy": 0.838087, "precision": 0.852606, "recall": 0.783862,
        "f1": 0.815278, "auc": 0.897508,
    },
    "cleveland/cls/Decision Tree (d=3)": {
        "accuracy": 0.778634, "precision": 0.766305, "recall": 0.74709,
        "f1": 0.755238, "auc": 0.809088,
    },
    "cleveland/reg/Linear Regression": {
        "r2": 0.522537, "mae": 0.634911, "rmse": 0.842442,
    },
    "cleveland/reg/Ridge (alpha=1)": {
        "r2": 0.52276, "mae": 0.634785, "rmse": 0.842275,
    },
    "cleveland/reg/Lasso (alpha=0.01)": {
        "r2": 0.525461, "mae": 0.632992, "rmse": 0.840237,
    },
    "cleveland/reg/SVR (RBF, C=1)": {
        "r2": 0.43059, "mae": 0.65973, "rmse": 0.913401,
    },
    "cleveland/reg/Regression Tree (d=3)": {
        "r2": 0.29215, "mae": 0.712077, "rmse": 1.013903,
    },
}
