"""4.5 四项消融 + 4.6 讨论题③（回归模型当分类器）+ 两条任务线的一致性。

运行方式：

    python assignment4/ablations.py

四项消融：
  A1 去掉 ca 与 thal（造影/灌注的“确诊后”信息）重跑分类与回归；
  A2 标准化 vs 不标准化：SVM 的最优 C 是否漂移，树为什么不受影响；
  A3 回归树在 10 个随机 80/20 划分上的 R² 波动 vs 固定五折；
  A4 岭回归 + 二次多项式特征 + 较强正则化：容量增加是否有益。

另外两项分析：
  B1 把回归模型的预测值当分数：AUC、阈值 0.3/0.5/0.7/1.0 下的分类指标；
  B2 分类（线性 SVM 系数）与回归（Ridge/Lasso 系数、树重要性）是否指向同一批特征。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:  # 兼容脚本与包两种运行方式
    from . import common as cmn
    from . import classify
    from . import regress
except ImportError:  # pragma: no cover
    import common as cmn
    import classify
    import regress

SEED = cmn.SEED
DROP = ("ca", "thal")

CLASS_MODELS = [
    ("SVM linear C=1", classify.svm_factory(1.0, "scale", "linear")),
    ("SVM linear C=0.01（调参后）", classify.svm_factory(0.01, "scale", "linear")),
    ("SVM RBF C=1 gamma=0.01（调参后）", classify.svm_factory(1.0, 0.01, "rbf")),
    ("SVM RBF C=1 gamma=scale", classify.svm_factory(1.0, "scale", "rbf")),
]
REG_MODELS = [
    ("Ridge alpha=10（调参后）", regress.ridge_factory(10.0)),
    ("Lasso alpha=0.03（调参后）", regress.lasso_factory(0.03)),
    ("回归树 d=3", regress.tree_factory(3)),
]


# --------------------------------------------------------------------------- #
# A1. 去掉 ca / thal
# --------------------------------------------------------------------------- #
def drop_features() -> dict:
    df_f, feats_f, X_f, y_f, yb_f = cmn.load_data()
    df_d, feats_d, X_d, y_d, yb_d = cmn.load_data(drop=DROP)
    cv = cmn.load_cv()

    cls_rows, cls_data = [], {}
    for name, factory in CLASS_MODELS:
        a = cmn.cv_classification(factory, X_f, yb_f, cv)["mean"]
        b = cmn.cv_classification(factory, X_d, yb_d, cv)["mean"]
        cls_data[name] = {"full": a, "drop": b}
        cls_rows.append([name, a["accuracy"], b["accuracy"], b["accuracy"] - a["accuracy"],
                         a["f1_pos"], b["f1_pos"], b["f1_pos"] - a["f1_pos"],
                         a["auc"], b["auc"], b["auc"] - a["auc"]])

    reg_rows, reg_data = [], {}
    for name, factory in REG_MODELS:
        a = cmn.cv_regression(factory, X_f, y_f, cv)["mean"]
        b = cmn.cv_regression(factory, X_d, y_d, cv)["mean"]
        reg_data[name] = {"full": a, "drop": b}
        reg_rows.append([name, a["r2"], b["r2"], b["r2"] - a["r2"],
                         a["mae"], b["mae"], b["mae"] - a["mae"],
                         a["rmse"], b["rmse"], b["rmse"] - a["rmse"]])

    cls_md = cmn.md_table(
        ["模型", "全 acc", "去两列 acc", "Δacc", "全 F1", "去两列 F1", "ΔF1",
         "全 AUC", "去两列 AUC", "ΔAUC"], cls_rows)
    reg_md = cmn.md_table(
        ["模型", "全 R²", "去两列 R²", "ΔR²", "全 MAE", "去两列 MAE", "ΔMAE",
         "全 RMSE", "去两列 RMSE", "ΔRMSE"], reg_rows)
    return {"cls_md": cls_md, "reg_md": reg_md, "cls": cls_data, "reg": reg_data,
            "features_full": feats_f, "features_drop": feats_d}


# --------------------------------------------------------------------------- #
# A2. 标准化 vs 不标准化
# --------------------------------------------------------------------------- #
def scaling() -> dict:
    _, feats, X, y_num, yb = cmn.load_data()
    cv = cmn.load_cv()
    grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    rows, data = [], {}
    for scale in (True, False):
        aucs, accs = [], []
        for C in grid:
            res = cmn.cv_classification(classify.svm_factory(C, "scale", "linear", scale), X, yb, cv)
            aucs.append(res["mean"]["auc"])
            accs.append(res["mean"]["accuracy"])
        best_i = int(np.argmax(aucs))
        at1 = grid.index(1.0)
        data[scale] = {"grid": grid, "auc": aucs, "acc": accs,
                       "best_C": grid[best_i], "best_auc": aucs[best_i],
                       "auc_at_1": aucs[at1], "acc_at_1": accs[at1]}
        rows.append(["标准化" if scale else "不标准化", grid[best_i], aucs[best_i],
                     aucs[at1], accs[at1]])
    md = cmn.md_table(["设置", "最优 C", "最优 CV AUC", "C=1 的 AUC", "C=1 的 accuracy"], rows)

    # 树对单调变换不变：加不加 StandardScaler，预测应逐例相同
    tree_scaled = cmn.cv_regression(regress.tree_factory(3), X, y_num, cv)["mean"]
    raw = []
    for tr, te in cv:
        est = cmn.make_pipe(DecisionTreeRegressor(max_depth=3, random_state=SEED), scale=False)
        est.fit(X[tr], y_num[tr])
        raw.append(est.predict(X[te]))
    scaled_pred = []
    for tr, te in cv:
        est = cmn.make_pipe(DecisionTreeRegressor(max_depth=3, random_state=SEED), scale=True)
        est.fit(X[tr], y_num[tr])
        scaled_pred.append(est.predict(X[te]))
    max_diff = float(max(np.max(np.abs(a - b)) for a, b in zip(raw, scaled_pred)))
    tree_md = cmn.md_table(
        ["检查项", "数值"],
        [["回归树 d=3 CV R²（不标准化）", tree_scaled["r2"]],
         ["回归树 d=3 逐例预测最大差异（标准化 vs 不标准化）", f"{max_diff:.3e}"],
         ["结论", "单调变换不改变树的划分，预测完全一致"]])
    return {"md": md, "data": data, "tree_md": tree_md, "tree_max_diff": max_diff,
            "tree_r2": tree_scaled["r2"]}


# --------------------------------------------------------------------------- #
# A3. 随机 80/20 划分的波动
# --------------------------------------------------------------------------- #
def split_variability(n_seeds: int = 10) -> dict:
    from sklearn.model_selection import train_test_split
    _, feats, X, y_num, yb = cmn.load_data()
    cv = cmn.load_cv()

    rows, tree_r2, ridge_r2, tree_mae = [], [], [], []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = train_test_split(X, y_num, test_size=0.2, random_state=seed)
        t = cmn.make_pipe(DecisionTreeRegressor(max_depth=3, random_state=SEED), scale=False)
        t.fit(Xtr, ytr)
        pt = t.predict(Xte)
        r = cmn.make_pipe(Ridge(alpha=10.0), scale=True)
        r.fit(Xtr, ytr)
        pr = r.predict(Xte)
        tr2 = float(1 - np.sum((yte - pt) ** 2) / np.sum((yte - yte.mean()) ** 2))
        rr2 = float(1 - np.sum((yte - pr) ** 2) / np.sum((yte - yte.mean()) ** 2))
        tree_r2.append(tr2)
        ridge_r2.append(rr2)
        tree_mae.append(float(np.mean(np.abs(yte - pt))))
        rows.append([seed, len(Xtr), len(Xte), tr2, rr2,
                     float(np.mean(np.abs(yte - pt)))])

    tree_r2 = np.array(tree_r2)
    ridge_r2 = np.array(ridge_r2)
    fixed_tree = cmn.cv_regression(regress.tree_factory(3), X, y_num, cv)["mean"]["r2"]
    fixed_ridge = cmn.cv_regression(regress.ridge_factory(10.0), X, y_num, cv)["mean"]["r2"]

    per_seed = cmn.md_table(["seed", "n_train", "n_test", "树 R²", "Ridge R²", "树 MAE"], rows)
    summary = cmn.md_table(
        ["模型", "最小 R²", "最大 R²", "均值", "标准差", "固定五折 R²"],
        [["回归树 d=3", tree_r2.min(), tree_r2.max(), tree_r2.mean(), tree_r2.std(ddof=1), fixed_tree],
         ["Ridge alpha=10", ridge_r2.min(), ridge_r2.max(), ridge_r2.mean(), ridge_r2.std(ddof=1), fixed_ridge]])
    return {"per_seed_md": per_seed, "summary_md": summary,
            "tree_r2": tree_r2.tolist(), "ridge_r2": ridge_r2.tolist(),
            "tree_mae": tree_mae,
            "tree_stats": {"min": float(tree_r2.min()), "max": float(tree_r2.max()),
                           "mean": float(tree_r2.mean()), "std": float(tree_r2.std(ddof=1))},
            "ridge_stats": {"min": float(ridge_r2.min()), "max": float(ridge_r2.max()),
                            "mean": float(ridge_r2.mean()), "std": float(ridge_r2.std(ddof=1))},
            "fixed_tree": fixed_tree, "fixed_ridge": fixed_ridge}


# --------------------------------------------------------------------------- #
# A4. 多项式 + 岭回归
# --------------------------------------------------------------------------- #
def polynomial(alpha_grid=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)) -> dict:
    _, feats, X, y_num, yb = cmn.load_data()
    cv = cmn.load_cv()

    def factory(alpha):
        return lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("sc", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    rows, data = [], {}
    for a in alpha_grid:
        res = cmn.cv_regression(factory(a), X, y_num, cv)
        data[a] = {"train_r2": res["train_r2"], **res["mean"]}
        rows.append([a, res["train_r2"], res["mean"]["r2"], res["mean"]["mae"], res["mean"]["rmse"]])
    best = max(data, key=lambda a: data[a]["r2"])
    linear = cmn.cv_regression(regress.ridge_factory(10.0), X, y_num, cv)["mean"]
    md = cmn.md_table(["alpha（多项式岭）", "训练 R²", "CV R²", "MAE", "RMSE"], rows)
    return {"md": md, "data": data, "best_alpha": best, "best": data[best],
            "linear": linear, "n_features": 104}


# --------------------------------------------------------------------------- #
# B1. 回归模型当分类器
# --------------------------------------------------------------------------- #
def regression_as_classifier() -> dict:
    _, feats, X, y_num, yb = cmn.load_data()
    cv = cmn.load_cv()
    reg = cmn.cv_regression(regress.ridge_factory(10.0), X, y_num, cv)
    oof = reg["oof"]
    auc = float(roc_auc_score(yb, oof))

    rows = []
    for t in (0.3, 0.5, 0.7, 1.0):
        pred = (oof >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0, 1]).ravel()
        rows.append([f"Ridge 预测值 >= {t}", t, accuracy_score(yb, pred),
                     precision_score(yb, pred, zero_division=0),
                     recall_score(yb, pred), f1_score(yb, pred),
                     tn, fp, fn, tp])

    svm = cmn.cv_classification(
        classify.svm_factory(0.01, "scale", "linear"), X, yb, cv)
    sm = svm["mean"]
    s_pred = svm["oof_pred"]
    tn, fp, fn, tp = confusion_matrix(yb, s_pred, labels=[0, 1]).ravel()
    rows.append(["SVM linear C=0.01（阈值 0）", 0.0, sm["accuracy"],
                 sm["precision_pos"], sm["recall_pos"], sm["f1_pos"], tn, fp, fn, tp])
    md = cmn.md_table(
        ["分数来源 / 阈值", "阈值", "accuracy", "precision", "recall", "F1",
         "TN", "FP", "FN", "TP"], rows)

    # ROC 曲线数据
    fpr_r, tpr_r, _ = roc_curve(yb, oof)
    fpr_s, tpr_s, _ = roc_curve(yb, svm["oof_score"])
    auc_md = cmn.md_table(
        ["分数", "AUC（连续分数）"],
        [["Ridge 回归预测值", auc], ["SVM 线性决策分数", sm["auc"]]])
    return {"md": md, "auc_md": auc_md, "auc": auc, "reg": reg, "oof": oof.tolist(),
            "svm_auc": float(sm["auc"]), "svm_acc": float(sm["accuracy"]),
            "svm_f1": float(sm["f1_pos"]), "svm_precision": float(sm["precision_pos"]),
            "svm_recall": float(sm["recall_pos"]),
            "roc": {"reg": [fpr_r.tolist(), tpr_r.tolist()],
                    "svm": [fpr_s.tolist(), tpr_s.tolist()]},
            "thresholds": [0.3, 0.5, 0.7, 1.0]}


# --------------------------------------------------------------------------- #
# B2. 分类与回归是否依赖同一批特征
# --------------------------------------------------------------------------- #
def consistency() -> dict:
    from scipy.stats import spearmanr
    _, feats, X, y_num, yb = cmn.load_data()
    imp = SimpleImputer(strategy="median").fit(X)
    Xi = imp.transform(X)
    Z = StandardScaler().fit_transform(Xi)

    svm = classify.svm_factory(0.01, "scale", "linear")().fit(X, yb)
    svm_coef = svm.named_steps["model"].coef_.ravel()
    ridge_coef = Ridge(alpha=10.0).fit(Z, y_num).coef_
    lasso_coef = Lasso(alpha=0.03, max_iter=50000).fit(Z, y_num).coef_
    tree_imp = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(Xi, y_num).feature_importances_

    rho_sr = spearmanr(svm_coef, ridge_coef)
    rho_sl = spearmanr(svm_coef, lasso_coef)
    rho_st = spearmanr(np.abs(svm_coef), tree_imp)
    rows = [[f, svm_coef[i], ridge_coef[i], lasso_coef[i], tree_imp[i]]
            for i, f in enumerate(feats)]
    rows.sort(key=lambda r: -abs(r[1]))
    md = cmn.md_table(["特征", "SVM 线性系数", "Ridge 系数", "Lasso 系数", "树重要性"], rows)
    corr_md = cmn.md_table(
        ["对比", "Spearman rho", "p 值"],
        [["SVM 系数 vs Ridge 系数", rho_sr.statistic, rho_sr.pvalue],
         ["SVM 系数 vs Lasso 系数", rho_sl.statistic, rho_sl.pvalue],
         ["|SVM 系数| vs 树重要性", rho_st.statistic, rho_st.pvalue]])
    return {"md": md, "corr_md": corr_md,
            "rho": {"svm_ridge": float(rho_sr.statistic),
                    "svm_lasso": float(rho_sl.statistic),
                    "svm_tree": float(rho_st.statistic)},
            "svm_coef": dict(zip(feats, svm_coef.tolist())),
            "ridge_coef": dict(zip(feats, ridge_coef.tolist()))}


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def make_figures(drop, split, poly, rac) -> list[str]:
    plt = cmn.setup_matplotlib()
    figs = []

    # A1 Δ 图
    names = list(drop["cls"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].barh(names, [drop["cls"][n]["drop"]["auc"] - drop["cls"][n]["full"]["auc"]
                         for n in names], color="C3")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("ΔAUC（去 ca/thal 减去 全特征）")
    axes[0].set_title("(a) 分类：去掉确诊后信息的代价")
    rnames = list(drop["reg"].keys())
    axes[1].barh(rnames, [drop["reg"][n]["drop"]["r2"] - drop["reg"][n]["full"]["r2"]
                          for n in rnames], color="C0")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("ΔR²（去 ca/thal 减去 全特征）")
    axes[1].set_title("(b) 回归：去掉确诊后信息的代价")
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "ablation_drop.png"))
    plt.close(fig)

    # A3 随机划分波动
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    seeds = np.arange(len(split["tree_r2"]))
    ax.scatter(seeds, split["tree_r2"], label="回归树 d=3", color="C0")
    ax.scatter(seeds, split["ridge_r2"], label="Ridge alpha=10", color="C2", marker="s")
    ax.axhline(split["fixed_tree"], color="C0", ls="--", lw=1,
               label=f"树固定五折 R²={split['fixed_tree']:.3f}")
    ax.axhline(split["fixed_ridge"], color="C2", ls=":", lw=1,
               label=f"Ridge 固定五折 R²={split['fixed_ridge']:.3f}")
    ax.set_xlabel("随机 80/20 划分的 seed")
    ax.set_ylabel("测试集 R²")
    ax.set_title("同一份数据、不同随机划分：R² 的波动")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "split_variability.png"))
    plt.close(fig)

    # A4 多项式岭
    alphas = sorted(poly["data"])
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(alphas, [poly["data"][a]["r2"] for a in alphas], "o-", label="CV R²（多项式）")
    ax.plot(alphas, [poly["data"][a]["train_r2"] for a in alphas], "s--", label="训练 R²（多项式）")
    ax.axhline(poly["linear"]["r2"], color="C2", ls=":",
               label=f"线性 Ridge R²={poly['linear']['r2']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha（对数尺度）")
    ax.set_ylabel("R²")
    ax.set_title("二次多项式（104 个特征）+ 岭回归")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "poly_ridge.png"))
    plt.close(fig)

    # B1 ROC + 阈值曲线
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(rac["roc"]["reg"][0], rac["roc"]["reg"][1],
                 label=f"Ridge 预测值（AUC={rac['auc']:.3f}）")
    axes[0].plot(rac["roc"]["svm"][0], rac["roc"]["svm"][1],
                 label=f"SVM 决策分数（AUC={rac['svm_auc']:.3f}）")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("(a) 同一批患者，两种分数")
    axes[0].legend(fontsize=8)
    oof = np.asarray(rac["oof"])
    yb = cmn.load_data()[4]
    ts = np.linspace(0.0, 2.5, 60)
    accs = [accuracy_score(yb, (oof >= t).astype(int)) for t in ts]
    f1s = [f1_score(yb, (oof >= t).astype(int)) for t in ts]
    precs = [precision_score(yb, (oof >= t).astype(int), zero_division=0) for t in ts]
    recs = [recall_score(yb, (oof >= t).astype(int)) for t in ts]
    axes[1].plot(ts, accs, label="accuracy")
    axes[1].plot(ts, f1s, label="F1")
    axes[1].plot(ts, precs, "--", label="precision")
    axes[1].plot(ts, recs, ":", label="recall")
    for t in rac["thresholds"]:
        axes[1].axvline(t, color="gray", lw=0.5, alpha=0.6)
    axes[1].set_xlabel("把回归预测值二值化的阈值")
    axes[1].set_ylabel("指标")
    axes[1].set_title("(b) 阈值如何移动 operating point")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "reg_as_classifier.png"))
    plt.close(fig)
    return [f.name for f in figs]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    out = {"tables": {}, "numbers": {}}

    def say(md):
        if verbose:
            cmn.print_md(md)

    drop = drop_features()
    out["tables"]["A1分类"] = drop["cls_md"]
    out["tables"]["A1回归"] = drop["reg_md"]
    out["numbers"]["drop"] = {
        name: {"cls_full": drop["cls"][name]["full"], "cls_drop": drop["cls"][name]["drop"],
               "reg_full": drop["reg"].get(name, {}).get("full", {}),
               "reg_drop": drop["reg"].get(name, {}).get("drop", {})}
        for name in drop["cls"]}
    if verbose:
        print("## 4.5-A1 去掉 ca 与 thal（确诊后信息）")
        print("分类：")
        say(drop["cls_md"])
        print("回归：")
        say(drop["reg_md"])

    sc = scaling()
    out["tables"]["A2标准化"] = sc["md"]
    out["tables"]["A2树不变性"] = sc["tree_md"]
    if verbose:
        print("## 4.5-A2 标准化 vs 不标准化")
        say(sc["md"])
        say(sc["tree_md"])

    split = split_variability()
    out["tables"]["A3逐seed"] = split["per_seed_md"]
    out["tables"]["A3汇总"] = split["summary_md"]
    out["numbers"]["split"] = {"tree": split["tree_stats"], "ridge": split["ridge_stats"],
                               "fixed_tree": split["fixed_tree"], "fixed_ridge": split["fixed_ridge"]}
    if verbose:
        print("## 4.5-A3 10 个随机 80/20 划分")
        say(split["per_seed_md"])
        say(split["summary_md"])

    poly = polynomial()
    out["tables"]["A4多项式"] = poly["md"]
    out["numbers"]["poly"] = {"best_alpha": poly["best_alpha"], **poly["best"],
                              "linear": poly["linear"], "n_features": poly["n_features"]}
    if verbose:
        print("## 4.5-A4 二次多项式 + 岭回归（104 个特征）")
        say(poly["md"])
        print(f"最优 alpha={poly['best_alpha']}，CV R²={poly['best']['r2']:.3f}；"
              f"线性 Ridge 为 {poly['linear']['r2']:.3f}\n")

    rac = regression_as_classifier()
    out["tables"]["B1回归当分类器"] = rac["md"]
    out["tables"]["B1AUC"] = rac["auc_md"]
    out["numbers"]["rac"] = {"auc": rac["auc"], "svm_auc": rac["svm_auc"],
                             "svm_acc": rac["svm_acc"], "svm_f1": rac["svm_f1"]}
    if verbose:
        print("## 4.6-Q3 回归模型当分类器")
        say(rac["auc_md"])
        say(rac["md"])

    cons = consistency()
    out["tables"]["B2一致性"] = cons["md"]
    out["tables"]["B2相关"] = cons["corr_md"]
    out["numbers"]["consistency"] = cons["rho"]
    if verbose:
        print("## B2 分类与回归是否依赖同一批特征")
        say(cons["corr_md"])
        say(cons["md"])

    figs = make_figures(drop, split, poly, rac)
    out["figures"] = figs
    return out


if __name__ == "__main__":
    run(verbose=True)
