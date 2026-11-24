"""Task A：同一批特征上的 SVM 分类（线性核 vs RBF 核）。

对应讲义 4.2 节。运行方式：

    python assignment4/classify.py     # 打印全部表格并刷新 figures/

内容包括：
  1. 复现核对：按 baselines.py 的固定超参重跑 6 个分类基线，逐项比对；
  2. C=1 默认设置下两个核的对比（含逐类指标、硬标签/连续分数两种 AUC）；
  3. RBF 核的 C × gamma 网格（CV AUC 与训练-CV 差距），并选定 (C, gamma)；
  4. gamma 过拟合曲线（训练 AUC / CV AUC / 支持向量占比）；
  5. 线性核的 C 扫描，标准化与不标准化各一遍；
  6. 分类结果总表（基线列 + 我的最优列 + 多数类参考线）与最优模型的混淆矩阵。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:  # 兼容 python assignment4/classify.py 与 python -m assignment4.classify
    from . import common as cmn
    from . import baseline_reference as base_ref
except ImportError:  # pragma: no cover
    import common as cmn
    import baseline_reference as base_ref

SEED = cmn.SEED

# 与 baselines.py 完全一致的固定超参（用于复现核对）
BASELINE_CLASSIFIERS = [
    ("KNN (k=5)", lambda: cmn.make_pipe(KNeighborsClassifier(5), scale=True)),
    ("Naive Bayes", lambda: cmn.make_pipe(GaussianNB(), scale=False)),
    ("Logistic Regression", lambda: cmn.make_pipe(LogisticRegression(max_iter=3000), scale=True)),
    ("SVM (linear)", lambda: cmn.make_pipe(SVC(kernel="linear"), scale=True)),
    ("SVM (RBF)", lambda: cmn.make_pipe(SVC(), scale=True)),
    ("Decision Tree (d=3)",
     lambda: cmn.make_pipe(DecisionTreeClassifier(max_depth=3, random_state=SEED), scale=False)),
]

RBF_CS = (0.1, 1.0, 10.0, 100.0)
RBF_GAMMAS = (0.01, 0.1, 1.0, 10.0)
LINEAR_CS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMA_SWEEP = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def svm_factory(C: float = 1.0, gamma="scale", kernel: str = "rbf", scale: bool = True):
    """返回一条未拟合的 SVM Pipeline（中位数填补 -> 标准化可选 -> SVC）。"""
    return lambda: cmn.make_pipe(SVC(C=C, gamma=gamma, kernel=kernel), scale=scale)


# --------------------------------------------------------------------------- #
# 1. 复现核对
# --------------------------------------------------------------------------- #
def reproduce(X, y, cv) -> dict:
    rows = []
    for name, factory in BASELINE_CLASSIFIERS:
        res = cmn.cv_classification(factory, X, y, cv)
        ref = base_ref.BASELINE[f"cleveland/cls/{name}"]
        for metric, key in [
            ("accuracy", "accuracy"), ("precision", "precision_pos"),
            ("recall", "recall_pos"), ("f1", "f1_pos"), ("auc", "auc"),
        ]:
            mine = res["mean"][key]
            rows.append({
                "method": name, "metric": metric,
                "baseline": ref[metric], "mine": mine, "diff": mine - ref[metric],
            })

    md = cmn.md_table(
        ["方法", "指标", "基线", "我的", "差值"],
        [[r["method"], r["metric"], r["baseline"], r["mine"], f"{r['diff']:+.3f}"] for r in rows],
    )
    diffs = [abs(r["diff"]) for r in rows]
    return {"rows": rows, "md": md, "max_abs_diff": float(max(diffs)),
            "n_over_tol": int(sum(d > base_ref.TOL for d in diffs)),
            "worst": max(rows, key=lambda r: abs(r["diff"]))}


# --------------------------------------------------------------------------- #
# 2. C=1 两个核的对比
# --------------------------------------------------------------------------- #
def kernel_comparison(X, y, cv) -> dict:
    tables = {}
    rows, per_class = [], []
    results = {}
    for name, kernel in [("SVM (linear)", "linear"), ("SVM (RBF)", "rbf")]:
        res = cmn.cv_classification(svm_factory(1.0, "scale", kernel), X, y, cv, return_train=True)
        results[name] = res
        m = res["mean"]
        rows.append([name, m["accuracy"], m["precision_pos"], m["recall_pos"],
                     m["f1_pos"], m["f1_macro"], m["auc"], m["auc_hard"]])
        for cls, tag in [(1, "正例 num>0"), (0, "负例 num=0")]:
            if cls == 1:
                per_class.append([name, tag, m["precision_pos"], m["recall_pos"], m["f1_pos"]])
            else:
                per_class.append([name, tag, m["precision_neg"], m["recall_neg"], m["f1_neg"]])

    tables["kernel"] = cmn.md_table(
        ["模型", "accuracy", "precision", "recall", "F1", "F1(macro)",
         "AUC(连续)", "AUC(硬标签)"], rows)
    tables["per_class"] = cmn.md_table(
        ["模型", "类别", "precision", "recall", "F1"], per_class)

    # 硬标签 vs 连续分数：对同一模型给出两种口径，并检查排序是否翻转
    hard_soft = []
    for name in results:
        m = results[name]["mean"]
        hard_soft.append([name, m["auc_hard"], m["auc"], m["auc"] - m["auc_hard"]])
    tables["auc_hard_soft"] = cmn.md_table(
        ["模型", "硬标签 AUC", "连续分数 AUC", "差"], hard_soft)

    return {"results": results, "tables": tables,
            "hard_soft": hard_soft,
            "linear_auc": results["SVM (linear)"]["mean"]["auc"],
            "rbf_auc": results["SVM (RBF)"]["mean"]["auc"]}


# --------------------------------------------------------------------------- #
# 3. RBF 的 C × gamma 网格
# --------------------------------------------------------------------------- #
def rbf_grid(X, y, cv) -> dict:
    cells = {}
    for Cv in RBF_CS:
        for g in RBF_GAMMAS:
            res = cmn.cv_classification(svm_factory(Cv, g), X, y, cv, return_train=True)
            cells[(Cv, g)] = {
                "cv_auc": res["mean"]["auc"], "cv_acc": res["mean"]["accuracy"],
                "cv_f1": res["mean"]["f1_pos"], "cv_auc_std": res["std"]["auc"],
                "train_auc": res["train_mean"]["auc"],
                "gap": res["train_mean"]["auc"] - res["mean"]["auc"],
            }

    # gamma='scale' 的等效数值（标准化后每个特征方差为 1，故 1/(13*1)）
    scale_ref = 1.0 / (X.shape[1] * 1.0)

    header = ["C \\ gamma"] + [f"{g:g}" for g in RBF_GAMMAS] + ["scale*"]
    rows = []
    for Cv in RBF_CS:
        row = [Cv]
        for g in RBF_GAMMAS:
            row.append(cells[(Cv, g)]["cv_auc"])
        # gamma='scale' 在每个 C 下单独评估
        res = cmn.cv_classification(svm_factory(Cv, "scale"), X, y, cv, return_train=True)
        cells[(Cv, "scale")] = {
            "cv_auc": res["mean"]["auc"], "cv_acc": res["mean"]["accuracy"],
            "cv_f1": res["mean"]["f1_pos"], "cv_auc_std": res["std"]["auc"],
            "train_auc": res["train_mean"]["auc"],
            "gap": res["train_mean"]["auc"] - res["mean"]["auc"],
        }
        row.append(cells[(Cv, "scale")]["cv_auc"])
        rows.append(row)

    md = cmn.md_table(header, rows, nd=3)
    best_key = max(cells, key=lambda k: cells[k]["cv_auc"])
    return {"cells": cells, "md": md, "best_key": best_key,
            "best": cells[best_key], "gamma_scale_value": scale_ref}


# --------------------------------------------------------------------------- #
# 4. gamma 过拟合曲线（固定 C=1）
# --------------------------------------------------------------------------- #
def gamma_sweep(X, y, cv, C_value: float = 1.0) -> dict:
    """固定 C，扫描 gamma；表末附 gamma='scale' 基线行（可复现）。"""
    def evaluate(gamma):
        train_auc, test_auc, test_acc, sv_frac = [], [], [], []
        for tr, te in cv:
            est = svm_factory(C_value, gamma)()
            est.fit(X[tr], y[tr])
            svc = est.named_steps["model"]
            train_auc.append(cmn.roc_auc_score(y[tr], cmn._scores(est, X[tr])))
            test_auc.append(cmn.roc_auc_score(y[te], cmn._scores(est, X[te])))
            test_acc.append(cmn.accuracy_score(y[te], est.predict(X[te])))
            sv_frac.append(svc.n_support_.sum() / len(tr))
        return {"train_auc": float(np.mean(train_auc)),
                "cv_auc": float(np.mean(test_auc)),
                "cv_acc": float(np.mean(test_acc)),
                "sv_frac": float(np.mean(sv_frac))}

    rows, data = [], {}
    for g in GAMMA_SWEEP:
        data[g] = evaluate(g)
        d = data[g]
        rows.append([f"{g:g}", d["train_auc"], d["cv_auc"],
                     d["train_auc"] - d["cv_auc"], d["cv_acc"], d["sv_frac"]])

    scale = evaluate("scale")
    rows.append(["scale*", scale["train_auc"], scale["cv_auc"],
                 scale["train_auc"] - scale["cv_auc"], scale["cv_acc"], scale["sv_frac"]])

    md = cmn.md_table(
        ["gamma", "训练 AUC", "CV AUC", "训练-CV", "CV accuracy", "支持向量占比"], rows)
    return {"data": data, "scale": scale, "md": md, "C": C_value}


# --------------------------------------------------------------------------- #
# 5. 线性核的 C 扫描（标准化 / 不标准化）
# --------------------------------------------------------------------------- #
def linear_scan(X, y, cv, scale: bool = True) -> dict:
    rows, data = [], {}
    for Cv in LINEAR_CS:
        res = cmn.cv_classification(svm_factory(Cv, "scale", "linear", scale=scale),
                                    X, y, cv, return_train=True)
        data[Cv] = {
            "cv_auc": res["mean"]["auc"], "cv_acc": res["mean"]["accuracy"],
            "cv_f1": res["mean"]["f1_pos"], "train_auc": res["train_mean"]["auc"],
            "gap": res["train_mean"]["auc"] - res["mean"]["auc"],
        }
        rows.append([Cv, res["train_mean"]["auc"], res["mean"]["auc"],
                     res["train_mean"]["auc"] - res["mean"]["auc"],
                     res["mean"]["accuracy"], res["mean"]["f1_pos"]])
    best = max(data, key=lambda c: data[c]["cv_auc"])
    md = cmn.md_table(
        ["C", "训练 AUC", "CV AUC", "训练-CV", "CV accuracy", "CV F1"], rows)
    return {"data": data, "md": md, "best_C": best, "best": data[best],
            "scale": scale}


# --------------------------------------------------------------------------- #
# 6. 汇总表与混淆矩阵
# --------------------------------------------------------------------------- #
def final_table(X, y, cv, lin, rbf, naive) -> dict:
    rows = []
    # 基线固定超参
    for name, Cv, g in [("SVM (linear)", 1.0, "scale"), ("SVM (RBF)", 1.0, "scale")]:
        kernel = "linear" if "linear" in name else "rbf"
        res = cmn.cv_classification(svm_factory(Cv, g, kernel), X, y, cv)
        m = res["mean"]
        rows.append([f"基线 {name} (C=1)", m["accuracy"], m["precision_pos"],
                     m["recall_pos"], m["f1_pos"], m["auc"]])
    # 扫描后的最优
    rows.append([f"我的最优 SVM (linear) (C={lin['best_C']:g})",
                 lin["best"]["cv_acc"], None, None, lin["best"]["cv_f1"], lin["best"]["cv_auc"]])
    bk = rbf["best_key"]
    rows.append([f"我的最优 SVM (RBF) (C={bk[0]:g}, gamma={bk[1]})",
                 rbf["best"]["cv_acc"], None, None, rbf["best"]["cv_f1"], rbf["best"]["cv_auc"]])
    # 朴素参考线
    rows.append(["参考线 全预测多数类", naive["accuracy"], naive["precision_pos"],
                 naive["recall_pos"], naive["f1_macro"], naive["auc"]])

    # 补全最优行的 precision/recall（重新评估）
    best_lin = cmn.cv_classification(svm_factory(lin["best_C"], "scale", "linear"), X, y, cv)
    best_rbf = cmn.cv_classification(svm_factory(bk[0], bk[1]), X, y, cv)
    rows[2][2] = best_lin["mean"]["precision_pos"]
    rows[2][3] = best_lin["mean"]["recall_pos"]
    rows[3][2] = best_rbf["mean"]["precision_pos"]
    rows[3][3] = best_rbf["mean"]["recall_pos"]

    md = cmn.md_table(["模型", "accuracy", "precision", "recall", "F1", "AUC"], rows)
    return {"rows": rows, "md": md, "best_linear": best_lin, "best_rbf": best_rbf}


def confusion_block(res) -> dict:
    cm = res["confusion"]
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    md = cmn.md_table(
        ["", "预测 0", "预测 1"],
        [["真实 0", tn, fp], ["真实 1", fn, tp]],
    )
    metrics = cmn.md_table(
        ["指标", "值", "含义"],
        [["sensitivity (recall+)", sensitivity, "真实阳性中被检出的比例"],
         ["specificity (recall-)", specificity, "真实阴性中被排除的比例"],
         ["precision (PPV)", precision, "预测阳性中真正阳性的比例"],
         ["NPV", npv, "预测阴性中真正阴性的比例"],
         ["accuracy", (tp + tn) / cm.sum(), "总体正确率"]],
    )
    return {"cm": cm, "md": md, "metrics_md": metrics,
            "sensitivity": float(sensitivity), "specificity": float(specificity),
            "precision": float(precision), "npv": float(npv)}


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def _heatmap(plt, fig, ax, mat, xticks, yticks, title, cmap="viridis"):
    im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xticks)), [str(x) for x in xticks])
    ax.set_yticks(range(len(yticks)), [str(y) for y in yticks])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title(title)
    ax.grid(False)
    lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            frac = (mat[i, j] - lo) / (hi - lo + 1e-12)
            color = "white" if frac < 0.55 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color=color)
    return im


def make_figures(grid, gamma, lin_scaled, lin_unscaled) -> list[str]:
    plt = cmn.setup_matplotlib()
    paths = []

    # 图：RBF C×gamma 的 CV AUC 与 过拟合差距
    cv_mat = np.array([[grid["cells"][(Cv, g)]["cv_auc"] for g in RBF_GAMMAS] for Cv in RBF_CS])
    gap_mat = np.array([[grid["cells"][(Cv, g)]["gap"] for g in RBF_GAMMAS] for Cv in RBF_CS])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    im0 = _heatmap(plt, fig, axes[0], cv_mat, RBF_GAMMAS, RBF_CS, "(a) 五折 CV AUC")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    im1 = _heatmap(plt, fig, axes[1], gap_mat, RBF_GAMMAS, RBF_CS,
                   "(b) 训练 AUC - CV AUC（过拟合程度）", cmap="magma")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.tight_layout()
    paths.append(str(cmn.savefig(fig, "rbf_grid.png")))
    plt.close(fig)

    # 图：gamma 过拟合曲线
    gs = list(gamma["data"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(gs, [gamma["data"][g]["train_auc"] for g in gs], "o-", label="训练集 AUC")
    ax.plot(gs, [gamma["data"][g]["cv_auc"] for g in gs], "s-", label="五折 CV AUC")
    ax.set_xscale("log")
    ax.set_xlabel("gamma（对数尺度）")
    ax.set_ylabel("AUC")
    ax.set_title(f"RBF 核：gamma 增大时训练 AUC 与 CV AUC 的分叉（C={gamma['C']:g}）")
    ax.legend()
    ax.set_ylim(0.45, 1.02)
    fig.tight_layout()
    paths.append(str(cmn.savefig(fig, "gamma_overfit.png")))
    plt.close(fig)

    # 图：线性核 C 扫描，标准化 vs 不标准化
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(LINEAR_CS, [lin_scaled["data"][c]["cv_auc"] for c in LINEAR_CS], "o-",
            label="标准化后")
    ax.plot(LINEAR_CS, [lin_unscaled["data"][c]["cv_auc"] for c in LINEAR_CS], "s--",
            label="不标准化")
    ax.set_xscale("log")
    ax.set_xlabel("C（对数尺度）")
    ax.set_ylabel("五折 CV AUC")
    ax.set_title("线性核：C 扫描与标准化（AUC 对 C 的敏感性）")
    ax.legend()
    fig.tight_layout()
    paths.append(str(cmn.savefig(fig, "linear_c_scan.png")))
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    df, feats, X, y_num, y_bin = cmn.load_data()
    cv = cmn.load_cv()
    out = {"tables": {}, "numbers": {}}

    def say(md):
        if verbose:
            cmn.print_md(md)

    # 1 复现核对
    rep = reproduce(X, y_bin, cv)
    out["tables"]["复现核对"] = rep["md"]
    out["numbers"]["reproduce"] = {
        "max_abs_diff": rep["max_abs_diff"], "n_over_tol": rep["n_over_tol"],
        "worst": rep["worst"],
    }
    if verbose:
        print("## Task A - 1. 复现核对（固定超参，容差 0.01）")
        say(rep["md"])
        print(f"最大绝对差值 {rep['max_abs_diff']:.4f}，超出容差项数 {rep['n_over_tol']}\n")

    # 2 C=1 两个核
    kern = kernel_comparison(X, y_bin, cv)
    out["tables"]["核对比"] = kern["tables"]["kernel"]
    out["tables"]["逐类指标"] = kern["tables"]["per_class"]
    out["tables"]["AUC口径"] = kern["tables"]["auc_hard_soft"]
    if verbose:
        print("## Task A - 2. 两个核在 C=1 默认设置下的对比")
        say(kern["tables"]["kernel"])
        print("逐类指标：")
        say(kern["tables"]["per_class"])
        print("AUC 两种口径：")
        say(kern["tables"]["auc_hard_soft"])

    # 3 RBF 网格
    grid = rbf_grid(X, y_bin, cv)
    out["tables"]["RBF网格"] = grid["md"]
    out["numbers"]["rbf_best"] = {"key": list(grid["best_key"]), **grid["best"],
                                  "gamma_scale": grid["gamma_scale_value"]}
    if verbose:
        print("## Task A - 3. RBF 核 C × gamma 网格（单元格为五折 CV AUC）")
        say(grid["md"])
        print(f"最优 (C, gamma) = {grid['best_key']}，CV AUC = {grid['best']['cv_auc']:.3f}；"
              f"gamma='scale' 等效于 {grid['gamma_scale_value']:.4f}\n")

    # 4 gamma 过拟合
    gam = gamma_sweep(X, y_bin, cv)
    out["tables"]["gamma扫描"] = gam["md"]
    if verbose:
        print("## Task A - 4. gamma 扫描：训练 AUC vs CV AUC（C=1）")
        say(gam["md"])

    # 5 线性 C 扫描
    lin = linear_scan(X, y_bin, cv, scale=True)
    uns = linear_scan(X, y_bin, cv, scale=False)
    out["tables"]["线性C扫描"] = lin["md"]
    out["tables"]["未标准化C扫描"] = uns["md"]
    if verbose:
        print("## Task A - 5. 线性核 C 扫描（标准化）")
        say(lin["md"])
        print("## Task A - 5b. 线性核 C 扫描（不标准化）")
        say(uns["md"])
        print(f"标准化下最优 C={lin['best_C']:g}（AUC {lin['best']['cv_auc']:.3f}）；"
              f"不标准化下最优 C={uns['best_C']:g}（AUC {uns['best']['cv_auc']:.3f}），"
              f"AUC 损失 {lin['best']['cv_auc'] - uns['best']['cv_auc']:+.3f}\n")

    # 6 总表 + 混淆矩阵
    naive = cmn.naive_classification(y_bin, cv)
    final = final_table(X, y_bin, cv, lin, grid, naive)
    out["tables"]["分类总表"] = final["md"]
    conf = confusion_block(final["best_rbf"])
    out["tables"]["混淆矩阵"] = conf["md"]
    out["tables"]["混淆矩阵指标"] = conf["metrics_md"]
    if verbose:
        print("## Task A - 6. 分类结果总表（基线 vs 我的最优 vs 参考线）")
        say(final["md"])
        print("最优 RBF 模型的混淆矩阵（五折测试集汇总）：")
        say(conf["md"])
        print(conf["metrics_md"])

    figs = make_figures(grid, gam, lin, uns)
    out["figures"] = [f.split("\\")[-1].split("/")[-1] for f in figs]

    out["numbers"].update({
        "naive": naive,
        "linear_default": kern["linear_auc"],
        "rbf_default": kern["rbf_auc"],
        "linear_best": {"C": lin["best_C"], **lin["best"]},
        "linear_unscaled_best": {"C": uns["best_C"], **uns["best"]},
        "rbf_best_auc": grid["best"]["cv_auc"],
        "gamma_scan": gam["data"],
        "gamma_scale": gam["scale"],
        "confusion": conf["cm"].tolist(),
        "sensitivity": conf["sensitivity"], "specificity": conf["specificity"],
        "precision": conf["precision"], "npv": conf["npv"],
    })
    return out


if __name__ == "__main__":
    run(verbose=True)
