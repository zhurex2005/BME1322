"""Task B：同一批特征上的回归（回归树 vs 岭回归 / Lasso）。

对应讲义 4.3 节。运行方式：

    python assignment4/regress.py

内容包括：
  1. 复现核对：按 baselines.py 的固定超参重跑 5 个回归基线；
  2. 回归树的 max_depth 扫描（预剪枝），附叶节点数、树深与叶内样本量；
  3. 代价复杂度后剪枝（ccp_alpha）：候选序列、剪枝前后对比、树结构图；
  4. 岭回归与 Lasso 的 alpha 对数扫描、13 个特征的系数、Lasso 的稀疏性与稳定性；
  5. 回归结果总表（基线列 + 我的最优列 + 训练均值参考线）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, plot_tree

try:  # 兼容 python assignment4/regress.py 与 python -m assignment4.regress
    from . import common as cmn
    from . import baseline_reference as base_ref
except ImportError:  # pragma: no cover
    import common as cmn
    import baseline_reference as base_ref

SEED = cmn.SEED

DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, None)
RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
LASSO_ALPHAS = (0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 1.0, 10.0)
STABILITY_ALPHAS = (0.1, 0.2)
CCP_POINTS = 10
CCP_CURVE_POINTS = 40


def tree_factory(max_depth=None, ccp_alpha=0.0):
    return lambda: cmn.make_pipe(
        DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha,
                              random_state=SEED),
        scale=False,
    )


def ridge_factory(alpha):
    return lambda: cmn.make_pipe(Ridge(alpha=alpha), scale=True)


def lasso_factory(alpha):
    return lambda: cmn.make_pipe(Lasso(alpha=alpha, max_iter=50000), scale=True)


# --------------------------------------------------------------------------- #
# 1. 复现核对
# --------------------------------------------------------------------------- #
def reproduce(X, y, cv) -> dict:
    models = [
        ("Linear Regression", lambda: cmn.make_pipe(LinearRegression(), scale=False)),
        ("Ridge (alpha=1)", ridge_factory(1.0)),
        ("Lasso (alpha=0.01)", lasso_factory(0.01)),
        ("SVR (RBF, C=1)",
         lambda: cmn.make_pipe(__import__("sklearn.svm", fromlist=["SVR"]).SVR(C=1.0), scale=True)),
        ("Regression Tree (d=3)", tree_factory(3)),
    ]
    rows = []
    for name, factory in models:
        res = cmn.cv_regression(factory, X, y, cv)
        ref = base_ref.BASELINE[f"cleveland/reg/{name}"]
        for metric in ("r2", "mae", "rmse"):
            mine = res["mean"][metric]
            rows.append({"method": name, "metric": metric,
                         "baseline": ref[metric], "mine": mine,
                         "diff": mine - ref[metric]})
    md = cmn.md_table(
        ["方法", "指标", "基线", "我的", "差值"],
        [[r["method"], r["metric"], r["baseline"], r["mine"], f"{r['diff']:+.3f}"] for r in rows])
    diffs = [abs(r["diff"]) for r in rows]
    return {"rows": rows, "md": md, "max_abs_diff": float(max(diffs)),
            "n_over_tol": int(sum(d > base_ref.TOL for d in diffs)),
            "worst": max(rows, key=lambda r: abs(r["diff"]))}


# --------------------------------------------------------------------------- #
# 2. 深度扫描 + 叶内样本统计
# --------------------------------------------------------------------------- #
def depth_scan(X, y, cv) -> dict:
    rows, data = [], {}
    for d in DEPTHS:
        res = cmn.cv_regression(tree_factory(d), X, y, cv, structure=True)
        data[d] = {
            "leaves": res["leaves"], "depth": res["depth"],
            "train_r2": res["train_r2"], **res["mean"],
        }
        rows.append([d if d is not None else "None", res["leaves"], res["depth"],
                     res["train_r2"], res["mean"]["r2"], res["mean"]["mae"],
                     res["mean"]["rmse"]])
    best = max((k for k in data if k is not None), key=lambda k: data[k]["r2"])
    md = cmn.md_table(
        ["max_depth", "叶节点数", "树深", "训练 R²", "CV R²", "MAE", "RMSE"], rows)
    return {"data": data, "md": md, "best_depth": best, "best": data[best]}


def leaf_stats(X, y, cv) -> dict:
    """折内拟合的树：叶节点平均训练样本数、小叶子（<=2 样本）比例。"""
    rows, data = [], {}
    for d in DEPTHS:
        sizes, small = [], []
        for tr, te in cv:
            est = tree_factory(d)()
            est.fit(X[tr], y[tr])
            tree = est.named_steps["model"].tree_
            leaf_n = tree.n_node_samples[tree.children_left == -1]
            sizes.append(leaf_n.mean())
            small.append((leaf_n <= 2).mean())
        data[d] = {"mean_leaf": float(np.mean(sizes)), "small_leaf": float(np.mean(small))}
        rows.append([d if d is not None else "None", np.mean(sizes), np.mean(small)])
    return {"data": data,
            "md": cmn.md_table(["max_depth", "叶均训练样本数", "叶样本数<=2 的叶子占比"], rows)}


# --------------------------------------------------------------------------- #
# 3. 代价复杂度后剪枝
# --------------------------------------------------------------------------- #
def _ccp_path(X, y):
    imp = SimpleImputer(strategy="median").fit(X)
    tree = DecisionTreeRegressor(random_state=SEED).fit(imp.transform(X), y)
    path = tree.cost_complexity_pruning_path(imp.transform(X), y)
    return path.ccp_alphas, path.impurities, tree


def ccp_scan(X, y, cv) -> dict:
    alphas, impurities, full_tree = _ccp_path(X, y)
    positive = alphas[alphas > 0]
    max_alpha = float(positive.max())
    # 固定的可读网格（覆盖 0 到单节点树的最大 alpha）；CV 曲线在此数据上并不平滑，
    # 因此另外用密网格画曲线，报告里说明 alpha 的选取误差。
    candidates = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
                  0.1, 0.15, 0.2, 0.3]
    candidates = [a for a in candidates if a < max_alpha]
    candidates.append(round(max_alpha, 6))

    rows, data = [], {}
    for a in candidates:
        res = cmn.cv_regression(tree_factory(None, a), X, y, cv, structure=True)
        data[a] = {"leaves": res["leaves"], "depth": res["depth"],
                   "train_r2": res["train_r2"], **res["mean"]}
        rows.append([a, res["leaves"], res["depth"], res["train_r2"],
                     res["mean"]["r2"], res["mean"]["mae"], res["mean"]["rmse"]])
    best = max(data, key=lambda a: data[a]["r2"])
    md = cmn.md_table(
        ["ccp_alpha", "叶节点数", "树深", "训练 R²", "CV R²", "MAE", "RMSE"], rows)

    # 更密的曲线用于作图（0 到 0.15 是有效区间）
    dense = list(np.linspace(0.0, min(max_alpha, 0.15), CCP_CURVE_POINTS))
    curve = []
    for a in dense:
        res = cmn.cv_regression(tree_factory(None, float(a)), X, y, cv, structure=True)
        curve.append({"alpha": float(a), "r2": res["mean"]["r2"],
                      "mae": res["mean"]["mae"], "leaves": res["leaves"],
                      "train_r2": res["train_r2"]})
    return {"md": md, "data": data, "best_alpha": best, "best": data[best],
            "candidates": candidates, "curve": curve,
            "path": {"alphas": alphas, "impurities": impurities},
            "n_path": int(len(alphas))}


def tree_figure(X, y, alpha) -> dict:
    """用选定 alpha 在全表上拟合一棵树并画出结构；返回前两层分裂信息。"""
    imp = SimpleImputer(strategy="median").fit(X)
    Xi = imp.transform(X)
    model = DecisionTreeRegressor(random_state=SEED, ccp_alpha=alpha).fit(Xi, y)
    plt = cmn.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(model, feature_names=cmn.FEATURES, filled=True, rounded=True,
              impurity=False, fontsize=7, ax=ax)
    ax.set_title(f"回归树（ccp_alpha={alpha:g}，全表拟合，叶节点 {model.get_n_leaves()} 个，"
                 f"深度 {model.get_depth()}）")
    fig.tight_layout()
    path = cmn.savefig(fig, "tree_structure.png")
    plt.close(fig)

    t = model.tree_
    splits = []
    node = 0
    for level in range(2):
        if t.children_left[node] == -1:
            break
        splits.append({
            "level": level + 1,
            "feature": cmn.FEATURES[t.feature[node]],
            "threshold": float(t.threshold[node]),
            "n_left": int(t.n_node_samples[t.children_left[node]]),
            "n_right": int(t.n_node_samples[t.children_right[node]]),
            "value_left": float(t.value[t.children_left[node]].ravel()[0]),
            "value_right": float(t.value[t.children_right[node]].ravel()[0]),
        })
        node = t.children_left[node]
    return {"figure": path.name, "splits": splits,
            "leaves": int(model.get_n_leaves()), "depth": int(model.get_depth())}


# --------------------------------------------------------------------------- #
# 4. 岭回归 / Lasso
# --------------------------------------------------------------------------- #
def ridge_scan(X, y, cv) -> dict:
    rows, data = [], {}
    for a in RIDGE_ALPHAS:
        res = cmn.cv_regression(ridge_factory(a), X, y, cv)
        data[a] = {"train_r2": res["train_r2"], **res["mean"]}
        rows.append([a, res["train_r2"], res["mean"]["r2"], res["mean"]["mae"],
                     res["mean"]["rmse"]])
    best = max(data, key=lambda a: data[a]["r2"])
    md = cmn.md_table(["alpha", "训练 R²", "CV R²", "MAE", "RMSE"], rows)
    return {"data": data, "md": md, "best_alpha": best, "best": data[best]}


def lasso_scan(X, y, cv) -> dict:
    rows, data = [], {}
    for a in LASSO_ALPHAS:
        res = cmn.cv_regression(lasso_factory(a), X, y, cv)
        # 非零系数个数：折内拟合后取平均
        nnz = []
        for tr, te in cv:
            est = lasso_factory(a)()
            est.fit(X[tr], y[tr])
            coef = est.named_steps["model"].coef_
            nnz.append(int(np.sum(np.abs(coef) > 1e-8)))
        data[a] = {"train_r2": res["train_r2"], "nnz": float(np.mean(nnz)), **res["mean"]}
        rows.append([a, np.mean(nnz), res["train_r2"], res["mean"]["r2"],
                     res["mean"]["mae"], res["mean"]["rmse"]])
    best = max(data, key=lambda a: data[a]["r2"])
    md = cmn.md_table(["alpha", "非零系数个数", "训练 R²", "CV R²", "MAE", "RMSE"], rows)
    return {"data": data, "md": md, "best_alpha": best, "best": data[best]}


def coefficient_table(X, y) -> dict:
    """全表标准化后拟合，比较 OLS / Ridge / Lasso 的 13 个系数。"""
    imp = SimpleImputer(strategy="median").fit(X)
    Xi = imp.transform(X)
    from sklearn.preprocessing import StandardScaler
    Xi = StandardScaler().fit_transform(Xi)

    def coefs(model):
        model.fit(Xi, y)
        return np.asarray(model.coef_, dtype=float)

    cols = {
        "OLS": coefs(LinearRegression()),
        "Ridge a=0.01": coefs(Ridge(alpha=0.01)),
        "Ridge a=1": coefs(Ridge(alpha=1.0)),
        "Ridge a=10": coefs(Ridge(alpha=10.0)),
        "Lasso a=0.01": coefs(Lasso(alpha=0.01, max_iter=50000)),
        "Lasso a=0.03": coefs(Lasso(alpha=0.03, max_iter=50000)),
        "Lasso a=0.1": coefs(Lasso(alpha=0.1, max_iter=50000)),
    }
    rows = [[f, *[cols[c][i] for c in cols]] for i, f in enumerate(cmn.FEATURES)]
    rows.append(["非零个数", *[int(np.sum(np.abs(cols[c]) > 1e-8)) for c in cols]])
    md = cmn.md_table(["特征", *cols.keys()], rows, nd=3)
    return {"md": md, "coefs": {c: cols[c].tolist() for c in cols}}


def coefficient_paths(X, y) -> dict:
    """在 alpha 网格上计算系数路径（全表拟合，仅用于作图）。"""
    imp = SimpleImputer(strategy="median").fit(X)
    Xi = imp.transform(X)
    from sklearn.preprocessing import StandardScaler
    Xi = StandardScaler().fit_transform(Xi)
    ridge_grid = np.logspace(-3, 4, 70)
    lasso_grid = np.logspace(-3, 0, 50)
    ridge, lasso = [], []
    for a in ridge_grid:
        ridge.append(Ridge(alpha=a).fit(Xi, y).coef_.copy())
    for a in lasso_grid:
        lasso.append(Lasso(alpha=a, max_iter=50000).fit(Xi, y).coef_.copy())
    return {"ridge_alphas": ridge_grid, "lasso_alphas": lasso_grid,
            "ridge": np.array(ridge), "lasso": np.array(lasso)}


def lasso_stability(X, y, alphas=STABILITY_ALPHAS) -> dict:
    """Lasso 变量选择的稳定性：20 次 80% 子样本 + 5 折，统计入选频率。

    两个 alpha 使用同一批重抽样索引，因此可以逐特征比较惩罚强度的影响。
    """
    rng = np.random.default_rng(SEED)
    n = len(y)
    freq = {a: np.zeros(X.shape[1]) for a in alphas}
    runs = 0
    for _ in range(20):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        for a in alphas:
            est = lasso_factory(a)()
            est.fit(X[idx], y[idx])
            freq[a] += (np.abs(est.named_steps["model"].coef_) > 1e-8)
        runs += 1
    for tr, te in cmn.load_cv():
        for a in alphas:
            est = lasso_factory(a)()
            est.fit(X[tr], y[tr])
            freq[a] += (np.abs(est.named_steps["model"].coef_) > 1e-8)
        runs += 1
    for a in alphas:
        freq[a] /= runs
    order = np.argsort(-freq[alphas[-1]])
    rows = [[cmn.FEATURES[i], *[freq[a][i] for a in alphas]] for i in order]
    md = cmn.md_table(["特征", *[f"入选频率 @alpha={a:g}" for a in alphas]], rows)
    return {"freq": {a: freq[a].tolist() for a in alphas},
            "md": md, "runs": runs, "alphas": list(alphas)}


def correlation_analysis(df, features) -> dict:
    corr = df[features].corr()
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda x: -abs(x[2]))
    md = cmn.md_table(["特征对", "Pearson r"], [[a, b, r] for a, b, r in pairs[:8]])
    return {"corr": corr, "pairs": pairs, "md": md}


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def make_figures(depth, ccp, ridge, lasso, paths, beta, stability) -> list[str]:
    plt = cmn.setup_matplotlib()
    figs = []

    # 深度扫描
    ds = [d for d in DEPTHS]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(range(len(ds)), [depth["data"][d]["r2"] for d in ds], "o-", label="CV R²")
    ax.plot(range(len(ds)), [depth["data"][d]["train_r2"] for d in ds], "s--", label="训练 R²")
    ax.set_xticks(range(len(ds)), [str(d) if d is not None else "None" for d in ds])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R²")
    ax.set_title("回归树：深度扫描（训练 vs CV）")
    ax.legend()
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "tree_depth.png"))
    plt.close(fig)

    # ccp 曲线
    curve = ccp["curve"]
    xs = [c["alpha"] for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(xs, [c["r2"] for c in curve], "o-", label="CV R²")
    axes[0].plot(xs, [c["train_r2"] for c in curve], "s--", label="训练 R²")
    axes[0].axvline(ccp["best_alpha"], color="C3", ls=":", label=f"选定 alpha={ccp['best_alpha']:.4f}")
    axes[0].set_xlabel("ccp_alpha")
    axes[0].set_ylabel("R²")
    axes[0].set_title("(a) 后剪枝：R² 随 ccp_alpha 变化")
    axes[0].legend()
    axes[1].plot(xs, [c["leaves"] for c in curve], "o-")
    axes[1].axvline(ccp["best_alpha"], color="C3", ls=":")
    axes[1].set_xlabel("ccp_alpha")
    axes[1].set_ylabel("叶节点数（折均）")
    axes[1].set_title("(b) 树规模：叶节点数随 ccp_alpha 收缩")
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "ccp_path.png"))
    plt.close(fig)

    # 系数路径
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    handles = []
    for i, f in enumerate(cmn.FEATURES):
        h, = axes[0].plot(paths["ridge_alphas"], paths["ridge"][:, i], lw=1.2, color=colors[i])
        axes[1].plot(paths["lasso_alphas"], paths["lasso"][:, i], lw=1.2, color=colors[i])
        handles.append(h)
    for ax, title, xlabel in [
            (axes[0], "(a) 岭回归：系数平滑收缩，但不会精确为 0", "alpha（对数尺度）"),
            (axes[1], "(b) Lasso：系数被精确压到 0", "alpha（对数尺度）")]:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("标准化系数")
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.6)
    fig.legend(handles, cmn.FEATURES, loc="lower center", ncol=7, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    figs.append(cmn.savefig(fig, "coef_paths.png"))
    plt.close(fig)

    # 相关性热图
    corr = beta["corr"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(beta["corr"].columns)), beta["corr"].columns, rotation=90)
    ax.set_yticks(range(len(beta["corr"].columns)), beta["corr"].columns)
    ax.set_title("13 个特征的 Pearson 相关矩阵")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "correlation.png"))
    plt.close(fig)

    # Lasso 选择频率（较强惩罚的工作点，便于观察不稳定性）
    alpha_show = stability["alphas"][-1]
    freq = np.asarray(stability["freq"][alpha_show])
    order = np.argsort(-freq)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.barh([cmn.FEATURES[i] for i in order][::-1], freq[order][::-1], color="C0")
    ax.axvline(1.0, color="gray", ls=":")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"入选频率（{stability['runs']} 次重抽样）")
    ax.set_title(f"Lasso 变量选择的稳定性（alpha={alpha_show:g}）")
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "lasso_stability.png"))
    plt.close(fig)
    return [f.name for f in figs]


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

    rep = reproduce(X, y_num, cv)
    out["tables"]["复现核对"] = rep["md"]
    out["numbers"]["reproduce"] = {"max_abs_diff": rep["max_abs_diff"],
                                   "n_over_tol": rep["n_over_tol"], "worst": rep["worst"]}
    if verbose:
        print("## Task B - 1. 复现核对（固定超参，容差 0.01）")
        say(rep["md"])
        print(f"最大绝对差值 {rep['max_abs_diff']:.4f}，超出容差项数 {rep['n_over_tol']}\n")

    depth = depth_scan(X, y_num, cv)
    leaves = leaf_stats(X, y_num, cv)
    out["tables"]["深度扫描"] = depth["md"]
    out["tables"]["叶内样本"] = leaves["md"]
    out["numbers"]["tree_best_depth"] = {"depth": depth["best_depth"], **depth["best"]}
    if verbose:
        print("## Task B - 2. 回归树 max_depth 扫描（预剪枝）")
        say(depth["md"])
        print("叶内样本量：")
        say(leaves["md"])

    ccp = ccp_scan(X, y_num, cv)
    tree_info = tree_figure(X, y_num, ccp["best_alpha"])
    out["tables"]["ccp扫描"] = ccp["md"]
    out["numbers"]["ccp"] = {"best_alpha": ccp["best_alpha"], **ccp["best"],
                             "n_path": ccp["n_path"]}
    out["numbers"]["tree_structure"] = tree_info
    if verbose:
        print("## Task B - 3. 代价复杂度后剪枝（ccp_alpha）")
        say(ccp["md"])
        print(f"选定 alpha={ccp['best_alpha']:.6g}，叶节点 {ccp['best']['leaves']:.1f}，"
              f"CV R²={ccp['best']['r2']:.3f}\n")
        print("前两层分裂：")
        for s in tree_info["splits"]:
            print(f"  第{s['level']}层: {s['feature']} <= {s['threshold']:.3f} "
                  f"(左 {s['n_left']} 例, 均值 {s['value_left']:.2f}; "
                  f"右 {s['n_right']} 例, 均值 {s['value_right']:.2f})")
        print()

    ridge = ridge_scan(X, y_num, cv)
    lasso = lasso_scan(X, y_num, cv)
    coef = coefficient_table(X, y_num)
    paths = coefficient_paths(X, y_num)
    stability = lasso_stability(X, y_num)
    corr = correlation_analysis(df, feats)

    out["tables"]["ridge扫描"] = ridge["md"]
    out["tables"]["lasso扫描"] = lasso["md"]
    out["tables"]["系数"] = coef["md"]
    out["tables"]["lasso稳定性"] = stability["md"]
    out["tables"]["相关结构"] = corr["md"]
    out["numbers"]["ridge_best"] = {"alpha": ridge["best_alpha"], **ridge["best"]}
    out["numbers"]["lasso_best"] = {"alpha": lasso["best_alpha"], **lasso["best"]}
    out["numbers"]["coefs"] = coef["coefs"]
    out["numbers"]["corr_pairs"] = corr["pairs"][:8]
    out["numbers"]["lasso_freq"] = {str(a): dict(zip(cmn.FEATURES, stability["freq"][a]))
                                    for a in stability["alphas"]}
    if verbose:
        print("## Task B - 4. 岭回归 alpha 扫描")
        say(ridge["md"])
        print("## Task B - 5. Lasso alpha 扫描")
        say(lasso["md"])
        print("## Task B - 6. 全表标准化后的 13 个系数")
        say(coef["md"])
        print("## Task B - 7. 相关结构与 Lasso 选择稳定性")
        say(corr["md"])
        say(stability["md"])

    figs = make_figures(depth, ccp, ridge, lasso, paths, corr, stability)

    # 汇总表
    naive = cmn.naive_regression(y_num, cv)
    rows = [
        ["基线 回归树 (max_depth=3)", 0.29215, 0.712077, 1.013903],
        ["基线 岭回归 (alpha=1)", 0.52276, 0.634785, 0.842275],
        ["我的最优 回归树（预剪枝 d=%d）" % depth["best_depth"],
         depth["best"]["r2"], depth["best"]["mae"], depth["best"]["rmse"]],
        ["我的最优 回归树（ccp 后剪枝 a=%.4g）" % ccp["best_alpha"],
         ccp["best"]["r2"], ccp["best"]["mae"], ccp["best"]["rmse"]],
        ["我的最优 岭回归 (alpha=%g)" % ridge["best_alpha"],
         ridge["best"]["r2"], ridge["best"]["mae"], ridge["best"]["rmse"]],
        ["我的最优 Lasso (alpha=%g)" % lasso["best_alpha"],
         lasso["best"]["r2"], lasso["best"]["mae"], lasso["best"]["rmse"]],
        ["参考线 全预测训练折均值", naive["r2"], naive["mae"], naive["rmse"]],
    ]
    out["tables"]["回归总表"] = cmn.md_table(["模型", "R²", "MAE", "RMSE"], rows)
    out["numbers"]["naive"] = naive
    out["numbers"]["final_rows"] = rows
    if verbose:
        print("## Task B - 8. 回归结果总表（基线 vs 我的最优 vs 参考线）")
        say(out["tables"]["回归总表"])

    out["figures"] = figs
    out["figures"] += [tree_info["figure"]]
    return out


if __name__ == "__main__":
    run(verbose=True)
