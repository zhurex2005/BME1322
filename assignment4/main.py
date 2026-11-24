"""一键复现 Assignment 4 的全部表与图。

用法（在仓库根目录）：

    python assignment4/main.py

产物：
  figures/*.png            全部图
  results/results.json     全部数值结果（便于核对）
  results/tables.md        全部 Markdown 表格
  results/mine.json        复现核对用的我的指标（交给 baselines.py --verify）
  results/baseline_verify.txt  课程基线脚本的核对输出（若数据目录存在）
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:  # 兼容 python assignment4/main.py 与 python -m assignment4.main
    from . import common as cmn
    from . import classify
    from . import regress
    from . import ablations
except ImportError:  # pragma: no cover
    import common as cmn
    import classify
    import regress
    import ablations

HERE = cmn.HERE
RESULTS = HERE / "results"


# --------------------------------------------------------------------------- #
# 数据探索
# --------------------------------------------------------------------------- #
def explore() -> dict:
    df, feats, X, y_num, y_bin = cmn.load_data()
    plt = cmn.setup_matplotlib()
    figs = []
    tables = {}
    numbers = {}

    counts = df["num"].value_counts().sort_index()
    rows = []
    cum = 0
    for k, v in counts.items():
        cum += v
        rows.append([int(k), int(v), v / len(df), cum / len(df)])
    tables["标签分布"] = cmn.md_table(["num", "例数", "占比", "累计占比"], rows)
    numbers["label_counts"] = {int(k): int(v) for k, v in counts.items()}
    numbers["positive_rate"] = float(y_bin.mean())

    missing = df[feats].isna().sum()
    miss_rows = [[f, int(missing[f])] for f in feats if missing[f] > 0]
    tables["缺失值"] = cmn.md_table(["特征", "缺失单元格数"], miss_rows)
    numbers["missing"] = {f: int(missing[f]) for f in feats if missing[f] > 0}

    summary = [[f, int(missing[f]), df[f].mean(), df[f].std(), df[f].min(), df[f].max()]
               for f in feats]
    tables["特征摘要"] = cmn.md_table(
        ["特征", "缺失", "均值", "标准差", "最小值", "最大值"], summary)

    key_feats = ["oldpeak", "thalach", "ca", "thal", "exang", "cp", "age"]
    rows = []
    for f in key_feats:
        vals = [df.loc[df["num"] == k, f].mean() for k in range(5)]
        rho = spearmanr(df[f].fillna(df[f].median()), y_num)
        rows.append([f, *vals, float(rho.statistic)])
    tables["分组均值"] = cmn.md_table(
        ["特征", "num=0", "num=1", "num=2", "num=3", "num=4", "Spearman rho"], rows)

    # 图 1：标签分布
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axes[0].bar(counts.index.astype(int), counts.values, color="C0")
    for x, v in zip(counts.index.astype(int), counts.values):
        axes[0].text(x, v + 2, str(v), ha="center", fontsize=9)
    axes[0].set_xlabel("num（血管狭窄程度，0=无）")
    axes[0].set_ylabel("例数")
    axes[0].set_title("(a) 五级标签分布（n=303）")
    binary = np.bincount(y_bin.astype(int), minlength=2)
    axes[1].bar(["num=0（无病）", "num>0（有病）"], binary, color=["C2", "C3"])
    for x, v in zip(range(2), binary):
        axes[1].text(x, v + 2, str(v), ha="center", fontsize=9)
    axes[1].set_ylabel("例数")
    axes[1].set_title("(b) 二分类后的类别分布")
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "label_distribution.png"))
    plt.close(fig)

    # 图 2：两个特征与标签
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, f, title in [
            (axes[0], "oldpeak", "(a) oldpeak（运动相对 ST 压低）"),
            (axes[1], "thalach", "(b) thalach（最大心率）")]:
        data = [df.loc[df["num"] == k, f].dropna().values for k in range(5)]
        bp = ax.boxplot(data, tick_labels=[str(k) for k in range(5)], showmeans=True)
        ax.set_xlabel("num")
        ax.set_ylabel(f)
        ax.set_title(title)
        for med in bp["medians"]:
            med.set_color("C3")
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "feature_vs_num.png"))
    plt.close(fig)

    # 图 3：各特征与 num 的 Spearman 相关
    rhos = []
    for f in feats:
        rhos.append((f, float(spearmanr(df[f].fillna(df[f].median()), y_num).statistic)))
    rhos.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh([t[0] for t in rhos], [t[1] for t in rhos],
            color=["C3" if t[1] > 0 else "C0" for t in rhos])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("与 num 的 Spearman 相关系数")
    ax.set_title("13 个特征与标签的单调关联")
    fig.tight_layout()
    figs.append(cmn.savefig(fig, "feature_target_corr.png"))
    plt.close(fig)

    numbers["feature_rho"] = {f: r for f, r in rhos}
    return {"tables": tables, "numbers": numbers, "figures": [f.name for f in figs]}


# --------------------------------------------------------------------------- #
# 复现核对：生成 mine.json 并调用课程脚本
# --------------------------------------------------------------------------- #
def baseline_check() -> dict:
    df, feats, X, y_num, y_bin = cmn.load_data()
    cv = cmn.load_cv()
    mine = {}

    crep = classify.reproduce(X, y_bin, cv)
    for r in crep["rows"]:
        mine.setdefault(f"cleveland/cls/{r['method']}", {})[r["metric"]] = float(r["mine"])
    rrep = regress.reproduce(X, y_num, cv)
    for r in rrep["rows"]:
        mine.setdefault(f"cleveland/reg/{r['method']}", {})[r["metric"]] = float(r["mine"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    mine_path = RESULTS / "mine.json"
    with open(mine_path, "w", encoding="utf-8") as f:
        json.dump(mine, f, indent=2, ensure_ascii=False)

    out = {"mine": mine, "verify_text": "", "max_abs_diff": max(crep["max_abs_diff"],
                                                               rrep["max_abs_diff"]),
           "n_over_tol": crep["n_over_tol"] + rrep["n_over_tol"]}

    data = cmn.data_dir()
    script = data / "baselines.py"
    if script.exists():
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, "baselines.py", "--verify", str(mine_path)],
            cwd=str(data), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env)
        text = proc.stdout + proc.stderr
        (RESULTS / "baseline_verify.txt").write_text(text, encoding="utf-8")
        out["verify_text"] = text
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Assignment 4：医学数据的分类与回归（Cleveland Heart Disease）")
    print("=" * 78)

    print("\n[0/5] 数据探索 ...")
    exp = explore()
    for k, md in exp["tables"].items():
        print(f"\n### {k}")
        cmn.print_md(md)

    print("\n[1/5] Task A 分类实验 ...")
    cls = classify.run(verbose=False)
    print("\n[2/5] Task B 回归实验 ...")
    reg = regress.run(verbose=False)
    print("\n[3/5] 4.5 消融与 4.6-Q3 ...")
    abl = ablations.run(verbose=False)

    print("[4/5] 复现核对（生成 mine.json 并运行 baselines.py --verify）...")
    base = baseline_check()

    # 汇总表格
    sections = [
        ("数据探索", exp["tables"]),
        ("Task A 分类", cls["tables"]),
        ("Task B 回归", reg["tables"]),
        ("4.5 消融 / 4.6-Q3 / 一致性", abl["tables"]),
    ]
    chunks = ["# Assignment 4 结果表（由 assignment4/main.py 自动生成）\n"]
    for title, tables in sections:
        chunks.append(f"\n## {title}\n")
        for name, md in tables.items():
            chunks.append(f"\n### {name}\n\n{md}\n")
    (RESULTS / "tables.md").write_text("\n".join(chunks), encoding="utf-8")

    figures = []
    for part in (exp, cls, reg, abl):
        for f in part.get("figures", []):
            if f not in figures:
                figures.append(f)

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "seed": cmn.SEED,
            "n_samples": 303,
            "n_features": 13,
            "cv": "assignment4-data/splits.json（固定五折）",
        },
        "exploration": exp["numbers"],
        "classify": cls["numbers"],
        "regress": reg["numbers"],
        "ablations": abl["numbers"],
        "baseline_check": {"max_abs_diff": base["max_abs_diff"],
                           "n_over_tol": base["n_over_tol"]},
        "figures": figures,
    }
    (RESULTS / "results.json").write_text(
        json.dumps(cmn.to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"完成：{len(figures)} 张图，表格写入 results/tables.md")
    print(f"复现核对最大绝对差值 = {base['max_abs_diff']:.4f}，超容差项数 = {base['n_over_tol']}")
    if base["verify_text"]:
        tail = [ln for ln in base["verify_text"].splitlines() if "全部在容差内" in ln
                or "超出容差" in ln or "项超出" in ln]
        for ln in tail:
            print(ln.strip())
        print("核对输出已保存到 results/baseline_verify.txt")
    else:
        print("未找到 assignment4-data/baselines.py，跳过课程脚本核对（mine.json 已生成）")
    print("=" * 78)
    return payload


if __name__ == "__main__":
    main()
