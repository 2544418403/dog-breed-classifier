"""
scripts/run_seeds.py —— 多种子稳健性：把"训练随机性"这一层不确定性单独量化

================================================================================
这是一个很容易被漏掉、但会直接导致决策错误的层次
================================================================================

bootstrap CI 回答的是：**这个模型**在**这批测试样本**上的抽样噪声。
多种子回答的是：**这个训练配方**换一个随机种子会变成什么样。

两者是**独立的误差来源**。只报前者，等于假设"我这次训练的运气是必然的"。
本项目实测：同一个配方、同一份数据，三个种子的 test_acc 跨度 4.8pp
（0.901 / 0.940 / 0.949），**比绝大多数消融差异都大**。
也就是说：单种子下看起来显著的 +4pp，很可能只是抽签抽得好。

判据（这里采用的，比较保守）：
    配方的均值差异 必须超过 **种子间标准差** 才敢称之为改进。
    更严格一点：要求不同种子下的表现**不重叠**（min(A) > max(B)）。

用法：
    python scripts/run_seeds.py --bases baseline cw_sqrt gray_aug img224 --seeds 42 7 2024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import (  # noqa: E402
    DATASETS, OUTPUTS as OUT, FIGURES as FIG, WEIGHTS, LABEL_NAMES, ensure_dirs,
)

ensure_dirs()   # 幂等：artifacts/ figures/ weights/ datasets/ logs/ 缺哪个补哪个


from eval.kit import per_class_metrics

MANIFEST = DATASETS / "pet5" / "manifest.csv"

IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
MINORITY = [IDX["shiba_inu"], IDX["yorkshire_terrier"]]


def metrics(lg, y):
    yp = lg.argmax(1)
    pc = per_class_metrics(y, yp, 5, LABEL_NAMES)
    return {
        "accuracy": float((yp == y).mean()),
        "macro_f1": float(pc.f1.mean()),
        "macro_recall": float(pc.recall.mean()),
        "minority_recall": float(np.mean([pc.recall.iloc[i] for i in MINORITY])),
        "minority_precision": float(np.mean([pc.precision.iloc[i] for i in MINORITY])),
        "worst_class_f1": float(pc.f1.min()),
    }


def fname(base: str, seed: int) -> str:
    """seed=42 时沿用无后缀文件名（run_matrix 单种子时不加 --tag）。"""
    return base if seed == 42 else f"{base}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", nargs="+", default=["baseline", "cw_sqrt", "gray_aug", "img224"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = []
    for b in args.bases:
        for s in args.seeds:
            f = OUT / f"preds_{fname(b, s)}.npz"
            if not f.exists():
                print(f"  [缺] {f.name}")
                continue
            z = np.load(f, allow_pickle=True)
            m = metrics(z[f"{args.split}_logits"], z[f"{args.split}_y"].astype(int))
            rows.append({"exp": b, "seed": s, **{k: round(v, 4) for k, v in m.items()}})

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有任何种子结果可聚合")
        return

    pd.set_option("display.width", 200)
    print("=" * 100)
    print(f"多种子逐次结果  split={args.split}")
    print("=" * 100)
    print(df.to_string(index=False))

    agg = df.groupby("exp").agg(
        n_seeds=("seed", "size"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
        mf1_mean=("macro_f1", "mean"), mf1_std=("macro_f1", "std"),
        mf1_min=("macro_f1", "min"), mf1_max=("macro_f1", "max"),
        minprec_mean=("minority_precision", "mean"), minprec_std=("minority_precision", "std"),
    ).round(4).reset_index()

    print("\n" + "=" * 100)
    print("聚合（mean ± std，跨种子）")
    print("=" * 100)
    print(agg.to_string(index=False))

    base_row = agg[agg.exp == "baseline"]
    if len(base_row):
        b = base_row.iloc[0]
        print(f"\nbaseline: acc {b.acc_mean:.4f} ± {b.acc_std:.4f}  "
              f"macro-F1 {b.mf1_mean:.4f} ± {b.mf1_std:.4f}")
        print("\n相对 baseline 的判断：")
        for _, r in agg.iterrows():
            if r["exp"] == "baseline":
                continue
            d_acc = r.acc_mean - b.acc_mean
            d_f1 = r.mf1_mean - b.mf1_mean
            # 保守判据：均值差 > 两者种子标准差的合成，才算"超出噪声"
            noise = np.sqrt(b.mf1_std ** 2 + r.mf1_std ** 2) if (b.mf1_std == b.mf1_std and r.mf1_std == r.mf1_std) else np.nan
            no_overlap = r.mf1_min > b.mf1_max
            verdict = "真实改进" if (d_f1 > noise) else ("疑似噪声" if d_f1 > 0 else "无效/变坏")
            if no_overlap:
                verdict += "（三种子区间完全不重叠，最稳）"
            print(f"  {r['exp']:12s}  Δacc {d_acc:+.4f}   Δmacro-F1 {d_f1:+.4f}   "
                  f"噪声尺度 {noise if noise==noise else float('nan'):.4f}   -> {verdict}")

    df.to_csv(OUT / "seeds_raw.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT / "seeds_agg.csv", index=False, encoding="utf-8-sig")

    # ---- 图：每个配方的种子分布 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    order = [b for b in args.bases if b in set(df.exp)]
    for ax, col, title in ((axes[0], "accuracy", "Accuracy across 3 seeds"),
                           (axes[1], "macro_f1", "Macro-F1 across 3 seeds")):
        data = [df[df.exp == b][col].to_numpy() for b in order]
        bp = ax.boxplot(data, tick_labels=order, widths=.5, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", mfc="#c0392b", mec="#222", ms=5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#9ec5e8"); patch.set_alpha(.9)
        for i, b in enumerate(order, start=1):
            xs = np.random.default_rng(i).normal(i, 0.045, len(df[df.exp == b]))
            ax.scatter(xs, df[df.exp == b][col], s=22, color="#20415f", zorder=3, alpha=.85)
        ax.set_ylabel(col); ax.set_title(title + "\n(diamond = mean, dots = individual seeds)")
        ax.grid(axis="y", alpha=.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_seeds_box.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[完成] outputs/seeds_agg.csv, figures/fig_seeds_box.png")


if __name__ == "__main__":
    main()
