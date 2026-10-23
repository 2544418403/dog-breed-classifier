"""
curves.py —— 本周所有图表的绘制

约定：图内文字一律英文（品种名本来就是英文，指标名用标准缩写），
中文解读写在报告里。这样彻底避开 matplotlib 中文字体缺字变方框的问题。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# 浅色主题：与 Week3 的浅色图表风格一致
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "grid.color": "#e6e6e6",
    "font.size": 8.5,
    "legend.frameon": False,
    "savefig.dpi": 150,
})

C_MAJOR = "#3b6ea5"
C_WARN = "#c0392b"
C_OK = "#2e7d5b"
C_GRAY = "#8a8a8a"


def _save(fig, out: Path):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------
def plot_confusion(
    C: np.ndarray,
    class_names: Sequence[str],
    out: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix (row = true)",
) -> Path:
    """混淆矩阵热力图。默认按行归一化 —— 因为想看的是"真值行往哪漏"。"""
    M = np.asarray(C, dtype=np.float64)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            M = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    K = len(class_names)
    fig, ax = plt.subplots(figsize=(1.15 * K + 2.2, 1.0 * K + 1.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, M.max()))
    ax.set_xticks(range(K)); ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title + ("  [row-normalized]" if normalize else ""))
    thr = (M.max() * 0.62) if M.max() > 0 else 0.5
    for i in range(K):
        for j in range(K):
            v = M[i, j]
            txt = f"{v:.2f}" if normalize else f"{int(v)}"
            if (normalize and v > 0.005) or (not normalize and v > 0):
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if v > thr else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out)


def plot_roc_pr(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    summary: Dict[str, float],
    out: Path,
    title_prefix: str = "",
) -> Path:
    """每个类的 ROC 与 PR 曲线，双图并排。

    PR 图里那根虚线是"随机猜测基线" = 正例占比 —— 少类的基线很低，
    所以 AP 必须和基线一起读才有意义。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(curves))))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", c=C_GRAY, lw=1, label="chance")
    ax1 = axes[1]
    for idx, (name, (fpr, tpr, rec, prec)) in enumerate(curves.items()):
        c = palette[idx % len(palette)]
        ax.plot(fpr, tpr, c=c, lw=1.5, label=f"{name} ({summary.get(name, {}).get('auroc', float('nan')):.3f})"
                if isinstance(summary.get(name, None), dict) else name)
        # PR 基线 = 该类正例占比
        base = None
        if len(prec) and not np.isnan(prec).all():
            ax1.plot(rec, prec, c=c, lw=1.5, label=name)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{title_prefix}ROC (one-vs-rest)".strip())
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=.3)

    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_ylim(0, 1.02); ax1.set_xlim(0, 1.02)
    ax1.set_title(f"{title_prefix}PR curve (AP)".strip())
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(alpha=.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_per_class(
    per_class: pd.DataFrame,
    out: Path,
    title: str = "Per-class precision / recall / F1 vs support",
) -> Path:
    """每类 P/R/F1 柱 + support 折线（右轴）。

    这个图一眼就能看出"少类是不是塌了"：柱子在最右边突然矮下去，
    而 support 曲线同时在谷底 —— 这就是 prior 塌缩的标准长相。
    """
    df = per_class.reset_index()
    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - w, df.precision, w, color="#8fb3d9", label="precision", edgecolor="white")
    ax.bar(x, df.recall, w, color=C_MAJOR, label="recall", edgecolor="white")
    ax.bar(x + w, df.f1, w, color="#20415f", label="F1", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["class"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.legend(fontsize=7.5, ncol=3)
    ax.grid(axis="y", alpha=.3)

    ax2 = ax.twinx()
    ax2.plot(x, df.support, "o--", color=C_WARN, lw=1.4, ms=5, label="support (test)")
    ax2.set_ylabel("support", color=C_WARN)
    ax2.tick_params(axis="y", labelcolor=C_WARN)
    for i, v in enumerate(df.support):
        ax2.annotate(int(v), (i, v), textcoords="offset points", xytext=(0, 6),
                     fontsize=7, color=C_WARN, ha="center")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def plot_metric_ladder(
    summary: Dict[str, float],
    out: Path,
    title: str = "Accuracy vs macro-F1 vs balanced accuracy",
) -> Path:
    """把"accuracy 虚高"这件事画成一根台阶。"""
    keys = ["accuracy", "balanced_acc", "macro_f1", "macro_recall", "worst_class_f1"]
    keys = [k for k in keys if k in summary]
    vals = [summary[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = [C_WARN, "#e08b3c", C_MAJOR, C_OK, "#7b4ea3"]
    bars = ax.barh(range(len(keys)), vals, color=colors[: len(keys)], edgecolor="white")
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5, color="#222")
    ax.set_xlabel("score")
    ax.set_title(title + "   (gap = how much accuracy hides)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_pair_diagnosis(pairs: pd.DataFrame, out: Path) -> Path:
    """错分方向的散点诊断：x = 对称性，y = 错误数。

    左上角（对称、量大）= 真难区分对；左下角（单向、量大）= prior 塌缩。
    这张图把"哪些类差"精确分诊成"两种不同的病"。
    """
    if len(pairs) == 0:
        return out
    df = pairs.copy()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    n = df["n"].to_numpy(dtype=float)
    s = ax.scatter(df.symmetry, df.n, s=40 + 6 * n, c=df.get("adj_resid", np.zeros(len(df))),
                   cmap="RdYlBu_r", alpha=.85, edgecolor="#444", linewidth=.4)
    for _, r in df.iterrows():
        ax.annotate(f"{r['true'][:4]}→{r['pred'][:4]}", (r.symmetry, r.n),
                    fontsize=6.8, xytext=(4, 5), textcoords="offset points", color="#333")
    ax.axvline(0.5, ls="--", c=C_GRAY, lw=1)
    ax.set_xlabel("symmetry  min(n_ab, n_ba) / max(n_ab, n_ba)\n→ 1 = mutually confusable (hard pair)   |   → 0 = one-way (prior collapse)")
    ax.set_ylabel("number of samples misclassified")
    ax.set_title("Error-direction diagnosis")
    ax.grid(alpha=.3)
    fig.colorbar(s, ax=ax, label="adjusted residual", fraction=.046, pad=.04)
    fig.tight_layout()
    return _save(fig, out)


def plot_reliability(bins: pd.DataFrame, ece: float, out: Path) -> Path:
    """可靠性图（校准曲线）：置信度分箱 vs 实际准确率。"""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", c=C_GRAY, lw=1, label="perfect")
    ax.bar(bins.conf, bins.acc, width=1.0 / max(3, len(bins)) * 0.9,
           color="#9ec5e8", edgecolor="#3b6ea5", label="accuracy")
    ax.plot(bins.conf, bins.conf, ls=":", c=C_WARN, lw=1.2, label="confidence")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram   ECE = {ece:.4f}")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_confidence_strata(strata: pd.DataFrame, out: Path) -> Path:
    """置信度分层错误率：单调递减才健康。"""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(strata["bin"], strata.error_rate, color=C_MAJOR, edgecolor="white")
    for i, r in strata.iterrows():
        ax.text(i, r.error_rate + 0.012, f"{r.error_rate:.2f}\nn={int(r.n)}",
                ha="center", fontsize=7, color="#333")
    ax.set_ylabel("error rate"); ax.set_xlabel("confidence bin")
    ax.set_title("Error rate by confidence stratum (should decrease)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_ablation(
    df: pd.DataFrame,
    metric: str,
    out: Path,
    ci_lo_col: Optional[str] = None,
    ci_hi_col: Optional[str] = None,
    title: Optional[str] = None,
) -> Path:
    """消融对比条形图，带 bootstrap 误差棒。"""
    d = df.copy()
    if metric not in d.columns:
        raise KeyError(f"缺少列 {metric}，现有: {list(d.columns)}")
    fig, ax = plt.subplots(figsize=(8.2, max(2.6, 0.55 * len(d) + 1.4)))
    y = np.arange(len(d))
    vals = d[metric].to_numpy(dtype=float)
    xerr = None
    if ci_lo_col and ci_hi_col and ci_lo_col in d.columns and ci_hi_col in d.columns:
        lo = d[ci_lo_col].to_numpy(dtype=float)
        hi = d[ci_hi_col].to_numpy(dtype=float)
        xerr = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
    base = vals[0] if len(vals) else 0
    colors = [C_GRAY] + [C_OK if v > base else (C_WARN if v < base else C_GRAY) for v in vals[1:]]
    ax.barh(y, vals, xerr=xerr, color=colors, edgecolor="white",
            error_kw=dict(ecolor="#555", capsize=3, lw=1))
    ax.set_yticks(y); ax.set_yticklabels(d["exp"].tolist())
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.006, i, f"{v:.4f}", va="center", fontsize=8, color="#222")
    ax.set_xlabel(metric)
    ax.set_title(title or f"Ablation · {metric}")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_group_compare(per_stratum: pd.DataFrame, out: Path, mask_name: str = "masked") -> Path:
    """分层对照：每个类里 mask 组 vs 对照组的错误率。"""
    if len(per_stratum) == 0:
        return out
    d = per_stratum.reset_index(drop=True)
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(x - w / 2, d.err_masked, w, color=C_WARN, label=mask_name, edgecolor="white")
    ax.bar(x + w / 2, d.err_ctrl, w, color="#9ec5e8", label="control", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(d["stratum"], rotation=25, ha="right")
    ax.set_ylabel("error rate")
    ax.set_title(f"Error rate: {mask_name} vs within-class control")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    for i, r in d.iterrows():
        ax.text(i - w / 2, r.err_masked + .01, f"n={int(r.n_masked)}", fontsize=6.5, ha="center", color=C_WARN)
        ax.text(i + w / 2, r.err_ctrl + .01, f"n={int(r.n_ctrl)}", fontsize=6.5, ha="center", color="#3b6ea5")
    fig.tight_layout()
    return _save(fig, out)


def plot_quality_boxes(err_df: pd.DataFrame, ok_df: pd.DataFrame, cols: Sequence[str], out: Path) -> Path:
    """错分样本 vs 正确样本的客观质量分布对比。"""
    cols = [c for c in cols if c in err_df.columns and c in ok_df.columns]
    if not cols:
        return out
    nc = len(cols)
    fig, axes = plt.subplots(1, nc, figsize=(2.5 * nc + 1, 3.6), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        a = err_df[c].dropna().to_numpy()
        b = ok_df[c].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            ax.axis("off"); continue
        # matplotlib >=3.9 把 boxplot 的 labels 参数改名为 tick_labels
        try:
            bp = ax.boxplot([a, b], tick_labels=["err", "ok"], widths=.55, patch_artist=True,
                            showfliers=False, medianprops=dict(color="#222"))
        except TypeError:
            bp = ax.boxplot([a, b], labels=["err", "ok"], widths=.55, patch_artist=True,
                            showfliers=False, medianprops=dict(color="#222"))
        for patch, col in zip(bp["boxes"], [C_WARN, "#9ec5e8"]):
            patch.set_facecolor(col); patch.set_alpha(.85)
        ax.set_title(c, fontsize=8.5)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Objective image quality: misclassified vs correct", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, out)
