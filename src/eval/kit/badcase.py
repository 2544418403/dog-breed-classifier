"""
badcase.py —— 错分样本的导出、分组与"给人看"的工程

================================================================================
本周的灵魂。前面所有指标都只是在说"模型不好"，只有 badcase 分析能说清楚
"哪一类不好、往哪个方向错、为什么"。
================================================================================

粗粝但正确的流程：

  1. 导出**全部**错分样本（不许只抽 20 张 —— 抽样会让你漏掉最值钱的模式）
  2. 按"错分方向" (true -> pred) 分组，而不是按类别分组
     —— 因为"havanese 被认成 yorkshire"和"yorkshire 被认成 havane"
        是**两个不同的病因**，混在一起看就永远归因不了
  3. 每组输出量化诊断（对称性 / 置信度 / 相对期望的偏离）
  4. 生成拼图让人眼确认（这一步不可替代）
  5. 归因假设 -> 去 attribution.py 做对照实验验证

本模块最重要的一个概念：对称性
--------------------------------------------------------------------
    假设 A->B 有 30 个，B->A 有 28 个  :  对称性 0.93 -> 双向都错
        => 这两个类在特征空间里真的挨着，**类间不可分**。
           解决办法是换更强的特征/更高分辨率/更多数据，加权重没用。

    假设 A->B 有 40 个，B->A 只有 2 个  :  对称性 0.05 -> 单向错误
        => B 少 A 多，模型在"猜不准时默认往多数类站"。
           这就是 **prior 塌缩**，加 class weight / 重采样最有效。

有这一个数，就能把"某些类差"这件模糊的事，拆成两种病因、两种药。
这一步是"会评估"和"会调模型"的分水岭。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ArrayLike = Union[np.ndarray, Sequence]


# ------------------------------------------------------------------------------
# 1. 跑预测，把概率全都留下
# ------------------------------------------------------------------------------
@torch.no_grad()
def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    return_path: bool = True,
    amp: bool = True,
) -> Dict[str, np.ndarray]:
    """跑一遍 loader，收集 probability / logit / y_true / (可选) path。

    **一定要存概率，不能只存 argmax**。理由很简单：
    错因的可观测证据（欠自信 vs 过度自信、 margin 多小）全都在概率里，
    只有 argmax 的话，你除了知道错了，什么都不知道 —— 那不叫分析。
    """
    model.eval()
    probs, ys, idxs, paths = [], [], [], []
    use_amp = amp and str(device).startswith("cuda")
    for batch in loader:
        x = batch[0]
        y = batch[1]
        it = iter(batch[2:])
        idx = next(it, None) if return_path else None
        pth = next(it, None)
        if return_path and pth is None and idx is not None:
            pth, idx = idx, None

        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
        if pth is not None:
            paths.extend(list(pth))
        elif idx is not None:
            idxs.append(idx.numpy())

    out = {
        "y_true": np.concatenate(ys),
        "probs": np.concatenate(probs),
    }
    if paths:
        out["path"] = np.asarray(paths, dtype=object)
    if idxs:
        out["index"] = np.concatenate(idxs)
    return out


def prob_stats(probs: np.ndarray) -> Dict[str, np.ndarray]:
    """从概率矩阵导出一组诊断量，全部按列拼成 array。

    margin     top1 - top2，越小说明越摇摆
    p_true     真类拿到的概率（低 = 模型压根没想到你）
    entropy    预测分布熵，衡量"模型有多不确定"
    """
    p_sorted = np.sort(probs, axis=1)
    top1 = p_sorted[:, -1]
    top2 = p_sorted[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1) / math.log(probs.shape[1])
    return {
        "conf": top1,
        "margin": top1 - top2,
        "entropy": entropy,
    }


# ------------------------------------------------------------------------------
# 2. 错分表：一张 tidy DataFrame 承载全部信息
# ------------------------------------------------------------------------------
def build_frame(
    pred: Dict[str, np.ndarray],
    meta: Optional[pd.DataFrame] = None,
    class_names: Optional[Sequence[str]] = None,
    meta_key: str = "path",
) -> pd.DataFrame:
    """把预测结果 + 元数据拼成一张大表。每一行一个样本，对了错了都在里面。

    meta 可以传 manifest（含 subject_id / plate_id / is_grayscale / exif ...），
    这是后面做归因的关键：没有这些协变量，你只能说"这张图错了"，
    没法说"灰度图的错误率显著更高"。
    """
    probs = pred["probs"]
    yt = pred["y_true"].astype(int)
    yp = probs.argmax(1)
    st = prob_stats(probs)
    K = probs.shape[1]
    names = [str(n) for n in (class_names or list(range(K)))]

    df = pd.DataFrame(
        {
            "y_true": yt,
            "y_pred": yp,
            "correct": (yt == yp).astype(int),
            "conf": st["conf"],
            "margin": st["margin"],
            "entropy": st["entropy"],
            "p_true": probs[np.arange(len(yt)), np.clip(yt, 0, K - 1)],
        }
    )
    df["class_true"] = [names[i] if 0 <= i < K else "?" for i in yt]
    df["class_pred"] = [names[i] for i in yp]
    df["error_dir"] = np.where(
        df.correct == 1, "✓正确",
        df.class_true + "→" + df.class_pred,
    )
    if "path" in pred:
        df.insert(0, "path", pred["path"])
    elif "index" in pred:
        df.insert(0, "index", pred["index"])

    if meta is not None and meta_key in df.columns and meta_key in meta.columns:
        df = df.merge(meta, on=meta_key, how="left", suffixes=("", "_meta"))
    return df


def error_table(df: pd.DataFrame, sort_by: str = "margin") -> pd.DataFrame:
    """只要错分样本，并按诊断最有价值的顺序排。

    默认按 margin **升序**：margin 小 = 模型在最摇摆的地方错，往往是类边界问题、
    标注歧义 —— 这类错误最有归因价值。

    反过来 cheapest 的做法是只看 top-loss 样本 sprint，但那会漏掉 silent
    majority。**两个方向都要留出口**，由调用方通过 sort_by 控制：
        margin   -> 最摇摆的错（类定义/边界问题）
        conf     -> 最自信的错（标注噪声 / 分布外）
        p_true   -> 模型最没想到的（难样本）
    """
    e = df[df.correct == 0].copy()
    asc = sort_by in ("margin", "p_true")
    return e.sort_values(sort_by, ascending=asc).reset_index(drop=True)


# ------------------------------------------------------------------------------
# 3. 错分方向组 —— 归因的主战场
# ------------------------------------------------------------------------------
def confusion_pairs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: Optional[Sequence[str]] = None,
    num_classes: Optional[int] = None,
) -> pd.DataFrame:
    """对每个 (true -> pred) 错误方向，给出诊断量。

    列的含义
    ----------
    n            该方向错了多少个
    rate         n / support_true  （这类里有多大比例被吸到 pred 去）
    n_reverse    反方向 (pred -> true) 错了多少个
    symmetry     min/前反之比，**接近 1 = 双向互错（真的难区分）；接近 0 = 单向（prior 主导）**
    mean_conf    该组错误样本的平均预测置信度 → 高 = 模型"理直气壮地错"
    adj_resid    列联表调整残差,>2 表示这个格子错的频次显著超出"边缘分布期望"
    """
    from .metrics import confusion_matrix

    yt, yp = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    K = num_classes or max(yt.max(), yp.max()) + 1
    names = [str(n) for n in (class_names or list(range(K)))]

    C = confusion_matrix(yt, yp, K)
    support = C.sum(axis=1)
    resid = adjusted_residuals(C)

    rows = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            n = int(C[i, j])
            if n == 0:
                continue
            n_rev = int(C[j, i])
            denom = max(n, n_rev)
            rows.append(
                {
                    "true": names[i],
                    "pred": names[j],
                    "n": n,
                    "rate": n / max(1, int(support[i])),
                    "n_reverse": n_rev,
                    "symmetry": (min(n, n_rev) / denom) if denom > 0 else np.nan,
                    "adj_resid": float(resid[i, j]) if not np.isnan(resid[i, j]) else np.nan,
                    "support_true": int(support[i]),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def adjusted_residuals(C: np.ndarray) -> np.ndarray:
    """混淆矩阵的**调整残差**（adjusted residual）。

        E_ij = R_i * C_j / N
        r_ij = (O_ij - E_ij) / sqrt(E_ij * (1 - R_i/N) * (1 - C_j/N))

    |r| > 1.96 ≈ 该格的频次偏离随机期望达 95% 显著水平。

    为什么要用它而不是直接看个数：个数多的格子多半只是因为类本身大。
    "havanese 被错得多" 很大程度是因为 havanese 本来就多 —— 残差把这个
    基线扣掉了，剩下的才是真正的**偏好性**。
    """
    C = np.asarray(C, dtype=np.float64)
    N = C.sum()
    R = C.sum(axis=1, keepdims=True)
    Cc = C.sum(axis=0, keepdims=True)
    E = R @ Cc / max(1.0, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(E * (1 - R / N) * (1 - Cc / N))
        resid = (C - E) / denom
    resid[~np.isfinite(resid)] = np.nan
    return resid


def group_errors(
    err: pd.DataFrame,
    true_col: str = "class_true",
    pred_col: str = "class_pred",
) -> Dict[str, pd.DataFrame]:
    """按错分方向切成一组一组，返回 {方向字符串: 该组 DataFrame}。

    组按"组内样本数降序"排列，刚好对应"先投入注意力到人多的组"的工作顺序。
    """
    groups = {}
    for (t, p), sub in err.groupby([true_col, pred_col], sort=True):
        groups[f"{t}→{p}"] = sub.sort_values("margin").reset_index(drop=True)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def error_summary(err: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """每个错分方向的概览：人数、对称性、平均置信度、平均 margin。"""
    if len(err) == 0:
        return pd.DataFrame()
    g = (
        err.groupby(["class_true", "class_pred"])
        .agg(
            n=("margin", "size"),
            mean_conf=("conf", "mean"),
            mean_margin=("margin", "mean"),
            mean_ptrue=("p_true", "mean"),
        )
        .sort_values("n", ascending=False)
        .head(top)
        .reset_index()
    )
    g["direction"] = g.class_true + "→" + g.class_pred
    # 反向数量
    rev = {(f"{a}→{b}"): 0 for a, b in zip(g.class_true, g.class_pred)}
    cnt = err.groupby(["class_true", "class_pred"]).size().to_dict()
    g["n_reverse"] = [int(cnt.get((b, a), 0)) for a, b in zip(g.class_true, g.class_pred)]
    g["symmetry"] = [
        min(n, r) / max(n, r) if max(n, r) > 0 else np.nan
        for n, r in zip(g.n, g.n_reverse)
    ]
    return g[["direction", "class_true", "class_pred", "n", "n_reverse",
              "symmetry", "mean_conf", "mean_margin", "mean_ptrue"]]


# ------------------------------------------------------------------------------
# 4. 置信度分层 —— 把"难样本是不是扎堆"量化
# ------------------------------------------------------------------------------
def confidence_strata(
    df: pd.DataFrame,
    bins: Sequence[float] = (0.0, 0.5, 0.7, 0.85, 0.95, 1.001),
) -> pd.DataFrame:
    """按预测置信度分层统计错误率。

    靠谱的模型应该单调：置信度越高错误率越低。如果不单调，说明
    模型的置信度不携带信息 —— 要么没校准，要么过拟合。
    """
    labels = []
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df.conf >= lo) & (df.conf < hi)
        n = int(m.sum())
        nerr = int((~df.loc[m, "correct"].astype(bool)).sum())
        labels.append(f"[{lo:.2f},{hi:.2f})")
        rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": n,
                     "n_err": nerr,
                     "error_rate": nerr / n if n else np.nan,
                     "mean_conf": float(df.loc[m, "conf"].mean()) if n else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 5. 人眼看图：错分样本拼图
# ------------------------------------------------------------------------------
def contact_sheet(
    rows: pd.DataFrame,
    root: Union[str, Path],
    out_path: Union[str, Path],
    cols: int = 5,
    rows_per_group: int = 10,
    thumb: Tuple[int, int] = (200, 200),
    max_groups: int = 6,
    group_col: str = "error_dir",
    title: Optional[str] = None,
) -> List[Path]:
    """按错分方向把样本拼成联系表（contact sheet），一张图一个方向。

    这一步存在的意义：**数字只能告诉你 where 错，只有看图能告诉你 why 错**。
    很多归因（"这根本不是狗脸是新标注标注员标错了"、"背景是笼子"、
    "图片被裁掉了头"）在指标上完全不可见，一眼就能看出来。

    返回生成的图片路径列表。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    root = Path(root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = rows.groupby(group_col)
    order = sorted(groups.groups.keys(), key=lambda k: -len(groups.get_group(k)))[:max_groups]

    made = []
    for gi, key in enumerate(order):
        sub = groups.get_group(key)
        sub = sub.head(cols * rows_per_group)
        n = len(sub)
        ncols = min(cols, max(1, n))
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.3))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for k, (_, r) in enumerate(sub.iterrows()):
            ax = axes[k]
            p = root / str(r["path"])
            try:
                img = mpimg.imread(str(p))
                ax.imshow(img)
            except Exception as ex:  # 读失败也要留位置，方便发现坏路径
                ax.text(0.5, 0.5, f"load fail\n{ex.__class__.__name__}",
                        ha="center", va="center", fontsize=7, color="crimson")
            conf = float(r["conf"])
            ax.set_title(f"true={r['class_true']}\npred={r['class_pred']} {conf:.2f}",
                         fontsize=6.5, color="#333", pad=2)
            for s in ax.spines.values():
                s.set_linewidth(1.2)
                s.set_edgecolor((conf, 0.25, 0.25) if conf > 0.8 else (0.3, 0.3, 0.3))
                s.set_visible(True)
        ttl = title or "错分样本"
        fig.suptitle(f"{ttl} · {key}  (n={len(groups.get_group(key))}, 展示 {n})",
                     fontsize=9, color="#111")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = out_path.parent / f"{out_path.stem}_g{gi:02d}{out_path.suffix}"
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        made.append(fp)
    return made
