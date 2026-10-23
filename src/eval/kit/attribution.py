"""
attribution.py —— 把"归因假设"变成"可反驳的实验"

================================================================================
一个合格的归因，必须同时给出：
  1. 对照组定义（错分样本 vs **谁**比？）
  2. 统计量（错误率差 / 相对风险 / 提升幅度）
  3. 不确定度（bootstrap CI 或检验的 p 值）
  4. 方向（这个证据支持/否定了哪个假设）

缺任何一条，就是"讲了个故事"。
================================================================================

本模块内置三个最常见的对照：

  A. 组间对照        "灰度组的错误率是不是显著更高"（H3 采集偏移）
  B. 类内配对对照    "少数类塌缩是不是 prior 造成的" —— 用预测分布的长尾结构检验（H1）
  C. 特征空间对照    "这对类是不是真的在嵌入空间里挨着" —— 类心余弦相似度（H2）

"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .metrics import bootstrap_ci, paired_bootstrap_test  # noqa: F401


# ------------------------------------------------------------------------------
# A. 组间对照：masked 子集 vs 同类对照组
# ------------------------------------------------------------------------------
def group_compare(
    df: pd.DataFrame,
    mask: Sequence[bool],
    stratify_by: str = "class_true",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """比较 mask=True 子集与 mask=False 子集的错误率差。

    **分层是必须的，不是可选的**：
    如果灰度图恰好集中在 havanese（大类，错误率本来就低），
    全局比较会把"类构成差异"误读成"灰度的影响"。
    这里在每个 class 内部分别计算错误率差，再按类支持度加权平均 ——
    相当于一个分层加权的因果效应估计（stratified treatment effect）。

    返回：err_a, err_b, diff（带 CI）, relative_risk（带 CI）, n_a, n_b
    """
    df = df.copy()
    m = np.asarray(mask, dtype=bool)
    if len(m) != len(df):
        raise ValueError(f"mask 长度 {len(m)} != df 行数 {len(df)}")
    df["_m"] = m

    def _err_rate(sub: pd.DataFrame) -> Tuple[float, int]:
        if len(sub) == 0:
            return (float("nan"), 0)
        return (float(1.0 - sub["correct"].mean()), len(sub))

    # --- 分层加权：每层内算 diff，再按 (n_a+n_b) 加权 ---
    weights, diffs = [], []
    per_class = []
    for cls, sub in df.groupby(stratify_by):
        a, na = _err_rate(sub[sub._m])
        b, nb = _err_rate(sub[~sub._m])
        if na == 0 or nb == 0:
            continue
        diffs.append(a - b)
        weights.append(na + nb)
        per_class.append({"stratum": cls, "err_masked": a, "n_masked": na,
                          "err_ctrl": b, "n_ctrl": nb, "diff": a - b})
    w = np.asarray(weights, dtype=np.float64)
    diff_strat = float(np.sum(np.asarray(diffs) * w) / w.sum()) if w.sum() > 0 else float("nan")

    # --- 整体（未分层）作为对照参考 ---
    a_all, na_all = _err_rate(df[df._m])
    b_all, nb_all = _err_rate(df[~df._m])

    # --- bootstrap over classes（层内重抽样）---
    rng = np.random.default_rng(seed)
    boots = []
    classes = [p["stratum"] for p in per_class]
    for _ in range(n_boot):
        pick = rng.integers(0, len(classes), len(classes))
        ws, ds = [], []
        for i in pick:
            p = per_class[i]
            ws.append(p["n_masked"] + p["n_ctrl"])
            ds.append(p["err_masked"] - p["err_ctrl"])
        ws, ds = np.asarray(ws, dtype=float), np.asarray(ds, dtype=float)
        if ws.sum() > 0:
            boots.append(float((ds * ws).sum() / ws.sum()))
    boots = np.asarray(boots, dtype=np.float64)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan))

    return {
        "err_masked": round(a_all, 4),
        "err_ctrl": round(b_all, 4),
        "n_masked": na_all,
        "n_ctrl": nb_all,
        "diff_raw": round(a_all - b_all, 4),
        "diff_stratified": round(diff_strat, 4) if diff_strat == diff_strat else float("nan"),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "significant": bool(not (lo <= 0 <= hi)) if boots.size else False,
        "per_stratum": pd.DataFrame(per_class),
    }


def fisher_exact_2x2(mask: Sequence[bool], wrong: Sequence[bool]) -> Dict[str, float]:
    """2x2 卡方/Fisher 精确检验（小样本下必须用 Fisher，卡方只是渐近正确）。"""
    from scipy.stats import fisher_exact

    m = np.asarray(mask, dtype=bool)
    w = np.asarray(wrong, dtype=bool)
    a = int((m & w).sum())      # masked 且错
    b = int((m & ~w).sum())     # masked 且对
    c = int((~m & w).sum())     # 对照 且错
    d = int((~m & ~w).sum())    # 对照 且对
    odds, p = fisher_exact([[a, b], [c, d]])
    return {"a_mw": a, "b_mok": b, "c_cw": c, "d_cok": d,
            "odds_ratio": float(odds) if odds == odds else float("inf"),
            "p_value": float(p)}


# ------------------------------------------------------------------------------
# B. prior 塌缩检验：错误是不是系统性地流向多数类
# ------------------------------------------------------------------------------
def prior_collapse_test(
    df: pd.DataFrame,
    class_col: str = "class_true",
    train_support: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """检验"错分方向是否被训练集类先验吸引"。

    做法：把所有错分样本拿出来，看被错成的类（pred）在训练集里的占比，
    与"正确样本的类占比"比较 —— 若错误显著集中于大类，
    则 prior 塌缩假设成立。

    输出：
      err_pred_share  每个类**被错成**的比例
      train_share     每个类在训练集的比例
      lift            前者/后者，>1 = 这个类在吸收别人的错误（= 大类的磁铁效应）
      spearman        错分流入量 与 log(train n) 的相关性
    """
    from scipy.stats import spearmanr

    err = df[df.correct == 0]
    ok = df[df.correct == 1]
    if len(err) == 0:
        return {"err_pred_share": {}, "lift": {}, "spearman": float("nan")}

    pred_share = err["class_pred"].value_counts(normalize=True).to_dict()
    train_share = (
        {k: v / sum(train_support.values()) for k, v in train_support.items()}
        if train_support else ok["class_true"].value_counts(normalize=True).to_dict()
    )

    lift = {}
    n_in = err["class_pred"].value_counts().to_dict()
    n_train = dict(train_support) if train_support else ok["class_true"].value_counts().to_dict()
    for k, ps in pred_share.items():
        base = train_share.get(k, np.nan)
        lift[k] = float(ps / base) if base and base == base else np.nan

    classes = sorted(n_in.keys())
    xs = [math.log(max(1, n_train.get(c, 1))) for c in classes]
    ys = [n_in[c] for c in classes]
    rho = float(spearmanr(xs, ys).statistic) if len(classes) >= 3 else float("nan")

    return {
        "n_in": n_in,
        "err_pred_share": pred_share,
        "train_share": train_share,
        "lift": lift,
        "spearman_vs_log_n": rho,
    }


# ------------------------------------------------------------------------------
# C. 特征空间对照：这对类到底"长得有多像"
# ------------------------------------------------------------------------------
@torch.no_grad()  # type: ignore[misc]
def embed(model, loader, device: str = "cuda", layer: str = "penultimate") -> np.ndarray:
    """抽倒数第二层（分类头之前的 512/2048 维）特征。

    为什么用 penultimate：它是模型"眼里"的样本表示 —— 讨论类与类在
    特征空间的距离，必须用模型自己的几何，而不是人拍的 L2 距离。
    """
    feats = []
    # hook 住 classifier 前的 forward
    model.eval()

    def _penultimate(m, inp, out):
        # fc/classifier 的输入就是倒数第二层
        feats.append(inp[0].detach().float().cpu().numpy() if torch.is_tensor(inp[0]) else inp[0])

    head = None
    for name in ("fc", "head.fc", "classifier", "head"):
        if hasattr(model, name):
            head = getattr(model, name)
            break
    if head is None:
        raise AttributeError("找不到分类头（fc/head/classifier）")
    h = head.register_forward_hook(_penultimate)
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        model(x)
    h.remove()
    return np.concatenate(feats, axis=0)


def class_geometry(
    feats: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """类心余弦相似度矩阵：量化"哪些类在模型眼里是一坨"。

    诊断量：
      cos_centroid   类心间余弦 —— 高 = 类不可分的直接证据
      intra/inter    类内平均距离 / 类间平均距离 —— 越接近 1 越难分
    """
    y = np.asarray(y_true).astype(int)
    names = [str(n) for n in class_names]
    K = len(names)

    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    cents = np.stack([fn[y == k].mean(0) for k in range(K)])
    cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    cos = cents @ cents.T

    intra = []
    inter = []
    for k in range(K):
        d = 1.0 - fn[y == k] @ cents[k]
        intra.append(float(d.mean()))
    for i in range(K):
        for j in range(i + 1, K):
            inter.append(float(1.0 - cents[i] @ cents[j]))
    ratio = float(np.mean(intra) / max(1e-9, np.mean(inter)))

    df = pd.DataFrame(cos, index=names, columns=names)
    df.attrs["intra_inter_ratio"] = ratio
    return df, cos
