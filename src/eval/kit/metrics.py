"""
metrics.py —— 分类评估的全部指标，一个都不少

================================================================================
先说一句最重要的话：指标不是算出来的装饰品，指标定义了你会交付哪个模型
================================================================================

本项目的 test 集是真实长尾（havanese 214 : yorkshire 23 ≈ 9.3 : 1）。在这种
分布下，一个"什么都不学、只预测众数"的模型就有 39% 的 accuracy。也就是说：

    accuracy 这个数字里，**有将近四成是白送的**

而它完全掩盖了"少数类 recall 可能是 20%"这件事。这就是为什么本周的第一条
纪律：**永远不要单独报 accuracy**。

各类指标到底在回答什么问题（这一段值得背下来，面试高频）：

    accuracy       总体猜对了多少          —— 被多数类主导，长尾下会骗人
    per-class R    某个类**漏检**了多少      —— 少放过一张远期 Dodgy 的代价 vs 误报的代价
    per-class P    报某个类时**错报**了多少  —— 误报成本高时看它（工业停机、误诊）
    recall 低      模型看不见这个类          —— 通常是样本少/特征弱
    precision 低   模型见了就报，尺度太松    —— 通常是类跪 priors 或类定义太宽

    要 U 你想要的组合 -> 用哪个平均：
      macro    各类平等加权，跟"关心每个类"的业务一致，**长尾下默认用它**
      micro    按样本加权，等价于 accuracy（多分类时），会掩盖少数类
      weighted 按 support 加权，介于两者之间，本质上还是多数类说了算

    AUROC(ROC-AUC)  排序质量：随机抽一正一负，模型给正的打分更高的概率。
                    对 **负样本数量爆炸** 不敏感 —— 所以它可能在长尾下很好看。
    PR-AUC(AP)      在 **正例很少** 时才露馅。是长尾/检索场景的正确选择。
                    >> 这就是为什么 AP 和 AUROC 必须同时报：
                       两者出现巨大分歧，就是在告诉你"模型靠多数类刷分"。

    ECE            置信度校准误差：模型说 0.9 的时候，真的有 90% 对吗？
                   现代网络（大容量 + label smoothing）普遍**过度自信**，
                    calibration 没校准好，你就不敢把概率接到"自动通过/自动拦截"
                    的下游流程里 —— 这是工业场景最常见的"指标很好但上不了线"。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, Sequence]


# ------------------------------------------------------------------------------
# 0. 基础工具
# ------------------------------------------------------------------------------
def _as_int(y: ArrayLike) -> np.ndarray:
    return np.asarray(y, dtype=np.int64).ravel()


def _num_classes(y_true: np.ndarray, y_prob: Optional[np.ndarray] = None, num_classes: Optional[int] = None) -> int:
    if num_classes is not None:
        return int(num_classes)
    if y_prob is not None and y_prob.ndim == 2:
        return int(y_prob.shape[1])
    return int(y_true.max()) + 1


def _safe_div(a, b, fill=0.0):
    """0/0 -> fill，而不是 nan / RuntimeWarning。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(np.broadcast(a, b).shape, float(fill), dtype=np.float64)
    np.divide(a, b, out=out, where=(np.asarray(b) != 0))
    return out


# ------------------------------------------------------------------------------
# 1. 混淆矩阵 —— 约定：行 = 真实，列 = 预测
# ------------------------------------------------------------------------------
def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """混淆矩阵 C[i, j] = 真 i 被预测成 j 的样本数。**行永远是真实**。

    约定必须先说清楚，因为 sklearn 的默认值也是行=真实，但很多画图代码
    （以及很多论文图）用的是列=真实，转置一下结论就全反了 —— 这是实际
    工作里最常见的"看起来很像 Machiavelli 实际是转置了"的事故。
    """
    yt, yp = _as_int(y_true), _as_int(y_pred)
    K = _num_classes(yt, None, num_classes)
    C = np.zeros((K, K), dtype=np.int64)
    for i, j in zip(yt, yp):
        if 0 <= i < K and 0 <= j < K:
            C[i, j] += 1
    return C


# ------------------------------------------------------------------------------
# 2. 每类指标
# ------------------------------------------------------------------------------
def per_class_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """每一类的 TP/FP/FN/TN 与 precision/recall/specificity/F1/jaccard。

    support 必须一起列出来 —— 没有 support 的 recall 是没有意义的：
    23 张图上的 recall=0.30 和 979 张图上的 recall=0.30，可信度差了两个数量级。
    """
    C = confusion_matrix(y_true, y_pred, num_classes)
    K = C.shape[0]
    tp = np.diag(C).astype(np.float64)
    fp = C.sum(axis=0).astype(np.float64) - tp      # 列和 - 对角 = 被报成该类但真不是
    fn = C.sum(axis=1).astype(np.float64) - tp      # 行和 - 对角 = 真是该类但没报出来
    tn = C.sum() - tp - fp - fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    jaccard = _safe_div(tp, tp + fp + fn)
    # balanced accuracy per class == (recall + specificity) / 2
    bal_acc = (recall + specificity) / 2.0

    names = list(class_names) if class_names is not None else [str(i) for i in range(K)]
    names = [str(n) for n in names][:K] + [str(i) for i in range(len(names), K)]

    return pd.DataFrame(
        {
            "class": names,
            "support": C.sum(axis=1).astype(int),
            "pred_count": C.sum(axis=0).astype(int),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "jaccard": jaccard,
            "balanced_acc": bal_acc,
        }
    ).set_index("class")


def kappa(y_true: ArrayLike, y_pred: ArrayLike, num_classes: Optional[int] = None) -> float:
    """Cohen's kappa：扣掉"蒙对了"之后的真实一致性。

    判断两个标注员的种族/class 是否一致时常用；此处用于衡量"预测和真值的
    一致性比随机好多少"。kappa<0.4 通常意味着模型基本在猜。
    """
    C = confusion_matrix(y_true, y_pred, num_classes)
    n = C.sum()
    po = np.diag(C).sum() / max(1, n)                       # 观察一致率 = accuracy
    pe = (C.sum(axis=0) @ C.sum(axis=1)) / max(1, n * n)    # 期望一致率（边缘独立假设下）
    return float(_safe_div(po - pe, 1 - pe, fill=0.0))


# ------------------------------------------------------------------------------
# 3. macro / micro / weighted 三种平均 —— 差别到底在哪
# ------------------------------------------------------------------------------
def average_precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    num_classes: Optional[int] = None,
    exclude_empty: bool = False,
) -> pd.DataFrame:
    """一次给出 macro / micro / weighted 三套 P、R、F1，让你亲眼看到它们的分歧。

    exclude_empty=True 时会跳过 support==0 的类。默认 False：宁可笑醒糊涂着读
    0 分也不要偷偷丢掉一个类 —— 静默丢弃会让"我加了这个方法 macro-F1 涨了"
    变成假象（其实是那个类被跳过了）。
    """
    df = per_class_metrics(y_true, y_pred, num_classes).reset_index()
    if exclude_empty:
        df = df[df.support > 0]

    support = df.support.to_numpy(dtype=np.float64)
    macro = {m: float(df[m].mean()) for m in ("precision", "recall", "f1")}
    weighted = {m: float(np.average(df[m], weights=support)) if support.sum() > 0 else 0.0
                for m in ("precision", "recall", "f1")}

    # micro 是从全局 TP/FP/FN 算的（不是对 per-class 再求平均） —— 这点很多人搞错
    tp = df.tp.sum()
    fp = df.fp.sum()
    fn = df.fn.sum()
    m_p = _safe_div(tp, tp + fp)
    m_r = _safe_div(tp, tp + fn)
    micro = {"precision": float(m_p), "recall": float(m_r),
             "f1": float(_safe_div(2 * m_p * m_r, m_p + m_r))}

    rows = []
    for name, d in (("macro", macro), ("micro", micro), ("weighted", weighted)):
        rows.append({"average": name, **d})
    out = pd.DataFrame(rows).set_index("average")
    out["accuracy"] = [np.nan, float(micro["recall"]), np.nan]  # 多分类下 micro-recall == accuracy
    return out


# ------------------------------------------------------------------------------
# 4. 概率输出相关：AUROC / PR-AUC（AP）
# ------------------------------------------------------------------------------
def _one_vs_rest(y_true: np.ndarray, k: int) -> np.ndarray:
    return (y_true == k).astype(np.int64)


def roc_pr_curves(y_true_bin: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """给一组 (0/1 真值, 连续打分)，返回 ROC 与 PR 曲线点。

    返回一个 tuple (fpr, tpr, recall_c, precision_c)。
    实现要点：必须按 **阈值** 从高到低扫并且在相同 score 处**一次性切分**，
    否则归并="/阻塞点绘制曲线会出错（这正是 sklearn 的 drop_intermediate 在
    处理的事情）。
    """
    yt = _as_int(y_true_bin)
    pos, neg = yt.sum(), (1 - yt).sum()
    if pos == 0 or neg == 0:
        # 全为一类 -> 曲线退化；返回 nan 让上层显式 display，而不是伪装成 0.5
        return (np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))

    order = np.argsort(-score, kind="mergesort")   # 从高分到低分
    yt_s, sc_s = yt[order], score[order]

    # 在 score 变化处才切一刀（同一分数的样本必须同时进/出，不能拆开）
    distinct = np.r_[np.where(np.diff(sc_s) != 0)[0], len(sc_s) - 1]

    tps = np.cumsum(yt_s)
    fps = np.cumsum(1 - yt_s)
    tps = tps[distinct]
    fps = fps[distinct]

    tpr = np.r_[0.0, tps / pos]
    fpr = np.r_[0.0, fps / neg]
    prec = np.r_[1.0, _safe_div(tps, tps + fps, fill=1.0)]

    # PR 曲线首点约定 recall=0, precision=1
    return fpr, tpr, tpr, prec


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """梯形面积。x 必须单调递增。"""
    if np.isnan(x).any() or len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    dx = np.diff(x)
    return float(np.sum(dx * (y[:-1] + y[1:]) / 2.0))


def auroc_pr_auc(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """逐类的 AUROC 与 PR-AUC(AP)，以及 macro / micro 平均。

    返回的 dict 里包含：
        auroc_macro / auroc_micro / ap_macro / ap_micro / ap_weighted

    为什么要 macro 而不是 micro AUROC：
      micro (=把 K 个二分类拧成一个大二分类) 会被样本多的类主导，
      而你在长尾任务上关心的恰恰是少类。
    """
    yt = _as_int(y_true)
    P = np.asarray(y_prob, dtype=np.float64)
    if P.ndim == 1:  # 二分类时允许只给正类概率
        P = np.stack([1 - P, P], axis=1)
    K = _num_classes(yt, P, num_classes)

    names = list(class_names) if class_names is not None else [str(i) for i in range(K)]
    names = [str(n) for n in names][:K] + [str(i) for i in range(len(names), K)]

    rows = []
    curves = {}
    for k in range(K):
        yb = _one_vs_rest(yt, k)
        npos = int(yb.sum())
        if npos == 0 or npos == len(yb):
            rows.append({"class": names[k], "support_pos": npos, "auroc": np.nan, "ap": np.nan})
            continue
        fpr, tpr, rec, prec = roc_pr_curves(yb, P[:, k])
        a_roc = _auc(fpr, tpr)
        # AP = Σ_k (R_k - R_{k-1}) · P_k，用**前向**差分。
        # （写成 -diff(rec) 那种后向差分是新手常犯的错，会把 AP 算到完全不对的量级）
        ap = float(np.sum(np.diff(rec, prepend=0.0) * prec))
        curves[names[k]] = (fpr, tpr, rec, prec)
        rows.append({"class": names[k], "support_pos": npos, "auroc": a_roc, "ap": ap})

    df = pd.DataFrame(rows).set_index("class")
    valid = df.dropna(subset=["auroc", "ap"])

    # --- micro：（进阶）把所有类的 OvR 摞起来，用总体TPR/FPR 求一次面积 ---
    micro = {}
    try:
        all_y = np.concatenate([_one_vs_rest(yt, k) for k in range(K)])
        all_s = np.concatenate([P[:, k] for k in range(K)])
        fpr, tpr, _, _ = roc_pr_curves(all_y, all_s)
        micro["auroc_micro"] = _auc(fpr, tpr)
    except Exception:
        micro["auroc_micro"] = float("nan")

    summary = {
        "auroc_macro": float(valid.auroc.mean()) if len(valid) else float("nan"),
        "ap_macro": float(valid.ap.mean()) if len(valid) else float("nan"),
        **micro,
    }
    support_pos = df.support_pos.to_numpy(dtype=np.float64)
    summary["ap_weighted"] = float(np.average(np.nan_to_num(df.ap), weights=support_pos)) if support_pos.sum() else float("nan")

    df.attrs["curves"] = curves
    return df, summary


# ------------------------------------------------------------------------------
# 5. 校准：ECE / MCE / Brier
# ------------------------------------------------------------------------------
def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 15,
    scheme: str = "equal_width",
) -> Dict[str, float]:
    """ECE：把预测置信度分箱，看"平均置信度"和"实际准确率"的加权平均差。

    ECE 大的模型：说 0.95 但其实只有 70% 对。这种模型 **不能直接用来做
    自动放行/自动拦截**，因为它的输出不可当概率用。

    现代网络（大容量 + label smoothing）普遍欠校准/过自信，所以这一项
    经常在本周的实验里露出来。
    """
    yt = _as_int(y_true)
    P = np.asarray(y_prob, dtype=np.float64)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == yt).astype(np.float64)

    if scheme == "equal_count":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        idx = np.clip(np.searchsorted(edges[1:-1], conf), 0, n_bins - 1)
    else:
        idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)

    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        gap = abs(acc_b - conf_b)
        ece += m.sum() / len(conf) * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "n": int(m.sum()), "acc": float(acc_b),
                     "conf": float(conf_b), "gap": float(gap)})

    brier = float(np.mean(np.sum((P - np.eye(P.shape[1])[yt]) ** 2, axis=1)))
    out = {"ece": float(ece), "mce": float(mce), "brier": brier}
    out["bins"] = pd.DataFrame(rows)
    return out


# ------------------------------------------------------------------------------
# 6. Bootstrap 置信区间 —— 小样本下没有 CI 的结论都是故事
# ------------------------------------------------------------------------------
def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    stratified: bool = True,
    digits: int = 4,
) -> Tuple[float, float, float]:
    """对任意 (y_true, y_score) -> float 的指标做 bootstrap，返回 (point, lo, hi)。

    为什么必须要有这个：
      test 里 yorkshire 只有 23 张，recall 从 0.30 掉到 0.26 可能只是 1 张图的差别。
      没有 CI 就去写"加了 class weight 让少类 recall 涨了 4pp"，是臆造。

    实现细节（决定了 CI 可不可信）：
      - stratified=True 按类别分层重抽样，否则少数类的重抽样样本里可能一个正例都没有，
        CI 会假性变宽（这是新手常踩的坑）。
      - percentile 法：简单稳健，小样本下比 normal-approx 好。
      - seed 固定 -> CI 可复现。
    """
    yt = _as_int(y_true)
    S = np.asarray(y_score)
    n = len(yt)

    point = float(metric_fn(yt, S))

    rng = np.random.default_rng(seed)
    if stratified:
        groups = [np.where(yt == c)[0] for c in np.unique(yt)]
    else:
        groups = [np.arange(n)]

    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        try:
            vals.append(float(metric_fn(yt[pick], S[pick])))
        except Exception:
            vals.append(np.nan)
    v = np.array(vals, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return point, float("nan"), float("nan")
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(v, [alpha * 100, (1 - alpha) * 100])
    return round(point, digits), round(float(lo), digits), round(float(hi), digits)


def paired_bootstrap_test(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """**成对** bootstrap：判断"改动 A 比 B 好"是真信号还是噪声。

    和上面的单样本 CI 不同，这里 A/B 在**同一批重抽样样本**上比较，
    消掉了共同的样本难度，因此更敏感也更诚实。

    输出 diff = A - B，ci_lo / ci_hi 是差的置信区间。
    判据：**ci 是否整体在 0 的同侧**。包含 0 -> 不能声称有提升。
    """
    yt = _as_int(y_true)
    A, B = np.asarray(score_a), np.asarray(score_b)
    n = len(yt)
    rng = np.random.default_rng(seed)
    groups = [np.where(yt == c)[0] for c in np.unique(yt)]

    point = float(metric_fn(yt, A) - metric_fn(yt, B))
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        yt_p = yt[pick]
        try:
            diffs.append(float(metric_fn(yt_p, A[pick]) - metric_fn(yt_p, B[pick])))
        except Exception:
            diffs.append(np.nan)
    d = np.array(diffs, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"diff": point, "ci_lo": np.nan, "ci_hi": np.nan, "p_gt0": np.nan}
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "diff": round(point, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        # 贝叶斯式读法：多次抽样里 A 严格优于 B 的比例
        "p_gt0": round(float((d > 0).mean()), 4),
    }


# ------------------------------------------------------------------------------
# 7. 聚类 bootstrap：当样本不独立时，普通 bootstrap 会骗你
# ------------------------------------------------------------------------------
def cluster_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    y_score: ArrayLike,
    clusters: ArrayLike,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    digits: int = 4,
) -> Tuple[float, float, float]:
    """**聚类** bootstrap：按 cluster（个体/病人/设备）整组重抽样。

    为什么必须有它
    --------------
    普通图片级 bootstrap 的隐含假设是"样本独立"。但本项目实测：
    90 个错分样本只来自 42 个个体，20 个个体的多帧**一起**错。
    同一个体的帧高度相关 -> 把它们当独立样本会**低估方差**，
    CI 假性收窄，进而把"碰运气"当成"真提升"。

    聚类 bootstrap 的做法：把每个 cluster 当成一个不可拆的"积木"，
    对 cluster 有放回重抽样，抽中的 cluster **带全部帧**进入。
    代价：CI 会变宽 —— 但那是**诚实**的宽度。

    实现细节：重抽样抽出的 cluster 数 = 原始 cluster 数（不是图片数），
    这样保持"集群层面的样本量"一致，这是 cluster bootstrap 的标准做法。
    """
    yt = _as_int(y_true)
    S = np.asarray(y_score)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    # 预先把每个 cluster 的行号收集好，避免循环里做布尔掩码（慢 10 倍以上）
    idx_by_cl = {c: np.where(cl == c)[0] for c in uniq}

    point = float(metric_fn(yt, S))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick_cl = rng.integers(0, len(uniq), len(uniq))
        pick = np.concatenate([idx_by_cl[uniq[c]] for c in pick_cl])
        try:
            vals.append(float(metric_fn(yt[pick], S[pick])))
        except Exception:
            vals.append(np.nan)
    v = np.array(vals, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return point, float("nan"), float("nan")
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(v, [alpha * 100, (1 - alpha) * 100])
    return round(point, digits), round(float(lo), digits), round(float(hi), digits)


def paired_cluster_bootstrap_test(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    clusters: ArrayLike,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """成对 + 聚类 bootstrap：判断 A/B 差异时同时尊重"样本不独立"这一事实。

    与 paired_bootstrap_test 的区别：重抽样单位是 cluster 而不是图片。
    本项目的结论都要过这一关才算数。
    """
    yt = _as_int(y_true)
    A, B = np.asarray(score_a), np.asarray(score_b)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    idx_by_cl = {c: np.where(cl == c)[0] for c in uniq}

    point = float(metric_fn(yt, A) - metric_fn(yt, B))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick_cl = rng.integers(0, len(uniq), len(uniq))
        pick = np.concatenate([idx_by_cl[uniq[c]] for c in pick_cl])
        try:
            diffs.append(float(metric_fn(yt[pick], A[pick]) - metric_fn(yt[pick], B[pick])))
        except Exception:
            diffs.append(np.nan)
    d = np.array(diffs, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"diff": point, "ci_lo": np.nan, "ci_hi": np.nan, "p_gt0": np.nan}
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": round(point, 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "p_gt0": round(float((d > 0).mean()), 4)}
