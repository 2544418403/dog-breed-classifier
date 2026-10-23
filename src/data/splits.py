"""
splits.py —— 划分：按"实体"而不是按"图片"

================================================================================
为什么这一页是整个数据管道里最重要的代码
================================================================================

随机划分（sklearn.train_test_split(df, ...)）隐含一个假设：
    **每一行 i.i.d. 独立同分布**

真实数据几乎从不满足。常见的"隐性重复单元"：

    医学    同一病人的多次拍片 / 同一台 CT 机的所有片子
    工业    同一卷钢带连续轧制的帧 / 同一批次（batch）的样品
    生物    同一条鱼连拍的多帧 / 同一块孔板的孔 / 同一天的成像
    遥感    相邻重叠切片的 patch
    视频    相邻帧

一旦重复单元被打散到 train 和 val 两边：
    - val 里那张图和 train 里某张图，除了传感器噪声几乎一模一样
    - 模型不需要学会"这是什么病/什么品种"，只需要记住"我见过这张图"
    - val_acc 于是变成了一个 **记忆力的度量**，而不是 **泛化能力的度量**
    - 上线后遇到真正没见过的病人/设备 -> 性能断崖

这就是"随机划分高估模型性能"的全部机制。注意它高估得非常安静：
loss 正常下降、曲线漂亮、AUC 很高，直到部署那天才暴露。

本模块提供：
    random_split()       按图片随机（对照组，用来量化泄漏有多严重）
    group_split()        按实体分组（正确的做法）
    stratified_group()   分组 + 尽量保持各类别比例
    assert_disjoint()    划分完成后强制校验（CI 里应该 assert 这个）
    group_kfold()        分组交叉验证
选择哪个 group_col 才是难点 —— 见 `suggest_group_column()` 与报告 §3。
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SEED = 42


# ------------------------------------------------------------------------------
# 1. 按图片随机划分（对照组）
# ------------------------------------------------------------------------------
def random_split(
    df: pd.DataFrame,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    names: Sequence[str] = ("train", "val", "test"),
    stratify_col: Optional[str] = "label",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """按行随机划分 —— **默认不要用**，除非你能证明每行互相独立。

    返回带 `split` 列的 df 副本。
    """
    df = df.copy()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))

    if stratify_col is not None and stratify_col in df.columns:
        parts: List[np.ndarray] = [np.array([], dtype=int) for _ in ratios]
        for _, grp in df.groupby(stratify_col, sort=True):
            g = rng.permutation(grp.index.to_numpy())
            cuts = _ratio_cuts(len(g), ratios)
            for k in range(len(ratios)):
                parts[k] = np.concatenate([parts[k], g[cuts[k] : cuts[k + 1]]])
    else:
        idx = rng.permutation(idx)
        cuts = _ratio_cuts(len(idx), ratios)
        parts = [idx[cuts[k] : cuts[k + 1]] for k in range(len(ratios))]

    split = pd.Series("unk", index=df.index, dtype=object)
    for name, p in zip(names, parts):
        split.loc[p] = name
    df["split"] = split.values
    return df


# ------------------------------------------------------------------------------
# 2. 按实体分组划分（正确的做法）
# ------------------------------------------------------------------------------
def group_split(
    df: pd.DataFrame,
    group_col: str,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    names: Sequence[str] = ("train", "val", "test"),
    stratify_col: Optional[str] = "label",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """按 `group_col` 分组划分：**同一实体的所有图片一定落在同一个 split**。

    算法（贪心 + 分层）：
      1. 把每个 group 聚成一个"超样本"，记录它的大小与主导类别
      2. 在 group 层面按主导类别分层抽样
      3. 贪心地把 group 塞进当前"样本数缺口最大"的那个 split，
         使最终样本数比例尽量贴近 ratios（而不是 group 数比例贴近）

    注意第 3 步：group 大小往往不均（有的病人拍了 20 张，有的只拍 1 张），
    按 group 个数切会让样本数严重偏离目标，必须按样本数贪心。
    """
    if group_col not in df.columns:
        raise KeyError(f"df 中找不到分组列 '{group_col}'，现有列: {list(df.columns)}")

    df = df.copy()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # --- group 级汇总 ---
    g = df.groupby(group_col, sort=True)
    sizes = g.size().astype(int)
    if stratify_col is not None and stratify_col in df.columns:
        # 一个实体理论上只属于一个类别；取众数以防脏数据
        dom = g[stratify_col].agg(lambda s: s.value_counts().idxmax())
    else:
        dom = pd.Series("__all__", index=sizes.index)

    gdf = pd.DataFrame({"group": sizes.index, "size": sizes.values, "dom": dom.values})
    gdf = gdf.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # 先整体打乱

    total = int(gdf["size"].sum())
    counts = [0.0] * len(ratios)

    order: List[np.ndarray] = [[] for _ in ratios]
    if stratify_col is not None and stratify_col in df.columns:
        # ------------------------------------------------------------------
        # 类别平衡贪心：让**每个类别**在三个 split 里的数量都尽量贴近目标比例。
        #
        # 为什么不能按"组内主导类别"分层（v1 的 bug）：
        #   当组本身是混合的（一块板上有所有类别），按主导类把组硬塞进
        #   "该类的桶"会导致某类只有 1~2 个大组 -> 整组原子分配 -> 比例失控，
        #   甚至 test 分到 0 张图（本项目实测复现过）。
        #
        # v2 策略：逐组扫描（大的先放），对每个候选桶计算加入该组后的
        #   "类别加权归一化缺口" sum_c cnt_c * max(0, target[r][c]-acc[r][c]) / target[r][c]
        #   选缺口最大的桶 —— 同时兼顾总量比例和类别比例。
        # ------------------------------------------------------------------
        class_totals = df.groupby(stratify_col).size().to_dict()
        classes = sorted(class_totals)
        # 每组的类别构成
        comp = df.groupby([group_col, stratify_col]).size().unstack(fill_value=0)
        comp = comp.reindex(columns=classes, fill_value=0)

        target = [{c: r * class_totals[c] for c in classes} for r in ratios]
        acc = [{c: 0.0 for c in classes} for _ in ratios]
        size_acc = [0.0] * len(ratios)
        size_target = [r * total for r in ratios]

        # 大组先放（大组自由度小，先安排能显著降不平衡）
        gorder = gdf.sort_values("size", ascending=False)
        buckets: List[List[str]] = [[] for _ in ratios]
        for _, row in gorder.iterrows():
            gid, gsize = row["group"], float(row["size"])
            cnt = comp.loc[gid]
            best_k, best_score = 0, -1e18
            for k in range(len(ratios)):
                # 总量缺口（归一化）
                size_def = (size_target[k] - size_acc[k]) / max(1.0, size_target[k])
                # 类别缺口：该组带来的每类样本，按目标归一化后求和（只算还没填满的）
                cls_def = 0.0
                for c in classes:
                    t = target[k][c]
                    if t <= 0:
                        continue
                    d = (t - acc[k][c]) / t
                    cls_def += cnt.get(c, 0) * max(0.0, d) / max(1.0, t) * len(classes)
                score = size_def * gsize / max(1.0, total * ratios[k]) + cls_def / max(1.0, gsize)
                if score > best_score:
                    best_score, best_k = score, k
            buckets[best_k].append(gid)
            size_acc[best_k] += gsize
            for c in classes:
                acc[best_k][c] += float(cnt.get(c, 0))
        for k in range(len(ratios)):
            order[k] = np.array(buckets[k])
    else:
        for _, row in gdf.iterrows():
            deficit = [target[k] - counts[k] for k in range(len(ratios))]
            k = int(np.argmax(deficit))
            order[k] = np.append(order[k], row["group"])
            counts[k] += row["size"]

    mapping = {}
    for name, arr in zip(names, order):
        for gg in arr:
            mapping[gg] = name
    df["split"] = df[group_col].map(mapping).values
    return df


def group_kfold(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    """分组 K 折：yield (train_df, val_df)。用于小数据集上更稳的结论。"""
    from sklearn.model_selection import GroupKFold

    df = df.copy()
    gkf = GroupKFold(n_splits=n_splits)
    X = np.zeros(len(df))
    for tr_i, va_i in gkf.split(X, groups=df[group_col].values):
        yield df.iloc[tr_i], df.iloc[va_i]


# ------------------------------------------------------------------------------
# 3. 划分后的强制校验
# ------------------------------------------------------------------------------
def assert_disjoint(df: pd.DataFrame, split_col: str, group_col: str) -> None:
    """断言任意两个 split 之间**没有共享实体**。不满足直接抛异常。

    这一行应该写进 CI / 单元测试里。数据规模一大，靠肉眼是看不出来的。
    """
    if split_col not in df.columns:
        raise KeyError(f"缺少列 '{split_col}'")
    if group_col not in df.columns:
        raise KeyError(f"缺少列 '{group_col}'")

    per = {s: set(g) for s, g in df.groupby(split_col)[group_col]}
    keys = list(per)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = per[keys[i]] & per[keys[j]]
            if overlap:
                sample = list(overlap)[:5]
                raise AssertionError(
                    f"数据泄漏：{group_col} 同时出现在 '{keys[i]}' 和 '{keys[j]}' 中，"
                    f"共 {len(overlap)} 个，例如 {sample}"
                )


def overlap_report(df: pd.DataFrame, split_col: str, group_col: str) -> pd.DataFrame:
    """给出两两 split 之间的实体重叠统计（用于对比"随机划分 vs 分组划分"）。"""
    per = {s: set(g) for s, g in df.groupby(split_col)[group_col]}
    keys = sorted(per)
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = per[keys[i]], per[keys[j]]
            ov = a & b
            rows.append(
                {
                    "split_a": keys[i],
                    "split_b": keys[j],
                    "groups_a": len(a),
                    "groups_b": len(b),
                    "shared_groups": len(ov),
                    "shared_ratio_of_b": len(ov) / max(1, len(b)),
                }
            )
    return pd.DataFrame(rows)


def split_summary(df: pd.DataFrame, split_col: str = "split", label_col: str = "label") -> pd.DataFrame:
    """每个 split 的样本数 + 各类别分布，一眼看出划分是否失衡。"""
    rows = []
    for s, sub in df.groupby(split_col, sort=True):
        d = {"split": s, "n": len(sub)}
        if label_col in sub.columns:
            vc = sub[label_col].value_counts().sort_index()
            for k, v in vc.items():
                d[f"cls_{k}"] = int(v)
        rows.append(d)
    return pd.DataFrame(rows).set_index("split")


# ------------------------------------------------------------------------------
# 4. 该用哪一列做分组？一个能给出建议的诊断函数
# ------------------------------------------------------------------------------
def suggest_group_column(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    label_col: str = "label",
) -> pd.DataFrame:
    """评估每个候选分组列的"泄漏风险"。

    直觉：如果一个 group 内部的标签是**纯的**（一个实体只对应一个类别），
    那么随机划分就会让模型直接记住"实体 -> 标签"的查表，泄漏最严重。
    group 越大、纯度越高，随机划分的水分就越大。

    这里用 group 内标签纯度（众数占比）和平均 group 大小来量化。
    """
    rows = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        g = df.groupby(c)
        sizes = g.size()
        purity = (
            g[label_col].agg(lambda s: s.value_counts().iloc[0] / len(s))
            if label_col in df.columns
            else pd.Series(1.0, index=sizes.index)
        )
        rows.append(
            {
                "column": c,
                "n_groups": int(len(sizes)),
                "mean_group_size": round(float(sizes.mean()), 2),
                "max_group_size": int(sizes.max()),
                "label_purity": round(float(purity.mean()), 4),
                "leak_risk": "高" if (sizes.mean() > 1.5 and purity.mean() > 0.95) else ("中" if sizes.mean() > 1.2 else "低"),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["leak_risk", "mean_group_size"], ascending=[True, False])
    return out


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def _ratio_cuts(n: int, ratios: Sequence[float]) -> List[int]:
    s = float(sum(ratios))
    r = [x / s for x in ratios]
    cuts = [0]
    acc = 0
    for k in range(len(r) - 1):
        acc += int(round(n * r[k]))
        cuts.append(acc)
    cuts.append(n)
    return cuts
