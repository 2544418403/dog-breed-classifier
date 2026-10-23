"""
imbalance.py —— 类别极不平衡的处理

三种主流手段，选哪个取决于你的评估指标：

  1. class_weight（改 loss）
     最安全：不改变数据分布，val 指标口径不变。
     适合：当你关心的是**在真实分布上的** recall / AUC。

  2. 过采样 / WeightedRandomSampler（改数据）
     少数类被反复看到，等价于给它们更大的权重，但会引入更多过拟合风险。
     适合：模型对少数类完全学不动时。

  3. 欠采样（改数据）
     砍多数类。会丢信息，但训练快。
     适合：多数类占比极端（>90%）且冗余样本多。

**重要**：重采样只能作用于 train，**val/test 必须保持原始分布**。
在 val 上做平衡采样，等于伪造指标 —— 这是新手最常见的翻车点。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def class_weights(
    labels: Sequence[int],
    num_classes: Optional[int] = None,
    mode: str = "inverse",
    beta: float = 0.999,
) -> torch.Tensor:
    """计算每个类别的 loss 权重。

    mode:
      'inverse'    w_c = N / (K * n_c)          经典 sklearn balanced
      'sqrt'       w_c = sqrt(N / n_c)          更温和，最常用
      'effective'  w_c = (1-beta)/(1-beta^n_c)  Cui et al. 2019（类别平衡损失）
      'none'       全 1
    """
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = num_classes or int(labels.max()) + 1
    counts = np.bincount(labels[labels >= 0], minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    N = counts.sum()

    if mode == "none":
        w = np.ones(n_classes)
    elif mode == "inverse":
        w = N / (n_classes * counts)
    elif mode == "sqrt":
        w = np.sqrt(N / counts)
    elif mode == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.mean()
    else:
        raise ValueError(f"未知 mode: {mode}")
    return torch.tensor(w, dtype=torch.float32)


def make_balanced_sampler(labels: Sequence[int], num_classes: Optional[int] = None) -> WeightedRandomSampler:
    """每个类别被采到的概率相同（等价于对少数类过采样、多数类欠采样）。"""
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = num_classes or int(labels.max()) + 1
    counts = np.bincount(labels[labels >= 0], minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    # 该样本所属类的权重 = 1 / n_c
    per_class_w = 1.0 / counts
    sample_w = per_class_w[labels]
    return WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.double), num_samples=len(labels), replacement=True)


def imbalance_report(df: pd.DataFrame, label_col: str = "label", split_col: Optional[str] = "split") -> str:
    """打印每个 split 的类别分布与不平衡比。"""
    lines = []
    splits = df[split_col].unique().tolist() if (split_col and split_col in df.columns) else [None]
    for s in splits:
        sub = df if s is None else df[df[split_col] == s]
        vc = sub[label_col].value_counts().sort_index()
        ratio = float(vc.max() / max(1, vc.min()))
        lines.append(f"--- {s or 'all'} (n={len(sub)}) ---")
        for k, v in vc.items():
            bar = "#" * int(40 * v / max(1, vc.max()))
            lines.append(f"  class {k}: {v:>6d} ({100*v/len(sub):5.1f}%) {bar}")
        lines.append(f"  不平衡比 max/min = {ratio:.2f}")
        if ratio > 10:
            lines.append("  >> 严重不平衡：建议 class_weight='sqrt' 或 balanced sampler")
        elif ratio > 3:
            lines.append("  >> 中度不平衡：建议 class_weight='sqrt'")
    return "\n".join(lines)
