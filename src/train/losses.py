"""
losses.py —— 消融实验要用到的损失函数

本周围绕"长尾"做消融，损失函数是最直接的一味药。三个东西要分清：

    1. class_weight（改 CrossEntropy 的 weight）
       做法：w_c = N/(K·n_c) 或 w_c = sqrt(N/n_c)
       作用机制：把少数类的梯度放大，**等价于给少数类更大的学习率**
       副作用：会让模型整体变得更"激进"， calibration 会变差
       —— 这是它最常被忽略的代价：少数类 recall 涨了，ECE 也涨了

    2. Focal Loss (Lin et al. 2017)
       做法：FL = -(1-p_t)^γ · log(p_t)
       作用机制：**降低简单样本的权重**，把容量留给难样本
       本质是"自动在线难例挖掘"，不只是治不平衡（检测里原用于前景背景比）
       γ 越大越激进；与 α 搭配（α-balanced focal）才同时治不平衡

    3. Class-Balanced Loss (Cui et al. 2019)
       做法：w_c = (1-β)/(1-β^{n_c})
       理论依据：**有效样本数**（effective number）—— 第 n 个同类样本带来的
       新信息递减，所以少数类"每个样本含金量更高"
       比 inverse 更平滑，不容易把大类打崩

本模块把三者都实现，并保留一个**纯 CE 的对照**，保证消融只有一个变量。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """α-balanced Focal Loss。α=None 时退化成标准 Focal（不含类别平衡项）。"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none",
                             weight=self.alpha, label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)                    # p_t：模型给真类的概率（带 α 时含 α 因子，可接受）
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """Cui et al. 2019 的类别平衡损失，可叠加在 CE 或 Focal 上。

    weights = (1-β)/(1-β^{n_c})，归一化后传入底层 loss 的 weight 参数。
    """

    def __init__(self, counts: np.ndarray, beta: float = 0.999, base: str = "ce",
                 gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.mean()
        self.register_buffer("weights", torch.tensor(w, dtype=torch.float32))
        self.base = base
        if base == "focal":
            self.loss = FocalLoss(gamma=gamma, alpha=None, label_smoothing=label_smoothing)
        else:
            self.loss = None
            self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none",
                             weight=self.weights, label_smoothing=getattr(self, "ls", 0.0))
        if self.base == "focal":
            ce = self.loss(logits, target)     # 直接用已含 α 的 focal
            return ce
        return ce.mean()


def make_loss(
    kind: str,
    counts: Optional[np.ndarray] = None,
    weight_mode: str = "sqrt",
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    beta: float = 0.999,
    device: Optional[str] = None,
) -> nn.Module:
    """损失函数工厂 —— 消融实验里唯一需要换的旋钮。

    kind:
      ce                 纯交叉熵（对照组）
      ce_weighted        交叉熵 + 类别权重（weight_mode 选 inverse/sqrt/effective）
      focal              α-balanced focal（α 由 weight_mode 给出）
      cb                 class-balanced (Cui 2019)，base 可用 cb_ce / cb_focal

    device: 传入时把 weight/alpha 搬到对应设备。
    ------------------------------------------------------------------
    **这是一个真实踩过的坑**：CrossEntropyLoss 的 weight 是普通属性而
    不是 registered buffer，所以 `loss.to(device)` **不会**搬运它；
    FocalLoss 里用 register_buffer 注册的 alpha 才会被搬运。
    结果就是：CPU 上的 weight + CUDA 上的 logits -> 第一个 batch 就
    RuntimeError。传 device 参数在这里统一处理，调用方就不用各显神通。
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    try:
        from datapipe.imbalance import class_weights as _cw  # 复用第三周的实现
    except Exception:
        _cw = None

    if kind == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device) if device is not None else t

    if counts is None:
        raise ValueError(f"{kind} 需要 counts（训练集各类样本数）")

    if kind == "ce_weighted":
        if _cw is not None:
            w = _cw(np.repeat(np.arange(len(counts)), counts.astype(int).tolist()),
                    num_classes=len(counts), mode=weight_mode)
        else:  # 兜底实现
            c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
            N, K = c.sum(), len(c)
            w = {"inverse": N / (K * c), "sqrt": np.sqrt(N / c),
                 "effective": (1 - beta) / (1 - np.power(beta, c)) / ((1 - beta) / (1 - np.power(beta, c))).mean(),
                 "none": np.ones(K)}[weight_mode]
            w = torch.tensor(w, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=_to(w), label_smoothing=label_smoothing)

    if kind == "focal":
        if _cw is not None:
            w = _cw(np.repeat(np.arange(len(counts)), counts.astype(int).tolist()),
                    num_classes=len(counts), mode=weight_mode)
        else:
            c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
            w = torch.tensor(np.sqrt(c.sum() / c), dtype=torch.float32)
        return FocalLoss(gamma=gamma, alpha=_to(w), label_smoothing=label_smoothing)

    if kind.startswith("cb"):
        base = "focal" if kind.endswith("focal") else "ce"
        m = ClassBalancedLoss(counts, beta=beta, base=base, gamma=gamma,
                              label_smoothing=label_smoothing)
        return m.to(device) if device is not None else m

    raise ValueError(f"未知 loss kind: {kind}")
