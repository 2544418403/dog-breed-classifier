"""
calibrate.py —— 温度缩放（Temperature Scaling）

================================================================================
为什么校准是"落地最后一公里"
================================================================================

一个准确率 82%、ECE 0.12 的模型，和一个准确率 82%、ECE 0.02 的模型，
**在离线指标上完全一样，在生产上完全不是一回事**。

第二种才敢接：
    - 自动放行：conf > 0.95 直接通过，人工只看剩下的（需要 conf 是真概率）
    - 风险打分：把 conf 和下游代价矩阵相乘算期望损失
    - 拒答/转人工：低 conf 转人工

温度缩放：p_calib = softmax(logits / T)，**只学一个标量**。
它是唯一一个不改变 argmax（因此不改变 accuracy/F1/AUC）、却能把
ECE 打下去的方法 —— 这也是它成为工业标配的原因。

铁律：T 必须在 **val** 上拟合，在 **test** 上评估。
在 test 上拟合 T 再在 test 上报 ECE，是自欺欺人（等价于把 test 当训练集用）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray, max_iter: int = 200) -> float:
    """在 val logits 上最小化 NLL，求最优温度 T。

    用 LBFGS 优化 log(T) 保证 T>0。T>1 -> 软化分布（治过度自信），
    T<1 -> 锐化分布。
    """
    lg = torch.as_tensor(np.asarray(logits_val, dtype=np.float32))
    y = torch.as_tensor(np.asarray(y_val, dtype=np.int64))
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / torch.exp(logT).clamp_min(1e-6), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).detach().clamp_min(1e-3))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """softmax(logits / T)。"""
    lg = torch.as_tensor(np.asarray(logits, dtype=np.float32)) / float(T)
    return torch.softmax(lg, dim=1).numpy()


def calibrate_report(
    y_true: np.ndarray,
    logits: np.ndarray,
    y_val: np.ndarray,
    logits_val: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """一次性做完：fit on val -> apply to test -> 返回前后 ECE。

    返回 (T, calibrated_probs, {"ece_before":..., "ece_after":...})
    """
    from .metrics import expected_calibration_error

    T = fit_temperature(logits_val, y_val)
    p_raw = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    p_cal = apply_temperature(logits, T)
    before = expected_calibration_error(y_true, p_raw, n_bins=n_bins)
    after = expected_calibration_error(y_true, p_cal, n_bins=n_bins)
    return T, p_cal, {
        "T": round(T, 4),
        "ece_before": round(before["ece"], 4),
        "ece_after": round(after["ece"], 4),
        "mce_before": round(before["mce"], 4),
        "mce_after": round(after["mce"], 4),
        "brier_before": round(before["brier"], 4),
        "brier_after": round(after["brier"], 4),
        "bins_before": before["bins"],
        "bins_after": after["bins"],
    }


# ------------------------------------------------------------------------------
# 先验校正（logit adjustment / prior shift 修正）
# ------------------------------------------------------------------------------
def fit_prior_correction(
    logits_val: np.ndarray,
    y_val: np.ndarray,
    pi_train: np.ndarray,
    grid=None,
    objective: str = "macro_f1",
) -> dict:
    """在 val 上拟合先验校正强度 τ。

    背景：训练分布的类先验 π_train 和部署分布 π_deploy 往往不一致
    （本项目实测：train 里 shiba 16.5%、york 8.2%；test 里只剩 5.1%、4.2%）。
    模型的 logit 里带着训练先验的偏置，结果就是**过预测少类**、
    少数类 precision 被冲垮。

    修正：logits' = logits + τ · (log π_target - log π_train)
      τ=0   不修正
      τ=1   完全按经典 prior-shift 公式修正

    τ 不能拍脑袋，**必须在 val 上拟合**（这里按目标指标做网格搜索）。
    π_target 用 val 的真实类分布估计（val 有标注，且未参与选模）。
    """
    if grid is None:
        grid = np.round(np.arange(0, 1.61, 0.1), 2)

    pi_target = np.bincount(np.asarray(y_val, dtype=int), minlength=len(pi_train)).astype(float)
    pi_target = pi_target / pi_target.sum()
    shift = np.log(np.maximum(pi_target, 1e-9)) - np.log(np.maximum(pi_train, 1e-9))

    from .metrics import per_class_metrics

    def _score(lg):
        yp = lg.argmax(1)
        pc = per_class_metrics(y_val, yp, len(pi_train))
        if objective == "macro_f1":
            return float(pc.f1.mean())
        if objective == "macro_recall":
            return float(pc.recall.mean())
        if objective == "accuracy":
            return float((yp == np.asarray(y_val)).mean())
        raise ValueError(objective)

    rows = []
    best = (-1.0, 0.0)
    for t in grid:
        s = _score(logits_val + t * shift)
        rows.append({"tau": float(t), objective: round(s, 4)})
        if s > best[0]:
            best = (s, float(t))

    return {"tau": best[1], "best_val_score": round(best[0], 4),
            "shift": shift.tolist(), "pi_target": pi_target.tolist(),
            "curve": rows}


def apply_prior_correction(logits: np.ndarray, tau: float, shift: np.ndarray) -> np.ndarray:
    return logits + tau * np.asarray(shift, dtype=np.float64)
