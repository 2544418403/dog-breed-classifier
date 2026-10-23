"""
viz.py —— 增强可视化（任务书第 6 条：人工确认增强"合理"）

为什么要人眼看一遍：
    增强是一个**无声的标签破坏器**。它不会报错，不会让 loss 变 nan，
    只会安静地让模型学到错误的东西。

    真实案例：
      - MNIST 上开 RandomHorizontalFlip -> "6" 变成 "9"，标签仍是 6
      - 医学左右位分类开 hflip -> 左肺变右肺，标签仍是"左"
      - 染色病理开 hue 抖动 -> 染色深浅是诊断依据，被随机改掉了
      - OCR 上开 90° 旋转 -> 字符完全不可读

    所以流程必须是：写增强 -> 画出来 -> 人确认 -> 才开训练。

本模块给两个视图：
    show_augmentations()  同一张图增强 N 次（看"扰动幅度合不合理"）
    show_batch()          一个 batch 的样张（看"整体有没有被破坏"）
    show_policy_check()   并排对比"允许/禁止某增强"的反例（数字 hflip 反例）
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from .transforms import AugPolicy, denormalize  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def to_display(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalized tensor -> (H,W,C) uint8 numpy。"""
    x = denormalize(t.float().cpu())
    return (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def show_augmentations(
    img: Image.Image,
    tfm,
    n: int = 8,
    cols: int = 4,
    title: str = "同一张图的多次随机增强",
    save: Optional[Union[str, Path]] = None,
    dpi: int = 110,
) -> plt.Figure:
    """同一张原图连续增强 n 次 —— 用来判断扰动强度是否过头。"""
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i == 0:
            ax.imshow(np.asarray(img))
            ax.set_title("原图", fontsize=10)
        elif i - 1 < n:
            ax.imshow(to_display(tfm(img)))
            ax.set_title(f"aug #{i}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def show_batch(
    batch_x: torch.Tensor,
    batch_y: Optional[Sequence] = None,
    class_names: Optional[Sequence[str]] = None,
    n: int = 16,
    cols: int = 4,
    title: str = "一个 batch 的增强结果（请人工确认）",
    save: Optional[Union[str, Path]] = None,
    dpi: int = 110,
) -> plt.Figure:
    """画一个 batch 的前 n 张。"""
    n = min(n, len(batch_x))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(to_display(batch_x[i]))
            if batch_y is not None:
                y = int(batch_y[i])
                name = class_names[y] if (class_names and y < len(class_names)) else str(y)
                ax.set_title(f"label={name}", fontsize=10, color="#c0392b")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig


def show_policy_check(
    imgs: Sequence[Image.Image],
    tfm_safe,
    tfm_unsafe,
    titles: Sequence[str] = ("原图", "安全策略（禁水平翻转）", "危险策略（允许水平翻转）"),
    save: Optional[Union[str, Path]] = None,
    dpi: int = 110,
) -> plt.Figure:
    """三列对照：原图 / 安全增强 / 危险增强。用于人工确认"这个增强对本任务是否合法"。"""
    k = len(imgs)
    fig, axes = plt.subplots(k, 3, figsize=(9.0, 3.1 * k))
    axes = np.atleast_2d(axes)
    for r, img in enumerate(imgs):
        for c, (ax, tfm) in enumerate(zip(axes[r], [None, tfm_safe, tfm_unsafe])):
            ax.axis("off")
            if c == 0:
                ax.imshow(np.asarray(img))
            else:
                ax.imshow(to_display(tfm(img)))
            if r == 0:
                ax.set_title(titles[c], fontsize=11,
                             color="#c0392b" if c == 2 else "#2c3e50")
    fig.suptitle("增强合法性人工确认：同一个输入在不同策略下的输出", fontsize=13)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig
