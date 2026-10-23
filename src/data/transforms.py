"""
transforms.py —— 增强策略工厂

设计要点不是"把增强堆上去"，而是 **增强必须可审计**：

  1. 每个策略都有一个名字和一个人类可读的描述（`describe()`），
     出问题时能立刻回答"训练时到底对图做了什么"
  2. 提供 `AugPolicy` 显式声明**哪些增强是禁止的**
     —— 任务 6 验收点："数字识别不该用水平翻转"
     6 -> 水平翻转 -> 9，标签就错了；竖直翻转 9 -> 6 同理。
     同理还有：
        - 遥感/病理切片：可以任意旋转（朝向无语义）
        - OCR / 数字 / 交通标志：禁 hflip、禁 vflip、禁 90° 旋转
        - 医学左右位（左眼/右眼、左肺/右肺）：禁 hflip
        - 显微图像带刻度尺/方向标记：禁旋转
  3. train / val / test 用**不同的** transform：
     val/test 必须确定性（Resize + CenterCrop），否则评估指标会抖，
     而且抖动会被误读成"这个 epoch 变好了"

包含的增强（任务书要求）：RandomResizedCrop / RandomHorizontalFlip / ColorJitter / RandAugment
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence

import torch
from torchvision import transforms as T

try:  # torchvision >= 0.15 有 RandAugment
    from torchvision.transforms import RandAugment

    _HAS_RANDAUG = True
except Exception:  # pragma: no cover
    RandAugment = None  # type: ignore
    _HAS_RANDAUG = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class AugPolicy:
    """显式声明增强的边界。

    默认配置是"自然图像"的合理默认（水平翻转安全）。
    做 OCR / 数字 / 左右位医学图像时，把 allow_hflip 设成 False。
    """

    img_size: int = 224
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD

    # --- 几何 ---
    random_resized_crop: bool = True
    crop_scale: Sequence[float] = (0.6, 1.0)
    crop_ratio: Sequence[float] = (0.75, 1.333)
    allow_hflip: bool = True       # 数字/文本/左右位 -> False
    allow_vflip: bool = False      # 绝大多数任务竖直翻转都不合理
    allow_rotation: bool = False   # 需要旋转不变性才开（遥感/病理）
    rotation_degrees: float = 15.0

    # --- 光度 ---
    color_jitter: bool = True
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05             # 医学染色图建议 0（H 通道抖动会改染色）

    # --- 自动增强 ---
    randaugment: bool = False
    randaug_n: int = 2
    randaug_m: int = 9

    # --- 正则 ---
    random_erase: float = 0.0      # p，>0 时开启
    gaussian_blur: float = 0.0     # p，模拟失焦/运动模糊

    # --- 评估 ---
    val_resize: int = 256          # val/test 先 resize 到这个尺寸再 center crop

    def to_dict(self) -> dict:
        return asdict(self)


def build_transform(split: str = "train", policy: Optional[AugPolicy] = None) -> T.Compose:
    """按 split 构造 transform。

    split='train'  -> 随机增强
    split='val'/'test' -> 确定性预处理（Resize + CenterCrop + Normalize）
    """
    p = policy or AugPolicy()
    s = split.lower()
    if s in ("val", "valid", "validation", "test", "eval", "inference"):
        return T.Compose(
            [
                T.Resize(p.val_resize if p.val_resize else int(p.img_size * 1.15)),
                T.CenterCrop(p.img_size),
                T.ToTensor(),
                T.Normalize(list(p.mean), list(p.std)),
            ]
        )

    ops: List[torch.nn.Module] = []
    if p.random_resized_crop:
        ops.append(
            T.RandomResizedCrop(
                p.img_size, scale=tuple(p.crop_scale), ratio=tuple(p.crop_ratio), interpolation=T.InterpolationMode.BILINEAR
            )
        )
    else:
        ops.append(T.Resize(p.img_size))
        ops.append(T.CenterCrop(p.img_size))

    if p.allow_hflip:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    if p.allow_vflip:
        ops.append(T.RandomVerticalFlip(p=0.5))
    if p.allow_rotation:
        ops.append(T.RandomRotation(degrees=p.rotation_degrees))

    if p.color_jitter:
        ops.append(
            T.ColorJitter(
                brightness=p.brightness, contrast=p.contrast, saturation=p.saturation, hue=p.hue
            )
        )

    if p.randaugment:
        if not _HAS_RANDAUG:
            raise RuntimeError("当前 torchvision 不支持 RandAugment，请升级到 >=0.15")
        ops.append(RandAugment(num_ops=p.randaug_n, magnitude=p.randaug_m))

    if p.gaussian_blur > 0:
        ops.append(T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))
        # 用 RandomApply 控制概率
        ops[-1] = T.RandomApply([ops[-1]], p=p.gaussian_blur)

    ops.append(T.ToTensor())

    if p.random_erase > 0:
        ops.append(T.RandomErasing(p=p.random_erase, scale=(0.02, 0.25), ratio=(0.3, 3.3)))

    ops.append(T.Normalize(list(p.mean), list(p.std)))
    return T.Compose(ops)


def describe(policy: AugPolicy, split: str = "train") -> List[str]:
    """把策略翻译成人话，写进实验记录/报告。"""
    p = policy
    s = split.lower()
    if s in ("val", "test", "eval", "inference"):
        return [f"Resize({p.val_resize})", f"CenterCrop({p.img_size})", "ToTensor", "Normalize(ImageNet)"]

    d = []
    if p.random_resized_crop:
        d.append(f"RandomResizedCrop({p.img_size}, scale={tuple(p.crop_scale)})")
    else:
        d.append(f"Resize+CenterCrop({p.img_size})")
    d.append(f"RandomHorizontalFlip(p=0.5)" if p.allow_hflip else "水平翻转: 已禁用")
    if p.allow_vflip:
        d.append("RandomVerticalFlip(p=0.5)")
    if p.allow_rotation:
        d.append(f"RandomRotation(±{p.rotation_degrees}°)")
    if p.color_jitter:
        d.append(
            f"ColorJitter(b={p.brightness}, c={p.contrast}, s={p.saturation}, h={p.hue})"
        )
    if p.randaugment:
        d.append(f"RandAugment(n={p.randaug_n}, m={p.randaug_m})")
    if p.random_erase > 0:
        d.append(f"RandomErasing(p={p.random_erase})")
    if p.gaussian_blur > 0:
        d.append(f"GaussianBlur(p={p.gaussian_blur})")
    d += ["ToTensor", "Normalize(ImageNet)"]
    return d


# ------------------------------------------------------------------------------
# 常用预设
# ------------------------------------------------------------------------------
def policy_natural(img_size: int = 224) -> AugPolicy:
    """自然图像（宠物/动物/场景）—— 默认。"""
    return AugPolicy(
        img_size=img_size,
        val_resize=max(64, int(img_size * 1.15)),  # 低分辨率实验(如64px)时同步缩放，别再用 256
        randaugment=True,
        randaug_n=2,
        randaug_m=9,
    )


def policy_digits(img_size: int = 224) -> AugPolicy:
    """数字/文本识别：**禁止水平翻转**，hue 抖动也关掉（颜色无语义）。"""
    return AugPolicy(
        img_size=img_size,
        allow_hflip=False,
        allow_vflip=False,
        allow_rotation=True,
        rotation_degrees=10.0,
        color_jitter=True,
        brightness=0.2,
        contrast=0.2,
        saturation=0.0,
        hue=0.0,
        randaugment=False,
    )


def policy_medical(img_size: int = 224) -> AugPolicy:
    """医学影像：允许旋转（朝向无语义），但 hue=0（染色颜色是诊断依据）、禁 vflip。"""
    return AugPolicy(
        img_size=img_size,
        allow_hflip=True,
        allow_vflip=False,
        allow_rotation=True,
        rotation_degrees=30.0,
        color_jitter=True,
        brightness=0.15,
        contrast=0.15,
        saturation=0.1,
        hue=0.0,
        randaugment=True,
        randaug_n=2,
        randaug_m=6,
    )


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """把 normalized tensor 还原成 [0,1] 便于可视化。x: (C,H,W) 或 (B,C,H,W)。"""
    mean = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(std, device=x.device).view(-1, 1, 1)
    if x.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (x * std + mean).clamp(0, 1)
