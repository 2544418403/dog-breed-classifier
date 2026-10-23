"""预处理：训练时的 val transform 的「无 torch 版」等价实现。

==============================================================================
为什么这段函数要单独存在，还要被训练和部署两侧共用
==============================================================================
这是整个项目里最容易静默出错的地方：如果服务端的预处理和训练时差一点，
模型指标会整体下滑，而**不会报错**——你只会看到线上效果"莫名不好"。

训练侧用的是 torchvision 的流水线（见 data.transforms.build_transform('val')）：
    Resize(短边 → resize_to) → CenterCrop(img_size) → ToTensor(/255) → Normalize
这里用 PIL + numpy 手抄一遍同样的语义，好处是：

  - 服务端不需要为了一个 resize 去装 torch / torchvision（镜像少 1GB）
  - 量化校准（models.export）和服务在线推理走的是**同一份代码**，
    校准分布与实际输入分布不可能漂移

三步 duration 的坑都写在注释里了，改之前请先读。
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_short_side(img: Image.Image, resize_to: int) -> Image.Image:
    """把**短边**缩到 resize_to，保持长宽比。

    注意 torchvision.transforms.Resize(int) 的语义是"短边对齐"，
    不是 resize((n, n))。直接拉成正方形会把狗压变形，
    对依赖体型/脸型比例的细粒度分类是致命的。
    """
    w, h = img.size
    scale = resize_to / min(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """中心裁剪。极小图先 pad 到目标尺寸（黑边），保证 shape 永远对得上。"""
    w, h = img.size
    if w < size or h < size:
        canvas = Image.new("RGB", (max(w, size), max(h, size)))
        canvas.paste(img, ((canvas.width - w) // 2, (canvas.height - h) // 2))
        img = canvas
        return img.crop((0, 0, size, size))
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_pil(img: Image.Image, img_size: int, resize_to: int,
                   mean, std) -> np.ndarray:
    """PIL 图 -> (3, img_size, img_size) float32 CHW，已归一化。"""
    img = img.convert("RGB")
    img = center_crop(resize_short_side(img, resize_to), img_size)

    x = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    x = np.transpose(x, (2, 0, 1))                          # -> CHW
    mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    x = (x - mean) / std
    return np.ascontiguousarray(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)   # 减最大值防 exp 溢出
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
