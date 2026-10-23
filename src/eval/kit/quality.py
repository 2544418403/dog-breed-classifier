"""
quality.py —— 用**客观量**替代"这张图看起来糊"

================================================================================
归因时最容易犯的错："我觉得这批错分样本主要是图糊" —— 这句话无法验证、
无法度量、无法复现。把它翻译成：

    错分样本组的**拉普拉斯方差**显著低于同类正确样本（bootstrap 95% CI）

就从一个形容词变成了一个可以反驳、可以复现的结论。
================================================================================

本模块把每张图压成一组标量：
    brightness   亮度均值（过曝/欠曝）
    contrast     灰度标准差（对比度）
    sharpness    拉普拉斯方差（业内通用的清晰度代理；<100 基本是糊的）
    hist_entropy 直方图熵（细节丰富度，纯色块会很低）
    colorfulness Hasler-Süsstrunk 颜色丰富度（灰度图 ≈ 0）
    is_gray      R/G/B 三通道是否几乎相等 —— **本项目注入的那 90 张灰度图可以直接被这个函数捞出来**
    nbytes_log   文件体积对数（间接反映压缩强度/信息量）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

from PIL import Image


# ------------------------------------------------------------------------------
def laplacian_var(gray: np.ndarray) -> float:
    """清晰度代理：拉普拉斯算子的响应方差。

    平坦区域（失焦、纯色背景、纯噪点）响应小 -> 方差小。
    """
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if _HAS_CV2:
        return float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    # 无 cv2 时用 numpy 卷积兜底（3x3 拉普拉斯核）
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape
    pad = np.pad(gray.astype(np.float64), 1, mode="reflect")
    out = (
        k[0, 1] * pad[:-2, 1:-1] + k[1, 0] * pad[1:-1, :-2] + k[1, 1] * pad[1:-1, 1:-1]
        + k[1, 2] * pad[1:-1, 2:] + k[2, 1] * pad[2:, 1:-1]
    )
    del h, w
    return float(out.var())


def hist_entropy(gray: np.ndarray, bins: int = 64) -> float:
    """灰度直方图熵，除以 log(bins) 归一化到 [0,1]。

    极低 = 画面几乎只有几个灰阶（纯色背景、过度曝光、二值图）。
    """
    h = np.histogram(gray, bins=bins, range=(0, 256))[0].astype(np.float64)
    p = h / max(1e-12, h.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) 的颜色丰富度。

    rg = R-G, yb = 0.5(R+G)-B；colorfulness = sqrt(σ²_rg + σ²_yb) + 0.3*|µ_rg+µ_yb|
    灰度图的 rg=yb=0 -> 分数 ≈ 0。这个量正好能把注入的灰度样本识别出来。
    """
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.abs(rg.mean() + yb.mean()))


def probe_image(path: os.PathLike) -> Dict[str, float]:
    """单张图 -> 一组客观质量标量。读取失败时返回全 nan（不静默丢样本）。"""
    p = Path(path)
    try:
        nbytes = p.stat().st_size
    except Exception:
        nbytes = -1
    try:
        img = Image.open(p)
        img.load()
        rgb = np.asarray(img.convert("RGB"))
    except Exception:
        return {"ok": 0, "brightness": np.nan, "contrast": np.nan, "sharpness": np.nan,
                "hist_entropy": np.nan, "colorfulness": np.nan, "is_gray": np.nan,
                "width": np.nan, "height": np.nan, "nbytes": float(nbytes)}

    gray = rgb.mean(axis=2)
    return {
        "ok": 1,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": laplacian_var(gray),
        "hist_entropy": hist_entropy(gray),
        "colorfulness": colorfulness(rgb),
        # 三通道两两最大差 < 2 视为灰度
        "is_gray": float(max(float(np.abs(rgb[..., i] - rgb[..., j]).max())
                             for i, j in ((0, 1), (1, 2), (0, 2))) < 2.0),
        "width": float(rgb.shape[1]),
        "height": float(rgb.shape[0]),
        "nbytes": float(nbytes),
    }


def probe_dataframe(
    df: pd.DataFrame,
    root: os.PathLike,
    path_col: str = "path",
    show_progress: bool = True,
) -> pd.DataFrame:
    """对整张表跑探针，把质量列并回去。

    注意：**一定要用与训练时相同的读取链路**（这里直接用 PIL 读原始文件，
    不做任何 transform），因为要看的是"数据本身的缺陷"，不是增强的产物。
    """
    root = Path(root)
    rows: List[Dict[str, float]] = []
    paths = df[path_col].astype(str).tolist()
    it = range(len(paths))
    if show_progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, total=len(paths), desc="quality probe", unit="img")
        except Exception:
            pass
    for i in it:
        rows.append(probe_image(root / paths[i]))
    q = pd.DataFrame(rows).reset_index(drop=True)
    q.insert(0, path_col, paths)
    out = df.merge(q, on=path_col, how="left", suffixes=("", "_q"))
    # 同名列（df 里可能已有 width/height）以探针结果为准时加后缀 _probe
    return out


# ------------------------------------------------------------------------------
# 用质量向量判断"这张图是否分布外"
# ------------------------------------------------------------------------------
def ood_flag(
    qdf: pd.DataFrame,
    ref_mask: Sequence[bool],
    cols: Sequence[str] = ("brightness", "contrast", "sharpness", "hist_entropy", "colorfulness"),
    thresh: float = 3.5,
) -> pd.Series:
    """基于参考分布（通常是 train 集）的稳健 z-score 打分布外标。

    用**中位数 + MAD** 而不是均值+标准差：质量分布长尾严重，
    一个极端值就能把标准差拉爆，导致什么都不报。MAD 对离群值稳健。
    """
    flags = np.zeros(len(qdf), dtype=bool)
    for c in cols:
        if c not in qdf.columns:
            continue
        v = qdf[c].to_numpy(dtype=np.float64)
        ref = v[np.asarray(ref_mask, dtype=bool)]
        ref = ref[~np.isnan(ref)]
        if ref.size < 10:
            continue
        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) + 1e-9
        z = 0.6745 * (v - med) / mad
        flags |= (np.abs(z) > thresh) & ~np.isnan(v)
    return pd.Series(flags, index=qdf.index)
