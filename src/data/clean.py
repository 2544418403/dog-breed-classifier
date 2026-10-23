"""
clean.py —— 脏数据体检与清洗

四类脏数据（任务书点名）与处理策略：

  损坏文件
      不能"跳过就算了"。要统计、要留证据、要问一句"为什么会坏"
      （下载脚本没校验 md5？磁盘满了？）。本模块产出 bad.csv 让你能回去查。

  单通道图（灰度 / RGBA / 调色板 / CMYK）
      绝大多数 ImageNet 预训练模型吃 3 通道。两种处理：
        (a) 复制成 3 通道（默认，安全）
        (b) 直接丢弃（当灰度图占比很小且属于采集异常时）
      **不要在 collate 时才报错** —— 训练跑 3 小时崩在第 9000 个 batch 是最糟的体验。

  EXIF 旋转
      手机直出图的像素方向 ≠ 显示方向。PIL 的 ImageOps.exif_transpose 能修正。
      两种策略：
        (a) 读取时修正（无侵入，推荐）
        (b) 离线烧进像素并重写（下游用 cv2 / C++ 解码器时必须这么做）

  类别极不平衡
      见 imbalance.py。这里只负责报告不平衡程度，不替你做决定
      （重采样 / 加权 loss / 阈值移动，选哪个取决于指标口径）。

用法：
    report = scan(df, root)          # 体检，返回 (df_with_meta, summary)
    clean_df = apply_clean(...)      # 按策略清洗
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import fix_exif_inplace, probe

PathLike = Union[str, Path]


def scan(
    df: pd.DataFrame,
    root: PathLike = ".",
    path_col: str = "path",
    min_side: int = 32,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """逐张体检，返回 (带体检列的 df, 汇总 dict)。

    新增列：ok, reason, width, height, mode, n_channels, nbytes,
            exif_orientation, is_grayscale, needs_convert, is_suspicious_size
    """
    root = str(root)
    recs: List[Dict] = []
    it = df[path_col].astype(str).tolist()
    if show_progress:
        it = tqdm(it, desc="扫描图片", ncols=80)
    for rel in it:
        p = rel if _is_abs(rel) else f"{root}/{rel}"
        recs.append(probe(p, min_side=min_side).to_dict())

    meta = pd.DataFrame(recs).reset_index(drop=True)
    out = pd.concat([df.reset_index(drop=True), meta.drop(columns=["path"])], axis=1)

    n = len(out)
    bad = out[~out["ok"]]
    summary = {
        "total": int(n),
        "ok": int(out["ok"].sum()),
        "bad": int(len(bad)),
        "bad_reasons": bad["reason"].value_counts().head(20).to_dict() if len(bad) else {},
        "grayscale": int(out["is_grayscale"].sum()),
        "needs_convert": int(out["needs_convert"].sum()),
        "has_exif": int((out["exif_orientation"] > 1).sum()),
        "suspicious_size": int(out["is_suspicious_size"].sum()),
        "min_side_threshold": min_side,
        "size_stats": {
            "w_mean": float(out.loc[out["ok"], "width"].mean() or 0),
            "h_mean": float(out.loc[out["ok"], "height"].mean() or 0),
            "w_min": int(out.loc[out["ok"], "width"].min() or 0),
            "h_min": int(out.loc[out["ok"], "height"].min() or 0),
        },
    }
    return out, summary


def scan_report(summary: Dict) -> str:
    """把体检汇总渲染成文本。"""
    lines = [
        "=" * 62,
        "数据体检报告",
        "=" * 62,
        f"总图片数        : {summary['total']}",
        f"可用            : {summary['ok']}",
        f"不可用          : {summary['bad']}",
        f"  灰度/单通道   : {summary['grayscale']}  (会在读取时自动转 RGB)",
        f"  需转通道      : {summary['needs_convert']}  (L/LA/RGBA/P/CMYK -> RGB)",
        f"  带 EXIF 旋转  : {summary['has_exif']}  (读取时自动摆正)",
        f"  尺寸可疑      : {summary['suspicious_size']}",
        f"平均尺寸        : {summary['size_stats']['w_mean']:.0f} x {summary['size_stats']['h_mean']:.0f}",
        f"最小尺寸        : {summary['size_stats']['w_min']} x {summary['size_stats']['h_min']}",
    ]
    if summary["bad_reasons"]:
        lines.append("-" * 62)
        lines.append("不可用原因分布：")
        for k, v in list(summary["bad_reasons"].items())[:10]:
            lines.append(f"    {v:>5d}  {k}")
    lines.append("=" * 62)
    return "\n".join(lines)


def apply_clean(
    df: pd.DataFrame,
    drop_bad: bool = True,
    drop_grayscale: bool = False,
    drop_suspicious_size: bool = True,
    min_side: int = 32,
    burn_exif: bool = False,
    root: PathLike = ".",
    path_col: str = "path",
    save_bad_csv: Optional[PathLike] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """按策略清洗（要求 df 已经过 `scan()`，带 ok / is_grayscale 等列）。

    Returns
    -------
    (clean_df, log)
    """
    root = str(root)
    before = len(df)
    log: Dict[str, object] = {"before": before}

    bad_mask = ~df["ok"].astype(bool)
    log["dropped_bad"] = int(bad_mask.sum())

    drop = bad_mask.copy()
    if drop_grayscale:
        m = df["is_grayscale"].astype(bool)
        log["dropped_grayscale"] = int((m & ~bad_mask).sum())
        drop |= m
    if drop_suspicious_size:
        m = df["is_suspicious_size"].astype(bool) & df["ok"].astype(bool)
        log["dropped_tiny"] = int((m & ~drop).sum())
        drop |= m

    if save_bad_csv is not None and bad_mask.any():
        cols = [c for c in [path_col, "reason", "width", "height", "mode", "nbytes"] if c in df.columns]
        df.loc[bad_mask, cols].to_csv(save_bad_csv, index=False, encoding="utf-8-sig")
        log["bad_csv"] = str(save_bad_csv)

    if drop_bad:
        clean_df = df.loc[~drop].copy().reset_index(drop=True)
    else:
        clean_df = df.copy().reset_index(drop=True)
        clean_df["usable"] = ~drop

    # 可选：把 EXIF 旋转烧进像素
    if burn_exif:
        fixed = 0
        paths = df.loc[df["exif_orientation"] > 1, path_col].astype(str).tolist()
        for rel in tqdm(paths, desc="烧写 EXIF 方向", ncols=80):
            p = rel if _is_abs(rel) else f"{root}/{rel}"
            fixed += int(fix_exif_inplace(p))
        log["exif_burned"] = fixed

    log["after"] = len(clean_df)
    log["removed_pct"] = round(100.0 * (before - len(clean_df)) / max(1, before), 2)
    return clean_df, log


def class_balance(df: pd.DataFrame, label_col: str = "label", split_col: Optional[str] = "split") -> pd.DataFrame:
    """类别分布表（可按 split 拆开看）。"""
    if split_col is not None and split_col in df.columns:
        t = pd.crosstab(df[split_col], df[label_col])
    else:
        t = df[label_col].value_counts().sort_index().to_frame("count").T
    return t


def imbalance_ratio(df: pd.DataFrame, label_col: str = "label") -> float:
    """最大类 / 最小类。>10 通常就要上加权或重采样了。"""
    vc = df[label_col].value_counts()
    if len(vc) < 2 or vc.min() == 0:
        return float("nan")
    return float(vc.max() / vc.min())


def _is_abs(p: str) -> bool:
    return len(p) > 1 and (p[1] == ":" or p.startswith("/") or p.startswith("\\"))
