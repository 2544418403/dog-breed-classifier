"""
dataset.py —— 自定义 Dataset（在你给的骨架上做工程化扩展）

原始骨架：
    class MyDataset(Dataset):
        def __init__(self, df, tfm, root):
            self.df, self.tfm, self.root = df, tfm, root
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            img = Image.open(f"{self.root}/{row.path}").convert("RGB")
            return self.tfm(img), row.label

骨架能用，但扛不住真实数据，这里补了 6 件事：

  1. 读取走 io_utils.load_image_rgb  -> EXIF 旋转 + 灰度/RGBA 自动转 RGB + 损坏可控
  2. 路径拼接用 os.path.join         -> 兼容路径里带空格 / 正反斜杠 / df 里存绝对路径
  3. label 允许 None                 -> 支持 test 集（无标签）与推理场景
  4. return_index / return_path      -> 做混淆矩阵、错误样本分析、embedding 导出时必需
  5. on_error 策略                   -> 训练时跳过坏图（返回下一个样本），评估时抛错（不能静默少样本）
  6. 可选内存缓存                    -> 小数据集 + CPU 解码瓶颈时的加速开关（增强仍在 CPU 上随机，不影响随机性）

另外提供 `make_splits()`：一次性构造 train/val/test 三个 Dataset，保证
三者的 label<->index 映射（classes / class_to_idx）完全一致 —— 这一点手工做极易出错。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .io_utils import CorruptImageError, load_image_rgb

PathLike = Union[str, Path]


class MyDataset(Dataset):
    """表格驱动的图像数据集。

    Parameters
    ----------
    df : pd.DataFrame
        至少含 `path` 列（相对 root 或绝对路径）。有 `label` 列时返回标签，
        否则 label 恒为 -1（test / 推理模式）。
    tfm : Callable
        接收 PIL.Image，返回 Tensor 或任意对象。
    root : str | Path
        图片根目录。
    path_col, label_col : str
        列名。
    class_to_idx : dict[str, int] | None
        外部传入的类别映射。训练集构造时会自动推断并可通过
        `ds.class_to_idx` 取出，传给 val/test 以保证编号一致 —— **不要**让
        val 集自己再 infer 一遍，类别缺失时编号一错位就是无声的 bug。
    return_index / return_path : bool
        额外返回样本下标 / 路径。
    on_error : {'raise', 'skip', 'zero'}
        'raise' 抛异常（评估/测试推荐，暴露问题）
        'skip'  返回下一个可用样本（训练推荐，容错）
        'zero'  返回全 0 图（不推荐，会污染统计）
    cache : bool
        是否把解码后的 PIL 图缓存在内存（仅建议 <=2 万张的小数据集）。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tfm: Callable,
        root: PathLike = ".",
        path_col: str = "path",
        label_col: str = "label",
        class_to_idx: Optional[Dict[str, int]] = None,
        return_index: bool = False,
        return_path: bool = False,
        on_error: str = "raise",
        cache: bool = False,
    ):
        assert on_error in ("raise", "skip", "zero"), on_error
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.root = str(root)
        self.path_col = path_col
        self.label_col = label_col
        self.return_index = return_index
        self.return_path = return_path
        self.on_error = on_error
        self.cache = cache
        self._cache: Dict[int, Image.Image] = {}

        if path_col not in self.df.columns:
            raise KeyError(f"df 中找不到 path 列 '{path_col}'，现有列: {list(self.df.columns)}")
        self.has_label = label_col in self.df.columns

        # ---- 类别映射 ----
        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
        elif self.has_label:
            labels = sorted(self.df[label_col].dropna().unique().tolist())
            self.class_to_idx = {str(v): i for i, v in enumerate(labels)}
        else:
            self.class_to_idx = {}
        self.classes = [k for k, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

        # ---- 预解析 label / path，避免 __getitem__ 里反复做 pandas 行索引（很慢）----
        self._paths: List[str] = self.df[path_col].astype(str).tolist()
        if self.has_label:
            self._labels: List[int] = [
                _encode_label(v, self.class_to_idx) for v in self.df[label_col].tolist()
            ]
        else:
            self._labels = [-1] * len(self.df)

    # ---------- 基本信息 ----------
    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"MyDataset(n={len(self)}, root={self.root!r}, "
            f"classes={len(self.classes)}, tfm={_tfm_name(self.tfm)}, "
            f"on_error={self.on_error})"
        )

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self._labels, dtype=np.int64)

    def class_counts(self) -> Dict[str, int]:
        """每个类别的样本数（按 class_to_idx 顺序）。"""
        import collections

        c = collections.Counter(self._labels)
        return {cls: int(c.get(idx, 0)) for cls, idx in self.class_to_idx.items()}

    # ---------- 核心 ----------
    def _full_path(self, i: int) -> str:
        p = self._paths[i]
        # df 里存绝对路径就直接用；否则拼 root
        if os.path.isabs(p) or self.root in ("", "."):
            return p
        return os.path.join(self.root, p)

    def _load(self, i: int) -> Image.Image:
        if self.cache and i in self._cache:
            return self._cache[i].copy()
        img = load_image_rgb(self._full_path(i))
        if self.cache:
            self._cache[i] = img.copy()
        return img

    def __getitem__(self, i: int):
        n = len(self)
        seen = 0
        while True:
            try:
                img = self._load(i)
                break
            except CorruptImageError:
                if self.on_error == "raise":
                    raise
                if self.on_error == "zero":
                    img = Image.new("RGB", (224, 224), 0)
                    break
                # skip：轮转到下一个样本
                i = (i + 1) % n
                seen += 1
                if seen >= n:  # 全坏了，别死循环
                    raise RuntimeError("MyDataset: 所有样本都读取失败，无法构造 batch")

        x = self.tfm(img)
        y = self._labels[i]

        if not (self.return_index or self.return_path):
            return x, y
        out: List[Any] = [x, y]
        if self.return_index:
            out.append(i)
        if self.return_path:
            out.append(self._paths[i])
        return tuple(out)


def _encode_label(v, class_to_idx: Dict[str, int]) -> int:
    """把 label 列里的值（int 或 str）映射成连续 int。"""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return -1
    key = str(v)
    if key in class_to_idx:
        return int(class_to_idx[key])
    # 数值字符串（如 "3"）按 int 处理
    try:
        return int(float(key))
    except ValueError:
        return -1


def _tfm_name(tfm) -> str:
    return type(tfm).__name__ if not isinstance(tfm, torch.nn.Module) else tfm.__class__.__name__


def make_splits(
    dfs: Dict[str, pd.DataFrame],
    tfms: Dict[str, Callable],
    root: PathLike = ".",
    label_col: str = "label",
    path_col: str = "path",
    class_to_idx: Optional[Dict[str, int]] = None,
    **kwargs,
) -> Dict[str, MyDataset]:
    """一次构造 train/val/test 三个 Dataset，共享同一份类别映射。

    Example
    -------
    >>> dsets = make_splits({"train": tr_df, "val": va_df, "test": te_df},
    ...                     {"train": tfm_train, "val": tfm_eval, "test": tfm_eval},
    ...                     root="data/pet5")
    >>> dsets["train"].class_to_idx == dsets["val"].class_to_idx
    True
    """
    out: Dict[str, MyDataset] = {}
    shared = class_to_idx
    for name, df in dfs.items():
        ds = MyDataset(
            df,
            tfms.get(name, tfms.get("val")),
            root=root,
            path_col=path_col,
            label_col=label_col,
            class_to_idx=shared,
            **kwargs,
        )
        if shared is None and ds.class_to_idx:
            # 以第一个有标签的 split（通常 train）为准，后续全部复用
            shared = ds.class_to_idx
        out[name] = ds
    # 回填：让先建的 split 也持有完整映射
    if shared is not None:
        for ds in out.values():
            ds.class_to_idx = dict(shared)
            ds.classes = [k for k, _ in sorted(shared.items(), key=lambda kv: kv[1])]
    return out


def collate_with_metadata(batch: Sequence[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """配合 return_index=True 的 collate_fn：返回 (x, y, meta_list)。"""
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta = [b[2:] for b in batch]
    return xs, ys, meta
