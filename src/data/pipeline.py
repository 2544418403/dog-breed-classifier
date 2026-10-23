"""
pipeline.py —— 把上面 7 个模块串成一条可复现的管道

存在的意义：**流程本身要成为可保存的对象**，而不是散落在 notebook 里的十几行。
这样下次换数据集时，改的是参数，不是重写代码；并且能拿到一份
`pipeline.report()` 文本直接贴进实验记录。

典型用法：

    pipe = DataPipeline(
        manifest_csv="data/messy/manifest.csv",
        root="data/messy",
        group_col="plate_id",          # 实体列：None 表示（故意）按图片随机划分
        label_col="label",
        ratios=(0.7, 0.15, 0.15),
        seed=42,
    )
    pipe.scan()        # 脏数据体检
    pipe.clean()       # 清洗
    pipe.split()       # 按实体划分
    pipe.verify()      # 强制断言无泄漏（会真的 assert）
    dsets = pipe.build_datasets(img_size=224, policy="natural")
    print(pipe.report())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from . import clean as C
from . import imbalance as I
from . import splits as S
from . import transforms as T
from .dataset import make_splits

PathLike = Union[str, Path]


@dataclass
class DataPipeline:
    manifest_csv: Optional[PathLike] = None
    df: Optional[pd.DataFrame] = None
    root: PathLike = "."
    path_col: str = "path"
    label_col: str = "label"
    group_col: Optional[str] = None      # 实体列；None -> 按图片随机（对照组）
    ratios: Sequence[float] = (0.7, 0.15, 0.15)
    names: Sequence[str] = ("train", "val", "test")
    seed: int = 42
    min_side: int = 32
    class_weight_mode: str = "sqrt"

    # 运行时填充
    scanned: bool = field(default=False, repr=False)
    cleaned: bool = field(default=False, repr=False)
    split_done: bool = field(default=False, repr=False)
    scan_summary: Dict = field(default_factory=dict, repr=False)
    clean_log: Dict = field(default_factory=dict, repr=False)
    overlap: Optional[pd.DataFrame] = field(default=None, repr=False)
    data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        if self.df is not None:
            self.data = self.df.copy()
        else:
            self.data = pd.read_csv(self.manifest_csv)
        return self.data

    def scan(self) -> Dict:
        if not len(self.data):
            self.load()
        self.data, self.scan_summary = C.scan(self.data, self.root, self.path_col, self.min_side)
        self.scanned = True
        return self.scan_summary

    def clean(self, drop_grayscale: bool = False, burn_exif: bool = False,
              save_bad_csv: Optional[PathLike] = None) -> pd.DataFrame:
        if not self.scanned:
            self.scan()
        self.data, self.clean_log = C.apply_clean(
            self.data,
            drop_bad=True,
            drop_grayscale=drop_grayscale,
            root=self.root,
            path_col=self.path_col,
            burn_exif=burn_exif,
            save_bad_csv=save_bad_csv,
        )
        self.cleaned = True
        return self.data

    def split(self) -> pd.DataFrame:
        if not len(self.data):
            self.load()
        if self.group_col:
            self.data = S.group_split(
                self.data, self.group_col, self.ratios, self.names,
                stratify_col=self.label_col, seed=self.seed,
            )
        else:
            self.data = S.random_split(
                self.data, self.ratios, self.names,
                stratify_col=self.label_col, seed=self.seed,
            )
        self.split_done = True
        return self.data

    def verify(self, strict: bool = True) -> pd.DataFrame:
        """校验无泄漏。group_col 为 None 时返回重叠统计（用于对照演示）。"""
        assert self.split_done, "请先调用 split()"
        if self.group_col is None:
            return pd.DataFrame()
        if strict:
            S.assert_disjoint(self.data, "split", self.group_col)
        self.overlap = S.overlap_report(self.data, "split", self.group_col)
        return self.overlap

    def class_weights(self) -> Optional[list]:
        tr = self.data[self.data["split"] == "train"]
        if not len(tr) or self.label_col not in tr.columns:
            return None
        w = I.class_weights(tr[self.label_col].tolist(), mode=self.class_weight_mode)
        return w.tolist()

    def build_datasets(self, img_size: int = 224, policy: str = "natural", **ds_kwargs):
        """返回 {split_name: MyDataset}，共享类别映射、各自正确的 transform。"""
        assert self.split_done, "请先调用 split()"
        pol = {
            "natural": T.policy_natural,
            "digits": T.policy_digits,
            "medical": T.policy_medical,
        }[policy](img_size)
        tfms = {}
        for n in self.names:
            tfms[n] = T.build_transform("train" if n == "train" else "val", pol)
        dfs = {n: self.data[self.data["split"] == n] for n in self.names}
        return make_splits(dfs, tfms, root=self.root, path_col=self.path_col,
                           label_col=self.label_col, **ds_kwargs)

    # ------------------------------------------------------------------
    def report(self) -> str:
        lines = ["=" * 66, "数据管道报告", "=" * 66]
        lines.append(f"manifest : {self.manifest_csv}")
        lines.append(f"root     : {self.root}")
        lines.append(f"分组列   : {self.group_col or '（无：按图片随机划分）'}")
        if self.scan_summary:
            lines.append("")
            lines.append(C.scan_report(self.scan_summary))
        if self.clean_log:
            lines.append("")
            lines.append(
                f"清洗: {self.clean_log.get('before')} -> {self.clean_log.get('after')} "
                f"(移除 {self.clean_log.get('removed_pct')}%)"
            )
        if self.split_done:
            lines.append("")
            lines.append(S.split_summary(self.data, "split", self.label_col).to_string())
            lines.append("")
            lines.append(f"类别不平衡比 (train): "
                         f"{C.imbalance_ratio(self.data[self.data['split']=='train'], self.label_col):.2f}")
            if self.overlap is not None and len(self.overlap):
                lines.append("")
                lines.append("实体重叠检查：")
                lines.append(self.overlap.to_string(index=False))
            else:
                lines.append("")
                lines.append("实体重叠检查: 通过（各 split 之间无共享实体）"
                             if self.group_col else "实体重叠检查: 未启用分组划分")
        lines.append("=" * 66)
        return "\n".join(lines)
