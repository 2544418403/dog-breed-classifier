"""
datapipe —— 可复用、可验证的图像数据管道

设计目标（第 3 周）：把"读数据"从脚本里的几行代码，变成能独立 import、
能被测试、能被复用的模块。

    from data import MyDataset, build_transform, AugPolicy, group_split, assert_disjoint

    df = pd.read_csv("datasets/pet5/manifest.csv")
    df = group_split(df, group_col="plate_id", ratios=(.7, .15, .15))
    assert_disjoint(df, "split", "plate_id")          # CI 里就该有这一行

    ds = MyDataset(df[df.split == "train"], build_transform("train", policy), root="datasets/pet5")
    x, y = ds[0]

模块清单
--------
io_utils    健壮读取：损坏 / 单通道 / EXIF 旋转
dataset     MyDataset（表格驱动，train/val/test 共享类别映射）
splits      按实体分组划分 + 泄漏断言
transforms  增强策略工厂 + 可审计的策略描述
clean       脏数据体检与清洗
audit       train/val 泄漏自检（md5 / 感知哈希 / CNN 嵌入余弦）
imbalance   类别权重与平衡采样
viz         增强可视化（人工确认）
pipeline    一键串起全流程
"""

from . import io_utils, dataset, splits, transforms, clean, audit, imbalance, viz, pipeline  # noqa: F401
from .io_utils import (  # noqa: F401
    CorruptImageError,
    load_image_rgb,
    probe,
    fix_exif_inplace,
    set_load_truncated,
)
from .dataset import MyDataset, make_splits  # noqa: F401
from .splits import (  # noqa: F401
    random_split,
    group_split,
    group_kfold,
    assert_disjoint,
    overlap_report,
    split_summary,
    suggest_group_column,
)
from .transforms import (  # noqa: F401
    AugPolicy,
    build_transform,
    describe,
    policy_natural,
    policy_digits,
    policy_medical,
    denormalize,
)
from .clean import scan, scan_report, apply_clean, imbalance_ratio  # noqa: F401
from .audit import leakage_audit, hash_matrix, EmbeddingExtractor, exact_duplicate_report  # noqa: F401
from .imbalance import class_weights, make_balanced_sampler, imbalance_report  # noqa: F401
from .viz import show_augmentations, show_batch, show_policy_check, to_display  # noqa: F401
from .pipeline import DataPipeline  # noqa: F401

__version__ = "0.3.0"
__all__ = [
    "MyDataset",
    "make_splits",
    "DataPipeline",
    "random_split",
    "group_split",
    "group_kfold",
    "assert_disjoint",
    "overlap_report",
    "split_summary",
    "suggest_group_column",
    "AugPolicy",
    "build_transform",
    "describe",
    "policy_natural",
    "policy_digits",
    "policy_medical",
    "denormalize",
    "scan",
    "scan_report",
    "apply_clean",
    "imbalance_ratio",
    "leakage_audit",
    "hash_matrix",
    "EmbeddingExtractor",
    "exact_duplicate_report",
    "class_weights",
    "make_balanced_sampler",
    "imbalance_report",
    "show_augmentations",
    "show_batch",
    "show_policy_check",
    "to_display",
    "load_image_rgb",
    "probe",
    "fix_exif_inplace",
    "set_load_truncated",
    "CorruptImageError",
    "io_utils",
    "dataset",
    "splits",
    "transforms",
    "clean",
    "audit",
    "imbalance",
    "viz",
    "pipeline",
]
