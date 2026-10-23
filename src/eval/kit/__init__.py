"""
eval.kit —— 评估方法论与 badcase 归因工具箱

血统：第 4 周的核心交付物，原名 evalkit。第 8 周收编进 src/eval/kit/。

    from eval.kit import (
        confusion_matrix, per_class_metrics, auroc_pr_auc,
        build_frame, error_table, confusion_pairs,
        group_compare, prior_collapse_test,
        fit_temperature,
    )

设计哲学：本工具箱不追求"指标全"，而追求**每个数字都带着不确定度和对照**。
一个不能回答"这个提升是不是噪声"的评估工具，只会帮你把故事讲得更响。

模块
----
metrics      全套分类指标 + bootstrap 置信区间 + 成对检验
curves       全部图表（混淆矩阵 / ROC / PR / 每类指标 / 消融 / 校准）
badcase      错分样本导出、按错分方向分组、对称性诊断、拼图
attribution  组间分层对照、prior 塌缩检验、特征空间几何
quality      图像客观质量探针 + 分布外标记
calibrate    温度缩放校准（ECE 治理）
（损失函数已移到 train.losses：它是训练侧的东西，不该待在评估工具箱里）
"""

from . import (  # noqa: F401
    metrics, curves, badcase, attribution, quality, calibrate,
)
from .metrics import (  # noqa: F401
    confusion_matrix,
    per_class_metrics,
    average_precision_recall_f1,
    auroc_pr_auc,
    expected_calibration_error,
    roc_pr_curves,
    bootstrap_ci,
    paired_bootstrap_test,
    kappa,
)
from .badcase import (  # noqa: F401
    predict,
    prob_stats,
    build_frame,
    error_table,
    confusion_pairs,
    adjusted_residuals,
    group_errors,
    error_summary,
    confidence_strata,
    contact_sheet,
)
from .attribution import (  # noqa: F401
    group_compare,
    fisher_exact_2x2,
    prior_collapse_test,
    embed,
    class_geometry,
)
from .quality import probe_dataframe, probe_image, ood_flag  # noqa: F401
from .calibrate import fit_temperature, apply_temperature, calibrate_report, fit_prior_correction, apply_prior_correction  # noqa: F401
from .metrics import cluster_bootstrap_ci, paired_cluster_bootstrap_test  # noqa: F401
from . import curves as viz  # noqa: F401

__version__ = "0.4.0"
__all__ = [
    "metrics", "curves", "badcase", "attribution", "quality", "calibrate", "viz",
    "confusion_matrix", "per_class_metrics", "average_precision_recall_f1",
    "auroc_pr_auc", "expected_calibration_error", "roc_pr_curves",
    "bootstrap_ci", "paired_bootstrap_test", "kappa",
    "predict", "prob_stats", "build_frame", "error_table", "confusion_pairs",
    "adjusted_residuals", "group_errors", "error_summary", "confidence_strata", "contact_sheet",
    "group_compare", "fisher_exact_2x2", "prior_collapse_test", "embed", "class_geometry",
    "probe_dataframe", "probe_image", "ood_flag",
    "fit_temperature", "apply_temperature", "calibrate_report",
    "fit_prior_correction", "apply_prior_correction",
    "cluster_bootstrap_ci", "paired_cluster_bootstrap_test",
]
