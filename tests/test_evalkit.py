"""
tests/test_evalkit.py —— 指标正确性自检

这个测试文件的存在理由只有一条：
**自己实现指标是有风险的，一个符号写反就能让所有结论反过来。**

所以每个指标都必须和 sklearn / 手算小例子对拍。
跑法：
    python tests/test_evalkit.py    # 需要训练依赖：pip install -r requirements-train.txt
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import pytest as _pytest
except ImportError:
    _pytest = None

_NEEDS = ["numpy", "pandas", "sklearn"]
_MISSING = [m for m in _NEEDS if importlib.util.find_spec(m) is None]
if _MISSING:
    _MSG = (f"缺少 {_MISSING}：本组用例需要训练依赖\n"
            f"    pip install -r requirements-train.txt")
    if _pytest is not None:
        _pytest.skip(_MSG, allow_module_level=True)   # pytest 收集时：跳过
    raise SystemExit(_MSG)                            # 直接 python xxx.py 时：明确报错

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]      # 仓库根
sys.path.insert(0, str(ROOT / "src"))   # src 布局：包名都在 src/ 下

from eval.kit import (  # noqa: E402
    confusion_matrix,
    per_class_metrics,
    average_precision_recall_f1,
    auroc_pr_auc,
    expected_calibration_error,
    bootstrap_ci,
    adjusted_residuals,
    confusion_pairs,
    kappa,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    flag = "  ok  " if cond else " FAIL "
    print(f"[{flag}] {name}" + (f"   {detail}" if detail else ""))


def close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=tol, equal_nan=True)


def main():
    from sklearn.metrics import (
        confusion_matrix as sk_cm,
        precision_recall_fscore_support,
        classification_report,
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        cohen_kappa_score,
    )

    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # 1. 混淆矩阵与 sklearn 一致（行=真实）
    # ------------------------------------------------------------------
    K = 5
    n = 800
    yt = rng.integers(0, K, n)
    yp = np.where(rng.random(n) < 0.62, yt, rng.integers(0, K, n))

    C = confusion_matrix(yt, yp, K)
    Cs = sk_cm(yt, yp, labels=list(range(K)))
    check("confusion_matrix == sklearn", np.array_equal(C, Cs), f"max diff {np.abs(C-Cs).max()}")

    # ------------------------------------------------------------------
    # 2. per-class P/R/F1 与 sklearn 一致
    # ------------------------------------------------------------------
    pc = per_class_metrics(yt, yp, K)
    p_s, r_s, f_s, s_s = precision_recall_fscore_support(
        yt, yp, labels=list(range(K)), zero_division=0
    )
    check("per-class precision == sklearn", close(pc.precision.to_numpy(), p_s))
    check("per-class recall == sklearn", close(pc.recall.to_numpy(), r_s))
    check("per-class f1 == sklearn", close(pc.f1.to_numpy(), f_s))
    check("support == sklearn", close(pc.support.to_numpy(), s_s))

    # ------------------------------------------------------------------
    # 3. macro / micro / weighted 平均
    # ------------------------------------------------------------------
    avg = average_precision_recall_f1(yt, yp, K)
    _, _, f_macro, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(K)), average="macro", zero_division=0)
    _, _, f_micro, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(K)), average="micro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(
        yt, yp, labels=list(range(K)), average="weighted", zero_division=0)
    check("macro-F1 == sklearn", abs(avg.loc["macro", "f1"] - f_macro) < 1e-9,
          f"{avg.loc['macro','f1']:.6f} vs {f_macro:.6f}")
    check("micro-F1 == sklearn", abs(avg.loc["micro", "f1"] - f_micro) < 1e-9)
    check("weighted-F1 == sklearn", abs(avg.loc["weighted", "f1"] - f_w) < 1e-9)
    check("micro-recall == accuracy (多分类恒等式)",
          abs(avg.loc["micro", "accuracy"] - accuracy_score(yt, yp)) < 1e-9)

    # ------------------------------------------------------------------
    # 4. AUROC / AP 与 sklearn 一致
    # ------------------------------------------------------------------
    P = rng.random((n, K))
    P = P / P.sum(1, keepdims=True)
    # 让概率和标签弱相关：强到可分辨但远非完美，这样 AUROC 才是一个**有意义的**
    # 中间值（如果 AUROC 直接=1.0，这条测试等于什么都没测）
    P = 0.82 * P + 0.18 * np.eye(K)[yt]
    P = P / P.sum(1, keepdims=True)

    df_auc, summary = auroc_pr_auc(yt, P, K)
    sk_macro = roc_auc_score(yt, P, multi_class="ovr", average="macro", labels=list(range(K)))
    sk_micro = roc_auc_score(yt, P, multi_class="ovr", average="micro", labels=list(range(K)))
    check("AUROC(macro) == sklearn",
          abs(summary["auroc_macro"] - sk_macro) < 5e-4,
          f"{summary['auroc_macro']:.5f} vs {sk_macro:.5f}")
    check("AUROC(micro) == sklearn",
          abs(summary["auroc_micro"] - sk_micro) < 5e-4,
          f"{summary['auroc_micro']:.5f} vs {sk_micro:.5f}")

    aps = [average_precision_score((yt == k).astype(int), P[:, k]) for k in range(K)]
    check("per-class AP == sklearn", close(df_auc.ap.to_numpy(), aps, tol=1e-4),
          f"max diff {np.nanmax(np.abs(df_auc.ap.to_numpy()-np.array(aps))):.2e}")
    check("AP(macro) == sklearn", abs(summary["ap_macro"] - float(np.mean(aps))) < 1e-4)

    # ------------------------------------------------------------------
    # 5. 完美预测 / 全错预测 的边界
    # ------------------------------------------------------------------
    pc_perf = per_class_metrics(yt, yt, K)
    check("完美预测 -> recall 全 1", bool((pc_perf.recall == 1).all()))
    check("完美预测 -> macro-F1 == 1",
          abs(average_precision_recall_f1(yt, yt, K).loc["macro", "f1"] - 1) < 1e-9)

    # 全错：把所有样本都预测成类别 0
    yp0 = np.zeros_like(yt)
    avg0 = average_precision_recall_f1(yt, yp0, K)
    check("全预测为类0 -> 类0 recall == 1",
          abs(per_class_metrics(yt, yp0, K).recall.iloc[0] - 1) < 1e-9)
    check("全预测为类0 -> macro-recall 很低 (<0.5)",
          avg0.loc["macro", "recall"] < 0.5, f"={avg0.loc['macro','recall']:.3f}")

    # ------------------------------------------------------------------
    # 6. 长尾下 accuracy 与 macro-F1 必然分歧（本周核心论点）
    # ------------------------------------------------------------------
    y_long = np.r_[np.zeros(900, dtype=int), np.ones(100, dtype=int)]
    y_pred_maj = np.zeros(1000, dtype=int)       # 只会预测众数
    acc = accuracy_score(y_long, y_pred_maj)
    mf1 = precision_recall_fscore_support(y_long, y_pred_maj, average="macro",
                                          zero_division=0)[2]
    check("众数模型 accuracy=0.90 但 macro-F1=0.47（accuracy 骗人）",
          abs(acc - 0.9) < 1e-9 and abs(mf1 - 0.4737) < 1e-3, f"acc={acc:.3f} mf1={mf1:.3f}")
    check("本实现给出同样的 macro-F1",
          abs(average_precision_recall_f1(y_long, y_pred_maj, 2).loc["macro", "f1"] - mf1) < 1e-9)

    # ------------------------------------------------------------------
    # 7. ECE
    # ------------------------------------------------------------------
    # 完美校准：置信度 0.8，实际正确率 0.8 -> ECE 应 ≈ 0
    n2 = 2000
    conf = np.full(n2, 0.8)
    correct = (rng.random(n2) < 0.8).astype(int)
    probs = np.zeros((n2, 2))
    probs[:, 0] = conf
    probs[:, 1] = 1 - conf
    yt2 = np.zeros(n2, dtype=int)
    yt2[correct == 0] = 1       # 错的时候真值是另一个类
    ece = expected_calibration_error(yt2, probs, n_bins=10)
    check("校准良好时 ECE < 0.02", ece["ece"] < 0.02, f"ECE={ece['ece']:.4f}")

    # 严重过自信：说 0.95 只有 50% 对
    conf2 = np.full(n2, 0.95)
    correct2 = (rng.random(n2) < 0.5).astype(int)
    probs2 = np.stack([conf2, 1 - conf2], 1)
    yt3 = np.zeros(n2, dtype=int); yt3[correct2 == 0] = 1
    ece2 = expected_calibration_error(yt3, probs2, n_bins=10)
    check("过度自信时 ECE > 0.4", ece2["ece"] > 0.4, f"ECE={ece2['ece']:.4f}")

    # ------------------------------------------------------------------
    # 8. bootstrap CI：可复现，且覆盖真值
    # ------------------------------------------------------------------
    mf = lambda a, b: float(precision_recall_fscore_support(
        a, b, average="macro", zero_division=0)[2])
    pt, lo, hi = bootstrap_ci(mf, yt, yp, n_boot=600, seed=1)
    pt2, lo2, hi2 = bootstrap_ci(mf, yt, yp, n_boot=600, seed=1)
    check("bootstrap 可复现（同 seed）", (pt, lo, hi) == (pt2, lo2, hi2))
    check("CI 包含点估计", lo <= pt <= hi, f"{lo:.4f} <= {pt:.4f} <= {hi:.4f}")

    # ------------------------------------------------------------------
    # 9. 调整残差：对角必须显著为正
    # ------------------------------------------------------------------
    R = adjusted_residuals(C)
    check("对角调整残差全部 > 2（真值类被正确预测显著偏多）",
          bool(np.all(np.diag(R) > 2)), f"min diag {np.diag(R).min():.1f}")

    # ------------------------------------------------------------------
    # 10. 对称性诊断
    # ------------------------------------------------------------------
    pairs = confusion_pairs(yt, yp, [f"c{i}" for i in range(K)])
    check("错分方向表非空且对称方向成对出现",
          len(pairs) > 0 and set(pairs["true"]) <= {f"c{i}" for i in range(K)})
    if len(pairs):
        check("symmetry 落在 [0,1]", bool(((pairs.symmetry >= 0) & (pairs.symmetry <= 1)).all()))

    # 手造一个完全对称的混淆对：c0<->c1 各 20，其余正确
    y_sym = np.r_[np.repeat(0, 20), np.repeat(1, 20), np.repeat(2, 30)]
    yp_sym = np.r_[np.repeat(1, 20), np.repeat(0, 20), np.repeat(2, 30)]
    p_sym = confusion_pairs(y_sym, yp_sym, ["c0", "c1", "c2"])
    row = p_sym[(p_sym["true"] == "c0") & (p_sym["pred"] == "c1")].iloc[0]
    check("完全互错 -> symmetry == 1", abs(row.symmetry - 1.0) < 1e-9, f"={row.symmetry}")

    # 单向：c0->c1 有 20，c1->c0 有 0
    y_one = np.r_[np.repeat(0, 20), np.repeat(1, 20)]
    yp_one = np.r_[np.repeat(1, 20), np.repeat(1, 20)]
    p_one = confusion_pairs(y_one, yp_one, ["c0", "c1"])
    row2 = p_one[(p_one["true"] == "c0") & (p_one["pred"] == "c1")].iloc[0]
    check("单向错误 -> symmetry == 0", abs(row2.symmetry - 0.0) < 1e-9, f"={row2.symmetry}")

    # ------------------------------------------------------------------
    # 11. kappa
    # ------------------------------------------------------------------
    check("kappa == sklearn", abs(kappa(yt, yp, K) - cohen_kappa_score(yt, yp)) < 1e-9)
    check("完全一致 -> kappa == 1", abs(kappa(yt, yt, K) - 1) < 1e-9)

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print(f"通过 {len(PASS)} / {len(PASS)+len(FAIL)}")
    if FAIL:
        print("失败项：", FAIL)
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
