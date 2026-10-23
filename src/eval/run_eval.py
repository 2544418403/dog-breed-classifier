"""
scripts/run_eval.py —— 深度评估：把"准确率"拆成一张完整的体检报告

用法：
    python scripts/run_eval.py --exp baseline

产出：
    outputs/metrics_<exp>.json     全套指标（含 CI）
    outputs/perclass_<exp>.csv
    outputs/frame_<exp>.csv
    figures/fig_confusion_<exp>.png
    figures/fig_rocpr_<exp>.png
    figures/fig_perclass_<exp>.png
    figures/fig_ladder_<exp>.png
    figures/fig_reliability_<exp>.png
    figures/fig_conf_strata_<exp>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import (  # noqa: E402
    DATASETS, OUTPUTS as OUT, FIGURES as FIG, WEIGHTS, LABEL_NAMES, ensure_dirs,
)

ensure_dirs()   # 幂等：artifacts/ figures/ weights/ datasets/ logs/ 缺哪个补哪个


from eval.kit import (
    confusion_matrix, per_class_metrics, average_precision_recall_f1,
    auroc_pr_auc, expected_calibration_error, kappa, bootstrap_ci,
    build_frame, confidence_strata,
)
from eval.kit import curves as viz
from eval.kit.calibrate import apply_temperature, fit_temperature

MANIFEST = DATASETS / "pet5" / "manifest.csv"


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_recall(y, yp):
    return float(per_class_metrics(y, yp, len(LABEL_NAMES)).recall.mean())


def macro_f1(y, yp):
    return float(per_class_metrics(y, yp, len(LABEL_NAMES)).f1.mean())


def worst_f1(y, yp):
    return float(per_class_metrics(y, yp, len(LABEL_NAMES)).f1.min())


def minority_recall(y, yp, minority_idx=4):
    """少数类（默认 yorkshire_terrier）的 recall —— 长尾任务里最该盯的一个数。"""
    m = y == minority_idx
    if m.sum() == 0:
        return float("nan")
    return float((yp[m] == minority_idx).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    npz = np.load(OUT / f"preds_{args.exp}.npz", allow_pickle=True)
    logits = npz[f"{args.split}_logits"]
    y = npz[f"{args.split}_y"].astype(int)
    path = npz[f"{args.split}_path"]
    meta = pd.read_csv(MANIFEST, encoding="utf-8-sig")

    P = softmax_np(logits)
    yp = P.argmax(1)
    K = len(LABEL_NAMES)

    print("=" * 78)
    print(f"深度评估  exp={args.exp}  split={args.split}  n={len(y)}")
    print("=" * 78)

    acc = float((yp == y).mean())
    avg = average_precision_recall_f1(y, yp, K)
    pc = per_class_metrics(y, yp, K, LABEL_NAMES)
    auc_df, auc_sum = auroc_pr_auc(y, P, K, LABEL_NAMES)
    ece = expected_calibration_error(y, P, n_bins=10)

    ladder = {
        "accuracy": round(acc, 4),
        "balanced_acc": round(float(pc.balanced_acc.mean()), 4),
        "macro_f1": round(avg.loc["macro", "f1"], 4),
        "macro_recall": round(avg.loc["macro", "recall"], 4),
        "macro_precision": round(avg.loc["macro", "precision"], 4),
        "weighted_f1": round(avg.loc["weighted", "f1"], 4),
        "worst_class_f1": round(float(pc.f1.min()), 4),
        "minority_recall": round(minority_recall(y, yp), 4),
        "kappa": round(kappa(y, yp, K), 4),
        "ece": round(ece["ece"], 4),
        "mce": round(ece["mce"], 4),
        "brier": round(ece["brier"], 4),
        "auroc_macro": round(auc_sum["auroc_macro"], 4),
        "ap_macro": round(auc_sum["ap_macro"], 4),
        "auroc_micro": round(auc_sum["auroc_micro"], 4),
    }

    cis = {}
    for name, fn in (("accuracy", lambda a, b: float((a == b).mean())),
                     ("macro_f1", lambda a, b: macro_f1(a, b)),
                     ("macro_recall", lambda a, b: macro_recall(a, b)),
                     ("minority_recall", lambda a, b: minority_recall(a, b)),
                     ("worst_class_f1", lambda a, b: worst_f1(a, b))):
        pt, lo, hi = bootstrap_ci(fn, y, yp, n_boot=args.n_boot, seed=7)
        cis[name] = {"point": pt, "ci95": [lo, hi], "width": round(hi - lo, 4)}

    print("\n--- 指标阶梯（看 accuracy 藏了多少）---")
    for k, v in ladder.items():
        ci = cis.get(k)
        extra = f"   95%CI [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]" if ci else ""
        print(f"  {k:16s} {v:.4f}{extra}")

    print("\n--- 每类指标 ---")
    tbl = pc.copy()
    tbl["auroc"] = auc_df.auroc.values
    tbl["ap"] = auc_df.ap.values
    print(tbl.round(4).to_string())
    tbl.round(6).to_csv(OUT / f"perclass_{args.exp}_{args.split}.csv", encoding="utf-8-sig")

    print("\n--- 平均方式的分歧 ---")
    print(avg.round(4).to_string())

    print("\n--- 校准 ---")
    print(f"  ECE={ece['ece']:.4f}  MCE={ece['mce']:.4f}  Brier={ece['brier']:.4f}")

    if args.split == "test":
        T = fit_temperature(npz["val_logits"], npz["val_y"].astype(int))
        Pc = apply_temperature(logits, T)
        ece_c = expected_calibration_error(y, Pc, n_bins=10)
        ladder["T_fitted_on_val"] = round(T, 4)
        ladder["ece_after_T"] = round(ece_c["ece"], 4)
        print(f"  [温度缩放] T={T:.3f} (fit on val)  ->  test ECE {ece['ece']:.4f} => {ece_c['ece']:.4f}")
        viz.plot_reliability(ece_c["bins"], ece_c["ece"],
                             FIG / f"fig_reliability_{args.exp}_calibrated.png")

    C = confusion_matrix(y, yp, K)
    viz.plot_confusion(C, LABEL_NAMES, FIG / f"fig_confusion_{args.exp}_{args.split}.png")
    viz.plot_per_class(tbl.reset_index(), FIG / f"fig_perclass_{args.exp}_{args.split}.png")
    viz.plot_metric_ladder(ladder, FIG / f"fig_ladder_{args.exp}_{args.split}.png")
    viz.plot_reliability(ece["bins"], ece["ece"], FIG / f"fig_reliability_{args.exp}_{args.split}.png")

    curves = auc_df.attrs.get("curves", {})
    if curves:
        viz.plot_roc_pr(curves, {}, FIG / f"fig_rocpr_{args.exp}_{args.split}.png",
                        title_prefix=f"{args.exp} ")

    frame = build_frame({"y_true": y, "probs": P, "path": path}, meta=meta, class_names=LABEL_NAMES)
    strata = confidence_strata(frame)
    print("\n--- 置信度分层 ---")
    print(strata.round(3).to_string(index=False))
    viz.plot_confidence_strata(strata, FIG / f"fig_conf_strata_{args.exp}_{args.split}.png")

    frame.to_csv(OUT / f"frame_{args.exp}_{args.split}.csv", index=False, encoding="utf-8-sig")

    result = {"exp": args.exp, "split": args.split, "n": int(len(y)),
              "metrics": ladder, "bootstrap_ci": cis,
              "auroc_ap": {k: (None if v != v else float(v)) for k, v in auc_sum.items()}}
    (OUT / f"metrics_{args.exp}_{args.split}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[完成] 指标已写入 outputs/metrics_{args.exp}_{args.split}.json")


if __name__ == "__main__":
    main()
