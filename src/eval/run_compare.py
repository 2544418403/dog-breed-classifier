"""
scripts/run_compare.py —— 消融对比 + 结论的统计把关

三个层次的比较，缺一不可：
  1. 单点指标对比     （最浅，只能看趋势）
  2. 聚类 bootstrap CI （承认样本不独立，看差异是否落在噪声外）
  3. 成对聚类 bootstrap （同一批重抽样上比较 A/B，最敏感也最诚实）

用法：
    python scripts/run_compare.py                    # 对比所有已训练的实验
    python scripts/run_compare.py --exps baseline cw_sqrt focal

产出：
    outputs/ablation.csv         主表（含 CI 与显著性判定）
    outputs/ablation_paired.csv  与 baseline 的成对比较
    figures/fig_ablation_*.png
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
    per_class_metrics, confusion_matrix, cluster_bootstrap_ci,
    paired_cluster_bootstrap_test, fit_temperature, apply_temperature,
    fit_prior_correction, apply_prior_correction,
)
from eval.kit import curves as viz

MANIFEST = DATASETS / "pet5" / "manifest.csv"

IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
MINORITY = [IDX["shiba_inu"], IDX["yorkshire_terrier"]]
HARD_PAIRS = [("miniature_pinscher", "shiba_inu"), ("havanese", "yorkshire_terrier")]


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _minority_precision(y, yp):
    """少数类（shiba + yorkshire）的平均 precision —— 本任务里最敏感的指标。"""
    pc = per_class_metrics(y, yp, 5, LABEL_NAMES)
    return float(np.mean([pc.precision.iloc[i] for i in MINORITY]))


def metrics_from_logits(lg, y):
    yp = lg.argmax(1)
    pc = per_class_metrics(y, yp, 5, LABEL_NAMES)
    C = confusion_matrix(y, yp, 5)
    pair_err = {}
    for a, b in HARD_PAIRS:
        ia, ib = IDX[a], IDX[b]
        pair_err[f"{a[:4]}->{b[:4]}"] = int(C[ia, ib])
        pair_err[f"{b[:4]}->{a[:4]}"] = int(C[ib, ia])
    return {
        "accuracy": float((yp == y).mean()),
        "macro_f1": float(pc.f1.mean()),
        "macro_recall": float(pc.recall.mean()),
        "minority_recall": float(np.mean([pc.recall.iloc[i] for i in MINORITY])),
        "minority_precision": float(np.mean([pc.precision.iloc[i] for i in MINORITY])),
        "worst_class_f1": float(pc.f1.min()),
        **pair_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None, help="不传则自动发现所有 preds_*.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    meta = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    path2subj = dict(zip(meta.path, meta.subject_id))
    pi_train = meta[meta.split == "train"].label.value_counts().sort_index().to_numpy(dtype=float)
    pi_train = pi_train / pi_train.sum()

    if args.exps is None:
        args.exps = sorted({p.name[6:-4] for p in OUT.glob("preds_*.npz")})
    # uniform2 是 2 类子集证伪实验，其 preds 只覆盖 test 的一部分（51 个样本），
    # 与 5 类全量 preds 不可成对比较 —— 不进消融表，结论由 hierarchical_probe 单独呈现。
    args.exps = [e for e in args.exps if e != "uniform2"]
    print(f"对比实验：{args.exps}\n")

    rows, paired = [], []
    ref_lg = ref_y = ref_cl = None

    for exp in args.exps:
        f = OUT / f"preds_{exp}.npz"
        if not f.exists():
            print(f"  [跳过] 缺少 {f.name}")
            continue
        npz = np.load(f, allow_pickle=True)
        lg = npz[f"{args.split}_logits"].astype(np.float64)
        y = npz[f"{args.split}_y"].astype(int)
        paths = npz[f"{args.split}_path"]
        cl = np.array([path2subj[p] for p in paths])

        # ---------- 原始 ----------
        m = metrics_from_logits(lg, y)

        # ---------- 后处理 1：温度缩放（不改变 argmax，只改校准） ----------
        T = fit_temperature(npz["val_logits"], npz["val_y"].astype(int))

        # ---------- 后处理 2：先验校正（会改变 argmax） ----------
        pc_res = fit_prior_correction(npz["val_logits"], npz["val_y"].astype(int), pi_train,
                                      objective="macro_f1")
        lg_adj = apply_prior_correction(lg, pc_res["tau"], np.asarray(pc_res["shift"]))
        m_adj = metrics_from_logits(lg_adj, y)

        row = {"exp": exp, "T": round(T, 3), "tau_prior": pc_res["tau"], **{k: round(v, 4) for k, v in m.items()}}
        row["macro_f1_adj"] = round(m_adj["macro_f1"], 4)
        row["minority_prec_adj"] = round(m_adj["minority_precision"], 4)
        row["acc_adj"] = round(m_adj["accuracy"], 4)
        rows.append(row)

        # ---------- 聚类 bootstrap CI（关键指标） ----------
        for key, fn in (("accuracy", lambda a, b: float((a == b).mean())),
                        ("macro_f1", lambda a, b: float(per_class_metrics(a, b, 5).f1.mean())),
                        ("minority_precision", _minority_precision)):
            pt, lo, hi = cluster_bootstrap_ci(fn, y, lg.argmax(1), cl, n_boot=args.n_boot, seed=0)
            row[f"{key}_ci"] = f"[{lo:.3f},{hi:.3f}]"

        # ---------- 与 baseline 的成对聚类比较 ----------
        if ref_lg is None:
            ref_lg, ref_y, ref_cl = lg, y, cl
        else:
            for key, fn in (("macro_f1", lambda a, b: float(per_class_metrics(a, b, 5).f1.mean())),
                            ("minority_precision", _minority_precision),
                            ("accuracy", lambda a, b: float((a == b).mean()))):
                r = paired_cluster_bootstrap_test(fn, ref_y, lg.argmax(1), ref_lg.argmax(1),
                                                  ref_cl, n_boot=args.n_boot, seed=0)
                paired.append({"exp": exp, "metric": key, **r})

    df = pd.DataFrame(rows)
    cols_first = ["exp", "accuracy", "macro_f1", "minority_recall", "minority_precision",
                  "worst_class_f1", "minp->shib", "hava->york", "york->hava", "T", "tau_prior",
                  "acc_adj", "macro_f1_adj", "minority_prec_adj"]
    cols = [c for c in cols_first if c in df.columns] + [c for c in df.columns if c not in cols_first]
    df = df[cols]

    pd.set_option("display.width", 240)
    print("=" * 120)
    print(f"消融对比  split={args.split}   （minp->shib / hava->york = 该错分方向的样本数，越小越好）")
    print("=" * 120)
    print(df.to_string(index=False))

    df.to_csv(OUT / "ablation.csv", index=False, encoding="utf-8-sig")

    if paired:
        pdf = pd.DataFrame(paired)
        print("\n" + "=" * 120)
        print("与 baseline 的成对聚类 bootstrap 比较（diff = 该实验 - baseline）")
        print("判据：ci_lo > 0 才能声称显著提升；ci_hi < 0 才能声称显著下降")
        print("=" * 120)
        print(pdf.to_string(index=False))
        pdf.to_csv(OUT / "ablation_paired.csv", index=False, encoding="utf-8-sig")

    # ---------- 图 ----------
    if len(df) > 1:
        d = df.copy()
        d["label"] = d["exp"]
        for metric, title in (("macro_f1", "Macro-F1 across ablations"),
                              ("minority_precision", "Minority-class precision (shiba+yorkshire avg)"),
                              ("accuracy", "Accuracy across ablations")):
            sub = d[["exp", metric]].copy()
            viz.plot_ablation(sub, metric, FIG / f"fig_ablation_{metric}.png", title=title)

    print(f"\n[完成] outputs/ablation.csv")


if __name__ == "__main__":
    main()
