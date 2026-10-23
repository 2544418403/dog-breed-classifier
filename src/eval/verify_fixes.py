"""
scripts/verify_fixes.py —— 三条改进措施的量化验证

验收标准要求"3 条归因对应 3 条**可验证**的改进措施"。
"可验证"意味着：给得出提升幅度的点估计 + 置信区间 + 判定。

这里的三条措施都不需要重新训练，只做**后处理/推理侧**的改动，
所以可以在同一批 logits 上做**成对**比较 —— 统计上最干净：

    Fix-1  先验校正（logit adjustment）      治：少数类 precision 塌方
    Fix-2  个体级证据聚合                     治：稠密帧样本的"散弹式"错误
    Fix-3  温度缩放                           治：置信度过度自信（不改变 argmax）

每条都给出 before/after + 成对聚类 bootstrap 的判定。

用法：
    python scripts/verify_fixes.py --exp baseline
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
    per_class_metrics, expected_calibration_error, confusion_matrix,
    paired_cluster_bootstrap_test, cluster_bootstrap_ci,
    fit_temperature, apply_temperature, fit_prior_correction, apply_prior_correction,
)
from eval.kit import curves as viz

MANIFEST = DATASETS / "pet5" / "manifest.csv"
IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
MINORITY = [IDX["shiba_inu"], IDX["yorkshire_terrier"]]


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def m_all(y, yp):
    pc = per_class_metrics(y, yp, 5, LABEL_NAMES)
    return {
        "accuracy": float((yp == y).mean()),
        "macro_f1": float(pc.f1.mean()),
        "macro_recall": float(pc.recall.mean()),
        "minority_recall": float(np.mean([pc.recall.iloc[i] for i in MINORITY])),
        "minority_precision": float(np.mean([pc.precision.iloc[i] for i in MINORITY])),
        "worst_class_f1": float(pc.f1.min()),
    }


def subject_vote(probs, y, subjects):
    """个体级聚合：同一 subject 的多帧概率取平均（算术平均），再 argmax。

    参数 subjects 必须是**个体 id 数组**（与 probs 逐行对齐），
    不是路径 —— 这里曾经把路径传进来当分组键，结果 159 个个体被当成 546 个
    独立组，聚合完全失效，而"救回 4 个"的统计还照样算得出来（因为它用的是
    另一个 groupby）。**这类"看起来有结果、其实分组错了"的 bug 最难发现**，
    所以函数内部再断言一次分组数。

    只用该个体已出现的帧求平均，不引入任何全量统计，无信息泄漏。
    """
    subjects = np.asarray(subjects)
    K = probs.shape[1]
    df = pd.DataFrame(probs, columns=[f"p{k}" for k in range(K)])
    df["y"] = np.asarray(y)
    df["subj"] = subjects

    agg = df.groupby("subj")[[f"p{k}" for k in range(K)]].mean()
    subj_pred = agg.to_numpy().argmax(1)
    subj_true = df.groupby("subj")["y"].first().to_numpy()
    out_subj = agg.index.to_numpy()

    assert len(out_subj) < len(subjects), (
        f"聚合后组数 {len(out_subj)} 不少于行数 {len(subjects)}，"
        "说明分组键几乎唯一 —— 很可能误传了 path 而不是 subject_id"
    )
    return subj_pred, subj_true, out_subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    meta = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    p2s = dict(zip(meta.path, meta.subject_id))
    pi_train = meta[meta.split == "train"].label.value_counts().sort_index().to_numpy(float)
    pi_train /= pi_train.sum()

    z = np.load(OUT / f"preds_{args.exp}.npz", allow_pickle=True)
    lg = z[f"{args.split}_logits"].astype(np.float64)
    y = z[f"{args.split}_y"].astype(int)
    paths = z[f"{args.split}_path"]

    # 聚类单元必须是**个体**：同一只狗的多帧高度相关
    subj = np.array([p2s[p] for p in paths])

    # 严重性：同一只狗的多帧**一起**错，说明"这只狗整体不像这一类"，
    # 而不是"某一张图拍得差"。这就要求重抽样按个体聚合。
    base_pred = lg.argmax(1)
    base_m = m_all(y, base_pred)

    print("=" * 78)
    print(f"改进措施验证  exp={args.exp}  split={args.split}  n={len(y)}  个体数={len(np.unique(subj))}")
    print("=" * 78)
    print(f"baseline: acc={base_m['accuracy']:.4f} macroF1={base_m['macro_f1']:.4f} "
          f"少类P={base_m['minority_precision']:.4f} 少类R={base_m['minority_recall']:.4f}")

    rows, verdicts = [], []

    # ---------------- Fix-1 先验校正 ----------------
    pc_res = fit_prior_correction(z["val_logits"], z["val_y"].astype(int), pi_train, objective="macro_f1")
    tau = pc_res["tau"]
    lg1 = apply_prior_correction(lg, tau, np.asarray(pc_res["shift"]))
    m1 = m_all(y, lg1.argmax(1))
    r1 = paired_cluster_bootstrap_test(_minprec, y, lg1.argmax(1), base_pred, subj, n_boot=args.n_boot)
    r1b = paired_cluster_bootstrap_test(_mf1, y, lg1.argmax(1), base_pred, subj, n_boot=args.n_boot)
    print(f"\n[Fix-1] 先验校正 τ={tau}  (在 val 上按 macro-F1 拟合)")
    print(f"  少类 precision {base_m['minority_precision']:.4f} -> {m1['minority_precision']:.4f} "
          f"({100*(m1['minority_precision']-base_m['minority_precision']):+.2f}pp)  "
          f"配对聚类 95%CI [{r1['ci_lo']:+.4f},{r1['ci_hi']:+.4f}] p_gt0={r1['p_gt0']}")
    print(f"  macro-F1       {base_m['macro_f1']:.4f} -> {m1['macro_f1']:.4f} "
          f"({100*(m1['macro_f1']-base_m['macro_f1']):+.2f}pp)  "
          f"95%CI [{r1b['ci_lo']:+.4f},{r1b['ci_hi']:+.4f}]")
    print(f"  accuracy       {base_m['accuracy']:.4f} -> {m1['accuracy']:.4f}")
    rows.append({"fix": "Fix-1 prior correction", "tau_or_k": tau,
                 **{f"{k}_before": round(base_m[k], 4) for k in base_m},
                 **{f"{k}_after": round(m1[k], 4) for k in m1},
                 "d_minprec": round(m1["minority_precision"] - base_m["minority_precision"], 4),
                 "ci_minprec": f"[{r1['ci_lo']:+.4f},{r1['ci_hi']:+.4f}]",
                 "d_macrof1": round(m1["macro_f1"] - base_m["macro_f1"], 4),
                 "ci_macrof1": f"[{r1b['ci_lo']:+.4f},{r1b['ci_hi']:+.4f}]",
                 "significant_minprec": bool(r1["ci_lo"] > 0)})
    verdicts.append(("Fix-1", r1["ci_lo"] > 0, "少数类 precision 显著提升"))

    # ---------------- Fix-2 个体级聚合 ----------------
    P = softmax_np(lg)
    sp, st, su = subject_vote(P, y, subj)
    # 个体级 bootstrap：重抽样单位 = 个体（每个个体一行，天然独立）
    pt_acc_s, lo_acc_s, hi_acc_s = cluster_bootstrap_ci(
        lambda a, b: float((a == b).mean()), st, sp, su, n_boot=args.n_boot)

    # 图片级的聚类 CI（同口径，便于并列展示）
    pt_acc_i, lo_acc_i, hi_acc_i = cluster_bootstrap_ci(
        lambda a, b: float((a == b).mean()), y, base_pred, subj, n_boot=args.n_boot)

    # 有多少"部分帧错"的个体被救回
    dfr = pd.DataFrame({"subj": subj, "y": y, "correct": (base_pred == y).astype(int)})
    g = dfr.groupby("subj").agg(n=("correct", "size"), nerr=("correct", lambda s: int((s == 0).sum())))
    partial = g[(g.nerr > 0) & (g.nerr < g.n)].index
    rescued = [s for s in partial if sp[list(su).index(s)] == st[list(su).index(s)]]
    broke = [s for s in g[g.nerr == 0].index if sp[list(su).index(s)] != st[list(su).index(s)]]

    # 三个口径必须分清，混着比就会得出自相矛盾的结论：
    #   ① 图片级 acc        : 逐帧判对错（546 个决策）
    #   ② 个体级 all-or-nothing: 一只狗所有帧全对才算对（159 个决策，最严）
    #   ③ 个体级聚合后 acc   : 先按个体平均概率再 argmax（159 个决策，这是 Fix-2 的产出）
    # ① 和 ③ 的**决策单元不同**，不能直接相减当"提升"；能比的是 ② -> ③。
    allok = g.assign(ok_all=(g.nerr == 0)).ok_all.mean()
    vote_maj = _majority(y, subj, base_pred)
    print(f"\n[Fix-2] 个体级证据聚合（{len(su)} 个个体，平均 {len(y)/len(su):.2f} 帧/个体）")
    print(f"  ① 图片级 accuracy        {pt_acc_i:.4f}  [{lo_acc_i:.4f},{hi_acc_i:.4f}]   (n_img={len(y)})")
    print(f"  ② 个体级 all-or-nothing  {allok:.4f}   (n_subj={len(su)}，一只狗全对才算对)")
    print(f"  ③ 个体级 平均概率聚合     {pt_acc_s:.4f}  [{lo_acc_s:.4f},{hi_acc_s:.4f}]   (n_subj={len(su)})")
    print(f"     个体级 众数投票         {vote_maj:.4f}   （对照：两种聚合方式差别不大）")
    print(f"  「部分帧错」的个体 {len(partial)} 个 -> 聚合后救回 {len(rescued)} 个；"
          f" 原本全对的个体里被聚合改错 {len(broke)} 个")
    print(f"  净收益 {len(rescued)-len(broke):+d} 个个体")
    print(f"  ② -> ③ 提升 {100*(pt_acc_s-allok):+.1f}pp（这是真正该比的量）")
    rows.append({"fix": "Fix-2 subject-level aggregation", "tau_or_k": f"{len(y)/len(su):.2f}frames/subj",
                 "accuracy_before": round(allok, 4), "accuracy_after": round(pt_acc_s, 4),
                 "d_minprec": np.nan,
                 "ci_minprec": f"[{lo_acc_s:.4f},{hi_acc_s:.4f}] subj",
                 "d_macrof1": round(pt_acc_s - allok, 4),
                 "ci_macrof1": "",
                 "significant_minprec": None,
                 "image_level_acc": round(pt_acc_i, 4),
                 "majority_vote_acc": round(vote_maj, 4),
                 "rescued_subj": len(rescued), "broken_subj": len(broke)})
    verdicts.append(("Fix-2", (pt_acc_s - allok) > 0 and len(broke) == 0,
                     f"个体级 {allok:.3f}->{pt_acc_s:.3f}，救回 {len(rescued)} 个、改错 {len(broke)} 个"))

    # ---------------- Fix-3 温度缩放 ----------------
    T = fit_temperature(z["val_logits"], z["val_y"].astype(int))
    P3 = apply_temperature(lg, T)
    ece_b = expected_calibration_error(y, P, n_bins=10)
    ece_a = expected_calibration_error(y, P3, n_bins=10)
    same_argmax = float((P3.argmax(1) == P.argmax(1)).mean())
    print(f"\n[Fix-3] 温度缩放 T={T:.3f}（在 val 上拟合 NLL）")
    print(f"  ECE {ece_b['ece']:.4f} -> {ece_a['ece']:.4f}    MCE {ece_b['mce']:.4f} -> {ece_a['mce']:.4f}")
    print(f"  argmax 一致率 {same_argmax:.4f}  ->  准确率/macro-F1 **不变**（这是特性不是缺陷）")
    rows.append({"fix": "Fix-3 temperature scaling", "tau_or_k": round(T, 4),
                 "ece_before": round(ece_b["ece"], 4), "ece_after": round(ece_a["ece"], 4),
                 "mce_before": round(ece_b["mce"], 4), "mce_after": round(ece_a["mce"], 4),
                 "argmax_same": same_argmax, "significant_minprec": None})
    verdicts.append(("Fix-3", ece_a["ece"] < ece_b["ece"], f"ECE {ece_b['ece']:.3f}->{ece_a['ece']:.3f}"))

    # ---------------- 汇总 ----------------
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"fixes_{args.exp}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"fixes_{args.exp}.json").write_text(
        json.dumps({"exp": args.exp, "split": args.split, "baseline": base_m,
                    "fix1": m1, "fix2_subject_level_acc": pt_acc_s, "fix2_image_level_acc": pt_acc_i,
                    "fix2_rescued": len(rescued), "fix2_broken": len(broke),
                    "fix3": {"T": T, "ece_before": ece_b["ece"], "ece_after": ece_a["ece"]},
                    "tau": tau},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    viz.plot_reliability(ece_a["bins"], ece_a["ece"], FIG / f"fig_reliability_fix3_{args.exp}.png")

    print("\n" + "=" * 78)
    print("三条措施判定")
    print("=" * 78)
    for name, ok, desc in verdicts:
        print(f"  {name}  {'成立' if ok else '不成立'}   ({desc})")


def _minprec(y, yp):
    return m_all(y, yp)["minority_precision"]


def _mf1(y, yp):
    return m_all(y, yp)["macro_f1"]


def _majority(y, subj, pred):
    """个体级众数投票（与平均概率对照用）。"""
    import pandas as pd

    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pred), "s": np.asarray(subj)})
    m = d.groupby("s")["p"].agg(lambda s: s.value_counts().idxmax()).to_numpy()
    t = d.groupby("s")["y"].first().to_numpy()
    return float((m == t).mean())


if __name__ == "__main__":
    main()


