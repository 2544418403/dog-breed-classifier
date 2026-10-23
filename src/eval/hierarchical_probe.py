"""
scripts/hierarchical_probe.py —— 逐级归因：错误到底发生在哪一步？

================================================================================
动机
================================================================================
只看混淆矩阵，你会说"shiba 和 york 互相错了很多，说明它们长得像"。
但这句话是**错的**，而且错得很典型。混淆矩阵把两种完全不同的错误混在了一起：

    A. 类间混淆     模型说"这是 shiba 或 york 之一"，然后选错了那一个
    B. 路由失败     模型说"这是 pomeranian"，根本没把样本放进正确的候选组

这两件事的病因和治疗**完全不同**：
    A 是特征表达不够（要更高分辨率/更好 backbone/更多数据）
    B 是决策边界被先验/分布偏移推歪了（要校准/重加权/调先验）

本脚本把 5 类决策拆成两级，量出每一级的准确率：
    第 1 级  路由：样本能否进入正确的"候选组"{shiba, york} / {havanese, pomeranian} / {min pin}
    第 2 级  组内区分：进了组之后，能否分对组内的两个类

结论（本项目实测）见报告 §4 —— 第 2 级准确率是 **100%**，所有错误都在第 1 级。
这一个数字就足以推翻"类间不可分"的假设。
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


IDX = {n: i for i, n in enumerate(LABEL_NAMES)}

# 由特征空间几何 + 混淆结构归纳出的两个"易混组"
GROUPS = {
    "A_shiba_york": [IDX["shiba_inu"], IDX["yorkshire_terrier"]],
    "B_hava_pom": [IDX["havanese"], IDX["pomeranian"]],
    "C_minpin": [IDX["miniature_pinscher"]],
}


def group_of(cls_id: int) -> str:
    for g, members in GROUPS.items():
        if cls_id in members:
            return g
    return "?"


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def analyze(preds_file: Path) -> dict:
    z = np.load(preds_file, allow_pickle=True)
    lg = z["test_logits"]
    y = z["test_y"].astype(int)
    P = softmax(lg)
    yp = P.argmax(1)

    rec = {"model": preds_file.stem.replace("preds_", ""), "n": int(len(y)),
           "overall_acc": round(float((yp == y).mean()), 4), "groups": {}}

    for gname, members in GROUPS.items():
        # 只对"真值属于本组"的样本评估
        sel = np.isin(y, members)
        if sel.sum() == 0:
            continue
        # 第 1 级：路由 —— 预测是否落入本组
        routed = np.isin(yp[sel], members)
        stage1 = float(routed.mean())
        # 第 2 级：组内区分 —— 对已正确路由的样本，是否猜对组内具体类
        if routed.sum() > 0:
            stage2 = float((yp[sel][routed] == y[sel][routed]).mean())
        else:
            stage2 = float("nan")
        # 若因单一类别（min pin）无组内区分，stage2 记 nan
        if len(members) == 1:
            stage2 = float("nan")

        # 错误样本的去向（诊断"被谁抢走"）
        lost_to = {}
        for c in range(len(LABEL_NAMES)):
            k = int(((yp[sel] == c) & (y[sel] != c)).sum())
            if k:
                lost_to[LABEL_NAMES[c]] = k

        rec["groups"][gname] = {
            "members": [LABEL_NAMES[i] for i in members],
            "n_samples": int(sel.sum()),
            "stage1_routing_acc": round(stage1, 4),
            "stage2_within_acc": (None if stage2 != stage2 else round(stage2, 4)),
            "end_to_end": round(float((yp[sel] == y[sel]).mean()), 4),
            "errors_lost_to": lost_to,
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    files = ([OUT / f"preds_{m}.npz" for m in args.models] if args.models
             else sorted(OUT.glob("preds_*.npz")))
    files = [f for f in files if f.exists()]

    rows, detail = [], {}
    for f in files:
        r = analyze(f)
        detail[r["model"]] = r
        row = {"model": r["model"], "overall_acc": r["overall_acc"]}
        for g, v in r["groups"].items():
            row[f"{g}__route"] = v["stage1_routing_acc"]
            row[f"{g}__within"] = v["stage2_within_acc"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("overall_acc")
    pd.set_option("display.width", 220)
    print("=" * 110)
    print("逐级归因：route = 样本能否进入正确候选组 | within = 进组后能否分对组内类")
    print("=" * 110)
    print(df.to_string(index=False))

    print("\n各组错误去向（每行一个模型）：")
    for m, r in detail.items():
        for g, v in r["groups"].items():
            if v["errors_lost_to"]:
                print(f"  [{m:12s}] {g:14s} n={v['n_samples']:3d}  "
                      f"route={v['stage1_routing_acc']:.3f} within={v['stage2_within_acc']}  "
                      f"错分去向={v['errors_lost_to']}")

    df.to_csv(OUT / "hierarchical.csv", index=False, encoding="utf-8-sig")
    (OUT / "hierarchical.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n[完成] outputs/hierarchical.csv")


if __name__ == "__main__":
    main()
