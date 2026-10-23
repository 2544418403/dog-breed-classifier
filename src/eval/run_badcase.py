"""
scripts/run_badcase.py —— badcase 挖掘：从"有 51 个错分"到"这 51 个错分是谁"

流程（顺序很重要）：
  1. 导出全部错分样本（不抽样）
  2. 按 **错分方向** 分组 + 对称性诊断
  3. 挂上协变量（灰度/EXIF/清晰度/板号…），做分层对照 —— 这一步把"感觉"变成"数字"
  4. 特征空间几何：类心余弦，验证"这对类是不是真的挨着"
  5. 生成拼图，人眼确认

用法：
    python scripts/run_badcase.py --exp baseline [--pool]   # --pool 表示 val+test 合并以增加样本量

产出：
    outputs/errors_<exp>.csv          全部错分样本（含协变量）
    outputs/pairs_<exp>.csv           错分方向诊断
    outputs/subgroup_<exp>.csv        子组对照结果
    outputs/geometry_<exp>.csv        类心余弦矩阵
    figures/fig_pairs_<exp>.png
    figures/fig_badcase_*_<exp>.png   每个错分方向一张拼图
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
    build_frame, error_table, confusion_pairs, error_summary, group_errors,
    contact_sheet, confusion_matrix,
)
from eval.kit import curves as viz
from eval.kit.quality import probe_dataframe
from eval.kit.attribution import group_compare, class_geometry, prior_collapse_test

CKPT = WEIGHTS
DATA = DATASETS / "pet5"
MANIFEST = DATASETS / "pet5" / "manifest.csv"


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--pool", action="store_true",
                    help="val+test 合并分析。val 仅用于监控（未参与选模），"
                         "合并可把错分样本翻一倍，子组对照才有统计功效")
    ap.add_argument("--max-sheets", type=int, default=4)
    args = ap.parse_args()

    npz = np.load(OUT / f"preds_{args.exp}.npz", allow_pickle=True)
    meta = pd.read_csv(MANIFEST, encoding="utf-8-sig")

    splits = ["val", "test"] if args.pool else ["test"]
    frames = []
    for s in splits:
        P = softmax_np(npz[f"{s}_logits"])
        f = build_frame({"y_true": npz[f"{s}_y"].astype(int), "probs": P,
                         "path": npz[f"{s}_path"]}, meta=meta, class_names=LABEL_NAMES)
        f["split"] = s
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)

    print("=" * 78)
    print(f"badcase 挖掘  exp={args.exp}  splits={splits}  n={len(frame)}")
    print("=" * 78)

    # ---------------------------------------------------------- 1. 错分样本
    err = error_table(frame, sort_by="margin")
    print(f"\n错分样本 {len(err)} / {len(frame)} = {100*len(err)/len(frame):.2f}%")

    # ---------------------------------------------------------- 2. 客观质量探针
    print("\n[1/5] 图像客观质量探针 ...")
    frame = probe_dataframe(frame, DATA, path_col="path", show_progress=False)
    err = frame[frame.correct == 0].copy().sort_values("margin")
    ok = frame[frame.correct == 1].copy()

    qcols = ["brightness", "contrast", "sharpness", "hist_entropy", "colorfulness"]
    print("\n  错分 vs 正确 的质量均值：")
    cmp_tbl = pd.DataFrame({
        "err_mean": err[qcols].mean(),
        "ok_mean": ok[qcols].mean(),
    })
    cmp_tbl["diff"] = cmp_tbl.err_mean - cmp_tbl.ok_mean
    print(cmp_tbl.round(3).to_string())
    viz.plot_quality_boxes(err, ok, qcols, FIG / f"fig_quality_{args.exp}.png")

    # ---------------------------------------------------------- 3. 错分方向诊断
    print("\n[2/5] 错分方向诊断")
    pairs = confusion_pairs(frame.y_true.to_numpy(), frame.y_pred.to_numpy(), LABEL_NAMES)
    print("\n" + pairs.round(3).to_string(index=False))
    pairs.to_csv(OUT / f"pairs_{args.exp}.csv", index=False, encoding="utf-8-sig")
    viz.plot_pair_diagnosis(pairs, FIG / f"fig_pairs_{args.exp}.png")

    # 错误流向 vs 按 support 的期望（虹吸效应检验）
    sup = frame.y_true.value_counts().sort_index()
    dest = err.y_pred.value_counts().sort_index()
    exp_dest = (len(err) * sup / sup.sum()).sort_index()
    flow = pd.DataFrame({
        "class": LABEL_NAMES,
        "n_error_absorbed": dest.reindex(range(5)).fillna(0).astype(int).values,
        "expected_if_proportional": exp_dest.values.round(1),
        "support": sup.reindex(range(5)).values,
    })
    flow["lift"] = (flow.n_error_absorbed / flow.expected_if_proportional).round(2)
    print("\n  错误被哪个类吸收（vs 按 support 的等比例期望）：")
    print(flow.round(2).to_string(index=False))
    flow.to_csv(OUT / f"error_flow_{args.exp}.csv", index=False, encoding="utf-8-sig")

    collapse = prior_collapse_test(frame)
    print(f"\n  错误流入量 vs log(train n) 的 Spearman = {collapse['spearman_vs_log_n']:.3f}")
    print("  （负值 = 错误反而流向**少**类，即虹吸；正值 = 经典的 prior 塌缩）")

    # ---------------------------------------------------------- 4. 子组对照
    print("\n[3/5] 子组分层对照（在每个类别内部比较，排除类构成混淆）")
    subgroups = {}

    if "is_grayscale" in frame.columns:
        mask = (frame.is_grayscale == True).to_numpy()  # noqa: E712
        subgroups["injected_grayscale"] = mask
    # 探针判定的灰度（R=G=B）
    if "is_gray" in frame.columns:
        subgroups["probe_gray"] = (frame.is_gray == 1).to_numpy()
    if "exif_orientation" in frame.columns:
        subgroups["exif_rotated"] = (frame.exif_orientation != 0).to_numpy()
    # 低清晰度：用错分+正确样本合并分布的下四分位
    if "sharpness" in frame.columns:
        q = frame.sharpness.quantile(0.25)
        subgroups["low_sharpness"] = (frame.sharpness <= q).to_numpy()
    if "colorfulness" in frame.columns:
        q = frame.colorfulness.quantile(0.25)
        subgroups["low_color"] = (frame.colorfulness <= q).to_numpy()

    sg_rows = []
    for name, mask in subgroups.items():
        n = int(mask.sum())
        if n < 5 or n >= len(frame) - 5:
            print(f"  - {name}: n={n} 太少，跳过")
            continue
        r = group_compare(frame, mask, stratify_by="class_true", n_boot=2000, seed=0)
        sg_rows.append({"subgroup": name, "n": n, **{k: v for k, v in r.items()
                                                     if k != "per_stratum"}})
        flag = "显著" if r["significant"] else "不显著"
        print(f"  - {name:20s} n={n:4d}  错误率 {r['err_masked']:.3f} vs 对照 {r['err_ctrl']:.3f}  "
              f"分层diff {r['diff_stratified']:+.3f}  95%CI [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  {flag}")
        viz.plot_group_compare(r["per_stratum"], FIG / f"fig_subgroup_{name}_{args.exp}.png",
                               mask_name=name)
    sg = pd.DataFrame(sg_rows)
    sg.to_csv(OUT / f"subgroup_{args.exp}.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- 5. 特征空间几何
    print("\n[4/5] 特征空间几何（类心余弦）")
    try:
        geo = geometry(args.exp, splits)
        print("\n" + geo.round(3).to_string())
        geo.to_csv(OUT / f"geometry_{args.exp}.csv", encoding="utf-8-sig")
        print(f"\n  类内/类间平均距离比 = {geo.attrs.get('intra_inter_ratio', float('nan')):.3f}"
              "  （越接近 1 越难分）")
    except Exception as e:
        print(f"  [跳过] 特征空间几何失败: {e.__class__.__name__}: {e}")

    # ---------------------------------------------------------- 6. 拼图
    print("\n[5/5] 生成错分样本拼图")
    sheets = contact_sheet(err, DATA, FIG / f"fig_badcase_{args.exp}.png",
                           cols=5, rows_per_group=2, max_groups=args.max_sheets)
    for s in sheets:
        print(f"  {s.name}")

    # ---------------------------------------------------------- 落盘
    err.to_csv(OUT / f"errors_{args.exp}.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(OUT / f"frame_{args.exp}_pooled.csv", index=False, encoding="utf-8-sig")
    print(f"\n[完成] 错分表 outputs/errors_{args.exp}.csv  ({len(err)} 行)")


def geometry(exp: str, splits):
    """重建模型，抽倒数第二层特征，算类心余弦。"""
    import timm
    import torch
    from torchvision import transforms as T
    from datapipe import MyDataset, build_transform, policy_natural

    ck = torch.load(CKPT / f"{exp}.pth", map_location="cpu")
    cfg = ck["cfg"]
    img = cfg["img"]
    model = timm.create_model("resnet50", pretrained=False, num_classes=5)
    model.load_state_dict(ck["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    pol = policy_natural(img)
    tfm = build_transform("val", pol)
    meta = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    df = meta[meta.split.isin(splits)].reset_index(drop=True)
    ds = MyDataset(df, tfm, root=DATA, label_col="label", return_path=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    from eval.kit.attribution import embed
    feats = embed(model, loader, device)
    y = np.asarray(ds._labels)
    geo, _ = class_geometry(feats, y, LABEL_NAMES)
    return geo


if __name__ == "__main__":
    main()
