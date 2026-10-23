"""
scripts/train.py —— 统一训练器：消融实验只有一个变量在动

================================================================================
本周所有实验共用这一个脚本，"改了什么"全部由 --exp 预设 + CLI 覆盖控制。
这是消融能做对的前提：**如果不同实验走不同的代码路径，你比较的就不是变量本身**。
================================================================================

沿用的第三周约定（保证与前三周可比）：
  - timm resnet50 ImageNet 预训练，num_classes=5
  - 只解冻分类头 + 最后 2 个 stage，判别式学习率 (head 1e-3 / tail 1e-4)
  - 冻结未解冻层的 BN 统计量（第二周结论：解冻崩溃的元凶是 BN 漂移）
  - test 一律取第三周的 held-out plates（P00/P05/P19），**绝对标尺**

本周新增的旋钮：
  --loss   ce / ce_weighted / focal / cb_ce / cb_focal
  --aug    natural / none / robust（robust = 叠加灰度+90°旋转，模拟采集侧偏移）
  --img    224 / 384（细粒度任务的分辨率消融）
  --seed   多种子跑，用来区分"真提升"与"运气"

输出：
  ckpts/<exp>.pth
  outputs/preds_<exp>.npz      ← 存 logits（不是只存 argmax！后面所有分析靠它）
  outputs/history_<exp>.csv
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (  # noqa: E402
    DATASETS, FIGURES as FIG, OUTPUTS as OUT, WEIGHTS as CKPT, LABEL_NAMES, set_seed, ensure_dirs,
)
from data import MyDataset, build_transform, make_splits, policy_natural  # noqa: E402
from models.factory import build_model, freeze_bn, load_checkpoint as models_load  # noqa: E402
from train.losses import make_loss  # noqa: E402

ensure_dirs()

DATA = DATASETS / "pet5"
MANIFEST = DATA / "manifest.csv"
N_CLASSES = len(LABEL_NAMES)

# ------------------------------------------------------------------------------
# 实验预设：每一行只允许和 baseline 差 1~2 个字段，改动必须写在 name 里
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 为什么默认分辨率是 64 而不是 224
# ------------------------------------------------------------------------------
# 第三周的实验报告已经测过：224px 下 by_plate 的 val/test 都在 99.8% 附近，
# 属于 **饱和区** —— 那时候混淆矩阵几乎是纯对角，badcase 只有个位数，
# 做归因分析等于无米之炊。第三周为了看见"划分方式"的差异，自己就降到了 64px
# （by_plate test 93.22%）。本周继续沿用这个"能看见错误"的工作点。
#
# 这本身就是一个值得记住的教训：
#   **评估方法论必须在"模型不完美"的区间里练，在全对的模型上什么方法都学不到。**
# ------------------------------------------------------------------------------
PRESETS = {
    "baseline":     dict(loss="ce",           aug="natural", img=64,  epochs=12, bs=64),
    "no_aug":       dict(loss="ce",           aug="none",    img=64,  epochs=12, bs=64),
    "cw_sqrt":      dict(loss="ce_weighted",  aug="natural", img=64,  epochs=12, bs=64, weight_mode="sqrt"),
    "cw_inverse":   dict(loss="ce_weighted",  aug="natural", img=64,  epochs=12, bs=64, weight_mode="inverse"),
    "focal":        dict(loss="focal",        aug="natural", img=64,  epochs=12, bs=64, gamma=2.0, weight_mode="sqrt"),
    "cb_focal":     dict(loss="cb_focal",     aug="natural", img=64,  epochs=12, bs=64, gamma=2.0),
    "cb_ce":        dict(loss="cb_ce",        aug="natural", img=64,  epochs=12, bs=64),
    "img224":       dict(loss="ce",           aug="natural", img=224, epochs=12, bs=32),
    "img384":       dict(loss="ce",           aug="natural", img=384, epochs=12, bs=16),
    "res224":       dict(loss="ce",           aug="natural", img=224, epochs=12, bs=32),
    # 均匀先验训练：用 shiba/yorkshire 组成一个 2 类平衡子集，让模型在"无先验偏置"
    # 的条件下学习 shiba vs york 的**特征区分**。这不是为了上线，而是为了**证伪**：
    # 如果换成均匀先验后这一对仍然分不开，那归因就不是 prior，而是真的类间不可分。
    "uniform2":     dict(loss="ce",           aug="natural", img=224, epochs=12, bs=16,
                         subset_classes=(3, 4)),
    "gray_aug":     dict(loss="ce",           aug="robust",  img=64,  epochs=12, bs=64),
    # 组合技：在确认单因子有效之后再上
    "best_combo":   dict(loss="ce_weighted",  aug="robust",  img=224, epochs=12, bs=32, weight_mode="sqrt"),
}
DEFAULT = PRESETS["baseline"]


# ------------------------------------------------------------------------------
def build_transform_variant(kind: str, img_size: int):
    """在第三周策略基础上加/减增强，变量只有一个。

    natural  第三周默认（RandAugment + ColorJitter + hflip）
    none     只做确定性 resize —— 用来量化"增强到底贡献了多少"
    robust   叠加 RandomGrayscale + 90°旋转，**模拟注入的采集侧偏移**
    """
    from torchvision import transforms as T

    if kind == "none":
        pol = policy_natural(img_size)
        pol.random_resized_crop = False
        pol.allow_hflip = False
        pol.color_jitter = False
        pol.randaugment = False
        return build_transform("train", pol)

    pol = policy_natural(img_size)
    tfm = build_transform("train", pol)
    if kind == "robust":
        # RandomGrayscale 必须在 ToTensor 之前（它吃 PIL 图），所以插到最前面
        extra = [
            T.RandomGrayscale(p=0.15),
            # EXIF 方向错误的等效模拟：整图 90/180/270 旋转
            T.RandomApply([T.RandomRotation(degrees=[90, 90])], p=0.06),
        ]
        tfm = T.Compose(extra + list(tfm.transforms))
    return tfm


def build_loaders(cfg: dict, workers: int = 2):
    df = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    img = cfg["img"]
    pol = policy_natural(img)

    # 可选：只保留两类，构成平衡子集（用于证伪实验，不是为了上线）。
    # 注意 label 不重映射 —— 输出仍是 5 维，只是 0/1/2 类永远不出现在 loss 里。
    subset = cfg.get("subset_classes")
    if subset is not None:
        df = df[df.label.isin(list(subset))].reset_index(drop=True)
        cnt = df[df.split == "train"].label.value_counts().sort_index()
        print(f"  [subset] 只保留类 {list(subset)}，train 分布 {cnt.to_dict()} "
              f"(比例 {cnt.max()/max(1,cnt.min()):.2f}:1)")

    # 可选：按比例缩减训练集（分层抽样，保持长尾结构）。
    # 用途：当模型太强、错分样本太少时，把工作点推到"看得见错误"的区间。
    frac = cfg.get("train_frac", 1.0)
    if frac < 1.0:
        tr = df[df.split == "train"]
        keep = tr.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=42), include_groups=True)
        df = pd.concat([keep, df[df.split != "train"]], ignore_index=True)

    tfms = {
        "train": build_transform_variant(cfg["aug"], img),
        "val": build_transform("val", pol),
        "test": build_transform("val", pol),
    }
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    # 固定 class_to_idx 为完整 5 类：子集训练时（如 uniform2）模型输出仍是 5 维，
    # 类别编号必须与全局一致，否则 3/4 会被映射成 0/1，指标全错。
    dsets = make_splits(parts, tfms, root=DATA, label_col="label",
                        class_to_idx={str(i): i for i in range(N_CLASSES)},
                        return_path=True, on_error="skip")
    # manifest 里 label 就是 0..4 整数，且顺序与 LABEL_NAMES（字母序）一致。
    # 这里断言"编号连续且从 0 开始"，防止哪天 manifest 变了导致指标和名字错位
    # —— 类别名与编号错位是最难查的一类 bug（指标看起来正常，结论全反）。
    cls = sorted(int(c) for c in dsets["train"].class_to_idx)
    assert cls == list(range(N_CLASSES)), cls
    assert dsets["train"].class_to_idx == dsets["test"].class_to_idx

    mk = lambda d, sh: torch.utils.data.DataLoader(
        d, batch_size=cfg["bs"], shuffle=sh, num_workers=workers,
        pin_memory=True, drop_last=sh, persistent_workers=workers > 0,
    )
    return mk(dsets["train"], True), mk(dsets["val"], False), mk(dsets["test"], False)


@torch.no_grad()
def run_logits(model, loader, device) -> dict:
    """跑一遍并收集 logits / y_true / path。

    存 **logits** 而非 probs：温度缩放需要 logits，而 probs 是 logits 的有损变换。
    """
    model.eval()
    L, Y, Ps = [], [], []
    amp = str(device).startswith("cuda")
    for x, y, path in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(x)
        L.append(out.float().cpu().numpy())
        Y.append(y.numpy())
        Ps.extend(list(path))
    return {"logits": np.concatenate(L), "y_true": np.concatenate(Y), "path": np.asarray(Ps, dtype=object)}


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    ok = tot = 0
    amp = str(device).startswith("cuda")
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            p = model(x).argmax(1)
        ok += int((p.cpu() == y).sum()); tot += y.numel()
    return ok / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--img", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--aug", default=None, choices=("natural", "none", "robust"))
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--weight-mode", default=None, choices=("sqrt", "inverse", "effective"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="输出名后缀，多种子时用 --tag s1")
    ap.add_argument("--no-pretrain", action="store_true", help="从零训练（模拟无预训练权重的领域）")
    ap.add_argument("--train-frac", type=float, default=None, help="分层抽样缩减训练集，如 0.4")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="默认 datasets/pet5/manifest.csv；增量重训时指向带反馈数据的 manifest")
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="热启动权重：增量重训（见 train.retrain）时从此权重继续")
    args = ap.parse_args()

    global MANIFEST
    if args.manifest is not None:
        MANIFEST = args.manifest

    cfg = dict(PRESETS[args.exp])
    for k, v in (("epochs", args.epochs), ("img", args.img), ("bs", args.bs),
                 ("loss", args.loss), ("aug", args.aug), ("gamma", args.gamma),
                 ("weight_mode", args.weight_mode), ("train_frac", args.train_frac)):
        if v is not None:
            cfg[k] = v
    # backbone 必须写进 cfg：导出脚本靠它重建结构，写漏了就只能用默认 resnet50 猜
    cfg["backbone"] = args.backbone
    pretrained = not args.no_pretrain

    name = args.exp + (f"_{args.tag}" if args.tag else "")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print("=" * 78)
    print(f"实验 {name}   cfg={cfg}   seed={args.seed}   device={device}")
    print("=" * 78)

    tr_l, va_l, te_l = build_loaders(cfg, args.workers)
    counts = tr_l.dataset.df["label"].value_counts().sort_index().to_numpy()
    print(f"train 类分布 {dict(zip(LABEL_NAMES, counts.tolist()))}")
    print(f"  不平衡比 max/min = {counts.max()/max(1,counts.min()):.2f}")
    print(f"test  n={len(te_l.dataset)}   val n={len(va_l.dataset)}")

    model, groups = build_model(args.backbone, N_CLASSES, device, pretrained=pretrained)
    if args.init_ckpt is not None:
        # 增量重训：热启动。strict=True —— 结构对不上就直接停，不要"部分加载"然后
        # 用一堆随机初始化的层继续训，那会得到谁也说不清的结果。
        ck = models_load(args.init_ckpt, device)
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"  热启动自 {args.init_ckpt.name}"
              f"  (exp={ck.get('exp')}  seed={ck.get('seed')}  cfg={ck.get('cfg')})")
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit = make_loss(cfg["loss"], counts=counts, weight_mode=cfg.get("weight_mode", "sqrt"),
                     gamma=cfg.get("gamma", 2.0), label_smoothing=0.1, device=device)
    print(f"loss = {crit}")

    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rows, best_val, t0 = [], 0.0, time.time()

    for ep in range(1, cfg["epochs"] + 1):
        model.train(); freeze_bn(model)
        tot, seen, ok = 0.0, 0, 0
        for x, y, _ in tr_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * y.numel(); seen += y.numel()
            ok += int((out.argmax(1) == y).sum())
        sch.step()
        va = accuracy(model, va_l, device)
        best_val = max(best_val, va)
        rows.append({"epoch": ep, "train_loss": tot / max(1, seen), "train_acc": ok / max(1, seen),
                     "val_acc": va})
        print(f"  ep{ep:02d} loss {tot/max(1,seen):.4f}  train {100*ok/max(1,seen):5.2f}%  "
              f"val {100*va:5.2f}%   ({time.time()-t0:.0f}s)")

    te = accuracy(model, te_l, device)
    print(f"  --> best val {best_val*100:.2f}%  test {te*100:.2f}%")

    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "classes": LABEL_NAMES,
                "seed": args.seed, "exp": args.exp}, CKPT / f"{name}.pth")

    res = {"val": run_logits(model, va_l, device), "test": run_logits(model, te_l, device)}
    np.savez_compressed(
        OUT / f"preds_{name}.npz",
        val_logits=res["val"]["logits"], val_y=res["val"]["y_true"], val_path=res["val"]["path"],
        test_logits=res["test"]["logits"], test_y=res["test"]["y_true"], test_path=res["test"]["path"],
        classes=np.array(LABEL_NAMES, dtype=object),
    )
    pd.DataFrame(rows).to_csv(OUT / f"history_{name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "exp": args.exp, "name": name, "seed": args.seed, "cfg": cfg,
        "best_val_acc": round(best_val, 4), "test_acc": round(te, 4),
        "minutes": round((time.time() - t0) / 60, 2),
    }
    (OUT / f"summary_{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] {name}  test_acc={te:.4f}  用时 {summary['minutes']} min")


if __name__ == "__main__":
    main()
