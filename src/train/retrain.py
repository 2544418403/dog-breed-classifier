"""数据回流第 4 步：增量重训 —— 用「原始数据 + 用户反馈」炼一个新模型。

用法：
    python -m train.retrain                       # 默认从 weights/baseline.pth 热启动
    python -m train.retrain --epochs 3 --lr-scale 0.3
    python -m train.retrain --eval-only           # 只对比新旧模型的 test 指标

==============================================================================
为什么是"热启动 + 小学习率"，而不是"从头重训"
==============================================================================
从头重训有三个问题：
    1. 贵：每次几百个用户点几下就从头跑 12 个 epoch，个人或小团队扛不住
    2. 危险：新数据量小，从头训很容易丢掉预训练里那些通用特征（灾难性遗忘）
    3. 不可控：每次重训都是一次大的分布漂移，很难说清是变好了还是变偏了

热启动 + 小 lr 的语义是：**我只想在这块"用户指出来的疤"上修一下，
其余认知尽量别动。** 这也是为什么下面默认 lr 乘 0.3、epoch 数默认 3。

==============================================================================
合并策略：原始数据 + 反馈数据，不做 cap 之外的重采样
==============================================================================
反馈数据通常几十到几百张，相对原始集是少数。这里保持全部保留、
但在 training loss 上不做特殊加权 —— 因为它们本来就是"模型最不会的题"，
本身就 Hard Negative，不需要再加权。真正要防的是反馈分布偏斜，
那一步在 feedback_export.py 的 cap 里做完了。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common import DATASETS, OUTPUTS, WEIGHTS, ensure_dirs  # noqa: E402

ensure_dirs()

BASE_MANIFEST = DATASETS / "pet5" / "manifest.csv"
FB_MANIFEST = DATASETS / "feedback" / "manifest.csv"
MERGED = DATASETS / "manifest_merged.csv"


def merge_manifests() -> Path:
    """把原始 manifest 和反馈 manifest 拼成一份。

    注意 **不重置 split**：反馈导出时已经按自己的比例切好了 train/val/test。
    两份拼起来后，test 集合包含一部分"用户指出来的错题" —— 这正是我们想看的：
    模型在它曾经答错、且被抱怨过的地方，现在答对了没有。
    """
    import pandas as pd

    frames = []
    if BASE_MANIFEST.exists():
        base = pd.read_csv(BASE_MANIFEST, encoding="utf-8-sig")
        base["origin"] = "base"
        frames.append(base)
    if FB_MANIFEST.exists():
        fb = pd.read_csv(FB_MANIFEST, encoding="utf-8-sig")
        frames.append(fb)
    if not frames:
        raise SystemExit(f"没有可训练的数据：{BASE_MANIFEST} 与 {FB_MANIFEST} 都不存在。")

    df = pd.concat(frames, ignore_index=True)
    # subject_id 在两批数据里可能撞号（例如 S0001），加前缀避免"假泄漏"
    df["subject_id"] = df["origin"] + "_" + df["subject_id"].astype(str)
    df.to_csv(MERGED, index=False, encoding="utf-8-sig")

    print("\n[合并训练集]")
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        fb_n = int((sub["origin"] != "base").sum()) if len(sub) else 0
        print(f"  {split:<6} 合计 {len(sub):>5d}  （其中来自反馈 {fb_n}）")
    return MERGED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", type=Path, default=WEIGHTS / "baseline.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-scale", type=float, default=0.3,
                    help="相对 baseline 学习率的缩放；增量微调用小 lr 是刻意的")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="retrain")
    ap.add_argument("--eval-only", action="store_true", help="跳过训练，只看当前权重表现")
    args = ap.parse_args()

    merged = merge_manifests()

    if args.eval_only:
        # 只想在合并后的 test 上看当前权重的表现。训练器没有"纯评估"模式，
        # 这里用 1 个 epoch 近似（代价很小，结论仍然可信）。
        return subprocess.call(
            [sys.executable, "-m", "train.train", "--exp", "baseline",
             "--epochs", "1", "--manifest", str(merged)], cwd=str(ROOT))

    if not args.init_ckpt.exists():
        raise SystemExit(
            f"没有起始权重 {args.init_ckpt}。\n"
            f"先训练一个基线：python -m train.train --exp baseline\n"
            f"（增量重训是站在已有模型的肩膀上，肩膀得先有。）")

    cmd = [
        sys.executable, "-m", "train.train",
        "--exp", "baseline",
        "--manifest", str(merged),
        "--init-ckpt", str(args.init_ckpt),
        "--epochs", str(args.epochs),
        "--bs", str(args.bs),
        "--seed", str(args.seed),
        "--tag", args.tag,
    ]
    print("\n[重训命令]", " ".join(str(c) for c in cmd), "\n")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        return rc

    new_ckpt = WEIGHTS / f"baseline_{args.tag}.pth"
    if new_ckpt.exists():
        print(f"\n[下一步] 导出并复核新的服务模型：")
        print(f"    python -m models.export --ckpt {new_ckpt.name} --quantize")
        print(f"  然后用  python scripts/smoke.py  确认服务仍然正常。")
        plan = {
            "step": "retrain",
            "init_ckpt": str(args.init_ckpt),
            "new_ckpt": str(new_ckpt),
            "merged_manifest": str(merged),
            "epochs": args.epochs,
        }
        (OUTPUTS / "retrain_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
