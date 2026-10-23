"""
scripts/00_prepare_data.py —— 把评测数据预处理一次，缓存成 numpy

==============================================================================
为什么要单独做这一步？因为"公平对比"是全周的生命线
==============================================================================
本周要对比 5 种推理形态的**延迟**。但一张图从磁盘走到模型输出要经过：
    解码 JPEG → Resize → CenterCrop → 归一化 → 模型 forward
前面四步（预处理）对所有形态是**完全一样、且相当耗时**的。

如果每个引擎都"从读图片开始计时"，那么：
  - 预处理噪声会把 5~10ms 量级的模型差异彻底淹没
  - 磁盘 IO / 页缓存命中的抖动会让结果不可复现

正确做法是**控制变量**：预处理只做一次并把结果冻成 npy，
之后每个引擎吃的都是同一块内存里的同一个 tensor。
这样测出来的差异，才 100% 是"推理引擎本身"的差异。

（真实服务当然要连预处理一起算，那个数字第五周已经测过了：
  c1 并发 P50 12.22ms，包含了 HTTP + 解码 + 预处理 + 推理。）

==============================================================================
预处理必须与训练期完全一致（第一周的教训）
==============================================================================
Resize(73) → CenterCrop(64) → ToTensor → Normalize(ImageNet mean/std)
这条链路取自第三周 datapipe.transforms.policy_natural(img_size=64)，
第五周 app/model.py 的在线预处理也是同一套，三者必须一致，
否则量出来的准确率是在骗自己。

==============================================================================
用法
==============================================================================
    python scripts/00_prepare_data.py

产物：
    outputs/test_x.npy        (546, 3, 64, 64) float32
    outputs/test_y.npy        (546,)            int64
    outputs/calib_train.npy   (200, 3, 64, 64) float32   ← 静态量化校准用
    outputs/data_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
WEEK3 = ROOT.parent / "第三周"
MANIFEST = WEEK3 / "data" / "messy" / "manifest_with_split.csv"
IMG_ROOT = WEEK3 / "data" / "messy"
OUT = ROOT / "outputs"

LABEL_NAMES = ["havanese", "miniature_pinscher", "pomeranian", "shiba_inu", "yorkshire_terrier"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_val_transform(img_size: int = 64, resize_to: int | None = None):
    """确定性预处理：Resize → CenterCrop → ToTensor → Normalize。

    注意 interpolation 用 BILINEAR 默认的即可，和第三周一致。
    这里刻意【不使用】任何随机增强——val/test 一旦引入随机，
    指标就会随 seed 抖动，量化造成的 0.2% 差异会被彻底盖住。
    """
    r = resize_to if resize_to else max(64, int(img_size * 1.15))
    return T.Compose([
        T.Resize(r, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
    ])


def run_split(df: pd.DataFrame, tfm, desc: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    bad = 0
    for i, row in enumerate(df.itertuples(index=False)):
        p = IMG_ROOT / row.path
        try:
            with Image.open(p) as im:
                if im.mode != "RGB":
                    im = im.convert("RGB")
                xs.append(tfm(im).numpy())
            ys.append(int(row.label))
        except Exception as e:
            bad += 1
            print(f"  [skip] {row.path}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {desc}: {i + 1}/{len(df)}")
    if bad:
        print(f"  [warn] {desc} 跳过 {bad} 张坏图")
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tfm = build_val_transform(args.img)

    print("=" * 78)
    print("第七周 · 00 准备评测数据")
    print("=" * 78)

    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    # 只保留第三周清洗后判定为可用的样本（本项目 manifest 里 ok 全为 True）
    m = m[m["ok"] == True]  # noqa: E712

    test_df = m[m.split == "test"].reset_index(drop=True)
    train_df = m[m.split == "train"].reset_index(drop=True)

    print(f"\n[1/3] 预处理 test split：{len(test_df)} 张")
    tx, ty = run_split(test_df, tfm, "test")
    np.save(OUT / "test_x.npy", tx)
    np.save(OUT / "test_y.npy", ty)
    print(f"  test_x {tuple(tx.shape)}  test_y {tuple(ty.shape)}  classes={sorted(set(ty.tolist()))}")

    print(f"\n[2/3] 预处理 train split 的前 {args.calib_n} 张（静态量化校准集）")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_df), size=min(args.calib_n, len(train_df)), replace=False)
    calib_df = train_df.iloc[sorted(idx)].reset_index(drop=True)
    cx, _ = run_split(calib_df, tfm, "calib")
    np.save(OUT / "calib_train.npy", cx)
    print(f"  calib_train {tuple(cx.shape)}")

    print("\n[3/3] 写元信息")
    meta = {
        "manifest": str(MANIFEST),
        "img_size": args.img,
        "resize_to": max(64, int(args.img * 1.15)),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "classes": LABEL_NAMES,
        "n_test": int(len(ty)),
        "n_calib": int(len(cx)),
        "test_class_counts": pd.Series(ty).value_counts().sort_index().to_dict(),
    }
    (OUT / "data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "-" * 78)
    print(f"test  {len(ty)} 张 → outputs/test_x.npy")
    print(f"calib {len(cx)} 张 → outputs/calib_train.npy")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
