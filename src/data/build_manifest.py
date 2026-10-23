"""
第 3 周的血统，第 8 周收编：构造一个"像真实世界一样脏、一样有实体结构"的数据集

为什么必须自己造一个
--------------------
标准数据集（Oxford Pet / CIFAR / ImageNet）都是**已经按图片独立采样好**的，
你没法在上面演示"同一个体跨集泄漏"，因为它根本没有"个体"这个概念。

所以这里基于第二周已有的 1000 张 pet5 图，构造一个**有实体层级**的采集场景：

    plate_id (设备/孔板，10 个)
        └── subject_id (个体/一只狗，1000 个)
                └── frame  (连拍帧，1~5 帧)

并刻意注入四类真实脏数据：
    1. 损坏文件        截断 JPEG + 0 字节 + 假图（文本伪装成 .jpg）
    2. 单通道图        灰度 L 模式
    3. EXIF 旋转       像素存的是横躺的，方向写在 EXIF Orientation=6
    4. 类别极不平衡    各类帧数 5:4:3:2:1（稀有类采集次数少）

关键设计：plate 与 label 之间存在**软相关**（每个类别偏好 2 块板）
-------------------------------------------------------------
这是现实中最常见的泄漏形态：A 医院的重症病人多，B 医院轻症多。
于是"背景/设备特征"和标签之间产生了虚假相关 —— 模型可以完全不看病灶，
只靠"这张图是哪个设备拍的"就能猜对大半。

这让我们能做出一个三档对照：
    按图片随机划分  -> 个体 + 设备 双重泄漏         -> val_acc 最虚高
    按 subject 划分 -> 个体泄漏堵住，设备仍跨集      -> 仍然偏高（粒度不够细！）
    按 plate 划分   -> 正确的粒度                    -> 诚实的性能

输出
----
    data/messy/manifest.csv   列: path,label,class_name,subject_id,plate_id,date,frame
    data/messy/images/...     实际图片
    outputs/dirty_injection.json  注入清单（用于验证清洗模块有没有抓全）

用法：
    python scripts/build_messy.py
"""

from __future__ import annotations

import json
import os
import random

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import DATASETS, LABEL_NAMES, OUTPUTS, ensure_dirs

ensure_dirs()

SRC = Path(os.getenv("DOGBREED_PET5_SRC", DATASETS / "pet5_raw"))   # 由 download_pets.py 生成
OUT = DATASETS / "pet5"
IMG_DIR = OUT / "images"

CLASSES = LABEL_NAMES
# 每个类别每个个体的连拍帧数 —— 制造 5:4:3:2:1 的类别不平衡
FRAMES_PER_CLASS = [5, 4, 3, 2, 1]
N_PLATES = 20
HOME_WEIGHT = 4.0        # 每个类别对"专属板"的偏好倍率
SEED = 42

# 脏数据注入比例
P_CORRUPT = 0.020
P_GRAY = 0.030
P_EXIF = 0.045

rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)


# ==============================================================================
# 设备（孔板）特征：每块板有自己固定的"成像签名"
#   真实来源：镜头暗角、光源不均匀、白平衡漂移、传感器噪声模式
# ==============================================================================
def make_plate_signatures(n_plates: int) -> list:
    sigs = []
    for i in range(n_plates):
        sigs.append(
            {
                "angle": np_rng.uniform(0, np.pi),
                "freq": np_rng.uniform(0.8, 2.0),
                "phase": np_rng.uniform(0, 1),
                "illum_amp": np_rng.uniform(0.10, 0.18),     # 光照不均匀幅度
                "cast": np_rng.uniform(0.90, 1.10, size=3),  # 白平衡漂移
                "vignette": np_rng.uniform(0.15, 0.35),      # 暗角强度
                "noise": np_rng.uniform(1.5, 4.0),           # 传感器噪声 sigma
            }
        )
    return sigs


def plate_multiplier(w: int, h: int, sig: dict) -> np.ndarray:
    """生成 (h,w,3) 的逐像素乘子：光照梯度 × 白平衡 × 暗角。"""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    proj = xx * np.cos(sig["angle"]) + yy * np.sin(sig["angle"])
    illum = 1.0 + sig["illum_amp"] * np.sin(2 * np.pi * (sig["freq"] * proj + sig["phase"]))
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.7071
    vig = 1.0 - sig["vignette"] * (r ** 2)
    return (illum[..., None] * sig["cast"][None, None, :] * vig[..., None]).astype(np.float32)


def apply_plate(img: Image.Image, sig: dict) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    m = plate_multiplier(img.width, img.height, sig)
    a = np.clip(a * m, 0, 255)
    if sig["noise"] > 0:
        a = a + np_rng.normal(0, sig["noise"], size=a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make_frame(src: Image.Image, sig: dict) -> Image.Image:
    """同一只狗的"下一帧"：轻微位移 / 缩放 / 曝光变化 / 偶发运动模糊。"""
    w, h = src.size
    # 轻微仿射：平移 ±3%，缩放 0.96~1.04，旋转 ±2.5°
    dx = np_rng.uniform(-0.03, 0.03) * w
    dy = np_rng.uniform(-0.03, 0.03) * h
    scale = np_rng.uniform(0.96, 1.04)
    ang = np_rng.uniform(-2.5, 2.5)
    im = src.transform(
        (w, h),
        Image.AFFINE,
        (scale * np.cos(np.deg2rad(ang)), scale * np.sin(np.deg2rad(ang)), dx,
         -scale * np.sin(np.deg2rad(ang)), scale * np.cos(np.deg2rad(ang)), dy),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )
    # 曝光漂移
    im = Image.fromarray(np.clip(np.asarray(im).astype(np.float32) * np_rng.uniform(0.88, 1.12), 0, 255).astype(np.uint8))
    # 10% 概率运动模糊
    if np_rng.random() < 0.10:
        im = im.filter(ImageFilter.GaussianBlur(radius=np_rng.uniform(0.6, 1.6)))
    return im


# ==============================================================================
# 脏数据注入
# ==============================================================================
def inject_corrupt(path: Path, kind: str) -> None:
    data = path.read_bytes()
    if kind == "truncate":
        path.write_bytes(data[: max(1, int(len(data) * np_rng.uniform(0.15, 0.45)))])
    elif kind == "zero":
        path.write_bytes(b"")
    elif kind == "fake":
        path.write_bytes(b"<html><head><title>403 Forbidden</title></head><body>CDN error</body></html>")


def inject_exif(path: Path) -> None:
    """把像素旋转 90° 存盘，同时在 EXIF 里写 Orientation=8。

    语义对齐（这是最容易踩的坑）：
        像素 = 原图顺时针转 90°（左边缘被转到了顶部）  ->  0th row = 视觉左侧
        Orientation=8 的定义正是"0th row 是视觉左侧，显示时需逆时针转 90°"
    -> 不做 exif_transpose 的读取器（cv2.imread、朴素 PIL.open）会看到躺着的图
    -> 正确处理后会看到正立的图
    （如果这里写成 Orientation=6，exif_transpose 会把它转成倒立的——差 180°）
    """
    img = Image.open(path).convert("RGB")
    img = img.rotate(-90, expand=True)          # 像素存成"横躺"（顺时针 90°）
    exif = img.getexif()
    exif[274] = 8                                # Orientation = 8: 显示时逆时针转 90°
    img.save(path, quality=90, exif=exif)


def inject_grayscale(path: Path) -> None:
    img = Image.open(path).convert("L")
    img.save(path, quality=90)


# ==============================================================================
def main():
    # 注意：不做整体 rmtree（3000+ 文件的批量删除需要额外确认，也容易误删）。
    # 生成是确定性的：同 seed 下文件名集合固定，直接原地覆盖即可。
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS).mkdir(exist_ok=True)

    plate_sigs = make_plate_signatures(N_PLATES)

    # ---- 收集源图：每个类别 200 张（pet5 的 train+val 合并）----
    src_by_cls = {c: [] for c in CLASSES}
    for split in ["train", "val"]:
        for c in CLASSES:
            d = SRC / split / c
            if d.exists():
                src_by_cls[c] += sorted(d.glob("*.jpg"))
    for c in CLASSES:
        print(f"[src] {c:<22} {len(src_by_cls[c])} 张")
        if not src_by_cls[c]:
            raise SystemExit(f"找不到源图: {c}")

    rows = []
    dirty_log = {"corrupt": [], "grayscale": [], "exif": []}
    sid = 0

    for ci, cname in enumerate(CLASSES):
        home = [(4 * ci) % N_PLATES, (4 * ci + 1) % N_PLATES]   # 该类别偏好的 2 块板（错开分布）
        n_frames = FRAMES_PER_CLASS[ci]
        for src_path in tqdm(src_by_cls[cname], desc=f"{cname}", ncols=80):
            # --- 设备分配：与类别软相关（偏好板权重 4 倍）---
            weights = np.array([HOME_WEIGHT if p in home else 1.0 for p in range(N_PLATES)])
            weights = weights / weights.sum()
            plate = int(np_rng.choice(N_PLATES, p=weights))
            date = f"2026-{np_rng.integers(1, 6):02d}-{np_rng.integers(1, 28):02d}"

            subject_id = f"S{sid:04d}"
            sid += 1
            try:
                base = Image.open(src_path).convert("RGB").resize((384, 384), Image.BILINEAR)
            except Exception:
                continue

            for f in range(n_frames):
                frame = make_frame(base, plate_sigs[plate])
                frame = apply_plate(frame, plate_sigs[plate])
                rel = f"images/{cname}/{subject_id}_f{f}.jpg"
                out_p = OUT / rel
                out_p.parent.mkdir(parents=True, exist_ok=True)
                frame.save(out_p, quality=88)
                rows.append(
                    {
                        "path": rel,
                        "label": ci,
                        "class_name": cname,
                        "subject_id": subject_id,
                        "plate_id": f"P{plate:02d}",
                        "date": str(date),
                        "frame": f,
                    }
                )

    df_rows = rows
    print(f"\n[生成] 共 {len(df_rows)} 张图，{sid} 个个体，{N_PLATES} 块板")

    # ---- 注入脏数据 ----
    n = len(df_rows)
    idxs = list(range(n))
    np_rng.shuffle(idxs)
    k_corrupt = int(n * P_CORRUPT)
    k_gray = int(n * P_GRAY)
    k_exif = int(n * P_EXIF)

    cur = 0
    for i in idxs[cur : cur + k_corrupt]:
        kind = ["truncate", "zero", "fake"][i % 3]
        p = OUT / df_rows[i]["path"]
        inject_corrupt(p, kind)
        dirty_log["corrupt"].append({"path": df_rows[i]["path"], "kind": kind})
    cur += k_corrupt
    for i in idxs[cur : cur + k_gray]:
        inject_grayscale(OUT / df_rows[i]["path"])
        dirty_log["grayscale"].append({"path": df_rows[i]["path"]})
    cur += k_gray
    for i in idxs[cur : cur + k_exif]:
        inject_exif(OUT / df_rows[i]["path"])
        dirty_log["exif"].append({"path": df_rows[i]["path"]})
    cur += k_exif

    print(f"[脏数据] 损坏 {len(dirty_log['corrupt'])} / 灰度 {len(dirty_log['grayscale'])} / EXIF {len(dirty_log['exif'])}")

    import pandas as pd

    df = pd.DataFrame(df_rows)
    df.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    (OUTPUTS / "dirty_injection.json").write_text(
        json.dumps(dirty_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 统计 ----
    print("\n[类别分布]（注意 5:4:3:2:1 的不平衡）")
    for ci, cname in enumerate(CLASSES):
        sub = df[df["label"] == ci]
        print(f"  {cname:<22} {len(sub):>5d} 张  {sub['subject_id'].nunique():>4d} 个体")
    print(f"  不平衡比 = {df['label'].value_counts().max() / df['label'].value_counts().min():.2f}")

    print("\n[plate x label 分布]（看得出 plate 与 label 的软相关）")
    ct = pd.crosstab(df["plate_id"], df["label"])
    print(ct.to_string())
    print("\n  每块板的主导类别占比（随机基线应为 20%）：")
    for p, row in ct.iterrows():
        print(f"    {p}: 主导类 = class{int(row.idxmax())}，占比 {100*row.max()/row.sum():.1f}%")

    print(f"\n[完成] manifest -> {OUT/'manifest.csv'}")


if __name__ == "__main__":
    main()
