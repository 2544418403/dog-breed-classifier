"""
从 Oxford-IIIT Pet 官方数据集抽取 5 个易混淆小型犬品种 -> datasets/pet5_raw/

==============================================================================
这条路是为"想自己重训一遍"的人准备的
==============================================================================
只想跑推理服务的话**不需要跑这个脚本** —— 仓库里已经带训好的 INT8 权重
（weights/dogbreed_resnet50_64_int8.onnx，22.7MB）。

==============================================================================
数据来源（两个源都要，各取所需）
==============================================================================
1. parquet（图像本体）
     https://www.modelscope.cn/api/v1/datasets/timm/oxford-iiit-pet/repo
        ?Revision=master&FilePath=data/train-00000-of-00001.parquet
     选 ModelScope 而不是原始 Kaggle 815MB 的 zip，原因很实在：
     国内直连能跑满带宽，而 Kaggle 那个包我在第 2 周写了三个版本的断点续传
     脚本（dl_pet.sh / dl_pet_par.sh / download_kaggle.sh）才下完。
     parquet 只有 ~800MB 且带标签列，省掉一次解压和一次标注对齐。

2. list.txt（品种名 -> label 编号的映射表）
     https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
     parquet 里的 label 是 0~36 的整数，没有品种名。品种名和编号的对应关系
     只在官方 annotations 包里，必须取。

==============================================================================
为什么是这 5 个品种
==============================================================================
刻意挑了一批毛色体型都接近的小型犬。选"哈士奇 vs 柯基"那种差距明显的数据集，
ImageNet 预训练特征一检就开，阶段 A（冻结骨干）就 98% 了，后面几周的实验
（解冻策略、BN 冻结、损失加权、量化损失压测）全部失去区分度 —— 曲线没故事。
现在的组合让"不用 classifier 也能看懂模型在干什么"这件事成立。

用法：
    python -m data.download_pets            # 下载并抽帧 -> datasets/pet5_raw/
"""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image
from pyarrow import parquet as pq

from common import DATASETS, ensure_dirs

ensure_dirs()

RAW = DATASETS / "_raw"                     # 下载的原始文件放这里（不进 git）
PARQUET_DIR = RAW / "parquet"
ANN_DIR = RAW / "annotations"
OUT = DATASETS / "pet5_raw"                 # 产物：给 build_manifest.py 吃的干净池


MS_BASE = ("https://www.modelscope.cn/api/v1/datasets/timm/oxford-iiit-pet/repo"
           "?Revision=master&FilePath=")
PARQUET_FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]
ANN_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# 5 个易混淆的小型品种（毛色/体型相近，靠五官和花纹区分）
CLASSES = [
    "havanese",            # 哈瓦那犬
    "miniature_pinscher",  # 迷你杜宾
    "pomeranian",          # 博美
    "shiba_inu",           # 柴犬
    "yorkshire_terrier",   # 约克夏
]
VAL_RATIO = 0.2
SEED = 42

CHUNK = 1 << 20


def _download(url: str, dst: Path, desc: str) -> None:
    """带进度条的下载。目标文件已存在且非空就跳过（幂等，方便重跑）。"""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [skip] {desc} 已存在（{dst.stat().st_size / 1048576:.1f} MB）")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  [get ] {desc}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dog-breed-classifier/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done / total * 100
                    print(f"\r        {done / 1048576:7.1f} / {total / 1048576:7.1f} MB"
                          f"  ({pct:5.1f}%)", end="", flush=True)
            print()
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"下载失败：{desc}\n  {e}\n"
            f"  URL: {url}\n"
            "  断网环境可以手动下载后放到该路径再重跑本脚本。"
        )
    tmp.replace(dst)


def _extract_list_txt(tar_gz: Path) -> Path:
    """annotations.tar.gz -> 取出 annotations/list.txt。只取这一个文件，不全解包。"""
    list_txt = ANN_DIR / "list.txt"
    if list_txt.exists():
        return list_txt
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith("list.txt")), None)
        if member is None:
            raise SystemExit(f"{tar_gz} 里找不到 list.txt")
        src = tf.extractfile(member)
        if src is None:
            raise SystemExit(f"{tar_gz} 中 list.txt 读取失败")
        list_txt.write_bytes(src.read())
    print(f"  [ok  ] 解析出 {list_txt.name}")
    return list_txt


def build_name_to_label(list_txt: Path) -> dict:
    """品种名 -> label(0-indexed)。"""
    name_to_cid: dict[str, int] = {}
    for line in list_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        img_name, cid = parts[0], int(parts[1])
        breed = img_name.rsplit("_", 1)[0]
        name_to_cid.setdefault(breed, cid)
    return name_to_cid


def main() -> None:
    print("=" * 68)
    print("步骤 1/3  下载 parquet 图像包 + 官方标注")
    print("=" * 68)
    parquets = []
    for rel in PARQUET_FILES:
        dst = PARQUET_DIR / Path(rel).name
        _download(MS_BASE + rel, dst, f"parquet/{dst.name}")
        parquets.append(dst)
    ann_tar = RAW / "annotations.tar.gz"
    _download(ANN_URL, ann_tar, "annotations.tar.gz")
    list_txt = _extract_list_txt(ann_tar)

    print("\n" + "=" * 68)
    print("步骤 2/3  挑出 5 个品种")
    print("=" * 68)
    name_to_cid = build_name_to_label(list_txt)
    missing = [c for c in CLASSES if c not in name_to_cid]
    if missing:
        raise SystemExit(f"list.txt 中找不到这些品种: {missing}")

    cls_to_label = {c: name_to_cid[c] - 1 for c in CLASSES}   # parquet 里是 0-indexed
    label_to_cls = {v: k for k, v in cls_to_label.items()}
    print("  品种 -> label:", cls_to_label)

    frames = []
    for p in parquets:
        t = pq.read_table(p, columns=["image", "label", "image_id"]).to_pandas()
        print(f"  {p.name}: {len(t)} 行")
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)

    sub = df[df["label"].isin(cls_to_label.values())].copy()
    sub["breed"] = sub["label"].map(label_to_cls)
    print(f"  目标 5 类合计 {len(sub)} 张")
    for c in CLASSES:
        print(f"      {c:<22} {(sub['breed'] == c).sum():>4d} 张")

    if not 300 <= len(sub) <= 2000:
        raise SystemExit(f"图片总数 {len(sub)} 不在预期的 300~2000 区间")

    print("\n" + "=" * 68)
    print(f"步骤 3/3  落盘到 {OUT}（分层 train/val）")
    print("=" * 68)
    tr, va = [], []
    for c in CLASSES:
        d = sub[sub["breed"] == c].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n_val = max(1, round(len(d) * VAL_RATIO))
        va.append(d.iloc[:n_val])
        tr.append(d.iloc[n_val:])

    if OUT.exists():
        shutil.rmtree(OUT)

    for split_name, part in [("train", pd.concat(tr, ignore_index=True)),
                             ("val", pd.concat(va, ignore_index=True))]:
        out_dir = OUT / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(part.itertuples(index=False)):
            blob = row.image
            if isinstance(blob, dict):
                blob = blob.get("bytes") or blob.get("path")
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            img.save(out_dir / f"{row.breed}_{i:04d}.jpg", quality=95)
        print(f"  {split_name:<6} {len(part):>4d} 张 -> {out_dir}")

    print(f"\n[完成] 干净数据池就绪：{OUT}")
    print("  下一步：python -m data.build_manifest   （制造 messy 分布 + 分组划分）")


if __name__ == "__main__":
    sys.exit(main())
