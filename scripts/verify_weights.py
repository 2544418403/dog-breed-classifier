"""
校验仓库里那一份 weights/*.onnx 的真实精度 —— 让 README 里的数字可以对得上。

==============================================================================
为什么要有这个脚本
==============================================================================
README 里写着"INT8 模型 test top-1 = 0.890"。这个数字必须能被验证，
否则它只是一句营销话术。有了这个脚本，任何人 clone 下来、构造完数据集之后：

    python scripts/verify_weights.py --manifest datasets/pet5/manifest.csv \\
                                     --root datasets/pet5

就能拿到自己机器上的数字。若和 README 差超过 1 个百分点，说明环境或
预处理有漂移 —— 这正是部署事故最常见的来源（详见 common/preprocess.py 的注释）。

==============================================================================
它同时回答一个工程问题
==============================================================================
FP32 -> INT8 量化到底牺牲了多少精度？这里对同一个 ckpt 的两个 ONNX 行为一张表：
table(gap)。< 0.5pp 通常可以接受；如果 gap 很大，优先怀疑**校准集分布**，
而不是量化算法本身（详见 models/export.py 里 --calib 的说明）。

用法：
    python scripts/verify_weights.py                         # 只测默认的 INT8
    python scripts/verify_weights.py --weights weights/a.onnx weights/b.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402
import onnxruntime as ort                             # noqa: E402
from PIL import Image                                 # noqa: E402

from common import DATASETS, WEIGHTS                  # noqa: E402
from common.preprocess import preprocess_pil          # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 64
RESIZE_TO = 73

CN = {
    "havanese": "哈瓦那犬",
    "miniature_pinscher": "迷你品捷犬",
    "pomeranian": "博美犬",
    "shiba_inu": "柴犬",
    "yorkshire_terrier": "约克夏梗",
}


def load_rows(manifest: Path, split: str):
    # utf-8-sig 不是随手写的：manifest 来自 pandas.to_csv，Windows 下带 BOM，
    # 用 utf-8 读会让第一列名变成 "\ufeffpath"，然后 KeyError('path')。
    # （pandas 会自动吃掉 BOM，所以这个坑只在 csv 模块里才会咬人。）
    with manifest.open(encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("split") == split]


def run(model_path: Path, rows: list[dict], root: Path, batch: int = 32) -> dict:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    # 模型自己声明的类别数和当前 img size，比写死更可靠
    n_cls = int(out.shape[-1]) if isinstance(out.shape[-1], int) else 5

    y_true, y_pred, lat = [], [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        xs, ys = [], []
        for r in chunk:
            p = root / r["path"]
            img = Image.open(p)
            xs.append(preprocess_pil(img, IMG_SIZE, RESIZE_TO, MEAN, STD))
            ys.append(int(r["label"]))
        X = np.stack(xs, axis=0)
        t0 = time.perf_counter()
        logits = sess.run([out.name], {inp.name: X})[0]
        lat.append((time.perf_counter() - t0) / len(chunk) * 1000)
        y_pred.extend(np.argmax(logits, axis=1).tolist())
        y_true.extend(ys)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred[: len(y_true)])
    acc = float((y_true == y_pred).mean())

    # 每类召回 + 混淆里最严重的一对
    per_class = {}
    for c in range(n_cls):
        mask = y_true == c
        per_class[c] = float((y_pred[mask] == c).mean()) if mask.any() else 0.0

    return {
        "model": model_path.name,
        "variant": "int8" if "int8" in model_path.name else "fp32",
        "size_mb": round(model_path.stat().st_size / 1048576, 2),
        "n": len(y_true),
        "accuracy": round(acc, 4),
        "per_class_recall": per_class,
        "mean_per_img_ms": round(float(np.mean(lat)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 ONNX 权重在 held-out test 集上的精度")
    ap.add_argument("--manifest", default=str(DATASETS / "pet5" / "manifest.csv"))
    ap.add_argument("--root", default=str(DATASETS / "pet5"),
                    help="manifest 里 path 列的相对根目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", nargs="+", default=[str(WEIGHTS / "dogbreed_resnet50_64_int8.onnx")])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--json", help="把结果写到 JSON 文件")
    args = ap.parse_args()

    manifest, root = Path(args.manifest), Path(args.root)
    if not manifest.exists():
        print(f"[!] 找不到 {manifest}", file=sys.stderr)
        print("    这个脚本需要数据集。构造方式见 README「完整复现」一节：", file=sys.stderr)
        print("      python -m data.download_pets && python -m data.build_manifest",
              file=sys.stderr)
        print("    只想验证模型能跑？用 scripts/smoke.py。", file=sys.stderr)
        return 2

    rows = load_rows(manifest, args.split)
    if not rows:
        print(f"[!] manifest 里没有 split={args.split} 的行", file=sys.stderr)
        return 2
    print(f"[数据] {manifest.name}  split={args.split}  n={len(rows)}  root={root}\n")

    results = []
    for w in args.weights:
        p = Path(w)
        if not p.exists():
            print(f"[!] 跳过不存在的权重：{p}", file=sys.stderr)
            continue
        r = run(p, rows, root, args.batch)
        results.append(r)
        print(f"{r['model']}  [{r['variant']}]  {r['size_mb']}MB")
        print(f"  test top-1      {r['accuracy']:.4f}   (n={r['n']})")
        print(f"  推理单张        {r['mean_per_img_ms']:.2f} ms")
        print("  每类召回：")
        for c, v in sorted(r["per_class_recall"].items(), key=lambda kv: kv[1]):
            name = list(CN)[c] if c < len(CN) else str(c)
            bar = "#" * int(v * 40)
            print(f"    {CN.get(name, name):<8} {v:.3f}  {bar}")
        print()

    if len(results) == 2:
        a, b = results
        gap = (a["accuracy"] - b["accuracy"]) * 100
        print(f"[量化 gap] {a['variant']} - {b['variant']} = {gap:+.2f} pp"
              f"    (体积 {a['size_mb']}MB vs {b['size_mb']}MB)")

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n[写出] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
