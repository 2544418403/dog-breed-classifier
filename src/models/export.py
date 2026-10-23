"""训练态 → 服务态：把 .pth 导出成 ONNX，并可选地静态量化成 INT8。

==============================================================================
为什么必须有这一步
==============================================================================
`xxx.pth` 是 torch 状态字典，它只有在三样东西同时存在时才能跑：
    torch(2GB+)  +  timm  +  和训练时一模一样的模型定义代码
线上服务不该背这个包袱：
    镜像体积   torch(CPU) ~200MB，带 CUDA 的 wheel 2.5GB；onnxruntime ~40MB
    冷启动     import torch 2~5s；import onnxruntime ~0.2s
    依赖面     torch 版本 / 驱动 / CUDA runtime 任一漂移，服务就起不来

代价是：导出后必须**对拍**（同一批输入，torch 与 onnxruntime 的 logits 差多少）。
本脚本强制做这件事，不对拍不许写 meta，更不许上线。

==============================================================================
用法
==============================================================================
    python -m models.export --ckpt weights/baseline.pth
    python -m models.export --ckpt weights/baseline.pth --quantize --calib-n 200

产出：
    weights/<name>.onnx               FP32
    weights/<name>_int8.onnx          静态 INT8（--quantize 时）
    weights/model_meta.json           服务启动时读的元信息（类别顺序 / 归一化参数等）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_NAMES,
    WEIGHTS,
    ensure_dirs,
)
from common.preprocess import preprocess_pil
from PIL import Image

ensure_dirs()

# 训练侧用 timm 创建，导出侧也用 timm 创建 —— 保证结构与 key 完全一致。
# （曾经试过"用 torchvision 重建来少一个依赖"，可行，但等于多了一处要人工对齐的假设。）


def build_model(backbone: str, n_classes: int) -> nn.Module:
    import timm

    return timm.create_model(backbone, pretrained=False, num_classes=n_classes)


# ---------------------------------------------------------------------------
# 1) FP32 导出
# ---------------------------------------------------------------------------
def export_fp32(model: nn.Module, out: Path, img_size: int, opset: int) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,      # 把 BN 折叠进 conv，推理更快
        input_names=["input"],
        output_names=["logits"],
        # batch 维必须是动态的：/predict_batch 靠它一次 forward 跑 N 张
        # 写死 1 的话批量接口只能靠循环，batching 的收益全没了
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,                   # 用稳定的 TorchScript 导出器，不依赖 onnxscript
    )


@torch.no_grad()
def cross_check(model: nn.Module, onnx_path: Path, img_size: int, n: int = 8) -> float:
    """torch vs onnxruntime 对拍，返回 max|Δlogit|。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    torch.manual_seed(0)
    x = torch.randn(n, 3, img_size, img_size)     # 真实量级的随机输入，不是全 0
    ref = model(x).numpy()
    got = sess.run(["logits"], {"input": x.numpy()})[0]
    return float(np.abs(ref - got).max())


# ---------------------------------------------------------------------------
# 2) 静态量化
# ---------------------------------------------------------------------------
class _CalibReader:
    """喂给 ORT 量化器的校准数据读取器。

    静态量化的原理：INT8 只有 256 个格子，必须知道每层激活值真实的取值范围
    才能把 float 铺进去。动态量化是"来一个 batch 现场算范围"（慢，且每批都变）；
    静态量化用一批有代表性的图先量出范围，写死进图里 —— 所以更快更准，
    但**校准集必须来自真实分布**。用随机噪声校准 = 精度崩。
    """

    def __init__(self, batches: list[np.ndarray], input_name: str):
        self._it = iter(batches)
        self.input_name = input_name

    def get_next(self) -> dict | None:
        b = next(self._it, None)
        return None if b is None else {self.input_name: b}

    def rewind(self) -> None:
        raise NotImplementedError("单次校准不需要 rewind")


def build_calibration_batches(manifest: Path, data_root: Path, img_size: int,
                              resize_to: int, n_images: int, batch: int = 1,
                              seed: int = 0) -> list[np.ndarray]:
    import pandas as pd

    if not manifest.exists():
        raise FileNotFoundError(
            f"找不到 {manifest}。校准需要真实图片："
            f"先跑 `python -m data.build_pet5` 生成数据集，或去掉 --quantize。")

    df = pd.read_csv(manifest, encoding="utf-8-sig")
    df = df[df["split"] == "train"]
    rows = df.sample(n=min(n_images, len(df)), random_state=seed).to_dict("records")

    items = []
    for r in rows:
        try:
            img = Image.open(data_root / r["path"])
            img.load()
        except Exception:
            continue  # 校准集里混进坏图没关系，跳过即可
        items.append(preprocess_pil(img, img_size, resize_to, IMAGENET_MEAN, IMAGENET_STD))

    if not items:
        raise RuntimeError("校准集为空：一张图都没读成功，检查 manifest 里的 path 列。")
    return [np.stack(items[i:i + batch]) for i in range(0, len(items), batch)]


def quantize_static(fp32_path: Path, out_path: Path, calib_batches: list[np.ndarray]) -> None:
    from onnxruntime import InferenceSession
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    input_name = InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=_CalibReader(calib_batches, input_name),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )


def evaluate_variant(onnx_path: Path, manifest: Path, data_root: Path,
                     img_size: int, resize_to: int, limit: int = 400,
                     seed: int = 0) -> dict | None:
    """在 held-out test 集上粗评一个新导出的模型 —— 导出完立刻知道有没有崩。

    为什么要这一步：量化是在改模型的数值行为。一个没有复核就去上线的量化模型，
    和"直接把权重动一下"没有区别。这里只用几百张图，几秒钟，换来一个可信的答案。
    """
    import onnxruntime as ort
    import pandas as pd

    if not manifest.exists():
        return None

    df = pd.read_csv(manifest, encoding="utf-8-sig")
    df = df[df["split"] == "test"]
    if len(df) == 0:
        return None
    rows = df.sample(n=min(limit, len(df)), random_state=seed).to_dict("records")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ok = 0
    xs, ys = [], []
    for r in rows:
        try:
            img = Image.open(data_root / r["path"])
            img.load()
        except Exception:
            continue
        xs.append(preprocess_pil(img, img_size, resize_to, IMAGENET_MEAN, IMAGENET_STD))
        ys.append(int(r["label"]))
    if not xs:
        return None
    logits = sess.run(["logits"], {name: np.stack(xs)})[0]
    ok = int((logits.argmax(1) == np.asarray(ys)).sum())
    return {"top1": round(ok / len(ys), 4), "n": len(ys)}


def main() -> int:
    ap = argparse.ArgumentParser(description="pth -> onnx(+int8)")
    ap.add_argument("--ckpt", type=Path, default=WEIGHTS / "baseline.pth")
    ap.add_argument("--out", type=Path, default=None, help="默认 weights/<ckpt 名>.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--img", type=int, default=None, help="默认读 ckpt['cfg']['img']")
    ap.add_argument("--quantize", action="store_true", help="额外产出静态 INT8")
    ap.add_argument("--calib-n", type=int, default=200, help="校准图片数")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args()

    from common import DATASETS

    manifest = args.manifest or (DATASETS / "pet5" / "manifest.csv")
    data_root = args.data_root or (DATASETS / "pet5")

    # ---------- 1. 加载权重 ----------
    if not args.ckpt.exists():
        print(f"[!] 找不到权重 {args.ckpt}。先训练：`python -m train.train --exp baseline`")
        return 2
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    classes = ckpt.get("classes", LABEL_NAMES)
    backbone = cfg.get("backbone", "resnet50")
    img_size = args.img or int(cfg.get("img", 64))
    # 训练 val transform：Resize(短边→resize_to) + CenterCrop(img_size)
    resize_to = max(img_size, int(img_size * 1.15))
    if classes != LABEL_NAMES:
        print(f"[warn] ckpt 的 classes={classes}，与本仓库常量不一致 —— 以 ckpt 为准")

    torch.manual_seed(0)
    model = build_model(backbone, len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)   # strict：错一个 key 就炸，别静默
    model.eval()
    print(f"[1/5] 权重已加载 {args.ckpt.name}  backbone={backbone}  img={img_size}")

    # ---------- 2. 导出 FP32 ----------
    out = args.out or WEIGHTS / f"{args.ckpt.stem}.onnx"
    export_fp32(model, out, img_size, args.opset)
    fp32_mb = out.stat().st_size / 1024 / 1024
    print(f"[2/5] ONNX FP32 已导出 {out.name}  ({fp32_mb:.1f} MB, opset={args.opset})")

    # ---------- 3. 对拍 ----------
    try:
        max_abs = cross_check(model, out, img_size)
    except ImportError:
        print("[3/5] 跳过对拍：此环境没有 onnxruntime（导出不该装在服务同一环境里）")
        max_abs = None
    if max_abs is not None:
        ok = max_abs < 2e-4
        print(f"[3/5] 对拍 8 张随机图 max|Δlogit|={max_abs:.3e} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("      对拍不过就不要上线。检查 preprocessing 与 opset。")

    meta = {
        "model_file": out.name,
        "source_ckpt": str(args.ckpt),
        "arch": backbone,
        "classes": classes,
        "img_size": img_size,
        "resize_to": resize_to,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "opset": args.opset,
        "onnx_mb": round(fp32_mb, 2),
        "max_abs_diff": max_abs,
    }

    # ---------- 4. 静态量化 ----------
    int8_mb = None
    if args.quantize:
        batches = build_calibration_batches(
            manifest, data_root, img_size, resize_to, args.calib_n)
        int8_out = out.with_name(out.stem + "_int8.onnx")
        quantize_static(out, int8_out, batches)
        int8_mb = int8_out.stat().st_size / 1024 / 1024
        print(f"[4/5] INT8 静态量化完成 {int8_out.name} "
              f"({fp32_mb:.1f}MB -> {int8_mb:.1f}MB，用 {sum(len(b) for b in batches)} 张校准)")

        rep = evaluate_variant(int8_out, manifest, data_root, img_size, resize_to)
        print(f"      INT8 top-1 = {rep['top1']:.4f} (n={rep['n']})" if rep else "      （跳过复核）")
        meta["int8_model_file"] = int8_out.name
        meta["int8_mb"] = round(int8_mb, 2)
        meta["int8_top1"] = rep["top1"] if rep else None
        meta["calib_images"] = sum(len(b) for b in batches)
    else:
        print("[4/5] 跳过量化（加 --quantize 开启）")

    # ---------- 5. 写 meta ----------
    rep32 = evaluate_variant(out, manifest, data_root, img_size, resize_to)
    if rep32:
        meta["fp32_top1"] = rep32["top1"]
        print(f"[5/5] FP32 top-1 = {rep32['top1']:.4f} (n={rep32['n']})")
    meta_path = out.parent / "model_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5/5] 元信息 -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
