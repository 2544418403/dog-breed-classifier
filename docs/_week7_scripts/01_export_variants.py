"""
scripts/01_export_variants.py —— 把一份训练态权重，派生出所有「服务候选形态」

==============================================================================
本周的第一个问题：同一个模型，到底有几种「能跑」的形态？
==============================================================================
第四周留下的 baseline.pth 只是**权重**，需要宿主（torch + 模型定义）才能跑。
但"能跑的方式"不止一种，每种形态的**体积 / 延迟 / 精度**都不同，
本周就是把这些形态全部做出来，放到同一把尺子下量。

本脚本派生 4 类形态：

  A. torch-fp32        PyTorch 原生 Float32            ← 训练态直出，性能基准
  B. onnx-fp32         ONNX + onnxruntime Float32      ← 第五周部署现状
  C. torch-int8-dyn    PyTorch 动态量化 INT8           ← torch.quantization.quantize_dynamic
  D. onnx-int8-dyn     ONNX 动态量化 INT8              ← ORT 权重压到 1/4
  E. onnx-int8-static  ONNX 静态量化 INT8              ← 带校准集，本周的最终赢家

（04_quant_tune.py 会告诉你为什么 D 是"看起来很美、实际很差"的一个选项）

==============================================================================
两个踩坑点，写在前头（这两个坑是真实踩出来的，不是教科书的）
==============================================================================
【坑 1】Windows 上 torch 的动态量化对 Conv2d 不生效
    查不到 fbgemm 后端时，torch 只会把 Linear / LSTM 换成量化版本，
    53 个 Conv2d 全部原样保留为 FP32 —— 于是体积几乎不变，速度也没提升。
    本脚本会**打印出到底有几个模块真被量化了**，不要相信"调用了 API 就一定生效"。
    这也是为什么在实际项目里，CNN 量化通常走 ONNX Runtime 而不是 torch eager API。

【坑 2】onnxruntime 的量化工具离不开「纯英文路径」
    ORT 的 quantize_* 会在**输入文件旁边**生成 xxx-inferred.onnx 中间文件，
    路径含非 ASCII 字符时它会 FileNotFoundError；torch.jit.save 同理。
    解决办法：量化动作全部在 tempfile 的英文临时目录里做，做完再拷回 models/。
    这类问题不会出现在报错信息里，只能靠环境观测提前知道。

==============================================================================
关键概念：动态量化 vs 静态量化
==============================================================================
两者都把**权重**从 32 位浮点压到 8 位整数（体积直降约 4 倍），区别在于**激活值**：

  动态量化：权重 scale 离线算好，激活值 scale 在**每次推理时现算**。
            → 无需校准数据，一行代码，鲁棒。
            → 代价：运行时要额外扫一遍激活值求 min/max，有额外开销。

  静态量化：提前用一批代表性数据「校准」，把激活值 scale 也**离线冻住**。
            → 推理时零额外开销，通常更快。
            → 代价：校准集必须代表真实分布，否则某些层被削道，精度掉得毫无征兆。

一句话：**动态量化图省事，静态量化图极限但要喂数据**。

==============================================================================
用法
==============================================================================
    python scripts/01_export_variants.py

产物：models/ 下各形态文件 + models/variants.json（供 02_benchmark.py 消费）
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
WEEK3 = ROOT.parent / "第三周"
WEEK4 = ROOT.parent / "第四周"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"

# ---- 与训练期严格对齐的常量（改任何一个都会导致精度对不上）----
LABEL_NAMES = ["havanese", "miniature_pinscher", "pomeranian", "shiba_inu", "yorkshire_terrier"]
N_CLASSES = 5
IMG_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CKPT = WEEK4 / "ckpts" / "baseline.pth"


def build_model() -> nn.Module:
    """重建 resnet50 骨架（torchvision 版，与第五周 export_onnx.py 同源）。"""
    from torchvision import models

    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    return m


def load_source_model(ckpt: Path, img: int) -> tuple[nn.Module, dict]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    classes = ck.get("classes", LABEL_NAMES)
    img = int(cfg.get("img", img))
    model = build_model()
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    print(f"  源权重已加载 {ckpt.name}  img={img}  classes={len(classes)}")
    return model, {"cfg": cfg, "classes": classes, "img": img}


def mb(p: Path) -> float:
    return round(p.stat().st_size / 1024 / 1024, 2)


def count_quantized(model: nn.Module) -> dict:
    """统计真正被替换成量化模块的算子 —— 验证「API 调了，但它到底生效没有」。

    判定方式不看类名（量化版 Conv2d 的类名也可能就叫 Conv2d），而是看它
    属于哪个 module：只要来自 torch.ao.nn.quantized.* 家族，就是量化版。
    """
    conv_q = conv_fp = lin_q = lin_fp = 0
    for _, mo in model.named_modules():
        mod = type(mo).__module__.lower()
        name = type(mo).__name__
        is_q = "quantized" in mod or "nnq" in mod
        if name == "Conv2d":
            conv_q += int(is_q)
            conv_fp += int(not is_q)
        elif name == "Linear":
            lin_q += int(is_q)
            lin_fp += int(not is_q)
    return {"conv_q": conv_q, "conv_fp": conv_fp, "lin_q": lin_q, "lin_fp": lin_fp}


# ---------------------------------------------------------------------------
class CalibReader:
    """给 ORT quantize_static 喂校准数据的标准接口（实现 get_next 即可）。

    刻意用 **train 集** 而不是 test 集校准：校准是在告诉量化器"激活值真实
    分布是什么"，用测试集等于让模型偷看答案，测试集上的精度就不再可信。
    """

    def __init__(self, x: np.ndarray, input_name: str, n: int = 200):
        self.x = x[:n]
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        batch = self.x[self.i : self.i + 1]
        self.i += 1
        return {self.input_name: batch}

    def rewind(self):
        self.i = 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--img", type=int, default=IMG_SIZE)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-torch-quant", action="store_true")
    ap.add_argument("--skip-onnx-quant", action="store_true")
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    variants: list[dict] = []

    print("=" * 78)
    print("第七周 · 01 导出所有推理形态")
    print("=" * 78)
    print(f"torch={torch.__version__}  量化后端可用={torch.backends.quantized.supported_engines}  "
          f"当前={torch.backends.quantized.engine}")

    model, meta = load_source_model(args.ckpt, args.img)
    img = meta["img"]
    dummy = torch.zeros(1, 3, img, img, dtype=torch.float32)

    # ---------- A. torch-fp32 ----------
    src_mb = round(args.ckpt.stat().st_size / 1024 / 1024, 2)
    variants.append({
        "name": "torch-fp32", "kind": "torch", "file": str(args.ckpt),
        "size_mb": src_mb, "note": "PyTorch state_dict + 模型定义（训练态直出）",
    })
    print(f"\n[A] torch-fp32 {src_mb:>8.2f} MB   训练态直出，无需导出")

    # ---------- B. onnx-fp32 ----------
    fp32_onnx = MODELS / "fp32.onnx"
    if not fp32_onnx.exists():
        torch.onnx.export(
            model, dummy, str(fp32_onnx),
            export_params=True, opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    variants.append({
        "name": "onnx-fp32", "kind": "onnx", "file": str(fp32_onnx),
        "size_mb": mb(fp32_onnx), "note": "第五周部署现状：ONNX + onnxruntime CPU",
    })
    print(f"[B] onnx-fp32 {mb(fp32_onnx):>8.2f} MB   opset={args.opset}  BN已折叠")

    # ---------- C. torch 动态量化 ----------
    if not args.skip_torch_quant:
        try:
            q_model = torch.quantization.quantize_dynamic(
                model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
            )
            s = count_quantized(q_model)
            conv_quant, conv_fp32 = s["conv_q"], s["conv_fp"]
            lin_quant = s["lin_q"]

            out = MODELS / "torch_dynamic_int8.pt"
            torch.save(q_model.state_dict(), str(out))

            note = (f"quantize_dynamic：Conv2d 量化 {conv_quant}/53（残留 FP32={conv_fp32}），"
                    f"Linear 量化 {lin_quant}/1")
            if conv_quant == 0:
                note += "；【注意】本环境缺 fbgemm 后端，Conv2d 动态量化未生效"
            variants.append({
                "name": "torch-int8-dyn", "kind": "torch", "file": str(out),
                "size_mb": mb(out), "note": note,
                "quantized_ops": {"conv": conv_quant, "conv_fp32_left": conv_fp32, "linear": lin_quant},
            })
            print(f"[C] torch-int8-dyn {mb(out):>8.2f} MB   {note}")
        except Exception as e:
            print(f"[C] torch-int8-dyn 失败：{e}")
            traceback.print_exc()

    # ---------- D. ONNX 动态量化（必须在英文临时目录里完成）----------
    if not args.skip_onnx_quant:
        scratch = Path(tempfile.gettempdir()) / "wk7_quant"
        scratch.mkdir(parents=True, exist_ok=True)
        src_copy = scratch / "fp32.onnx"
        shutil.copy(fp32_onnx, src_copy)
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            dyn_out = scratch / "onx_dyn_int8.onnx"
            quantize_dynamic(
                model_input=str(src_copy), model_output=str(dyn_out),
                weight_type=QuantType.QInt8,
                per_channel=True,      # 逐通道 scale：精度损失显著小于整层一个 scale
                reduce_range=False,    # 现代 x86 CPU 不需要
            )
            dst = MODELS / "onnx_dynamic_int8.onnx"
            shutil.copy(dyn_out, dst)
            variants.append({
                "name": "onnx-int8-dyn", "kind": "onnx", "file": str(dst),
                "size_mb": mb(dst), "note": "ORT 动态量化：per-channel QInt8，无需校准集",
            })
            print(f"[D] onnx-int8-dyn {mb(dst):>8.2f} MB   per-channel QInt8（在临时目录中转）")
        except Exception as e:
            print(f"[D] onnx-int8-dyn 失败：{e}")
            traceback.print_exc()

        # ---------- E. ONNX 静态量化 ----------
        try:
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

            calib_x = np.load(OUT / "calib_train.npy")
            dst = MODELS / "onnx_static_int8.onnx"
            st_out = scratch / "onx_static_int8.onnx"
            rd = CalibReader(calib_x, "input", n=min(args.calib_n, len(calib_x)))
            quantize_static(
                model_input=str(src_copy), model_output=str(st_out),
                calibration_data_reader=rd,
                quant_format=QuantFormat.QOperator,
                weight_type=QuantType.QInt8,      # 权重有符号
                activation_type=QuantType.QUInt8, # 激活值无符号（ReLU 后非负，多 1 bit 精度）
                per_channel=True,
            )
            shutil.copy(st_out, dst)
            variants.append({
                "name": "onnx-int8-static", "kind": "onnx", "file": str(dst),
                "size_mb": mb(dst),
                "note": f"ORT 静态量化；用 {len(calib_x)} 张 train 图校准激活值范围",
            })
            print(f"[E] onnx-int8-static {mb(dst):>8.2f} MB   校准 {len(calib_x)} 张 train 图")
        except Exception as e:
            print(f"[E] onnx-int8-static 失败：{e}")
            traceback.print_exc()

    # ---------- 汇总 ----------
    out = {
        "source_ckpt": str(args.ckpt),
        "arch": "resnet50",
        "classes": meta["classes"],
        "img_size": img,
        "opset": args.opset,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "torch_version": torch.__version__,
        "quant_engines": torch.backends.quantized.supported_engines,
        "variants": variants,
    }
    (MODELS / "variants.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "-" * 78)
    print(f"{'形态':<20}{'体积(MB)':>11}{'压缩比':>10}")
    print("-" * 78)
    for v in variants:
        print(f"{v['name']:<20}{v['size_mb']:>11.2f}{src_mb / v['size_mb']:>9.2f}x")
    print("-" * 78)
    print(f"models/variants.json 已写出，共 {len(variants)} 个形态")
    return 0


if __name__ == "__main__":
    sys.exit(main())
