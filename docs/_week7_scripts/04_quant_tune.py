"""
scripts/04_quant_tune.py —— 为什么 INT8 会掉点？把各种量化配方摆到同一张表里

==============================================================================
02_benchmark.py 跑出来一个刺眼的结果
==============================================================================
    onnx-int8-dyn：体积 89.6MB → 22.6MB（省了 75%），
                   但 top-1 从 0.8919 掉到 0.8333，**整整 -5.9 个百分点**，
                   预测翻转率高达 14.3%。

教科书上 INT8 量化通常只掉零点几个点，这个数字明显不对劲。
所以本脚本做的事情是：**把量化配方拆开，逐个变量试**，
找出到底是哪个开关在偷走精度。

==============================================================================
本脚本要回答的三个问题
==============================================================================
1. per_channel 有没有帮助？（逐通道 scale 应该比整层一个 scale 更准）
2. 最敏感的是哪一层？—— 用「逐层排除」的二分法定位。
3. 静态量化（带校准集）能不能救回来？

==============================================================================
核心诊断方法：逐层排除法定位误差源
==============================================================================
一种实用做法是 nodes_to_exclude：把某些节点的权重**排除在量化之外**，
让它们保持 FP32。先排除最可疑的那一层（通常是第一层 conv1，
它直接吃原始输入，量化误差会在后面的 50 层里被放大），
看精度能不能回来——能回来，就说明误差源定位到了。

用法：
    python scripts/04_quant_tune.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODELS = ROOT / "models"

x = np.load(OUT / "test_x.npy")
y = np.load(OUT / "test_y.npy")
calib = np.load(OUT / "calib_train.npy")

SCRATCH = Path(tempfile.gettempdir()) / "wk7_qtune"
SCRATCH.mkdir(parents=True, exist_ok=True)
FP32_SRC = SCRATCH / "fp32.onnx"
if not FP32_SRC.exists():
    shutil.copy(MODELS / "fp32.onnx", FP32_SRC)

import onnxruntime as ort  # noqa: E402


def eval_ort(path: Path, threads: int = 1, bs: int = 32) -> dict:
    """在 546 张 test 图上算准确率 + 单张延迟。"""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    outs = []
    for i in range(0, len(x), bs):
        outs.append(sess.run(["logits"], {name: np.ascontiguousarray(x[i : i + bs])})[0])
    logits = np.concatenate(outs, 0)
    acc = float((logits.argmax(1) == y).mean())

    for i in range(20):
        sess.run(["logits"], {name: x[i : i + 1]})
    ts = []
    for i in range(100):
        t = time.perf_counter()
        sess.run(["logits"], {name: x[i % len(x) : i % len(x) + 1]})
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {
        "acc": round(acc, 4),
        "p50_ms": round(ts[len(ts) // 2], 2),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "logits": logits,
    }


class CalibReader:
    def __init__(self, x, input_name, n=200):
        self.x, self.input_name, self.n, self.i = x[:n], input_name, n, 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        b = self.x[self.i : self.i + 1]
        self.i += 1
        return {self.input_name: b}

    def rewind(self):
        self.i = 0


def node_names_by_pattern(pattern: str) -> list[str]:
    """找出名字里包含某模式的节点，用于「排除量化」。"""
    import onnx

    m = onnx.load(str(FP32_SRC), load_external_data=False)
    return [n.name for n in m.graph.node if pattern in n.name]


def main() -> int:
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static,
    )

    print("=" * 78)
    print("第七周 · 04 量化配方调优")
    print("=" * 78)

    base = eval_ort(FP32_SRC)
    ref_logits = base.pop("logits")
    print(f"\n基线 onnx-fp32 : top1={base['acc']:.4f}  P50={base['p50_ms']}ms  {base['size_mb']}MB")

    results = [{"name": "onnx-fp32(基线)", **base, "flip": 0.0}]
    trial = lambda tag: SCRATCH / f"{tag}.onnx"

    def report(tag: str, note: str, path: Path):
        r = eval_ort(path)
        lg = r.pop("logits")
        flip = float((lg.argmax(1) != ref_logits.argmax(1)).mean())
        d = {"name": tag, "note": note, "flip": round(flip, 4), **r}
        results.append(d)
        print(f"  {tag:<26} acc={r['acc']:.4f} (Δ{(r['acc']-base['acc'])*100:+.2f}pp)  "
              f"翻转={flip:.2%}  P50={r['p50_ms']}ms  {r['size_mb']}MB")
        return d

    # ---------- 配方 1：动态量化，per_channel 开/关 ----------
    print("\n[1] 动态量化 · per_channel 开关")
    for pc in (True, False):
        tag = f"dyn_pc{int(pc)}"
        quantize_dynamic(model_input=str(FP32_SRC), model_output=str(trial(tag)),
                         weight_type=QuantType.QInt8, per_channel=pc, reduce_range=False)
        report(tag, f"per_channel={pc}, reduce_range=False", trial(tag))

    # ---------- 配方 2：动态量化 · reduce_range ----------
    print("\n[2] 动态量化 · reduce_range")
    tag = "dyn_pc1_rr"
    quantize_dynamic(model_input=str(FP32_SRC), model_output=str(trial(tag)),
                     weight_type=QuantType.QInt8, per_channel=True, reduce_range=True)
    report(tag, "per_channel=True, reduce_range=True(7bit)", trial(tag))

    # ---------- 配方 3：排除最敏感层 ----------
    print("\n[3] 逐层排除法定位误差源")
    try:
        excl = node_names_by_pattern("conv1")
        tag = "dyn_excl_conv1"
        quantize_dynamic(model_input=str(FP32_SRC), model_output=str(trial(tag)),
                         weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
                         nodes_to_exclude=excl if excl else None)
        report(tag, f"排除 {len(excl)} 个 conv1 类节点", trial(tag))
    except Exception as e:
        print(f"  排除 conv1 失败：{e}")

    # ---------- 配方 4：静态量化（带真实校准集）----------
    print("\n[4] 静态量化 · 用 200 张 train 图校准")
    try:
        tag = "static_int8"
        rd = CalibReader(calib, "input", n=200)
        quantize_static(model_input=str(FP32_SRC), model_output=str(trial(tag)),
                        calibration_data_reader=rd,
                        quant_format=QuantFormat.QOperator,
                        weight_type=QuantType.QInt8,
                        activation_type=QuantType.QUInt8,
                        per_channel=True)
        report(tag, "静态量化 QOperator + per_channel", trial(tag))
    except Exception as e:
        print(f"  静态量化失败：{e}")

    # ---------- 结论 ----------
    print("\n" + "=" * 78)
    print(f"{'配方':<26}{'体积MB':>8}{'P50ms':>9}{'top1':>9}{'Δ vs 基线':>11}{'翻转率':>9}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<26}{r['size_mb']:>8.1f}{r['p50_ms']:>9.2f}{r['acc']:>9.4f}"
              f"{(r['acc'] - base['acc']) * 100:>+10.2f}pp{r['flip']:>9.2%}")
    print("-" * 78)

    clean = [{k: v for k, v in r.items()} for r in results]
    (OUT / "quant_tune.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 outputs/quant_tune.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
