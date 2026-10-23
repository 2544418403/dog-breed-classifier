"""离线延迟基准：直接对 ONNX 文件测体积 / P50 延迟 / 吞吐。

和 scripts/stress_test.py 的分工：
    stress_test  测的是「HTTP 服务」整体端到端（含网络、解码、排队）
    bench_latency 测的是「模型本身」—— 谁慢、量化赢了多少，与 Web 框架无关

用法：
    python scripts/bench_latency.py --all
    python scripts/bench_latency.py weights/dogbreed_resnet50_64_int8.onnx --batch 1 8 32

历史上这张表得出的结论（见 README）：
    FP32 -> 静态 INT8：延迟 8.60ms -> 2.57ms，体积 89.6MB -> 22.7MB，精度 -0.18pp
    *动态*量化那条路走不通：onnx-int8-dyn 延迟反而涨到 100ms（循环而是è°调度开销），
    这个「否定结论」和正面结论一样值钱。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREADS = 1


def make_session(path: Path, threads: int) -> tuple[ort.InferenceSession, str]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def bench(sess, input_name, batch: int, size: int, n: int = 60) -> dict:
    x = np.random.randn(batch, 3, size, size).astype(np.float32)
    out = sess.get_outputs()[0].name

    for _ in range(8):                      # 预热：不预热会把一次性成本算进 P50
        sess.run([out], {input_name: x})

    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run([out], {input_name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "batch": batch,
        "p50_ms": round(statistics.median(ts), 3),
        "p95_ms": round(ts[int(len(ts) * 0.95)], 3),
        "per_image_ms": round(statistics.median(ts) / batch, 3),
        "img_per_s": round(batch * 1000 / statistics.median(ts), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", type=Path, help="onnx 文件路径")
    ap.add_argument("--all", action="store_true", help="自动发现 weights/*.onnx")
    ap.add_argument("--batch", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=60)
    args = ap.parse_args()

    models = args.models or (sorted((ROOT / "weights").glob("*.onnx")) if args.all else [])
    if not models:
        print("没有指定模型。用法：python scripts/bench_latency.py --all")
        return 1

    print(f"\n{'model':<40} {'MB':>7} {'batch':>6} {'P50 ms':>9} {'per-img':>9} {'img/s':>9}")
    print("-" * 86)
    for m in models:
        if not m.exists():
            print(f"{m.name:<40} (找不到文件)")
            continue
        sess, inp = make_session(m, args.threads)
        mb = m.stat().st_size / 1024 / 1024
        real_size = sess.get_inputs()[0].shape[2]
        for b in args.batch:
            r = bench(sess, inp, b, real_size, args.repeat)
            tag = m.name if b == args.batch[0] else ""
            print(f"{tag:<40} {mb:>7.1f} {b:>6} {r['p50_ms']:>9.2f} "
                  f"{r['per_image_ms']:>9.2f} {r['img_per_s']:>9.0f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
