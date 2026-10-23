"""
scripts/02_benchmark.py —— 把所有形态放到同一把尺子上量

==============================================================================
本周唯一的硬指标：拿出三个维度的真实数字
==============================================================================
  ① 模型体积    → 磁盘占多少、镜像多大、冷启动要读多久
  ② 单次延迟    → 用户上传一张图要等多久（P50 / P95，不是平均值！）
  ③ 准确率      → 优化有没有把模型优化坏了（top-1 + 预测翻转率）

==============================================================================
方法论：几个必须做对的细节，否则数字不可信
==============================================================================
1. 输入先冻住
   所有形态吃的是同一个 test_x.npy，预处理早已完成并缓存。
   这样测出来的差异 100% 来自推理引擎，而不是"这次解码快了一点"。

2. 分位数而不是平均值
   延迟的均值会被偶发抖动（OS 调度、页缓存）带偏。
   服务端真正关心的是 P95 / P99 —— 那才是用户会投诉的那一批请求。

3. 预热
   ORT 第一次 run 要做图优化，torch 第一次要初始化线程池和 oneDNN 计划。
   不预热，第一个数能比稳态慢 10 倍，整个基准就废了。

4. 线程数必须对齐生产环境
   第六周 Dockerfile 里写的是 ORT_INTRA_OP_THREADS=1（单agem进程单线程池）。
   所以主表跑 threads=1；另外单独跑 threads=4 看多线程下的收益差异。
   **量化的收益在不同线程数下完全不同**，这是很多人忽略的点。

5. 输出─┐除此之外，还要看「预测翻转率」
   准确率是一个粗粒度的指标：掉了 0.2% 你可能看不出发生了什么。
   翻转率告诉你：有多少样本的预测结果**变了**（哪怕另一份样本被反向修好，
   准确率看起来纹丝不动）。这是量化体检里最灵敏的一根探针。

==============================================================================
用法
==============================================================================
    python scripts/02_benchmark.py                    # CPU 全量 + GPU（若可用）
    python scripts/02_benchmark.py --no-gpu
    python scripts/02_benchmark.py --repeat 300       # 每个形态多测几次

产物：outputs/benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
WEEK4 = ROOT.parent / "第四周"
OUT = ROOT / "outputs"

LABEL_NAMES = ["havanese", "miniature_pinscher", "pomeranian", "shiba_inu", "yorkshire_terrier"]
N_CLASSES = 5


# ===========================================================================
# 统一的「引擎」抽象：底下是 torch 还是 ORT 无所谓，对外只有 predict(x)->logits
# ===========================================================================
class Engine:
    name: str = ""
    kind: str = ""

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def build_model() -> nn.Module:
    from torchvision import models

    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    return m


class TorchFp32(Engine):
    """A. PyTorch 原生 Float32 —— 性能基准线。"""

    def __init__(self, ckpt: Path, threads: int):
        self.name = "torch-fp32"
        self.kind = "torch"
        torch.set_num_threads(threads)
        t0 = time.perf_counter()
        m = build_model()
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=True)
        m.eval()
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.m = m
        self.size_mb = round(ckpt.stat().st_size / 1024 / 1024, 2)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        # CPU tensor 的 .numpy() 是零拷贝，成本可忽略；
        # 统一返回 numpy，和 ORT 引擎口径一致。
        return self.m(torch.from_numpy(x)).numpy()


class TorchDynInt8(Engine):
    """C. torch.quantization.quantize_dynamic。

    注意：正如 01 脚本验证过的，本机的动态量化只对 Linear 生效，
    53 个 Conv2d 全部保持 FP32 —— 所以这个形态大概率**不会更快**，
    把它放进表里是为了用数据而不是记忆来判断。
    """

    def __init__(self, ckpt: Path, quant_file: Path, threads: int):
        self.name = "torch-int8-dyn"
        self.kind = "torch"
        torch.set_num_threads(threads)
        t0 = time.perf_counter()
        m = build_model()
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=True)
        m.eval()
        q = torch.quantization.quantize_dynamic(m, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
        # 量化版模块的 state_dict 里存的是压缩后的 int8 权重，
        # 必须先把结构搭出来再灌进去，不能直接 load 到普通 resnet50 上。
        q.load_state_dict(torch.load(quant_file, map_location="cpu", weights_only=False))
        q.eval()
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.m = q
        self.size_mb = round(quant_file.stat().st_size / 1024 / 1024, 2)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        # CPU tensor 的 .numpy() 是零拷贝，成本可忽略；
        # 统一返回 numpy，和 ORT 引擎口径一致。
        return self.m(torch.from_numpy(x)).numpy()


class Onnx(Engine):
    """B / D. ONNX Runtime CPU 推理。"""

    def __init__(self, onnx_path: Path, threads: int, name: str):
        import onnxruntime as ort

        self.name = name
        self.kind = "onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.in_name = self.sess.get_inputs()[0].name
        self.size_mb = round(onnx_path.stat().st_size / 1024 / 1024, 2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(["logits"], {self.in_name: x})[0]


# ===========================================================================
# 测量工具
# ===========================================================================
def percentile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    f, c = int(np.floor(k)), int(np.ceil(k))
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)


def measure_latency(eng: Engine, x: np.ndarray, repeat: int = 200, warmup: int = 30) -> dict:
    """单张推理延迟。x 形状 (N,C,H,W)，逐张送进去。"""
    n = min(repeat, len(x))
    for i in range(warmup):
        eng.predict(x[i % len(x) : i % len(x) + 1])
    ts = []
    for i in range(n):
        t = time.perf_counter()
        eng.predict(x[i : i + 1])
        ts.append((time.perf_counter() - t) * 1000)
    return {
        "mean_ms": round(statistics.fmean(ts), 3),
        "p50_ms": round(percentile(ts, 0.50), 3),
        "p95_ms": round(percentile(ts, 0.95), 3),
        "p99_ms": round(percentile(ts, 0.99), 3),
        "min_ms": round(min(ts), 3),
        "n": n,
    }


def measure_batch(eng: Engine, x: np.ndarray, bs_list=(1, 4, 8, 16, 32), reps: int = 20) -> dict:
    """吞吐：不同 batch size 下每秒能处理多少张图。"""
    res = {}
    for bs in bs_list:
        if bs > len(x):
            continue
        batch = np.ascontiguousarray(x[:bs])
        for _ in range(5):
            eng.predict(batch)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            eng.predict(batch)
            ts.append((time.perf_counter() - t) * 1000)
        med = percentile(ts, 0.5)
        res[str(bs)] = {
            "batch_ms": round(med, 3),
            "img_per_s": round(bs / (med / 1000), 1),
            "per_img_ms": round(med / bs, 3),
        }
    return res


def run_full(eng: Engine, x: np.ndarray, batch: int = 32) -> np.ndarray:
    """整批跑完拿到 logits，用于算准确率。"""
    outs = []
    for i in range(0, len(x), batch):
        outs.append(eng.predict(np.ascontiguousarray(x[i : i + batch])))
    return np.concatenate(outs, axis=0)


def top1(logits: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(logits).argmax(1) == y).mean())


# ===========================================================================
def build_engines(args, threads: int) -> list[Engine]:
    """按线程数重建所有引擎（线程数是 Session/torch 级别的，不能中途改）。"""
    engs: list[Engine] = []
    engs.append(TorchFp32(args.ckpt, threads))
    engs.append(Onnx(args.fp32, threads, "onnx-fp32"))
    qfile = ROOT / "models" / "torch_dynamic_int8.pt"
    if qfile.exists():
        engs.append(TorchDynInt8(args.ckpt, qfile, threads))
    if args.int8.exists():
        engs.append(Onnx(args.int8, threads, "onnx-int8-dyn"))
    if args.static.exists():
        engs.append(Onnx(args.static, threads, "onnx-int8-static"))
    return engs


def bench_cpu(args, x: np.ndarray, y: np.ndarray) -> list[dict]:
    results: list[dict] = []
    for threads in args.threads:
        print(f"\n{'=' * 78}\nCPU  thread={threads}（第六周容器实际配置是 intra_threads=1）\n{'=' * 78}")
        engs = build_engines(args, threads)
        for eng in engs:
            print(f"\n--- {eng.name}  文件 {eng.size_mb} MB  加载 {eng.load_ms:.0f} ms ---")
            rec = {"name": eng.name, "kind": eng.kind, "size_mb": eng.size_mb,
                   "load_ms": round(eng.load_ms, 1), "threads": threads}

            rec["latency"] = measure_latency(eng, x, args.repeat, args.warmup)
            L = rec["latency"]
            print(f"  延迟   P50={L['p50_ms']:.2f}ms  P95={L['p95_ms']:.2f}ms  "
                  f"P99={L['p99_ms']:.2f}ms  mean={L['mean_ms']:.2f}ms")

            rec["batch"] = measure_batch(eng, x, tuple(args.batch_list))
            b1 = rec["batch"].get("1", {})
            print(f"  吞吐   bs1={b1.get('img_per_s', 0):.0f} img/s", end="")
            for k in ("8", "32"):
                if k in rec["batch"]:
                    print(f"   bs{k}={rec['batch'][k]['img_per_s']:.0f} img/s", end="")
            print()

            logits = run_full(eng, x)
            acc = top1(logits, y)
            rec["acc"] = round(acc, 4)
            rec["logits"] = logits           # 留着算翻转率
            print(f"  准确率 top1={acc:.4f}")

            if eng.name == "torch-fp32":
                args._ref_logits = logits
                rec["flip_rate"] = 0.0
                rec["max_logit_diff"] = 0.0
            else:
                ref = getattr(args, "_ref_logits", None)
                if ref is not None:
                    flip = float((logits.argmax(1) != ref.argmax(1)).mean())
                    md = float(np.abs(logits - ref).max())
                    rec["flip_rate"] = round(flip, 4)
                    rec["max_logit_diff"] = round(md, 6)
                    print(f"  与 torch-fp32 对比：预测翻转率={flip:.3%}  max|Δlogit|={md:.3e}")

            results.append(rec)
            del eng
            gc.collect()
    return results


# ===========================================================================
# GPU 部分（任务 4：理解 batch 聚合 / FP16 / torch.compile 的定位）
# ===========================================================================
def bench_gpu(args, x: np.ndarray, y: np.ndarray) -> dict | None:
    if args.no_gpu or not torch.cuda.is_available():
        print("\n[GPU] 不可用或已跳过")
        return None

    dev = torch.device("cuda")
    print(f"\n{'=' * 78}\nGPU  {torch.cuda.get_device_name(0)}\n{'=' * 78}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    def fresh_model(half: bool = False) -> nn.Module:
        """每次都从 state_dict 重建一个干净模型。

        千万别写成 m16 = base.to(dev).half()：`.to()/.half()` 对 Module 是
        **原地生效**的，所有引用共享同一批参数，改一个就把另一个也改了。
        """
        m = build_model()
        m.load_state_dict(sd, strict=True)
        m.eval()
        m = m.to(dev)
        if half:
            m = m.half()
        return m

    out: dict = {"device": torch.cuda.get_device_name(0), "runs": []}

    def time_batch(m, xnp, bs, reps=30, warmup=10, half=False, input_provider=None):
        batch_cpu = np.ascontiguousarray(xnp[:bs])
        with torch.no_grad():
            b = torch.from_numpy(batch_cpu).to(dev)
            if half:
                b = b.half()
            for _ in range(warmup):
                m(b)
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                m(b)
                torch.cuda.synchronize()          # 必须！GPU 是异步执行的
                ts.append((time.perf_counter() - t) * 1000)
        med = percentile(ts, 0.5)
        return {"bs": bs, "batch_ms": round(med, 3), "per_img_ms": round(med / bs, 3),
                "img_per_s": round(bs / (med / 1000), 1)}

    variants_gpu = [("cuda-fp32", fresh_model(False), False)]

    if args.fp16:
        variants_gpu.append(("cuda-fp16", fresh_model(True), True))

    if args.compile:
        variants_gpu.append(("cuda-fp32-compile", torch.compile(fresh_model(False)), False))

    for name, m, half in variants_gpu:
        rows = []
        for bs in (1, 8, 32):
            rows.append(time_batch(m, x, bs, half=half))
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        acc = None
        try:
            outs = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    b = torch.from_numpy(np.ascontiguousarray(x[i : i + 64])).to(dev)
                    if half:
                        b = b.half()
                    outs.append(m(b).float().cpu().numpy())
            acc = round(top1(np.concatenate(outs), y), 4)
        except Exception as e:
            print(f"  [{name}] 准确率计算失败：{e}")
        rec = {"name": name, "rows": rows, "acc": acc, "peak_mem_mb": round(peak, 1)}
        out["runs"].append(rec)
        line = "  " + f"{name:<20}"
        for r in rows:
            line += f"  bs{r['bs']:<3}={r['img_per_s']:>7.0f} img/s"
        print(f"{line}   acc={acc}  峰值显存={peak:.0f}MB")

    for _, m, _ in variants_gpu:
        del m
    torch.cuda.empty_cache()
    return out


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=WEEK4 / "ckpts" / "baseline.pth")
    ap.add_argument("--fp32", type=Path, default=ROOT / "models" / "fp32.onnx")
    ap.add_argument("--int8", type=Path, default=ROOT / "models" / "onnx_dynamic_int8.onnx")
    ap.add_argument("--static", type=Path, default=ROOT / "models" / "onnx_static_int8.onnx")
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--batch-list", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()

    x = np.load(OUT / "test_x.npy")
    y = np.load(OUT / "test_y.npy")
    print("=" * 78)
    print(f"第七周 · 02 性能基准   样本 {x.shape}   标签 {len(y)}")
    print("=" * 78)

    res = bench_cpu(args, x, y)
    gpu = bench_gpu(args, x, y)

    # 干净化：logits 很大，不落盘
    for r in res:
        r.pop("logits", None)

    payload = {
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_physical": psutil.cpu_count(logical=False),
            "cpu_logical": psutil.cpu_count(logical=True),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_samples": int(len(x)),
        },
        "cpu": res,
        "gpu": gpu,
    }
    (OUT / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 已写出 outputs/benchmark.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
