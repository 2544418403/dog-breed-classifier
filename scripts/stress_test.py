"""
scripts/stress_test.py —— 用 requests 做并发压测，测 QPS 和 P99

==============================================================================
压测里最容易自欺欺人的三件事（本脚本逐条规避）
==============================================================================
1. **不预热就计时** → 第一个请求要付一次性成本（线程池、内存分配、JIT），
   这个冷启动延迟会被算进 P99，让 P99 虚高好几倍。
   解决：正式计时前先跑 --warmup 次不计入统计。

2. **用平均延迟当指标** → 均值会把长尾藏起来。服务化的真实体验由 P99 决定，
   因为一个页面可能要串行调你好几次。所以这里只看 P50/P95/P99，不看均值。

3. **并发数靠"循环里发 100 次"实现** → 那不是并发，那是串行 100 次，
   测出来的是延迟的倒数，不是 QPS。真正的并发必须有 N 个**同时在飞**的请求。
   解决：ThreadPoolExecutor(max_workers=--concurrency)，每个线程持有一个
   独立的 requests.Session（Session 不是线程安全的，不能共享）。

==============================================================================
用法
==============================================================================
    python scripts/stress_test.py --concurrency 100 --requests 1000
    python scripts/stress_test.py --concurrency 100 --mode batch --batch-size 8
    python scripts/stress_test.py --url http://127.0.0.1:8000 --image assets/shiba.jpg
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

# 本机 127.0.0.1 不该走 HTTP_PROXY（CI / 内网常常配了代理）
requests.Session.trust_env = False
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "assets" / "demo.jpg"

_local = threading.local()


def get_session() -> requests.Session:
    """每个工作线程一个 Session。

    requests 的 Session 自带连接池，复用它能省掉每个请求的 TCP+TLS 握手。
    但它**不是线程安全的**，所以不能全局共享一个 —— 每个线程建自己的。
    """
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        # pool_maxsize 必须 >= 并发数，否则连接池本身会成为瓶颈，
        # urllib3 会打 "Connection pool is full, discarding connection" 警告
        a = requests.adapters.HTTPAdapter(pool_connections=256, pool_maxsize=256)
        s.mount("http://", a)
        s.mount("https://", a)
        _local.session = s
    return s


def make_image_bytes(path: Path | None, size: int = 320) -> tuple[bytes, str]:
    """准备请求体。没有真实图片时就地合成一张（压测关心的是吞吐，不是精度）。"""
    if path and path.exists():
        return path.read_bytes(), path.name
    rng = np.random.default_rng(0)
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "synthetic.jpg"


def one_request(url: str, payload: bytes, filename: str,
                mode: str, batch_size: int, timeout: float):
    """发一个请求，返回 (延迟ms, HTTP状态码, 错误类型)。"""
    s = get_session()
    if mode == "batch":
        files = [("files", (filename, payload, "image/jpeg")) for _ in range(batch_size)]
    else:
        files = [("file", (filename, payload, "image/jpeg"))]

    t0 = time.perf_counter()
    try:
        r = s.post(url, files=files, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000
        r.content  # 读完 body，否则连接不会归还连接池
        return dt, r.status_code, None
    except requests.exceptions.Timeout:
        return (time.perf_counter() - t0) * 1000, 0, "timeout"
    except Exception as e:
        return (time.perf_counter() - t0) * 1000, 0, type(e).__name__


def pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=100, help="同时在飞的请求数")
    ap.add_argument("--requests", type=int, default=1000, help="总请求数")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--tag", default="", help="结果文件的标记，方便多组对比")
    args = ap.parse_args()

    endpoint = f"{args.url}/predict" if args.mode == "single" else f"{args.url}/predict_batch"
    payload, fname = make_image_bytes(args.image)

    # ---- 连通性检查 ----
    try:
        h = requests.get(f"{args.url}/health", timeout=5).json()
        print(f"服务：{h['model']['file']}  输入 {h['model']['input_shape']}  "
              f"已运行 {h['uptime_s']}s  并发上限 {h['max_inflight']}")
    except Exception as e:
        print(f"连不上 {args.url}：{e}\n先启动服务：uvicorn app.main:app --port 8000")
        return 1

    # ---- 1) 预热：不计入统计 ----
    print(f"预热 {args.warmup} 次…", end="", flush=True)
    for _ in range(args.warmup):
        one_request(endpoint, payload, fname, args.mode, args.batch_size, args.timeout)
    print(" 完成")

    # ---- 2) 正式压测 ----
    print(f"压测：并发 {args.concurrency} × 共 {args.requests} 请求 → {endpoint}")
    lat, codes, errs = [], [], []
    lock = threading.Lock()

    def task(_i):
        dt, code, err = one_request(endpoint, payload, fname, args.mode, args.batch_size, args.timeout)
        with lock:
            lat.append(dt)
            codes.append(code)
            if err:
                errs.append(err)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(task, range(args.requests)))
    wall = time.perf_counter() - t_start

    a = np.array(lat)
    ok = np.array([c == 200 for c in codes])
    n_ok = int(ok.sum())
    qps = n_ok / wall
    images = n_ok * (args.batch_size if args.mode == "batch" else 1)
    ips = images / wall

    from collections import Counter
    code_cnt = dict(sorted(Counter(codes).items(), key=lambda kv: -kv[1]))
    err_cnt = dict(Counter(errs))

    res = {
        "tag": args.tag or f"c{args.concurrency}-{args.mode}",
        "mode": args.mode,
        "batch_size": args.batch_size if args.mode == "batch" else 1,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "wall_s": round(wall, 3),
        "success": n_ok,
        "error_rate": round(1 - n_ok / max(1, len(codes)), 4),
        "qps": round(qps, 2),
        "images_per_sec": round(ips, 2),
        "latency_ms": {
            "min": round(float(a.min()), 2),
            "p50": round(pct(a, 50), 2),
            "p90": round(pct(a, 90), 2),
            "p95": round(pct(a, 95), 2),
            "p99": round(pct(a, 99), 2),
            "max": round(float(a.max()), 2),
        },
        "status_codes": {str(k): v for k, v in code_cnt.items()},
        "errors": err_cnt,
    }

    L = res["latency_ms"]
    print("\n" + "=" * 62)
    print(f"  QPS              {res['qps']:.1f} req/s"
          + (f"   ({res['images_per_sec']:.1f} 张图/s, batch={args.batch_size})" if args.mode == "batch" else ""))
    print(f"  P50 / P95 / P99  {L['p50']:.1f} / {L['p95']:.1f} / {L['p99']:.1f} ms"
          f"   (min {L['min']:.1f}, max {L['max']:.1f})")
    print(f"  成功率            {(1 - res['error_rate']) * 100:.1f}%  （{n_ok}/{len(codes)}）")
    print(f"  状态码分布        {res['status_codes']}")
    if err_cnt:
        print(f"  异常              {err_cnt}")
    print(f"  总耗时            {wall:.2f}s")
    print("=" * 62)

    out = ROOT / "outputs" / f"bench_{res['tag']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
