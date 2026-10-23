"""可观测性：一次请求到底发生了什么，必须有一行机器可读的记录。

==============================================================================
第 8 周的要求是"记录输入尺寸、预测结果、耗时"。真正难的不是记，而是
**三个月后你怎么用**。这套字段是围绕四个真实问题倒推出来的：
==============================================================================

Q1 线上精度是不是在掉？
    → 每条记 shot(img_w, img_h, nbytes) + pred.label + pred.confidence
      离线按天聚合出"平均置信度"和"类别分布"。分布突然变了几乎一定是出问题了
      （用户画像变了 / 上游接入了别的设备 / 有人传二维码进来）。

Q2 慢是谁慢？
    → 把一次请求拆成 read_ms / decode_ms / preprocess_ms / infer_ms / total_ms
      只记 total 的话，你会发现"变慢了"却不知道该优化 IO 还是模型。

Q3 有没有异常输入在系统性地打穿模型？
    → 记 image_mode / aspect / grayscale 这类"形状"字段。
      曾踩过：一批 EXIF 旋转没处理的竖图把某类的召回打下去了。

Q4 用户点反馈的那批请求，当时到底预测了啥？
    → 靠 request_id 把 request log 和 feedback 记录 join 起来。
      这也是为什么 request_id 要同时返回给前端、写进响应头。

==============================================================================
落盘策略：JSON Lines（一行一条）
==============================================================================
选 JSONL 而不是 DB：单机 / 单容器场景下，多一个 Postgres 就多一套运维。
JSONL 可以被 filebeat / promtail / 甚至 `jq` 直接消费，迁移到 ELK 时不用改代码。

==============================================================================
写入方式：后台单线程 + 有界队列（这是被压测逼出来的改动，不是设计癖好）
==============================================================================
最初这里图省事，直接在请求线程里 `open(...).write(...)`。实测代价：
16 并发下 QPS 从 **355 掉到 69（-81%）**，P50 从 43ms 涨到 230ms。
根因是每请求一次 open/write/close 系统调用，且它和外层的互斥锁绑在一起，
把并发请求序列化成了串行。

——监控系统把被监控对象拖慢 5 倍，这是最讽刺的失败方式。

所以现在：请求线程只往内存队列丢一条 dict（有界，默认 4096），
真正的写盘由一个 daemon 线程做，文件句柄长期持有、每秒 flush。
队列满时**丢日志而不是拖慢请求**，并把 drop 计数暴露到 /metrics：
宁可少几条日志，也不能让观测反过来成为 SLO 的瓶颈。
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime

from typing import Any

from serve.app.config import (
    ENABLE_REQUEST_LOG,
    MODEL_VARIANT,
    REQUEST_LOG_DIR,
    TELEMETRY_FLUSH_S,
    TELEMETRY_QUEUE_SIZE,
)

_QUEUE_MAX_DROP_LOG = 200   # 丢弃告警打太多次会刷屏，只打前 200 次

_LOCK = threading.Lock()      # 只保护内存滑动窗口，务必轻量

# 内存里的滑动窗口：给 /metrics 用。
# 为什么还要留一份在内存里：磁盘上的历史可以慢慢分析，但"现在是不是在报警"
# 需要立刻能回答，而且要能被 Prometheus 那种 15s 一次的抓取读到。
_WINDOW: deque = deque(maxlen=2000)

_START_TS = time.time()

_LOG_QUEUE: queue.Queue = queue.Queue(maxsize=int(TELEMETRY_QUEUE_SIZE))
_DROPPED = 0
_WRITTEN = 0
_READY = threading.Event()


def _open_for_today(handle=None):
    """按天切文件。当天变化时关掉旧句柄换新句柄。"""
    REQUEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    if handle is not None and getattr(handle, "_dogbreed_day", None) == today:
        return handle
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 - 关不掉就算了，别把线程带崩
            pass
    fh = (REQUEST_LOG_DIR / f"requests-{today}.jsonl").open("a", encoding="utf-8")
    fh._dogbreed_day = today     # 偷偷挂个属性标记当前日期，避免每次都 stat
    return fh


def _writer_loop() -> None:
    """后台写线程：直到队列空且距上次 flush 超过间隔，才真正 flush。"""
    global _WRITTEN
    fh = None
    last_flush = time.time()
    try:
        while True:
            timeout = max(0.05, TELEMETRY_FLUSH_S)
            try:
                item = _LOG_QUEUE.get(timeout=timeout)
            except queue.Empty:
                if fh is not None and time.time() - last_flush >= TELEMETRY_FLUSH_S:
                    try:
                        fh.flush()
                    except Exception:  # noqa: BLE001
                        pass
                    last_flush = time.time()
                continue

            if item is None:        # 停止信号：排空剩余后退出
                break

            try:
                fh = _open_for_today(fh)
                fh.write(item)
                _WRITTEN += 1
            except Exception as e:  # noqa: BLE001 - 观测失败永远不许冒泡到业务
                print(f"[telemetry] 写请求日志失败：{e}", flush=True)

            if time.time() - last_flush >= TELEMETRY_FLUSH_S:
                if fh is not None:
                    try:
                        fh.flush()
                    except Exception:  # noqa: BLE001
                        pass
                last_flush = time.time()
    finally:
        if fh is not None:
            try:
                fh.flush()
                fh.close()
            except Exception:  # noqa: BLE001
                pass


_writer_thread: threading.Thread | None = None


def start_writer() -> None:
    """在 app 启动时调用（lifespan）。幂等。"""
    global _writer_thread
    if not ENABLE_REQUEST_LOG or _writer_thread is not None:
        return
    _writer_thread = threading.Thread(target=_writer_loop, name="telemetry-writer",
                                      daemon=True)
    _writer_thread.start()
    _READY.set()


def stop_writer() -> None:
    """停服时把队列里剩下的写干净。日志丢尾巴，排查时最气人。"""
    if _writer_thread is None or not _writer_thread.is_alive():
        return
    try:
        _LOG_QUEUE.put(None, timeout=2)
    except queue.Full:
        pass
    _writer_thread.join(timeout=5)


def _drain_now() -> None:
    """同步兜底：进程被 kill(SIGKILL/signal 9) 时 flush 来不及，
    平时靠 atexit 至少保证正常退出路径不丢。"""
    stop_writer()


if ENABLE_REQUEST_LOG:
    start_writer()
    atexit.register(_drain_now)


def log_request(record: dict[str, Any]) -> None:
    """写一条请求记录：内存窗口 + 异步落盘 JSONL。

    这个函数在**请求的关键路径上**，所以它必须只做 O(1) 的内存操作。
    刻意**不做**任何可能抛异常往上冒泡的事：观测代码把业务请求搞挂，
    是监控系统最经典的翻车方式。
    """
    global _DROPPED
    record["ts"] = datetime.now().isoformat(timespec="milliseconds")
    record.setdefault("variant", MODEL_VARIANT)

    _WINDOW.append(record)          # deque.append 本身线程安全，无需锁

    if not ENABLE_REQUEST_LOG:
        return

    try:
        _LOG_QUEUE.put_nowait(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except queue.Full:
        # 队列满 -> 宁可丢日志也不阻塞请求。这是主动选择，不是漏笔画 behaviors。
        _DROPPED += 1
        if _DROPPED <= _QUEUE_MAX_DROP_LOG:
            print(f"[telemetry] 日志队列满，已丢弃 {_DROPPED} 条（服务压力过高）",
                  flush=True)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def snapshot(window: int = 500) -> dict:
    """给 /metrics 的聚合视图。只用最近 window 条，反映"此刻"。"""
    recs = list(_WINDOW)[-window:]      # 不加锁：deque 的切片取快照本身就是一次性的原子拷贝

    n = len(recs)
    if n == 0:
        base = {"requests": 0, "uptime_s": round(time.time() - _START_TS, 1)}
        base.update(TELEMETRY_HEALTH)
        return base

    lat = [float(r.get("total_ms", 0)) for r in recs]
    inf = [float(r.get("infer_ms", 0)) for r in recs]
    conf = [float(r["confidence"]) for r in recs if r.get("confidence") is not None]
    batch = [int(r.get("batch", 1)) for r in recs]

    labels: dict[str, int] = {}
    for r in recs:
        lab = r.get("label")
        if lab:
            labels[lab] = labels.get(lab, 0) + 1

    errors = sum(1 for r in recs if r.get("status", 200) >= 400)

    out = {
        "uptime_s": round(time.time() - _START_TS, 1),
        "requests_in_window": n,
        "qps_window": round(n / max(1e-9, sum(lat) / 1000.0), 1),
        "latency_ms": {
            "p50": round(_pct(lat, 0.50), 2),
            "p95": round(_pct(lat, 0.95), 2),
            "p99": round(_pct(lat, 0.99), 2),
            "max": round(max(lat), 2),
        },
        "infer_ms": {
            "p50": round(_pct(inf, 0.50), 2),
            "p95": round(_pct(inf, 0.95), 2),
            "p99": round(_pct(inf, 0.99), 2),
        },
        "confidence": {
            "mean": round(sum(conf) / len(conf), 4) if conf else None,
            "below_0.5": round(sum(1 for c in conf if c < 0.5) / max(1, len(conf)), 4),
        },
        "input": {
            "avg_bytes": round(sum(int(r.get("bytes", 0)) for r in recs) / n),
            "avg_width": round(sum(int(r.get("width", 0)) for r in recs) / n),
            "avg_height": round(sum(int(r.get("height", 0)) for r in recs) / n),
            "avg_batch": round(sum(batch) / n, 2),
        },
        "label_distribution": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "error_rate": round(errors / n, 4),
        "model_variant": MODEL_VARIANT,
    }
    return out


def health() -> dict:
    """日志子系统自己的健康度 —— 监控组件本人也需要被监控。

    queue_depth 持续上涨 / dropped 只增不减，说明写盘速度跟不上请求速度：
    这时候该扩容日志盘或调大 TELEMETRY_QUEUE_SIZE，而不是假装没看见。
    """
    return {
        "enabled": bool(ENABLE_REQUEST_LOG),
        "queue_depth": _LOG_QUEUE.qsize(),
        "queue_capacity": int(TELEMETRY_QUEUE_SIZE),
        "written": _WRITTEN,
        "dropped": _DROPPED,
        "writer_alive": bool(_writer_thread is not None and _writer_thread.is_alive()),
        "log_dir": str(REQUEST_LOG_DIR),
    }


def recent(limit: int = 20) -> list[dict]:
    """最近 N 条原始记录 —— 排查具体问题时比聚合值有用得多。"""
    with _LOCK:
        return list(_WINDOW)[-limit:]
