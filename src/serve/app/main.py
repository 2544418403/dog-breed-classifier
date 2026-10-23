"""HTTP 层：把模型变成一个可靠的接口。

==============================================================================
三个决定了"能不能上线"的问题
==============================================================================

1) 模型在进程里"活在哪"？
   错误写法（新手最常犯）：每个请求 load_model() → 读几十 MB 权重 → QPS 从几百掉到个位数。
   正确写法：**进程启动时加载一次，所有请求共享**。FastAPI 给这件事留的标准位置是
   `lifespan`：yield 之前 = 启动，yield 之后 = 清理，uvicorn 保证整个进程只跑一次。

2) 推理是 CPU 密集的，会阻塞事件循环吗？
   `async def` 里的同步代码**不会**自动变并发。直接在 async 函数里调 `session.run()`，
   那 15ms 里整个事件循环都被卡住。所以：
     读上传内容用 await（IO 密集）／ 推理用 run_in_threadpool（CPU 密集）

3) 出问题的时候你怎么知道？
   这一周补上的答案：每次请求写一行结构化日志（telemetry.log_request），
   `/metrics` 给出滚动窗口聚合值，前端把 request_id 还给用户用作反馈主键。

==============================================================================
接口清单
==============================================================================
    GET  /              上传页面
    GET  /health        readiness：模型加载了吗、现在还能接活吗
    GET  /meta          自描述：类别顺序 / 输入尺寸 / 批量上限
    GET  /metrics       滚动窗口 P50/P95/P99、置信度分布、类别分布、错误率
    POST /predict       单张推理
    POST /predict_batch 批量推理（一次 forward 跑 N 张）
    POST /feedback      用户纠错（数据回流的起点）

启动：
    uvicorn serve.app.main:app --host 127.0.0.1 --port 8000 --workers 1
注意 --workers：每个 worker 是独立进程，会各加载一份模型。
单进程 + 内部并发（MAX_INFLIGHT）对 CPU 推理通常是最优解。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from serve.app.config import (
    ALLOWED_CONTENT_TYPES,
    CLASS_CN,
    CLASSES,
    ENABLE_FEEDBACK,
    INFER_TIMEOUT_S,
    MAX_BATCH,
    MAX_FILE_BYTES,
    MAX_INFLIGHT,
    MODEL_PATH,
    MODEL_VARIANT,
    QUEUE_WAIT_S,
    READ_TIMEOUT_S,
    STATIC_DIR,
)
from serve.app import feedback as fb
from serve.app import telemetry
from serve.app.model import OnnxClassifier
from serve.app.security import BodyTooLarge, InputRejected, decode_image, read_limited

import logging

from common import setup_logging

log = setup_logging("serve")

# 进程级单例：模型、并发闸门、统计计数器都挂在这里。
# 用模块级 dict 而不是全局变量，是为了让 lifespan 里 clear() 能一次清干净。
MODEL: dict = {}
STATS: dict = {
    "requests": 0,       # 进到推理函数的请求数
    "ok": 0,             # 成功返回的
    "errors": 0,         # 5xx
    "rejected": 0,       # 4xx（被防护挡下的）
    "feedback": 0,       # 收到的用户反馈条数
    "inflight": 0,       # 正在推理的
    "inflight_http": 0,  # 正在处理的 HTTP 请求（含排队/上传）
}
_START_TS = time.time()


# ==============================================================================
# 生命周期：加载一次
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("启动：加载模型 %s ...", MODEL_PATH.name)
    t0 = time.perf_counter()
    MODEL["clf"] = await run_in_threadpool(OnnxClassifier, MODEL_PATH)
    MODEL["sem"] = asyncio.Semaphore(MAX_INFLIGHT)   # 并发闸门
    boot_ms = (time.perf_counter() - t0) * 1000

    clf: OnnxClassifier = MODEL["clf"]
    log.info(
        "模型就绪：加载 %.0fms（含预热 %.0fms）  input=%s  variant=%s  并发上限=%d",
        boot_ms, clf.warmup_ms, clf.input_shape, MODEL_VARIANT, MAX_INFLIGHT,
    )
    MODEL["boot_ms"] = boot_ms

    telemetry.start_writer()        # 幂等；日志落盘线程在这里显式启动

    yield   # ←───────── 服务在这里跑完一辈子 ─────────

    telemetry.stop_writer()         # 把队列里剩下的写干净，别丢日志尾巴
    log.info("关闭：释放模型（日志已落盘 %d 条）", telemetry.health()["written"])
    MODEL.clear()


app = FastAPI(
    title="犬种分类推理服务",
    version="1.0.0",
    description=(
        "ResNet50 / Oxford-Pets-5 类犬种分类器。\n\n"
        "模型以 ONNX INT8 静态量化形态部署，CPU 上 P50 ≈ 2.6ms。\n"
        "支持单张 / 批量推理、结构化请求日志 /metrics，以及用户纠错回流的 /feedback。"
    ),
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# 中间件 0：body 硬上限 —— 在框架解析 multipart 之前把超大请求掐掉
# ==============================================================================
# 为什么这一层必须存在（而不是只靠路由里的 read_limited）：
#   FastAPI 对 `UploadFile` 的解析发生在**进入你的函数之前**（await request.form()）。
#   一个 2GB 的上传在你拿到 file 之前就已经被完整落盘了，路由里再限制是马后炮。
#
# 为什么写成 ASGI 包装器而不是 @app.middleware("http")：
#   后者只能拿到 Request，想拦 body 就得替换 request._receive，容易破坏 ASGI 语义
#   （实测直接导致 multipart 解析失败）。这里直接包一层 ASGI app，原样转发每个
#   message，只在 http.request 的 body 上累加计数，语义最干净。
class BodyLimitMiddleware:
    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        limit = self.limit

        # 快筛：带了 content-length 且明显超标，一个字节都不用读
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > limit:
                    STATS["rejected"] += 1
                    return await _send_413(send, limit)
                break

        state = {"received": 0, "over": False}

        async def guarded_receive():
            msg = await receive()
            if msg["type"] == "http.request":
                state["received"] += len(msg.get("body", b""))
                if state["received"] > limit:
                    state["over"] = True
                    # 截断：不再向下游交付更多字节
                    return {"type": "http.request", "body": b"", "more_body": False}
            return msg

        async def guarded_send(msg):
            # 解析器拿到"截断的"body 要么提前结束、要么报错；两种情况都拦成 413
            if state["over"] and msg["type"] == "http.response.start" and msg["status"] == 200:
                STATS["rejected"] += 1
                msg = dict(msg)
                msg["status"] = 413
            await send(msg)

        await self.app(scope, guarded_receive, guarded_send)


async def _send_413(send, limit: int):
    import json as _json
    body = _json.dumps({"request_id": "-", "error": {
        "code": 413,
        "message": f"请求体过大：超过 {limit / 1024 / 1024:.1f}MB 上限"}},
        ensure_ascii=False).encode("utf-8")
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


# 上限 = 单文件上限 × (批量上限 + 余量)，余量留给 multipart 边界和头部
app.add_middleware(BodyLimitMiddleware, limit=MAX_FILE_BYTES * (MAX_BATCH + 4))


# ==============================================================================
# 中间件 1：请求 ID + 耗时 + 全局在途计数
# ==============================================================================
# request_id 是整个可观测体系的**主键**：
#   响应头带回去 → 用户在页面上复制它 → 点反馈时带回来 →
#   feedback 记录与当时的请求日志用同一个 id join 起来。
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.rid = rid
    t0 = time.perf_counter()
    STATS["inflight_http"] += 1
    try:
        response = await call_next(request)
    except BodyTooLarge as e:
        STATS["rejected"] += 1
        log.warning("[%s] 请求体超限：%.1fMB", rid, e.limit / 1024 / 1024)
        response = JSONResponse(
            status_code=413,
            content={"request_id": rid, "error": {
                "code": 413, "message": f"请求体过大：超过 {e.limit / 1024 / 1024:.1f}MB 上限（已中断读取）"}},
        )
    except Exception:
        # 兜底中的兜底：中间件层也得接住，否则客户端拿到的是空响应
        STATS["errors"] += 1
        log.exception("[%s] 未捕获异常", rid)
        response = JSONResponse(
            status_code=500,
            content={"request_id": rid, "error": {"code": 500, "message": "内部错误"}},
        )
    finally:
        STATS["inflight_http"] -= 1
    dt = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time"] = f"{dt:.1f}ms"
    return response


# ==============================================================================
# 统一的异常出口：客户端永远拿到同构 JSON，永远不会看到 Python 堆栈
# ==============================================================================
@app.exception_handler(InputRejected)
async def on_input_rejected(request: Request, exc: InputRejected):
    STATS["rejected"] += 1
    log.warning("[%s] 拒绝输入：%s", request.state.rid, exc.reason)
    return JSONResponse(
        status_code=exc.status,
        content={"request_id": request.state.rid,
                 "error": {"code": exc.status, "message": exc.reason}},
    )


@app.exception_handler(HTTPException)
async def on_http_exc(request: Request, exc: HTTPException):
    STATS["errors"] += 1
    return JSONResponse(
        status_code=exc.status_code,
        content={"request_id": request.state.rid,
                 "error": {"code": exc.status_code, "message": str(exc.detail)}},
    )


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    STATS["errors"] += 1
    log.exception("[%s] 未处理异常", request.state.rid)
    return JSONResponse(
        status_code=500,
        content={"request_id": request.state.rid,
                 "error": {"code": 500, "message": "服务内部错误，请联系管理员"}},
    )


# ==============================================================================
# 推理执行器：并发闸门 + 超时
# ==============================================================================
async def guarded_infer(batch):
    """带着闸门和超时跑一次推理。

    两道闸：
      1. 并发闸门（Semaphore）：超过上限快速失败 503（load shedding），
         而不是让所有人一起变慢。调用方看到 503 后退避重试。
      2. 超时：任何一次推理卡住，3 秒后返回 504。注意这是**软**超时 ——
         已经在线程池里跑起来的线程没法被安全杀死。
    """
    sem: asyncio.Semaphore = MODEL["sem"]

    # wait_for 的 timeout 不能设 0 —— 那样连"立刻能拿到"的情况也会被判超时
    try:
        await asyncio.wait_for(sem.acquire(), timeout=QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        STATS["rejected"] += 1
        raise HTTPException(503, "服务繁忙，请稍后重试") from None

    STATS["inflight"] += 1
    try:
        return await asyncio.wait_for(
            run_in_threadpool(MODEL["clf"].predict_array, batch),
            timeout=INFER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        STATS["errors"] += 1
        raise HTTPException(504, f"推理超时（>{INFER_TIMEOUT_S}s）") from None
    finally:
        STATS["inflight"] -= 1
        sem.release()


# ==============================================================================
# 可观测性
# ==============================================================================
@app.get("/health")
def health():
    """健康检查。

    /health 该回答的不是"进程活着吗"（那是 k8s 的 liveness），
    而是"**这个实例现在能接活吗**"（readiness）：模型加载成功了吗？排队压力大不大？
    所以返回 model_loaded 和 inflight，负载均衡据此决定是否摘流量。
    """
    clf: OnnxClassifier | None = MODEL.get("clf")
    if clf is None:
        raise HTTPException(503, "模型尚未加载完成")
    return {
        "status": "ok",
        "model": {
            "file": MODEL_PATH.name,
            "variant": MODEL_VARIANT,
            "classes": len(clf.classes),
            "input_shape": list(clf.input_shape),
            "load_ms": round(MODEL.get("boot_ms", 0), 1),
        },
        "inflight": STATS["inflight"],
        "inflight_http": STATS["inflight_http"],
        "max_inflight": MAX_INFLIGHT,
        "uptime_s": round(time.time() - _START_TS, 1),
        "stats": {"requests": STATS["requests"], "ok": STATS["ok"],
                  "errors": STATS["errors"], "rejected": STATS["rejected"],
                  "feedback": STATS["feedback"]},
    }


@app.get("/meta")
def meta():
    """给调用方的自描述接口：类别顺序是什么、图片会被怎么处理。

    没有它，调用方只能猜"返回的 3 是柴犬还是博美"。
    类别顺序错位是线上最常见也最难发现的 bug 之一。
    """
    clf: OnnxClassifier = MODEL["clf"]
    return {
        "classes": clf.classes,
        "classes_cn": {c: CLASS_CN.get(c, c) for c in clf.classes},
        "input_size": clf.input_shape,
        "max_batch": MAX_BATCH,
        "variant": MODEL_VARIANT,
        "feedback_enabled": ENABLE_FEEDBACK,
    }


@app.get("/metrics")
def metrics(window: int = Query(500, ge=1, le=2000)):
    """最近 N 条请求的聚合指标。

    Prometheus 要的是文本格式，这里给 JSON：
      - 一个自用的服务，JSON + curl 就能看，不需要再加一层 exporter
      - 真要接 Prometheus，用 pushgateway 或一个 15 行的 exporter 转换即可
      - `window` 可调，是为了能同时看"此刻"和"这一小时"
    """
    snap = telemetry.snapshot(window)
    return {
        "serve": snap,
        "counters": {
            "requests": STATS["requests"],
            "ok": STATS["ok"],
            "errors": STATS["errors"],
            "rejected": STATS["rejected"],
            "feedback": STATS["feedback"],
        },
        # 监控组件自己的健康度：日志队列堵了也要能被看见
        "telemetry": telemetry.health(),
        "feedback_summary": fb.count_summary(),
    }


@app.get("/metrics/recent")
def metrics_recent(limit: int = Query(20, ge=1, le=200)):
    """最近 N 条原始请求记录 —— 排查具体问题时比聚合值有用得多。"""
    return {"items": telemetry.recent(limit)}


# ==============================================================================
# 接口：单张推理
# ==============================================================================
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    STATS["requests"] += 1
    rid = request.state.rid
    t0 = time.perf_counter()

    # --- 校验：Content-Type 快筛（便宜，先做） ---
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise InputRejected(415, f"不支持的格式：{file.content_type}（允许 {sorted(ALLOWED_CONTENT_TYPES)}）")

    # --- 读取：限大小 + 限时间 ---
    data = await read_limited(file, MAX_FILE_BYTES, READ_TIMEOUT_S)

    # --- 解码：魔数校验 + 真解码（CPU 密集，丢线程池） ---
    t_dec = time.perf_counter()
    img = await run_in_threadpool(
        decode_image, data, file.content_type, ALLOWED_CONTENT_TYPES, MAX_FILE_BYTES
    )
    decode_ms = (time.perf_counter() - t_dec) * 1000

    t_pre = time.perf_counter()
    x = await run_in_threadpool(MODEL["clf"].preprocess, img)
    pre_ms = (time.perf_counter() - t_pre) * 1000

    # --- 推理：一次 forward ---
    try:
        preds, infer_ms = await guarded_infer(x[None, ...])
    except HTTPException as e:
        telemetry.log_request({
            "request_id": rid, "endpoint": "/predict", "status": e.status_code,
            "bytes": len(data), "width": img.width, "height": img.height,
            "declined_reason": str(e.detail),
        })
        raise

    STATS["ok"] += 1
    p = preds[0]
    total_ms = (time.perf_counter() - t0) * 1000

    topk = sorted(
        ({"label": c, "label_cn": CLASS_CN.get(c, c), "confidence": round(float(pp), 4)}
         for c, pp in zip(MODEL["clf"].classes, p.probs)),
        key=lambda d: -d["confidence"],
    )[:3]

    telemetry.log_request({
        "request_id": rid,
        "endpoint": "/predict",
        "status": 200,
        "filename": file.filename,
        "bytes": len(data),
        "width": img.width,
        "height": img.height,
        "aspect": round(img.width / max(1, img.height), 3),
        "mode": img.mode,
        "batch": 1,
        "read_decode_ms": round(decode_ms, 2),
        "preprocess_ms": round(pre_ms, 2),
        "infer_ms": round(infer_ms, 2),
        "total_ms": round(total_ms, 2),
        "label": p.label,
        "confidence": round(p.confidence, 4),
        "second_label": topk[1]["label"] if len(topk) > 1 else None,
        "margin": round(float(topk[0]["confidence"] - (topk[1]["confidence"] if len(topk) > 1 else 0)), 4),
    })

    return {
        "request_id": rid,
        "filename": file.filename,
        "label": p.label,
        "label_cn": p.label_cn,
        "class_index": p.index,
        "confidence": round(p.confidence, 4),
        "topk": topk,
        "timing": {
            "decode_ms": round(decode_ms, 2),
            "preprocess_ms": round(pre_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "total_ms": round(total_ms, 2),
        },
    }


# ==============================================================================
# 接口：批量推理
# ==============================================================================
@app.post("/predict_batch")
async def predict_batch(request: Request, files: list[UploadFile] = File(...)):
    """批量推理。

    **意义不是"省 HTTP 往返"，而是"省 forward"。**
    一张一张跑：N 次 kernel launch + N 次内存搬运
    堆成 (N,3,64,64) 一次跑：1 次 launch，算子内部并行度高得多。
    代价是单次延迟随 batch 变大，所以必须有 MAX_BATCH。
    """
    STATS["requests"] += 1
    rid = request.state.rid
    t0 = time.perf_counter()

    if len(files) == 0:
        raise InputRejected(400, "没有收到任何文件")
    if len(files) > MAX_BATCH:
        raise InputRejected(400, f"批量过大：{len(files)} > 上限 {MAX_BATCH}")

    xs, names, metas = [], [], []
    for f in files:
        if f.content_type and f.content_type not in ALLOWED_CONTENT_TYPES:
            raise InputRejected(415, f"{f.filename}: 不支持的格式 {f.content_type}")
        data = await read_limited(f, MAX_FILE_BYTES, READ_TIMEOUT_S)
        img = await run_in_threadpool(
            decode_image, data, f.content_type, ALLOWED_CONTENT_TYPES, MAX_FILE_BYTES
        )
        xs.append(await run_in_threadpool(MODEL["clf"].preprocess, img))
        names.append(f.filename)
        metas.append({"bytes": len(data), "width": img.width, "height": img.height})

    batch = np.stack(xs, axis=0)          # (N,3,64,64) —— batching 就发生在这行
    preds, infer_ms = await guarded_infer(batch)
    STATS["ok"] += 1
    total_ms = (time.perf_counter() - t0) * 1000

    telemetry.log_request({
        "request_id": rid,
        "endpoint": "/predict_batch",
        "status": 200,
        "files": len(files),
        "batch": len(files),
        "bytes": sum(m["bytes"] for m in metas),
        "width": round(sum(m["width"] for m in metas) / len(metas)),
        "height": round(sum(m["height"] for m in metas) / len(metas)),
        "infer_ms": round(infer_ms, 2),
        "per_image_ms": round(infer_ms / max(1, len(preds)), 2),
        "total_ms": round(total_ms, 2),
        "labels": [p.label for p in preds],
        "mean_confidence": round(float(np.mean([p.confidence for p in preds])), 4),
    })

    return {
        "request_id": rid,
        "count": len(preds),
        "infer_ms": round(infer_ms, 2),
        "per_image_ms": round(infer_ms / max(1, len(preds)), 2),
        "total_ms": round(total_ms, 2),
        "results": [
            {
                "filename": n,
                "label": p.label,
                "label_cn": p.label_cn,
                "class_index": p.index,
                "confidence": round(p.confidence, 4),
            }
            for n, p in zip(names, preds)
        ],
    }


# ==============================================================================
# 接口：用户反馈（数据回流入口）
# ==============================================================================
@app.post("/feedback")
async def feedback(
    request: Request,
    image: UploadFile = File(..., description="那张被判错的图"),
    predicted: str = Body(..., description="当时模型给出的类别（/predict 返回的 label）"),
    confidence: float = Body(..., ge=0.0, le=1.0, description="当时模型给出的置信度"),
    request_id: str = Body("", description="当时那次推理的 request_id，用于和请求日志 join"),
    user_label: str | None = Body(None, description="用户认为正确的类别；不确定时留空"),
    topk: list[dict] | None = Body(None, description="/predict 返回的 topk，原样带回即可"),
    note: str = Body("", description="补充说明"),
    src: str = Body("web", description="来源标记：web / cli / bot"),
):
    """用户说"你认错了"的落点 —— 数据回流闭环的第一步。

    三个诚实的实现决定：

    1. **fig 由客户端重传，而不是服务端按 request_id 回捞。**
       否则服务必须长期持有用户上传的明文图片，隐私面凭空变大。
       让用户显式再传一次，存下来的每张图都是用户授权的。

    2. **user_label 允许为空**（用户只知道"不对"，不知道正确答案）。
       这类样本价值极高：它标记了"模型自信但错误"的区域，
       是下一轮数据收集优先级最高的地方。

    3. **反馈路径不许影响推理路径**。写盘失败要让前端知道（accepted=false），
       但绝不让它把 5xx 抛到 /predict 上去。
    """
    rid = request.state.rid
    if not ENABLE_FEEDBACK:
        raise HTTPException(404, "本实例未开启反馈采集")

    if predicted not in CLASSES:
        raise InputRejected(400, f"未知类别：{predicted}（可选 {CLASSES}）")
    if user_label is not None and user_label not in CLASSES:
        raise InputRejected(400, f"未知类别：{user_label}（可选 {CLASSES}）")

    data = await read_limited(image, MAX_FILE_BYTES, READ_TIMEOUT_S)

    # 只为了记下原图形状，不跑模型 —— 反馈不该消耗 GPU/CPU 预算
    shape = (0, 0)
    try:
        from PIL import Image as _Img
        import io as _io

        with _Img.open(_io.BytesIO(data)) as im:
            shape = (im.width, im.height)
    except Exception:  # noqa: BLE001 - 形状只是附带信息，取不到就算了
        pass

    record = fb.save_feedback(
        request_id=request_id or rid,
        data=data,
        predicted=predicted,
        confidence=confidence,
        topk=topk or [],
        user_label=user_label,
        image_shape=shape,
        infer_ms=0.0,
        src=src,
        note=note,
    )
    STATS["feedback"] += 1
    log.info("[%s] 收到反馈 ref=%s predicted=%s user_label=%s shape=%s",
             rid, request_id, predicted, user_label, shape)
    return {"request_id": rid, "accepted": bool(record.get("stored")), "record": record}


# ==============================================================================
# 前端页面
# ==============================================================================
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
