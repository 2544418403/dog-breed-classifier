# ==============================================================================
# 推理服务镜像：多阶段构建 · 不含 torch · 非 root 运行
#
# 构建（上下文必须是仓库根目录）：
#     docker build -t dogbreed-serve:1.0.0 .
# 运行：
#     docker run -d --name dogbreed -p 8000:8000 dogbreed-serve:1.0.0
# ==============================================================================
#
# 为什么用多阶段：builder 里为了编译可能装的东西（gcc / 头文件 / pip 缓存），
# 一点都不会进最终镜像。这是多阶段构建唯一但极其重要的价值 —— 最终层里只有 venv。
#
# 为什么不装 torch：第五周已把模型导出成 ONNX，第七周又量化成了 INT8。
# 服务只需要 onnxruntime（~40MB）。装 torch 的话镜像会从 0.35GB 涨到 3.2GB。
# ==============================================================================

# ---------- 阶段 1：builder —— 只负责装依赖 ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------- 阶段 2：runtime —— 真正跑起来的镜像 ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="dogbreed-serve" \
      org.opencontainers.image.description="犬种分类推理服务（ONNX INT8 / CPU）"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # src 布局：src/ 下的每个子目录都是顶层包名，靠 PYTHONPATH 指过去。
    # 另外官方文档建议 Python 的日志直接刷出来，否则 docker logs 要等缓冲区满才看得到。
    PYTHONPATH=/app/src \
    ORT_INTRA_OP_THREADS=1 \
    MAX_INFLIGHT=8 \
    DOGBREED_LOG_JSON=1

# libgomp1 —— OpenMP，onnxruntime 的多线程靠它，缺了直接起不来
# curl     —— 排查用（进容器 curl 一下比什么都快）
# 刻意没装 libgl1 / libglib2.0-0：那是 OpenCV 的依赖，我们用 PIL 解码，装了纯属浪费 ~90MB
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# 非 root 运行：容器被攻破时攻击者拿不到 root（最小权限原则）
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# 只拷运行必需的东西：源码 + 权重 meta。日志与反馈落 /app/logs /app/feedback
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser weights/model_meta.json weights/*.onnx ./weights/

# 这两个目录是容器运行期要写的，必须先建好（非 root 用户不能凭空建目录）
RUN mkdir -p /app/logs /app/feedback && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# HEALTHCHECK 打 /health 而不是 / —— /health 会检查"模型加载完了吗"，
# 而 / 只是个静态页面；进程活着但模型没加载完时 / 照样 200，那是假健康。
# --start-period 给模型加载留时间，没它启动期的几次失败会把容器标成 unhealthy。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys,urllib.request; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# --workers 1：每个 worker 是独立进程，会各加载一份模型。
# 并发能力已由应用内的 MAX_INFLIGHT 信号量 + 线程池提供（实测单进程 ~350 QPS），
# 横向扩容应该靠多容器 + 负载均衡，不是单容器内堆 worker。
CMD ["uvicorn", "serve.app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
