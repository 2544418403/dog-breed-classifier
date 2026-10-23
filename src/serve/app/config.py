"""所有可调旋钮集中在一个文件里。

为什么单独拆一个 config？
  防护参数（大小上限、超时、并发数）最怕"散落在业务代码里"。
  一旦散落，你就没法回答"我们服务到底允许多大的文件"这种问题，
  调参时要改 5 个地方，改漏一个就是线上事故。集中起来，一眼能审完。

每个常量都允许用环境变量覆盖 —— 容器里靠 `docker compose` 的 environment 注入，
不需要为了改一个超时重新打镜像。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from common import FEEDBACK, LOGS, ROOT, WEIGHTS, ensure_dirs

ensure_dirs()

MODEL_DIR = Path(os.getenv("DOGBREED_MODEL_DIR", WEIGHTS))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
APP_ROOT = ROOT        # 仓库根（由 common.paths 统一给出，不要在别处再算一遍）

# ---------------- 元信息：由模型告诉服务"我是什么" ----------------
# 这一层的价值：**服务不硬编码任何和训练有关的常量**。
# 类别顺序、归一化参数、输入尺寸全部从导出时写下的 meta 里读。
# 顺序错位不会报错，只会让所有结论反向 —— 所以必须机器可读、不可手抄。
META_PATH = Path(os.getenv("DOGBREED_MODEL_META", MODEL_DIR / "model_meta.json"))


def _load_meta() -> dict:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"找不到模型元信息 {META_PATH}。\n"
            f"仓库自带一份可直接使用的 INT8 权重与 model_meta.json；\n"
            f"如需重新导出：python -m models.export --ckpt weights/baseline.pth --quantize"
        )
    return json.loads(META_PATH.read_text(encoding="utf-8"))


META = _load_meta()

CLASSES: list[str] = META["classes"]
IMG_SIZE: int = int(META["img_size"])
RESIZE_TO: int = int(META.get("resize_to", max(IMG_SIZE, int(IMG_SIZE * 1.15))))
MEAN = tuple(META["mean"])
STD = tuple(META["std"])

# 默认优先用 INT8：精度基本不变，体积 1/4、延迟 1/3（见 README 结果表）。
# 想对比 FP32：DOGBREED_MODEL_FILE=baseline.onnx
DEFAULT_MODEL_FILE = META.get("int8_model_file") or META["model_file"]
MODEL_PATH = MODEL_DIR / os.getenv("DOGBREED_MODEL_FILE", DEFAULT_MODEL_FILE)
MODEL_VARIANT = "int8" if "int8" in MODEL_PATH.name else "fp32"

# 类别的中文名（纯展示用，不参与任何计算）
CLASS_CN = {
    "havanese": "哈瓦那犬",
    "miniature_pinscher": "迷你品捷犬",
    "pomeranian": "博美犬",
    "shiba_inu": "柴犬",
    "yorkshire_terrier": "约克夏梗",
}


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ---------------- 观测：请求追踪 ----------------
# 每次推理往 logs/requests-YYYYMMDD.jsonl 追加一行 JSON。
# 为什么不用 print：上线三个月后你想知道"上周三下午是不是有一批竖图都预测错了"，
# print 出来的文本没法聚合，JSONL 一句 jq 就有答案。
ENABLE_REQUEST_LOG = _env_bool("DOGBREED_REQUEST_LOG", True)
REQUEST_LOG_DIR = Path(os.getenv("DOGBREED_LOG_DIR", LOGS))

# 写盘的节流参数。这两个值是压测压出来的，不是拍脑袋定的：
#   同步写盘时（老实现）16 并发 QPS 只有 69，改成异步后回到 355。
#   队列容量 4096 = 满载时约 12s 的日志缓冲，足够扛住写盘的一次抖动。
TELEMETRY_QUEUE_SIZE = _env_int("DOGBREED_LOG_QUEUE_SIZE", 4096)
TELEMETRY_FLUSH_S = _env_float("DOGBREED_LOG_FLUSH_S", 1.0)

# ---------------- 反馈闭环 ----------------
# 用户点"预测错了"时，把原图 + 预测 + 用户认为正确的类别落到这里，
# 供后续人工复核 → 增量重训（见 README「数据回流」一节）。
ENABLE_FEEDBACK = _env_bool("DOGBREED_FEEDBACK", True)
FEEDBACK_DIR = Path(os.getenv("DOGBREED_FEEDBACK_DIR", FEEDBACK))
FEEDBACK_MAX_PER_DAY = _env_int("DOGBREED_FEEDBACK_MAX_PER_DAY", 2000)

# ---------------- 防护：输入侧 ----------------
# 单文件大小上限。5MB 对 64px 输入的模型来说已非常宽松
# （一张 64px JPEG 通常 < 20KB）。上限的意义是不让"大文件"成为攻击面。
MAX_FILE_BYTES = _env_int("MAX_FILE_BYTES", 5 * 1024 * 1024)

# 允许的 MIME 白名单。注意 content_type 是客户端填的、**可以伪造**，
# 所以它只能做第一道快筛，真正的判定看文件头的魔数（见 security.py）。
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# 图片像素上限，防止 decompression bomb（一张 64000x64000 的 PNG 只有几 KB，
# 解码后能吃掉几个 GB 内存）。PIL 默认阈值 ~89M 像素，对 64px 模型太松。
MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 32_000_000)

# ---------------- 防护：批量侧 ----------------
# 批量不是越大越好：单次延迟随 batch 变长、内存线性增长，
# 而且一个大 batch 会拖慢它后面排队的每个请求（队头阻塞）。
MAX_BATCH = _env_int("MAX_BATCH", 32)

# ---------------- 防护：并发与超时 ----------------
# 同时在跑推理的请求数上限。CPU 核数有限，放任 100 个请求同时推理，
# 结果是所有人一起变慢、QPS 反而不涨。超限快速失败 503，叫 load shedding。
MAX_INFLIGHT = _env_int("MAX_INFLIGHT", 8)

# 排队多久还没拿到推理名额就放弃（秒）
QUEUE_WAIT_S = _env_float("QUEUE_WAIT_S", 2.0)

# 单次推理的软超时（秒）。
# 诚实说明：这是"软"超时 —— 到点返回 504，但那个卡在线程池里的线程没法被
# 强行杀死（Python 没有安全的 thread kill）。毫秒级推理够用，
# 但要知道边界：真·硬超时需要进程池或直接重启进程。
INFER_TIMEOUT_S = _env_float("INFER_TIMEOUT_S", 3.0)

# 读取上传内容的超时（秒）。慢速客户端会把连接一直占着。
READ_TIMEOUT_S = _env_float("READ_TIMEOUT_S", 10.0)

# onnxruntime 算子内并行线程数。并发已由 MAX_INFLIGHT 提供，
# 这里再开多线程会 oversubscription（线程打架），反而更慢。
ORT_INTRA_OP_THREADS = _env_int("ORT_INTRA_OP_THREADS", 1)
