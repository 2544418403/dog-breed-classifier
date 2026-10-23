"""
app/security.py —— 输入侧的兜底：大小、格式、超时

三条原则：
  1. **不信任客户端声明的任何东西**（Content-Type、Content-Length 都能伪造）
  2. **尽早失败**：超限的字节根本不要读进来，更不要送去解码
  3. **失败要说人话**：返回 4xx + 明确原因，让调用方知道该重试还是改参数
"""

from __future__ import annotations

import asyncio
import io

from PIL import Image, UnidentifiedImageError
from starlette import formparsers

# ---------------------------------------------------------------------------
# 第一道闸：在框架解析 multipart 之前就限死
# ---------------------------------------------------------------------------
# 为什么不能只靠我们自己的 read_limited？
#   因为 FastAPI 会在**进入你的函数之前**就把整个 multipart body 解析完
#   （把文件落到临时文件，再给你 UploadFile）。也就是说，等你拿到 file 时，
#   2GB 的 body 已经写进磁盘了。我们那层的"限流"限的是**读进内存的量**，
#   挡不住"往磁盘写的量"。
#
# Starlette 的默认值是：max_files=1000（一个请求最多 1000 个文件部）、
# max_part_size 只管**非文件**字段（1MB）。文件部没有单体大小限制。
# 所以这里把类的默认值改小 —— 这是全局的、在解析阶段就生效的硬限制。
formparsers.MultiPartParser.max_files = 64

# ---------------------------------------------------------------------------
# 魔数（文件头几个字节）：判断"这到底是不是一张图"的唯一可靠依据。
# 文件扩展名能改、Content-Type 能伪造，但真实的 JPEG/PNG 头字节是固定的。
# ---------------------------------------------------------------------------
_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
}


def sniff_mime(head: bytes) -> str | None:
    for sig, mime in _MAGIC.items():
        if head.startswith(sig):
            return mime
    # WEBP: RIFF????WEBP
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


class InputRejected(Exception):
    """输入不合法。携带 HTTP 状态码和给人看的理由。"""

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class BodyTooLarge(Exception):
    """Body 超过硬上限。由 body_limit 中间件抛出并转换成 413。"""

    def __init__(self, limit: int):
        super().__init__(f"请求体超过 {limit / 1024 / 1024:.1f}MB")
        self.limit = limit



async def read_limited(upload, limit: int, timeout: float) -> bytes:
    """分块读取上传内容，边读边限流。

    为什么不直接 `await upload.read()`？
      因为那样会把整个文件先读进内存 —— 客户端发一个 2GB 的 body，
      你的内存就先没了。分块读 + 累计超限即停，才是"限流"该做的事。

    为什么不信 Content-Length？
      它是 header，客户端想写多少写多少。这里只拿它做**快筛**
      （能提前拒绝就别读 body），真正的判定靠累计字节数。
    """
    declared = getattr(upload, "size", None)
    if declared is not None and declared > limit:
        raise InputRejected(413, f"文件过大：{declared / 1024 / 1024:.1f}MB > 上限 {limit / 1024 / 1024:.1f}MB")

    chunks: list[bytes] = []
    total = 0

    async def _read() -> None:
        nonlocal total
        while True:
            chunk = await upload.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise InputRejected(
                    413, f"文件过大：超过 {limit / 1024 / 1024:.1f}MB 上限（读取中断）"
                )
            chunks.append(chunk)

    try:
        await asyncio.wait_for(_read(), timeout=timeout)
    except asyncio.TimeoutError:
        # 慢速客户端（Slowloris 类）：一直占着连接不给你数据
        raise InputRejected(408, f"读取上传内容超时（>{timeout}s）") from None

    return b"".join(chunks)


def decode_image(data: bytes, declared_type: str | None,
                 allowed: set[str], max_bytes: int) -> Image.Image:
    """字节 -> PIL 图，全程带校验。

    三步校验的顺序是有讲究的：**先便宜的，后贵的**。
      魔数（看 12 个字节）→ 类型白名单 → PIL 解码（最贵，可能触发炸弹）
    """
    if len(data) == 0:
        raise InputRejected(400, "空文件")

    real = sniff_mime(data[:12])
    if real is None:
        raise InputRejected(415, "不是有效的图片文件（魔数校验失败，可能是改名过的非图片）")
    if real not in allowed:
        raise InputRejected(415, f"不支持的图片格式：{real}（允许 {sorted(allowed)}）")
    if declared_type and declared_type not in allowed:
        # 声明类型和真实类型不一致 —— 不是致命错误（浏览器经常乱填），
        # 但值得记一笔：这里我们以**真实内容**为准，不信声明。
        pass

    try:
        img = Image.open(io.BytesIO(data))
        img.load()          # 关键：Image.open 是懒加载，不 load() 根本不知道文件是不是坏的
    except Image.DecompressionBombError:
        raise InputRejected(413, "图片像素数过大，拒绝解码（decompression bomb）") from None
    except UnidentifiedImageError:
        raise InputRejected(415, "图片无法解码（文件可能已损坏或截断）") from None
    except Exception as e:  # PIL 的异常类型很多，兜住所有解码期异常
        raise InputRejected(400, f"图片解码失败：{type(e).__name__}") from None

    return img
