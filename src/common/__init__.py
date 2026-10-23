"""公共工具：路径、随机种子、结构化日志。

单独抽出来的理由很简单：`data/ models/ train/ eval/ serve/` 五个模块都要知道
"仓库根目录在哪"、"日志往哪写"，各自算一遍就会出现两份不一致的答案
（比如 serve 把日志写到 logs/，train 写到 outputs/，然后谁也找不到）。
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# src/common/paths.py -> parents: [common, src, repo-root]
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
WEIGHTS = Path(os.getenv("DOGBREED_WEIGHTS_DIR", ROOT / "weights"))
DATASETS = Path(os.getenv("DOGBREED_DATA_DIR", ROOT / "datasets"))
FIGURES = Path(os.getenv("DOGBREED_FIGURES_DIR", ROOT / "figures"))
OUTPUTS = Path(os.getenv("DOGBREED_OUTPUT_DIR", ROOT / "artifacts"))
LOGS = Path(os.getenv("DOGBREED_LOG_DIR", ROOT / "logs"))
FEEDBACK = Path(os.getenv("DOGBREED_FEEDBACK_DIR", ROOT / "feedback"))

# PET5 的类别顺序：训练、导出、服务三处必须完全一致。
# 顺序错位是这类项目最难查的 bug —— 指标看起来正常，结论全反。
LABEL_NAMES = [
    "havanese",
    "miniature_pinscher",
    "pomeranian",
    "shiba_inu",
    "yorkshire_terrier",
]

LABEL_CN = {
    "havanese": "哈瓦那犬",
    "miniature_pinscher": "迷你品捷犬",
    "pomeranian": "博美犬",
    "shiba_inu": "柴犬",
    "yorkshire_terrier": "约克夏梗",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    """把该有的目录建出来。幂等，可以随便调。"""
    for d in (WEIGHTS, DATASETS, FIGURES, OUTPUTS, LOGS, FEEDBACK):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # 服务侧没有 torch，也不需要可训练的可复现性


# ==============================================================================
# 日志：给人看的格式 vs 给机器看的格式
# ==============================================================================
# 服务有两个读者：
#   人（终端 tail -f）—— 想要彩色、短行、能一眼扫出错误
#   机器（ELK / Loki / 一个简单的 jq）—— 想要 JSON Lines，字段稳定、可聚合
#
# 只给其中一种，另一种一定会痛。所以这里两个都留：
#   DOGBREED_LOG_JSON=1 -> stdout 输出 JSONL（容器里推荐）
#   默认                -> 人读的行格式（本地开发推荐）
class JsonFormatter(logging.Formatter):
    """把一条日志记录序列化成一行 JSON。

    不用第三方 structlog 是因为：这里只需要"把字段塞进 JSON"这一件事，
    多一个依赖就多一个版本漂移点。但字段名的约定照抄 12-factor 的做法：
    ts / level / logger / msg，其余全是业务字段。
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # extras：业务代码用 logger.info("...", extra={"request_id": ...}) 传进来
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in logging.LogRecord.__dict__ or key in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message", "module",
                "msecs", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "taskName",
            ):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def setup_logging(name: str = "dogbreed", *, json_lines: bool | None = None,
                  level: str | None = None) -> logging.Logger:
    """配置并返回一个 logger。重复调用是安全的（幂等）。"""
    if json_lines is None:
        json_lines = os.getenv("DOGBREED_LOG_JSON", "0") == "1"
    level = level or os.getenv("DOGBREED_LOG_LEVEL", "INFO").upper()

    log = logging.getLogger(name)
    log.setLevel(_LEVELS.get(level, logging.INFO))
    log.handlers.clear()
    log.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    if json_lines:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    return log
