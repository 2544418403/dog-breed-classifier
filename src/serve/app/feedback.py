"""用户反馈采集：闭环的起点。

==============================================================================
为什么一个 demo 项目也要认真做这一步
==============================================================================
模型的寿命取决于它见过多少种"它会错的样本"。离线 test 集是冻结的，
真实流量每天都在给你的模型出新的考试卷。没有回流通道，模型上线那天就是它
最好的一天；有回流通道，它才会越用越好。

这条链路刻意做得**很土**：写文件、人工看 CSV。刻意不上标注平台、不做自动
打标，因为：

  1. 冷启动阶段（几百条）用人的判断最准，任何自动流程都在引入你还没验证的假设
  2. 落盘格式一旦和"训练数据 manifest"对齐，后面接什么标注系统都是替换后端
  3. 手动流程跑通一遍，你才知道自动化到底该自动化哪一步

==============================================================================
落盘内容
==============================================================================
feedback/
  inbox/YYYY-MM-DD/<request_id>__<predicted>.jpg      原图（留证 + 用于重训）
  feedback-YYYY-MM-DD.jsonl                           一行一条元数据

每条 JSONL 记录：
  request_id / ts / predicted / predicted_cn / confidence / topk
  user_label           用户认为正确的类别（可能为 null：只点了"不准"）
  image_shape          原图形状 —— 用来发现"某类尺寸系统性出错"
  infer_ms             当时耗时
  src                  来源标记（web / api）

reviewed 字段由 scripts/feedback_review.py 写回，形成：
  采集 → 复核 → 导出 → 重训 → 再导出 → 重新部署
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from serve.app.config import (
    CLASS_CN,
    ENABLE_FEEDBACK,
    FEEDBACK_DIR,
    FEEDBACK_MAX_PER_DAY,
)

_LOCK = threading.Lock()
_COUNT: dict[str, int] = {}
_COUNT_LOCK = threading.Lock()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _paths() -> tuple[Path, Path]:
    day = _today()
    inbox = FEEDBACK_DIR / "inbox" / day
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox, FEEDBACK_DIR / f"feedback-{day}.jsonl"


def save_feedback(*, request_id: str, data: bytes, predicted: str,
                  confidence: float, topk: list[dict], user_label: str | None,
                  image_shape: tuple[int, int], infer_ms: float,
                  src: str = "unknown", note: str = "") -> dict:
    """把一条用户反馈落盘。返回写下的记录（不含图片字节）。

    设计取舍：
      - 图片**直接存原图**。看起来浪费，但它是最不可替代的东西：
        以后想换导出分辨率、想重跑 EMBEDDING 做去重、想复查某个错判，
        都需要原图；而原图一旦丢了就再也拿不回来。
      - 配额（每天上限）保护磁盘：真实服务一定有人会写脚本刷你这个接口。
    """
    with _COUNT_LOCK:
        today = _today()
        _COUNT[today] = _COUNT.get(today, 0) + 1
        if _COUNT[today] > FEEDBACK_MAX_PER_DAY:
            return {"stored": False, "reason": f"当日反馈已达上限 {FEEDBACK_MAX_PER_DAY} 条"}

    record = {
        "request_id": request_id,
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "predicted": predicted,
        "predicted_cn": CLASS_CN.get(predicted, predicted),
        "confidence": round(float(confidence), 4),
        "topk": topk,
        "user_label": user_label,
        "user_label_cn": CLASS_CN.get(user_label, user_label) if user_label else None,
        "agreed": (user_label == predicted) if user_label else None,
        "image_shape": list(image_shape),
        "infer_ms": round(float(infer_ms), 2),
        "src": src,
        "note": note,
        "reviewed": 0,     # scripts/feedback_review.py 复核后置 1
        "stored": True,
    }

    if not ENABLE_FEEDBACK:
        return {**record, "stored": False, "reason": "服务端已关闭反馈采集"}

    inbox, jsonl = _paths()
    suffix = ".jpg"
    img_name = f"{request_id}__{predicted}{suffix}"
    (inbox / img_name).write_bytes(data)

    record["image"] = f"inbox/{_today()}/{img_name}"
    with _LOCK:
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def iter_records(days: list[str] | None = None):
    """遍历 feedback-*.jsonl 里的所有记录（按文件名排序）。"""
    files = sorted(FEEDBACK_DIR.glob("feedback-*.jsonl"))
    if days:
        files = [f for f in files if f.stem.replace("feedback-", "") in set(days)]
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def count_summary() -> dict:
    """给 /metrics 用的一眼概览：攒了多少、多少已复核、模型被推翻了几次。"""
    total = wrong = reviewed = 0
    per_label: dict[str, int] = {}
    for r in iter_records():
        total += 1
        reviewed += int(bool(r.get("reviewed")))
        if r.get("user_label") and r.get("user_label") != r.get("predicted"):
            wrong += 1
            per_label[r["user_label"]] = per_label.get(r["user_label"], 0) + 1
    return {
        "total": total,
        "reviewed": reviewed,
        "disagreed": wrong,
        "disagreed_by_label": dict(sorted(per_label.items(), key=lambda kv: -kv[1])),
        "enabled": ENABLE_FEEDBACK,
    }
