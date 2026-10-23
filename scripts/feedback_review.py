"""数据回流第 2 步：把用户反馈整理成一张"待标注清单"。

==============================================================================
整条闭环一共 5 步（README 有图）
==============================================================================
    ① 用户在网页点"认错了"          → serve/app/feedback.py 落盘
    ② 汇总成人能审核的清单（本脚本）  → feedback/review_queue.csv
    ③ 人工在 CSV 里填 final_label     → 人
    ④ 导出成训练数据（feedback_export.py）→ datasets/feedback/<split>/<class>/*.jpg
    ⑤ 增量重训 + 重新导出（train.retrain）→ weights/ 里的新 onnx

用法：
    python scripts/feedback_review.py                 # 生成/更新 feedback/review_queue.csv
    python scripts/feedback_review.py --stats         # 只看统计，不出表
    python scripts/feedback_review.py --priority-only # 只导出"高置信度还判错"的
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from serve.app.feedback import iter_records  # noqa: E402

QUEUE_CSV = ROOT / "feedback" / "review_queue.csv"
FIELDS = [
    "request_id", "ts", "image", "predicted", "confidence",
    "user_label", "agreed", "image_shape", "src", "note",
    # ↓ 以下三列是给人填的
    "final_label", "keep", "review_note",
]


def build_rows(priority_only: bool) -> list[dict]:
    rows = []
    for r in iter_records():
        if r.get("reviewed"):
            continue
        if priority_only and float(r.get("confidence", 0)) < 0.8:
            continue   # 低置信度错判是"已知无能"，高置信度错判才是"致命自信"
        image_rel = r.get("image", "")
        rows.append({
            "request_id": r.get("request_id", ""),
            "ts": r.get("ts", ""),
            "image": image_rel,
            "predicted": r.get("predicted", ""),
            "confidence": r.get("confidence", ""),
            "user_label": r.get("user_label") or "",
            "agreed": "" if r.get("agreed") is None else int(bool(r["agreed"])),
            "image_shape": "x".join(map(str, r.get("image_shape", [0, 0]))),
            "src": r.get("src", ""),
            "note": (r.get("note") or "").replace("\n", " "),
            # 默认策略：用户给了答案就先采信，人工再改
            "final_label": r.get("user_label") or "",
            "keep": 1,
            "review_note": "",
        })
    rows.sort(key=lambda d: -float(d["confidence"] or 0))
    return rows


def stats() -> None:
    recs = list(iter_records())
    if not recs:
        print("还没有任何反馈记录。先在网页上点几下「认错了」，或跑 `python scripts/smoke.py`。")
        return
    c = Counter(r.get("predicted") for r in recs)
    print(f"\n反馈总数          : {len(recs)}")
    print(f"已复核            : {sum(1 for r in recs if r.get('reviewed'))}")
    print(f"用户判为错（给了正确答案）: {sum(1 for r in recs if r.get('agreed') is False)}")
    print(f"用户只标了'不对'   : {sum(1 for r in recs if r.get('user_label') is None)}")
    print(f"用户认可预测       : {sum(1 for r in recs if r.get('agreed') is True)}")
    hi = [r for r in recs if r.get("agreed") is False and float(r.get("confidence", 0)) > 0.8]
    print(f"★ 高置信度错判     : {len(hi)}  <- 这批最该优先看")
    print("\n被质疑最多的预测类别:")
    for k, v in c.most_common():
        print(f"    {k:<22} {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--priority-only", action="store_true",
                    help="只导出高置信度（>=0.8）还被判错的样本")
    args = ap.parse_args()

    if args.stats:
        stats()
        return 0

    rows = build_rows(args.priority_only)
    QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    # 已有的人工填写要保住：按 request_id 合并
    old = {}
    if QUEUE_CSV.exists():
        for d in csv.DictReader(QUEUE_CSV.read_text(encoding="utf-8-sig").splitlines()):
            old[d["request_id"]] = d
    for r in rows:
        o = old.get(r["request_id"])
        if o:
            r["final_label"] = o.get("final_label", r["final_label"])
            r["keep"] = o.get("keep", r["keep"])
            r["review_note"] = o.get("review_note", "")

    with QUEUE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n待复核 {len(rows)} 条 -> {QUEUE_CSV}")
    print("下一步：用 Excel/记事本打开它，在 final_label 列填正确类别、keep 列填 0 表示丢弃，")
    print("       然后跑 `python scripts/feedback_export.py` 生成训练数据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
