"""数据回流第 3 步：人工复核过的结果 → 真的能喂给训练脚本的数据。

输入：feedback/review_queue.csv（人工填了 final_label / keep）
输出：
    datasets/feedback/<split>/<class>/*.jpg   重切分后的图片
    datasets/feedback/manifest.csv            与 datasets/pet5/manifest.csv 同构，可直接用

==============================================================================
两个容易被忽略、会把重训做废的细节
==============================================================================
1) **反馈样本必须进 val/test，不能全塞进 train。**
   用户的抱怨天然集中在"模型做不好"的区域。全部拿去训练，你会在测试集上
   看到虚高的提升 —— 因为根本没有新的一份数据能证明它在没见过的地方也变好了。
   这里按 7:1.5:1.5 分层切，并且**按 request_id 去重**，防止同一张图横跨两个 split。

2) **类别不平衡要复核。**
   用户报错的分布和真实分布不一样（长尾类更容易被错，也更容易被追着说错）。
   直接全量并入训练，模型会往用户抱怨最多的类倾倒。这里打印合并前后的
   类别占比，让你决定要不要做 cap（每个类最多补多少张）。
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUEUE = ROOT / "feedback" / "review_queue.csv"
FEEDBACK_DIR = ROOT / "feedback"
OUT = ROOT / "datasets" / "feedback"

SPLITS = ("train", "val", "test")
RATIO = (0.7, 0.15, 0.15)
DEFAULT_CAP_PER_CLASS = 200      # 单类最多补多少张，防止某类被抱怨太多而失衡


def read_queue() -> list[dict]:
    if not QUEUE.exists():
        raise SystemExit(f"找不到 {QUEUE}。先跑 python scripts/feedback_review.py")
    return list(csv.DictReader(QUEUE.read_text(encoding="utf-8-sig").splitlines()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, default=QUEUE)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP_PER_CLASS,
                    help="每个类别最多补多少张（防御用户分布偏斜）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [r for r in read_queue() if r.get("keep", "1").strip() not in ("0", "", "false")]
    usable = [r for r in rows if r.get("final_label", "").strip()]
    skipped = len(rows) - len(usable)
    if skipped:
        print(f"[warn] {skipped} 条没有 final_label，已跳过（这类样本标记为问题但还不能训练）")
    # 用户认可预测的那些样本不入训练集 —— 模型已经会了，再喂只是浪费 epoch
    usable = [r for r in usable if r.get("predicted") != r["final_label"]]
    if not usable:
        print("没有可用的新样本（要么还没填 final_label，要么全部与模型预测一致）。")
        return 1

    # ---- cap：防止某一类被灌爆 ----
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_class[r["final_label"]].append(r)
    kept: list[dict] = []
    print("\n[类别配额]")
    for c, lst in sorted(by_class.items()):
        take = lst[: args.cap]
        if len(lst) > args.cap:
            print(f"  {c:<22} {len(lst)} -> 取 {len(take)}（超过 cap={args.cap}）")
        else:
            print(f"  {c:<22} {len(take)}")
        kept.extend(take)

    # ---- 分层切分（在每个类别内部按比例切，保持各类都有 val/test） ----
    rng = random.Random(args.seed)
    assign: dict[str, str] = {}
    for c, lst in by_class.items():
        lst = [r for r in lst if r in kept]
        rng.shuffle(lst)
        n = len(lst)
        n_tr = max(1, round(n * RATIO[0]))
        n_va = max(1, round(n * RATIO[1]))
        for i, r in enumerate(lst):
            s = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            assign[r["request_id"]] = s

    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = []

    missing = 0
    for idx, r in enumerate(kept):
        src_rel = r.get("image", "")
        src_path = FEEDBACK_DIR / src_rel
        if not src_path.exists():
            missing += 1
            continue
        split = assign[r["request_id"]]
        cls = r["final_label"]
        dst_dir = OUT / "images" / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        name = f"fb{idx:05d}_{Path(src_rel).stem[:24]}.jpg"
        dst = dst_dir / name
        shutil.copy2(src_path, dst)
        manifest.append({
            "path": str(dst.relative_to(OUT)),
            "label": cls,
            "class_name": cls,
            "subject_id": f"FB{idx:05d}",     # 反馈样本每张都是独立个体
            "plate_id": "FEEDBACK",
            "date": r.get("ts", "")[:10],
            "frame": 0,
            "split": split,
            "origin": "user_feedback",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    print(f"\n[完成] 新增可用样本 {len(manifest)} 张" + (f"（{missing} 张原图缺失已跳过）" if missing else ""))
    for s in SPLITS:
        print(f"  {s:<6} {sum(1 for m in manifest if m['split'] == s):>4d}")
    empty = [s for s in SPLITS if not any(m["split"] == s for m in manifest)]
    if empty:
        # 样本太少时（比如只有几条反馈）按比例取整必然切不出 val/test。
        # 这一步的数据不能被当作"能证明泛化提升"的证据 —— 只能先看效果、攒够再正式重训。
        print(f"\n[warn] {empty} 集合为空：样本太少，无法在这批数据上做可靠的测试。")
        print("       建议至少攒到每类 10 张以上再跑 retrain，否则你看到的提升没法被证伪。")
    print(f"\n  manifest -> {OUT / 'manifest.csv'}")
    print("\n下一步：python -m train.retrain   （合并原始数据集 + 这批反馈，增量重训）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
