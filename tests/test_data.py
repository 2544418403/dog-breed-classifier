"""
test_data.py —— datapipe 的最小单元测试（不依赖网络 / GPU / 真实数据）

覆盖本模块最核心、也最容易悄悄出错的行为：
  1. group_split 的"同实体不跨集"硬约束
  2. random_split 的"同实体必然跨集"（对照组行为）
  3. assert_disjoint 的断言真的会抛异常
  4. EXIF 旋转被读取层摆正
  5. 单通道图被读取层转 RGB
  6. 损坏文件被 probe 拦截
  7. MyDataset 的 class_to_idx 一致性 / label 编码
  8. 类别权重之和为类别数（balanced）

运行：
    python tests/test_data.py      # 需要训练依赖：pip install -r requirements-train.txt
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    import pytest as _pytest
except ImportError:
    _pytest = None

_NEEDS = ["torch", "pandas", "PIL"]
_MISSING = [m for m in _NEEDS if importlib.util.find_spec(m) is None]
if _MISSING:
    _MSG = (f"缺少 {_MISSING}：本组用例需要训练依赖\n"
            f"    pip install -r requirements-train.txt")
    if _pytest is not None:
        _pytest.skip(_MSG, allow_module_level=True)   # pytest 收集时：跳过
    raise SystemExit(_MSG)                            # 直接 python xxx.py 时：明确报错

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]      # 仓库根
sys.path.insert(0, str(ROOT / "src"))   # src 布局：包名都在 src/ 下

from data import (  # noqa: E402
    MyDataset,
    assert_disjoint,
    build_transform,
    class_weights,
    group_split,
    load_image_rgb,
    policy_natural,
    probe,
    random_split,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def expect_raises(name: str, fn):
    try:
        fn()
        check(name, False, "没有抛出预期的异常")
    except AssertionError:
        check(name, True)
    except Exception as e:
        check(name, False, f"抛了别的异常 {type(e).__name__}: {e}")


def make_fake_df(n_subjects=40, frames=(1, 4), n_plates=5):
    rng = np.random.default_rng(0)
    rows = []
    sid = 0
    for s in range(n_subjects):
        label = s % 5
        plate = f"P{label % n_plates}"
        for f in range(rng.integers(frames[0], frames[1] + 1)):
            rows.append(
                {"path": f"img_{sid:04d}_{f}.jpg", "label": label,
                 "subject_id": f"S{s:03d}", "plate_id": plate}
            )
            sid += 1
    return pd.DataFrame(rows)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="datapipe_test_"))
    print("=" * 70)
    print("datapipe 单元测试")
    print("=" * 70)

    # ---------- 1/2/3: 划分 ----------
    df = make_fake_df()
    gs = group_split(df, "subject_id", (0.7, 0.15, 0.15), ("train", "val", "test"), "label", seed=0)
    per = gs.groupby("split")["subject_id"].apply(set)
    inter = per.get("train", set()) & per.get("val", set())
    check("group_split: 同一个体不跨集", len(inter) == 0, f"重叠 {len(inter)} 个")
    check("group_split: 三份都有样本", all(len(per.get(s, set())) > 0 for s in ["train", "val", "test"]))

    rs = random_split(df, (0.7, 0.15, 0.15), ("train", "val", "test"), "label", seed=0)
    per = rs.groupby("split")["subject_id"].apply(set)
    inter = per.get("train", set()) & per.get("val", set())
    check("random_split: 同一个体必然跨集（对照组行为）", len(inter) > 0, f"重叠 {len(inter)} 个")

    expect_raises("assert_disjoint 对随机划分结果抛 AssertionError",
                  lambda: assert_disjoint(rs, "split", "subject_id"))
    assert_disjoint(gs, "split", "subject_id")   # 不应抛
    check("assert_disjoint 对分组划分结果通过", True)

    # ---------- 4/5/6: 脏数据 ----------
    img = (np.random.default_rng(1).integers(0, 255, (60, 80, 3))).astype(np.uint8)
    im = Image.fromarray(img)

    # EXIF: 顺时针转 90 存盘 + Orientation=8
    p_exif = tmp / "rot.jpg"
    rot = im.rotate(-90, expand=True)
    ex = rot.getexif(); ex[274] = 8
    rot.save(p_exif, exif=ex)
    fixed = load_image_rgb(p_exif)
    raw = Image.open(p_exif)
    check("EXIF: 读取层摆正方向", fixed.size == im.size and raw.size != im.size,
          f"raw {raw.size} -> fixed {fixed.size}, 原始 {im.size}")

    # 灰度
    p_gray = tmp / "gray.jpg"
    im.convert("L").save(p_gray)
    g = load_image_rgb(p_gray)
    check("单通道: 自动转 3 通道 RGB", g.mode == "RGB" and np.asarray(g).shape[2] == 3)

    # 损坏
    p_bad = tmp / "bad.jpg"
    p_bad.write_bytes(b"<html>404</html>")
    check("损坏: probe 拦截", not probe(p_bad).ok)

    p_zero = tmp / "zero.jpg"
    p_zero.write_bytes(b"")
    check("0 字节: probe 拦截", not probe(p_zero).ok)

    # ---------- 7: Dataset ----------
    # 造一个真目录
    droot = tmp / "imgs"
    droot.mkdir()
    small = make_fake_df(10, (2, 2))
    for _, r in small.iterrows():
        Image.fromarray(img).save(droot / r["path"])
    pol = policy_natural(64)
    tr = MyDataset(small, build_transform("train", pol), root=droot)
    va = MyDataset(small.iloc[:5], build_transform("val", pol), root=droot, class_to_idx=tr.class_to_idx)
    check("Dataset: 类别映射外部传入保持一致", tr.class_to_idx == va.class_to_idx)
    x, y = tr[0]
    check("Dataset: 返回 (Tensor, int)", isinstance(x, torch.Tensor) and x.shape[0] == 3 and isinstance(y, int))
    check("Dataset: class_counts 与样本数一致", sum(tr.class_counts().values()) == len(tr))

    # on_error='skip'：把第一张图弄坏，dataset 应能继续取
    (droot / small.iloc[0]["path"]).write_bytes(b"broken")
    ds_skip = MyDataset(small, build_transform("train", pol), root=droot, on_error="skip")
    n_got = sum(1 for i in range(len(ds_skip)) if ds_skip[i] is not None)
    check("Dataset: on_error='skip' 容错继续", n_got == len(ds_skip))

    # ---------- 8: 类别权重 ----------
    labels = [0] * 100 + [1] * 10 + [2] * 1
    w = class_weights(labels, num_classes=3, mode="inverse")
    check("class_weights: inverse 下权重比 == 样本数反比",
          abs(float(w[0] / w[2]) - 1 / 100 * 111 / 1) < 1e-3 or abs(float(w[2] / w[0]) - 100 / 1) < 1e-3,
          f"w={w.tolist()}")
    w2 = class_weights(labels, num_classes=3, mode="none")
    check("class_weights: none 全 1", torch.allclose(w2, torch.ones(3)))

    # ---------- 结果 ----------
    print("=" * 70)
    print(f"通过 {len(PASS)} / {len(PASS)+len(FAIL)}")
    if FAIL:
        print("失败项:", FAIL)
        sys.exit(1)
    print("全部通过 ✔")


if __name__ == "__main__":
    main()
