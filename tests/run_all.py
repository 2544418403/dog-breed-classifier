"""一次跑全部测试。

为什么需要这个入口：仓库里有三种测试风格，直接用 pytest 会漏掉两个。
    1. tests/test_serve.py   pytest 风格（服务层，只需要服务依赖）
    2. tests/test_data.py    main() 风格（数据管道，需要 torch）
    3. tests/test_evalkit.py main() 风格（评估工具，需要对拍 sklearn）

第 2、3 是"自打印 + 汇总退出码"的脚本 —— 它们的输出是给人读的一张表，
用 pytest 包装反而会丢信息。这里统一驱动三类，缺依赖就跳过而不是假装通过。

用法：
    python tests/run_all.py
    python tests/run_all.py --skip-heavy      # 跳过需要 torch 的部分
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent

PYCACHE_MARKERS = ("__pycache__",)


def _has(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run_script(name: str) -> int:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    r = subprocess.run([sys.executable, str(TESTS / name)], cwd=str(ROOT))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heavy", action="store_true", help="跳过需要 torch/sklearn 的测试")
    args = ap.parse_args()

    failures = []

    print(f"\n{'#' * 70}\n# 服务层（pytest）\n{'#' * 70}")
    if _has("httpx"):
        r = subprocess.run([sys.executable, "-m", "pytest", str(TESTS / "test_serve.py"),
                            "-q", "--no-header"], cwd=str(ROOT))
        failures.append(("test_serve.py", r.returncode))
    else:
        print("跳过：缺 httpx（pip install httpx）")

    if not args.skip_heavy:
        if _has("torch") and _has("torchvision"):
            failures.append(("test_data.py", run_script("test_data.py")))
        else:
            print("\n跳过 test_data.py：需要 torch / torchvision")
        if _has("sklearn"):
            failures.append(("test_evalkit.py", run_script("test_evalkit.py")))
        else:
            print("\n跳过 test_evalkit.py：需要 scikit-learn")
    else:
        print("\n--skip-heavy：跳过 torch / sklearn 相关测试")

    print(f"\n{'#' * 70}\n# 汇总")
    for name, rc in failures:
        print(f"  {'OK  ' if rc == 0 else 'FAIL'}  {name}")
    bad = [n for n, rc in failures if rc != 0]
    print(f"{'#' * 70}\n{'全部通过 ✔' if not bad else '失败：' + ', '.join(bad)}\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
