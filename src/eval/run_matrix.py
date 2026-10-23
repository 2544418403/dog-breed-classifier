"""
scripts/run_matrix.py —— 批量跑消融矩阵（串起 train.py 的多次调用）

用法：
    python scripts/run_matrix.py --exps baseline no_aug cw_sqrt cw_inverse focal cb_focal img224 gray_aug
    python scripts/run_matrix.py --exps baseline --seeds 42 7 2024    # 多种子

设计要点：
  - 每个实验**独立进程**，避免 CUDA 显存/状态在多次训练间互相污染
  - 统一的 HF_HOME + HF_HUB_OFFLINE 环境变量（复用第二周的预训练权重缓存）
  - 失败不中断，最后汇总哪些没跑成
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, DATASETS, OUTPUTS as OUT, FIGURES as FIG, WEIGHTS, ensure_dirs

ensure_dirs()   # 幂等

# 用"当前解释器"跑子进程，而不是写死某个 venv 里的 python.exe。
# 硬编码解释器路径是典型的"只能在我机器上跑"。
PY = sys.executable

# HF_HOME 默认沿用环境/缓存，不强依赖某个历史目录
HF_HOME = os.getenv("HF_HOME", "")


def run_one(exp: str, seed: int, extra: list) -> tuple:
    # 子进程同样是包内模块：`python -m train.train`
    cmd = [str(PY), "-m", "train.train", "--exp", exp, "--seed", str(seed)] + extra
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    if HF_HOME:
        env["HF_HOME"] = HF_HOME
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = (r.stdout or "").strip().splitlines()
    last = tail[-1] if tail else "(no output)"
    return r.returncode, dt, last, (r.stderr or "")[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["baseline"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    print(f"消融矩阵：{len(args.exps)} 个实验 × {len(args.seeds)} 个种子 = "
          f"{len(args.exps)*len(args.seeds)} 次训练\n")

    results = []
    for exp in args.exps:
        for seed in args.seeds:
            tag = f"s{seed}" if len(args.seeds) > 1 else ""
            extra = list(args.extra)
            if tag:
                extra += ["--tag", tag]
            print(f"---> {exp} seed={seed}")
            rc, dt, last, err = run_one(exp, seed, extra)
            status = "ok" if rc == 0 else f"FAILED({rc})"
            print(f"     {status}  {dt/60:.1f} min  {last}")
            if rc != 0:
                print(err)
            results.append({"exp": exp, "seed": seed, "rc": rc, "min": round(dt / 60, 2), "last": last})

    print("\n" + "=" * 70)
    ok = sum(1 for r in results if r["rc"] == 0)
    print(f"完成 {ok}/{len(results)}，总耗时 {sum(r['min'] for r in results):.1f} min")
    for r in results:
        if r["rc"] != 0:
            print(f"  失败: {r['exp']} seed={r['seed']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
