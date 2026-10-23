"""
scripts/03_report.py —— 把 benchmark.json 变成给人看的结论

==============================================================================
产出
==============================================================================
    figures/fig_size_vs_latency.png   体积 vs 延迟（一眼看出谁在第几象限）
    figures/fig_batch_scaling.png     batch size → 吞吐曲线（看懂 batch 聚合的价值）
    figures/fig_thread_scaling.png    线程数 → 延迟（量化在不同线程下的收益差异）
    figures/fig_tradeoff.png          三维归一化的权衡图
    实验报告.md                        本周主报告：数据 + 结论 + 优化权衡表
    实验报告.html                      同内容的网页版（图表内联）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 中文字体：Windows 用微软雅黑
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = {
    "torch-fp32": "#9b9b9b",
    "torch-int8-dyn": "#d98c3f",
    "onnx-fp32": "#3f7fb5",
    "onnx-int8-dyn": "#c0504d",
    "onnx-int8-static": "#2e9e5b",
}
CN_NAME = {
    "torch-fp32": "PyTorch FP32",
    "torch-int8-dyn": "Torch 动态量化",
    "onnx-fp32": "ONNX FP32",
    "onnx-int8-dyn": "ONNX 动态量化",
    "onnx-int8-static": "ONNX 静态量化",
}


def load():
    return json.loads((OUT / "benchmark.json").read_text(encoding="utf-8"))


def pick(rows, threads=1):
    return [r for r in rows if r["threads"] == threads]


def fig_size_latency(rows, threads=1):
    r = pick(rows, threads)
    offsets = {  # 手工微调标签位置，避免右下角三点重叠
        "torch-fp32": (12, -16), "torch-int8-dyn": (14, 6), "onnx-fp32": (12, -6),
        "onnx-int8-dyn": (12, 0), "onnx-int8-static": (12, -6),
    }
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=140)
    for d in r:
        ax.scatter(d["size_mb"], d["latency"]["p50_ms"], s=190, color=COLORS.get(d["name"], "#666"),
                   alpha=.85, edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{CN_NAME.get(d['name'], d['name'])}\n{d['size_mb']}MB / {d['latency']['p50_ms']}ms",
                    (d["size_mb"], d["latency"]["p50_ms"]), textcoords="offset points",
                    xytext=offsets.get(d["name"], (12, -6)), fontsize=9.5, color="#333")
    ax.set_xlabel("模型体积 (MB)", fontsize=11)
    ax.set_ylabel("单次推理延迟 P50 (ms)", fontsize=11)
    ax.set_title(f"体积 vs 延迟（CPU threads={threads}）· 左下角最优", fontsize=13, pad=12)
    ax.grid(alpha=.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = FIG / "fig_size_vs_latency.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_batch_scaling(rows, threads=1):
    r = pick(rows, threads)
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=140)
    for d in r:
        b = sorted(d["batch"].items(), key=lambda kv: int(kv[0]))
        xs = [int(k) for k, _ in b]
        ys = [v["img_per_s"] for _, v in b]
        ax.plot(xs, ys, marker="o", lw=2.2, ms=7, color=COLORS.get(d["name"], "#666"),
                label=CN_NAME.get(d["name"], d["name"]))
        ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -4), fontsize=9, color=COLORS.get(d["name"], "#666"))
    ax.set_xlabel("batch size", fontsize=11)
    ax.set_ylabel("吞吐 (images/s)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "4", "8", "16", "32"])
    ax.set_title(f"batch 聚合的价值（CPU threads={threads}）", fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = FIG / "fig_batch_scaling.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_thread(rows):
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=140)
    names = []
    for d in rows:
        if d["name"] not in names:
            names.append(d["name"])
    x = np.arange(len(names))
    dfp = pick(rows, 1)
    if not dfp:
        return None
    w = 0.35
    for i, th in enumerate([1, 4]):
        r = {d["name"]: d["latency"]["p50_ms"] for d in pick(rows, th)}
        vals = [r.get(n, 0) for n in names]
        bars = ax.bar(x + (i - .5) * w, vals, w, label=f"threads={th}",
                      color=["#b9c6d4", "#4a7fa8"][i], edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                    ha="center", fontsize=8.5, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([CN_NAME.get(n, n) for n in names], fontsize=10)
    ax.set_ylabel("P50 延迟 (ms)", fontsize=11)
    ax.set_title("线程数对延迟的影响", fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=.25, axis="y", linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = FIG / "fig_thread_scaling.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gpu(gpu):
    if not gpu or not gpu.get("runs"):
        return None
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=140)
    palette = ["#7a5aa8", "#c2571f", "#1f7a63"]
    for i, run in enumerate(gpu["runs"]):
        xs = [r["bs"] for r in run["rows"]]
        ys = [r["img_per_s"] for r in run["rows"]]
        ax.plot(xs, ys, marker="s", lw=2.2, ms=7, color=palette[i % len(palette)], label=run["name"])
        ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -4), fontsize=9, color=palette[i % len(palette)])
    ax.set_xlabel("batch size", fontsize=11)
    ax.set_ylabel("吞吐 (images/s)", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32])
    ax.set_xticklabels(["1", "8", "32"])
    ax.set_title(f"GPU 吞吐：{gpu.get('device', 'cuda')}", fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = FIG / "fig_gpu_scaling.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def print_table(rows, threads=1):
    r = pick(rows, threads)
    base = next((d for d in r if d["name"] == "torch-fp32"), r[0])
    print(f"\n{'形态':<18}{'体积MB':>8}{'加载ms':>8}{'P50ms':>8}{'P95ms':>8}"
          f"{'img/s':>9}{'top1':>8}{'翻转率':>8}")
    print("-" * 80)
    for d in r:
        print(f"{d['name']:<18}{d['size_mb']:>8.1f}{d['load_ms']:>8.0f}"
              f"{d['latency']['p50_ms']:>8.2f}{d['latency']['p95_ms']:>8.2f}"
              f"{d['batch']['1']['img_per_s']:>9.0f}{d['acc']:>8.4f}"
              f"{d.get('flip_rate', 0):>8.2%}")
    print("-" * 80)


def main() -> int:
    data = load()
    rows = data["cpu"]
    gpu = data.get("gpu")

    threads_main = 1
    p1 = fig_size_latency(rows, threads_main)
    p2 = fig_batch_scaling(rows, threads_main)
    p3 = fig_thread(rows)
    p4 = fig_gpu(gpu)

    for th in sorted({r["threads"] for r in rows}):
        print(f"\n### threads={th}")
        print_table(rows, th)

    if gpu:
        print(f"\n### GPU {gpu.get('device')}")
        for run in gpu["runs"]:
            line = f"  {run['name']:<20}"
            for rw in run["rows"]:
                line += f"  bs{rw['bs']:<3}={rw['img_per_s']:>7.0f} img/s"
            print(f"{line}  acc={run['acc']}")

    print("\n图表：")
    for p in (p1, p2, p3, p4):
        if p:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
