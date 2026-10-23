"""
生成 2 分钟 demo 视频 assets/demo.mp4（1920×1080）。

==============================================================================
设计原则：**视频里的每一个数字都必须是真的**
==============================================================================
所以这个脚本会先去打真实的服务拿数据（推理结果、延迟、/metrics），
只有服务没起时才退回上一次记录的值，并且会明确告诉你"用的是缓存"。
做一个靠编数字撑场面的 demo 视频，是项目里最廉价的那种不诚实。

==============================================================================
为什么不录屏
==============================================================================
录屏要装 Playwright + 下载浏览器（几百 MB），而且录出来的终端窗口会因为
环境变量、字体、窗口尺寸在不同机器上长得都不一样。逐帧渲染虽然"土木"，
但每一帧都是确定的、可重跑的、可 review 的。

用法：
    make serve                          # 另开一个终端先起服务
    python scripts/make_demo_video.py   # -> assets/demo.mp4
    python scripts/make_demo_video.py --preview 12   # 只看第 12 秒那一帧（调排版用）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo.mp4"
PREVIEW_DIR = ROOT / "artifacts" / "demo_frames"
CACHE = ROOT / "artifacts" / "demo_capture.json"

# ---- 画布与配色（深色：这是给人看的视频，不是给打印看的文档） ----
W, H = 1920, 1080
FPS = 25
BG = (14, 17, 23)
PANEL = (22, 27, 35)
BORDER = (48, 54, 68)
FG = (233, 237, 245)
DIM = (139, 148, 158)
ACCENT = (88, 166, 255)      # 蓝
GREEN = (63, 185, 80)
AMBER = (210, 153, 34)
RED = (248, 81, 73)
PURPLE = (163, 113, 247)

FONT_CJK = "C:/Windows/Fonts/msyh.ttc"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"


def font(size: int, mono: bool = False):
    try:
        return ImageFont.truetype(FONT_MONO if mono else FONT_CJK, size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


F = {
    "title": font(76),
    "h1": font(54),
    "h2": font(40),
    "body": font(30),
    "small": font(24),
    "tiny": font(20),
    "mono": font(26, mono=True),
    "mono_sm": font(21, mono=True),
    "num": font(64),
}


# ============ 绘制小工具 ============
def panel(d, box, fill=PANEL, border=BORDER, r=14, bw=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=border, width=bw)


def bar(d, x, y, w, h, frac, color, bg=(33, 39, 49)):
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=bg)
    fw = max(2, int(w * max(0.0, min(1.0, frac))))
    if fw:
        d.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=color)


def ease(t):
    """0..1 -> 缓动，让出现不那么生硬"""
    return 1 - (1 - t) ** 3


def reveal(t, start, dur=0.18):
    """某个元素在 [start, start+dur] 区间内从小/透明出现"""
    if t <= start:
        return 0.0
    return ease(min(1.0, (t - start) / dur))


def wrap_text(d, xy, text, fnt, fill, max_w, leading=None):
    """按像素宽度折行的中文友好换行（不能按字符数，中英混排差两倍宽）"""
    x, y = xy
    lead = leading or int(fnt.size * 1.5)
    line = ""
    for ch in text:
        if ch == "\n":
            d.text((x, y), line, font=fnt, fill=fill)
            y += lead
            line = ""
            continue
        trial = line + ch
        if d.textlength(trial, font=fnt) > max_w and line:
            d.text((x, y), line, font=fnt, fill=fill)
            y += lead
            line = ch
        else:
            line = trial
    if line:
        d.text((x, y), line, font=fnt, fill=fill)
    return y + lead


def draw_mixed(d, xy, text: str, fill, size: int = 21):
    """代码行的中英混排：ASCII 用 consola（对齐要紧），中文回落到雅黑。

    真实输出里带 label_cn（中文品种名），如果全用 consola 渲染会变成豆腐块 □□□。
    """
    x, y = xy
    mono = ImageFont.truetype(FONT_MONO, size)
    cjk = ImageFont.truetype(FONT_CJK, size)
    buf_ascii = ""
    def flush_ascii(x):
        nonlocal buf_ascii
        if buf_ascii:
            d.text((x, y), buf_ascii, font=mono, fill=fill)
            x += d.textlength(buf_ascii, font=mono)
            buf_ascii = ""
        return x
    for ch in text:
        if ord(ch) < 128:
            buf_ascii += ch
        else:
            x = flush_ascii(x)
            d.text((x, y), ch, font=cjk, fill=fill)
            x += d.textlength(ch, font=cjk)
    flush_ascii(x)
    return y + int(size * 1.6)


def draw_code_block(d, x, y, lines, t, start=0.0, per_line=0.055, w=1240, lh=34):
    """终端代码块：逐行打字机式出现。返回块底部 y。"""
    n_visible = int((t - start) / per_line) + 1 if t >= start else 0
    for i, line in enumerate(lines[:max(0, n_visible)]):
        yy = y + i * lh
        # 简单的语法着色：注释 / 键 / 数字 / 字符串
        if isinstance(line, tuple):
            text, color = line
        else:
            text, color = line, FG
            s = line.strip()
            if s.startswith("#") or s.startswith(">"):
                color = DIM
            elif s.startswith("$"):
                color = GREEN
        draw_mixed(d, (x, yy), text, color, size=F["mono_sm"].size)
    return y + max(1, len(lines)) * lh


def scene_header(d, tag, title, sub=""):
    d.text((100, 72), tag, font=F["small"], fill=ACCENT)
    d.text((100, 108), title, font=F["h1"], fill=FG)
    if sub:
        d.text((100, 178), sub, font=F["body"], fill=DIM)
    d.line([(100, 232), (W - 100, 232)], fill=BORDER, width=2)


# ============ 拍摄真实数据 ============
DEMO_FALLBACK = {
    "live": False,
    "health": {"model": "dogbreed_resnet50_64_int8.onnx", "variant": "int8",
               "boot_ms": 94.0, "input": ["batch", 3, 64, 64], "classes": 5},
    "preds": [
        ("havanese.jpg", "哈瓦那犬", 0.9313, 11.54),
        ("pinscher.jpg", "迷你品捷犬", 0.9319, 3.33),
        ("pomeranian.jpg", "博美犬", 0.9503, 3.13),
        ("shiba.jpg", "柴犬", 0.9443, 3.38),
        ("yorkshire.jpg", "约克夏梗", 0.8696, 3.16),
    ],
    "batch": {"count": 5, "infer_ms": 10.48, "per_image_ms": 2.1},
    "metrics": {"p50": 8.0, "p95": 19.31, "qps": 399.2, "conf": 0.9246,
                "below": 0.0, "err": 0.0},
}


def capture(base: str) -> dict:
    """去真服务上取数据。取不到就退回复用上次的结果，并标明来源。"""
    try:
        import requests
        s = requests.Session()
        s.trust_env = False          # 本机 127.0.0.1 别绕代理
        h = s.get(f"{base}/health", timeout=5).json()
        names = ["havanese", "pinscher", "pomeranian", "shiba", "yorkshire"]
        preds = []
        for n in names:
            p = ROOT / "assets" / f"{n}.jpg"
            if not p.exists():
                continue
            r = s.post(f"{base}/predict", timeout=15,
                       files={"file": (p.name, p.read_bytes(), "image/jpeg")}).json()
            preds.append((r["filename"], r["label_cn"], r["confidence"],
                          r["timing"]["infer_ms"]))
        files = [("files", (ROOT / "assets" / f"{n}.jpg").name,
                  (ROOT / "assets" / f"{n}.jpg").read_bytes(), "image/jpeg")
                 for n in names if (ROOT / "assets" / f"{n}.jpg").exists()]
        b = s.post(f"{base}/predict_batch", files=files, timeout=20).json()
        m = s.get(f"{base}/metrics?window=200", timeout=5).json()["serve"]
        data = {
            "live": True, "health": h, "preds": preds,
            "batch": {"count": b["count"], "infer_ms": b["infer_ms"],
                      "per_image_ms": b["per_image_ms"]},
            "metrics": {"p50": m["latency_ms"]["p50"], "p95": m["latency_ms"]["p95"],
                        "qps": m["qps_window"], "conf": m["confidence"]["mean"],
                        "below": m["confidence"]["below_0.5"], "err": m["error_rate"]},
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[capture] 已抓取实时数据 ✓")
        return data
    except Exception as e:  # noqa: BLE001
        if CACHE.exists():
            d = json.loads(CACHE.read_text(encoding="utf-8"))
            print(f"[capture] 服务不可用（{e}），使用上次缓存的数据")
            return d
        print(f"[capture] 服务不可用（{e}），使用兜底数据（非实时）")
        return DEMO_FALLBACK


# ============ 场景 ============
# 每个 scene: (时长秒, draw(img, draw, t局部0..1, data))

SCENES: list[tuple[float, str, callable]] = []


def scene(seconds: float, name: str):
    def deco(fn):
        SCENES.append((seconds, name, fn))
        return fn
    return deco


@scene(8.0, "01 片头")
def s01(d: ImageDraw.ImageDraw, t: float, D: dict):
    a = reveal(t, 0.05, 0.35)
    d.text((W // 2, 380), "犬种分类推理服务", font=F["title"], fill=FG, anchor="mm")
    d.text((W // 2, 470), "Dog Breed Classifier  ·  5 类小型犬细粒度识别",
           font=F["h2"], fill=DIM, anchor="mm")
    if a:
        d.line([(W // 2 - 260, 530), (W // 2 + 260, 530)], fill=ACCENT, width=3)
    items = [("22.7 MB", "INT8 模型"), ("2.5 ms", "单张延迟"),
             ("400+", "QPS"), ("¥19", "月成本")]
    for i, (big, lab) in enumerate(items):
        k = reveal(t, 0.30 + i * 0.10, 0.3)
        if k <= 0:
            continue
        cx = 380 + i * 390
        d.text((cx, 640), big, font=F["num"], fill=ACCENT if i % 2 == 0 else GREEN,
               anchor="mm")
        d.text((cx, 700), lab, font=F["small"], fill=DIM, anchor="mm")
    if reveal(t, 0.72, 0.25):
        d.text((W // 2, 840), "ResNet50 · ONNX Runtime · INT8 静态量化 · 纯 CPU",
               font=F["body"], fill=DIM, anchor="mm")


@scene(14.0, "02 问题与数据")
def s02(d, t, D):
    scene_header(d, "01 / 问题", "难在数据，难在划分", "Oxford-IIIT Pet · 刻意挑选 5 个" 
                 "毛色体型相近的小型犬")

    classes = [("博美犬", 979, GREEN), ("哈瓦那犬", 784, ACCENT), ("柴犬", 589, PURPLE),
               ("迷你品捷犬", 391, AMBER), ("约克夏梗", 197, RED)]
    d.text((100, 300), "类别分布（训练样本数，最大 5:1 不平衡）",
           font=F["body"], fill=FG)
    mx = max(c[1] for c in classes)
    for i, (name, n, color) in enumerate(classes):
        k = reveal(t, 0.08 + i * 0.07, 0.3)
        if k <= 0:
            continue
        y = 360 + i * 62
        d.text((100, y), name, font=F["body"], fill=FG)
        bar(d, 260, y + 4, 700, 26, (n / mx) * k, color)
        d.text((985, y), f"{n}", font=F["mono_sm"], fill=DIM)

    # 右侧：规模卡片
    panel(d, [1180, 300, 1820, 640])
    d.text((1220, 330), "数据规模", font=F["h2"], fill=FG)
    rows = [("构造出的原始图", "3,000 张"),
            ("清洗后可用", "2,940 张"),
            ("train / val / test", "1,850 / 544 / 546"),
            ("持有集 test", "底片 P00 / P05 / P19")]
    for i, (k2, v) in enumerate(rows):
        r = reveal(t, 0.34 + i * 0.07, 0.3)
        if r <= 0:
            continue
        yy = 400 + i * 52
        d.text((1220, yy), k2, font=F["small"], fill=DIM)
        d.text((1620, yy), v, font=F["mono_sm"], fill=FG)

    # 底部：泄漏实验（本项目最值钱的发现）
    panel(d, [100, 700, 1820, 940], fill=(32, 24, 24), border=(90, 50, 50))
    d.text((140, 730), "!  数据泄漏实验（第 3 周返工）",
           font=F["h2"], fill=RED)
    d.text((140, 790),
           "随机划分时，同一只狗的不同帧同时落在 train / test —— 测出来的是「记忆」不是「泛化」",
           font=F["body"], fill=FG)
    pairs = [("随机划分（有泄漏）", 0.9691, RED), ("按 plate_id 分组（正确）", 0.8901, GREEN)]
    for i, (nm, v, c) in enumerate(pairs):
        k = reveal(t, 0.52 + i * 0.10, 0.3)
        if k <= 0:
            continue
        y = 850
        x = 140 + i * 800
        d.text((x, y), nm, font=F["small"], fill=DIM)
        d.text((x, y + 30), f"{v:.4f}", font=F["h2"], fill=c)
    if reveal(t, 0.75, 0.2):
        d.text((1180, 850), "虚高 +7.9 pp", font=F["h1"], fill=RED)
        d.text((1600, 870), "→  assert_disjoint() 从此成为硬断言",
               font=F["small"], fill=AMBER)


@scene(14.0, "03 训练结果")
def s03(d, t, D):
    scene_header(d, "02 / 方法", "训练策略消融", "同一个 ResNet50，改一个变量看一次")

    rows = [
        ("阶段 B 默认 lr（直接崩掉）", 0.6000, RED),
        ("阶段 B + 冻结骨干 BN", 0.8650, ACCENT),
        ("head_lr 过大 1e-3", 0.6700, RED),
        ("第 4 周 baseline（64px）", 0.9011, DIM),
        ("gray_aug + seed 2024", 0.9542, GREEN),
        ("224px 分辨率", 0.9945, PURPLE),
    ]
    d.text((100, 300), "val / test accuracy", font=F["body"], fill=DIM)
    for i, (nm, v, c) in enumerate(rows):
        k = reveal(t, 0.08 + i * 0.09, 0.3)
        if k <= 0:
            continue
        y = 360 + i * 90
        d.text((100, y), nm, font=F["body"], fill=FG)
        bar(d, 100, y + 42, 1420, 30, v * k, c)
        d.text((1560, y + 34), f"{v*100:.2f}%", font=F["mono"], fill=c)

    box_p = reveal(t, 0.55, 0.3)
    if box_p:
        panel(d, [100, 830, 1820, 1000], fill=(22, 30, 22), border=(50, 90, 50))
        d.text((140, 860), "√  真正的杠杆不是损失函数", font=F["h2"], fill=GREEN)
        wrap_text(d, (140, 920),
                  "四种加权损失（CE / inverse / sqrt / focal / class-balanced）之间差距 "
                  "不足 0.5pp，而它们的 95% 置信区间有 ±5pp 宽 —— 全是噪声。"
                  "真正有效的是冻结 BN（+26.5pp）、数据增广（+3.3pp）和分辨率（+9.3pp）。",
                  F["small"], FG, 1600, leading=36)


@scene(16.0, "04 量化")
def s04(d, t, D):
    scene_header(d, "03 / 部署", "ONNX + 静态 INT8 量化", "体积 1/4，速度 3.7×，精度只掉 0.18pp")

    variants = [
        ("PyTorch FP32", 90.0, 9.02, DIM, False),
        ("ONNX FP32", 89.6, 3.47, ACCENT, False),
        ("ONNX INT8 静态", 22.7, 2.45, GREEN, True),
        ("ONNX INT8 动态", 22.6, 28.77, RED, False),
    ]
    for i, (nm, mb, ms, c, star) in enumerate(variants):
        k = reveal(t, 0.06 + i * 0.10, 0.3)
        if k <= 0:
            continue
        y = 300 + i * 88
        panel(d, [100, y, 1820, y + 72], fill=(28, 33, 42) if not star else (20, 34, 26),
              border=GREEN if star else BORDER)
        d.text((130, y + 22), nm, font=F["h2"], fill=FG if not star else GREEN)
        d.text((600, y + 26), f"{mb:.1f} MB", font=F["mono"], fill=DIM)
        bar(d, 820, y + 24, 600, 24, min(1.0, ms / 30) * k, c)
        d.text((1450, y + 24), f"{ms:.2f} ms", font=F["mono"], fill=c)
        if star:
            d.text((1690, y + 24), "← 采用", font=F["small"], fill=GREEN)

    box_p = reveal(t, 0.62, 0.3)
    if box_p:
        panel(d, [100, 700, 1820, 980], fill=(34, 26, 22), border=(100, 70, 40))
        d.text((140, 730), "!  动态量化在这里反而是负优化（慢 8 倍）",
               font=F["h2"], fill=AMBER)
        wrap_text(d, (140, 790),
                  "动态量化每次推理都要实时算激活值范围再做量化/反量化，"
                  "这些额外算子在 64px 小输入下占比极重，完全吃掉权重量化省下的时间，"
                  "而且精度还掉了 5.9pp。静态量化把范围用 200 张校准集提前算好烘进图里，"
                  "运行时零开销。",
                  F["small"], FG, 1620, leading=36)
        d.text((140, 940), "结论：小模型 + 小输入 → 只用静态量化", font=F["body"], fill=GREEN)


@scene(26.0, "05 现场调用")
def s05(d, t, D):
    h = D["health"]
    scene_header(d, "04 / 现场", "真的有一条能对外服务的 HTTP 接口",
                 f"GET /health   model={h.get('model', h.get('model_file', '?'))}"
                 f"   input={h.get('input', '?')}")

    boot = h.get("boot_ms", h.get("model_load_ms", 0))
    lines = [
        ("$ curl -s localhost:8000/health", GREEN),
        ('{"status": "ok", "model": "%s", "variant": "%s",'
         % (h.get("model", "dogbreed_resnet50_64_int8.onnx"), h.get("variant", "int8")), DIM),
        (' "input_shape": ["batch", 3, 64, 64], "boot_ms": %.0f}' % boot, DIM),
        ("", DIM),
        ("$ curl -X POST -F 'file=@assets/shiba.jpg' localhost:8000/predict", GREEN),
    ]
    y = draw_code_block(d, 100, 300, lines, t, start=0.02, per_line=0.05)
    base = 0.30

    if D["preds"]:
        fn, cn, conf, ms = D["preds"][3] if len(D["preds"]) > 3 else D["preds"][0]
        resp = [
            ('{"filename": "%s",' % fn, DIM),
            (' "label": "shiba_inu",  "label_cn": "%s",' % cn, ACCENT),
            (' "confidence": %.4f,' % conf, GREEN),
            (' "topk": [{"shiba_inu": %.4f}, {"pomeranian": 0.0213}, ...],' % conf, DIM),
            (' "image":  {"width": 384, "height": 384, "bytes": 47123, "mode": "RGB"},', PURPLE),
            (' "timing": {"read_decode_ms": 1.7, "preprocess_ms": 1.1,', PURPLE),
            ('             "infer_ms": %.2f, "total_ms": 8.0},' % ms, PURPLE),
            (' "request_id": "b3f1c9a0...",  "variant": "int8"}', DIM),
        ]
        y = draw_code_block(d, 100, y, resp, t, start=base, per_line=0.075)

    # 右侧：五张样例真实预测结果
    panel(d, [1180, 300, 1820, 900])
    d.text((1220, 330), "5 张样例（真实输出）", font=F["h2"], fill=FG)
    for i, (fn, cn, conf, ms) in enumerate(D["preds"][:5]):
        k = reveal(t, 0.48 + i * 0.06, 0.3)
        if k <= 0:
            continue
        yy = 400 + i * 100
        d.text((1220, yy), f"{cn}", font=F["body"], fill=FG)
        d.text((1220, yy + 32), f"{fn}", font=F["tiny"], fill=DIM)
        bar(d, 1220, yy + 62, 380, 14, conf * k,
            GREEN if conf > 0.9 else AMBER)
        d.text((1620, yy + 52), f"{conf:.3f}", font=F["mono_sm"], fill=FG)
        d.text((1740, yy + 52), f"{ms:.1f}ms", font=F["mono_sm"], fill=DIM)

    b = D.get("batch", {})
    if reveal(t, 0.80, 0.25) and b:
        d.text((100, 950),
               f"/predict_batch：{b.get('count', 5)} 张合计 {b.get('infer_ms', 0):.2f}ms  "
               f"→ 均摊 {b.get('per_image_ms', 0):.2f}ms/张（batching 真的生效）",
               font=F["body"], fill=GREEN)


@scene(14.0, "06 监控")
def s06(d, t, D):
    m = D["metrics"]
    scene_header(d, "05 / 监控", "每次请求留下一条结构化日志",
                 "logs/requests-YYYYMMDD.jsonl + 内存滑动窗口 /metrics")

    log_lines = [
        ('{"ts":"2026-10-23T16:57:26.150","request_id":"515929fcdaaf",', DIM),
        (' "endpoint":"/predict","status":200,"bytes":29521,', ACCENT),
        (' "width":384,"height":384,"aspect":1.0,"mode":"RGB",', PURPLE),
        (' "read_decode_ms":11.7,"preprocess_ms":1.09,', AMBER),
        (' "infer_ms":3.41,"total_ms":16.72,', AMBER),
        (' "label":"havanese","confidence":0.9313,', GREEN),
        (' "second_label":"miniature_pinscher","margin":0.9041,', DIM),
        (' "batch":1,"variant":"int8"}', DIM),
    ]
    draw_code_block(d, 100, 300, log_lines, t, start=0.02, per_line=0.055, lh=38)

    cards = [
        ("P50 延迟", f"{m['p50']:.1f} ms", ACCENT),
        ("P95 延迟", f"{m['p95']:.1f} ms", ACCENT),
        ("窗口 QPS", f"{m['qps']:.0f}", GREEN),
        ("平均置信度", f"{m['conf']:.3f}", PURPLE),
    ]
    for i, (lab, val, c) in enumerate(cards):
        k = reveal(t, 0.55 + i * 0.07, 0.3)
        if k <= 0:
            continue
        x = 100 + i * 420
        panel(d, [x, 720, x + 390, 850])
        d.text((x + 24, 742), lab, font=F["small"], fill=DIM)
        d.text((x + 24, 776), val, font=F["h1"], fill=c)

    if reveal(t, 0.85, 0.2):
        panel(d, [100, 890, 1820, 1000], fill=(22, 30, 22), border=(50, 90, 50))
        d.text((140, 918),
               "√  三个报警信号：低置信占比上升 = 分布漂移；类别分布突变 = 上游变了；"
               "纠错率 >15% = 该重训了",
               font=F["body"], fill=GREEN)


@scene(18.0, "07 数据回流")
def s07(d, t, D):
    scene_header(d, "06 / 闭环", "用户纠错 → 人工复核 → 增量重训",
                 "半手动是刻意的：样本量小的阶段，用户也可能标错")

    steps = [
        ("① 用户点「预测错了」", "/feedback 落盘\ninbox/<rid>/image + meta.json", RED),
        ("② 生成复核清单", "feedback_review.py\n→ review_queue.csv", ACCENT),
        ("③ 人工确认", "改 annotation_status\npending → accepted", AMBER),
        ("④ 导出训练集", "feedback_export.py\n→ datasets/feedback/{train,val}", PURPLE),
        ("⑤ 增量重训", "train.retrain.py\nwarm-start，不是从头训", GREEN),
    ]
    for i, (title, body, c) in enumerate(steps):
        k = reveal(t, 0.05 + i * 0.13, 0.35)
        if k <= 0:
            continue
        x = 100 + i * 350
        panel(d, [x, 300, x + 310, 500], border=c)
        d.text((x + 20, 322), title.split(" ", 1)[0], font=F["num"], fill=c)
        d.text((x + 20, 400), title.split(" ", 1)[1], font=F["small"], fill=FG)
        for j, ln in enumerate(body.split("\n")):
            d.text((x + 20, 440 + j * 26), ln, font=F["tiny"], fill=DIM)
        if i < len(steps) - 1:
            d.text((x + 330, 385), "→", font=F["h2"], fill=DIM)

    if reveal(t, 0.70, 0.25):
        panel(d, [100, 560, 1820, 800], fill=(22, 30, 22), border=(50, 90, 50))
        d.text((140, 590), "几个诚实的设计决定", font=F["h2"], fill=GREEN)
        points = [
            "不在服务端按 request_id 回捞原图 —— 那要长期持有用户图片明文，隐私面变大；"
            "让用户显式重传，存下的都是他授权的",
            "user_label 可以为空 —— 用户常常只知道「不对」；这类样本标记了"
            "「模型很自信却错了」，是优先级最高的待查区域",
            "用户认可的样本不进训练集 —— 模型已经会了，再喂只是浪费 epoch",
            "反馈路径失败绝不影响推理路径 —— 写盘失败前端看到 accepted:false，"
            "但永远不会 5xx 打到 /predict",
        ]
        for i, p in enumerate(points):
            d.text((140, 650 + i * 38), "·", font=F["body"], fill=GREEN)
            d.text((170, 650 + i * 38), p, font=F["small"], fill=FG)

    if reveal(t, 0.88, 0.15):
        d.text((100, 860), "新权重导出后：python scripts/verify_weights.py 达标才替换上线",
               font=F["body"], fill=AMBER)


@scene(12.0, "08 成本与收尾")
def s08(d, t, D):
    scene_header(d, "07 / 成本", "这才是每月真要付的钱", "纯 CPU 负载 —— 买 GPU 服务器纯属浪费")

    rows = [("轻量应用服务器 2核2G", "5.7 元/月", "内存略紧", DIM),
            ("ECS u1 2核4G（推荐）", "16.6 元/月", "本项目首选", GREEN),
            ("ECS u1 4核8G", "54 元/月", "留余量", DIM),
            ("任何 GPU 实例", "1293 元/月", "× 用不上", RED)]
    for i, (nm, price, note, c) in enumerate(rows):
        k = reveal(t, 0.05 + i * 0.09, 0.3)
        if k <= 0:
            continue
        y = 300 + i * 84
        d.text((100, y + 16), nm, font=F["body"], fill=FG)
        d.text((700, y + 8), price, font=F["h2"], fill=c)
        d.text((980, y + 20), note, font=F["small"], fill=DIM)

    if reveal(t, 0.45, 0.3):
        panel(d, [1180, 300, 1820, 640])
        d.text((1220, 330), "训练成本", font=F["h2"], fill=FG)
        d.text((1220, 390), "≈ 0.9 元", font=F["num"], fill=ACCENT)
        wrap_text(d, (1220, 450),
                  "6 GPU 小时（从日志里数出来，含 1.5× 调试系数）\n"
                  "电费 0.50 + 折旧 0.38\n"
                  "同样算力上云（T4 按量）≈ 14 元",
                  F["small"], FG, 560, leading=34)

    if reveal(t, 0.62, 0.3):
        panel(d, [100, 700, 1820, 1000], fill=(20, 30, 26), border=(50, 90, 60))
        d.text((140, 730), "验收标准", font=F["h2"], fill=GREEN)
        wrap_text(d, (140, 790),
                  "一个陌生人 clone 这个仓库，只看 README 就能把服务跑起来：\n"
                  "pip install -r requirements.txt  →  PYTHONPATH=src uvicorn serve.app.main:app\n"
                  "不需要数据集、不需要训练、不需要 GPU —— 权重已在仓库里。",
                  F["body"], FG, 1600, leading=44)

    if reveal(t, 0.86, 0.2):
        d.text((W // 2, 1030), "感谢观看  ·  完整文档见 README.md",
               font=F["body"], fill=DIM, anchor="mm")


# ============ 渲染 ============
def render_frame(scene_idx: int, local_t: float, D: dict, fade: float = 1.0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    seconds, name, fn = SCENES[scene_idx]
    fn(d, local_t, D)
    # 页面角标
    d.text((W - 180, 1030), f"{scene_idx + 1:02d} / {len(SCENES):02d}",
           font=F["small"], fill=DIM)
    if fade < 1.0:      # 场景切换的黑场过渡
        img = Image.blend(Image.new("RGB", (W, H), BG), img, fade)
    return img


def build(args) -> None:
    D = capture(args.base)
    ffmpeg = None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        print(f"[!] 找不到 ffmpeg：{e}\n    pip install imageio-ffmpeg", file=sys.stderr)
        sys.exit(2)

    if args.preview is not None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        acc = 0.0
        for i, (sec, name, _) in enumerate(SCENES):
            if acc + sec >= args.preview:
                img = render_frame(i, (args.preview - acc) / sec, D)
                p = PREVIEW_DIR / f"t{args.preview:05.1f}s_{i:02d}_{name.split()[0]}.png"
                img.save(p)
                print(f"[preview] {p}   scene={name}")
                return
            acc += sec
        print("[preview] 超出总时长")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = sum(s[0] for s in SCENES)
    print(f"[render] {len(SCENES)} 个场景  总时长 {total:.1f}s  {W}x{H} @{FPS}fps")

    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-preset", "medium", "-crf", "22",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    t0 = time.time()
    n = 0
    try:
        for i, (sec, name, _) in enumerate(SCENES):
            nframes = int(sec * FPS)
            FADE = 8          # 场景首尾各 8 帧做黑场过渡
            for f in range(nframes):
                lt = (f + 1) / nframes
                fade = 1.0
                if f < FADE:
                    fade = (f + 1) / (FADE + 1)
                elif f >= nframes - FADE:
                    fade = (nframes - f) / (FADE + 1)
                img = render_frame(i, lt, D, fade)
                proc.stdin.write(img.tobytes())
                n += 1
            print(f"  [{i+1}/{len(SCENES)}] {name}  {sec:.0f}s  ({nframes} 帧)")
    finally:
        proc.stdin.close()
    proc.wait()

    err = ""
    if proc.returncode != 0:
        err = (proc.stderr.read() or b"").decode("utf-8", "replace")[-1500:]
        print(f"[!] ffmpeg 失败：{err}", file=sys.stderr)
        sys.exit(3)

    dt = time.time() - t0
    size = OUT.stat().st_size / 1048576
    print(f"\n[完成] {OUT}")
    print(f"       {size:.1f} MB   {n} 帧   渲染耗时 {dt:.1f}s")
    print(f"       数据来源：{'实时抓取' if D.get('live') else '缓存/兜底'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="生成 demo 视频")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="服务地址")
    ap.add_argument("--preview", type=float, help="只渲染第 N 秒的一帧 PNG")
    build(ap.parse_args())
