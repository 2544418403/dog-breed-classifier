"""端到端冒烟测试：跑完这一条命令，你就知道这个仓库是不是真的活着。

用法：
    python scripts/smoke.py                       # 默认 http://127.0.0.1:8000
    python scripts/smoke.py --url http://ip:8000  # 测远程部署

设计原则：**测试要回答"能不能用"，而不是"代码有没有报错"**。
所以每一步检查的都是用户真正依赖的东西：服务在线吗？模型认得出狗吗？
结果对吗？耗时合理吗？反馈写得进去吗？日志留下了痕迹吗？

退出码：0 = 全通过，1 = 有失败。可以挂进 CI。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
sys.path.insert(0, str(ROOT / "src"))

# 类名 -> 期望预测。之所以敢写死期望值：这五张图来自数据集，
# 模型在它们上的预测是稳定且已知的（见 README 结果表）。冒烟测试的职责
# 就是发现"变了的那一刻"，所以硬断言比对捣糨糊的 exists() 检查有用得多。
EXPECT = {
    "havanese.jpg": "havanese",
    "pinscher.jpg": "miniature_pinscher",
    "pomeranian.jpg": "pomeranian",
    "shiba.jpg": "shiba_inu",
    "yorkshire.jpg": "yorkshire_terrier",
}

# trust_env=False：CI / 公司内网常配了 HTTP_PROXY，而本机 127.0.0.1
# 显然不该绕一圈代理。requests 默认会读环境变量里的代理。
SESSION = requests.Session()
SESSION.trust_env = False

_pass = fail = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _pass, fail
    if ok:
        _pass += 1
        print(f"  [PASS] {name}{('  ' + detail) if detail else ''}")
    else:
        fail += 1
        print(f"  [FAIL] {name}  {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--latency-budget-ms", type=float, default=50.0,
                    help="单请求 P50 预算；超过说明机器太忙或量化没生效")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    print(f"\n=== 冒烟测试 {base} ===\n")

    # ---- 1. health ----
    print("[1] 健康检查 /health")
    try:
        h = SESSION.get(f"{base}/health", timeout=10).json()
    except Exception as e:  # noqa: BLE001
        check("/health 可达", False, str(e))
        return 1
    check("状态 ok", h.get("status") == "ok")
    check("已加载模型", h.get("model", {}).get("file") is not None,
          f"file={h.get('model', {}).get('file')} variant={h.get('model', {}).get('variant')}")
    check("冷启动在 5s 内", h.get("model", {}).get("load_ms", 1e9) < 5000,
          f"load_ms={h.get('model', {}).get('load_ms')}")

    # ---- 2. meta ----
    print("\n[2] 自描述 /meta")
    m = SESSION.get(f"{base}/meta", timeout=10).json()
    check("类别数 = 5", len(m.get("classes", [])) == 5, str(m.get("classes")))
    check("input_shape 是 (batch,3,64,64)", m.get("input_size") == ["batch", 3, 64, 64],
          str(m.get("input_size")))

    # ---- 3. 单张推理 ----
    print("\n[3] 单张推理 /predict")
    lats = []
    for fn, want in EXPECT.items():
        p = ASSETS / fn
        if not p.exists():
            check(f"{fn} 存在", False, f"缺少样例图 {p}")
            continue
        t0 = time.perf_counter()
        r = SESSION.post(f"{base}/predict", timeout=30,
                         files={"file": (fn, p.read_bytes(), "image/jpeg")})
        dt = (time.perf_counter() - t0) * 1000
        lats.append(dt)
        if not check(f"{fn} -> HTTP 200", r.status_code == 200, r.text[:160]):
            continue
        j = r.json()
        check(f"{fn} 预测正确", j["label"] == want,
              f"得到 {j['label']} ({j['confidence']:.3f})  期望 {want}  "
              f"infer={j['timing']['infer_ms']:.2f}ms")
        check(f"{fn} 返回了 request_id", bool(j.get("request_id")), j.get("request_id", ""))

    p50 = sorted(lats)[len(lats) // 2] if lats else 0
    check(f"端到端 P50 在预算内（<{args.latency_budget_ms}ms）",
          p50 < args.latency_budget_ms, f"P50={p50:.1f}ms")

    # ---- 4. 批量推理 ----
    print("\n[4] 批量推理 /predict_batch")
    files = [("files", (fn, (ASSETS / fn).read_bytes(), "image/jpeg")) for fn in EXPECT
             if (ASSETS / fn).exists()]
    rb = SESSION.post(f"{base}/predict_batch", timeout=30, files=files)
    if check("HTTP 200", rb.status_code == 200, rb.text[:160]):
        j = rb.json()
        check(f"返回 {len(files)} 条", j.get("count") == len(files), str(j.get("count")))
        check("单张均摊 < 单次推理耗时（batching 真的生效了）",
              float(j.get("per_image_ms", 1e9)) < float(j.get("infer_ms", 0)),
              f"per_image={j.get('per_image_ms')}ms infer={j.get('infer_ms')}ms")

    # ---- 5. 防护 ----
    print("\n[5] 输入防护")
    bad = SESSION.post(f"{base}/predict", timeout=30,
                       files={"file": ("evil.txt", b"not an image at all", "text/plain")})
    check("非图片被拒（4xx）", 400 <= bad.status_code < 500, f"status={bad.status_code}")
    check("错误信息结构化", isinstance(bad.json().get("error"), dict), bad.text[:160])

    # ---- 6. 反馈闭环 ----
    print("\n[6] 反馈闭环 /feedback")
    rf = SESSION.post(
        f"{base}/feedback", timeout=30,
        files={"image": ("shiba.jpg", (ASSETS / "shiba.jpg").read_bytes(), "image/jpeg")},
        data={"predicted": "shiba_inu", "confidence": 0.9, "user_label": "pomeranian",
              "note": "smoke test", "src": "smoke"},
    )
    if check("HTTP 200", rf.status_code == 200, rf.text[:200]):
        check("accepted = true", rf.json().get("accepted") is True, str(rf.json().get("record", {}))[:160])

    # ---- 7. 可观测性 ----
    print("\n[7] 可观测性 /metrics")
    met = SESSION.get(f"{base}/metrics?window=200", timeout=10).json()
    s = met.get("serve", {})
    check("有请求样本", s.get("requests_in_window", 0) >= 5, str(s.get("requests_in_window")))
    check("记录了延迟分位", "p50" in s.get("latency_ms", {}), str(s.get("latency_ms")))
    check("记录了输入尺寸", s.get("input", {}).get("avg_width", 0) > 0, str(s.get("input")))
    check("记录了类别分布", len(s.get("label_distribution", {})) > 0, str(s.get("label_distribution")))
    check("反馈计数 > 0", met.get("counters", {}).get("feedback", 0) > 0,
          str(met.get("feedback_summary")))

    print(f"\n=== 通过 {_pass} / 失败 {fail} ===")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
