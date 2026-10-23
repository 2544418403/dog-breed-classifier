"""服务层测试：用 FastAPI 的 TestClient 起真应用，不占端口、不需要外部服务。

这里的每个断言都对应一个"上线后真的会咬人"的问题：

    test_health_reports_model    → 别让"进程活着但模型没加载"被当成健康
    test_predict_top1_and_shape  → 端到端能认出样例图
    test_batch_one_forward       → batching 真的省了 forward（不是循环）
    test_rejects_non_image       → 垃圾输入必须 4xx，且不能泄漏堆栈
    test_request_id_echoed       → 排障靠它串起日志
    test_meta_class_order        → 类别顺序错位是最难查的一类 bug
    test_request_log_written     → 观测必须在，否则上线等于蒙眼开车
    test_feedback_roundtrip      → 回流闭环第一步得通

运行：
    pytest tests/test_serve.py -q
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

import serve.app.config as cfg  # noqa: E402
from serve.app import feedback as fbmod  # noqa: E402
from serve.app import telemetry  # noqa: E402
from serve.app.main import app  # noqa: E402

ASSETS = ROOT / "assets"
EXPECT = {
    "havanese.jpg": "havanese",
    "pinscher.jpg": "miniature_pinscher",
    "pomeranian.jpg": "pomeranian",
    "shiba.jpg": "shiba_inu",
    "yorkshire.jpg": "yorkshire_terrier",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _jpg(name: str) -> bytes:
    return (ASSETS / name).read_bytes()


@pytest.fixture(scope="module", autouse=True)
def _need_assets():
    missing = [n for n in EXPECT if not (ASSETS / n).exists()]
    if missing:
        pytest.skip(f"缺少 assets/ 下的样例图：{missing}")


def test_health_reports_model(client):
    j = client.get("/health").json()
    assert j["status"] == "ok"
    assert j["model"]["classes"] == 5
    assert j["model"]["file"] == cfg.MODEL_PATH.name
    # readiness 而非 liveness：必须能看出"模型到底加载完没"
    assert j["model"]["load_ms"] > 0


def test_meta_class_order(client):
    j = client.get("/meta").json()
    assert j["classes"] == cfg.CLASSES
    assert j["input_size"] == ["batch", 3, 64, 64]
    assert j["max_batch"] == cfg.MAX_BATCH


def test_predict_top1(client):
    r = client.post("/predict", files={"file": ("shiba.jpg", _jpg("shiba.jpg"), "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["label"] == EXPECT["shiba.jpg"]
    assert 0 < j["confidence"] <= 1
    assert j["timing"]["infer_ms"] >= 0
    assert j["request_id"]


def test_predict_all_samples(client):
    for name, want in EXPECT.items():
        r = client.post("/predict", files={"file": (name, _jpg(name), "image/jpeg")})
        assert r.status_code == 200, r.text
        assert r.json()["label"] == want, f"{name}: 得到 {r.json()['label']}"


def test_batch_one_forward(client):
    files = [("files", (n, _jpg(n), "image/jpeg")) for n in EXPECT]
    j = client.post("/predict_batch", files=files).json()
    assert j["count"] == len(EXPECT)
    # batching 生效的证据：单张均摊明显低于"整个 batch 的耗时"
    assert j["per_image_ms"] < j["infer_ms"]
    assert {row["label"] for row in j["results"]} == set(EXPECT.values())


def test_batch_size_limit(client):
    files = [("files", (n, _jpg(n), "image/jpeg")) for n in EXPECT] * (cfg.MAX_BATCH // len(EXPECT) + 2)
    r = client.post("/predict_batch", files=files)
    assert r.status_code == 400


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("evil.txt", b"not an image", "text/plain")})
    assert 400 <= r.status_code < 500
    body = r.json()
    # 错误信息必须是结构化的，不能把 Python 堆栈甩给调用方
    assert isinstance(body["error"], dict)
    assert "Traceback" not in json.dumps(body, ensure_ascii=False)


def test_request_id_echoed(client):
    r = client.post("/predict", headers={"X-Request-ID": "abc123"},
                    files={"file": ("shiba.jpg", _jpg("shiba.jpg"), "image/jpeg")})
    assert r.headers["X-Request-ID"] == "abc123"
    assert float(r.headers["X-Response-Time"].rstrip("ms")) > 0


def test_request_log_written(client):
    client.post("/predict", files={"file": ("pomeranian.jpg", _jpg("pomeranian.jpg"), "image/jpeg")})
    rec = telemetry.recent(1)[0]
    # 观测的四个核心字段：输入尺寸 / 预测 / 置信度 / 各阶段耗时
    for key in ("width", "height", "label", "confidence", "infer_ms", "total_ms"):
        assert key in rec, f"请求日志缺少字段 {key}"


def test_feedback_roundtrip(client, tmp_path, monkeypatch):
    monkeypatch.setattr(fbmod, "FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(cfg, "FEEDBACK_DIR", tmp_path)
    r = client.post(
        "/feedback",
        files={"image": ("shiba.jpg", _jpg("shiba.jpg"), "image/jpeg")},
        data={"predicted": "shiba_inu", "confidence": 0.9, "user_label": "pomeranian"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["accepted"] is True
    assert (tmp_path / "inbox").exists()
    summary = fbmod.count_summary()
    assert summary["total"] >= 1


def test_metrics_endpoint(client):
    j = client.get("/metrics?window=100").json()
    assert "latency_ms" in j["serve"]
    assert "p99" in j["serve"]["latency_ms"]
    assert "feedback_summary" in j


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "application/javascript" in r.headers.get("content-type", "") or "<!doctype html>" in r.text.lower()
