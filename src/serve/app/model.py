"""模型的"包装层"：加载、预处理、推理。

这一层存在的唯一目的：**让 main.py 里看不到任何模型细节**。
接口层只认识 `clf.predict(pil_image) -> (label, conf)`，
换模型（resnet50 → mobilenet）、换运行时（onnxruntime → torch）
都只改这一个文件，HTTP 那部分一行不动。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from common.preprocess import preprocess_pil, softmax
from serve.app.config import (
    CLASS_CN,
    CLASSES,
    IMG_SIZE,
    MAX_IMAGE_PIXELS,
    MEAN,
    ORT_INTRA_OP_THREADS,
    RESIZE_TO,
    STD,
)

# 让解码炸弹在"吃掉内存之前"就抛错（配合 security.py 里捕获 DecompressionBombError）
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass
class Prediction:
    index: int
    label: str
    label_cn: str
    confidence: float
    probs: list[float]


class OnnxClassifier:
    """一次加载，反复使用。

    这个类**必须**只被实例化一次（见 main.py 的 lifespan）。
    每次请求 new 一个，就是重新读几十 MB 权重 + 重建执行计划，
    QPS 会从几百掉到个位数 —— 这是服务化里最常见的性能错误。
    """

    def __init__(self, onnx_path: Path):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = ORT_INTRA_OP_THREADS
        so.inter_op_num_threads = 1
        # log_severity=3 关掉 ORT 自己的 INFO 日志，不然每请求一行刷屏
        so.log_severity_level = 3

        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.load_ms = (time.perf_counter() - t0) * 1000

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape  # ['batch', 3, 64, 64]
        self.classes = CLASSES
        self.path = Path(onnx_path)

        self.warmup()

    # ------------------------------------------------------------------ #
    def warmup(self, n: int = 2) -> float:
        """预热：跑几张假图。

        不做预热的后果：第一个真实请求会替所有人付"一次性成本"
        —— 内存分配器初始化、线程池建立、图优化后的 kernel 准备。
        实测能把首请求延迟从几百毫秒压到几十毫秒。
        压测脚本也必须先 warmup 再计时，否则 P99 会被这一个冷请求污染。
        """
        t0 = time.perf_counter()
        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for _ in range(n):
            self.sess.run([self.output_name], {self.input_name: dummy})
        self.warmup_ms = (time.perf_counter() - t0) * 1000
        return self.warmup_ms

    # ------------------------------------------------------------------ #
    def preprocess(self, img: Image.Image) -> np.ndarray:
        """PIL 图 -> (3, H, W) float32 CHW。

        必须和训练时的 val/test 预处理**逐位一致**，否则精度会静默下降。
        这里直接复用 common.preprocess.preprocess_pil —— 量化校准、
        离线评估和在线推理用的是同一份代码，不可能漂移。
        """
        return preprocess_pil(img, IMG_SIZE, RESIZE_TO, MEAN, STD)

    # ------------------------------------------------------------------ #
    def predict_array(self, batch: np.ndarray) -> tuple[list[Prediction], float]:
        """batch: (N,3,H,W) float32 -> (预测列表, 本次推理耗时 ms)。

        注意这里是**一次 forward 跑 N 张**，不是 for 循环跑 N 次。
        这就是 /predict_batch 存在的全部理由（导出时 dynamic_axes 开了 batch 维）。
        """
        t0 = time.perf_counter()
        logits = self.sess.run([self.output_name], {self.input_name: batch})[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        probs = softmax(logits, axis=1)
        idx = probs.argmax(axis=1)
        out = []
        for i, k in enumerate(idx):
            name = self.classes[int(k)]
            out.append(
                Prediction(
                    index=int(k),
                    label=name,
                    label_cn=CLASS_CN.get(name, name),
                    confidence=float(probs[i, k]),
                    probs=[round(float(p), 6) for p in probs[i]],
                )
            )
        return out, infer_ms

    def predict_one(self, img: Image.Image) -> tuple[Prediction, float, float]:
        t0 = time.perf_counter()
        x = self.preprocess(img)[None, ...]
        pre_ms = (time.perf_counter() - t0) * 1000
        preds, infer_ms = self.predict_array(x)
        return preds[0], pre_ms, infer_ms
