# ==============================================================================
# 一条命令走完常见动作。PYTHONPATH=src 是 src 布局的代价：包名在 src/ 下。
# ==============================================================================
PY      ?= python
SRC     := src
export PYTHONPATH := $(SRC)

.PHONY: help install install-train run serve smoke bench stress docker-build docker-up test clean

help:                          ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:                       ## 装服务依赖（不含 torch）
	$(PY) -m pip install -r requirements.txt

install-train:                 ## 装训练/评估依赖（torch 等）
	$(PY) -m pip install -r requirements-train.txt

serve:                         ## 启动推理服务 http://127.0.0.1:8000
	$(PY) -m uvicorn serve.app.main:app --host 127.0.0.1 --port 8000 --workers 1

data:                          ## 下载 Oxford-IIIT Pet 并构造 pet5 数据集
	$(PY) -m data.download_pets
	$(PY) -m data.build_manifest

train:                         ## 训练基线模型 -> weights/baseline.pth
	$(PY) -m train.train --exp baseline

export:                        ## 导出 ONNX FP32 + INT8
	$(PY) -m models.export --ckpt weights/baseline.pth --quantize

retrain:                       ## 增量重训（需先跑 feedback_review / feedback_export）
	$(PY) scripts/feedback_review.py
	$(PY) scripts/feedback_export.py
	$(PY) -m train.retrain

smoke:                         ## 端到端冒烟（需服务已启动）
	$(PY) scripts/smoke.py

bench:                         ## 模型延迟基准
	$(PY) scripts/bench_latency.py --all

stress:                        ## 并发压测 QPS / P99
	$(PY) scripts/stress_test.py --concurrency 100 --requests 1000

test:                          ## 跑全部测试（服务层 pytest + 数据/评估脚本）
	$(PY) tests/run_all.py

docker-build:                  ## 构建镜像
	docker build -t dogbreed-serve:1.0.0 .

docker-up:                     ## docker compose 起服务
	docker compose up -d

clean:                         ## 清理运行产物（不动权重和代码）
	rm -rf artifacts/ logs/ feedback/inbox feedback/feedback-*.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
