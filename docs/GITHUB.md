# 推上 GitHub & 简历描述

## 一键推送

仓库已完成 `git init` 和首次提交。推送只需三步：

```bash
# ① 在 GitHub 上建一个空仓库（不要勾 "Add a README"，我们要推自己的）

# ② 关联远端
git remote add origin https://github.com/<YOUR_ID>/dog-breed-classifier.git

# ③ 推送
git push -u origin main
```

> **关于 22.7MB 的权重文件**（`weights/dogbreed_resnet50_64_int8.onnx`）：
> 它在仓库里是有意为之 —— README 的验收标准是「clone 之后不训练、不下数据
> 就能跑起服务」。GitHub 单文件建议 <50MB，我们在预算内，**不需要 Git LFS**。
> 如果以后模型变大超过 100MB，再迁 LFS 不迟：
> `git lfs install && git lfs track "*.onnx" && git add .gitattributes && git commit`

> **私有 vs 公开**：如果这项目要进简历，建议 Public。
> 敏感信息已检查：仓库里没有密钥、内网地址或个人凭证（只有本机路径注释
> 和阿里云公开报价）。

## 可选的仓库设置（在 GitHub 网页上）

| 设置 | 值 |
|---|---|
| Description | `犬种细粒度分类推理服务：ResNet50 → ONNX INT8，CPU 2.5ms/张，含监控与数据回流闭环` |
| Topics | `pytorch` `onnx` `fastapi` `quantization` `computer-vision` `fine-grained` |
| About → Website | （部署后填你的服务地址） |
| Social preview | `figures/fig-size-vs-latency.png` 或 demo 视频截图 |

## 简历上的一行项目描述

**三档，按空间选：**

**一句话版（推荐，放在项目列表里）：**
> **犬种细粒度分类推理服务** — ResNet50 迁移学习 + ONNX INT8 量化，CPU 上 2.5ms/张、400 QPS，含请求日志监控与用户纠错回流重训闭环（Python / FastAPI / ONNX Runtime）

**两行版（放在重点项目里）：**
> 独立完成从数据管线到上线的全流程：Oxford-Pet 五分类（test top-1 0.89），用分组划分修复数据泄漏（虚高 -7.9pp）；静态 INT8 量化实现体积 1/4、速度 3.7× 而精度仅损 0.18pp；压测定位并修复同步日志把吞吐拖慢 5 倍的问题（69→399 QPS）。

**英文版（投递外企用）：**
> **Dog Breed Classifier** — End-to-end ML service: ResNet50 transfer learning → ONNX INT8 quantization (4× smaller, 3.7× faster, -0.18pp accuracy); FastAPI serving with structured request logging and a human-in-the-loop feedback/retraining pipeline; 2.5ms CPU inference, 400+ QPS single instance.
