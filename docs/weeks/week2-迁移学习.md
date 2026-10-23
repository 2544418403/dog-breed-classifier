# 第二周实验报告 · 迁移学习微调（三段式）

> **主题**：用 `timm` 预训练骨干做三段式微调，量化对比"冻结 vs 微调"
> **任务**：Oxford-IIIT Pet 五类细粒度品种分类
> **环境**：RTX 4060 Laptop 8G · PyTorch 2.9.1+cu126 · timm 1.0.30 · Python 3.13
> **骨干**：resnet50 / efficientnet_b0 / convnext_tiny（3 个，任务要求 ≥2）
> **实验总量**：16 组训练 × 10 epoch，全部真实跑完，数据见 `outputs/summary.csv`

---

## 摘要（先看结论）

三段式流程完整跑通：**3 个骨干 × 3 个阶段 × 10 epoch**，外加 4 组对照/消融实验。

最重要的发现是：**"解冻后准确率下降"的元凶不是学习率，而是 BatchNorm 的统计量。**
这个结论由三条相互独立的证据链支撑：

1. **自然对照**：三个骨干里唯一**没有 BN** 的 convnext_tiny（用 LayerNorm），
   解冻后不但不崩，反而从 99.0% 升到 **100.0%**；而有两个 BN 骨干分别下降
   （resnet50 99.5→98.0，efficientnet_b0 90.5→**77.0**）。
2. **直接干预**：把骨干的 BN 全部冻住（`--freeze-all-bn`），resnet50 阶段 B 在 224px
   从 98.0% 回到 **99.5%**（且逐 epoch 零波动），在 64px 从崩掉的 60.0% 回到 **86.5%**。
3. **学习率消融**：不冻 BN 时，骨干 lr 越小反而越糟（1e-5 → 60.0%，1e-3 → 88.5%）；
   这只有用"BN 统计量与权重失配"才能解释（§四）。

另外还有一个**选题层面的方法论发现**：这份数据上"冻结 vs 微调"本来就不会有差距，
因为 Oxford Pet 的 37 个品种全部落在 ImageNet 语义空间内（§3.1 线性探针实测）。

---

## 一、任务与数据集

### 1.1 任务

猫犬**细粒度品种分类**（5 类）。选细分类而不是"猫/狗/车/花"这类大跨度类别，
是为了避免"预训练模型零样本就能分对、三个阶段全是平线"的无效实验。

**入选 5 个品种**（小型犬，外观相近）：

| 类别 | 中文 | train | val |
|---|---|---|---|
| havanese | 哈瓦那犬 | 160 | 40 |
| miniature_pinscher | 迷你杜宾 | 160 | 40 |
| pomeranian | 博美 | 160 | 40 |
| shiba_inu | 柴犬 | 160 | 40 |
| yorkshire_terrier | 约克夏梗 | 160 | 40 |
| **合计** | | **800** | **200** |

- **总规模 1000 张**，落在任务要求的 300~2000 张区间内。
- **来源**：Oxford-IIIT Pet Dataset（VGG, University of Oxford），全量 37 个品种。
- **划分**：Oxford 每品种共 200 张（官方 trainval 100 + test 100）。把两个划分合并成池子
  后，**自己按类别做 8:2 分层抽样**（`random_state=42`）得到 train/val，
  让验证集足够大（200 张）、准确率估计更稳。
- 脚本：`scripts/build_dataset.py`。

> **可复现性说明**：本机网络受限（HuggingFace / PyTorch 官方源均不可达），
> 数据集与预训练权重均改从 **ModelScope（魔搭）** 拉取。它完整镜像了 `timm/oxford-iiit-pet`
> 与 `timm/resnet50.a1_in1k` 等仓库，路径与 HuggingFace 命名一致，速度 1.7~2.9 MB/s。
> 权重按 HF 缓存目录结构落盘，训练时 `HF_HUB_OFFLINE=1` 离线加载，全程不依赖外网。

### 1.2 预处理与增强

- **训练**：`Resize(256) → RandomResizedCrop(224, scale=(0.7,1.0)) → RandomHorizontalFlip → ColorJitter(0.2) → Normalize(ImageNet)`
- **验证**：`Resize(224) → Normalize`（**不做随机增强**，否则 val_acc 抖动、无法跨阶段比较）

> 增强刻意温和：细粒度任务里大角度旋转和强裁剪会破坏品种判别特征
> （耳朵立/垂、脸型长宽比），过度增强反而压低上限。

---

## 二、三段式微调的实现

### 2.1 阶段设计

| 阶段 | 解冻范围 | 分类头 lr | 骨干 lr | epoch |
|---|---|---|---|---|
| **A** 冻结骨干 | 无（仅分类头） | 1e-3 | 0 | 10 |
| **B** 部分解冻 | 最后 2 个 stage | 1e-4 | **1e-5** | 10 |
| **C** 全网微调 | 全部 | 1e-5 | **1e-6** | 10 |

阶段间**热启动**：B 从 A 的最佳权重出发，C 从 B 的最佳权重出发。
损失 `CrossEntropyLoss(label_smoothing=0.1)`，优化器 `AdamW(weight_decay=1e-4)`，
学习率按 `CosineAnnealingLR` 退火。

> **为什么 B/C 的分类头 lr 也要递减**：阶段 A 的分类头是**随机初始化**的，需要 1e-3
> 才能快速学会；但 B/C 是热启动，头已经训好了，再从大 lr 退火等于先把学好的头打乱。
> （不过 §3.5 的消融会说明：这一项**不是**本实验中崩溃的主因。）

### 2.2 三个关键实现细节

**细节 1 —— 判别式学习率（discriminative LR）**

```python
param_groups = [
    {"params": head_params,     "lr": 1e-4},   # 已训好的头，只需微调
    {"params": backbone_params, "lr": 1e-5},   # 预训练权重，只需轻推
]
opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
```

**细节 2 —— 未解冻部分的 BatchNorm 统计量必须冻住（本实验的主线发现）**

只设 `requires_grad=False` **不够**。BN 在 `model.train()` 下依然会累加
`running_mean/running_var`——即使权重不更新。用 1000 张小数据去改写 ImageNet 学到的
统计量，等于变相破坏骨干。所以**每个 epoch 都要重新执行**：

```python
def freeze_backbone_bn(model, unfreeze_n, freeze_all=False):
    groups = collect_stage_modules(model)          # [stem, stage1, stage2, ...]
    if freeze_all or unfreeze_n <= 0:
        frozen = groups                            # 阶段A：全冻
    elif unfreeze_n >= len(groups):
        frozen = []
    else:
        frozen = groups[:-unfreeze_n]              # 阶段B：只冻未解冻的部分
    for g in frozen:
        for st in g:
            for m in st.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
```

**细节 3 —— 一个真实的 bug：stem 也要算进 stage**

第一版 `collect_stage_modules()` 只返回 `layer1..layer4`，**漏掉了 stem（conv1/bn1）**，
结果 stem 的 BN 在阶段 A 从未被冻住。修正后 64px 阶段 A 从 83.0% 升到 **86.5%** ——
一个容易被"看起来对"骗过的 bug，必须用 `assert` 级别的验证去确认每组的 BN 状态。

**细节 4 —— 跨骨干统一"最后 N 个 stage"**

```python
if hasattr(model, "layer1") and hasattr(model, "conv1"):   # ResNet
    stem = [model.conv1, model.bn1]; stage_mods = [model.layer1, ..., model.layer4]
elif hasattr(model, "blocks"):                             # EfficientNet
    stem = [model.conv_stem, model.bn1]; stage_mods = list(model.blocks)
elif hasattr(model, "stages"):                             # ConvNeXt / Swin
    stem = [model.stem]; stage_mods = list(model.stages)
```

修正后各骨干的结构（这也是 §3.3 分析的基础）：

| 骨干 | BN 总数 | 解冻部分含 BN | LayerNorm |
|---|---|---|---|
| resnet50 | 53 | 29 | 0 |
| efficientnet_b0 | 48 | 15 | 0 |
| convnext_tiny | **0** | 0 | 23 |

---

## 三、实验结果

### 3.1 【方法论】选题诊断：为什么这份数据"冻结 vs 微调"天然没差距

第一轮冒烟测试出现反常：**阶段 A 只训 1 个 epoch，val_acc 就 98.5%**。
这说明冻结特征几乎瞬间就把任务学完了。

我没有直接接受这个结果，而是做了**量化诊断**（`scripts/probe_subsets.py`）：
用冻结的 resnet50 提 2048 维特征，对若干候选 5 类子集各做一次**逻辑回归线性探针**
（8:2 分层划分，与主实验同随机种子）。如果探针准确率已经很高，
说明特征本来就线性可分，跟训练策略无关。

| 候选 5 类子集 | 图片数 | 冻结特征线性探针 val_acc |
|---|---|---|
| 当前 5 犬种（哈瓦那/迷你杜宾/博美/柴犬/约克夏） | 1000 | **99.50%** |
| 5 种短毛猫（阿比西尼亚/孟加拉/伯曼/孟买/俄罗斯蓝） | 982 | **95.94%** |
| 5 种长毛猫（波斯/布偶/缅因/英短/斯芬克斯） | 1000 | **97.50%** |
| 鼠尾犬系（约克夏/迷你杜宾/博美/吉娃娃/日本狆） | 1000 | **98.50%** |

**换哪 5 类都一样。** Oxford-IIIT Pet 的 37 个品种**全部落在 ImageNet-1k 的语义空间内**
（该数据集本身就是围绕 ImageNet 犬猫类别构建的），所以 ResNet50 的 ImageNet 特征
天然几乎完美线性可分。

> **普适结论**：迁移学习中"需要解冻微调"的前提，是目标任务相对预训练任务存在
> **域偏移**或**语义新颖性**。目标类别已在预训练语义空间内时，冻结特征已足够。
> **可操作建议：选题前先跑一个线性探针**。探针 >95% 就该换更"远离"预训练分布的任务
> （自己拍的瑕疵图、医学影像、显微图像等）。

因此下面的主实验里"解冻不涨反跌"**不是 bug，而是这个诊断的必然结果**；
我随后补做了 §3.4（域偏移）和 §3.6（BN 消融）把机制讲清楚。

### 3.2 主实验：冻结 vs 微调 的准确率曲线（224px）

![冻结vs微调_三阶段曲线](figures/fig1_冻结vs微调_三阶段曲线.png)

| 骨干 | BN层数 | 阶段A 冻结 | 阶段B 解冻最后2 stage | 阶段C 全网 | 
|---|---|---|---|---|
| resnet50 | 53 | **99.50%** | 98.00% ↓1.5 | 98.50% |
| efficientnet_b0 | 48 | 90.50% | **77.00% ↓13.5** | 72.50% ↓18.0 |
| convnext_tiny | **0** | 99.00% | **100.00% ↑1.0** | **100.00% ↑1.0** |

**读图**（每格是一个骨干的 30 epoch 连续曲线，底色区分 A/B/C，竖虚线是阶段分界）：

- **resnet50**：阶段 A 平稳 99.5%，进入阶段 B 的瞬间跌到 94.5%，再慢慢回到 98% —— 
  **解冻带来了净损失**。
- **efficientnet_b0**：阶段 A 到 90.5%，进入 B 的瞬间崩到 63%，最终只回到 77%；
  进入 C 又再跌。**解冻是灾难性的**。
- **convnext_tiny**：阶段 A 99.0%，阶段 B 稳步升到 **100.0%** 并保持。**解冻带来真实收益**。

三个骨干在"阶段分界处的行为"差异巨大，而它们的**训练协议完全相同**——
唯一的结构性差异就是 §2.2 表里的 **BN 层数**。这就是 §3.6 要验证的假设。

### 3.3 骨干对比

![骨干对比](figures/fig2_骨干对比.png)

| 观察 | 说明 |
|---|---|
| convnext_tiny（无 BN）**唯一**在解冻后提升 | 99.0% → 100.0%，是全部实验里唯一的正向收益 |
| efficientnet_b0 跌得最惨 | 90.5% → 77.0%；它的 BN 都在卷积块里，且基线本来就低 |
| 参数量与性价比 | efficientnet_b0 只有 5.3M 参数（resnet50 25.6M / convnext_tiny 28.6M） |
| **结论** | **在这个"同域小任务"上，最优选择是 convnext_tiny 或"最小骨干 + 冻结"**；骨干能力差异被任务天花板掩盖，但**对 BN 的敏感度差异被放大了** |

### 3.4 对照实验一：域偏移（224px → 64px）

为了让"冻结特征不够用"的场景真实出现，构造了一个**受控域偏移**：把输入分辨率从 224
降到 64。低分辨率下预训练模型依赖的纹理/细节大量丢失，ImageNet 特征不再够用。

![域偏移对照](figures/fig3_域偏移对照.png)

| 分辨率 | 阶段A 冻结 val_acc | 说明 |
|---|---|---|
| 224px（同域） | 99.50% | 冻结特征已够用 |
| **64px（域偏移）** | **86.50%** | 掉了 13 个点，冻结特征不够用了 |

64px 下训练集准确率能到 96%，但验证集只有 86.5% —— **过拟合**，
说明冻结特征的表达力已经不是瓶颈，瓶颈在于特征本身不适配低分辨率输入。

**这组对照把"什么时候需要微调"从口号变成了可测量**：同一骨干、同一份代码、同一批数据，
只改输入分布，冻结基线就从 99.5% 掉到 86.5%，"微调是否有价值"才变成一个真问题。

### 3.5 对照实验二：学习率消融（64px 阶段 B）

固定 64px、固定阶段 B（都解冻最后 2 个 stage）、都从**同一个** 64px 阶段 A 权重热启动，
**只改学习率**：

| 配置 | head_lr | backbone_lr | 最佳 val_acc |
|---|---|---|---|
| 冻结基线（阶段A） | — | — | 86.50% |
| 推荐配置 | 1e-4 | 1e-5 | **60.00%** ↓26.5 |
| 只调大分类头 lr | **1e-3** | 1e-5 | 67.00% ↓19.5 |
| 只调大骨干 lr | 1e-4 | **1e-3** | **88.50% ↑2.0** |
| 阶段 C（更小的 lr） | 1e-5 | 1e-6 | **46.00%** ↓40.5 |

![学习率消融](figures/fig4_学习率消融.png)

**这个结果和教科书说法完全相反：骨干 lr 越大反而越好。**

`bb=1e-3` 那条曲线是一条**干净的正常微调曲线**（loss 1.24→0.48，train_acc 60%→98%，
val 74%→88.5%），是真实学习，不是巧合。而 `bb=1e-5` 从第 1 个 epoch 就崩到 60% 且再也起不来。

**为什么？** 权重在 `1e-5` 下 25 步/epoch × 10 epoch 只能移动极小距离，**不可能**造成
26 个点的崩塌。唯一能瞬间改变前向行为的是 **BatchNorm 的 running statistics** ——
它们以固定 momentum（0.1）更新，**与学习率无关**。这就引出 §3.6 的直接验证。

### 3.6 【核心发现】对照实验三：BN 消融 —— 崩溃的元凶是 BatchNorm 统计量

假设：解冻后，被解冻 stage 的 BN 统计量会立刻被 800 张小图改写，
导致 **"权重还是 ImageNet 的、统计量却已经变了"** 的失配，分类头随之失效。

预测：如果把骨干**所有** BN 都冻在 eval（`--freeze-all-bn`），
崩溃应当完全消失。实测：

| 场景 | 阶段B，BN 照常更新 | 阶段B，骨干 BN 全部冻住 | 阶段A 冻结基线 |
|---|---|---|---|
| 224px resnet50 | 98.00%（第1轮跌到 94.50%） | **99.50%（全程零波动）** | 99.50% |
| 64px resnet50 | 60.00%（崩） | **86.50%（零崩溃）** | 86.50% |
| 224px efficientnet_b0 | 77.00%（崩） | **94.00%（反超基线）** | 90.50% |

![BN消融_崩溃元凶](figures/fig6_BN消融_崩溃元凶.png)

**三条独立证据链在此闭合：**

1. **自然对照**：convnext_tiny 用 LayerNorm、没有 BN，是唯一解冻后提升的骨干（99.0→100.0）。
2. **直接干预**：冻住 BN 后，两个 BN 骨干在两种分辨率下**全部恢复到冻结基线或更好**，
   且曲线从"先崩再缓慢恢复"变成"从第 1 个 epoch 就平稳"。
3. **lr 消融反推**：不冻 BN 时小 lr 最糟、大 lr 最好——因为大 lr 让权重能追上统计量的漂移；
   冻住 BN 后，这个反常消失（224px 小 lr 也能稳定 99.5%）。

> **efficientnet_b0 的 94.00% 尤其说明问题**：冻住 BN 后它**超过了**自己的冻结基线 90.50%
> （+3.5 个点）——这才是教科书里"解冻微调带来提升"的正常形态。也就是说，
> **efficientnet_b0 之前跌到 77% 完全是 BN 统计量被污染造成的，不是"微调没用"。**

### 3.7 混淆矩阵

![混淆矩阵](figures/fig5_混淆矩阵.png)

### 3.8 汇总实验数据表

完整表格见 `outputs/summary.csv` / `outputs/summary.md`，共 16 组实验。
字段：`实验 / 骨干 / 阶段 / 阶段说明 / epochs / 最佳val_acc / 末轮val_acc / 最佳train_acc`。

---

## 四、核心问题：为什么解冻后需要更小的学习率？

这是本周验收重点。**本实验给出的答案比教科书更精确，而且每一条都有实测证据。**

### 教科书答案（仍然成立，但要加前提）

**原因一：预训练权重落在"深而窄"的谷底，大 lr 会把它踢出去。**
预训练收敛后的权重位于 ImageNet 损失面一个很深、很窄的极小值附近。
更新量 ∝ `lr × 梯度`：谷底梯度本身很小，但谷底也很窄，大 lr 的位移足以把参数冲出谷底，
掉进更差的区域 —— **灾难性遗忘**。典型现象是 loss 先飙升再收敛到更差的解。
本质是：**更新步长必须小于谷底的宽度**。

**原因二：分类头是随机初始化的，前几步会释放"脏梯度"。**
阶段 B 刚开始时头的 loss 很大、梯度方向混乱，会反向污染刚解冻的骨干。
判别式学习率（头大 lr / 骨干小 lr）就是为了隔离这个污染。

### 本实验的重要修正：上面两条都成立，但**都不是**本实验中崩溃的主因

实测数据（§3.5、§3.6）指出第三个、也是本实验里**最强**的效应：

**原因三（本实验的主发现）：BN 的 running statistics 与学习率无关地被改写，
造成"权重-统计量失配"。**

- BN 统计量在 `train()` 模式下按固定 momentum（0.1）更新，** lr 再小也照更新不误**；
- 阶段 A 里被冻住的 BN，其统计量是 ImageNet 的值，分类头是在这套统计量定义的特征上学出来的；
- 一旦解冻，统计量立刻开始向 1000 张小图的统计量漂移 → 特征整体平移 → 头失效；
- 此时若骨干 lr 很小，**权重跟不上统计量的漂移**，失配持续存在 → 崩（60.0%）；
- 若骨干 lr 足够大，权重能追上统计量，一起适配新域 → 反而最好（88.5%）。

**所以正确的表述是：**

> **"解冻后要用小学习率"的前提是：BN 的统计量也要一起冻住。**
> 只冻权重、不冻 BN，是最容易踩的坑 —— 此时小 lr 不仅无益，反而是**最坏**的选择。
>
> 正确的微调配方（小 batch / 小数据 / 低分辨率场景）：
> 1. 冻结骨干所有 BN 统计量（`model.train()` 后把 BN 全部 `eval()`）；
> 2. 骨干用小 lr（保护预训练特征，防灾难性遗忘）；
> 3. 分类头按需给 lr（随机初始化用大 lr，热启动用小 lr）。

**一句话总结**：**先决定 BN 怎么办，再决定学习率。** 顺序反了，小 lr 会变成毒药。

---

## 五、结论

1. **阶段 A 的上限与收敛速度**：224px 下 1 个 epoch 即近饱和（99.5%），说明冻结特征
   对这份任务已足够；64px 下冻结基线降到 86.5%，出现真实的提升空间。
2. **阶段 B 是否带来提升，取决于两件事**：
   - 任务是否有域偏移（224px 无 → 无空间；64px 有 → 有空间）；
   - **BN 统计量是否被处理好**（这是本实验的主发现）。
3. **阶段 C 是否值得**：在 BN 未处理时阶段 C 是最糟的（64px 46.0%）；
   convnext_tiny 上阶段 C 稳定在 100%。**先修 BN，再谈阶段 C。**
4. **骨干差异**：convnext_tiny（无 BN）是唯一解冻后提升的骨干；
   在同域小任务上应选它或"最小骨干+冻结"。**对 BN 的敏感度是选骨干时被忽视的维度。**
5. **方法论收获（最重要）**：
   - **选题前先跑线性探针**：探针 >95% 说明冻结特征已够用，三段式微调不会产生有意义的对比。
   - **调试迁移学习时，先怀疑 BN 再怀疑 lr**：一个 `freeze_backbone_bn()` 的位置错误
     （stem 被漏掉）就足以让 64px 阶段 A 差 3.5 个点。
   - **"解冻后用小 lr"不是普适真理，而是有前提的经验法则**；用消融实验验证前提，
     比背诵结论有价值得多。

---

## 六、踩坑记录

**坑 1 · 所有常规权重/数据源被封**
HuggingFace（含 hf-mirror）、download.pytorch.org、GitHub release 大文件、Google 系全部不可达，
`timm.create_model(..., pretrained=True)` 直接抛 502。
解法：**ModelScope（魔搭）**完整镜像 `timm/` 命名空间，路径与 HF 一致；
按 HF 缓存结构落盘后 `HF_HUB_OFFLINE=1` 离线加载。

**坑 2 · 一个依赖把 torch 装坏**
安装某第三方 SDK 时，它带进来的旧组件覆盖了 `torch/__init__.py`（新版要求
`torch.random.thread_safe_generator`，磁盘上的 `random.py` 是旧版），`import torch` 直接 ImportError。
定位：**比对 pip `RECORD` 里记录的 sha256 与磁盘实际哈希** —— `random.py` 一致（未污染）、
`__init__.py` 不一致（被替换），立刻锁定。修复：`--force-reinstall --no-deps` 重装 torch。
教训：**装重依赖前后都要 `python -c "import torch"` 验一次**。

**坑 3 · stem 的 BN 被漏掉（自己写的 bug）**
`collect_stage_modules()` 只返回 `layer1..layer4`，stem（conv1/bn1）不在任何"stage"里，
导致阶段 A 的 stem BN 一直在更新统计量。修正后 64px 阶段 A 从 83.0% → 86.5%。
教训：**"冻结"这种声明式的设计，必须写一个断言去逐 module 验证 `training` 标志位**。

**坑 4 · 并发 3 路训练触发 OpenBLAS 线程爆炸**
`OpenBLAS error: Memory allocation still failed after 10 retries` 刷屏，
进程**空转卡死**（18 分钟累计 CPU 只 46 秒）——不是忙，是阻塞。
根因：20 逻辑核 × 多进程 × 每进程默认 OpenBLAS 线程池 → 线程/内存分配爆炸。
解法：`export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1`，
并发限制为 2 路。**教训：判断"卡死"要看进程的累计 CPU 时间，而不是日志有没有动
（print 未 flush + 管道会块缓冲，日志本来就不实时）。**

**坑 5 · 后台下载被回收 & 分段下载饿死**
`nohup ... &` 起的后台任务会被环境回收；12 段并发下载瞬时 13MB/s 但几秒后全部连接挂起。
解法：用工具自身的后台机制 + **单连接断点续传 + 失败重试**（稳定 2~9 MB/s）；
分段下载的 SIZE 必须按服务端 `Content-Length` 精确计算（zip 中央目录在末尾，差一点就废）。

**坑 6 · 教学数据集"太简单"是隐形陷阱**
见 §3.1。原以为"挑 5 个相似品种"就能制造难度，实测完全无效。
**不先做诊断，整份报告都会建立在一条平线上。**

---

## 附：复现方式

```bash
cd D:/develop/深度学习八周实验/第二周

# 0) 下载数据与权重（ModelScope 镜像）
bash data/dl_pet_ms.sh
bash scripts/prepare_weights_ms.sh

# 1) 构建 5 分类数据集（1000 张，8:2 分层）
python scripts/build_dataset.py

# 2) 选题诊断：线性探针（强烈建议先跑）
python scripts/probe_subsets.py

# 3) 主实验：3 骨干 × 3 阶段 × 10 epoch
python scripts/run_all.py

# 4) 对照实验
python scripts/run_all.py --backbones resnet50 --img-size 64 --tag _res64      # 域偏移
python scripts/train.py --backbone resnet50 --stage B --img-size 64 \
    --tag _res64_hlr1e3 --init checkpoints/resnet50_A_res64.pt --head-lr 1e-3  # 头lr消融
python scripts/train.py --backbone resnet50 --stage B --img-size 64 \
    --tag _res64_blr1e3 --init checkpoints/resnet50_A_res64.pt --backbone-lr 1e-3  # 骨干lr消融
python scripts/train.py --backbone resnet50 --stage B --tag _bnz --freeze-all-bn   # BN消融
python scripts/train.py --backbone resnet50 --stage B --img-size 64 \
    --tag _res64_bnz --init checkpoints/resnet50_A_res64.pt --freeze-all-bn
python scripts/train.py --backbone efficientnet_b0 --stage B \
    --init checkpoints/efficientnet_b0_A.pt --tag _bnz --freeze-all-bn

# 5) 出图与汇总表
python scripts/plot.py
```

| 脚本 | 作用 |
|---|---|
| `scripts/build_dataset.py` | 从 parquet 建 5 分类数据集（8:2 分层） |
| `scripts/probe_subsets.py` | 线性探针选题诊断 |
| `scripts/train.py` | 单次训练（`--backbone --stage --epochs --img-size --tag --init --head-lr --backbone-lr --freeze-all-bn`） |
| `scripts/run_all.py` | 一键跑完某骨干 A→B→C（支持 `--img-size --tag`） |
| `scripts/plot.py` | 出 6 张图 + 汇总表 |
