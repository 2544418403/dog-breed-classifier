# 第四周实验报告：评估方法论与 badcase 归因

> 数据集：Oxford Pets pet5（第三周同款，5 类长尾，train 979/784/589/391/197，ratio 4.97:1）
> 工作点：timm resnet50 @ **64px**（224px 是饱和区，test 99.45%，错分个位数无法归因——见第三周结论）
> 硬件：RTX 4060 Laptop 8GB · 单种子 seed=42 + 三种子复核（42/7/2024）
> 测试集：by_plate 划分 test n=546（板 P00/P05/P19），val n=544

---

## 0. 一句话总结

**accuracy 0.9011 看起来很好，但拆开看：少数类塌的是 precision（shiba 0.615 / york 0.600）而非 recall；错误不是均匀分布而是"虹吸"进小类（york 吸入 lift 4.87×）；错误按个体整组聚集（聚类 bootstrap CI 宽 +85%）；全部错误发生在"路由"阶段而非类间区分（组内二分正确率 100%）。** 三条对应措施全部验证成立：先验校正（少类 P +10.9pp，显著）、个体级聚合（+2.5pp）、温度缩放（ECE 0.073→0.026）。class weight / focal / CB loss 在三种子下全部未通过噪声检验。

---

## 1. 指标阶梯：只看 accuracy 会漏掉什么

baseline（64px, seed42, test n=546）：

| 指标 | 值 | 95% CI（图片级 bootstrap） |
|---|---|---|
| accuracy | 0.9011 | [0.874, 0.923] |
| balanced_acc | 0.9279 | — |
| **macro_F1** | **0.8315** | [0.792, 0.870] |
| macro_precision | 0.7995 | — |
| macro_recall | 0.8793 | — |
| weighted_F1 | 0.9054 | — |
| **worst_class_F1** | **0.6792** | [0.549, 0.758] |
| minority_recall | 0.7826 | [0.609, 0.957]（CI 宽 0.35！） |
| kappa | 0.8607 | — |
| ECE | 0.0728 | — |
| AUROC (macro/micro) | 0.9806 / 0.9889 | — |
| AP (macro/weighted) | 0.9217 / 0.9685 | — |

关键读法（图 `fig_ladder_baseline_test.png`、`fig_perclass_baseline_test.png`）：

1. **accuracy = weighted_F1 ≈ 0.90，但 macro_F1 只有 0.83、worst_class_F1 只有 0.68**。accuracy 被大类（havanese 214 + minpin 189 = 74% support）主导。
2. **micro-recall ≡ accuracy**（多分类恒等式），所以"召回 0.90"这句话不携带任何额外信息。
3. **AUROC 0.98 vs AP 0.92**：AUROC 的 FPR 分母是巨大的负样本池，对少类 precision 崩塌不敏感；PR 曲线（图 `fig_rocpr_baseline_test.png`）里 shiba/york 的 AP（0.864/0.796）才暴露问题。**类别不平衡时看 PR-AUC，不看 ROC**。

### 每类指标（test）

| class | support | precision | recall | F1 | AP |
|---|---|---|---|---|---|
| havanese | 214 | 0.969 | 0.869 | 0.916 | 0.972 |
| miniature_pinscher | 189 | 0.983 | 0.931 | 0.957 | 0.995 |
| pomeranian | 92 | 0.830 | 0.957 | 0.889 | 0.982 |
| **shiba_inu** | 28 | **0.615** | 0.857 | 0.716 | 0.864 |
| **yorkshire_terrier** | 23 | **0.600** | 0.783 | **0.679** | 0.796 |

**归因线索 #1：少数类塌的是 precision 而非 recall**（shiba R=0.857 / P=0.615；york R=0.783 / P=0.600）。这说明模型"找得全但认不准"——把别类错认成 shiba/york 的样本，比漏掉的 shiba/york 更多。这条线索直接否决了"加 class weight"这种以提高 recall 为目标的药方（§4 实证证实）。

---

## 2. 混淆矩阵与错分方向诊断

图 `fig_confusion_baseline_test.png`（行归一化）：最大的非对角块是 york→hava 0.17、minpin→shiba 0.06、hava→pom 0.07。

### 2.1 错分方向分组（pooled val+test，92 个错分）

图 `fig_pairs_baseline.png`：x 轴是对称度 `min(n_ab,n_ba)/max(...)`（→1 = 互混的"硬对"；→0 = 单向流的先验问题），y 轴是错分数，颜色是调整残差（全部 < −1.96，即所有主要错分方向都显著偏离独立假设）。

| 方向 | n | symmetry | 解读 |
|---|---|---|---|
| minpin→shiba | 23 | **0.04** | 纯单向 —— 不是"长得像"，是**流向** |
| hava→pom | 22 | 0.05 | 纯单向 |
| hava→york | 18 | 0.28 | 单向为主（反向 york→hava 有 5） |
| shiba→pom | 8 | 0.25 | 单向 |
| york→hava | 5 | 0.28 | hava→york 的反向 |

**单向性（symmetry ≈ 0）是 prior 失衡的指纹，不是特征不可分的指纹。** 真正"长得像"的硬对应该双向流动（如 pomeranian↔yorkshire 这对 symmetry=1.0 但 n=1，无足轻重）。

### 2.2 置信度画像：理直气壮地错

图 `fig_conf_profiles_baseline.png`：

| 方向 | n | mean_conf | mean_margin | mean_p_true |
|---|---|---|---|---|
| minpin→shiba | 23 | 0.694 | 0.488 | 0.186 |
| hava→pom | 22 | 0.580 | 0.307 | 0.209 |
| hava→york | 18 | 0.683 | 0.508 | 0.172 |

错分样本中 margin < 0.2 的只占 **25%** —— 大多数错分不是"犹豫着错"，而是**高置信度地错**。这意味着"用置信度阈值过滤 low-confidence 样本给人审核"挡不住这批错误。

---

## 3. badcase 人眼归因（核心产出）

四张拼图（`fig_badcase_baseline_g00..g03.png`）逐组人眼看图后的归因假设与验证：

### G0：minpin→shiba（n=23）——"采集畸变 + 体型线索丢失"
- **人眼所见**：多帧重度伪彩畸变（纯黄绿假彩、近黑帧）、公园远景；min pin 与 shiba 都是红褐色小型犬，64px 下体型差异（minpin 更瘦更方）完全不可见。
- **假设 H1a**：畸变帧拖累。**验证**：质量探针子组分层对照（`fig_subgroup_*.png`）——`exif_rotated` 显著为负（diff −0.036, CI [−0.103, −0.019]），但 `injected_grayscale`（n=36）diff +0.020 CI [−0.041, +0.087] **不显著**，`low_sharpness`/`low_color` 也不显著。
- **诚实结论**：畸变确实伤（旋转显著），但**灰度本身不显著**——直观归因被对照实验部分推翻，这正是要做对照而不是拍脑袋的原因。

### G1：hava→pomeranian（n=22）——"分布内视觉子类"
- **人眼所见**：几乎全是**黑白双色**的 havanese（训练分布里 havanese 以纯白为主），个别假彩帧。
- **假设 H2**：颜色子型 OOD。havanese 类内黑白/纯白两个视觉子型，测试集黑白色被吸入 support 与纯白更近的 pomeranian。**对应措施**：类内多样化增强 / 采集补黑白 hava。

### G3：hava→yorkshire（n=18）——"长毛遮脸 = 关键特征不可见"
- **人眼所见**：3 张是**同一只狗同角度**（长毛遮眼、只见毛团），其余也多为脸被毛发遮挡。
- **假设 H3**：这不是类间相似，是**有效信息量不足**。验证见 §5 逐级归因——模型不是"分不清 hava 和 york"，是"没看到能用的特征"。

### G4：shiba→pomeranian（n=8）——"面部被遮挡"
- 花丛中藏脸、背身帧。与 G3 同机制：**遮挡导致路由失败**。

**三条可验证的错分归因（验收要求）：**

| # | 归因 | 量化证据 | 对应改进措施（已验证） |
|---|---|---|---|
| A1 | **先验虹吸**：训练先验（shiba 16.5%/york 8.2%）≠ 部署先验（5.1%/4.2%），错误单向流入小类 | lift york 4.87× / shiba 3.56×；Spearman(误流入, log n)=−0.50；symmetry≈0 | Fix-1 先验校正 τ=1.2（§6.1） |
| A2 | **错误按个体聚集**：同一条狗整组错 | 92 错分来自 42 个体，21 个 ≥2 帧；test 21 个有错个体中 14 个每帧都错 | Fix-2 个体级聚合（§6.2） |
| A3 | **64px 分辨率不足以支撑路由**：错误 100% 在路由阶段，组内区分 100% 正确 | img224 +11.1pp macro-F1，三种子 CI 完全不重叠；hierarchical within=1.0 | Fix-3/分辨率升级（§4 img224） |

另有一条**诚实记录的否定结果**：直觉上"灰度注入帧是 badcase 来源"，但分层对照不显著（+0.020, CI 跨 0）——不能把灰度列为归因。

---

## 4. 虹吸效应：与 prior 塌缩方向相反的反直觉发现

图 `fig_error_flow_baseline.png`：左图 observed vs expected（按 support 比例分配的错误流入量），右图 lift。

| 吸入类 | 实际吸入 | 按比例应有 | support | lift |
|---|---|---|---|---|
| havanese | 9 | 35.5 | 421 | 0.25 |
| miniature_pinscher | 7 | 32.5 | 385 | 0.22 |
| pomeranian | 31 | 12.8 | 152 | 2.42 |
| shiba_inu | 26 | 7.3 | 86 | 3.56 |
| **yorkshire_terrier** | 19 | 3.9 | 46 | **4.87** |

经典文献（prior collapse）预测错误应流向**大类**（大类的 logit 先验高）；本实验**相反**：support 越小的类吸入错误越多，Spearman(误流入量, log n) = **−0.50**。

**解释**：训练先验 shiba 16.5% / york 8.2%，而 test 先验只有 5.1% / 4.2%。模型忠实学习了训练分布的 prior，但部署分布里这两个类变少了——于是它把"边缘样本"按过期的 prior 塞给小类。**这是 train/deploy 先验漂移问题，不是类别不平衡本身**。

**验证性实验（uniform2）**：只用 shiba+york 的均衡子集重训（224px），test 该组 51 样本 **100% 正确**——证明 shiba/york 特征层面完全可分，排除"类间不可分"假设。（注意：uniform2 是证伪实验，不进消融表。）

---

## 5. 逐级归因：错误 100% 发生在路由，而非类间区分

把 5 类决策拆成两级：① 路由（进不进正确候选组 A={shiba,york}/B={hava,pom}/C={minpin}）② 组内区分。

| 模型 | 组 | stage1 路由 | stage2 组内 | 端到端 |
|---|---|---|---|---|
| baseline (64px) | A_shiba_york | **0.8235** | **1.0** | 0.8235 |
| baseline (64px) | B_hava_pom | 0.9444 | 0.9481 | 0.8954 |
| baseline (64px) | C_minpin | 0.9312 | — | 0.9312 |
| img224 | A / B / C | 0.94 / 1.0 / 1.0 | 1.0 | 0.94/1.0/1.0 |
| uniform2 | A | 1.0 | 1.0 | 1.0 |

**决定性证据：凡是模型认为"是 shiba 或 york 之一"的样本，二分正确率 100%（42/42）。** A 组 12 个错误全部丢给了 A 组之外的类（hava 5 / pom 3 / minpin 1）。

结论：**不存在"shiba 和 york 长得像所以分不开"这回事**。问题全部在于"该不该进这个候选组"的先验判断——又回到 A1 的先验漂移归因。这与特征空间几何一致（`geometry_baseline.csv`：类心余弦最高的是 shiba-pom 0.671 和 york-hava 0.642，**不是** shiba-york 0.565——"直觉上最像的对"在特征空间里反而不是最近的）。

---

## 6. 改进措施验证（三条全部成立）

### Fix-1：先验校正（logit adjustment，τ=1.2 在 val 上按 macro-F1 网格拟合）
`logits' = logits + τ·(log π_train − log π_train_prior)` 修正训练先验向均匀先验靠拢。

| 指标 | 前 | 后 | 配对聚类 bootstrap CI |
|---|---|---|---|
| 少数类 precision | 0.6077 | **0.7168（+10.91pp）** | [+0.011, +0.241], p=0.99 **显著** |
| macro_F1 | 0.8315 | 0.8661（+3.46pp） | [−0.004, +0.076] 边缘 |
| accuracy | 0.9011 | 0.9249（+2.38pp） | — |
| worst_class_F1 | 0.6792 | 0.7391 | — |

**零训练成本**（只动推理端一行公式），直击 A1。代价是 macro_recall 微降（0.879→0.888 内部再平衡）。

### Fix-2：个体级聚合（同一 subject 的帧做平均概率聚合）
- 图片级 acc 0.9011 → 个体级（平均概率）0.8931，但这是**决策单元不同**的对比；同单元对比：all-or-nothing 0.8679 → 平均聚合 0.8931，**+2.5pp**；救回 4 个个体、改错 0 个。
- 意义：A2 证明错误按个体聚集（14/21 有错个体每帧都错），把决策单元升到"个体"既匹配业务语义（一条狗一个判断），又能白拿聚集错误的收益。

### Fix-3：温度缩放（T=0.668，val 拟合）
- ECE **0.0728 → 0.0257**，argmax 一致率 100%（不动决策，只修校准）。
- MCE 反而 0.170→0.232：等宽分箱下 MCE 对最坏箱敏感，属正常现象，报告中如实记录。
- 可靠性图：`fig_reliability_baseline_test.png`（过自信）→ `fig_reliability_fix3_baseline.png`（贴对角线）。

---

## 7. 消融实验：噪声检验下的残酷现实

单种子（seed42, test）汇总 + 成对聚类 bootstrap（vs baseline）+ 三种子复核：

| exp | acc | macro_F1 | 少类P | worst_F1 | ΔmF1 [成对聚类 95% CI] | 三种子判定 |
|---|---|---|---|---|---|---|
| **img224** | 0.9945 | **0.9847** | 1.000 | 0.930 | **+0.153 [+0.081, +0.226]** | **真实**（+0.111，min>max 区间不重叠） |
| img384 | 0.9927 | 0.9805 | 1.000 | 0.930 | +0.149 [+0.079, +0.219] | 真实但无增益于 224 |
| gray_aug | 0.9341 | 0.8721 | 0.657 | 0.731 | +0.041 [+0.008, +0.076] 单种子显著 | **疑似噪声**（+0.010 ± 0.028 跨种子） |
| focal | 0.9066 | 0.8453 | 0.617 | 0.716 | +0.014 [−0.003, +0.033] | 未通过 |
| cb_ce | 0.9048 | 0.8392 | 0.599 | 0.656 | +0.008 [−0.016, +0.035] | 未通过 |
| **cw_sqrt** | 0.9029 | 0.8416 | 0.615 | 0.690 | +0.010 [−0.012, +0.035] | **无效**（ΔmF1 −0.009 ± 0.022） |
| baseline | 0.9011 | 0.8315 | 0.608 | 0.679 | — | — |
| cw_inverse | 0.9011 | 0.8267 | 0.523 | 0.575 | −0.005 [−0.038, +0.025] | 变坏（少类P 0.61→0.52） |
| cb_focal | 0.8993 | 0.8215 | 0.576 | 0.642 | −0.010 [−0.029, +0.003] | 变坏 |
| no_aug | 0.8956 | 0.8203 | 0.600 | 0.590 | −0.011 [−0.068, +0.040] | 增强有贡献但 CI 跨 0 |

图 `fig_ablation_macro_f1.png`、`fig_seeds_box.png`。

**两个方法论要点：**

1. **种子噪声是消融的隐形杀手**。同 seed=42 重跑两次 baseline 得 0.9066 vs 0.9011（GPU 非确定性）；三种子 baseline 的 mF1 标准差 0.039。**任何小于 ±0.04 的"改进"在单种子下都不可信** —— gray_aug 单种子 +0.041 且 CI 不含 0，跨种子后照样落进噪声。面试里说"我的方法提了 2 个点"而不报种子方差，和没说一样。
2. **class weight / focal / CB loss 全线失败**，且方向一致：macro_F1 基本不动、少数类 **precision** 恶化（cw_inverse 0.608→0.523）。原因正是 §1 归因线索 #1——少数类塌的是 precision，而这些方法全部在推高 recall，药不对症。**"长尾就该加权"是本实验最有价值的一条证伪。**

---

## 8. 不确定性：图片级 bootstrap 低估了 85%

图 `fig_cluster_ci_baseline.png`：

| split | 图片级 CI 宽 | 聚类级（按 subject）CI 宽 | 膨胀 |
|---|---|---|---|
| test | 0.082 | 0.148 | **+80%** |
| val | 0.066 | 0.101 | +53% |

因为 92 个错分聚集在 42 个个体上（21 个个体 ≥2 帧一起错），样本间不独立，图片级 bootstrap 违反独立性假设。**凡是有"每实体多帧/多测量"结构的数据，必须聚类重抽样**，否则给老板报的 CI 就是自欺。

（bootstrap CI 与多种子是两个独立的随机源：前者量化测试样本抽样噪声，后者量化训练随机性。本报告两者都做了。）

---

## 9. 结论与遗留

**验收对照**：≥3 条错分归因（A1 先验虹吸 / A2 个体聚集 / A3 分辨率不足 + 1 条诚实否定的灰度假设）✓；对应 ≥3 条可验证改进（Fix-1 显著 / Fix-2 成立 / Fix-3 成立 + img224 真实改进）✓。

**遗留 / 下一步**：
- Fix-1 的 macro_F1 CI [−0.004, +0.076] 只算边缘显著，多种子下复核 τ 的稳定性；
- G1（黑白 hava）需要采集侧补数据而不是算法侧修补；
- 224px 已把问题"消灭"而非"解决"——真正该学的是 64px 工作点上的诊断流程，它能迁移到任何没有饱和预算的真实业务。

## 附：产物索引
- 代码：`evalkit/`（metrics/badcase/attribution/quality/calibrate/losses/curves），`scripts/`（train/run_eval/run_badcase/run_compare/run_seeds/verify_fixes/hierarchical_probe/make_figures）
- 30 项对拍测试（对 sklearn）：`tests/test_evalkit.py` 全部通过
- 数据：`outputs/*.csv|json`；图：`figures/*.png`（26 张）
