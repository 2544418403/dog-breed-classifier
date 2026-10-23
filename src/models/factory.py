"""模型工厂：骨干选择 + 部分解冻 + 判别式学习率分组。

==============================================================================
为什么这里是这段代码最重要的三个决策
==============================================================================

1) 为什么默认 resnet50，而且用 timm 创而不是手抄一份实现
   "自己写 ResNet" 在这个项目里是负收益：结构一旦和 timm 有细微差别，
   ImageNet 预训练权重就加载不进去（或装进去但不匹配），而这恰恰是最难排查的
   一类 bug——模型能跑、loss 会降，只是永远收敛不到该有的水平。
   我们要比的是训练策略，不是网络搭建手速。

2) 为什么只解冻「分类头 + 最后 2 个 stage」
   第二周实测：全网微调在本任务上验证集反而更差，而且训练时间贵好几倍。
   语义上说得通——ImageNet 的底层纹理/边缘特征对任何自然图像都通用，
   需要重新适配的是高层语义部分。

3) 判别式学习率（discriminative learning rates）
   分类头是随机初始化的，需要从 0 学 → 给大 lr；
   骨干是预训练的，只想轻推 → 给小 lr。
   两组参数放进同一个 optimizer，用不同的 lr 牵引。
   全网一个 lr 的结果只有两种：要么头部学不动，要么骨干被推坏。
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_UNFREEZE_STAGES = 2

# 不同骨干的「分类头」不叫同一个名字，逐个排除这些才能拿到真正的 stage 列表
_SKIP_MODULES = {"global_pool", "fc", "head", "classifier", "pool", "poolfcn"}

PARAM_GROUPS_LR = {"head": 1e-3, "tail": 1e-4}


def build_model(backbone: str, n_classes: int, device: str = "cpu",
                pretrained: bool = True) -> tuple[nn.Module, list[dict]]:
    """返回一个 (model, optimizer_param_groups)。

    param_groups 已经是 [{'params': head, 'lr': 1e-3}, {'params': tail, 'lr': 1e-4}]，
    直接塞给 optimizer 即可。之所以让工厂顺手把参数组也造出来，是因为
    「哪些层解冻」和「给它们什么 lr」本质是同一个决策，拆到两处必然会对不上。
    """
    import timm

    model = timm.create_model(backbone, pretrained=pretrained, num_classes=n_classes).to(device)

    # ---- 先把所有参数冻住，再按需放开：白名单比黑名单安全 ----
    for p in model.parameters():
        p.requires_grad = False

    head_params = list(model.get_classifier().parameters())
    for p in head_params:
        p.requires_grad = True

    stages = [n for n, _ in model.named_children() if n not in _SKIP_MODULES]
    tail_names = stages[-N_UNFREEZE_STAGES:]

    head_ids = {id(p) for p in head_params}
    tail_params: list[nn.Parameter] = []
    for name in tail_names:
        for p in getattr(model, name).parameters():
            p.requires_grad = True
            # 有些骨干的 classifier 挂在最后一个 stage 名下，去重避免重复计费 lr
            if id(p) not in head_ids:
                head_ids.add(id(p))
                tail_params.append(p)

    groups = [
        {"params": head_params, "lr": PARAM_GROUPS_LR["head"]},
        {"params": tail_params, "lr": PARAM_GROUPS_LR["tail"]},
    ]
    return model, groups


def freeze_bn(model: nn.Module) -> None:
    """冻结所有 BatchNorm 的 running 统计量。

    只设 requires_grad=False **不够** —— BN 的 running_mean/running_var
    不通过梯度更新，只要层处于 train 模式就会继续滑动。
    第二周的血泪结论：batch size 小时解冻 BN 会让 val_acc 直接崩掉，
    而且曲线看上去像"过拟合"，很容易误判成网络容量问题。
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def load_checkpoint(path, device: str = "cpu") -> dict:
    """读取 UP 训练脚本保存的 ckpt。

    约定格式：{"state_dict", "cfg", "classes", "seed", "exp"}
    用 weights_only=False 是因为 cfg / classes 是普通 Python 对象；
    这是我们自己产生的文件，风险可控。
    """
    return torch.load(path, map_location=device, weights_only=False)
