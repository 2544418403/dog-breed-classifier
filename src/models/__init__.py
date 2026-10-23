"""模型相关：结构定义（工厂）、格式转换与量化导出。

这里只做三件事：
    1. factory.py  用什么骨干、哪些层解冻、给哪些层多大学习率
    2. export.py   训练态(pth) -> 服务态(onnx / int8 onnx)
依赖 torch；服务运行时不 import 本包。
"""

from .factory import PARAM_GROUPS_LR, build_model, load_checkpoint

__all__ = ["build_model", "load_checkpoint", "PARAM_GROUPS_LR"]
