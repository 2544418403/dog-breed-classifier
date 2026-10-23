"""dog-breed-classifier —— 五个子包：data / models / train / eval / serve。

src 布局（src-layout）的意义：仓库根目录下的 `weights/` `datasets/` 等数据目录
不会被意外当成 Python 包 import 进来，避免 `import models` 拿到的是权重目录。
详见 README「目录结构」一节。
"""

__version__ = "1.0.0"
