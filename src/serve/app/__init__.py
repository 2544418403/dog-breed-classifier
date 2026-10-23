"""serve.app —— HTTP 应用层。

分层约定（改代码前先看这个）：
    config.py     所有旋钮
    model.py      唯一知道模型细节的地方
    security.py   唯一做输入校验的地方
    telemetry.py  唯一决定"一条请求留下什么痕迹"的地方
    feedback.py   唯一写用户反馈的地方
    main.py       只做编排：接请求 -> 调上面几个 -> 回 JSON
"""
