# -*- coding: utf-8 -*-
"""quotes — 行情源层

只做网络请求与源格式归一化，不含业务判定。
腾讯(实时报价) / 东财(板块榜对照) / akshare(涨停池) / 大QMT(实时报价备选)。

fetch_quotes 按 config.QUOTE_SOURCE 分发: tx=腾讯http(默认) | qmt=大QMT。
切换只需环境变量 QUOTE_SOURCE=qmt（或写入.env），调用方代码不变。
"""
from config import QUOTE_SOURCE  # noqa: E402

if QUOTE_SOURCE == "qmt":
    from quotes.qmt import fetch_quotes  # noqa: E402,F401
else:
    from quotes.tx import fetch_quotes  # noqa: E402,F401
