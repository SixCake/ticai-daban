# -*- coding: utf-8 -*-
"""股票代码口径双向映射 — rqalpha 与 ticai-daban 的唯一转换点

项目内部(core/ apps/ collect/ quotes/)统一用 tushare 口径 `000001.SZ`;
rqalpha 内部统一用米筐口径 `000001.XSHE`(下单/持仓/Instrument 全是这套)。
混用会在下单 API 处静默失败, 故转换只允许发生在本模块。

约定: 策略代码只见 rqalpha 口径; 框架注入 API(api.py) 返回给策略的数据
一律已转成 rqalpha 口径, 策略不需要也不应该调用本模块。

北交所(.BJ)rqalpha 无对应后缀, 且项目已在雷达宇宙三道拦截中剔除,
此处返回 None 表示不支持(调用方须跳过而非伪造代码)。
"""
from functools import lru_cache

# tushare 后缀 → 米筐后缀
TS2RQ = {"SZ": "XSHE", "SH": "XSHG"}
RQ2TS = {v: k for k, v in TS2RQ.items()}
# 米筐交易所代码(Instrument.exchange 字段)
EXCHANGE_OF = {"XSHE": "XSHE", "XSHG": "XSHG"}


def to_rq(ts_code: str) -> str | None:
    """`000001.SZ` → `000001.XSHE`; 不支持的后缀返回 None"""
    if not ts_code or "." not in ts_code:
        return None
    num, suf = ts_code.rsplit(".", 1)
    rq_suf = TS2RQ.get(suf.upper())
    return f"{num}.{rq_suf}" if rq_suf else None


def from_rq(order_book_id: str) -> str | None:
    """`000001.XSHE` → `000001.SZ`; 不支持的后缀返回 None"""
    if not order_book_id or "." not in order_book_id:
        return None
    num, suf = order_book_id.rsplit(".", 1)
    ts_suf = RQ2TS.get(suf.upper())
    return f"{num}.{ts_suf}" if ts_suf else None


def to_rq_many(ts_codes) -> list:
    """批量转换, 自动丢弃不支持项(北交所等)"""
    return [r for r in (to_rq(c) for c in ts_codes) if r]


def from_rq_many(order_book_ids) -> list:
    return [r for r in (from_rq(c) for c in order_book_ids) if r]


@lru_cache(maxsize=8192)
def exchange_of(order_book_id: str) -> str:
    """米筐交易所代码; 未知返回空串"""
    if not order_book_id or "." not in order_book_id:
        return ""
    return EXCHANGE_OF.get(order_book_id.rsplit(".", 1)[1].upper(), "")


def is_stock(order_book_id: str) -> bool:
    """是否 A 股个股(排除指数/虚拟基准)"""
    return exchange_of(order_book_id) in ("XSHE", "XSHG")
