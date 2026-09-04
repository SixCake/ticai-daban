# -*- coding: utf-8 -*-
"""akshare 当日涨停池（东财口径）

注意: 未开盘日期请求会返回上一交易日数据, 调用方须自行定标。
"""
import time

import pandas as pd

from core.codes import ts_code_of
from core.times import hhmmss6


def fetch_pool(date: str) -> pd.DataFrame | None:
    """拉取指定日期涨停池, 3次重试退避; 空/失败返回None"""
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.stock_zt_pool_em(date=date)
            return df if df is not None and len(df) else None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def norm_pool(df: pd.DataFrame) -> pd.DataFrame:
    """归一化: 补 ts_code / first_time / last_time(6位字符串)
    封板时间走 core/times.py 使缺失保持 NA — 否则 pandas 2.x 下会得
    '000nan', 而下游 apps/poller.py 用字符串比较算早封板占比时
    '000nan' <= '094500' 为 True → 未封板票被误算成早封板。"""
    p = df.copy()
    p["ts_code"] = p["代码"].astype(str).str.zfill(6).map(ts_code_of)
    p["first_time"] = hhmmss6(p["首次封板时间"])
    p["last_time"] = hhmmss6(p["最后封板时间"])
    return p
