# -*- coding: utf-8 -*-
"""交易时段判定（唯一出处）

两种口径语义不同，勿混用：
  is_trading_hours — 严格连续竞价时段（雷达扫描用，午休休眠）
  is_polling_hours — 宽口径含集合竞价与午休（poller用，涨停池快照午休不变但仍轮询）
"""
from datetime import datetime


def is_trading_hours(now: datetime) -> bool:
    """工作日 09:25-11:30 与 13:00-15:00"""
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H%M")
    return "0925" <= hm <= "1130" or "1300" <= hm <= "1500"


def is_polling_hours(now: datetime) -> bool:
    """工作日 09:15-15:05（含集合竞价与午休）"""
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return 915 <= hm <= 1505
