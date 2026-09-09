# -*- coding: utf-8 -*-
"""交易时段判定（唯一出处）

两种口径语义不同，勿混用：
  is_trading_hours — 严格连续竞价时段（信号生产用，午休休眠）
  is_radar_hours   — 雷达驻留时段，比 is_trading_hours 早 10 分钟起（09:15），
                     用于竞价推送预热（订阅必须早于 09:25:00~04 撮合 tick 波）
  is_polling_hours — 宽口径含集合竞价与午休（poller用，涨停池快照午休不变但仍轮询）
"""
from datetime import datetime


def is_trading_hours(now: datetime) -> bool:
    """工作日 09:25-11:30 与 13:00-15:00"""
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H%M")
    return "0925" <= hm <= "1130" or "1300" <= hm <= "1500"


def is_radar_hours(now: datetime) -> bool:
    """工作日 09:15-11:30 与 13:00-15:00

    比 is_trading_hours 提前 10 分钟: 09:15~09:25 是竞价预热段，雷达
    在此段只建推送订阅不产信号。QMT 推送是增量流，全市场竞价撮合
    tick 在 09:25:00~09:25:04 一次性下发，订阅晚于这波就永久采不到
    竞价量（20260904/20260907 竞价量比全缺的根因）。"""
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H%M")
    return "0915" <= hm <= "1130" or "1300" <= hm <= "1500"


def is_polling_hours(now: datetime) -> bool:
    """工作日 09:15-15:05（含集合竞价与午休）"""
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return 915 <= hm <= 1505
