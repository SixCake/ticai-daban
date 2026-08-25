# -*- coding: utf-8 -*-
"""时序窗口差分：涨速与题材热度趋势的通用原语"""


def window_diff(series, secs: int, t: float) -> float:
    """series: 按时间升序的 (ts, value) 序列(deque/list)
    返回 最新值 − secs秒前值；窗口内无样本时回退到最早样本；空序列返回0"""
    if not series:
        return 0.0
    now_v = series[-1][1]
    target = t - secs
    for ts, v in series:
        if ts >= target:
            return round(now_v - v, 2)
    return round(now_v - series[0][1], 2)
