# -*- coding: utf-8 -*-
"""现实格口径（研究02无前视格）— 唯一出处

大热点(独占涨停≥8家) + 炸板≥1次后回封 + 炸板≤3次 + 最后封板≤11:00(午前回封)
v2归属下历史均值 +3.64%/笔 (n=4237, 日聚类t=36.0)。

输入为事件富化表口径(events_enriched + attribution + theme_day merge后),
需含列: zt_cnt / open_times / last_time / is_yizi / is_st。
盘中poller基于akshare涨停池的候选识别是该口径的实时近似（池内无
is_yizi/is_st字段, 由池自身过滤承担）。
"""
import pandas as pd

from core.times import is_before


def reality_mask(df: pd.DataFrame) -> pd.Series:
    """无前视现实格: 大热点+炸板早回封+午前+炸板≤3
    封板时间比较走 core/times.py: 缺失一律判 False, 不随 pandas 版本漂移。
    原写法 `lastm < '140000'` 与 `lastm <= '110000'` 双条件中后者恒更严,
    前者无独立作用, 故合并为单一午前判据(结果等价)。"""
    return ((df["zt_cnt"] >= 8) & (df["open_times"] >= 1) &
            (df["open_times"] <= 3) &
            is_before(df["last_time"], "110000") &
            (~df["is_yizi"]) & (~df["is_st"]))
