# -*- coding: utf-8 -*-
"""题材波次面板（归因自由活跃面板 + 波段切分，单一出处）

为何独立成模块: 波次口径必须完全一致，否则 build/theme_daily.py（离线落盘）
与 apps/poller.py（盘中实时）会算出不同的 wave_no，看板与复盘页互相矛盾。
切波逻辑在 core.cycle.segment_waves，本模块只负责建面板与调用它。

口径（研究36~39 链路，不可随意改）:
  · 活跃面板用 kpl_events 当日**全部** theme 标注直标，不经独占投票 ——
    theme.day 的独占 zt_cnt 只取该股首标签，受标签排序随机性污染，且
    「题材在场但独占为0」的日子会从 theme.day 里消失；
  · 在场 = 关联涨停家数 ≥ core.cycle.WAVE_MIN_ZT；
  · 新波 = 在场日 且 与上一「有任何涨停的日子」空档 ≥ WAVE_COOLDOWN
    个交易日（研究38 定档；cooldown=2/3 会把同波拖动误判成新波）；
  · 波次 = 在 [T-WAVE_LOOKBACK, T] 窗口内开启的波段个数，**不是全历史
    累计**（研究40 定档：全历史累计在看板实际面对的近期样本里方向反置）；
  · ths 源无 kpl 标注 → 返回空面板，波次置空（旧口径列不受影响）。

用法:
    from core.theme_wave import activity_panel, wave_table, wave_state
    panel = activity_panel()                    # 全历史题材-日关联家数
    wt = wave_table(panel, dates)               # 加 wave_no 列(滚动窗口)
    base = wave_state(panel, dates)             # {concept: (starts, last_pos, zt)}
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_SOURCE  # noqa: E402
from core.attribute import split_themes  # noqa: E402
from core.cycle import (WAVE_COOLDOWN, WAVE_LOOKBACK, WAVE_MIN_ZT,  # noqa: E402
                        segment_waves)
from datastore import load  # noqa: E402

_EMPTY = pd.DataFrame(columns=["trade_date", "concept_code", "zt_all"])


def activity_panel() -> pd.DataFrame:
    """归因自由题材-日活跃面板: (trade_date, concept_code) → zt_all

    zt_all = 当日该题材被 kpl 标注的涨停家数（全 tag，一股多题材各计一次）。
    """
    if CONCEPT_SOURCE != "kpl":
        return _EMPTY.copy()
    k = load("limitup.kpl_events",
             columns=["trade_date", "ts_code", "tag", "theme"])
    z = k[(k["tag"] == "涨停") & k["theme"].notna()]
    rows = []
    for d, th in zip(z["trade_date"], z["theme"]):
        for t in split_themes(th):
            rows.append((d, t))
    if not rows:
        return _EMPTY.copy()
    e = pd.DataFrame(rows, columns=["trade_date", "concept_code"])
    return (e.groupby(["trade_date", "concept_code"]).size()
            .rename("zt_all").reset_index())


def wave_table(panel: pd.DataFrame, dates: list,
               lookback: int | None = WAVE_LOOKBACK) -> pd.DataFrame:
    """给活跃面板加 wave_no 列（研究38/39/40 验证口径）

    dates:    升序交易日历（用于把日期换算成下标算空档）。面板里不在日历内
              的行会被剔除。切波在**所有有涨停的日子**上算，在场日才能开新波。
    lookback: 回看交易日数，波次 = 窗口内开启的波段个数（研究40 定档）。
    """
    if panel.empty:
        out = panel.copy()
        out["wave_no"] = pd.Series(dtype=int)
        return out
    dpos = {d: i for i, d in enumerate(dates)}
    act = panel[panel["trade_date"].isin(dpos)].copy()
    wno = []
    for _, g in act.groupby("concept_code", sort=False):
        g = g.sort_values("trade_date")
        pos = g["trade_date"].map(dpos).tolist()
        active = (g["zt_all"] >= WAVE_MIN_ZT).tolist()
        wno.append(pd.Series(segment_waves(pos, active, WAVE_COOLDOWN,
                                           lookback), index=g.index))
    act["wave_no"] = pd.concat(wno).sort_index() if wno else pd.Series(
        dtype=int)
    return act


def wave_state(panel: pd.DataFrame, dates: list,
               lookback: int | None = WAVE_LOOKBACK) -> dict:
    """盘中基线: {concept_code: (starts, last_pos, last_zt_all)}

    starts     = 该题材在 [最后日-lookback, 最后日] 窗口内的波段起点
                 交易日下标列表（滚动波次需要起点列表，不是单个波次）;
    last_pos   = 最后一个「有任何涨停」的交易日下标，供盘中判断今日
                 与它的空档是否 ≥ WAVE_COOLDOWN（是则今日若在场即为新一波）。
    """
    wt = wave_table(panel, dates, lookback)
    if wt.empty:
        return {}
    dpos = {d: i for i, d in enumerate(dates)}
    out = {}
    for k, g in wt.groupby("concept_code", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)
        pos = g["trade_date"].map(dpos).tolist()
        active = (g["zt_all"] >= WAVE_MIN_ZT).tolist()
        # 累计切波找起点（与 segment_waves 同口径），再取窗口内的
        starts, prev = [], None
        for p, a in zip(pos, active):
            if prev is None or (a and p - prev - 1 >= WAVE_COOLDOWN):
                starts.append(p)
            prev = p
        last = pos[-1]
        lo = last - lookback if lookback is not None else -1
        out[k] = ([s for s in starts if lo < s <= last], last,
                  int(g["zt_all"].iloc[-1]))
    return out


def next_wave_no(base: tuple | None, cur_pos: int, zt_all: int,
                 lookback: int | None = WAVE_LOOKBACK) -> int:
    """盘中推今日波次（与 segment_waves 同口径，滚动回看窗口）

    base:    wave_state 给出的 (starts, last_pos, last_zt_all)，可为 None
    cur_pos: 今日在交易日历中的下标
    zt_all:  今日该题材关联涨停家数
    """
    starts, last_pos = (base[0], base[1]) if base else ([], None)
    # 今日是否开新波: 在场 且 与上一有涨停日空档≥cooldown
    new_start = zt_all >= WAVE_MIN_ZT and (
        last_pos is None or cur_pos - last_pos - 1 >= WAVE_COOLDOWN)
    lo = cur_pos - lookback if lookback is not None else -1
    ss = [s for s in starts if lo < s <= cur_pos]
    if new_start:
        ss = ss + [cur_pos]
    return len(ss) if len(ss) >= 1 else 1
