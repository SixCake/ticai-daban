# -*- coding: utf-8 -*-
"""定稿卖出规则的离线模拟 —— 唯一出处

定稿卖出端(见 strategies/v5_daban/strategy.py 与研究36)分层:
  回落止损5%  参考价 = max(买入价 pb, 买入当日最高价)
  P0 跌停开/深低开≤-5%   跌破VWAP即卖 / 线上方观察至首次回落即卖
  P1 回封观察            深跌5% / 3穿插 / 回封续持 / 14:57强平
  P2 10:10 封板唯一判据   10:10 后未封板即卖, 已封板续持
  P3 冲高回落            高2低1
  封板续持无上限

本模块只模拟**可用分时数据复现的三条**(止损5% / P2 10:10 / 14:57强平),
P0/P1/P3 需要 VWAP 与更细的形态判定, 暂不模拟 —— 调用方必须知道这是
部分模拟, 不得当作完整策略回测。

口径要点:
  · 收益以买入价 pb 为基准, 返回百分比
  · 封死判据 SEAL_EPS=0.9995(贴死涨停价), 同 core/theme_signal
  · 分时轨迹不足(末点<14:57 且未触发上述规则)时回退收盘价, 并显式标记
    回退原因 —— **不冒充完整模拟**(intraday_px 来自 radar_log, 只录
    pct≥1% 或 prob≥0.2 的票, 实测 58% 的票轨迹在 14:00 前就断了)

调用方: research/36_s_redefine.py(方案评估) / research/backfill_tsig.py
(历史信号回填)。两处共用本模块避免口径分叉。
"""
STOP_LOSS = 0.95                    # 回落止损 5%
P2_SEC = 10 * 3600 + 10 * 60        # P2: 10:10 封板唯一判据
FLAT_SEC = 14 * 3600 + 57 * 60      # 14:57 强平
SEAL_EPS = 0.9995                   # 封死判据(贴死涨停价)


def _sec(hms) -> int:
    s = str(hms or "").replace(":", "")
    if len(s) < 6 or not s[:6].isdigit():
        return 0
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def simulate_exit(pts: list, pb: float, day_hi: float,
                  limit_px: float, close_px: float,
                  hi_pts: list | None = None) -> tuple:
    """按定稿卖出规则逐点模拟离场 → (收益%, 离场原因)

    pts      离场日分时轨迹 [[HHMMSS|HH:MM:SS, price, ...], ...]
    pb       买入价(挂限价单成交价)
    day_hi   买入当日最高价(止损参考价用); 0/None 则退化为只用 pb
    limit_px 离场日涨停价(判封死用); 0 则无法判封板
    close_px 离场日收盘价(轨迹不足时的回退价)
    hi_pts   **可选**。与 pts 对齐的最高价 [[HHMMSS, high], ...]。
             提供则止损参考价随分时推进动态取 max(pb, 已过bar最高价)。
             **当日盘中入场场景必须传此参**(如研究39 半路板):
             入场与离场在同一日, 若用全天最高价作参考价则含未来信息,
             且对触板股而言全天高点≈涨停价 → 止损线高于入场价,
             第一根bar 就误触发止损(实测导致封板率63.5% 但胜率仅11.65%)。
             None 则用固定 day_hi —— 适用于 T+1 离场场景(买入日已完结,
             无未来信息), 即研究36 / backfill_tsig 的原口径。
    """
    if not pb or pb <= 0:
        return None, "无买价"
    # hi_pts 提供时: 参考价从 pb 起算, 随已过 bar 的最高价上移(无前视)
    hi_map = {}
    if hi_pts:
        for e in hi_pts:
            if len(e) >= 2 and e[1] and e[1] > 0:
                hi_map[_sec(e[0])] = float(e[1])
    ref = pb if hi_pts else (max(pb, day_hi) if day_hi and day_hi > 0 else pb)
    stop = ref * STOP_LOSS
    sealed = False
    for e in sorted(pts or [], key=lambda x: str(x[0])):
        if len(e) < 2 or not e[1] or e[1] <= 0:
            continue
        px, t = float(e[1]), _sec(e[0])
        if hi_map:                       # 动态上移止损参考价
            h = hi_map.get(t)
            if h and h > ref:
                ref = h
                stop = ref * STOP_LOSS
        if px <= stop:
            return (px / pb - 1) * 100, "回落止损5%"
        if limit_px > 0 and px >= limit_px * SEAL_EPS:
            sealed = True
        if t >= P2_SEC and not sealed:
            return (px / pb - 1) * 100, "P2 10:10未封板"
        if t >= FLAT_SEC:
            return ((px / pb - 1) * 100,
                    "14:57强平(封板续持)" if sealed else "14:57强平")
    if close_px and close_px > 0:
        return ((close_px / pb - 1) * 100,
                "封板续持·回退收盘" if sealed else "轨迹不足回退收盘")
    return None, "无数据"


def pl_stats(values: list) -> tuple:
    """盈亏统计 → (样本数, 胜率, 平均盈利, 平均亏损, 盈亏比, 期望)

    胜率   = 收益>0 的占比
    盈亏比 = 平均盈利 / |平均亏损|
    期望   = 胜率×平均盈利 + (1-胜率)×平均亏损
    """
    v = [x for x in values if x is not None]
    if not v:
        return 0, None, None, None, None, None
    w = [x for x in v if x > 0]
    l = [x for x in v if x <= 0]
    aw = sum(w) / len(w) if w else 0.0
    al = sum(l) / len(l) if l else 0.0
    ratio = (aw / abs(al)) if al < 0 else None
    exp = (len(w) / len(v)) * aw + (len(l) / len(v)) * al
    return len(v), len(w) / len(v), aw, al, ratio, exp
