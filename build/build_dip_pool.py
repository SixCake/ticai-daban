# -*- coding: utf-8 -*-
"""首板回调5日线低吸候选池 — ma5_dip 策略私有 feed 生产者

对每个目标交易日 T(池生效日), 用严格早于 T 的数据选股:
  · 近 BOARD_WIN 个交易日内出现过首板(limitup.events_enriched limit_times==1)
  · 首板之后至 T-1 未再涨停(处于回调态)
  · 排除 ST/退市/北交所/一字首板
  · T-1 收盘未破 MA5(close >= ma5×(1-BREAK_TOL), 已破位的不做低吸候选)
每票预计算 ma5(T-1..T-5收盘均值)与 atr(Wilder ATR14, 截至T-1) ——
策略盘中直接用, live/回放/回测同构, 且避免 live 盘中 history_bars
拿不到当日 bar 的口径漂移(见 SPEC/探索结论)。

产物: feeds.write_feed("dip_pool", date=T, strategy="ma5_dip")
  条目 topic=rqalpha口径代码, extra={ma5, atr, board_date, name, prev_close}
  ts=T-1日20:00 epoch(历史回填必须显式给, 否则被时间戳闸门滤掉,
  同 apps/ai_feed.py::_ts_for 的实测踩坑)

CLI:
  python build/build_dip_pool.py                          # 为最新数据日的下一交易日生成(daily_update 接线)
  python build/build_dip_pool.py --start 20260601 --end 20260917   # 批量回填历史(回测用)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load  # noqa: E402
from rqalpha_mod_ticai import feeds  # noqa: E402
from rqalpha_mod_ticai.codes import to_rq  # noqa: E402

BOARD_WIN = 8        # 首板回看窗口(交易日)
MA_N = 5             # 均线天数
ATR_N = 14           # Wilder ATR 周期
ATR_BARS = 60        # ATR 计算最多回看的 bar 数
BREAK_TOL = 0.01     # T-1 收盘破线容差(与策略同值): close < ma5×(1-tol) 不入池
FEED_NAME = "dip_pool"
STRATEGY = "ma5_dip"


def _open_dates() -> list:
    cal = load("meta.trade_cal")
    return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str))


def _ts_prev_evening(prev_day: str) -> float:
    """feed 条目时间戳 = T-1 日 20:00(生产者产出时刻语义)"""
    return datetime.strptime(f"{prev_day} 20:00:00",
                             "%Y%m%d %H:%M:%S").timestamp()


def _atr_wilder(high, low, pre_close, n=ATR_N) -> float | None:
    """Wilder ATR(n); 样本不足返回 None。pre_close 与 high/low 同长对齐"""
    if len(high) < n + 1:
        return None
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - pre_close),
                               np.abs(low - pre_close)))
    atr = float(tr[:n].mean())
    for x in tr[n:]:
        atr = (atr * (n - 1) + float(x)) / n
    return round(atr, 3)


def build_pool(T: str, dates: list, ev: pd.DataFrame,
               panel_by_code: dict) -> list:
    """目标日 T 的候选池条目列表(严格用 <T 数据)"""
    i = dates.index(T)
    if i < BOARD_WIN + ATR_N:
        return []
    prev = dates[i - 1]
    win = dates[i - BOARD_WIN:i]              # T-1 .. T-BOARD_WIN
    wset = set(win)

    boards = ev[ev["trade_date"].isin(wset) & (ev["limit_times"] == 1)
                & (~ev["is_yizi"].astype(bool)) & (~ev["is_st"].astype(bool))]
    boards = boards[~boards["ts_code"].str.endswith(".BJ")]
    boards = boards[~boards["name"].astype(str).str.contains("ST|退")]
    if boards.empty:
        return []
    last_board = boards.groupby("ts_code")["trade_date"].max()
    binfo = boards.sort_values("trade_date").groupby("ts_code").last()

    # 首板后至 T-1 又涨停的剔除(连板/二波, 非首板回调态)
    after = ev[(ev["trade_date"] > last_board.reindex(ev["ts_code"]).values)
               & (ev["trade_date"] < T)]
    rebored = set(after["ts_code"]) & set(last_board.index)

    ts_prev = _ts_prev_evening(prev)
    out = []
    for code in sorted(set(last_board.index) - rebored):
        bars = panel_by_code.get(code)
        if bars is None:
            continue
        bd, bh, bl, bc, bpc = bars
        j = np.searchsorted(bd, T)            # 严格 <T 的切片终点
        if j < max(MA_N, ATR_N + 1):
            continue
        closes = bc[j - ATR_BARS:j] if j > ATR_BARS else bc[:j]
        highs = bh[j - ATR_BARS:j] if j > ATR_BARS else bh[:j]
        lows = bl[j - ATR_BARS:j] if j > ATR_BARS else bl[:j]
        pcs = bpc[j - ATR_BARS:j] if j > ATR_BARS else bpc[:j]
        ma5 = float(closes[-MA_N:].mean())
        prev_close = float(closes[-1])
        if prev_close < ma5 * (1 - BREAK_TOL):
            continue                          # T-1 已破位, 不做候选
        atr = _atr_wilder(highs, lows, pcs)
        if atr is None or atr <= 0:
            continue
        rq = to_rq(code)
        if rq is None:
            continue
        out.append({
            "ts": ts_prev, "t": "20:00:00", "topic": rq, "score": None,
            "text": f"首板回调 {binfo.loc[code, 'name']}",
            "src": "rule",
            "extra": {"ma5": round(ma5, 3), "atr": atr,
                      "board_date": str(last_board[code]),
                      "name": str(binfo.loc[code, "name"]),
                      "prev_close": round(prev_close, 2)}})
    return out


def main():
    ap = argparse.ArgumentParser(description="ma5_dip 首板回调候选池生产者")
    ap.add_argument("--start", help="回填起点YYYYMMDD(池生效日)")
    ap.add_argument("--end", help="回填终点YYYYMMDD(默认=最新数据日的下一交易日)")
    args = ap.parse_args()

    dates = _open_dates()
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "name", "limit_times",
                       "is_yizi", "is_st"])
    ev["trade_date"] = ev["trade_date"].astype(str)
    ev = ev[ev["trade_date"] >= dates[max(0, len(dates) - 400)]]

    have = ev["trade_date"].max()             # 事件库最新日
    if args.start:
        lo = args.start
        hi = args.end or have
        targets = [d for d in dates if lo <= d <= hi]
    else:
        # 缺省: 为"最新数据日"的下一交易日生成(daily_update 盘后跑)
        nxt = next((d for d in dates if d > have), None)
        if nxt is None:
            print(f"交易日历无 {have} 之后的日期, 先更新 meta.trade_cal")
            return
        targets = [nxt]
    if not targets:
        print("目标日列表为空")
        return

    # 面板只加载窗口需要的段(池生效首日再往前 ATR_BARS+BOARD_WIN 个交易日)
    i0 = dates.index(targets[0])
    pfrom = dates[max(0, i0 - ATR_BARS - BOARD_WIN - 5)]
    pn = load("market.daily_panel",
              columns=["trade_date", "ts_code", "close", "high", "low",
                       "pre_close"])
    pn = pn[pn["trade_date"] >= pfrom]
    pn["trade_date"] = pn["trade_date"].astype(str)
    panel_by_code = {}
    for code, g in pn.sort_values("trade_date").groupby("ts_code"):
        panel_by_code[code] = (g["trade_date"].to_numpy(),
                               g["high"].to_numpy(float),
                               g["low"].to_numpy(float),
                               g["close"].to_numpy(float),
                               g["pre_close"].to_numpy(float))
    print(f"面板 {pfrom}~{pn['trade_date'].max()}, {len(panel_by_code)} 只")

    for T in targets:
        entries = build_pool(T, dates, ev, panel_by_code)
        p = feeds.write_feed(FEED_NAME, T, entries, strategy=STRATEGY)
        print(f"{T}: 候选 {len(entries)} 只 → {p}")


if __name__ == "__main__":
    main()
