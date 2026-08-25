# -*- coding: utf-8 -*-
"""事件富化: 涨停事件 × 日线面板 → 一字板标记 + T+1三口径收益

产物: limitup.events_enriched
  事件字段 + is_yizi + first_min + next_open_ret/next_close_ret/next_high_ret/next_low_ret
  + next_is_yizi（T+1一字影响次日卖出可行性）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load, save


def main():
    ev = load("limitup.events")
    panel = load("market.daily_panel")
    print(f"事件 {len(ev)} 行, 面板 {len(panel)} 行")

    # --- 当日OHLC → 一字板判定 ---
    day = panel[["trade_date", "ts_code", "open", "high", "low", "close"]].rename(
        columns={"open": "day_open", "high": "day_high", "low": "day_low",
                 "close": "day_close"})
    ev = ev.merge(day, on=["trade_date", "ts_code"], how="left")
    ev["is_yizi"] = ((ev["day_high"] == ev["day_low"]) &
                     (ev["day_low"] == ev["day_close"]))
    # 首封时间→分钟数(9:25=0基准, 便于排序)
    ft = ev["first_time"].astype(str).str.zfill(6)
    ev["first_min"] = (ft.str[:2].astype(int) * 60 + ft.str[2:4].astype(int)
                       + ft.str[4:].astype(int) / 60 - (9 * 60 + 25))

    # --- T+1收益 ---
    dates = np.array(sorted(panel["trade_date"].unique()))
    pos = np.searchsorted(dates, ev["trade_date"].values, side="right")
    nd = np.where(pos < len(dates), dates[np.minimum(pos, len(dates) - 1)], None)
    ev["next_date"] = nd
    t1 = panel[["trade_date", "ts_code", "open", "high", "low", "close",
                "pre_close"]].rename(
        columns={"trade_date": "next_date", "open": "t1_open", "high": "t1_high",
                 "low": "t1_low", "close": "t1_close", "pre_close": "t1_preclose"})
    ev = ev.merge(t1, on=["next_date", "ts_code"], how="left")
    has = ev["t1_preclose"].notna()
    for src, dst in [("t1_open", "next_open_ret"), ("t1_close", "next_close_ret"),
                     ("t1_high", "next_high_ret"), ("t1_low", "next_low_ret")]:
        ev.loc[has, dst] = ev.loc[has, src] / ev.loc[has, "t1_preclose"] - 1
    ev["next_is_yizi"] = ((ev["t1_high"] == ev["t1_low"]) &
                          (ev["t1_low"] == ev["t1_close"]))

    keep = [c for c in ev.columns if not c.startswith(("t1_", "day_"))]
    ev = ev[keep]
    p = save("limitup.events_enriched", ev)
    print(f"富化完成 {len(ev)} 行 → {p}")
    print(f"  一字板占比 {ev['is_yizi'].mean():.3%}, T+1可得 {ev['next_open_ret'].notna().mean():.3%}")


if __name__ == "__main__":
    main()
