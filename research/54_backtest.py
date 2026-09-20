# -*- coding: utf-8 -*-
"""研究54: 决策链路三杠杆方案历史回测（牛/熊/震荡三段）

把已落地 strategy.py 的三杠杆方案用历史大样本回测, 三段市况验证:
  杠杆① 入场价位闸: fulluniv_panel e3档(在3%买入, 低位)
  杠杆② E3续持闸: 封板票E3≥0.20续持(次日开盘卖), E3<0.20尾盘卖(当日收盘)
  杠杆③ 不止损: 不封板票持到收盘(当日收盘)

口径(吸取研究37教训: 买入基准不可用涨停价):
  · 买入基准 = e3_px(在3%涨幅买入, 可执行)
  · 封板续持 = next_open_ret(次日开盘卖, 真实可成交)
  · 不封板/E3低 = day_close/e3_px-1(当日收盘卖)
判据: 盈亏比优先, 三段市况(牛/熊/震荡)方向一致

用法: python research/54_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.exit_rules import pl_stats                        # noqa: E402

E3_HOLD = 0.20           # E3续持阈值(同 strategy.py)


def build_regimes(dates):
    """按月均全A中位涨幅三分位切牛/熊/震荡(同研究31/40)"""
    dp = pd.read_parquet(ROOT / "data" / "market" / "1d" / "daily_panel.parquet",
                         columns=["trade_date", "pct_chg"])
    dp["trade_date"] = dp["trade_date"].astype(str)
    mr = dp.groupby("trade_date")["pct_chg"].median()
    mon = mr.groupby(mr.index.str[:6]).mean().sort_values()
    n = len(mon)
    bear = set(mon.index[:n // 3])
    bull = set(mon.index[-n // 3:])
    return {dt: ("熊" if dt[:6] in bear else "牛" if dt[:6] in bull else "震荡")
            for dt in dates}


def main():
    # 杠杆①: fulluniv_panel e3档(在3%买入, 低位入场价位闸)
    d = pd.read_parquet(ROOT / "data" / "factor" / "fulluniv_panel.parquet",
                        columns=["trade_date", "ts_code", "sealed",
                                 "day_close", "e3_px"])
    d["trade_date"] = d["trade_date"].astype(str)
    d = d[d["e3_px"] > 0].copy()

    # 杠杆②: join E3封单强度 + 次日开盘收益
    ev = pd.read_csv("/tmp/e3.csv", dtype={"trade_date": str})
    d = d.merge(ev, on=["trade_date", "ts_code"], how="left")

    # 三杠杆离场收益
    def exit_ret(r):
        if r.sealed and pd.notna(r.E3) and r.E3 >= E3_HOLD \
                and pd.notna(r.next_open_ret):
            return r.next_open_ret * 100          # ② E3高续持(次日开盘卖)
        return (r.day_close / r.e3_px - 1) * 100  # ③ 不封板/E3低: 当日收盘
    d["ret"] = d.apply(exit_ret, axis=1)
    d["reg"] = d["trade_date"].map(build_regimes(d["trade_date"].unique()))

    print("# 研究54: 三杠杆方案历史回测(低位+E3续持+不止损)")
    print(f"宇宙: fulluniv e3档 {len(d)} 票·日 "
          f"({d['trade_date'].min()}~{d['trade_date'].max()})")

    n, wr, aw, al, ratio, exp = pl_stats(d["ret"].tolist())
    print(f"\n全样本: {n} 胜率{wr*100:.1f}% 盈亏比{ratio:.2f} 期望{exp:+.2f}")

    print(f"\n## 三段市况验证")
    print(f"{'市况':<6}{'样本':>9}{'胜率%':>8}{'盈亏比':>8}{'期望%':>8}")
    ok = 0
    for rg in ("牛", "熊", "震荡"):
        g = d[d["reg"] == rg]
        n2, wr2, aw2, al2, ratio2, exp2 = pl_stats(g["ret"].tolist())
        flag = "✓" if ratio2 >= 1 else "✗"
        ok += ratio2 >= 1
        print(f"{rg:<6}{n2:>9}{wr2*100:>8.1f}{ratio2:>8.2f}{exp2:>8.2f} {flag}")
    print(f"\n盈亏比≥1的市况: {ok}/3 段")


if __name__ == "__main__":
    main()
