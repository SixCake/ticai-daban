# -*- coding: utf-8 -*-
"""TicaiDataSource 单测（Spec Test Plan #3）

覆盖:
  1. history_bars(1d) 与 market.daily_panel 逐值对齐（取 3 只票）
  2. get_instruments() 数量与 qmt_names.json 一致（剔除北交所后）
  3. .SZ/.SH ↔ .XSHE/.XSHG 双向映射往返无损
  4. history_bars(bar_count=None) 返回截至 dt 的全部 bar（rqalpha 语义）
  5. is_suspended 对盘中 datetime 与 date 判定一致（不被时分秒误导）

运行: .venv/bin/python tests/test_ticai_data_source.py
不依赖 pytest（纯 assert + 退出码），便于在任意环境直接跑。
"""
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rqalpha_mod_ticai.codes import from_rq, to_rq  # noqa: E402
from rqalpha_mod_ticai.data_source import TicaiDataSource  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAIL.append(name)


def main():
    ds = TicaiDataSource()
    panel = pd.read_parquet(ROOT / "data/market/1d/daily_panel.parquet")

    # ---- 1. history_bars 与 daily_panel 逐值对齐 ----
    print("1) history_bars(1d) vs daily_panel 逐值对齐")
    for ts in ["000001.SZ", "600519.SH", "300750.SZ"]:
        rq = to_rq(ts)
        sub = panel[panel["ts_code"] == ts].sort_values("trade_date")
        if sub.empty:
            check(f"{ts} 面板有数据", False, "面板无该票")
            continue
        last_day = sub["trade_date"].iloc[-1]
        dt = pd.Timestamp(str(last_day))
        n = min(22, len(sub))
        bars = ds.history_bars(
            ds.get_instruments([rq])[0], n, "1d",
            ["datetime", "open", "high", "low", "close", "volume"], dt)
        exp = sub.tail(n)
        ok_len = bars is not None and len(bars) == n
        check(f"{ts} 长度={n}", ok_len,
              f"got {0 if bars is None else len(bars)}")
        if not ok_len:
            continue
        ok_close = np.allclose(bars["close"], exp["close"].values, atol=1e-6)
        ok_open = np.allclose(bars["open"], exp["open"].values, atol=1e-6)
        ok_high = np.allclose(bars["high"], exp["high"].values, atol=1e-6)
        ok_low = np.allclose(bars["low"], exp["low"].values, atol=1e-6)
        # volume: 面板为手, DataSource 转成股(×100)
        ok_vol = np.allclose(bars["volume"], exp["vol"].values * 100, atol=1e-6)
        check(f"{ts} close/open/high/low 对齐",
              ok_close and ok_open and ok_high and ok_low)
        check(f"{ts} volume=面板vol×100", ok_vol)

    # ---- 2. get_instruments 数量 vs 面板 ----
    print("2) get_instruments 数量 vs daily_panel(去北交所)")
    # instrument 以面板为准(qmt_names 只供显示名, 缺失时回退用代码作名);
    # 故期望 = 面板代码经 to_rq 后非 None 的集合
    panel_codes = set(panel["ts_code"].unique())
    expect = {to_rq(c) for c in panel_codes if to_rq(c) is not None}
    ins = ds.get_instruments()
    # 只比较个股(排除指数/基准)
    idx = {"000300.XSHG", "000985.XSHG", "000001.XSHG", "399006.XSHE",
           "DBBNCH.XSHG"}
    got_stocks = {i.order_book_id for i in ins} - idx
    check(f"个股 instrument 数 == 面板去北交所 ({len(expect)})",
          got_stocks == expect,
          f"got {len(got_stocks)} expect {len(expect)} "
          f"diff {list(got_stocks ^ expect)[:5]}")

    # ---- 3. 代码映射往返 ----
    print("3) .SZ/.SH ↔ .XSHE/.XSHG 往返无损")
    for ts in ["000001.SZ", "600519.SH", "300750.SZ", "688981.SH"]:
        check(f"{ts} 往返", from_rq(to_rq(ts)) == ts,
              f"{ts} -> {to_rq(ts)} -> {from_rq(to_rq(ts))}")
    check("北交所 .BJ 返回 None", to_rq("830799.BJ") is None)

    # ---- 4. bar_count=None 返回全部 ----
    print("4) history_bars(bar_count=None) 语义")
    rq = to_rq("000001.SZ")
    sub = panel[panel["ts_code"] == "000001.SZ"].sort_values("trade_date")
    last_day = sub["trade_date"].iloc[-1]
    dt = pd.Timestamp(str(last_day))
    allbars = ds.history_bars(ds.get_instruments([rq])[0], None, "1d",
                              ["datetime", "close"], dt)
    check(f"bar_count=None 长度 == 面板行数 ({len(sub)})",
          allbars is not None and len(allbars) == len(sub),
          f"got {0 if allbars is None else len(allbars)}")
    one = ds.history_bars(ds.get_instruments([rq])[0], 1, "1d",
                          ["datetime", "close"], dt)
    check("bar_count=1 长度==1 且为最后一根",
          one is not None and len(one) == 1
          and int(one["datetime"][0]) == int(last_day) * 1000000)

    # ---- 5. is_suspended 盘中 datetime 与 date 一致 ----
    print("5) is_suspended 不被时分秒误导")
    d = sub["trade_date"].iloc[-5]
    dd = datetime.strptime(str(d), "%Y%m%d")
    a = ds.is_suspended(rq, [dd])                       # 午夜
    b = ds.is_suspended(rq, [dd.replace(hour=10, minute=30)])  # 盘中
    c = ds.is_suspended(rq, [date(dd.year, dd.month, dd.day)])
    check(f"{d} 午夜/盘中/date 判定一致", a == b == c, f"{a} {b} {c}")
    check(f"{d} 该票当日不应停牌", a == [False], f"{a}")

    # ---- 6. 涨停买不进约束 active（sys_simulation price_limit）----
    print("6) 涨停买不进约束 (reaches_limit_up)")
    from rqalpha.utils.price_limits import reaches_limit_up
    tick = 0.01
    lu = 10.00
    check("价格==涨停价 → 拒买", reaches_limit_up(lu, lu, tick) is True
          or bool(reaches_limit_up(lu, lu, tick)))
    check("价格略低于涨停价 → 不拒", not reaches_limit_up(lu - 0.02, lu, tick))
    check("价格远低于涨停价 → 不拒", not reaches_limit_up(lu * 0.9, lu, tick))

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} 项 -> {FAIL}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
