# -*- coding: utf-8 -*-
"""采集指数日线面板（tushare index_daily, 增量续跑）

用途: rqalpha_mod_ticai 的宽基基准(沪深300/中证全指等)。缺失时策略模拟
框架仍可用 —— 只是不提供宽基基准, 仅有自建打板基准 DBBNCH(全A等权)。

产物: market.index_panel
  trade_date, ts_code, open, high, low, close, pre_close, pct_chg, vol, amount
  vol 单位=手(与 daily_panel 一致); amount 单位=千元(tushare index_daily口径)

用法:
  python collect/fetch_index_panel.py            # 增量: 补齐至最新交易日
  python collect/fetch_index_panel.py --start 20191128
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import START_DATE, get_pro  # noqa: E402
from datastore import path_of, save  # noqa: E402

pro = get_pro()
OUT = path_of("market.index_panel")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 基准指数清单(与 rqalpha_mod_ticai._INDEX_NAMES 对齐)
INDEXES = ["000300.SH", "000985.SH", "000001.SH", "399006.SZ"]


def fetch_one(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """单指数日线; tushare index_daily 单次上限约 8000 行, 分段拉取"""
    frames = []
    cur = start
    while cur <= end:
        seg_end = min(end, str(int(cur) + 10000))     # 粗略分段, 后面按交易日裁
        for attempt in range(3):
            try:
                df = pro.index_daily(ts_code=ts_code, start_date=cur,
                                     end_date=seg_end)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  {ts_code} {cur}~{seg_end} 失败: {e}")
                    df = None
                time.sleep(1.5 * (attempt + 1))
        if df is not None and len(df):
            frames.append(df)
            cur = str(int(df["trade_date"].max()) + 1)
        else:
            break
        time.sleep(0.2)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["trade_date"])


def main():
    start = START_DATE
    if "--start" in sys.argv:
        start = sys.argv[sys.argv.index("--start") + 1]
    today = pd.Timestamp.now().strftime("%Y%m%d")

    existing = None
    done_upto = {}
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        for c, g in existing.groupby("ts_code"):
            done_upto[c] = str(g["trade_date"].max())
        print(f"已有数据 {len(existing)} 行, 覆盖 "
              f"{sorted(done_upto.items())}")

    buf = []
    for idx in INDEXES:
        s = str(int(done_upto.get(idx, "0")) + 1) if idx in done_upto else start
        if s > today:
            print(f"  {idx} 已至最新, 跳过")
            continue
        print(f"  采集 {idx} {s}~{today} ...")
        df = fetch_one(idx, s, today)
        if df is None or not len(df):
            print(f"  {idx} 无新增")
            continue
        df["pre_close"] = df["close"].shift(-1)      # index_daily 按日期降序
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100
        buf.append(df)
        print(f"  {idx} 新增 {len(df)} 行")
        time.sleep(0.3)

    if not buf:
        print("无新增数据")
        return
    new = pd.concat(buf, ignore_index=True)
    cols = ["trade_date", "ts_code", "open", "high", "low", "close",
            "pre_close", "pct_chg", "vol", "amount"]
    new = new.reindex(columns=cols)
    merged = (pd.concat([existing, new], ignore_index=True)
              if existing is not None else new)
    merged = merged.drop_duplicates(subset=["trade_date", "ts_code"])
    merged = merged.sort_values(["ts_code", "trade_date"])
    save("market.index_panel", merged)
    print(f"保存 {len(merged)} 行 → {OUT}")


if __name__ == "__main__":
    main()
