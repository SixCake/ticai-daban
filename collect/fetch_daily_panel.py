# -*- coding: utf-8 -*-
"""构建日线面板: tushare全量(默认) 或 lab_333前复权日线(可选加速) + tushare补尾

数据源优先级:
  1. 已有 market.daily_panel → 仅补尾
  2. 环境变量 LAB333_DAILY_DIR 指向前复权日线parquet目录(可选, 本地有则快) → 载入+补尾rebase
  3. 纯tushare pro.daily 逐日全量(官方除权调整口径, 无需rebase)

产物: market.daily_panel
  trade_date, ts_code, open, high, low, close, pre_close, pct_chg, open_ret, vol
  pct_chg = close/pre_close-1, open_ret = open/pre_close-1 (官方除权调整口径)
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import START_DATE, get_pro
from datastore import path_of

pro = get_pro()
OUT = path_of("market.daily_panel")
OUT.parent.mkdir(parents=True, exist_ok=True)
LAB333 = Path(os.environ["LAB333_DAILY_DIR"]) if os.environ.get("LAB333_DAILY_DIR") else None

# 个股前缀: SSE 60x/68x, SZSE 00x/30x
STOCK_PREFIX = {("SSE", "60"), ("SSE", "68"), ("SZSE", "00"), ("SZSE", "30")}
EXCH_MAP = {"SSE": "SH", "SZSE": "SZ"}


def load_lab333() -> pd.DataFrame:
    rows = []
    files = [f for f in LAB333.iterdir() if f.suffix == ".parquet"]
    picked = 0
    for f in files:
        code, exch = f.stem.split(".")
        if (exch, code[:2]) not in STOCK_PREFIX:
            continue
        df = pd.read_parquet(f, columns=["datetime", "open", "high", "low",
                                         "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"] >= START_DATE]
        if df.empty:
            continue
        df["ts_code"] = f"{code}.{EXCH_MAP[exch]}"
        df["trade_date"] = df["datetime"].dt.strftime("%Y%m%d")
        rows.append(df[["trade_date", "ts_code", "open", "high", "low",
                        "close", "volume"]])
        picked += 1
        if picked % 500 == 0:
            print(f"  已读 {picked} 只")
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.rename(columns={"volume": "vol"})
    # 前复权序列内 shift 得到 pre_close（跨历史除权日的比率=含权总收益，正确）
    panel = panel.sort_values(["ts_code", "trade_date"])
    panel["pre_close"] = panel.groupby("ts_code")["close"].shift(1)
    panel["pct_chg"] = (panel["close"] / panel["pre_close"] - 1) * 100
    panel["open_ret"] = (panel["open"] / panel["pre_close"] - 1) * 100
    print(f"lab_333 载入 {picked} 只, {len(panel)} 行, 至 {panel['trade_date'].max()}")
    return panel


def trade_days(start: str, end: str) -> list[str]:
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end,
                        is_open="1")
    return sorted(cal["cal_date"].tolist())


def fetch_days(days: list[str]) -> pd.DataFrame | None:
    """tushare逐日拉取daily(官方口径), 带重试退避"""
    if not days:
        return None
    buf = []
    for i, d in enumerate(days):
        df = None
        for attempt in range(3):
            try:
                df = pro.daily(trade_date=d)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [FAIL] {d}: {e}")
                time.sleep(2 * (attempt + 1))
        if df is not None and len(df):
            buf.append(df)
        time.sleep(0.15)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(days)} ({d})")
    if not buf:
        return None
    tail = pd.concat(buf, ignore_index=True)
    return tail[["trade_date", "ts_code", "open", "high", "low", "close",
                 "pre_close", "pct_chg", "vol"]]


def fetch_tail(last_date: str) -> pd.DataFrame | None:
    """tushare补尾, 按股票rebase到面板最后一日收盘价"""
    today = pd.Timestamp.now().strftime("%Y%m%d")
    days = trade_days(last_date, today)[1:]  # 不含last_date本身
    if not days:
        print("补尾: 无缺失日期")
        return None
    print(f"补尾: {len(days)} 个交易日 ({days[0]}~{days[-1]})")
    tail = fetch_days(days)
    if tail is None:
        return None

    # rebase: 每股 factor = 面板最后收盘 / 补尾首日pre_close
    base = (pd.read_parquet(OUT, columns=["ts_code", "trade_date", "close"])
            if OUT.exists() else None)
    if base is not None and len(base):
        last = base.loc[base.groupby("ts_code")["trade_date"].idxmax()]
        last_map = last.set_index("ts_code")["close"]
        first_day = tail["trade_date"].min()
        fd = tail[tail["trade_date"] == first_day].copy()
        fd["factor"] = fd["ts_code"].map(last_map) / fd["pre_close"]
        fmap = fd.set_index("ts_code")["factor"]
        for c in ["open", "high", "low", "close", "pre_close"]:
            tail[c] = tail[c] * tail["ts_code"].map(fmap)
        n_rebased = fmap.notna().sum()
        print(f"  rebase {n_rebased} 只, 新股(无历史) {tail['ts_code'].nunique() - n_rebased} 只")
    tail["open_ret"] = (tail["open"] / tail["pre_close"] - 1) * 100
    return tail


def build_history_tushare() -> pd.DataFrame:
    """无本地lab_333时, 纯tushare逐日构建全历史(官方除权调整口径)"""
    today = pd.Timestamp.now().strftime("%Y%m%d")
    days = trade_days(START_DATE, today)
    print(f"tushare全量历史: {len(days)} 个交易日 ({days[0]}~{days[-1]}), 预计数分钟")
    panel = fetch_days(days)
    if panel is None or not len(panel):
        raise RuntimeError("tushare历史拉取失败, 检查token权限/网络")
    panel = panel[panel["ts_code"].str[:1].isin(["6", "0", "3"])]  # 仅主板/创业板/科创板
    panel["open_ret"] = (panel["open"] / panel["pre_close"] - 1) * 100
    print(f"历史面板 {len(panel)} 行, {panel['trade_date'].min()}~{panel['trade_date'].max()}")
    return panel


def main():
    if OUT.exists():
        panel = pd.read_parquet(OUT)
        print(f"已有面板 {len(panel)} 行, 至 {panel['trade_date'].max()}")
    elif LAB333 is not None and LAB333.exists():
        panel = load_lab333()
        panel.to_parquet(OUT, index=False)
    else:
        panel = build_history_tushare()
        panel.to_parquet(OUT, index=False)

    last_date = panel["trade_date"].max()
    tail = fetch_tail(last_date)
    if tail is not None and len(tail):
        merged = pd.concat([panel, tail], ignore_index=True)
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"],
                                        keep="last")
        merged.to_parquet(OUT, index=False)
        print(f"合并后面板 {len(merged)} 行, {merged['trade_date'].min()}~{merged['trade_date'].max()}")


if __name__ == "__main__":
    main()
