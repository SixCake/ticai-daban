# -*- coding: utf-8 -*-
"""采集涨停事件（tushare limit_list_d, 2019-11-28→今, 增量续跑）

产物: limitup.events
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import START_DATE, get_pro
from datastore import path_of

pro = get_pro()
OUT = path_of("limitup.events")
OUT.parent.mkdir(parents=True, exist_ok=True)


def trade_days(start: str, end: str) -> list[str]:
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end,
                        is_open="1")
    return sorted(cal["cal_date"].tolist())


def fetch_day(d: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            return pro.limit_list_d(trade_date=d, limit_type="U")
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [FAIL] {d}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def main():
    start = START_DATE
    if "--start" in sys.argv:
        start = sys.argv[sys.argv.index("--start") + 1]
    today = pd.Timestamp.now().strftime("%Y%m%d")
    days = trade_days(start, today)

    existing = None
    done_days = set()
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        done_days = set(existing["trade_date"].unique())
        print(f"已有数据 {len(existing)} 行, 最后日期 {max(done_days)}")

    todo = [d for d in days if d not in done_days]
    print(f"待采集 {len(todo)} 个交易日 ({days[0]}~{days[-1]})")

    buf = []
    for i, d in enumerate(todo):
        df = fetch_day(d)
        if df is not None and len(df):
            buf.append(df)
        time.sleep(0.15)
        if (i + 1) % 100 == 0:
            print(f"  进度 {i + 1}/{len(todo)}  累计新增 {sum(len(x) for x in buf)} 行")

    if buf:
        new = pd.concat(buf, ignore_index=True)
        new["name"] = new["name"].astype(str)
        new["is_st"] = new["name"].str.contains("ST", case=False, na=False)
        new["first_time"] = new["first_time"].astype(str).str.zfill(6)
        new["last_time"] = new["last_time"].astype(str).str.zfill(6)
        merged = (pd.concat([existing, new], ignore_index=True)
                  if existing is not None else new)
        merged = merged.drop_duplicates(subset=["trade_date", "ts_code"])
        merged = merged.sort_values(["trade_date", "ts_code"])
        merged.to_parquet(OUT, index=False)
        print(f"保存 {len(merged)} 行 → {OUT}")
    else:
        print("无新增数据")


if __name__ == "__main__":
    main()
