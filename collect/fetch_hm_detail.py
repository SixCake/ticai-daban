# -*- coding: utf-8 -*-
"""采集龙虎榜游资席位明细（tushare hm_detail）

每日收盘后晚间更新(约19~22点); 拉取失败/无数据不阻塞 daily_update 主流程。
产物:
  hmlist.detail  席位×个股买卖明细(按trade_date增量)

CLI:
  python collect/fetch_hm_detail.py                  # 增量补缺(截至今日)
  python collect/fetch_hm_detail.py --start 20260601 # 区间回填
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_pro  # noqa: E402
from datastore import load, path_of, save  # noqa: E402

pro = get_pro()

NUM_COLS = ["buy_amount", "sell_amount", "net_amount"]


def fetch_day(date: str) -> pd.DataFrame:
    """拉单日游资明细(失败重试3次; 数据未更新/非交易日返回空)"""
    for attempt in range(3):
        try:
            df = pro.hm_detail(trade_date=date)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            if attempt == 2:
                print(f"  {date}: [FAIL] {e}")
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="龙虎榜游资席位明细采集")
    ap.add_argument("--start", help="回填起点YYYYMMDD(默认已有库末日)")
    ap.add_argument("--end", help="回填终点YYYYMMDD(默认今日)")
    args = ap.parse_args()

    p = path_of("hmlist.detail")
    ev = load("hmlist.detail") if p.exists() else pd.DataFrame()
    if len(ev):
        print(f"已有明细 {len(ev)} 行, "
              f"{ev['trade_date'].min()}~{ev['trade_date'].max()}")

    last = max(ev["trade_date"]) if len(ev) else None
    start = args.start or last or datetime.now().strftime("%Y%m%d")
    end = args.end or datetime.now().strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end,
                        is_open="1")
    # 给了--start视为重拉该区间(重复行以新数据为准), 否则只增量补末日之后
    days = [d for d in sorted(cal["cal_date"].tolist())
            if d <= end and (args.start or last is None or d > last)]
    if not days:
        print(f"无需补拉({start}~{end})")
        return
    print(f"补拉 {len(days)} 个交易日: {days[0]}~{days[-1]}")

    add = []
    for d in days:
        df = fetch_day(d)
        if len(df):
            df["trade_date"] = d      # 接口偶缺该列, 统一以日历日为准
            add.append(df)
        print(f"  {d}: +{len(df)} 行", flush=True)
        time.sleep(0.35)

    if not add:
        print("本次无新增数据(当日明细可能尚未更新)")
        return
    ev = pd.concat([ev] + add, ignore_index=True) if len(ev) else \
        pd.concat(add, ignore_index=True)
    for c in NUM_COLS:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0)
    ev = ev.drop_duplicates(subset=["trade_date", "ts_code", "hm_name",
                                    "hm_orgs"], keep="last")
    ev = ev.sort_values(["trade_date", "hm_name", "ts_code"]).reset_index(
        drop=True)
    out = save("hmlist.detail", ev)
    print(f"明细 {len(ev)} 行 ({ev['trade_date'].min()}~"
          f"{ev['trade_date'].max()}) → {out}")


if __name__ == "__main__":
    main()
