# -*- coding: utf-8 -*-
"""采集同花顺涨跌停榜单（tushare limit_list_ths, 含涨停原因）

只取「涨停池」分类: lu_desc(涨停原因)仅涨停池/连扳池有值, 复盘只需涨停池;
当日16点左右更新(非T+1), 历史起点20231101。北交所标的不在榜单内,
复盘展示层用开盘啦事件库(limitup.kpl_events.lu_desc)兜底。

产物:
  limitup.ths_limit  同花顺涨停池榜单(全字段, 按trade_date增量)

CLI:
  python collect/fetch_ths_limit.py            # 增量补缺(截至今日)
  python collect/fetch_ths_limit.py --start 20231101   # 全量回填
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

# 接口历史起点(tushare文档: 20231101起提供)
THS_START = "20231101"
LIMIT_TYPE = "涨停池"

FIELDS = ('ts_code,trade_date,name,price,pct_chg,open_num,lu_desc,limit_type,'
          'tag,status,first_lu_time,last_lu_time,limit_order,limit_amount,'
          'turnover_rate,free_float,lu_limit_order,limit_up_suc_rate,turnover,'
          'sum_float,market_type')


def fetch_day(date: str) -> pd.DataFrame:
    """拉单日涨停池(失败重试3次; 当日16点前/非交易日返回空)"""
    for attempt in range(3):
        try:
            df = pro.limit_list_ths(trade_date=date, limit_type=LIMIT_TYPE,
                                    fields=FIELDS)
            if "limit_type" not in df.columns:
                df = df.assign(limit_type=LIMIT_TYPE)
            return df
        except Exception as e:
            if attempt == 2:
                print(f"  {date}: [FAIL] {e}")
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="同花顺涨跌停榜单采集")
    ap.add_argument("--start", help="回填起点YYYYMMDD(默认接口起点/已有库末日)")
    ap.add_argument("--end", help="回填终点YYYYMMDD(默认今日)")
    args = ap.parse_args()

    p = path_of("limitup.ths_limit")
    if p.exists() and not args.start:
        ev = load("limitup.ths_limit")
        print(f"已有榜单 {len(ev)} 行, "
              f"{ev['trade_date'].min()}~{ev['trade_date'].max()}")
    else:
        ev = load("limitup.ths_limit") if p.exists() else pd.DataFrame()

    last = max(ev["trade_date"]) if len(ev) else None
    start = args.start or last or THS_START
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
            add.append(df)
        print(f"  {d}: +{len(df)} 行", flush=True)
        time.sleep(0.25)          # 8000积分档每分钟500次, 留余量

    if not add:
        print("本次无新增数据(当日榜单可能尚未更新)")
        return
    ev = pd.concat([ev] + add, ignore_index=True) if len(ev) else \
        pd.concat(add, ignore_index=True)
    ev = ev.drop_duplicates(subset=["trade_date", "ts_code", "limit_type"],
                            keep="last")
    ev = ev.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    out = save("limitup.ths_limit", ev)
    n = ev["lu_desc"].notna().sum()
    print(f"榜单 {len(ev)} 行 ({ev['trade_date'].min()}~"
          f"{ev['trade_date'].max()}) 涨停原因非空 {n} → {out}")


if __name__ == "__main__":
    main()
