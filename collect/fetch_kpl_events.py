# -*- coding: utf-8 -*-
"""采集开盘啦涨停/炸板事件（tushare kpl_list, 含theme题材标注）

首次运行自动从 qmt-trade 项目导入已拉近两年的历史缓存
(xuntou/research/kb92/kpl_limit_up.parquet + kpl_break.parquet),
之后按交易日历增量补缺(kpl_list为T+1数据, 当日盘后次日才可拉)。

产物:
  limitup.kpl_events  开盘啦事件库(涨停+炸板, 全字段, tag区分)

theme字段是开盘啦对当日炒作题材的事件级标注(顿号分隔多题材,
第一个为主归属), 是题材归属层(core.attribute)的权威数据源。
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_pro  # noqa: E402
from datastore import load, path_of, save  # noqa: E402

pro = get_pro()

# qmt-trade 历史缓存(20240102~20260821, 已拉近两年)
QMT_TRADE_DIR = Path.home() / "aiproject/qmt-trade/xuntou/research/kb92"

FIELDS = ('ts_code,name,trade_date,lu_time,open_time,last_time,lu_desc,tag,'
          'theme,status,net_change,bid_amount,bid_change,bid_turnover,'
          'lu_bid_vol,pct_chg,bid_pct_chg,limit_order,amount,turnover_rate,'
          'free_float,lu_limit_order')


def import_history() -> pd.DataFrame | None:
    """从 qmt-trade 历史缓存导入(一次性, 无tag列则补)"""
    f_u = QMT_TRADE_DIR / "kpl_limit_up.parquet"
    f_z = QMT_TRADE_DIR / "kpl_break.parquet"
    if not f_u.exists():
        return None
    parts = []
    df_u = pd.read_parquet(f_u)
    if len(df_u):
        df_u = df_u.assign(tag="涨停")
        parts.append(df_u)
    if f_z.exists():
        df_z = pd.read_parquet(f_z)
        if len(df_z):
            df_z = df_z.assign(tag="炸板")
            parts.append(df_z)
    return pd.concat(parts, ignore_index=True) if parts else None


def fetch_day(date: str) -> pd.DataFrame:
    """拉单日涨停+炸板(kpl_list T+1, 失败重试3次)"""
    parts = []
    for tag in ("涨停", "炸板"):
        df = None
        for attempt in range(3):
            try:
                df = pro.kpl_list(trade_date=date, tag=tag, fields=FIELDS)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if df is None:
            df = pd.DataFrame()
        if len(df):
            if "tag" not in df.columns:
                df = df.assign(tag=tag)
            parts.append(df)
        time.sleep(0.35)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    p = path_of("limitup.kpl_events")
    if p.exists():
        ev = load("limitup.kpl_events")
        print(f"已有事件库 {len(ev)} 行, "
              f"{ev['trade_date'].min()}~{ev['trade_date'].max()}")
    else:
        hist = import_history()
        if hist is None or not len(hist):
            print("[FAIL] 无本地事件库且 qmt-trade 历史缓存不存在: "
                  f"{QMT_TRADE_DIR}")
            return
        ev = hist
        print(f"从 qmt-trade 导入历史 {len(ev)} 行, "
              f"{ev['trade_date'].min()}~{ev['trade_date'].max()}")

    # ---- 增量: 交易日历补缺(截至昨日, kpl_list当日T+1才可拉) ----
    today = datetime.now().strftime("%Y%m%d")
    last = max(ev["trade_date"])
    cal = pro.trade_cal(exchange="SSE", start_date=last, end_date=today,
                        is_open="1")
    days = [d for d in sorted(cal["cal_date"].tolist())
            if d > last and d < today]
    if days:
        print(f"增量补拉 {len(days)} 个交易日: {days[0]}~{days[-1]}")
        add = []
        for i, d in enumerate(days):
            df = fetch_day(d)
            if len(df):
                add.append(df)
            print(f"  {d}: +{len(df)} 行", flush=True)
        if add:
            ev = pd.concat([ev] + add, ignore_index=True)

    if len(ev):
        ev = ev.drop_duplicates(subset=["trade_date", "ts_code", "tag"])
        ev = ev.sort_values(["trade_date", "tag", "ts_code"]).reset_index(
            drop=True)
        out = save("limitup.kpl_events", ev)
        n_theme = ev["theme"].notna().sum()
        print(f"事件库 {len(ev)} 行 "
              f"({ev['trade_date'].min()}~{ev['trade_date'].max()}) "
              f"theme非空 {n_theme} → {out}")


if __name__ == "__main__":
    main()
