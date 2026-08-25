# -*- coding: utf-8 -*-
"""研究01: 题材打板的条件期望统计（第一层）

核心问题: 题材维度能否在打板侧筛出正期望子集?
打板收益口径: T日涨停价(收盘)买 → T+1开盘卖, ret = next_open_ret (pre_close口径, 含除权)
可行性过滤: 剔除一字板(排板无法成交) + ST

分桶:
  A 基准(全部可打板事件)
  B 题材集中度 zt_cnt: 1 / 2-3 / 4-7 / 8+
  C 题材内地位: 龙头 vs 跟风
  D 连板高度 × 题材集中度 交叉
  E 题材年龄 theme_age: 首日爆发 vs 持续
  F 首封时间: 竞价925 / 早盘<1000 / 午前 / 午后
  附: 逐年拆分看衰减
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load


def stat(df: pd.DataFrame, ret_col: str = "next_open_ret") -> pd.Series:
    x = df[ret_col].dropna() * 100
    n = len(x)
    if n == 0:
        return pd.Series({"n": 0, "mean%": np.nan, "med%": np.nan, "t": np.nan})
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if x.std(ddof=1) > 0 else np.nan
    return pd.Series({"n": n, "mean%": round(x.mean(), 3),
                      "med%": round(x.median(), 3), "t": round(t, 2)})


def bucket_zt(c):
    if c == 1:
        return "1(孤板)"
    if c <= 3:
        return "2-3"
    if c <= 7:
        return "4-7"
    return "8+(大热点)"


def bucket_time(m):
    if m <= 0:
        return "竞价一字/秒板"
    if m < 35:
        return "早盘<10:00"
    if m < 150:
        return "午前"
    return "午后"


def main():
    ev = load("limitup.events_enriched")
    att = load("theme.attribution")
    td = load("theme.day")

    df = ev.merge(att, on=["trade_date", "ts_code"], how="left")
    df = df.merge(td[["trade_date", "concept_code", "zt_cnt", "theme_age"]],
                  on=["trade_date", "concept_code"], how="left")
    leader_lookup = dict(zip(zip(td["trade_date"], td["concept_code"]),
                             td["leader_code"]))
    df["is_leader"] = [leader_lookup.get((d, k)) == c
                       for d, k, c in zip(df["trade_date"], df["concept_code"],
                                          df["ts_code"])]
    # 可行性过滤
    base = df[(~df["is_yizi"]) & (~df["is_st"])].copy()
    yizi_share = df["is_yizi"].mean()
    print(f"事件总数 {len(df)}, 一字板 {yizi_share:.1%}(剔除), ST {df['is_st'].sum()}, "
          f"可打板样本 {len(base)}")
    print(f"日期范围 {df['trade_date'].min()}~{df['trade_date'].max()}")

    # 卖出端风险
    print(f"\nT+1开盘跌停(<=-9.5%)占比: {(base['next_open_ret'] <= -0.095).mean():.2%}")
    print(f"T+1一字(卖不出好价)占比: {base['next_is_yizi'].fillna(False).mean():.2%}")

    print("\n=== A. 基准: 涨停价买→T+1开盘卖 ===")
    print(stat(base).to_frame("all").T.to_string())
    print(stat(base, "next_close_ret").to_frame("all(持有到T+1收盘)").T.to_string())

    print("\n=== B. 题材集中度(独占归属后涨停家数) ===")
    base["zt_b"] = base["zt_cnt"].fillna(0).astype(int).map(bucket_zt)
    print(base.groupby("zt_b").apply(stat, include_groups=False).to_string())

    print("\n=== C. 题材内地位 ===")
    base["role"] = np.where(base["zt_cnt"].fillna(0) <= 1, "孤板",
                            np.where(base["is_leader"], "龙头", "跟风"))
    print(base.groupby("role").apply(stat, include_groups=False).to_string())

    print("\n=== D. 连板高度 × 题材集中度 ===")
    base["hb"] = np.where(base["limit_times"] == 1, "首板", "2板+")
    print(base.groupby(["hb", "zt_b"]).apply(stat, include_groups=False).to_string())

    print("\n=== E. 题材年龄(连续有涨停天数) ===")
    base["age_b"] = np.where(base["theme_age"].fillna(1) == 1, "首日爆发", "持续2天+")
    print(base.groupby("age_b").apply(stat, include_groups=False).to_string())

    print("\n=== F. 首封时间 ===")
    base["time_b"] = base["first_min"].map(bucket_time)
    print(base.groupby("time_b").apply(stat, include_groups=False).to_string())

    print("\n=== 附: 逐年基准 ===")
    base["year"] = base["trade_date"].str[:4]
    print(base.groupby("year").apply(stat, include_groups=False).to_string())


if __name__ == "__main__":
    main()
