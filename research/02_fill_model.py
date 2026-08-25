# -*- coding: utf-8 -*-
"""研究02: 封单额成交建模（日线代理层）

核心问题: 能排进的板还赚不赚? (逆向选择量化)

板型分类:
  一封到底: open_times==0 —— 排板成交取决于封单厚度(fd_ratio越小=抛压消化越多=越易成交)
  炸板回封: open_times>=1 —— 开板时挂单几乎必成交, 实盘主要成交来源, 逆向选择样本
  尾盘回封: open_times>=1 且 last_time>=14:00 —— 最弱回封, 单独看

权衡曲线: 一封到底内部按 fd_ratio=fd_amount/amount 分位分层看T+1收益单调性
现实估计: 各题材集中度桶 × 板型 的收益矩阵 + 最强组合的板型分解
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA


def stat(df, ret_col="next_open_ret"):
    x = df[ret_col].dropna() * 100
    n = len(x)
    if n == 0:
        return pd.Series({"n": 0, "mean%": np.nan, "med%": np.nan})
    return pd.Series({"n": n, "mean%": round(x.mean(), 2),
                      "med%": round(x.median(), 2)})


def bucket_zt(c):
    if c == 1:
        return "1(孤板)"
    if c <= 3:
        return "2-3"
    if c <= 7:
        return "4-7"
    return "8+(大热点)"


def day_cluster_t(sub, ret_col="next_open_ret"):
    dm = sub.groupby("trade_date")[ret_col].mean() * 100
    if len(dm) < 5:
        return np.nan
    return round(dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))), 1)


def main():
    ev = pd.read_parquet(DATA / "events_enriched.parquet")
    att = pd.read_parquet(DATA / "attribution.parquet")
    td = pd.read_parquet(DATA / "theme_day.parquet")
    df = ev.merge(att, on=["trade_date", "ts_code"], how="left")
    df = df.merge(td[["trade_date", "concept_code", "zt_cnt"]],
                  on=["trade_date", "concept_code"], how="left")
    base = df[(~df["is_yizi"]) & (~df["is_st"]) &
              df["next_open_ret"].notna() & df["fd_amount"].notna()].copy()
    base["zt_b"] = base["zt_cnt"].fillna(0).astype(int).map(bucket_zt)
    base["fd_ratio"] = base["fd_amount"] / base["amount"].replace(0, np.nan)
    base["fd_mv"] = base["fd_amount"] / base["float_mv"].replace(0, np.nan)

    lastm = base["last_time"].astype(str).str.zfill(6)
    base["board_type"] = np.where(base["open_times"] == 0, "一封到底",
                                  np.where(lastm >= "140000", "尾盘回封", "炸板回封"))

    print(f"样本 {len(base)} (剔一字/ST/缺封单), fd_amount缺失率 "
          f"{1 - df['fd_amount'].notna().mean():.2%}")
    print(f"板型分布:\n{base['board_type'].value_counts(normalize=True).round(3).to_string()}")

    print("\n=== 1. 板型 × 收益 (总体) ===")
    print(base.groupby("board_type").apply(stat, include_groups=False).to_string())

    print("\n=== 2. 板型 × 题材集中度 ===")
    pv = base.pivot_table(index="zt_b", columns="board_type", values="next_open_ret",
                          aggfunc=lambda x: round(x.mean() * 100, 2))
    pn = base.pivot_table(index="zt_b", columns="board_type", values="next_open_ret",
                          aggfunc="count").astype(int)
    print("均值%:")
    print(pv.to_string())
    print("样本数:")
    print(pn.to_string())

    print("\n=== 3. 一封到底: fd_ratio分位权衡曲线(封单薄=易成交?) ===")
    yf = base[base["board_type"] == "一封到底"].copy()
    yf["fdq"] = pd.qcut(yf["fd_ratio"], 5, labels=["Q1封单最薄", "Q2", "Q3", "Q4", "Q5封单最厚"])
    g = yf.groupby("fdq", observed=True).apply(stat, include_groups=False)
    g["日聚类t"] = [day_cluster_t(yf[yf["fdq"] == q]) for q in g.index]
    print(g.to_string())

    print("\n=== 4. 一封到底: fd_mv(封单/流通市值)分位 ===")
    yf["fmq"] = pd.qcut(yf["fd_mv"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                        duplicates="drop")
    print(yf.groupby("fmq", observed=True).apply(stat, include_groups=False).to_string())

    print("\n=== 5. 炸板回封(可成交样本) × 题材集中度: 现实期望 ===")
    zb = base[base["board_type"].isin(["炸板回封", "尾盘回封"])]
    print(zb.groupby("zt_b").apply(stat, include_groups=False).to_string())
    print("日聚类t:")
    for b in ["1(孤板)", "2-3", "4-7", "8+(大热点)"]:
        print(f"  {b}: t={day_cluster_t(zb[zb['zt_b'] == b])}")

    print("\n=== 6. 最强组合分解: 大热点×早盘×龙头 按板型 ===")
    leader_lookup = dict(zip(zip(td["trade_date"], td["concept_code"]),
                             td["leader_code"]))
    base["is_leader"] = [leader_lookup.get((d, k)) == c
                         for d, k, c in zip(base["trade_date"], base["concept_code"],
                                            base["ts_code"])]
    combo = base[(base["zt_b"] == "8+(大热点)") &
                 base["first_min"].between(0, 35) & base["is_leader"]]
    print(combo.groupby("board_type").apply(stat, include_groups=False).to_string())

    print("\n=== 7. 逆向选择缺口 逐年 ===")
    gap = base.pivot_table(index=base["trade_date"].str[:4], columns="board_type",
                           values="next_open_ret", aggfunc=lambda x: x.mean() * 100)
    gap["缺口(一封-回封)"] = (gap.get("一封到底", np.nan) -
                              gap[["炸板回封", "尾盘回封"]].mean(axis=1)).round(2)
    print(gap.round(2).to_string())


if __name__ == "__main__":
    main()
