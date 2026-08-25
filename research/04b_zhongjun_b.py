# -*- coding: utf-8 -*-
"""研究04b: 中军B(成分级锚)验证

中军B = 题材成分内 当日涨幅>=5% 且非涨停 中成交额(vol*close)最大者
(研究04已否定池内中军A作为买点; 游资语义的中军=不涨停但放量大涨的大市值锚)

统计层:
  1. 出现率与画像 (涨幅/成交额/市值) 按题材规模桶, 阈值敏感性(5%/7%)
  2. 中军B → 题材延续: 次日存活率/次日家数 (按规模桶控制)
  3. 中军B自身 T+1 (T收盘买→T+1开盘卖) 参考
  4. 逐年延续性差距稳定性
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load


def bucket_zt(c):
    if c <= 3:
        return "2-3"
    if c <= 7:
        return "4-7"
    return "8+"


def main():
    dp = load("market.daily_panel")
    ev = load("limitup.events_enriched", columns=["trade_date", "ts_code"])
    td = load("theme.day")
    mem = load("theme.members")
    con = load("theme.concepts")
    theme_codes = set(con[con["is_theme"]]["ts_code"])
    mem_t = mem[mem["concept_code"].isin(theme_codes)]

    dp["amount"] = dp["vol"] * dp["close"]
    # 自身T+1: 收盘买→次日开盘卖
    dp = dp.sort_values(["ts_code", "trade_date"])
    dp["t1_open"] = dp.groupby("ts_code")["open"].shift(-1)
    dp["self_ret"] = dp["t1_open"] / dp["close"] - 1

    evset = set(zip(ev["trade_date"], ev["ts_code"]))

    dates = sorted(td["trade_date"].unique())
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    active = set(zip(td["trade_date"], td["concept_code"]))

    for thr in (5, 7):
        cand = dp[dp["pct_chg"] >= thr]
        cand = cand.merge(mem_t, left_on="ts_code", right_on="con_code",
                          how="inner")
        cand = cand[~pd.Series(zip(cand["trade_date"], cand["ts_code"]),
                               index=cand.index).isin(evset)]
        zj = (cand.sort_values("amount")
              .drop_duplicates(["trade_date", "concept_code"], keep="last"))
        zj = zj[zj["concept_code"].isin(theme_codes)]

        tdm = td.merge(zj[["trade_date", "concept_code", "ts_code", "pct_chg",
                           "amount", "self_ret", "close"]]
                       .rename(columns={"ts_code": "zj_code",
                                        "pct_chg": "zj_pct",
                                        "amount": "zj_amt",
                                        "self_ret": "zj_ret"}),
                       on=["trade_date", "concept_code"], how="left")
        tdm["has_zj"] = tdm["zj_code"].notna()
        tdm["next_d"] = tdm["trade_date"].map(nxt)
        tdm["survive"] = [(d, k) in active for d, k in
                          zip(tdm["next_d"], tdm["concept_code"])]
        nz = tdm.merge(td.rename(columns={"trade_date": "next_d"})[
            ["next_d", "concept_code", "zt_cnt"]].rename(
            columns={"zt_cnt": "next_zt"}),
            on=["next_d", "concept_code"], how="left")
        nz["zt_b"] = nz["zt_cnt"].map(bucket_zt)

        print(f"\n===== 阈值 >= {thr}% =====")
        sub = nz[nz["zt_cnt"] >= 4]
        print(f"题材-日(zt>=4) {len(sub)}, 中军B出现率 "
              f"{sub['has_zj'].mean():.1%}")
        prof = sub[sub.has_zj]
        print(f"画像: 中军涨幅中位 {prof.zj_pct.median():.1f}%, "
              f"成交额中位 {prof.zj_amt.median()/1e8:.1f}亿")
        rows = []
        for b in ["4-7", "8+"]:
            s2 = sub[sub["zt_b"] == b]
            for has in [True, False]:
                g = s2[s2["has_zj"] == has]
                rows.append({"规模": b, "中军B": "有" if has else "无",
                             "n": len(g),
                             "次日存活率": round(g["survive"].mean(), 3),
                             "次日家数": round(g["next_zt"].mean(), 2)})
        print(pd.DataFrame(rows).to_string())

        # 逐年: 8+桶 存活率差
        yr = []
        s8 = sub[sub["zt_b"] == "8+"].copy()
        s8["year"] = s8["trade_date"].str[:4]
        for y, g in s8.groupby("year"):
            a = g[g.has_zj]["survive"].mean()
            b0 = g[~g.has_zj]["survive"].mean()
            yr.append(f"{y}:{a - b0:+.2f}")
        print("8+桶逐年存活率差(有-无): " + " ".join(yr))

        # 自身T+1
        r = prof["zj_ret"].dropna() * 100
        print(f"中军B自身T+1(收盘买→次开卖): n={len(r)} "
              f"mean={r.mean():.2f}% med={r.median():.2f}%")


if __name__ == "__main__":
    main()
