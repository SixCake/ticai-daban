# -*- coding: utf-8 -*-
"""研究04: 龙头/中军/补涨 三角色统计验证

角色操作化定义(题材-日, 独占归属):
  龙头   = theme_day.leader_code (连板高度→封单额→首封早→炸板少)
  中军A  = 题材内非龙头成交额最大者 (仅zt_cnt>=4的题材计算)
  中军mv = 题材内非龙头流通市值最大者 (替代口径)
  补涨   = 题材波龄>=2 + 该股波内首次涨停 + 连板<=2
  共振首板 = 波龄==1(题材爆发日) 的首板
  跟风   = 其余

统计层:
  1. 角色T+1开盘收益 (剔一字/ST) + 日聚类t
  2. 龙头溢价: 按题材规模桶 龙头 vs 非龙头
  3. 补涨时机: 按波龄桶(2/3/4-5/6+) T+1 —— "补涨越晚越差=尾声信号"?
  4. 中军→题材延续: 大市值非龙头参与(非龙头流通市值max>=当日全池P80)
     vs 无, 次日题材存活率与次日家数 (按规模桶控制)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load


def stat(df, ret_col="next_open_ret"):
    x = df[ret_col].dropna() * 100
    n = len(x)
    if n == 0:
        return pd.Series({"n": 0, "mean%": np.nan, "med%": np.nan, "t": np.nan})
    dm = df.groupby("trade_date")[ret_col].mean().dropna() * 100
    t = (dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm)))) if len(dm) >= 5 else np.nan
    return pd.Series({"n": n, "mean%": round(x.mean(), 2),
                      "med%": round(x.median(), 2), "t": round(t, 1)})


def bucket_zt(c):
    if c <= 3:
        return "2-3"
    if c <= 7:
        return "4-7"
    return "8+"


def main():
    ev = load("limitup.events_enriched")
    att = load("theme.attribution")
    td = load("theme.day")

    att = att[att["concept_code"] != "UNASSIGNED"]
    df = ev.merge(att[["trade_date", "ts_code", "concept_code"]],
                  on=["trade_date", "ts_code"], how="inner")
    df = df.merge(td[["trade_date", "concept_code", "zt_cnt", "theme_age",
                      "leader_code"]], on=["trade_date", "concept_code"])

    # ---- 题材波id: theme_age==1 为波起点 ----
    td_s = td.sort_values(["concept_code", "trade_date"])
    td_s["wave_id"] = (td_s.groupby("concept_code")["theme_age"]
                       .transform(lambda s: (s == 1).cumsum()))
    wave_key = td_s.set_index(["trade_date", "concept_code"])["wave_id"]
    df["wave_id"] = [wave_key.get((d, k)) for d, k in
                     zip(df["trade_date"], df["concept_code"])]
    first_day = (df.groupby(["concept_code", "wave_id", "ts_code"])["trade_date"]
                 .min().rename("first_day").reset_index())
    df = df.merge(first_day, on=["concept_code", "wave_id", "ts_code"])
    df["is_first"] = df["trade_date"] == df["first_day"]

    # ---- 角色 ----
    df["is_leader"] = df["ts_code"] == df["leader_code"]
    df["is_bz"] = df["is_first"] & (df["theme_age"] >= 2) & (df["limit_times"] <= 2)
    df["is_res"] = df["is_first"] & (df["theme_age"] == 1) & (df["limit_times"] == 1)
    big = df[(df["zt_cnt"] >= 4) & ~df["is_leader"]]
    zj_a = (big.sort_values("amount")
            .drop_duplicates(["trade_date", "concept_code"], keep="last")
            [["trade_date", "concept_code", "ts_code"]])
    zj_m = (big.sort_values("float_mv")
            .drop_duplicates(["trade_date", "concept_code"], keep="last")
            [["trade_date", "concept_code", "ts_code"]])
    df["is_zj_a"] = df.set_index(
        ["trade_date", "concept_code", "ts_code"]).index.isin(
        zj_a.set_index(["trade_date", "concept_code", "ts_code"]).index)
    df["is_zj_mv"] = df.set_index(
        ["trade_date", "concept_code", "ts_code"]).index.isin(
        zj_m.set_index(["trade_date", "concept_code", "ts_code"]).index)

    base = df[(~df["is_yizi"]) & (~df["is_st"]) &
              df["next_open_ret"].notna()].copy()
    base["zt_b"] = base["zt_cnt"].map(bucket_zt)
    print(f"样本 {len(base)} (剔一字/ST), 角色重叠: "
          f"龙头∩中军A={int((df.is_leader & df.is_zj_a).sum())} "
          f"补涨∩中军A={int((df.is_bz & df.is_zj_a).sum())}")

    print("\n=== 1. 角色T+1开盘收益 (剔一字/ST) ===")
    roles = [("龙头", base.is_leader), ("中军A(额最大)", base.is_zj_a),
             ("中军mv(市值最大)", base.is_zj_mv), ("补涨", base.is_bz),
             ("共振首板", base.is_res)]
    out = []
    for nm, m in roles:
        out.append(stat(base[m]))
    out.append(stat(base[~base.is_leader & ~base.is_zj_a & ~base.is_bz
                         & ~base.is_res]))
    res1 = pd.DataFrame(out, index=["龙头", "中军A(额最大)", "中军mv(市值最大)",
                                    "补涨", "共振首板", "跟风(其余)"])
    print(res1.to_string())

    print("\n=== 2. 龙头溢价 按题材规模 ===")
    rows = []
    for b in ["2-3", "4-7", "8+"]:
        sub = base[base["zt_b"] == b]
        rows.append(pd.Series({"龙头": stat(sub[sub.is_leader])["mean%"],
                               "非龙头": stat(sub[~sub.is_leader])["mean%"],
                               "龙头n": int(sub.is_leader.sum())}))
    print(pd.DataFrame(rows, index=["2-3", "4-7", "8+"]).to_string())

    print("\n=== 3. 补涨时机: 波龄桶 × T+1 ===")
    bz = base[base.is_bz].copy()
    bz["age_b"] = pd.cut(bz.theme_age, [1, 2, 3, 5, 99],
                         labels=["2", "3", "4-5", "6+"])
    print(bz.groupby("age_b", observed=True)
          .apply(stat, include_groups=False).to_string())
    res = base[base.is_res]
    print(f"对照 共振首板(波龄1): n={len(res)} mean={res.next_open_ret.mean()*100:.2f}%")

    print("\n=== 4. 中军(大市值非龙头参与) → 次日题材延续 ===")
    day_p80 = df.groupby("trade_date")["float_mv"].quantile(0.8)
    df["day_p80"] = df["trade_date"].map(day_p80)
    flag = (df[(df["zt_cnt"] >= 4) & ~df["is_leader"]]
            .groupby(["trade_date", "concept_code"])
            .apply(lambda g: (g["float_mv"].max() >= g["day_p80"].iloc[0]),
                   include_groups=False).rename("has_zj"))
    dates = sorted(td["trade_date"].unique())
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    td2 = td.merge(flag, on=["trade_date", "concept_code"], how="left")
    td2 = td2[td2["zt_cnt"] >= 4].copy()
    td2["next_d"] = td2["trade_date"].map(nxt)
    active = set(zip(td["trade_date"], td["concept_code"]))
    td2["survive"] = [(d, k) in active for d, k in
                      zip(td2["next_d"], td2["concept_code"])]
    nz = td2.merge(td.rename(columns={"trade_date": "next_d"})[
        ["next_d", "concept_code", "zt_cnt"]].rename(
        columns={"zt_cnt": "next_zt"}), on=["next_d", "concept_code"], how="left")
    rows = []
    for b in ["4-7", "8+"]:
        sub = nz[nz["zt_cnt"].map(bucket_zt) == b]
        for has in [True, False]:
            g = sub[sub["has_zj"] == has]
            rows.append({"规模": b, "中军参与": "有" if has else "无",
                         "n": len(g),
                         "次日存活率": round(g["survive"].mean(), 3),
                         "次日家数": round(g["next_zt"].mean(), 2)})
    print(pd.DataFrame(rows).to_string())

    print("\n=== 5. 龙头/补涨 逐年 ===")
    for nm, m in [("龙头", base.is_leader), ("补涨", base.is_bz)]:
        sub = base[m]
        yr = sub.groupby(sub.trade_date.str[:4]).apply(
            lambda g: round(g.next_open_ret.mean() * 100, 2))
        print(f"  {nm}: " + " ".join(f"{y}:{v}" for y, v in yr.items()))


if __name__ == "__main__":
    main()
