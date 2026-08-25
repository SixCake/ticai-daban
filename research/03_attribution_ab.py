# -*- coding: utf-8 -*-
"""研究03: 独占归属tie-break A/B — 成分数(v1) vs 当日涨停密度(v2)

背景(两个实盘case暴露):
  深中华A(20260824)→两轮车: 两轮车raw1/成分77, 黄金概念raw3/成分82, v1平票取小成分→两轮车
  金健米业(20260819)→乳业: 乳业raw1/成分35, 粮食概念raw4/成分47, 国企改革raw10/成分1470
v1问题: 平票取成分数小 → 偏爱niche冷概念, 真实热点被拆碎
naive raw修复问题: 国企改革类杂烩raw高但非主题 → 用密度 raw/member_count 做tie-break

v2: sorted(ks, key=(-cnt, -raw/msize, msize, code))
cnt(已归属家数)仍为主键驱动级联; 密度只负责冷启动平票。

方法: 归属在当日全量涨停池上跑(与生产一致, 一字/ST也贡献家数), 统计再剔样本。
v1基线须复现生产数字(碎片率30.6%/漏标9.82%/现实格n=984均值4.81)作sanity check。

对比指标:
  A 碎片率: 归属家数==1 且 自身另有raw>=3概念; 大漏标: raw>=8却归属孤板 (全事件)
  B 现实格(无前视): n/均值/日聚类t/逐年 (剔一字/ST/缺T+1)
  C 集中度单调性: 1/2-3/4-7/8+ 桶均值
  D 定性: 20260819金健米业、20260824深中华A、20260820 TOP题材
  E 杂烩接管: 国企改革/养老概念/乡村振兴 进当日TOP5的天数
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load

JUNK = {"国企改革", "养老概念", "乡村振兴", "融资融券", "央企国企改革"}


def load_data():
    concepts = load("theme.concepts")
    members = load("theme.members")
    theme = concepts[concepts["is_theme"]]
    msize = theme.set_index("ts_code")["member_count"].to_dict()
    cname = theme.set_index("ts_code")["name"].to_dict()
    tcodes = set(theme["ts_code"])
    mem = members[members["concept_code"].isin(tcodes)]
    stock2con = mem.groupby("con_code")["concept_code"].apply(
        lambda s: sorted(set(s))).to_dict()
    ev = load("limitup.events_enriched")
    return ev, stock2con, msize, cname


def attribute_day(codes, stock2con, msize, mode):
    raw = defaultdict(int)
    for c in codes:
        for k in stock2con.get(c, []):
            raw[k] += 1
    active = set(raw)
    cand = {c: [k for k in stock2con.get(c, []) if k in active] for c in codes}
    dens = {k: raw[k] / msize.get(k, 10**9) for k in active}

    attr = {}
    for _ in range(20):
        cnt = defaultdict(int)
        for k in attr.values():
            cnt[k] += 1
        if mode == "v1":
            key = lambda k: (-cnt.get(k, 0), msize.get(k, 10**9), k)
        else:
            key = lambda k: (-cnt.get(k, 0), -dens.get(k, 0),
                             msize.get(k, 10**9), k)
        new_attr = {c: (sorted(ks, key=key)[0] if ks else "UNASSIGNED")
                    for c, ks in cand.items()}
        if new_attr == attr:
            break
        attr = new_attr
    return attr, raw


def day_cluster_t(dm):
    if len(dm) < 5:
        return np.nan
    return round(dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))), 1)


def main():
    ev, stock2con, msize, cname = load_data()
    lastm_all = ev["last_time"].astype(str).str.zfill(6)

    rows = {m: [] for m in ("v1", "v2")}
    for d, grp in ev.groupby("trade_date"):
        codes = grp["ts_code"].tolist()
        for mode in ("v1", "v2"):
            attr, raw = attribute_day(codes, stock2con, msize, mode)
            cnt = defaultdict(int)
            for k in attr.values():
                cnt[k] += 1
            for c in codes:
                ks = stock2con.get(c, [])
                rows[mode].append((d, c, attr.get(c, "UNASSIGNED"),
                                   cnt[attr.get(c, "UNASSIGNED")],
                                   max([raw[k] for k in ks], default=0)))
    cols = ["trade_date", "ts_code", "concept", "zt_cnt", "best_raw"]
    out = {}
    for m in ("v1", "v2"):
        d = pd.DataFrame(rows[m], columns=cols)
        d = d.merge(ev, on=["trade_date", "ts_code"])
        d["name_cn"] = d["concept"].map(cname).fillna("未分组")
        out[m] = d

    print("=== A. 碎片化(全事件) ===")
    for m in ("v1", "v2"):
        d = out[m]
        frag = ((d["zt_cnt"] == 1) & (d["best_raw"] >= 3)).mean()
        big = ((d["zt_cnt"] == 1) & (d["best_raw"] >= 8)).mean()
        print(f"  {m}: 碎片率 {frag:.1%}  大热点漏标 {big:.2%}")

    stat_sub = {m: out[m][(~out[m]["is_yizi"]) & (~out[m]["is_st"]) &
                          out[m]["next_open_ret"].notna()] for m in ("v1", "v2")}
    lastm = {m: stat_sub[m]["last_time"].astype(str).str.zfill(6)
             for m in ("v1", "v2")}

    print("=== B. 现实格(zt>=8+炸板1-3+午前回封) ===")
    for m in ("v1", "v2"):
        d = stat_sub[m]
        mask = ((d["zt_cnt"] >= 8) & (d["open_times"] >= 1) &
                (d["open_times"] <= 3) & (lastm[m] <= "110000"))
        sub = d[mask]
        dm = sub.groupby("trade_date")["next_open_ret"].mean() * 100
        print(f"  {m}: n={len(sub)} 均值{(sub['next_open_ret'].mean()*100):.2f}% "
              f"日聚类t={day_cluster_t(dm)}")
        yr = sub.groupby(sub["trade_date"].str[:4])["next_open_ret"].mean() * 100
        print("     逐年:", " ".join(f"{y}:{v:.2f}" for y, v in yr.items()))

    print("=== C. 集中度单调性 ===")
    for m in ("v1", "v2"):
        d = stat_sub[m].copy()
        d["b"] = pd.cut(d["zt_cnt"], [0, 1, 3, 7, 999],
                        labels=["1", "2-3", "4-7", "8+"])
        g = d.groupby("b", observed=True)["next_open_ret"].agg(["count", "mean"])
        print(f"  {m}: " + "  ".join(
            f"{b}:{round(r['mean']*100, 2)}%(n={int(r['count'])})"
            for b, r in g.iterrows()))

    print("=== D. 定性case ===")
    for m in ("v1", "v2"):
        d = out[m]
        a = d[(d["trade_date"] == "20260819") & (d["ts_code"] == "600127.SH")]
        b = d[(d["trade_date"] == "20260824") & (d["ts_code"] == "000017.SZ")]
        an = a["name_cn"].iloc[0] if len(a) else "?"
        bn = b["name_cn"].iloc[0] if len(b) else "?"
        print(f"  {m}: 金健米业→{an}  深中华A→{bn}")
        t20 = (d[d["trade_date"] == "20260820"].groupby("name_cn")
               .size().sort_values(ascending=False).head(5))
        print(f"     20260820 TOP5: {dict(t20)}")

    print("=== E. 杂烩进TOP5天数 ===")
    for m in ("v1", "v2"):
        days = 0
        for _, g in out[m].groupby("trade_date"):
            top5 = set(g.groupby("name_cn").size()
                       .sort_values(ascending=False).head(5).index)
            if top5 & JUNK:
                days += 1
        print(f"  {m}: {days} 天")


if __name__ == "__main__":
    main()
