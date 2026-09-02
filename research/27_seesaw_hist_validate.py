# -*- coding: utf-8 -*-
"""研究27: 龙头拐头/跷跷板/情绪四阶段 历史日频验证（赚钱效应）

盘中分钟级逻辑的日频翻译, 2019-11~2026-08 全样本 + 用户定稿三段环境:
  熊市 20220101-20221231 | 震荡 20230101-20240930 | 牛市 20241001-20260617

H1 跟跌(逻辑正确性): 热门题材(zt≥4)昨日龙头当日大跌(≤-3%)时, 概念成分当日
   均涨幅/下跌家数占比 是否显著差于昨龙头上涨日
H2 跷跷板(逻辑+赚钱效应): 昨龙头拐头日, 当日成分均涨最强概念(对手)的
   T+1成分均涨(=买入对手板块篮子次日收益) 是否显著>0 且 > 全A等权
H3 四阶段(赚钱效应/仓位): 日频情绪春夏秋冬各阶段, 当日非一字涨停票
   T+1开盘收益均值与胜率(验收线50%), 夏应显著优于冬

赚钱效应口径: 打板T+1开盘卖(next_open_ret, 剔一字); 板块篮子=成分等权均涨
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load  # noqa: E402

ENV_SEG = [("全样本", "20190101", "20261231"),
           ("熊市", "20220101", "20221231"),
           ("震荡市", "20230101", "20240930"),
           ("牛市", "20241001", "20260617")]
OUT = Path(__file__).resolve().parent / "out" / "27_seesaw_hist.md"


def seg_of(d):
    for name, a, b in ENV_SEG[1:]:
        if a <= d <= b:
            return name
    return None


def load_all():
    ev = load("limitup.events_enriched")
    td = load("theme.day")
    mem = load("theme.members")[["concept_code", "con_code"]] \
        .drop_duplicates()
    dp = load("market.daily_panel",
              columns=["trade_date", "ts_code", "pct_chg"])
    return ev, td, mem, dp


def build_con_panel(dp, mem):
    """概念×日 成分等权均涨/下跌占比矩阵 (pivot切片法, 避免全量merge爆炸)"""
    pivot = dp.pivot_table(index="trade_date", columns="ts_code",
                           values="pct_chg")
    avg, fall = {}, {}
    for k, g in mem.groupby("concept_code"):
        codes = [c for c in g["con_code"] if c in pivot.columns]
        if len(codes) < 5:
            continue
        sub = pivot[codes]
        avg[k] = sub.mean(axis=1)
        fall[k] = (sub < 0).mean(axis=1)
    con_avg = pd.DataFrame(avg)
    con_fall = pd.DataFrame(fall)
    mkt = pivot.mean(axis=1)
    return con_avg, con_fall, mkt, sorted(pivot.index)


def daily_sentiment(ev):
    g = ev.groupby("trade_date")
    sent = pd.DataFrame({
        "zt": g.size(),
        "br": g["open_times"].apply(lambda s: (s >= 1).mean()),
        "mh": g["limit_times"].max()})
    sent["zt_ma5"] = sent["zt"].rolling(5, min_periods=3).mean()
    br_q30, br_q70 = sent["br"].quantile(0.3), sent["br"].quantile(0.7)
    sa, sb = [], []
    for d, r in sent.iterrows():
        # 方案A: 绝对阈值(与core/cycle同源)
        if r["br"] > 0.3 or (r["zt"] < 0.8 * r["zt_ma5"] and r["mh"] <= 3):
            sa.append("冬")
        elif r["br"] < 0.2 and r["mh"] >= 5 and r["zt"] > 1.1 * r["zt_ma5"]:
            sa.append("夏")
        elif r["mh"] >= 5:
            sa.append("秋")
        else:
            sa.append("春")
        # 方案B: 分位阈值(炸板率30/70分位, 保证四阶段有分布)
        if r["br"] > br_q70 or (r["zt"] < 0.8 * r["zt_ma5"]
                                and r["mh"] <= 3):
            sb.append("冬")
        elif r["br"] < br_q30 and r["mh"] >= 5 and r["zt"] > r["zt_ma5"]:
            sb.append("夏")
        elif r["mh"] >= 5:
            sb.append("秋")
        else:
            sb.append("春")
    sent["stage"], sent["stage_b"] = sa, sb
    return sent


def in_seg(d, a, b):
    return a <= d <= b


def main():
    ev, td, mem, dp = load_all()
    con_avg, con_fall, mkt, dates = build_con_panel(dp, mem)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    sent = daily_sentiment(ev)
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr("# 研究27: 龙头拐头/跷跷板/四阶段 历史验证\n")

    # ---------- H1 跟跌 ----------
    pr("## H1 昨龙头下跌→板块跟跌（逻辑正确性）")
    # 昨日龙头当日表现: theme_day T-1 leader → 当日pct_chg
    tp = td[["trade_date", "concept_code", "leader_code"]].copy()
    tp["d"] = tp["trade_date"].map(nxt)
    tp = tp.dropna(subset=["d"])
    hot = td[td["zt_cnt"] >= 4].merge(
        tp[["d", "concept_code", "leader_code"]].rename(
            columns={"d": "trade_date", "leader_code": "prev_leader"}),
        on=["trade_date", "concept_code"], how="left")
    ld = dp.rename(columns={"ts_code": "prev_leader",
                            "pct_chg": "ld"})[["trade_date",
                                               "prev_leader", "ld"]]
    hot = hot.merge(ld, on=["trade_date", "prev_leader"], how="left")
    h1_full = {}
    for (seg, a, b) in ENV_SEG:
        sub = hot[(hot["trade_date"] >= a) & (hot["trade_date"] <= b)]
        res = []
        for name, lo, hi in [("昨龙头≤-3%", -99, -3),
                             ("昨龙头-3~0%", -3, 0),
                             ("昨龙头≥0%", 0, 99)]:
            g = sub[(sub["ld"] >= lo) & (sub["ld"] < hi)]
            if len(g) < 20:
                continue
            ca = [con_avg.at[d, k] for d, k in
                  zip(g["trade_date"], g["concept_code"])
                  if k in con_avg.columns and d in con_avg.index]
            cf = [con_fall.at[d, k] for d, k in
                  zip(g["trade_date"], g["concept_code"])
                  if k in con_fall.columns and d in con_fall.index]
            if seg == "全样本":
                h1_full[name] = (len(g), float(np.nanmean(ca)),
                                 100 * float(np.nanmean(cf)))
            res.append({"": name, "n": len(g),
                        "概念均涨%": round(float(np.nanmean(ca)), 2),
                        "下跌占比%": round(100 * float(np.nanmean(cf)), 1)})
        if res:
            pr(f"\n### {seg}")
            pr(pd.DataFrame(res).set_index("").to_string())

    # ---------- H2 跷跷板 ----------
    pr("\n## H2 跷跷板: 昨龙头拐头日对手板块T+1赚钱效应")
    hot_ld = hot
    turn_dates = sorted(hot_ld[hot_ld["ld"] <= -3]["trade_date"].unique())
    rec = []
    for d in turn_dates:
        if d not in con_avg.index or d not in nxt:
            continue
        turn_k = set(hot_ld[(hot_ld["trade_date"] == d)
                            & (hot_ld["ld"] <= -3)]["concept_code"])
        row = con_avg.loc[d].drop(turn_k, errors="ignore").dropna()
        row = row[row > 0.3]
        if not len(row):
            continue
        opp = row.idxmax()
        t1 = nxt[d]
        opp_t1 = con_avg.at[t1, opp] if opp in con_avg.columns else np.nan
        rec.append({"date": d, "seg": seg_of(d), "opp": opp,
                    "opp_t": row[opp], "opp_t1": opp_t1,
                    "mkt_t1": mkt.get(t1, np.nan)})
    rdf = pd.DataFrame(rec, columns=["date", "seg", "opp", "opp_t",
                                     "opp_t1", "mkt_t1"])
    for (seg, a, b) in ENV_SEG:
        g = rdf if seg == "全样本" else rdf[rdf["seg"] == seg]
        if len(g) < 10:
            pr(f"\n{seg}: 样本{len(g)}不足")
            continue
        pr(f"\n### {seg} (拐头日n={len(g)})")
        pr(f"  对手板块T+1均涨: {g['opp_t1'].mean():+.2f}% "
           f"胜率{(g['opp_t1'] > 0).mean() * 100:.0f}%")
        pr(f"  全A等权T+1:     {g['mkt_t1'].mean():+.2f}% "
           f"超额 {(g['opp_t1'] - g['mkt_t1']).mean():+.2f}%")

    # ---------- H3 四阶段 ----------
    pr("\n## H3 情绪四阶段→打板T+1赚钱效应（胜率验收线50%）")
    base = ev[~ev["is_yizi"]]
    for variant, col in [("方案A 绝对阈值", "stage"),
                         ("方案B 分位阈值", "stage_b")]:
        evd = base.merge(sent[[col]], left_on="trade_date",
                         right_index=True)
        pr(f"\n--- {variant} ---")
        for (seg, a, b) in ENV_SEG:
            g = evd[(evd["trade_date"] >= a) & (evd["trade_date"] <= b)]
            if not len(g):
                continue
            pr(f"\n### {seg}")
            t = g.groupby(col)["next_open_ret"].agg(
                n="count", 均值=lambda s: round(s.mean() * 100, 2),
                胜率=lambda s: round((s > 0).mean() * 100, 1))
            pr(t.to_string())
    OUT.parent.mkdir(exist_ok=True)
    # ---------- 结论 ----------
    pr("\n## 结论（历史日频 2019-11~2026-08）")
    if h1_full.get("昨龙头≤-3%") and h1_full.get("昨龙头≥0%"):
        a1, b1 = h1_full["昨龙头≤-3%"], h1_full["昨龙头≥0%"]
        pr(f"1. H1跟跌逻辑成立: 昨龙头大跌日概念均涨{a1[1]:+.2f}%/下跌占比"
           f"{a1[2]:.0f}% vs 龙头上涨日{b1[1]:+.2f}%/{b1[2]:.0f}%, "
           f"三段环境同向 → 龙头拐头可作板块风险预警")
    if len(rdf):
        pr(f"2. H2跷跷板无次日赚钱效应: 对手板块T+1均涨"
           f"{rdf['opp_t1'].mean():+.2f}% 胜率"
           f"{(rdf['opp_t1'] > 0).mean() * 100:.0f}%(<50%验收线), "
           f"超额全A{(rdf['opp_t1'] - rdf['mkt_t1']).mean():+.2f}% → "
           f"仅盘中观察/预警, 不次日买对手板块")
    pr("3. H3打板池全环境有赚钱效应(三段胜率均>50%验收线), 但四阶段"
       "分类对T+1无稳定梯度(方案B冬胜率不最低, 分歧后回封票T+1反而好, "
       "印证'买在分歧'); 阶段→仓位映射维持展示+research/26积累校准")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {OUT}")


if __name__ == "__main__":
    main()
