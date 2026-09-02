# -*- coding: utf-8 -*-
"""研究18: 第五轮迭代(研究17遗留问题闭环)

  A. h_cnt=0(首次涨停)train/test不一致复核: 分队列+60日窗口双口径
  B. G组(高开)挂单等回踩验证(L组已证无逆向选择)
  C. 接力×G稳封相: 月度稳定性 + bootstrap
输出: research/out/18_round5.md; 结论回写 docs/research_06 日志
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
CACHE1M = OUT / "1m_cache"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


df = pd.read_parquet(OUT / "17_enriched.parquet")
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

say("# 研究18: 第五轮迭代")

# ---------- A. h_cnt 复核 ----------
say("\n## A. 近20日涨停次数 h_cnt 复核(分队列)")
say("| 队列 | 桶 | train(n) | test(n) |")
say("|---|---|---|---|")
for cq in ["G", "L"]:
    for lo, hi in [(-0.5, 0), (0, 1), (1, 3), (3, 99)]:
        a = tr[(tr["cohort"] == cq) & (tr["h_cnt"] > lo)
               & (tr["h_cnt"] <= hi)]
        b = te[(te["cohort"] == cq) & (te["h_cnt"] > lo)
               & (te["h_cnt"] <= hi)]
        at = f"{a['y'].mean():.0%}({len(a)})" if len(a) >= 20 else "-"
        bt = f"{b['y'].mean():.0%}({len(b)})" if len(b) >= 20 else "-"
        say(f"| {cq} | ({lo},{hi}] | {at} | {bt} |")

# 60日窗口口径重建
import bisect  # noqa: E402
ev_s = ev.sort_values(["ts_code", "trade_date"])
hd = {c: g["trade_date"].tolist() for c, g in ev_s.groupby("ts_code")}


def cnt60(code, date):
    ds = hd.get(code)
    if not ds:
        return 0
    i = bisect.bisect_left(ds, date)
    lo = max(0, i - 60)
    return i - lo


df["h_cnt60"] = [cnt60(r.ts_code, r.date) for r in df.itertuples()]
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
say("\n60日窗口口径 h_cnt60:")
say("| 桶 | train(n) | test(n) | test总收益% |")
say("|---|---|---|---|")
for lo, hi in [(-0.5, 0), (0, 2), (2, 5), (5, 10), (10, 999)]:
    a = tr[(tr["h_cnt60"] > lo) & (tr["h_cnt60"] <= hi)]
    b = te[(te["h_cnt60"] > lo) & (te["h_cnt60"] <= hi)]
    at = f"{a['y'].mean():.0%}({len(a)})" if len(a) >= 20 else "-"
    bt = f"{b['y'].mean():.0%}({len(b)})" if len(b) >= 20 else "-"
    nr = b["next_ret"].dropna()
    nt = f"{nr.median():.2f}" if len(nr) >= 20 else "-"
    say(f"| ({lo},{hi}] | {at} | {bt} | {nt} |")

# ---------- B. G组挂单等回踩 ----------
say("\n## B. G组(高开)挂单等回踩(开盘后10min)")
day_cache = {}


def load_day(day):
    if day in day_cache:
        return day_cache[day]
    cf = CACHE1M / f"1m_{day}.parquet"
    d = {}
    if cf.exists():
        try:
            ck = pd.read_parquet(cf)
            for c, g in ck.groupby("code"):
                d[c] = g.reset_index(drop=True)
        except Exception:
            pass
    day_cache[day] = d
    return d


G = df[df["cohort"] == "G"].copy()
pb = []
for date, grp in G.groupby("date"):
    cd = load_day(date)
    for r in grp.itertuples():
        g = cd.get(r.ts_code)
        if g is None or len(g) < 12 or r.pre <= 0:
            pb.append(np.nan)
            continue
        nxt = g.iloc[1:11]              # 开盘后第2~11分钟
        min_low = float(nxt["low"].min())
        d_pct = float(g["close"].iloc[0])   # 首bar收盘≈决策价
        pb.append((d_pct - min_low) / r.pre * 100)
G["pb_g"] = pb
Gtr, Gte = G[G["split"] == "train"], G[G["split"] == "test"]
say(f"回踩覆盖 {G['pb_g'].notna().mean():.0%}")
say("\n回踩深度×封板率(逆向选择检验):")
say("| 桶 | train(n) | test(n) |")
say("|---|---|---|")
for lo, hi in [(-0.01, 0.2), (0.2, 0.8), (0.8, 1.5), (1.5, 3), (3, 99)]:
    a = Gtr[(Gtr["pb_g"] > lo) & (Gtr["pb_g"] <= hi)]
    b = Gte[(Gte["pb_g"] > lo) & (Gte["pb_g"] <= hi)]
    at = f"{a['y'].mean():.0%}({len(a)})" if len(a) >= 20 else "-"
    bt = f"{b['y'].mean():.0%}({len(b)})" if len(b) >= 20 else "-"
    say(f"| ({lo},{hi}] | {at} | {bt} |")
say("\n挂单模拟(test期G组, 对开盘首bar价):")
say("| 挂单深度 | 成交率 | 成交者封板率 | 总收益%(对挂单价) |")
say("|---|---|---|---|")
Gte2 = Gte.dropna(subset=["pb_g"])
for d in [0.0, 0.5, 1.0, 2.0]:
    filled = Gte2[Gte2["pb_g"] >= d]
    if len(filled) < 30:
        continue
    adj = filled["next_ret"] + d
    say(f"| -{d} | {len(filled)/len(Gte2):.0%} | {filled['y'].mean():.0%} "
        f"| {adj.median():.2f} |")

# ---------- C. 接力×稳封相 稳定性 ----------
say("\n## C. 接力×G稳封相 月度稳定性 + bootstrap")
cond = ((df["y_zt"] == 1) & (df["cohort"] == "G") & (df["odip"] <= 0.05))
df["month"] = df["date"].str[:6]
say("| 月份 | 封板率(n) |")
say("|---|---|")
for m, g in df[cond].groupby("month"):
    if len(g) >= 10:
        say(f"| {m} | {g['y'].mean():.0%}({len(g)}) |")
sub = te[cond.reindex(te.index).fillna(False)]
rng = np.random.default_rng(42)
yv = sub["y"].values
stats = [yv[rng.integers(0, len(yv), len(yv))].mean() for _ in range(500)]
lo, hi = np.quantile(stats, [0.025, 0.975])
nr = sub["next_ret"].dropna()
say(f"\ntest期: 封板率 {yv.mean():.0%} [{lo:.0%},{hi:.0%}], "
    f"EV {nr.median():.2f}%, 胜率 {(nr > 0).mean():.0%} (n={len(sub)})")

report = "\n".join(R)
(OUT / "18_round5.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/18_round5.md")
