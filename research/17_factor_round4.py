# -*- coding: utf-8 -*-
"""研究17: 因子深挖第四轮(板性习惯族/入场时机/接力交互)

全离线, 数据: 16_enriched.parquet + 1m缓存 + events事件库(2019起)
  A. 历史封板习惯族(新挖): 近20交易日涨停次数h_cnt / 平均首封时刻h_ft
     / 平均炸板次数h_zb —— 个股"板性"是否可前向预测
  B. 入场时机: 触+1%后5分钟回踩深度 × 封板率(逆向选择检验) +
     挂单等回踩的成交率×条件EV模拟
  C. 昨涨停接力 × 今日形态交互
输出: research/out/17_round4.md
"""
import bisect
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


df = pd.read_parquet(OUT / "16_enriched.parquet")
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ALL_DAYS = sorted(ev["trade_date"].unique())

# ---------- A. 历史封板习惯族 ----------
say("构建历史板性特征...")
ev_s = ev.sort_values(["ts_code", "trade_date"])
hist_dates, hist_ft, hist_zb = {}, {}, {}
for c, g in ev_s.groupby("ts_code"):
    hist_dates[c] = g["trade_date"].tolist()
    hist_ft[c] = g["first_time"].astype(str).str.zfill(6).map(
        lambda s: int(s[:2]) * 60 + int(s[2:4])).tolist()
    hist_zb[c] = g["open_times"].fillna(0).tolist()


def habit(code, date):
    ds = hist_dates.get(code)
    if not ds:
        return np.nan, np.nan, np.nan
    i = bisect.bisect_left(ds, date)
    lo = max(0, i - 20)
    if i - lo == 0:
        return 0, np.nan, np.nan
    fts = hist_ft[code][lo:i]
    zbs = hist_zb[code][lo:i]
    return (i - lo,
            float(np.mean(fts)) if fts else np.nan,
            float(np.mean(zbs)))


h = [habit(r.ts_code, r.date) for r in df.itertuples()]
df["h_cnt"] = [x[0] for x in h]
df["h_ft"] = [x[1] for x in h]
df["h_zb"] = [x[2] for x in h]
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

say("# 研究17: 因子深挖第四轮")
say(f"\n板性覆盖: 有历史涨停样本 {df['h_cnt'].notna().mean():.0%}, "
    f"近20日有涨停 {(df['h_cnt'] > 0).mean():.0%}")


def dual(factor, scope_tr, scope_te, bins, label=""):
    say(f"\n`{factor}` {label}")
    say("| 桶 | train封板率(n) | test封板率(n) | test总收益% |")
    say("|---|---|---|---|")
    for lo, hi in zip(bins[:-1], bins[1:]):
        a = scope_tr[(scope_tr[factor] > lo) & (scope_tr[factor] <= hi)]
        b = scope_te[(scope_te[factor] > lo) & (scope_te[factor] <= hi)]
        at = f"{a['y'].mean():.0%}({len(a)})" if len(a) >= 20 else "-"
        bt = f"{b['y'].mean():.0%}({len(b)})" if len(b) >= 20 else "-"
        nr = b["next_ret"].dropna()
        nt = f"{nr.median():.2f}" if len(nr) >= 20 else "-"
        say(f"| ({lo},{hi}] | {at} | {bt} | {nt} |")


say("\n## A. 历史板性族(walk-forward双列)")
dual("h_cnt", tr, te, [-0.5, 0, 1, 2, 3, 99], "(近20日涨停次数)")
dual("h_ft", tr.dropna(subset=["h_ft"]), te.dropna(subset=["h_ft"]),
     [-1, 580, 610, 660, 800, 9999], "(历史平均首封时刻,分钟)")
dual("h_zb", tr.dropna(subset=["h_zb"]), te.dropna(subset=["h_zb"]),
     [-0.01, 0.3, 0.8, 1.5, 99], "(历史平均炸板次数)")

# ---------- B. 入场时机: 触板后回踩 ----------
say("\n构建触板后5分钟回踩深度...")
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


pb_depth = []
for date, grp in df.groupby("date"):
    cd = load_day(date)
    for r in grp.itertuples():
        if r.cohort != "L":
            pb_depth.append(np.nan)
            continue
        g = cd.get(r.ts_code)
        if g is None or len(g) < 10 or r.pre <= 0:
            pb_depth.append(np.nan)
            continue
        tm = [str(x)[8:12] for x in g["tm"]]
        hh, mm = divmod(int(r.td), 60)
        tgt = f"{hh:02d}{mm:02d}"
        if tgt not in tm:
            pb_depth.append(np.nan)
            continue
        j = tm.index(tgt)
        nxt = g.iloc[j + 1:j + 6]
        if len(nxt) < 3:
            pb_depth.append(np.nan)
            continue
        min_low = float(nxt["low"].min())
        pb_depth.append(r.pct_d - (min_low / r.pre - 1) * 100)
df["pb_depth"] = pb_depth
L = df[df["cohort"] == "L"]
Ltr, Lte = L[L["split"] == "train"], L[L["split"] == "test"]
say(f"回踩深度覆盖 {L['pb_depth'].notna().mean():.0%}")

say("\n## B. 触+1%后5分钟回踩(逆向选择检验)")
say("假设: 回踩深的票更弱 → 挂单等回踩有逆向选择")
dual("pb_depth", Ltr, Lte, [-0.01, 0.1, 0.3, 0.6, 1.0, 99],
     "(回踩深度,pct点)")

say("\n挂单等回踩模拟(test期, L组):")
say("| 挂单位置 | 成交率 | 成交者封板率 | 成交者总收益%(对挂单价) |")
say("|---|---|---|---|")
Lte2 = Lte.dropna(subset=["pb_depth"])
for d in [0.0, 0.2, 0.4, 0.6]:
    filled = Lte2[Lte2["pb_depth"] >= d]
    if len(filled) < 20:
        continue
    # 总收益对挂单价: next_ret(对触板价) + d(挂单更低d个点)
    adj = filled["next_ret"] + d
    say(f"| 触板价-{d} | {len(filled)/len(Lte2):.0%} "
        f"| {filled['y'].mean():.0%} | {adj.median():.2f} |")

# ---------- C. 接力交互 ----------
say("\n## C. 昨涨停接力 × 今日形态(test期)")
say("| 组合 | n | 封板率 | 总收益% |")
say("|---|---|---|---|")
rel = te[te["y_zt"] == 1]
combos = {
    "接力全体": rel,
    "接力 & G组高开(gap>2)": rel[(rel["cohort"] == "G") & (rel["gap"] > 2)],
    "接力 & G稳封相(odip≤0.05)": rel[(rel["cohort"] == "G")
                                     & (rel["odip"] <= 0.05)],
    "接力 & 竞价量爆(open_vr>5)": rel[(rel["cohort"] == "G")
                                      & (rel["open_vr"] > 5)],
    "接力 & L颠簸高": rel[(rel["cohort"] == "L")
                          & (rel["pathvol"] > 0.93)],
    "非接力 & G稳封相": te[(te["y_zt"] == 0) & (te["cohort"] == "G")
                           & (te["gap"] > 5.2) & (te["odip"] <= 0.05)
                           & (te["cm20"] == 0)],
}
for name, sub in combos.items():
    nr = sub["next_ret"].dropna()
    if len(sub) >= 15:
        say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {nr.median():.2f} |")

report = "\n".join(R)
(OUT / "17_round4.md").write_text(report, encoding="utf-8")
df.to_parquet(OUT / "17_enriched.parquet", index=False)
print(f"\n报告: {OUT}/17_round4.md")
