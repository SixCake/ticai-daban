# -*- coding: utf-8 -*-
"""研究16: 因子深挖第三轮(VWAP位置/相对强度/情绪环境/复合叠加)

全离线, 数据: 15_enriched.parquet + 1m缓存 + events事件库
  A. VWAP位置因子: 决策时刻价格 vs 日内VWAP(1m缓存重建), 回踩不破VWAP假设
  B. 日内相对强度: 决策涨幅在当日样本内百分位
  C. 市场情绪环境: 当日涨停家数冷/中/热 × 规则表现
  D. 复合叠加: S3a×昨日收强×竞价量 / L颠簸×vtrend×昨日收强, 增量验证
  E. bootstrap 95%CI 鲁棒性
输出: research/out/16_round3.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
CACHE1M = OUT / "1m_cache"
TRAIN_END = "20260430"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


df = pd.read_parquet(OUT / "15_enriched.parquet")
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
zt_cnt = ev.groupby("trade_date").size()
df["zt_day"] = df["date"].map(zt_cnt)
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

# ---------- A. VWAP位置(1m缓存重建) ----------
say("重建决策时刻VWAP位置...")
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


vwap_pos = []
for date, grp in df.groupby("date"):
    cd = load_day(date)
    for r in grp.itertuples():
        g = cd.get(r.ts_code)
        if g is None or len(g) < 5:
            vwap_pos.append(np.nan)
            continue
        tm = [str(x)[8:12] for x in g["tm"]]
        hh, mm = divmod(int(r.td), 60)
        tgt = f"{hh:02d}{mm:02d}"
        j = tm.index(tgt) if tgt in tm else min(4, len(g) - 1)
        seg = g.iloc[:j + 1]
        v = seg["volume"].values.astype(float)
        c = seg["close"].values.astype(float)
        if v.sum() <= 0 or r.pre <= 0:
            vwap_pos.append(np.nan)
            continue
        vwap = (c * v).sum() / v.sum()
        vwap_pos.append(r.pct_d - (vwap / r.pre - 1) * 100)  # 价-VWAP(pct点)
df["vwap_pos"] = vwap_pos
say(f"VWAP覆盖 {df['vwap_pos'].notna().mean():.0%}")
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

say("# 研究16: 因子深挖第三轮")


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


say("\n## A. VWAP位置因子(价格-日内VWAP, pct点)")
dual("vwap_pos", tr[tr["cohort"] == "L"], te[te["cohort"] == "L"],
     [-99, -0.5, 0, 0.5, 1.5, 99], "(L组)")
dual("vwap_pos", tr[tr["cohort"] == "G"], te[te["cohort"] == "G"],
     [-99, -0.5, 0, 0.5, 1.5, 99], "(G组)")

# ---------- B. 日内相对强度 ----------
say("\n## B. 日内相对强度(决策涨幅当日样本内百分位)")
df["rs"] = df.groupby("date")["pct_d"].rank(pct=True)
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
dual("rs", tr, te, [-0.01, 0.25, 0.5, 0.75, 0.9, 1.01], "(全体)")

# ---------- C. 情绪环境 ----------
say("\n## C. 市场情绪环境(当日涨停家数) × 核心规则")
q33, q66 = df["zt_day"].quantile([0.33, 0.66]).tolist()
df["sent"] = pd.cut(df["zt_day"], [-1, q33, q66, 9999],
                    labels=["冷", "中", "热"])
RULES = {
    "S3a高开稳封": (df["cohort"] == "G") & (df["gap"] > 5.2)
        & (df["odip"] <= 0.05) & (df["cm20"] == 0),
    "竞价量爆(open_vr>5&10cm)": (df["cohort"] == "G")
        & (df["open_vr"] > 5) & (df["cm20"] == 0),
    "L颠簸高": (df["cohort"] == "L") & (df["pathvol"] > 0.93)
        & (df["cm20"] == 0),
}
say("| 规则 | 冷日封板率(n) | 中日 | 热日 |")
say("|---|---|---|---|")
for name, cond in RULES.items():
    cells = []
    for s in ["冷", "中", "热"]:
        sub = df[cond & (df["sent"] == s)]
        cells.append(f"{sub['y'].mean():.0%}({len(sub)})" if len(sub) >= 10
                     else "-")
    say(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")

# ---------- D. 复合叠加 ----------
say("\n## D. 复合叠加(test期)")
say("| 组合 | n | 封板率 | 总收益% | 胜率 |")
say("|---|---|---|---|---|")
combos = {
    "S3a": RULES["S3a高开稳封"],
    "S3a & 昨收强(y_cpos>0.6)": RULES["S3a高开稳封"] & (df["y_cpos"] > 0.6),
    "S3a & 昨收强 & 竞价量(open_vr>2)": RULES["S3a高开稳封"]
        & (df["y_cpos"] > 0.6) & (df["open_vr"] > 2),
    "竞价量爆 & gap≤5.2(非一字)": RULES["竞价量爆(open_vr>5&10cm)"]
        & (df["gap"] <= 5.2),
    "竞价量爆 & 昨收强": RULES["竞价量爆(open_vr>5&10cm)"]
        & (df["y_cpos"] > 0.6),
    "L颠簸高": RULES["L颠簸高"],
    "L颠簸高 & vtrend>1.5": RULES["L颠簸高"] & (df["vtrend"] > 1.5),
    "L颠簸高 & 昨收强": RULES["L颠簸高"] & (df["y_cpos"] > 0.6),
    "L颠簸高 & VWAP上方(vwap_pos>0)": RULES["L颠簸高"]
        & (df["vwap_pos"] > 0),
}
for name, cond in combos.items():
    sub = te[cond.reindex(te.index).fillna(False)]
    nr = sub["next_ret"].dropna()
    if len(sub) >= 15:
        say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {nr.median():.2f} | {(nr > 0).mean():.0%} |")

# ---------- E. bootstrap ----------
say("\n## E. bootstrap 95%CI(封板率, 500次重抽样, test期)")
rng = np.random.default_rng(42)
say("| 规则 | 封板率 | 95%CI |")
say("|---|---|---|")
for name, cond in [("S3a", combos["S3a"]),
                   ("竞价量爆&gap≤5.2", combos["竞价量爆 & gap≤5.2(非一字)"]),
                   ("L颠簸高", combos["L颠簸高"]),
                   ("S3a&昨收强&竞价量", combos["S3a & 昨收强 & 竞价量(open_vr>2)"])]:
    sub = te[cond.reindex(te.index).fillna(False)]
    if len(sub) < 15:
        continue
    yv = sub["y"].values
    stats = []
    for _ in range(500):
        idx = rng.integers(0, len(yv), len(yv))
        stats.append(yv[idx].mean())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    say(f"| {name} | {yv.mean():.0%} | [{lo:.0%}, {hi:.0%}] |")

report = "\n".join(R)
(OUT / "16_round3.md").write_text(report, encoding="utf-8")
df.to_parquet(OUT / "16_enriched.parquet", index=False)
print(f"\n报告: {OUT}/16_round3.md")
