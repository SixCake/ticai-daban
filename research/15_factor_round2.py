# -*- coding: utf-8 -*-
"""研究15: 因子深挖第二轮(全离线, 复用14数据+1m缓存)

新增/补验:
  A. L组蓄势形态因子 walk-forward 补验(tight/drift/volramp/base_hi/vtrend)
  B. G组竞价量能 open_vr 前向验证 + gap×open_vr 交互
  C. 昨日形态族(新挖, 从1m缓存重建): 昨日涨幅y_ret/昨日量比y_vr/
     昨日收盘位置y_cpos/昨日上影y_upper/昨日涨停接力y_zt
  D. 候选规则三段行情(偏多/偏空/震荡)稳定性
  E. EV视角: 入场→次日收盘总收益(封板率不再单独作目标)
输出: research/out/15_round2.md
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


df = pd.read_parquet(OUT / "14_dual_oos.parquet")
df12 = pd.read_parquet(OUT / "12_expanded_oos.parquet")
regime_by = df12.groupby("date")["regime"].first().to_dict()
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ev_set = set(zip(ev["trade_date"], ev["ts_code"]))
ALL_DAYS = sorted(ev["trade_date"].unique())
prev_day = dict(zip(ALL_DAYS[1:], ALL_DAYS[:-1]))
prev2_day = dict(zip(ALL_DAYS[2:], ALL_DAYS[:-2]))

# ---------- C. 昨日形态族(1m缓存重建) ----------
say("重建昨日形态特征...")
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


y_ret_l, y_vr_l, y_cpos_l, y_upper_l, y_zt_l = [], [], [], [], []
for date, grp in df.groupby("date"):
    d1 = prev_day.get(date)
    d0 = prev2_day.get(date)
    c1 = load_day(d1) if d1 else {}
    c0 = load_day(d0) if d0 else {}
    for r in grp.itertuples():
        g1 = c1.get(r.ts_code)
        g0 = c0.get(r.ts_code)
        if g1 is None or len(g1) < 30:
            y_ret_l.append(np.nan)
            y_vr_l.append(np.nan)
            y_cpos_l.append(np.nan)
            y_upper_l.append(np.nan)
        else:
            yclose = float(g1["close"].iloc[-1])
            if g0 is not None and len(g0):
                pre = float(g0["close"].iloc[-1])
                y_ret_l.append((yclose / pre - 1) * 100 if pre > 0 else np.nan)
                v0 = float(g0["volume"].sum())
                y_vr_l.append(float(g1["volume"].sum()) / v0 if v0 > 0
                            else np.nan)
            else:
                y_ret_l.append(np.nan)
                y_vr_l.append(np.nan)
            hi = float(g1["high"].max())
            lo = float(g1["low"].min())
            y_cpos_l.append((yclose - lo) / (hi - lo) if hi > lo else 0.5)
            y_upper_l.append((hi - yclose) / (hi - lo) if hi > lo else 0.0)
        y_zt_l.append(int((d1, r.ts_code) in ev_set) if d1 else 0)
    if len(y_ret_l) % 5000 < len(grp):
        print(f"  {date} 累计{len(y_ret_l)}", flush=True)

df["y_ret"] = y_ret_l
df["y_vr"] = y_vr_l
df["y_cpos"] = y_cpos_l
df["y_upper"] = y_upper_l
df["y_zt"] = y_zt_l
df["regime"] = df["date"].map(regime_by).fillna("震荡")
df["split"] = np.where(df["date"] <= TRAIN_END, "train", "test")
df["total_ret"] = df["next_ret"]           # 入场→次日收盘(含当日持有段)
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
say(f"样本 {len(df)}, 昨日特征覆盖 {df['y_ret'].notna().mean():.0%}")

# ---------- 工具: 双段分桶(train发现 → test确认) ----------


def dual(factor, scope_tr, scope_te, bins, label=""):
    say(f"\n`{factor}` {label}")
    say("| 桶 | train封板率(n) | test封板率(n) | test次日% |")
    say("|---|---|---|---|")
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        a = scope_tr[(scope_tr[factor] > lo) & (scope_tr[factor] <= hi)]
        b = scope_te[(scope_te[factor] > lo) & (scope_te[factor] <= hi)]
        at = f"{a['y'].mean():.0%}({len(a)})" if len(a) >= 20 else "-"
        bt = f"{b['y'].mean():.0%}({len(b)})" if len(b) >= 20 else "-"
        nr = b["next_ret"].dropna()
        nt = f"{nr.median():.2f}" if len(nr) >= 20 else "-"
        say(f"| ({lo},{hi}] | {at} | {bt} | {nt} |")
        if len(b) >= 20:
            out.append((lo, hi, b["y"].mean()))
    return out


base_tr, base_te = tr["y"].mean(), te["y"].mean()
say(f"# 研究15: 因子深挖第二轮\n\n基准封板率 train {base_tr:.0%} / "
    f"test {base_te:.0%}")

# ---------- A. L组蓄势形态补验 ----------
say("\n## A. L组(低拉)蓄势形态因子")
Ltr, Lte = tr[tr["cohort"] == "L"], te[te["cohort"] == "L"]
dual("tight", Ltr, Lte, [-0.01, 0.1, 0.2, 0.35, 99], "(平台紧度)")
dual("drift", Ltr, Lte, [-99, -0.5, 0, 0.5, 1.5, 99], "(蓄势段漂移)")
dual("volramp", Ltr, Lte, [-0.01, 0.5, 1, 2, 4, 999], "(量能爬坡)")
dual("base_hi", Ltr, Lte, [-0.01, 0.3, 0.6, 0.9, 1.2, 99], "(蓄势段最高)")
dual("vtrend", Ltr, Lte, [-0.01, 0.8, 1.5, 3, 999], "(近3min量能/前3min)")

# ---------- B. G组竞价量能 ----------
say("\n## B. G组(高开)竞价量能 open_vr")
Gtr, Gte = tr[tr["cohort"] == "G"], te[te["cohort"] == "G"]
dual("open_vr", Gtr, Gte, [-0.1, 1, 2, 5, 10, 9999], "(竞价量比)")
say("\ngap×open_vr 交互(test期):")
say("| gap档 | open_vr档 | n | 封板率 | 次日% |")
say("|---|---|---|---|---|")
for glo, ghi, gl in [(1, 3, "gap1-3"), (3, 5.2, "gap3-5.2"),
                     (5.2, 99, "gap>5.2")]:
    for vlo, vhi, vl in [(0, 2, "vr<2"), (2, 5, "vr2-5"), (5, 9999, "vr>5")]:
        b = Gte[(Gte["gap"] > glo) & (Gte["gap"] <= ghi)
                & (Gte["open_vr"] > vlo) & (Gte["open_vr"] <= vhi)]
        nr = b["next_ret"].dropna()
        if len(b) >= 30:
            say(f"| {gl} | {vl} | {len(b)} | {b['y'].mean():.0%} "
                f"| {nr.median():.2f} |")

# ---------- C. 昨日形态族 ----------
say("\n## C. 昨日形态族(全体)")
dual("y_ret", tr, te, [-99, -2, 0, 3, 6, 10, 99], "(昨日涨幅)")
dual("y_vr", tr, te, [-0.01, 0.5, 1, 2, 4, 999], "(昨日量比)")
dual("y_cpos", tr, te, [-0.01, 0.3, 0.6, 0.85, 1.01], "(昨日收盘位置)")
dual("y_upper", tr, te, [-0.01, 0.1, 0.3, 0.6, 1.01], "(昨日上影占比)")
say("\n昨日涨停接力:")
say("| 组 | train(n) | test(n) | test次日% |")
say("|---|---|---|---|")
for v, lab in [(1, "昨涨停"), (0, "非接力")]:
    a, b = tr[tr["y_zt"] == v], te[te["y_zt"] == v]
    nr = b["next_ret"].dropna()
    say(f"| {lab} | {a['y'].mean():.0%}({len(a)}) "
        f"| {b['y'].mean():.0%}({len(b)}) | {nr.median():.2f} |")

# ---------- D/E 候选规则: 三段行情 + EV ----------
say("\n## D/E 候选规则: 三段行情稳定性 + EV(入场→次日收盘)")
RULES = {
    "S3a 高开稳封(gap>5.2&odip≤0.05&10cm)": (df["cohort"] == "G")
        & (df["gap"] > 5.2) & (df["odip"] <= 0.05) & (df["cm20"] == 0),
    "G竞价量(open_vr>5&10cm)": (df["cohort"] == "G")
        & (df["open_vr"] > 5) & (df["cm20"] == 0),
    "L颠簸高(pathvol>0.93&10cm)": (df["cohort"] == "L")
        & (df["pathvol"] > 0.93) & (df["cm20"] == 0),
    "L暴拉(r3>4.8)": (df["cohort"] == "L") & (df["r3"] > 4.8),
}
say("| 规则 | test封板率 | 偏多 | 偏空 | 震荡 | test总收益%(中位) | 胜率 |")
say("|---|---|---|---|---|---|---|")
for name, cond in RULES.items():
    sub_te = te[cond.reindex(te.index).fillna(False)]
    if sub_te.empty:
        continue
    cells = []
    for rg in ["偏多", "偏空", "震荡"]:
        s = sub_te[sub_te["regime"] == rg]
        cells.append(f"{s['y'].mean():.0%}({len(s)})" if len(s) >= 10 else "-")
    nr = sub_te["total_ret"].dropna()
    say(f"| {name} | {sub_te['y'].mean():.0%}({len(sub_te)}) "
        f"| {cells[0]} | {cells[1]} | {cells[2]} "
        f"| {nr.median():.2f} | {(nr > 0).mean():.0%} |")

# 昨日因子与前向规则交互
say("\n昨日因子加持效果(test期, 在S3a/L颠簸高规则内):")
say("| 规则+昨日条件 | n | 封板率 | 总收益% |")
say("|---|---|---|---|")
for rname, cond in [("S3a", RULES["S3a 高开稳封(gap>5.2&odip≤0.05&10cm)"]),
                    ("L颠簸高", RULES["L颠簸高(pathvol>0.93&10cm)"])]:
    sub = te[cond.reindex(te.index).fillna(False)]
    for cname, c2 in [("昨日收强(y_cpos>0.6)", sub["y_cpos"] > 0.6),
                      ("昨日收弱(y_cpos≤0.6)", sub["y_cpos"] <= 0.6),
                      ("昨日放量(y_vr>1.5)", sub["y_vr"] > 1.5),
                      ("昨日缩量(y_vr≤0.8)", sub["y_vr"] <= 0.8)]:
        s2 = sub[c2.fillna(False)]
        nr = s2["total_ret"].dropna()
        if len(s2) >= 15:
            say(f"| {rname}+{cname} | {len(s2)} | {s2['y'].mean():.0%} "
                f"| {nr.median():.2f} |")

report = "\n".join(R)
(OUT / "15_round2.md").write_text(report, encoding="utf-8")
df.to_parquet(OUT / "15_enriched.parquet", index=False)
print(f"\n报告: {OUT}/15_round2.md")
