# -*- coding: utf-8 -*-
"""研究08: 半路抓涨停因子 多日OOS验证(20260812-20260825)

研究07单日发现: 高位(≥+7%)半路场景板型主导(10cm 93% vs 20cm 18%),
规则"10cm & prob≥0.5"精准93%。prob无法历史重建, 本脚本验证可重建因子:
  H15 板型效应在多日是否稳定(10cm vs 20cm 触+7%后封板率)
  H16 量比(触板时刻口径)是否增益
  H17 时段效应(早盘 vs 午后)是否稳定
  H18 规则命中队列的次日收益期望(可交易性)

数据: QMT FormulaServer 1m历史(2025-07-21起) + 1d横截面; 涨停事件库定正例,
负例=当日收盘涨幅5~9.7%(10cm)/8~19%(20cm)的非涨停股(盘中大概率触+7%)。
决策时刻=首根 high≥昨收×1.07 的分钟bar, 特征全部取该bar及之前(无未来信息)。
输出: research/out/08_halfway_oos.md
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


# ---------- QMT 客户端 ----------
BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()
sys.path.insert(0, str(BIGQMT_SRC))
os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
from bigqmt_signal_trader.xtquant_compat import configure, xtdata  # noqa: E402
configure(redis_config={"formula_server": {"failure_cooldown_seconds": 5}})

ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
DAYS = sorted(ev["trade_date"].unique())[-30:]
say(f"# 研究08: 半路因子多日OOS ({DAYS[0]}~{DAYS[-1]}, {len(DAYS)}个交易日)")


def limit_ratio(code: str) -> float:
    return 0.20 if code[:2] in ("30", "68") else 0.10


def day_1m(codes: list, day: str) -> dict:
    """分批拉当日1m, 返回 {code: DataFrame}"""
    out = {}
    for i in range(0, len(codes), 80):
        batch = codes[i:i + 80]
        try:
            res = xtdata.get_market_data_ex(
                field_list=["high", "close", "volume", "amount"],
                stock_list=batch, period="1m",
                start_time=day + "091500", end_time=day + "150500",
                dividend_type="none", chunk_size=0, timeout_seconds=30)
            out.update(res or {})
        except Exception as e:
            print(f"1m批次失败 {day} {i}: {e}", flush=True)
    return out


def day_1d(codes: list, day: str, count=6) -> dict:
    """以day为终点的日bar(含前收/前5日量)"""
    out = {}
    for i in range(0, len(codes), 400):
        batch = codes[i:i + 400]
        try:
            res = xtdata.get_market_data_ex(
                field_list=["close", "volume", "high"],
                stock_list=batch, period="1d", end_time=day,
                count=count, dividend_type="none", chunk_size=0,
                timeout_seconds=30)
            out.update(res or {})
        except Exception as e:
            print(f"1d批次失败 {day} {i}: {e}", flush=True)
    return out


rows = []
for di, day in enumerate(DAYS):
    t0 = time.time()
    # 1) 全市场1d横截面: 前收/收盘涨幅 → 选负例候选
    uni = sorted(load("theme.members")["con_code"].unique())
    uni = [c for c in uni if c.endswith((".SH", ".SZ"))
           and c[:2] in ("60", "68", "00", "30")]
    d1 = day_1d(uni, day, count=6)
    cand = {}
    for c, df in d1.items():
        try:
            if df is None or len(df) < 2:
                continue
            if str(df.index[-1])[:8] != day:
                continue
            pre = float(df["close"].iloc[-2])
            close = float(df["close"].iloc[-1])
            if pre <= 0 or close <= 0:
                continue
            cand[c] = {"pre": pre, "close_pct": (close / pre - 1) * 100,
                       "avg5v": float(df["volume"].iloc[:-1].tail(5).mean())}
        except Exception:
            continue
    # 2) 正例(涨停事件, 排一字/ST) + 负例(收盘5~9.7%/8~19%非涨停)
    zt = ev[(ev["trade_date"] == day) & ~ev["is_yizi"] & ~ev["is_st"]]
    pos = {r.ts_code for r in zt.itertuples()}
    neg = set()
    for c, v in cand.items():
        r = limit_ratio(c)
        lp_pct = r * 100 - 0.3
        lo, hi = (5.0, lp_pct) if r == 0.10 else (8.0, lp_pct)
        if lo <= v["close_pct"] < hi and c not in pos:
            neg.add(c)
    codes = sorted(pos | neg)
    m1 = day_1m(codes, day)
    # 3) 逐票重建决策时刻
    for c in codes:
        df = m1.get(c)
        if df is None or len(df) < 10:
            continue
        df = df[[str(ix)[:8] == day for ix in df.index]]
        df = df[df["volume"] > 0]           # 过滤假bar/停牌
        if len(df) < 10:
            continue
        v = cand.get(c, {})
        pre = v.get("pre")
        if not pre:
            continue
        thr7 = pre * 1.07
        lp = round(pre * (1 + limit_ratio(c)), 2)
        hit = df["high"] >= thr7
        if not hit.any():
            continue                        # 未触+7%(负例里的假候选)
        j = hit.values.argmax()
        tbar = str(df.index[j])
        hhmm = int(tbar[8:10]) * 60 + int(tbar[10:12])
        entry = float(df["close"].iloc[j])
        # 触板时刻量比: 累计量/已进行分钟 / (近5日均量/240)
        emin = max(j + 1, 1)
        a5 = v.get("avg5v", 0.0)
        vr7 = (df["volume"].iloc[:j + 1].sum() / emin) / (a5 / 240) \
            if a5 > 0 else np.nan
        sealed_close = float(df["close"].iloc[-1]) >= lp * 0.995
        touch_lp = bool((df["high"] >= lp * 0.998).any())
        rows.append({
            "date": day, "ts_code": c, "pos": c in pos,
            "cm20": int(c[:2] in ("30", "68")),
            "t7": hhmm, "entry_pct": (entry / pre - 1) * 100,
            "vr7": vr7, "sealed_close": sealed_close,
            "touch_lp": touch_lp, "entry": entry,
            "day_close": float(df["close"].iloc[-1]),
        })
    say(f"进度 {day}: 样本累计{len(rows)} 耗时{time.time()-t0:.0f}s")

df = pd.DataFrame(rows)
df["y"] = df["sealed_close"].astype(int)

# ---------- 验证 ----------
say(f"\n触+7%样本: {len(df)} = 涨停股{int(df['pos'].sum())} "
    f"负例{int((~df['pos']).sum())}; 整体封板率 {df['y'].mean():.0%}")


def check(name, factor, expect, scope=None, nbin=4):
    d = (scope if scope is not None else df).dropna(subset=[factor])
    if len(d) < 40:
        say(f"\n### {name}\n样本不足")
        return
    try:
        d = d.assign(bin=pd.qcut(d[factor], nbin, duplicates="drop"))
    except ValueError:
        d = d.assign(bin=pd.cut(d[factor], nbin))
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), zt=("y", "mean"), fmed=(factor, "median"))
    say(f"\n### {name}\n`{factor}`(预期{expect}) | 桶 | n | 封板率 |")
    say("|---|---|---|")
    for b, r in g.iterrows():
        say(f"| {b} | {int(r['n'])} | {r['zt']:.0%} |")
    rates = g["zt"].tolist()
    up = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    dn = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
    spread = max(rates) - min(rates)
    v = ("支持" if (expect == "↑" and up) or (expect == "↓" and dn)
         else "否定(反向)" if (expect == "↑" and dn) or (expect == "↓" and up)
         else "部分(非单调)")
    if spread < 0.05:
        v = "否定(无区分度)"
    say(f"**裁决: {v}**, 极差{spread:.0%}")


say("\n## H15 板型效应多日稳定性")
say("| 日期 | 10cm封板率(n) | 20cm封板率(n) |")
say("|---|---|---|")
for day in DAYS:
    s = df[df["date"] == day]
    a = s[s["cm20"] == 0]
    b = s[s["cm20"] == 1]
    say(f"| {day} | {a['y'].mean():.0%}({len(a)}) "
        f"| {b['y'].mean():.0%}({len(b)}) |")
a = df[df["cm20"] == 0]
b = df[df["cm20"] == 1]
say(f"\n全期: 10cm {a['y'].mean():.0%}(n={len(a)}) vs "
    f"20cm {b['y'].mean():.0%}(n={len(b)})")

check("H16 触+7%时刻量比", "vr7", "↑")
check("H17 触+7%时刻(早→晚)", "t7", "↓")

say("\n## H18 规则队列次日收益(可交易性)")
say("入场价=触+7%分钟bar收盘, 次日收益=次日收盘/入场价-1")
# 负例次日收盘/最高: 按日批量拉下一交易日1d
all_days = sorted(set(ev["trade_date"].unique()) | {"20260826"})
next_of = dict(zip(all_days[:-1], all_days[1:]))
next_close_map, next_high_map = {}, {}
for day in DAYS:
    nd = next_of.get(day)
    if not nd:
        continue
    codes_d = sorted(df[df["date"] == day]["ts_code"])
    res = day_1d(codes_d, nd, count=2)
    for c, d2 in res.items():
        try:
            rows2 = [(str(ix)[:8], float(cl), float(hi))
                     for ix, cl, hi in zip(d2.index, d2["close"],
                                           d2["high"] if "high" in d2
                                           else d2["close"])]
            for dstr, cl, hi in rows2:
                if dstr == nd and cl > 0:
                    next_close_map[(day, c)] = cl
                    next_high_map[(day, c)] = hi
        except Exception:
            continue
df["next_ret"] = [
    (next_close_map[(r.date, r.ts_code)] / r.entry - 1) * 100
    if (r.date, r.ts_code) in next_close_map else np.nan
    for r in df.itertuples()]
df["next_high_ret"] = [
    (next_high_map[(r.date, r.ts_code)] / r.entry - 1) * 100
    if (r.date, r.ts_code) in next_high_map else np.nan
    for r in df.itertuples()]
# 当日持有收益(触+7%入场 → 当日收盘): 封板者≈+2~3%肉, 未封者可能大面
df["day_ret"] = (df["day_close"] / df["entry"] - 1) * 100
pos_med = df[df["pos"] & df["next_ret"].notna()]["next_ret"].median()
say(f"(正例次日收益中位 {pos_med:.2f}% 与事件库 next_close_ret 口径互验)")
rules = {
    "全体触+7%": df,
    "仅10cm": df[df["cm20"] == 0],
    "10cm & vr7≥2": df[(df["cm20"] == 0) & (df["vr7"] >= 2)],
    "10cm & t7<600(10点前)": df[(df["cm20"] == 0) & (df["t7"] < 600)],
    "20cm": df[df["cm20"] == 1],
}
say("| 规则 | n | 封板率 | 当日收益% | 次日收益% | 次日胜率 |")
say("|---|---|---|---|---|---|")
for name, sub in rules.items():
    nr = sub[sub["next_ret"].notna()]["next_ret"]
    dr = sub["day_ret"]
    wr = (nr > 0).mean() if len(nr) else float("nan")
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {dr.median():.2f} | {nr.median():.2f}(n={len(nr)}) | {wr:.0%} |")

say("\n## H19 封板率与收益分离的根源: 封住 vs 未封住的条件收益")
say("(同为触+7%入场, 按当日是否封住分组)")
say("| 组 | n | 当日收益% | 次日收益% | 次日胜率 | 次日冲高% |")
say("|---|---|---|---|---|---|")
for lab, cond in [("10cm封住", (df["cm20"] == 0) & df["y"].astype(bool)),
                  ("10cm未封", (df["cm20"] == 0) & ~df["y"].astype(bool)),
                  ("20cm封住", (df["cm20"] == 1) & df["y"].astype(bool)),
                  ("20cm未封", (df["cm20"] == 1) & ~df["y"].astype(bool))]:
    sub = df[cond]
    nr = sub[sub["next_ret"].notna()]["next_ret"]
    nh = sub[sub["next_high_ret"].notna()]["next_high_ret"]
    wr = (nr > 0).mean() if len(nr) else float("nan")
    say(f"| {lab} | {len(sub)} | {sub['day_ret'].median():.2f} "
        f"| {nr.median():.2f} | {wr:.0%} | {nh.median():.2f} |")

say("\n## H20 板型×量比 完整决策表(封板率/当日/次日)")
say("| 组 | n | 封板率 | 当日% | 次日% | 次日胜率 |")
say("|---|---|---|---|---|---|")
for cm, lab in [(0, "10cm"), (1, "20cm")]:
    for lo, hi, vlab in [(0, 2, "vr<2"), (2, 5, "vr2-5"),
                         (5, 13, "vr5-13"), (13, 999, "vr≥13")]:
        sub = df[(df["cm20"] == cm) & (df["vr7"] >= lo) & (df["vr7"] < hi)]
        if len(sub) < 30:
            continue
        nr = sub[sub["next_ret"].notna()]["next_ret"]
        wr = (nr > 0).mean() if len(nr) else float("nan")
        say(f"| {lab}&{vlab} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {sub['day_ret'].median():.2f} | {nr.median():.2f} | {wr:.0%} |")

say("\n## H21 跨日稳定性(排除单日/牛市beta)")
say("同日对照: 触+7%的 10cm vs 20cm 次日收益中位")
say("| 日期 | 10cm次日% | 20cm次日% | 差值 |")
say("|---|---|---|---|")
wins = 0
for d in DAYS:
    s = df[df["date"] == d]
    a = s[(s["cm20"] == 0) & s["next_ret"].notna()]["next_ret"].median()
    b = s[(s["cm20"] == 1) & s["next_ret"].notna()]["next_ret"].median()
    if pd.notna(a) and pd.notna(b):
        wins += 1 if b > a else 0
        say(f"| {d} | {a:+.2f} | {b:+.2f} | {b - a:+.2f} |")
say(f"\n20cm跑赢天数: {wins}/{len(DAYS)}")
sub = df[(df["cm20"] == 1) & (df["vr7"] >= 2) & (df["vr7"] < 5)]
g = sub.groupby("date")["next_ret"].agg(["median", "count"]).round(2)
say("\n20cm&vr2-5 分日次日收益中位:")
say("| 日期 | 中位% | n |")
say("|---|---|---|")
for d, r in g.iterrows():
    say(f"| {d} | {r['median']:+.2f} | {int(r['count'])} |")
say(f"为正天数: {int((g['median'] > 0).sum())}/{len(g)}")

report = "\n".join(R)
(OUT / "08_halfway_oos.md").write_text(report, encoding="utf-8")
df.to_parquet(OUT / "08_halfway_oos.parquet", index=False)
print(f"\n报告: {OUT}/08_halfway_oos.md")
