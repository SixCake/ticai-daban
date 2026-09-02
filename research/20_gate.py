# -*- coding: utf-8 -*-
"""研究20: 指数/情绪环境闸门(坏日识别)

研究19发现 20260813/20260820 全规则胜率集体跳水。假设: 坏日可被
决策前可见的指数/情绪信号识别:
  H1 昨日上证跌幅(idx_prev_ret)
  H2 当日上证开盘跳空(idx_open_ret, 09:25竞价后决策前可见)
  H3 昨日涨停家数(zt_prev, 情绪高度)
闸门回测: 关闭坏日信号后, 规则组合胜率/EV改善幅度
输出: research/out/20_gate.md
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()
sys.path.insert(0, str(BIGQMT_SRC))
os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
from bigqmt_signal_trader.xtquant_compat import configure, xtdata  # noqa: E402
configure(redis_config={"formula_server": {"failure_cooldown_seconds": 5}})

df = pd.read_parquet(OUT / "17_enriched.parquet")
df["win"] = (df["next_ret"] > 0).astype(int)
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
zt_cnt = ev.groupby("trade_date").size()

# ---------- 指数日线 ----------
res = xtdata.get_market_data_ex(
    field_list=["open", "close"], stock_list=["000001.SH"], period="1d",
    start_time="20250801", end_time="20260826",
    dividend_type="none", chunk_size=0, timeout_seconds=20)
idx = res["000001.SH"]
idx.index = [str(ix)[:8] for ix in idx.index]
idx["prev_ret"] = idx["close"].pct_change() * 100
idx["open_ret"] = (idx["open"] / idx["close"].shift(1) - 1) * 100
idx_map_prev = idx["prev_ret"].to_dict()
idx_map_open = idx["open_ret"].to_dict()

days = sorted(df["date"].unique())
prev_day = dict(zip(days[1:], days[:-1]))
df["idx_prev"] = df["date"].map(idx_map_prev)
df["idx_open"] = df["date"].map(idx_map_open)
df["zt_prev"] = df["date"].map(lambda d: zt_cnt.get(prev_day.get(d), np.nan))

say("# 研究20: 指数/情绪环境闸门")

# ---------- 每日聚合胜率(定义坏日) ----------
RULES = {
    "竞价量爆(非一字)": (df["cohort"] == "G") & (df["open_vr"] > 5)
        & (df["cm20"] == 0) & (df["gap"] <= 5.2),
    "高开剧震": (df["cohort"] == "G") & (df["gap"] <= 5.2)
        & (df["amp3"] > 4.3) & (df["cm20"] == 0),
    "接力×稳封相": (df["y_zt"] == 1) & (df["cohort"] == "G")
        & (df["odip"] <= 0.05),
}
pool = pd.Series(False, index=df.index)
for cond in RULES.values():
    pool |= cond
day_wr = df[pool].groupby("date")["win"].mean()
bad_days = set(day_wr[day_wr < 0.45].index)
say(f"\n主力规则池每日胜率 <45% 定义坏日: {len(bad_days)}天")
say(f"坏日列表(近30): {sorted(d for d in bad_days if d >= '20260101')}")

# ---------- H1-H3 坏日识别 ----------
dd = pd.DataFrame({
    "date": sorted(df["date"].unique()),
}).set_index("date")
dd["wr"] = day_wr
dd["idx_prev"] = dd.index.map(idx_map_prev)
dd["idx_open"] = dd.index.map(idx_map_open)
dd["zt_prev"] = dd.index.map(lambda d: zt_cnt.get(prev_day.get(d), np.nan))
dd["bad"] = dd.index.isin(bad_days).astype(int)


def check_gate(f, bins):
    say(f"\n`{f}` × 坏日占比/当日胜率:")
    say("| 桶 | 天数 | 坏日占比 | 平均日胜率 |")
    say("|---|---|---|---|")
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = dd[(dd[f] > lo) & (dd[f] <= hi)].dropna(subset=["wr"])
        if len(s) >= 10:
            say(f"| ({lo},{hi}] | {len(s)} | {s['bad'].mean():.0%} "
                f"| {s['wr'].mean():.0%} |")


check_gate("idx_prev", [-99, -1.5, -0.5, 0, 0.5, 99])
check_gate("idx_open", [-99, -0.8, -0.3, 0, 0.3, 99])
check_gate("zt_prev", [-1, 30, 50, 70, 999])

# ---------- 闸门回测 ----------
say("\n## 闸门回测(主力规则池, 全窗口)")
say("闸门定义: 满足任一即关闸 — idx_open≤-0.8 或 idx_prev≤-1.5 或 zt_prev≤30")
gate_off = ((df["idx_open"] <= -0.8) | (df["idx_prev"] <= -1.5)
            | (df["zt_prev"] <= 30))
sub = df[pool]
on, off = sub[~gate_off.reindex(sub.index)], sub[gate_off.reindex(sub.index)]
say(f"\n| 状态 | 信号数 | 封板率 | 胜率 | EV% | 覆盖天数 |")
say("|---|---|---|---|---|---|")
for lab, s in [("开闸", on), ("关闸日", off)]:
    nr = s["next_ret"].dropna()
    say(f"| {lab} | {len(s)} | {s['y'].mean():.0%} | {s['win'].mean():.0%} "
        f"| {nr.median():.2f} | {s['date'].nunique()} |")
all_nr = sub["next_ret"].dropna()
on_nr = on["next_ret"].dropna()
say(f"\n全池: 胜率{sub['win'].mean():.0%} EV{all_nr.median():.2f} → "
    f"闸门后: 胜率{on['win'].mean():.0%} EV{on_nr.median():.2f}, "
    f"信号量保留{len(on)/len(sub):.0%}")

report = "\n".join(R)
(OUT / "20_gate.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/20_gate.md")
