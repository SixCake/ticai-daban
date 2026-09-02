# -*- coding: utf-8 -*-
"""研究22: longtou(龙头二波)思想因子迁移 — 打板场景前向验证

来源: docs/research_07_龙头二波策略借鉴.md 的 H1-H6 假设。
数据: 17_enriched(决策时刻样本, 封板率/当日/次日) +
      events_enriched(2019-2026涨停事件, 长窗次日复核) +
      daily_panel(全市场日线, 因子计算)。

口径纪律:
- 决策日 T 的因子只使用 T-1 及更早数据(市场/行业统计取T-1日,
  个股特征取T-1日行并按股票shift对齐);
- 17_enriched 自带 train/test 前向切分: 分档阈值只在 train 定;
- 长窗 events: 2019-2024 定方向, 2025-2026 复核;
- 三段行情(regime列) × 时段(td≤600=10:00前) 分别复核;
- 主指标: 次日胜率; 辅: 封板率/当日胜率/盈亏比/五档单调性。

近似口径说明:
- daily_panel 无成交额字段, 量比用 vol 口径(amount_ratio5→vol_ratio5);
- 涨/跌停价=pre_close×(1±rate), rate: 30/68→20%, 其余→10%, 未做ST修正;
- 行业映射取 events 最新 industry(点时漂移忽略)。
输出: research/out/22_longtou_factors.md + 22_features.parquet
"""
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


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# ================= 加载 =================
say("# 研究22: longtou思想因子迁移（打板前向验证）")
panel = pd.read_parquet(ROOT / "data/market/1d/daily_panel.parquet")
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
f17 = pd.read_parquet(OUT / "17_enriched.parquet")
say(f"\npanel={len(panel):,}行 {panel['trade_date'].min()}~{panel['trade_date'].max()}"
    f"；events={len(ev):,}；17样本={len(f17):,}"
    f"（train={int((f17['split'] == 'train').sum())}/test={int((f17['split'] == 'test').sum())}）")

panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
# 涨跌停价近似（30/68→20%，其余10%，未做ST修正）
rate = np.where(panel["ts_code"].str[:2].isin(["30", "68"]), 0.20, 0.10)
limit_px = np.round(panel["pre_close"] * (1 + rate), 2)
low_px = np.round(panel["pre_close"] * (1 - rate), 2)
panel["sealed"] = panel["close"] >= limit_px * 0.999
panel["touched"] = panel["high"] >= limit_px * 0.999
panel["sealed_dn"] = panel["close"] <= low_px * 1.001
panel["zb"] = panel["touched"] & ~panel["sealed"]          # 触板未封=炸板
panel["cpos"] = np.where(
    panel["high"] > panel["low"],
    (panel["close"] - panel["low"]) / (panel["high"] - panel["low"]), 0.5)
panel["neg"] = panel["pct_chg"] < 0

# 行业映射（events最新口径）
ind_map = (ev.sort_values("trade_date").groupby("ts_code")["industry"]
           .last().to_dict())
panel["industry"] = panel["ts_code"].map(ind_map)

# ================= 个股特征（行日D=用到D日数据；对齐决策日再shift1） =================
say("\n计算个股特征（7.5M行，需约1分钟）…")
g = panel.groupby("ts_code", sort=False)
panel["zb_cnt20"] = g["zb"].transform(lambda s: s.rolling(20, min_periods=10).sum())
panel["zb_cnt5"] = g["zb"].transform(lambda s: s.rolling(5, min_periods=3).sum())
panel["vol_ma5p"] = g["vol"].transform(lambda s: s.rolling(5, min_periods=3).mean().shift(1))
panel["y_volr5"] = panel["vol"] / panel["vol_ma5p"]
neg_id = (~panel["neg"]).groupby(panel["ts_code"]).cumsum()
panel["neg_streak"] = np.where(
    panel["neg"], panel.groupby([panel["ts_code"], neg_id]).cumcount() + 1, 0)
panel["neg_deep"] = g["pct_chg"].transform(
    lambda s: s.rolling(3, min_periods=1).min() <= -5.0)

# 行业内相对地位（按 trade_date×industry）
gi = panel.dropna(subset=["industry"]).groupby(["trade_date", "industry"])
panel["ind_rank"] = gi["pct_chg"].rank(ascending=False, method="min")
panel["ind_gap"] = gi["pct_chg"].transform("max") - panel["pct_chg"]
panel["ind_n"] = gi["pct_chg"].transform("size")

# shift(1)：行日D的特征 → 决策日D可见（只用≤D-1数据）；保留原始pct_chg供市场/行业统计
panel["pct_raw"] = panel["pct_chg"]
shift_cols = ["zb_cnt20", "zb_cnt5", "y_volr5", "neg_streak", "neg_deep",
              "ind_rank", "ind_gap", "ind_n", "cpos", "pct_raw"]
panel[shift_cols] = g[shift_cols].shift(1)
panel = panel.rename(columns={"cpos": "y_cpos2", "pct_raw": "y_pct"})

# ================= 市场/行业日频统计（决策日T取T-1值） =================
panel["up"] = ~panel["neg"]
md = panel.groupby("trade_date").agg(
    mkt_n=("ts_code", "size"),
    adv_n=("up", "sum"),
    zt_n=("sealed", "sum"), ld_n=("sealed_dn", "sum"),
    vol_tot=("vol", "sum")).sort_index()
md["adv_ratio"] = md["adv_n"] / md["mkt_n"]
md["vol_ratio"] = md["vol_tot"] / md["vol_tot"].shift(1)
md["cycle"] = (45 * md["adv_ratio"] + 25 * clip01(md["zt_n"] / 80)
               + 15 * clip01(1 - md["ld_n"] / 20)
               + 15 * clip01(md["vol_ratio"] / 1.1))
md["ldlr"] = md["ld_n"] / md["zt_n"].clip(lower=1)

panel["ind_up"] = panel["up"].astype(float)
idat = panel.dropna(subset=["industry"]).groupby(["trade_date", "industry"]).agg(
    ind_breadth=("ind_up", "mean"),
    ind_ztdens=("sealed", "mean"), ind_med=("pct_chg", "median"))
idat["ind_score"] = (45 * idat["ind_breadth"]
                     + 25 * clip01((idat["ind_med"] + 2) / 4)
                     + 30 * clip01(idat["ind_ztdens"] / 0.08))
idat = idat.reset_index()

dates = sorted(md.index)
prev_date = dict(zip(dates[1:], dates[:-1]))
mkt_lookup = md.shift(1)  # 行日D → 用D-1统计（日历shift，停牌不影响市场级）
mkt_lookup = mkt_lookup.rename(columns={
    "zt_n": "zt_prev", "ld_n": "ld_prev", "adv_ratio": "adv_prev",
    "cycle": "cycle_prev", "ldlr": "ldlr_prev", "vol_ratio": "mvol_prev"})
mkt_lookup = mkt_lookup[["zt_prev", "ld_prev", "adv_prev", "cycle_prev",
                         "ldlr_prev", "mvol_prev"]]

# 行业统计对齐到决策日：统计日d → 映射到下一交易日D（决策日T查到T-1统计）
date_to_next = dict(zip(dates[:-1], dates[1:]))
idat_prev = idat.copy()
idat_prev["trade_date"] = idat_prev["trade_date"].map(date_to_next)
idat_prev = idat_prev.dropna(subset=["trade_date"])

# ================= 合并到 17 决策样本 =================
say("合并到17决策样本…")
f = f17.merge(
    panel[["trade_date", "ts_code", "zb_cnt20", "zb_cnt5", "y_volr5",
           "neg_streak", "neg_deep", "ind_rank", "ind_gap", "ind_n",
           "y_cpos2", "y_pct", "industry"]],
    left_on=["date", "ts_code"], right_on=["trade_date", "ts_code"], how="left")
f = f.merge(mkt_lookup, left_on="date", right_index=True, how="left")
f = f.merge(idat_prev, left_on=["date", "industry"], right_on=["trade_date", "industry"],
            how="left", suffixes=("", "_i"))
f["early"] = f["td"] <= 600
f["same_win"] = f["entry_ret"] > 0
f["next_win"] = f["next_ret"] > 0
cov = f[["ld_prev", "cycle_prev", "ind_breadth", "zb_cnt20", "ind_rank",
         "neg_streak", "y_volr5"]].notna().mean()
say("因子覆盖率: " + " ".join(f"{k}={v:.0%}" for k, v in cov.items()))
f.to_parquet(OUT / "22_features.parquet")

FACTORS = [
    ("ld_prev", "H1 昨日跌停家数", [-0.5, 2.5, 5.5, 10.5, 19.5, 999]),
    ("ldlr_prev", "H1 昨日跌停/涨停比", [-0.01, 0.1, 0.25, 0.5, 1.0, 99]),
    ("adv_prev", "H1 昨日上涨家数占比", [-0.01, 0.35, 0.45, 0.55, 0.65, 1.01]),
    ("cycle_prev", "H1 昨日市场周期分", [-1, 38, 50, 65, 101]),
    ("ind_breadth", "H2 昨日行业上涨广度", [-0.01, 0.25, 0.5, 0.65, 1.01]),
    ("ind_ztdens", "H2 昨日行业涨停密度", [-0.001, 0.01, 0.03, 0.06, 1.0]),
    ("ind_score", "H2 昨日行业结构分", [-1, 35, 50, 72, 101]),
    ("y_volr5", "H3 昨日量比(vol/5日)", [-0.01, 0.55, 1.2, 2.2, 2.5, 99]),
    ("y_cpos2", "H3 昨日收盘位置", [-0.01, 0.3, 0.55, 0.8, 1.01]),
    ("zb_cnt20", "H4 近20日炸板次数", [-0.5, 0.5, 1.5, 2.5, 99]),
    ("zb_cnt5", "H4 近5日炸板次数", [-0.5, 0.5, 1.5, 99]),
    ("ind_rank", "H5 昨日行业内涨幅排名", [0.5, 3.5, 10.5, 30.5, 9999]),
    ("ind_gap", "H5 与行业第一涨幅差", [-0.01, 0.01, 2.0, 5.0, 99]),
    ("neg_streak", "H6 连续收跌天数", [-0.5, 0.5, 1.5, 2.5, 99]),
]


def metrics(s):
    nr = s["next_ret"].dropna()
    pos, negv = nr[nr > 0], nr[nr <= 0]
    plr = (pos.mean() / abs(negv.mean())) if len(negv) > 20 else np.nan
    return dict(n=len(s), seal=s["y"].mean(), same=s["same_win"].mean(),
                nwin=s["next_win"].mean(), nmed=nr.median(), plr=plr)


def fmt_row(label, m):
    return (f"| {label} | {m['n']} | {m['seal']:.1%} | {m['same']:.1%} "
            f"| {m['nwin']:.1%} | {m['nmed']:+.2f} | "
            f"{m['plr']:.2f} |" if np.isfinite(m.get('plr', np.nan)) else
            f"| {label} | {m['n']} | {m['seal']:.1%} | {m['same']:.1%} "
            f"| {m['nwin']:.1%} | {m['nmed']:+.2f} | - |")


say("\n## A. 17决策样本：分档单调性（train定档，test复核）")
for col, name, bins in FACTORS:
    sub = f.dropna(subset=[col, "next_ret"])
    if len(sub) < 2000:
        say(f"\n### {name}（{col}）— 样本不足，跳过")
        continue
    labels = [f"({bins[i]:g},{bins[i+1]:g}]" for i in range(len(bins) - 1)]
    say(f"\n### {name}（{col}）")
    say("| 档 | split | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
    say("|---|---|---|---|---|---|---|---|")
    for split in ["train", "test"]:
        s2 = sub[sub["split"] == split]
        cut = pd.cut(s2[col], bins=bins, labels=labels)
        for lab in labels:
            part = s2[cut == lab]
            if len(part) >= 50:
                say(fmt_row(f"{lab} {split}", metrics(part)))

say("\n## B. 三段行情复核（test段，次日胜率/封板率）")
KEY_GATES = [
    ("ld_prev", lambda x: x >= 10.5, "ld_prev≥11（跌停多）"),
    ("ldlr_prev", lambda x: x >= 0.5, "ldlr≥0.5（跌停≥涨停半数）"),
    ("cycle_prev", lambda x: x < 38, "cycle<38（退潮）"),
    ("ind_breadth", lambda x: x < 0.25, "行业广度<25%（退潮）"),
    ("ind_breadth", lambda x: x >= 0.65, "行业广度≥65%（主升）"),
    ("zb_cnt20", lambda x: x >= 2.5, "近20日炸板≥3次"),
    ("ind_rank", lambda x: x <= 3.5, "行业内排名前3"),
    ("neg_streak", lambda x: x >= 2.5, "连续收跌≥3日"),
    ("y_volr5", lambda x: (x >= 2.5), "昨日爆量≥2.5"),
]
tt = f[f["split"] == "test"]
say("| 闸门 | 行情段 | 侧 | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|---|---|")
for col, fn, label in KEY_GATES:
    for rg in ["偏多", "震荡", "偏空"]:
        s2 = tt[tt["regime"] == rg].dropna(subset=[col, "next_ret"])
        if len(s2) < 300:
            continue
        mask = fn(s2[col])
        for side, part in [("命中", s2[mask]), ("未中", s2[~mask])]:
            if len(part) >= 50:
                m = metrics(part)
                say(f"| {label} | {rg} | {side} | {m['n']} | {m['seal']:.1%} "
                    f"| {m['same']:.1%} | {m['nwin']:.1%} | {m['nmed']:+.2f} "
                    f"| {m['plr']:.2f} |")

say("\n## C. 时段效应（test段，10:00前/后）")
say("| 闸门 | 时段 | n | 封板率 | 次日胜率 | 次日中位% |")
say("|---|---|---|---|---|---|")
for col, fn, label in KEY_GATES:
    for early, tlab in [(True, "≤10:00"), (False, ">10:00")]:
        s2 = tt[(tt["early"] == early)].dropna(subset=[col, "next_ret"])
        mask = fn(s2[col])
        part = s2[mask]
        if len(part) >= 50:
            m = metrics(part)
            say(f"| {label} | {tlab} | {m['n']} | {m['seal']:.1%} "
                f"| {m['nwin']:.1%} | {m['nmed']:+.2f} |")

say("\n## D. H1增量检验：ld_prev × zt_prev（research20已证zt_prev≤30有效）")
say("（test段，次日胜率；zt_prev>30即情绪不冰的常规日）")
say("| ld_prev | zt_prev≤30 | zt_prev>30 |")
say("|---|---|---|")
for lo, hi, lab in [(-0.5, 2.5, "≤2"), (2.5, 5.5, "3-5"), (5.5, 10.5, "6-10"),
                    (10.5, 999, "≥11")]:
    cells = []
    for zlo, zhi in [(-1, 30.5), (30.5, 999)]:
        part = tt[(tt["ld_prev"] > lo) & (tt["ld_prev"] <= hi)
                  & (tt["zt_prev"] > zlo) & (tt["zt_prev"] <= zhi)]
        cells.append(f"{part['next_win'].mean():.1%}(n={len(part)})"
                     if len(part) >= 30 else "-")
    say(f"| {lab} | {cells[0]} | {cells[1]} |")

# ================= 长窗 events 复核（2019-2026） =================
say("\n## E. 长窗复核：涨停事件（非一字）次日胜率，2019-2024定方向/2025-2026验证")
evp = ev[ev["is_yizi"] == False].drop(columns=["industry"]).merge(  # noqa: E712
    panel[["trade_date", "ts_code", "zb_cnt20", "neg_streak", "ind_rank",
           "ind_gap", "y_volr5", "y_cpos2", "industry"]],
    on=["trade_date", "ts_code"], how="left")
evp = evp.merge(mkt_lookup, left_on="trade_date", right_index=True, how="left")
evp = evp.merge(idat_prev, left_on=["trade_date", "industry"],
                right_on=["trade_date", "industry"], how="left", suffixes=("", "_i"))
evp["win"] = evp["next_close_ret"] > 0
evp["period"] = np.where(evp["trade_date"] >= "20250101", "OOS25-26", "IS19-24")
evp.to_parquet(OUT / "22_events_features.parquet")
say(f"非一字涨停事件 {len(evp):,} 条，因子覆盖 "
    f"ld={evp['ld_prev'].notna().mean():.0%} zb={evp['zb_cnt20'].notna().mean():.0%}")

EV_GATES = [
    ("ld_prev", lambda x: x >= 10.5, "ld_prev≥11"),
    ("ldlr_prev", lambda x: x >= 0.5, "ldlr≥0.5"),
    ("cycle_prev", lambda x: x < 38, "cycle<38"),
    ("ind_breadth", lambda x: x < 0.25, "行业广度<25%"),
    ("ind_breadth", lambda x: x >= 0.65, "行业广度≥65%"),
    ("zb_cnt20", lambda x: x >= 2.5, "炸板≥3次"),
    ("zb_cnt20", lambda x: x <= 0.5, "无炸板疤痕"),
    ("ind_rank", lambda x: x <= 3.5, "行业内前3"),
    ("neg_streak", lambda x: x >= 2.5, "连跌≥3日"),
]
say("| 闸门 | 期间 | 侧 | n | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|")
for col, fn, label in EV_GATES:
    for per in ["IS19-24", "OOS25-26"]:
        s2 = evp[evp["period"] == per].dropna(subset=[col, "next_close_ret"])
        mask = fn(s2[col])
        for side, part in [("命中", s2[mask]), ("未中", s2[~mask])]:
            if len(part) >= 200:
                nr = part["next_close_ret"]
                pos, negv = nr[nr > 0], nr[nr <= 0]
                plr = pos.mean() / abs(negv.mean()) if len(negv) > 50 else np.nan
                say(f"| {label} | {per} | {side} | {len(part)} "
                    f"| {part['win'].mean():.1%} | {nr.median():+.2f} "
                    f"| {plr:.2f} |")

say("\n## F. 长窗五档单调性（OOS 2025-26，次日胜率）")
for col, name, bins in [("ld_prev", "昨日跌停家数", [-0.5, 2.5, 5.5, 10.5, 19.5, 999]),
                        ("zb_cnt20", "近20日炸板", [-0.5, 0.5, 1.5, 2.5, 99]),
                        ("ind_rank", "行业内排名", [0.5, 3.5, 10.5, 30.5, 9999]),
                        ("cycle_prev", "市场周期分", [-1, 38, 50, 65, 101])]:
    s2 = evp[evp["period"] == "OOS25-26"].dropna(subset=[col, "next_close_ret"])
    labels = [f"({bins[i]:g},{bins[i+1]:g}]" for i in range(len(bins) - 1)]
    cut = pd.cut(s2[col], bins=bins, labels=labels)
    row = " / ".join(
        f"{s2[cut == lab]['win'].mean():.1%}({int((cut == lab).sum())})"
        if (cut == lab).sum() >= 50 else "-" for lab in labels)
    say(f"- **{name}**: {row}")

(OUT / "22_longtou_factors.md").write_text("\n".join(R), encoding="utf-8")
say("\n报告已写入 research/out/22_longtou_factors.md")
