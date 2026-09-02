# -*- coding: utf-8 -*-
"""研究23: longtou因子组合闸门与复合评分 — 应用前后对比回测

研究22单因子验证后，本脚本回答「应用后能提高多少」：
1. 组合闸门：基线 vs 关闸（research20 zt_prev≤30 ∪ 新增 ldlr_prev≥0.5），
   在主力规则池上对比封板率/当日胜率/次日胜率/EV/盈亏比；
2. 次日质量复合分 qscore（4项研究22验证方向）：分档单调性 train→test 前向；
3. 封板概率复合分 sscore（聚焦度类特征）：分档封板率单调性；
4. 长窗 events IS/OOS 复核 qscore 次日胜率与盈亏比；
5. 时段效应复核组合闸门。

数据: research/out/22_features.parquet + 22_events_features.parquet
输出: research/out/23_combo_gate.md
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


f = pd.read_parquet(OUT / "22_features.parquet")
evp = pd.read_parquet(OUT / "22_events_features.parquet")
f["same_win"] = f["entry_ret"] > 0
f["next_win"] = f["next_ret"] > 0
evp["win"] = evp["next_close_ret"] > 0

say("# 研究23: longtou因子组合闸门与复合评分（应用前后对比）")
say(f"\n17样本={len(f):,}（train={int((f['split'] == 'train').sum())}/"
    f"test={int((f['split'] == 'test').sum())}）；长窗事件={len(evp):,}")

# ---------- 主力规则池（沿用 research/20 定义） ----------
RULES = {
    "竞价量爆(非一字)": (f["cohort"] == "G") & (f["open_vr"] > 5)
        & (f["cm20"] == 0) & (f["gap"] <= 5.2),
    "高开剧震": (f["cohort"] == "G") & (f["gap"] <= 5.2)
        & (f["amp3"] > 4.3) & (f["cm20"] == 0),
    "接力×稳封相": (f["y_zt"] == 1) & (f["cohort"] == "G")
        & (f["odip"] <= 0.05),
}
pool = pd.Series(False, index=f.index)
for cond in RULES.values():
    pool |= cond
f["pool"] = pool


def metrics(s):
    nr = s["next_ret"].dropna()
    pos, negv = nr[nr > 0], nr[nr <= 0]
    plr = pos.mean() / abs(negv.mean()) if len(negv) > 20 else np.nan
    return (len(s), s["y"].mean(), s["same_win"].mean(), s["next_win"].mean(),
            nr.median(), plr)


def mrow(label, s):
    n, seal, same, nwin, nmed, plr = metrics(s)
    plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
    return (f"| {label} | {n} | {seal:.1%} | {same:.1%} | {nwin:.1%} "
            f"| {nmed:+.2f} | {plr_s} | {s['date'].nunique()} |")


# ---------- 1. 组合闸门前后对比 ----------
gate_old = f["zt_prev"] <= 30                       # research20 已验证
gate_new = f["ldlr_prev"] >= 0.5                    # 研究22新增
gate_all = gate_old | gate_new

say("\n## 1. 组合闸门：主力规则池 基线 vs 关闸")
say("关闸定义: zt_prev≤30（research20）∪ ldlr_prev≥0.5（研究22新增）")
say("\n| 状态 | split | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 | 覆盖天 |")
say("|---|---|---|---|---|---|---|---|---|")
for split in ["train", "test"]:
    sub = f[f["pool"] & (f["split"] == split)]
    on, off = sub[~gate_all[sub.index]], sub[gate_all[sub.index]]
    say(mrow(f"基线(不关闸) {split}", sub))
    say(mrow(f"开闸(保留) {split}", on))
    say(mrow(f"关闸(剔除) {split}", off))

say("\n### 增量：仅新增 ldlr 闸门（在 research20 闸门已开的基础上）")
say("\n| 状态 | split | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 | 覆盖天 |")
say("|---|---|---|---|---|---|---|---|---|")
for split in ["train", "test"]:
    sub = f[f["pool"] & (f["split"] == split)]
    kept_old = sub[~gate_old[sub.index]]            # 老闸门后剩余
    on2 = kept_old[~gate_new[kept_old.index]]       # 再关新闸
    off2 = kept_old[gate_new[kept_old.index]]
    say(mrow(f"老闸门后剩余 {split}", kept_old))
    say(mrow(f"+新闸保留 {split}", on2))
    say(mrow(f"+新闸剔除 {split}", off2))

say("\n### 组合闸门 × 三段行情（test段）")
say("\n| 行情段 | 状态 | n | 封板率 | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|")
for rg in ["偏多", "震荡", "偏空"]:
    sub = f[f["pool"] & (f["split"] == "test") & (f["regime"] == rg)]
    on, off = sub[~gate_all[sub.index]], sub[gate_all[sub.index]]
    for lab, s in [("开闸", on), ("关闸", off)]:
        n, seal, same, nwin, nmed, plr = metrics(s)
        plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
        say(f"| {rg} | {lab} | {n} | {seal:.1%} | {nwin:.1%} | {nmed:+.2f} "
            f"| {plr_s} |")

say("\n### 组合闸门 × 时段（test段）")
say("\n| 时段 | 状态 | n | 封板率 | 次日胜率 |")
say("|---|---|---|---|---|")
for early, tlab in [(True, "≤10:00"), (False, ">10:00")]:
    sub = f[f["pool"] & (f["split"] == "test") & (f["early"] == early)]
    for lab, part in [("开闸", sub[~gate_all[sub.index]]),
                      ("关闸", sub[gate_all[sub.index]])]:
        n, seal, same, nwin, nmed, plr = metrics(part)
        say(f"| {tlab} | {lab} | {n} | {seal:.1%} | {nwin:.1%} |")

# ---------- 2. 次日质量复合分 qscore ----------
f["q_ldlr"] = (f["ldlr_prev"] < 0.5).astype(int)
f["q_rank"] = (f["ind_rank"] > 3.5).astype(int)
f["q_zb"] = (f["zb_cnt20"] <= 1.5).astype(int)
f["q_vol"] = ((f["y_volr5"] > 0.55) & (f["y_volr5"] <= 2.2)).astype(int)
f["qscore"] = f["q_ldlr"] + f["q_rank"] + f["q_zb"] + f["q_vol"]

say("\n## 2. 次日质量复合分 qscore（ldlr<0.5 + 行业排名>3 + 炸板≤1 + 量比甜蜜区）")
say("\n| qscore | split | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|---|")
for split in ["train", "test"]:
    sub = f[(f["split"] == split)].dropna(subset=["next_ret", "ldlr_prev"])
    for sc in sorted(sub["qscore"].unique()):
        part = sub[sub["qscore"] == sc]
        if len(part) >= 50:
            n, seal, same, nwin, nmed, plr = metrics(part)
            plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
            say(f"| {int(sc)} | {split} | {n} | {seal:.1%} | {same:.1%} "
                f"| {nwin:.1%} | {nmed:+.2f} | {plr_s} |")

say("\n### qscore 顶档(4) vs 底档(≤1)（主力规则池，test段，三段行情）")
say("\n| 行情段 | 档 | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|---|")
for rg in ["偏多", "震荡", "偏空"]:
    sub = f[f["pool"] & (f["split"] == "test") & (f["regime"] == rg)].dropna(
        subset=["next_ret", "ldlr_prev"])
    for lab, cond in [("qscore=4", sub["qscore"] == 4),
                      ("qscore≤1", sub["qscore"] <= 1)]:
        part = sub[cond]
        if len(part) >= 30:
            n, seal, same, nwin, nmed, plr = metrics(part)
            plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
            say(f"| {rg} | {lab} | {n} | {seal:.1%} | {same:.1%} | {nwin:.1%} "
                f"| {nmed:+.2f} | {plr_s} |")

# ---------- 3. 封板概率复合分 sscore ----------
f["s_zb"] = (f["zb_cnt20"] >= 0.5).astype(int)
f["s_ztd"] = (f["ind_ztdens"] >= 0.03).astype(int)
f["s_rank"] = (f["ind_rank"] <= 3.5).astype(int)
f["s_vol"] = (f["y_volr5"] < 2.5).astype(int)
f["s_brd"] = (f["ind_breadth"] >= 0.65).astype(int)
f["sscore"] = f["s_zb"] + f["s_ztd"] + f["s_rank"] + f["s_vol"] + f["s_brd"]

say("\n## 3. 封板概率复合分 sscore（炸板疤痕+行业涨停密度+行业排名前3+非爆量+高广度）")
say("\n| sscore | split | n | 封板率 | 次日胜率 | 盈亏比 |")
say("|---|---|---|---|---|---|")
for split in ["train", "test"]:
    sub = f[(f["split"] == split)].dropna(subset=["sscore"])
    sub = sub[sub[["s_zb", "s_ztd", "s_rank", "s_vol", "s_brd"]].notna().all(axis=1)]
    for sc in sorted(sub["sscore"].unique()):
        part = sub[sub["sscore"] == sc]
        if len(part) >= 100:
            n, seal, same, nwin, nmed, plr = metrics(part)
            plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
            say(f"| {int(sc)} | {split} | {n} | {seal:.1%} | {nwin:.1%} "
                f"| {plr_s} |")

# ---------- 4. 长窗 events 复核 qscore ----------
evp["q_ldlr"] = (evp["ldlr_prev"] < 0.5).astype(int)
evp["q_rank"] = (evp["ind_rank"] > 3.5).astype(int)
evp["q_zb"] = (evp["zb_cnt20"] <= 1.5).astype(int)
evp["q_vol"] = ((evp["y_volr5"] > 0.55) & (evp["y_volr5"] <= 2.2)).astype(int)
evp["qscore"] = evp["q_ldlr"] + evp["q_rank"] + evp["q_zb"] + evp["q_vol"]

say("\n## 4. 长窗复核：qscore × 次日胜率/盈亏比（非一字涨停事件）")
say("\n| qscore | 期间 | n | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|")
for per in ["IS19-24", "OOS25-26"]:
    sub = evp[(evp["period"] == per)].dropna(
        subset=["next_close_ret", "ldlr_prev", "ind_rank", "zb_cnt20", "y_volr5"])
    for sc in sorted(sub["qscore"].unique()):
        part = sub[sub["qscore"] == sc]
        if len(part) >= 200:
            nr = part["next_close_ret"]
            pos, negv = nr[nr > 0], nr[nr <= 0]
            plr = pos.mean() / abs(negv.mean()) if len(negv) > 50 else np.nan
            say(f"| {int(sc)} | {per} | {len(part)} | {part['win'].mean():.1%} "
                f"| {nr.median():+.2f} | {plr:.2f} |")

(OUT / "23_combo_gate.md").write_text("\n".join(R), encoding="utf-8")
say("\n报告已写入 research/out/23_combo_gate.md")
