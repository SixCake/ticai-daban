# -*- coding: utf-8 -*-
"""研究24b: 最终形态确认 — 两层架构候选方案并行回测

研究24消融给出两层架构结论，但组合级只测了 V4(结构过滤+盘中排序)。
本脚本并行回测全部候选最终形态，含月度稳定性（walk-forward 时序复核）：
  V0 随机基线（同池）
  V1 仅盘中（现有方案近似）
  V3 三组加权融合
  V4 g_chip≥0.6 过滤 + 盘中排序
  V5 g_chip≥0.6 过滤 + 融合分排序（推荐形态候选）
  V6 g_chip≥0.6 过滤 + 纯结构排序（结构选股极端）
口径: train 拟合分位/阈值，test 只应用；top3 分仓、30% 成交、次日收盘离场、MC200。
输出: 追加到 research/out/24_fusion_ablation.md §8
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


df = pd.read_parquet(OUT / "22_features.parquet")
df["same_win"] = df["entry_ret"] > 0
df["next_win"] = df["next_ret"] > 0

SCORE_COLS = ["r3", "pathvol", "ind_ztdens", "ind_rank", "ind_breadth",
              "zb_cnt20", "y_volr5", "ldlr_prev", "neg_streak"]
sub = df.dropna(subset=SCORE_COLS + ["next_ret"]).copy()
tr = sub[sub["split"] == "train"]


def pct_fit(col, invert=False):
    grid = np.sort(tr[col].values)

    def apply_(v):
        p = np.searchsorted(grid, v, side="right") / len(grid)
        return 1.0 - p if invert else p

    return apply_


p_r3 = pct_fit("r3")
p_pv = pct_fit("pathvol")
p_ztd = pct_fit("ind_ztdens")
p_rank = pct_fit("ind_rank", invert=True)
p_brd = pct_fit("ind_breadth")
p_zb = pct_fit("zb_cnt20")

sub["g_intra"] = (p_r3(sub["r3"].values) + p_pv(sub["pathvol"].values)) / 2
sub["g_eco"] = (p_ztd(sub["ind_ztdens"].values) + p_rank(sub["ind_rank"].values)
                + p_brd(sub["ind_breadth"].values) + p_zb(sub["zb_cnt20"].values)
                + ((sub["y_volr5"] > 0.55) & (sub["y_volr5"] <= 2.2)).values) / 5
sub["g_chip"] = (((sub["ldlr_prev"] < 0.5).astype(float)
                  + (sub["ind_rank"] > 3.5).astype(float)
                  + (sub["zb_cnt20"] <= 1.5).astype(float)
                  + ((sub["y_volr5"] > 0.55) & (sub["y_volr5"] <= 2.2)).astype(float)
                  + (sub["neg_streak"] >= 2.5).astype(float)) / 5)
sub["V1"] = sub["g_intra"]
sub["V3"] = (0.5 * sub["g_intra"] + 0.25 * sub["g_eco"] + 0.25 * sub["g_chip"])
CHIP_OK = sub["g_chip"] >= 0.6          # 结构层阈值（train 分布 40 分位附近）
say(f"# 研究24b: 最终形态确认（两层架构候选并行回测）")
say(f"\n评分样本={len(sub):,}；g_chip≥0.6 覆盖: train="
    f"{CHIP_OK[sub['split'] == 'train'].mean():.0%} / test="
    f"{CHIP_OK[sub['split'] == 'test'].mean():.0%}")


def metrics(s):
    nr = s["next_ret"].dropna()
    pos, negv = nr[nr > 0], nr[nr <= 0]
    plr = pos.mean() / abs(negv.mean()) if len(negv) > 10 else np.nan
    return (len(s), s["y"].mean(), s["same_win"].mean(), s["next_win"].mean(),
            nr.median(), plr)


N_RUNS = 200
TOPK = 3
rng = np.random.default_rng(42)

PLANS = {
    "V0 随机基线": (sub, None, True),
    "V1 仅盘中": (sub, "V1", False),
    "V3 加权融合": (sub, "V3", False),
    "V4 结构过滤+盘中排序": (sub[CHIP_OK], "g_intra", False),
    "V5 结构过滤+融合排序": (sub[CHIP_OK], "V3", False),
    "V6 结构过滤+结构排序": (sub[CHIP_OK], "g_chip", False),
}


def simulate(frame, score_col, split, random_pick=False):
    s2 = frame[frame["split"] == split]
    days = sorted(s2["date"].unique())
    runs = []
    for run in range(N_RUNS):
        daily = []
        for d in days:
            day = s2[s2["date"] == d]
            if len(day) == 0:
                continue
            if random_pick:
                k = day.sample(n=min(TOPK, len(day)),
                               random_state=run * 7919 + int(d) % 10**6)
            else:
                k = day.nlargest(min(TOPK, len(day)), score_col)
            fills = rng.random(len(k)) < 0.30
            rets = k["next_ret"].values[fills] / 100.0
            daily.append(rets.sum() / TOPK)
        runs.append(np.array(daily))
    return runs


def sharpe_of(r):
    if len(r) < 20 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(244)


say("\n## 8. 最终形态确认：候选方案并行回测（top3/日，30%成交，MC200）")
say("\n| 方案 | split | 选中样本 | 封板率 | 当日胜率 | 买入点胜率 | 次日胜率 "
    "| 盈亏比 | Sharpe(中位) | 年化(中位) | 日胜率 |")
say("|---|---|---|---|---|---|---|---|---|---|---|")
best = {}
for name, (frame, score_col, random_pick) in PLANS.items():
    for split in ["train", "test"]:
        if random_pick:
            sel = frame[frame["split"] == split]
        else:
            # 组合级=每日top3；样本级指标用每日top3的并集近似选中池
            sel = pd.concat([g.nlargest(min(TOPK, len(g)), score_col)
                             for _, g in frame[frame["split"] == split]
                             .groupby("date")])
        n, seal, same, nwin, nmed, plr = metrics(sel)
        bpw = ((sel["y"] == 1) & sel["same_win"]).mean()
        runs = simulate(frame, score_col, split, random_pick)
        sh = [sharpe_of(r) for r in runs]
        sh = [x for x in sh if np.isfinite(x)]
        ar = [r.mean() * 244 for r in runs if len(r) >= 20]
        wr = [(r > 0).mean() for r in runs if len(r) >= 20]
        plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
        say(f"| {name} | {split} | {n} | {seal:.1%} | {same:.1%} | {bpw:.1%} "
            f"| {nwin:.1%} | {plr_s} | {np.median(sh):.2f} "
            f"| {np.median(ar):.1%} | {np.median(wr):.1%} |")
        if split == "test":
            best[name] = (np.median(sh), nwin, plr, seal)

# ---------- 月度稳定性（test段，推荐形态 V5 vs V1/V3） ----------
say("\n## 9. 月度稳定性（test段每日top3选中样本：次日胜率 / 盈亏比）")
say("\n| 月份 | V1次日胜率 | V3次日胜率 | V5次日胜率 | V5盈亏比 | V5样本 |")
say("|---|---|---|---|---|---|")
chip_ok = sub[CHIP_OK & (sub["split"] == "test")]
v5_sel = pd.concat([g.nlargest(min(TOPK, len(g)), "V3")
                    for _, g in chip_ok.groupby("date")])
v1_all = sub[sub["split"] == "test"]
v1_sel = pd.concat([g.nlargest(min(TOPK, len(g)), "V1")
                    for _, g in v1_all.groupby("date")])
v3_sel = pd.concat([g.nlargest(min(TOPK, len(g)), "V3")
                    for _, g in v1_all.groupby("date")])
v1_sel["ym"] = v1_sel["date"].str[:6]
v3_sel["ym"] = v3_sel["date"].str[:6]
v5_sel["ym"] = v5_sel["date"].str[:6]
for ym in sorted(v5_sel["ym"].unique()):
    a, b = v1_sel[v1_sel["ym"] == ym], v3_sel[v3_sel["ym"] == ym]
    c = v5_sel[v5_sel["ym"] == ym]
    nr = c["next_ret"].dropna()
    pos, negv = nr[nr > 0], nr[nr <= 0]
    plr = pos.mean() / abs(negv.mean()) if len(negv) > 5 else np.nan
    plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
    say(f"| {ym} | {a['next_win'].mean():.0%} | {b['next_win'].mean():.0%} "
        f"| {c['next_win'].mean():.0%} | {plr_s} | {len(c)} |")
win_months = sum(
    1 for ym in sorted(v5_sel["ym"].unique())
    if v5_sel[v5_sel["ym"] == ym]["next_win"].mean()
    > v1_sel[v1_sel["ym"] == ym]["next_win"].mean())
n_months = v5_sel["ym"].nunique()
say(f"\nV5 月度跑赢 V1 的月份: {win_months}/{n_months}")

with open(OUT / "24_fusion_ablation.md", "a", encoding="utf-8") as fh:
    fh.write("\n" + "\n".join(R) + "\n")
say("\n已追加到 research/out/24_fusion_ablation.md")
