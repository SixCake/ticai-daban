# -*- coding: utf-8 -*-
"""研究24: 因子融合与消融 — 买入点胜率/次日胜率/盈亏比/Sharpe

沿用假设驱动闭环：假设 → 论证 → 融合 → 论证 → 消融 → 论证。

假设（基于研究17/22/23已验证方向）:
  H1 盘中动量组(r3暴拉, pathvol轨迹波动)与结构组(板块生态/筹码环境)信息正交，
     rank融合后 top 档买入点胜率(封板率×当日胜率)与次日胜率同时提升；
  H2 融合分 V3(盘中0.5+生态0.25+筹码0.25) 的组合级 Sharpe 高于
     现有方案近似 V1(仅盘中) 与无评分基线 V0；
  H3 两层方案 V4(筹码硬过滤+盘中排序) 在次日胜率/盈亏比上优于单层 V3，
     代价是覆盖下降；
  H4 消融：任一组移除后 top 档指标下降（各组边际贡献为正）。

口径纪律:
- rank 分位只在 train 拟合，test 只应用（前向）；
- 组合模拟: 每日按分选 top3，30% 买入成交概率（用户指定约束），
  入场=entry 价、出场=次日收盘(next_ret)，未成交=现金0；
  Monte Carlo 200 次，报告中位 Sharpe/年化/日胜率；
- 三段行情 × 10:00 时段复核。
数据: research/out/22_features.parquet（含17全部盘中因子+22新因子）
输出: research/out/24_fusion_ablation.md
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

say("# 研究24: 因子融合与消融（买入点胜率 → 次日胜率 → 盈亏比/Sharpe）")
say(f"\n样本={len(df):,}（train={int((df['split'] == 'train').sum())}/"
    f"test={int((df['split'] == 'test').sum())}）")

# ================= 因子分组与 rank 分位（train 拟合） =================
SCORE_COLS = ["r3", "pathvol", "ind_ztdens", "ind_rank", "ind_breadth",
              "zb_cnt20", "y_volr5", "ldlr_prev", "neg_streak"]
sub = df.dropna(subset=SCORE_COLS + ["next_ret"]).copy()
say(f"评分可用样本（因子全覆盖）: {len(sub):,} "
    f"(train={int((sub['split'] == 'train').sum())}/"
    f"test={int((sub['split'] == 'test').sum())})")

tr = sub[sub["split"] == "train"]


def pct_fit(col, invert=False):
    """train 分位拟合：值 → [0,1] 分位（searchsorted 前向应用）"""
    grid = np.sort(tr[col].values)

    def apply_(v):
        p = np.searchsorted(grid, v, side="right") / len(grid)
        return 1.0 - p if invert else p

    return apply_


p_r3 = pct_fit("r3")                     # 暴拉：越大越好
p_pv = pct_fit("pathvol")                # 轨迹波动：越大越好
p_ztd = pct_fit("ind_ztdens")            # 行业涨停密度：封板正向
p_rank = pct_fit("ind_rank", invert=True)  # 行业排名靠前：封板正向
p_brd = pct_fit("ind_breadth")           # 行业广度：封板正向
p_zb = pct_fit("zb_cnt20")               # 炸板疤痕：封板正向(次日负向,见H3)

sub["g_intra"] = (p_r3(sub["r3"].values) + p_pv(sub["pathvol"].values)) / 2
sub["g_eco"] = (p_ztd(sub["ind_ztdens"].values) + p_rank(sub["ind_rank"].values)
                + p_brd(sub["ind_breadth"].values) + p_zb(sub["zb_cnt20"].values)
                + ((sub["y_volr5"] > 0.55) & (sub["y_volr5"] <= 2.2)).values) / 5
sub["g_chip"] = (((sub["ldlr_prev"] < 0.5).astype(float)
                  + (sub["ind_rank"] > 3.5).astype(float)
                  + (sub["zb_cnt20"] <= 1.5).astype(float)
                  + ((sub["y_volr5"] > 0.55) & (sub["y_volr5"] <= 2.2)).astype(float)
                  + (sub["neg_streak"] >= 2.5).astype(float)) / 5)

# 融合变体（多方案并行回测）
sub["V1"] = sub["g_intra"]                                   # 现有方案近似: 仅盘中
sub["V2"] = 0.6 * sub["g_intra"] + 0.4 * sub["g_eco"]        # +生态
sub["V3"] = (0.5 * sub["g_intra"] + 0.25 * sub["g_eco"]
             + 0.25 * sub["g_chip"])                          # 三组融合
sub["V4"] = sub["V1"].where(sub["g_chip"] >= 0.6)             # 两层: 筹码过滤+盘中排序

VARIANTS = ["V1", "V2", "V3"]


def metrics(s):
    nr = s["next_ret"].dropna()
    pos, negv = nr[nr > 0], nr[nr <= 0]
    plr = pos.mean() / abs(negv.mean()) if len(negv) > 20 else np.nan
    return (len(s), s["y"].mean(), s["same_win"].mean(), s["next_win"].mean(),
            nr.median(), plr)


# ================= 论证1: 分组间正交性（H1前提） =================
say("\n## 1. 论证：三组分数相关性（train，Spearman近似=Pearson于rank）")
corr = sub[sub["split"] == "train"][["g_intra", "g_eco", "g_chip"]].corr()
say("\n| | g_intra | g_eco | g_chip |")
say("|---|---|---|---|")
for a in ["g_intra", "g_eco", "g_chip"]:
    say(f"| {a} | {corr.loc[a, 'g_intra']:.2f} | {corr.loc[a, 'g_eco']:.2f} "
        f"| {corr.loc[a, 'g_chip']:.2f} |")

# ================= 融合: 分档单调性（train定档 test复核） =================
say("\n## 2. 融合：各变体五分位单调性（分位档在 train 定）")
for var in VARIANTS:
    grid = np.quantile(tr[var].values if var in tr else
                       sub[sub["split"] == "train"][var].values,
                       [0.2, 0.4, 0.6, 0.8])
    say(f"\n### {var}")
    say("| 档 | split | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
    say("|---|---|---|---|---|---|---|---|")
    bins = [-np.inf, *grid, np.inf]
    labs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for split in ["train", "test"]:
        s2 = sub[sub["split"] == split]
        cut = pd.cut(s2[var], bins=bins, labels=labs)
        for lab in labs:
            part = s2[cut == lab]
            if len(part) >= 50:
                n, seal, same, nwin, nmed, plr = metrics(part)
                plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
                say(f"| {lab} | {split} | {n} | {seal:.1%} | {same:.1%} "
                    f"| {nwin:.1%} | {nmed:+.2f} | {plr_s} |")

# ================= 论证2: top10% 选股口径 =================
say("\n## 3. 论证：top10% 精选口径（test段）")
say("买入点胜率 = 封板率×当日胜率近似（封住且收盘不低于入场价的概率）")
say("\n| 变体 | n | 封板率 | 当日胜率 | 买入点胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|---|")
for var in VARIANTS:
    thr = np.quantile(sub[sub["split"] == "train"][var].values, 0.90)
    part = sub[(sub["split"] == "test") & (sub[var] >= thr)]
    n, seal, same, nwin, nmed, plr = metrics(part)
    bpw = ((part["y"] == 1) & part["same_win"]).mean()
    plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
    say(f"| {var} top10% | {n} | {seal:.1%} | {same:.1%} | {bpw:.1%} "
        f"| {nwin:.1%} | {nmed:+.2f} | {plr_s} |")
# V4 两层
v4 = sub[(sub["split"] == "test") & (sub["g_chip"] >= 0.6)]
thr1 = np.quantile(sub[(sub["split"] == "train") & (sub["g_chip"] >= 0.6)]
                   ["g_intra"].values, 0.90)
part = v4[v4["g_intra"] >= thr1]
n, seal, same, nwin, nmed, plr = metrics(part)
bpw = ((part["y"] == 1) & part["same_win"]).mean()
plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
say(f"| V4 两层top10% | {n} | {seal:.1%} | {same:.1%} | {bpw:.1%} "
    f"| {nwin:.1%} | {nmed:+.2f} | {plr_s} |")

# ================= 组合模拟: Sharpe（30%成交 MC） =================
N_RUNS = 200
TOPK = 3
rng = np.random.default_rng(42)


def simulate(score_frame, score_col, split, random_pick=False):
    """每日按分选 topK，30%成交，次日收盘离场；返回每次MC的日收益序列列表"""
    s2 = score_frame[score_frame["split"] == split].copy()
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
            daily.append(rets.sum() / TOPK)   # 预算3等分，未成交=现金
        runs.append(np.array(daily))
    return runs


def sharpe_stats(runs):
    sh, ar, wr = [], [], []
    for r in runs:
        if len(r) < 30 or r.std() == 0:
            continue
        sh.append(r.mean() / r.std() * np.sqrt(244))
        ar.append(r.mean() * 244)
        wr.append((r > 0).mean())
    return (float(np.median(sh)), float(np.median(ar)), float(np.median(wr)),
            float(np.percentile(sh, 25)), float(np.percentile(sh, 75)))


say("\n## 4. 组合级模拟（每日top3分仓，30%成交概率，次日收盘离场，MC200）")
say("V0=无评分随机选股基线（同池同日数随机top3）")
say("\n| 变体 | split | Sharpe(中位) | P25-P75 | 年化(中位) | 日胜率 | 交易日 |")
say("|---|---|---|---|---|---|---|")
for split in ["train", "test"]:
    r0 = simulate(sub, "V1", split, random_pick=True)
    sh, ar, wr, p25, p75 = sharpe_stats(r0)
    nd = len(r0[0]) if r0 else 0
    say(f"| V0 随机基线 | {split} | {sh:.2f} | {p25:.2f}~{p75:.2f} "
        f"| {ar:.1%} | {wr:.1%} | {nd} |")
    for var in VARIANTS + ["V4"]:
        if var == "V4":
            s2 = sub[sub["g_chip"] >= 0.6]
            runs = simulate(s2, "g_intra", split)
        else:
            runs = simulate(sub, var, split)
        sh, ar, wr, p25, p75 = sharpe_stats(runs)
        nd = len(runs[0]) if runs else 0
        say(f"| {var} | {split} | {sh:.2f} | {p25:.2f}~{p75:.2f} "
            f"| {ar:.1%} | {wr:.1%} | {nd} |")

# ================= 论证3: 三段行情 × 时段（test，V3 vs V1/V0） =================
say("\n## 5. 论证：V3 top10% × 三段行情/时段（test）")
thr3 = np.quantile(sub[sub["split"] == "train"]["V3"].values, 0.90)
top3 = sub[(sub["split"] == "test") & (sub["V3"] >= thr3)]
say("\n| 切面 | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 |")
say("|---|---|---|---|---|---|---|")
for rg in ["偏多", "震荡", "偏空"]:
    part = top3[top3["regime"] == rg]
    if len(part) >= 30:
        n, seal, same, nwin, nmed, plr = metrics(part)
        plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
        say(f"| {rg} | {n} | {seal:.1%} | {same:.1%} | {nwin:.1%} "
            f"| {nmed:+.2f} | {plr_s} |")
for early, tlab in [(True, "≤10:00"), (False, ">10:00")]:
    part = top3[top3["td"] <= 600] if early else top3[top3["td"] > 600]
    if len(part) >= 30:
        n, seal, same, nwin, nmed, plr = metrics(part)
        plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
        say(f"| {tlab} | {n} | {seal:.1%} | {same:.1%} | {nwin:.1%} "
            f"| {nmed:+.2f} | {plr_s} |")

# ================= 消融（H4） =================
say("\n## 6. 消融：V3 移除任一组后 top10% 指标变化（test）")
say("消融方式：组权重置零后重新归一（如 -生态 = 0.5盘中+0.25筹码 → 归一）")
abl = {
    "V3 完整": 0.5 * sub["g_intra"] + 0.25 * sub["g_eco"] + 0.25 * sub["g_chip"],
    "-盘中(生态+筹码)": (0.25 * sub["g_eco"] + 0.25 * sub["g_chip"]) / 0.5,
    "-生态(盘中+筹码)": (0.5 * sub["g_intra"] + 0.25 * sub["g_chip"]) / 0.75,
    "-筹码(盘中+生态)": (0.5 * sub["g_intra"] + 0.25 * sub["g_eco"]) / 0.75,
}
say("\n| 消融 | n | 封板率 | 当日胜率 | 次日胜率 | 次日中位% | 盈亏比 | Sharpe(test) |")
say("|---|---|---|---|---|---|---|---|")
for name, score in abl.items():
    tmp = sub.assign(abl=score)
    thr = np.quantile(tmp[tmp["split"] == "train"]["abl"].values, 0.90)
    part = tmp[(tmp["split"] == "test") & (tmp["abl"] >= thr)]
    n, seal, same, nwin, nmed, plr = metrics(part)
    plr_s = f"{plr:.2f}" if np.isfinite(plr) else "-"
    runs = simulate(tmp, "abl", "test")
    sh, _, _, _, _ = sharpe_stats(runs)
    say(f"| {name} | {n} | {seal:.1%} | {same:.1%} | {nwin:.1%} | {nmed:+.2f} "
        f"| {plr_s} | {sh:.2f} |")

(OUT / "24_fusion_ablation.md").write_text("\n".join(R), encoding="utf-8")
say("\n报告已写入 research/out/24_fusion_ablation.md")
