# -*- coding: utf-8 -*-
"""研究14c: 离线快跑版(零下载, 复用12_expanded_oos.parquet)

降级说明: 决策点=+2%触板(非+1%), 入场=触板bar收盘(非T+1min),
高开队列不可恢复(留待缓存版下载)。其余协议不变:
  - T1 时段效应复核: 封板时刻×次日收益(验证"10点前封板次日更好")
  - T2 决策时刻×封板率/次日收益
  - W1 walk-forward: train≤20260430拟合, test≥20260501纯前向
  - W2 研究12规则前向复核
输出: research/out/14c_offline_report.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
TRAIN_END = "20260430"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


df = pd.read_parquet(OUT / "12_expanded_oos.parquet")
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ft = ev.set_index(["trade_date", "ts_code"])["first_time"]
df["ft"] = [ft.get((r.date, r.ts_code), None) for r in df.itertuples()]
df["split"] = np.where(df["date"] <= TRAIN_END, "train", "test")
tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

say("# 研究14c: 离线前向验证(+2%决策, 复用研究12数据)")
say(f"\n样本 {len(df)} (train {len(tr)} / test {len(te)}), "
    f"基准封板率 全{df['y'].mean():.0%} train{tr['y'].mean():.0%} "
    f"test{te['y'].mean():.0%}")

# ---------- T1 时段效应复核 ----------
say("\n## T1 封板时刻 × 次日收益(复核'10点前封板次日更好')")
sealed = df[(df["y"] == 1) & df["ft"].notna()].copy()
sealed["ft_min"] = sealed["ft"].astype(str).str.zfill(6).map(
    lambda s: int(s[:2]) * 60 + int(s[2:4]))
say("| 封板时段 | n | 次日收益中位% | 次日胜率 | 次日收益均值% |")
say("|---|---|---|---|---|")
for lo, hi, lab in [(0, 571, "开盘即封"), (571, 600, "早盘<10点"),
                    (600, 690, "10:00-11:30"), (690, 901, "午后")]:
    sub = sealed[(sealed["ft_min"] >= lo) & (sealed["ft_min"] < hi)]
    nr = sub["next_ret"].dropna()
    if len(sub) >= 10:
        say(f"| {lab} | {len(sub)} | {nr.median():.2f} "
            f"| {(nr > 0).mean():.0%} | {nr.mean():.2f} |")

# ---------- T2 决策时刻 ----------
say("\n## T2 触+2%决策时刻 × 封板率/次日收益")
say("| 决策时段 | n | 封板率 | 次日收益中位%(封板者) |")
say("|---|---|---|---|")
for lo, hi, lab in [(569, 600, "<10点"), (600, 690, "10-11:30"),
                    (690, 901, "午后")]:
    sub = df[(df["t2"] >= lo) & (df["t2"] < hi)]
    sl = sub[(sub["y"] == 1)]["next_ret"].dropna()
    if len(sub) >= 50:
        say(f"| {lab} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {sl.median():.2f}(n={len(sl)}) |")

# ---------- W1 walk-forward ----------
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

FEATS = ["r3", "r5", "r10", "accel", "pathvol", "drawdown", "convex",
         "vr2", "vtrend", "co_con", "cm20", "t2"]
X = tr[FEATS].fillna(0).values
yy = tr["y"].values
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60,
                              class_weight="balanced").fit(X, yy)
say("\n## W1 walk-forward: 训练期(≤2026-04)决策树")
say("```")
say(export_text(tree, feature_names=FEATS, decimals=2))
say("```")
t_ = tree.tree_


def path_rules(xi):
    node, conds = 0, []
    while t_.children_left[node] >= 0:
        f = FEATS[t_.feature[node]]
        th = t_.threshold[node]
        if xi[t_.feature[node]] <= th:
            conds.append((f, "<=", th))
            node = t_.children_left[node]
        else:
            conds.append((f, ">", th))
            node = t_.children_right[node]
        if len(conds) > 5:
            break
    return conds


def mask_of(d, conds):
    m = pd.Series(True, index=d.index)
    for f, op, th in conds:
        s = d[f].fillna(0)
        m &= (s <= th) if op == "<=" else (s > th)
    return m


leaves = tree.apply(X)
base_te = te["y"].mean()
say("| 训练期正向叶子规则 | test n | test封板率 | lift | test次日%(封板者) |")
say("|---|---|---|---|---|")
for lf in sorted(set(leaves)):
    mask = leaves == lf
    if yy[mask].mean() <= yy.mean():
        continue
    conds = path_rules(X[mask][0])
    txt = " & ".join(f"{f}{op}{th:.2f}" for f, op, th in conds)
    sub = te[mask_of(te, conds)]
    sl = sub[sub["y"] == 1]["next_ret"].dropna()
    if len(sub) >= 30:
        say(f"| {txt} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {sub['y'].mean()/max(base_te,1e-9):.2f}x "
            f"| {sl.median():.2f}(n={len(sl)}) |")

# ---------- W2 研究12规则前向复核 ----------
say("\n## W2 研究12规则纯前向(test期)复核")
RULES = {
    "R2 暴拉 r3>4.8": te["r3"] > 4.8,
    "P1 10cm&pathvol>0.93": (te["cm20"] == 0) & (te["pathvol"] > 0.93),
    "P1a 10cm&pathvol∈(0.5,0.93]": (te["cm20"] == 0)
        & (te["pathvol"] > 0.5) & (te["pathvol"] <= 0.93),
    "R1 r3>1.2&vr2>1.1&r3≤4.8": (te["r3"] > 1.2) & (te["vr2"] > 1.1)
                                & (te["r3"] <= 4.8),
    "R3 蓄势(已被证伪)": (te["r3"] <= 1.2) & (te["accel"] > 0.3)
                         & (te["cm20"] == 0),
}
say("| 规则 | test n | 封板率 | lift | 次日%(封板者) | 次日%(全体) |")
say("|---|---|---|---|---|---|")
for name, cond in RULES.items():
    sub = te[cond.fillna(False)]
    if sub.empty:
        continue
    sl = sub[sub["y"] == 1]["next_ret"].dropna()
    na = sub["next_ret"].dropna()
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {sub['y'].mean()/max(base_te,1e-9):.2f}x "
        f"| {sl.median():.2f}(n={len(sl)}) | {na.median():.2f}(n={len(na)}) |")

report = "\n".join(R)
(OUT / "14c_offline_report.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/14c_offline_report.md")
