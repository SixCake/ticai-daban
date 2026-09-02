# -*- coding: utf-8 -*-
"""研究14b: 双队列前向验证分析

协议:
  - walk-forward: 训练≤20260430拟合, 测试≥20260501纯前向评估
  - 入场均为T+1min收盘×1.001滑点(数据entry列已含)
  - 时段效应复核: 决策时刻/封板时刻分桶 × 次日收益
输出: research/out/14_dual_report.md
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


df = pd.read_parquet(OUT / "14_dual_oos.parquet")
df["split"] = np.where(df["date"] <= TRAIN_END, "train", "test")
G = df[df["cohort"] == "G"]
L = df[df["cohort"] == "L"]

say("# 研究14: 双队列前向验证(高开G组 + 低拉L组)")
say(f"\n样本 {len(df)}: G组高开 {len(G)}(封板率{G['y'].mean():.0%}) "
    f"L组低拉 {len(L)}(封板率{L['y'].mean():.0%})")
say(f"次日收益中位: G {G['next_ret'].median():.2f}% / "
    f"L {L['next_ret'].median():.2f}%")

# ---------- T1 时段效应复核 ----------
say("\n## T1 时段效应复核(封板时刻 × 次日收益, 全体封板样本)")
sealed = df[(df["y"] == 1) & df["ft"].notna()].copy()
sealed["ft_min"] = sealed["ft"].astype(str).str.zfill(6).map(
    lambda s: int(s[:2]) * 60 + int(s[2:4]))
say("| 封板时段 | n | 次日收益中位% | 次日胜率 |")
say("|---|---|---|---|")
for lo, hi, lab in [(0, 570, "开盘一字"), (570, 600, "早盘<10点"),
                    (600, 690, "10-11:30"), (690, 901, "午后")]:
    sub = sealed[(sealed["ft_min"] >= lo) & (sealed["ft_min"] < hi)]
    nr = sub["next_ret"].dropna()
    if len(sub) >= 10:
        say(f"| {lab} | {len(sub)} | {nr.median():.2f} "
            f"| {(nr > 0).mean():.0%} |")

say("\n## T2 决策时刻 × 次日收益(以T+1入场价计)")
say("| 决策时段 | 队列 | n | 封板率 | 入场当日% | 次日% | 次日胜率 |")
say("|---|---|---|---|---|---|---|")
for lo, hi, lab in [(569, 600, "<10点"), (600, 690, "10-11:30"),
                    (690, 901, "午后")]:
    for cq in ["G", "L"]:
        sub = df[(df["cohort"] == cq) & (df["td"] >= lo) & (df["td"] < hi)]
        nr = sub["next_ret"].dropna()
        if len(sub) >= 20:
            say(f"| {lab} | {cq} | {len(sub)} | {sub['y'].mean():.0%} "
                f"| {sub['entry_ret'].median():.2f} | {nr.median():.2f} "
                f"| {(nr > 0).mean():.0%} |")

# ---------- G组 高开队列因子 ----------
say("\n## G1 高开队列因子分桶(全窗口)")


def bucket(scope, factor, bins):
    say(f"\n`{factor}`:")
    say("| 桶 | n | 封板率 | 入场当日% | 次日% |")
    say("|---|---|---|---|---|")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = scope[(scope[factor] > lo) & (scope[factor] <= hi)]
        nr = sub["next_ret"].dropna()
        if len(sub) >= 20:
            say(f"| ({lo},{hi}] | {len(sub)} | {sub['y'].mean():.0%} "
                f"| {sub['entry_ret'].median():.2f} | {nr.median():.2f} |")


bucket(G, "gap", [-99, 2, 3, 5, 7, 99])
bucket(G, "open_vr", [-0.1, 2, 5, 10, 9999])
bucket(G, "om3", [-99, -1, 0, 1, 99])
bucket(G, "odip", [-0.01, 0.5, 1, 2, 99])

# ---------- walk-forward ----------
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

FEATS_G = ["gap", "open_vr", "om3", "odip", "amp3", "cm20"]
FEATS_L = ["r3", "accel", "pathvol", "vr1", "vtrend", "tight", "drift",
           "volramp", "base_hi", "gap", "cm20", "td"]


def conds_to_mask(d, conds):
    m = pd.Series(True, index=d.index)
    for f, op, th in conds:
        s = d[f].fillna(0)
        m &= (s <= th) if op == "<=" else (s > th)
    return m


def walk_forward(cohort_df, feats, name):
    tr = cohort_df[cohort_df["split"] == "train"]
    te = cohort_df[cohort_df["split"] == "test"]
    X = tr[feats].fillna(0).values
    yy = tr["y"].values
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
                                  class_weight="balanced").fit(X, yy)
    say(f"\n## {name} 训练期决策树")
    say("```")
    say(export_text(tree, feature_names=feats, decimals=2))
    say("```")
    t_ = tree.tree_

    def path_rules(xi):
        node, conds = 0, []
        while t_.children_left[node] >= 0:
            f = feats[t_.feature[node]]
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

    leaves = tree.apply(X)
    base_te = te["y"].mean()
    say(f"| 叶子规则(train拟合) | test n | test封板率 | lift | test次日% |")
    say("|---|---|---|---|---|")
    for lf in sorted(set(leaves)):
        mask = leaves == lf
        if yy[mask].mean() <= yy.mean():
            continue                        # 只前向验证正向叶子
        conds = path_rules(X[mask][0])
        txt = " & ".join(f"{f}{op}{th:.2f}" for f, op, th in conds)
        sub = te[conds_to_mask(te, conds)]
        nr = sub["next_ret"].dropna()
        if len(sub) >= 15:
            say(f"| {txt} | {len(sub)} | {sub['y'].mean():.0%} "
                f"| {sub['y'].mean()/max(base_te,1e-9):.2f}x "
                f"| {nr.median():.2f} |")


walk_forward(G, FEATS_G, "W1 G组(高开)")
walk_forward(L, FEATS_L, "W2 L组(低拉)")

# ---------- 研究12规则前向复核 ----------
say("\n## W3 研究12规则在L组的前向复核(test期)")
te = L[L["split"] == "test"]
base_te = te["y"].mean()
RULES = {
    "R2 暴拉 r3>4.8": te["r3"] > 4.8,
    "P1 10cm&pathvol>0.93": (te["cm20"] == 0) & (te["pathvol"] > 0.93),
    "R1 r3>1.2&vr1>1.1": (te["r3"] > 1.2) & (te["vr1"] > 1.1)
                          & (te["r3"] <= 4.8),
}
say("| 规则 | test n | 封板率 | lift | 入场当日% | 次日% |")
say("|---|---|---|---|---|---|")
for name, cond in RULES.items():
    sub = te[cond.fillna(False)]
    nr = sub["next_ret"].dropna()
    if len(sub):
        say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {sub['y'].mean()/max(base_te,1e-9):.2f}x "
            f"| {sub['entry_ret'].median():.2f} | {nr.median():.2f} |")

report = "\n".join(R)
(OUT / "14_dual_report.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/14_dual_report.md")
