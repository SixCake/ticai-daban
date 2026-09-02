# -*- coding: utf-8 -*-
"""研究13b: walk-forward 前向验证分析(读13的数据产物)

协议:
  - 训练期 ≤20260430 拟合规则, 测试期 ≥20260501 纯样本外评估
  - 决策点=+1%首触, 入场=T+1分钟bar收盘×1.001滑点(已含在数据entry列)
  - 评估指标: 测试期封板率/lift、入场当日收益(entry_ret)、
    入场后同日最高(max_after)、信号提前量(决策到封板分钟差)
输出: research/out/13_forward.md
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


df = pd.read_parquet(OUT / "13_forward_oos.parquet")
tdf = pd.read_parquet(OUT / "13_forward_theme.parquet")
df["split"] = np.where(df["date"] <= TRAIN_END, "train", "test")
tr = df[df["split"] == "train"]
te = df[df["split"] == "test"]

say("# 研究13: 前向验证与提前感知 (+1%决策, T+1min可执行入场)")
say(f"\n全量样本 {len(df)} (train {len(tr)} / test {len(te)}), "
    f"+1%首触后最终封板率: 全{df['y'].mean():.0%} "
    f"train {tr['y'].mean():.0%} / test {te['y'].mean():.0%}")
say(f"入场后当日收益中位(全体): {df['entry_ret'].median():.2f}% "
    f"同日最高中位: {df['max_after'].median():.2f}%")

FEATS = ["r3", "accel", "pathvol", "vr1", "vtrend", "tight", "drift",
         "volramp", "base_hi", "gap", "cm20", "t1"]

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

# ---------- W1 训练期拟合 ----------
Xtr = tr[FEATS].fillna(0).values
ytr = tr["y"].values
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60,
                              class_weight="balanced").fit(Xtr, ytr)
say("\n## W1 训练期(≤2026-04)决策树拟合")
say("```")
say(export_text(tree, feature_names=FEATS, decimals=2))
say("```")
t_ = tree.tree_
leaves_tr = tree.apply(Xtr)


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
    return conds


def conds_to_mask(d, conds):
    m = pd.Series(True, index=d.index)
    for f, op, th in conds:
        s = d[f].fillna(0)
        m &= (s <= th) if op == "<=" else (s > th)
    return m


say("| 训练期叶子 | n | 封板率 | 规则 |")
say("|---|---|---|---|")
leaf_conds = {}
for lf in sorted(set(leaves_tr)):
    mask = leaves_tr == lf
    xi = Xtr[mask][0]
    conds = path_rules(xi)
    leaf_conds[lf] = conds
    txt = " & ".join(f"{f}{op}{th:.2f}" for f, op, th in conds)
    say(f"| {lf} | {int(mask.sum())} | {ytr[mask].mean():.0%} | {txt} |")

# ---------- W2 纯前向评估(test期) ----------
say("\n## W2 纯前向评估(测试期 ≥2026-05, 训练期规则原样套用)")
base_te = te["y"].mean()
say("| 规则 | test n | test封板率 | lift | 入场当日收益% | 同日最高% |")
say("|---|---|---|---|---|---|")
say(f"| 基准(test全体) | {len(te)} | {base_te:.0%} | 1.00x "
    f"| {te['entry_ret'].median():.2f} | {te['max_after'].median():.2f} |")
for lf, conds in leaf_conds.items():
    m = conds_to_mask(te, conds)
    sub = te[m]
    if len(sub) < 20:
        continue
    say(f"| 叶子{lf} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {sub['y'].mean()/max(base_te,1e-9):.2f}x "
        f"| {sub['entry_ret'].median():.2f} | {sub['max_after'].median():.2f} |")

# ---------- W3 研究12规则的前向复核(+1%决策口径) ----------
say("\n## W3 研究12规则前向复核(+1%决策+T+1入场)")
RULES = {
    "R2 暴拉 r3>4.8": te["r3"] > 4.8,
    "P1 10cm&pathvol>0.93": (te["cm20"] == 0) & (te["pathvol"] > 0.93),
    "R1 r3>1.2&vr1>1.1&r3≤4.8": (te["r3"] > 1.2) & (te["vr1"] > 1.1)
                                & (te["r3"] <= 4.8),
}
say("| 规则 | test n | 封板率 | lift | 入场当日% | 同日最高% |")
say("|---|---|---|---|---|---|")
for name, cond in RULES.items():
    sub = te[cond.fillna(False)]
    if sub.empty:
        say(f"| {name} | 0 | - | - | - | - |")
        continue
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {sub['y'].mean()/max(base_te,1e-9):.2f}x "
        f"| {sub['entry_ret'].median():.2f} | {sub['max_after'].median():.2f} |")

# ---------- W4 蓄势段新因子单调性(train拟合依据, test验证) ----------
say("\n## W4 蓄势段因子分桶(train/test 双列对照)")


def dual_bucket(factor, bins):
    say(f"\n`{factor}`:")
    say("| 桶 | train封板率(n) | test封板率(n) |")
    say("|---|---|---|")
    for lo, hi in zip(bins[:-1], bins[1:]):
        a = tr[(tr[factor] > lo) & (tr[factor] <= hi)]
        b = te[(te[factor] > lo) & (te[factor] <= hi)]
        at = f"{a['y'].mean():.0%}({len(a)})" if len(a) else "-"
        bt = f"{b['y'].mean():.0%}({len(b)})" if len(b) else "-"
        say(f"| ({lo},{hi}] | {at} | {bt} |")


dual_bucket("tight", bins=[-0.01, 0.15, 0.3, 0.6, 99])
dual_bucket("drift", bins=[-99, 0, 1, 2.5, 99])
dual_bucket("volramp", bins=[-0.01, 1, 2, 4, 999])
dual_bucket("base_hi", bins=[-0.01, 0.5, 1.0, 1.5, 99])
dual_bucket("gap", bins=[-99, -1, 0, 1, 3, 99])

# ---------- W5 题材前向因子(30日抽样) ----------
say("\n## W5 题材前向因子(测试期抽样30日)")
m = df.merge(tdf, on=["date", "ts_code"], how="inner")
say(f"匹配样本 {len(m)} (封板率 {m['y'].mean():.0%})")
for f, bins in [("peer_lead", [-0.5, 0, 2, 5, 999]),
                ("peer_up", [-0.5, 2, 5, 10, 9999]),
                ("peer_move", [-99, -0.5, 0, 0.5, 99])]:
    say(f"\n`{f}`:")
    say("| 桶 | n | 封板率 |")
    say("|---|---|---|")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub = m[(m[f] > lo) & (m[f] <= hi)]
        if len(sub) >= 10:
            say(f"| ({lo},{hi}] | {len(sub)} | {sub['y'].mean():.0%} |")

# ---------- W6 提前量统计 ----------
say("\n## W6 信号提前量(决策到收盘封板的距离)")
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ft = ev.set_index(["trade_date", "ts_code"])["first_time"]
df["ft"] = [ft.get((r.date, r.ts_code), None)
            for r in df.itertuples()]
sealed = df[(df["y"] == 1) & df["ft"].notna()].copy()
sealed["ft_min"] = sealed["ft"].astype(str).str.zfill(6).map(
    lambda s: int(s[:2]) * 60 + int(s[2:4]))
sealed["lead"] = sealed["ft_min"] - sealed["t1"]
say(f"最终封板样本 n={len(sealed)}; 决策(+1%触)到首次封板分钟差: "
    f"中位 {sealed['lead'].median():.0f}min, "
    f"25分位 {sealed['lead'].quantile(0.25):.0f}min")
say(f"决策时刻已封(lead≤0, 即决策慢于封板)占比: "
    f"{(sealed['lead'] <= 0).mean():.0%}")

report = "\n".join(R)
(OUT / "13_forward.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/13_forward.md")
