# -*- coding: utf-8 -*-
"""研究10: 提前抓涨停 — 启动初期(+1%附近)信号挖掘

目标更新: 从"半路(+7%)追高判别" → "提前(启动初期)预测"。
决策时刻 = 该股首个 pct∈[0.8,2.5] 的20s记录(雷达已因prob≥0.2收录),
特征全部取该时刻及之前。标签: 最终涨停/炸板/未板。

研究问题: 雷达早期预警(prob≥0.2@低位)的票里, 时序形态×量能×题材共振
×模型分, 谁能把最终涨停者提前分离出来?

注意: 日志收录条件(pct≥3或prob≥0.2)使样本=模型已预警子集,
本研究回答"预警内精选", 不回答"全市场早发现"。
输出: research/out/10_early_signal_20260826.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATE = "20260826"
OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


zt = pd.read_parquet(f"/tmp/zt_{DATE}.parquet")
zb = pd.read_parquet(f"/tmp/zb_{DATE}.parquet")
for d in (zt, zb):
    d["ts_code"] = d["代码"].astype(str).str.zfill(6).map(
        lambda x: x + (".SH" if x.startswith(("60", "68")) else ".SZ"))
zt_by = {r["ts_code"]: r for _, r in zt.iterrows()}
zb_by = {r["ts_code"]: r for _, r in zb.iterrows()}

log = []
with open(ROOT / f"data/live/radar_log_{DATE}.jsonl") as f:
    for line in f:
        log.append(json.loads(line))
by_code = defaultdict(list)
for r in sorted(log, key=lambda x: x["t"]):
    by_code[r["code"]].append(r)


def tsec(t: str) -> int:
    return int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


rows = []
for c, g in by_code.items():
    # 决策记录: 首个 pct∈[0.8,2.5] 且 时刻≥0950(保证轨迹积累)
    cand = [r for r in g if 0.8 <= r["pct"] <= 2.5 and r["t"] >= "095000"]
    if not cand:
        continue
    rec = cand[0]
    t0 = tsec(rec["t"])
    pre = [r for r in g if t0 - 600 <= tsec(r["t"]) < t0]
    if c in zt_by:
        out_grp = "涨停"
    elif c in zb_by:
        out_grp = "炸板"
    else:
        out_grp = "未板"

    def pct_at(sec_back):
        sel = [r for r in pre if tsec(r["t"]) <= t0 - sec_back]
        return sel[-1]["pct"] if sel else rec["pct"] - 0.5
    p600, p300, p180, p60 = (pct_at(600), pct_at(300), pct_at(180),
                             pct_at(60))
    accel = (rec["pct"] - p60) - (p60 - p180)
    pcts = np.array([r["pct"] for r in pre[-30:]] + [rec["pct"]])
    pathvol = float(np.diff(pcts).std()) if len(pcts) > 2 else 0.0
    cummax = np.maximum.accumulate(pcts)
    dd = float((cummax - pcts).max()) if len(pcts) else 0.0
    half = pct_at(300)
    convex = (rec["pct"] - half) - (half - p600)
    # 启动前蓄势: 决策前10min的最高涨幅(是否已试过盘)
    pre_hi = float(pcts[:-1].max()) if len(pcts) > 1 else 0.0
    rows.append({"ts_code": c, "name": rec["name"], "out": out_grp,
                 "t0": t0, "theme": rec["theme"], "pct": rec["pct"],
                 "r600": rec["pct"] - p600, "r300": rec["pct"] - p300,
                 "r60": rec["pct"] - p60, "accel": accel,
                 "pathvol": pathvol, "drawdown": dd, "convex": convex,
                 "pre_hi": pre_hi,
                 "vr": rec["vr"], "tover": rec["tover"],
                 "vpg": rec["vr"] - rec["pct"] * 0.8,
                 "heat": rec["heat"], "trank": rec["trank"],
                 "dheat": rec["dheat"],
                 "dist": rec["dist"],
                 "cm20": int(c[:2] in ("30", "68")),
                 "prob": rec["prob"], "dp": rec["dp"]})
df = pd.DataFrame(rows)
df["y"] = (df["out"] == "涨停").astype(int)

# 题材早期共振: ±5min内同题材也有≥1%记录的票数(含自身)
low_touch = [(r["t0"], r["theme"]) for r in rows]
df["co_theme"] = [
    sum(1 for t, th in low_touch
        if th == r["theme"] and abs(t - r["t0"]) <= 300)
    for r in rows]
df.to_parquet(OUT / f"10_early_{DATE}.parquet", index=False)

say(f"# 研究10: 提前抓涨停 — 启动初期信号({DATE})")
n_pos = int((df["out"] == "涨停").sum())
say(f"\n早期预警样本(首记录+1%附近): {len(df)} = 涨停{n_pos} "
    f"炸板{int((df['out']=='炸板').sum())} 未板{int((df['out']=='未板').sum())}")
say(f"基准: 预警票最终涨停率 {df['y'].mean():.0%} "
    f"(对比: 全市场涨停{len(zt_by)}只, 预警召回{n_pos}/{len(zt_by)})")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

FAMS = {
    "A时序": ["r600", "r300", "r60", "accel", "pathvol", "drawdown",
              "convex", "pre_hi"],
    "B量能": ["vr", "tover", "vpg"],
    "C题材": ["heat", "trank", "dheat", "co_theme"],
    "D位置": ["dist", "cm20", "pct"],
    "E模型": ["prob", "dp"],
}
FEATS = [f for fs in FAMS.values() for f in fs]
X = df[FEATS].fillna(0).values
y = df["y"].values
if y.sum() < 5:
    say("正例过少, 终止建模")
    sys.exit(0)

sc = StandardScaler()
Xs = sc.fit_transform(X)
lr = LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced").fit(
    Xs, y)
say("\n## L1 logistic系数(balance, |系数|降序)")
say("| 特征 | 系数 | 含义 |")
say("|---|---|---|")
for f, c in sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1])):
    if abs(c) >= 0.1:
        say(f"| {f} | {c:+.2f} | {'利成真' if c > 0 else '利落空'} |")

say("\n## L2 特征族消融(单族balance-logistic)")
say("| 特征集 | 准确率 | 涨停召回 | 预警精准 |")
say("|---|---|---|---|")


def evalfam(Xf):
    scf = StandardScaler()
    Xfs = scf.fit_transform(Xf)
    m = LogisticRegression(max_iter=3000, C=0.2,
                           class_weight="balanced").fit(Xfs, y)
    p = m.predict(Xfs)
    prec = p[y == 1].mean() if p.sum() else 0
    rec = p[y == 1].mean() if y.sum() else 0
    return m.score(Xfs, y), p[y == 1].mean() if y.sum() else 0, \
        (y[p == 1].mean() if p.sum() else 0)


for fam, fs in FAMS.items():
    acc, rec, prec = evalfam(df[fs].fillna(0).values)
    say(f"| {fam} | {acc:.2f} | {rec:.0%} | {prec:.0%} |")
acc, rec, prec = evalfam(X)
say(f"| 全量 | {acc:.2f} | {rec:.0%} | {prec:.0%} |")

say("\n## L3 决策树交互规则(depth=3, balance)")
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8,
                              class_weight="balanced").fit(X, y)
say("```")
say(export_text(tree, feature_names=FEATS, decimals=2))
say("```")
t_ = tree.tree_
leaves = tree.apply(X)


def path_rules(xi):
    node, conds = 0, []
    while t_.children_left[node] >= 0:
        f = FEATS[t_.feature[node]]
        th = t_.threshold[node]
        if xi[t_.feature[node]] <= th:
            conds.append(f"{f}≤{th:.1f}")
            node = t_.children_left[node]
        else:
            conds.append(f"{f}>{th:.1f}")
            node = t_.children_right[node]
    return conds


say("| 叶子 | n | 涨停率 | 路径条件 |")
say("|---|---|---|---|")
for lf in sorted(set(leaves)):
    mask = leaves == lf
    say(f"| {lf} | {int(mask.sum())} | {y[mask].mean():.0%} "
        f"| {' & '.join(path_rules(X[mask][0]))} |")

# 高分叶子命中明细(供人工核对)
say("\n## 高涨停率叶子明细")
for lf in sorted(set(leaves)):
    mask = leaves == lf
    if y[mask].mean() >= 0.15 and mask.sum() >= 8:
        sub = df[mask]
        say(f"\n叶子{lf}(n={len(sub)}, 涨停率{y[mask].mean():.0%}):")
        for _, r in sub[sub["y"] == 1].iterrows():
            say(f"  ✓ {r['name']} 决策时pct={r['pct']:.1f} "
                f"vr={r['vr']:.1f} heat={r['heat']:.0f} "
                f"co={int(r['co_theme'])} {r['theme']}")

report = "\n".join(R)
(OUT / f"10_early_signal_{DATE}.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/10_early_signal_{DATE}.md")
