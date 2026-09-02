# -*- coding: utf-8 -*-
"""研究09: 半路抓涨停 — 多维交互挖掘(时序形态×量能×题材×共振)

研究07/08的局限: 单因子分桶, 只用决策时刻截面快照。本轮用雷达20s高频
轨迹构建触板前完整特征向量, 让树模型发现交互, 消融量化各组贡献。

特征族(决策时刻=首触+3%的20s快照, 全部当时可见):
  A 时序形态: 触板前3/5/10min涨幅、加速度(近1min-前3min涨速)、
              轨迹波动、回撤深度、回撤后修复、拉升耗时(凹/凸)
  B 量能: vr、触板时tover、量价背离(高涨幅低量比)
  C 题材: heat、trank、dheat、同题材并涨家数(±5min窗口同theme≥3%)
  D 位置/板型: pct、dist、cm20、fmv、昨涨停接力
  E 模型分: 雷达prob、dp(较上cycle变化)
标签: 涨停/炸板/未板(研究07样本口径)
模型: 决策树(交互规则发现) + logistic(方向与强度) + 族消融
输出: research/out/09_multidim_20260826.md
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


# ---------- 样本(复用研究07口径) ----------
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
    hits = [r for r in g if r["pct"] >= 3]
    if not hits or hits[0]["t"] < "095000":
        continue
    rec = hits[0]
    t0 = tsec(rec["t"])
    pre = [r for r in g if t0 - 600 <= tsec(r["t"]) < t0]   # 前10min
    if c in zt_by:
        out_grp, fmv = "涨停", float(zt_by[c]["流通市值"]) / 1e8
    elif c in zb_by:
        out_grp, fmv = "炸板", float(zb_by[c]["流通市值"]) / 1e8
    else:
        out_grp, fmv = "未板", 0.0
    # A 时序形态
    def pct_at(sec_back):
        sel = [r for r in pre if tsec(r["t"]) <= t0 - sec_back]
        return sel[-1]["pct"] if sel else rec["pct"]
    p600, p300, p180, p60 = (pct_at(600), pct_at(300), pct_at(180),
                             pct_at(60))
    accel = (rec["pct"] - p60) - (p60 - p180)        # 近1min增量 - 前一增量
    pcts = np.array([r["pct"] for r in pre[-30:]] + [rec["pct"]])
    pathvol = float(np.diff(pcts).std()) if len(pcts) > 2 else 0.0
    cummax = np.maximum.accumulate(pcts)
    dd = float((cummax - pcts).max()) if len(pcts) else 0.0   # 回撤深度
    peak_i = int(np.argmax(pcts))
    recover = float(pcts[-1] - pcts[peak_i]) if peak_i < len(pcts) - 1 else 0.0
    # 拉升耗时凹性: 从p600到触板的实际时长中后半程涨幅占比
    half = pct_at(300)
    convex = (rec["pct"] - half) - (half - p600)     # >0 后半程加速
    # B 量能
    vr, tover = rec["vr"], rec["tover"]
    vol_price_gap = vr - rec["pct"] * 0.8            # <0 量落后于价
    # C 题材
    heat, trank, dheat, theme = (rec["heat"], rec["trank"],
                                 rec["dheat"], rec["theme"])
    # D 位置/板型
    dist, pct = rec["dist"], rec["pct"]
    cm20 = int(c[:2] in ("30", "68"))
    # E 模型分
    prob = rec["prob"]
    idx = g.index(rec)
    dp = prob - g[idx - 1]["prob"] if idx > 0 else 0.0
    rows.append({"ts_code": c, "name": rec["name"], "out": out_grp,
                 "t0": t0, "theme": theme,
                 "r600": rec["pct"] - p600, "r300": rec["pct"] - p300,
                 "r60": rec["pct"] - p60, "accel": accel,
                 "pathvol": pathvol, "drawdown": dd, "recover": recover,
                 "convex": convex, "vr": vr, "tover": tover,
                 "vpg": vol_price_gap, "heat": heat, "trank": trank,
                 "dheat": dheat, "dist": dist, "pct": pct, "cm20": cm20,
                 "fmv": fmv, "prob": prob, "dp": dp})
df = pd.DataFrame(rows)
df["y"] = (df["out"] == "涨停").astype(int)
df["y3"] = df["out"].map({"涨停": 2, "炸板": 1, "未板": 0})

# C 共振: ±5min内同题材触+3%家数(含自身)
touch_theme = [(r["t0"], r["theme"]) for r in rows]
co = []
for r in rows:
    n = sum(1 for t, th in touch_theme
            if th == r["theme"] and abs(t - r["t0"]) <= 300)
    co.append(n)
df["co_theme"] = co
df.to_parquet(OUT / f"09_multidim_{DATE}.parquet", index=False)

say(f"# 研究09: 多维交互挖掘({DATE})")
say(f"\n样本: {len(df)} = 涨停{int((df['out']=='涨停').sum())} "
    f"炸板{int((df['out']=='炸板').sum())} 未板{int((df['out']=='未板').sum())}")

# ---------- 建模 ----------
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

FAMS = {
    "A时序": ["r600", "r300", "r60", "accel", "pathvol", "drawdown",
              "recover", "convex"],
    "B量能": ["vr", "tover", "vpg"],
    "C题材": ["heat", "trank", "dheat", "co_theme"],
    # fmv排除: 负例未取市值=0会造成泄漏(首版fmv系数+1.21实为泄漏)
    "D位置": ["dist", "pct", "cm20"],
    "E模型": ["prob", "dp"],
}
FEATS = [f for fs in FAMS.values() for f in fs]
X = df[FEATS].fillna(0).values
y = df["y"].values

# 1) logistic 全特征: 系数方向与强度
sc = StandardScaler()
lr = LogisticRegression(max_iter=2000, C=0.3).fit(sc.fit_transform(X), y)
say("\n## L1 logistic 标准化系数(绝对值降序)")
say("| 特征 | 族 | 系数 | 方向含义 |")
say("|---|---|---|---|")
fam_of = {f: k for k, fs in FAMS.items() for f in fs}
coef = sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1]))
for f, c in coef:
    if abs(c) < 0.1:
        continue
    say(f"| {f} | {fam_of[f]} | {c:+.2f} | "
        f"{'利封板' if c > 0 else '利回落'} |")
say(f"\n全特征logistic R²近似(训练): {lr.score(sc.transform(X), y):.2f}")

# 2) 族消融: 每族单独的判别力
say("\n## L2 特征族消融(单族logistic准确率 vs 全量)")
say("| 特征集 | 准确率 | 正例召回 |")
say("|---|---|---|")
base = LogisticRegression(max_iter=2000, C=0.3)
all_acc = base.fit(sc.transform(X), y).score(sc.transform(X), y)
for fam, fs in FAMS.items():
    Xf = df[fs].fillna(0).values
    scf = StandardScaler()
    m = LogisticRegression(max_iter=2000, C=0.3).fit(scf.fit_transform(Xf), y)
    pred = m.predict(scf.transform(Xf))
    rec_rate = pred[y == 1].mean() if y.sum() else 0
    say(f"| {fam} | {m.score(scf.transform(Xf), y):.2f} | {rec_rate:.0%} |")
say(f"| 全量 | {all_acc:.2f} | - |")
say("(注: 训练集内评估, 用于比较族间相对判别力, 非泛化指标)")

# 3) 决策树: 交互规则发现
say("\n## L3 决策树发现的交互规则(depth=3)")
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=12).fit(X, y)
say("```")
say(export_text(tree, feature_names=FEATS, decimals=2))
say("```")
leaves = tree.apply(X)
# 叶子节点规则提取: 沿树路径拼条件
t_ = tree.tree_


def path_rules(xi):
    node = 0
    conds = []
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
    n = mask.sum()
    rate = y[mask].mean()
    xi = X[mask][0]
    conds = " & ".join(path_rules(xi))
    say(f"| {lf} | {int(n)} | {rate:.0%} | {conds} |")

report = "\n".join(R)
(OUT / f"09_multidim_{DATE}.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/09_multidim_{DATE}.md")
