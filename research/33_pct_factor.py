# -*- coding: utf-8 -*-
"""研究33: 分位化龙头因子 — 候选方案并行回测与三段市况复核

动机(承接研究32): 二值累加的 qscore/sscore 在全市场人群下塌缩(顶档占比
82.7%), 而当日横截面分位化可彻底消除塌缩(每档恒定20%)。研究32 已测出
分位化对【封板率】过闸(lift 1.75x/ρ=+1.00)、对【次日胜率】反向
(lift 0.90x/ρ=-0.90)。本研究回答: 哪个分位化方案最优, 且是否三段市况稳健。

方法论纪律:
- 目标变量锁定【封板率 y】(次日胜率维度已被研究32 判反向, 不再优化);
- 4个候选方案并行, 数据对比选最优, 不做单方案调参;
- 权重只在 train 段定, test 段复核(walk-forward, 无前视);
- 分位一律按【当日横截面】算(跨日分位会引入前视);
- 三段市况(偏多/震荡/偏空)分别复核, 任一段失效即不采纳;
- 白盒可解释: 每个候选给出显式公式, 无黑盒拟合。

候选方案:
  A0 基线   = sscore 二值累加(研究23 冻结版)
  P1 等权   = mean(pct*(zb_cnt20), pct*(ind_ztdens), pct↓(ind_rank),
                   pct↓(y_volr5), pct*(ind_breadth))
  P2 减法   = P1 剔除 train段 |rankIC|<0.02 或方向与假设反的成分
  P3 IC加权 = P1 各成分按 train段 |rankIC| 归一化加权
  ↑ = 越大越好, ↓ = 越小越好(取 1-pct)

评价: 5档单调性 + 顶/底 spread(pp) + lift + 四关(研究32) + 三段市况

数据: research/out/22_features.parquet (20250901~20260825, train/test)
输出: research/out/33_pct_factor.md

用法: python research/33_pct_factor.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
R = []

# 成分与方向假设(+ = 越大越好, - = 越小越好), 沿用 sscore 定义
COMPS = {"zb_cnt20": +1, "ind_ztdens": +1, "ind_rank": -1,
         "y_volr5": -1, "ind_breadth": +1}
IC_MIN = 0.02          # P2 减法阈: |rankIC| 低于此判为无信息量
N_Q = 5                # 分档数
# 四关(研究32 冻结)
G1_TOP_MAX, G2_MIN_B, G2_TOP_MAX = 0.40, 4, 0.50
G3_MIN_LIFT, G4_MIN_RHO = 1.30, 0.80


def say(s=""):
    R.append(s)
    print(s, flush=True)


def pct_score(df: pd.DataFrame, comps: dict, weights: dict | None = None) -> pd.Series:
    """当日横截面分位等权/加权求和 → [0,1]。按 date 分组算, 无跨日前视。"""
    out = pd.Series(np.nan, index=df.index)
    for d, g in df.groupby("date"):
        acc = pd.Series(0.0, index=g.index)
        wsum = 0.0
        for c, sgn in comps.items():
            v = g[c]
            if v.notna().sum() < 5:
                continue
            p = v.rank(pct=True, ascending=(sgn > 0))
            if sgn < 0:
                p = 1.0 - p
            w = 1.0 if weights is None else weights.get(c, 0.0)
            if w <= 0:
                continue
            acc = acc + p.fillna(0.5) * w
            wsum += w
        if wsum > 0:
            out.loc[g.index] = acc / wsum
    return out


def rank_ic(df: pd.DataFrame, col: str, target: str) -> float:
    """逐日横截面 rank IC 的均值(Spearman), 无前视"""
    ics = []
    for _, g in df.groupby("date"):
        s = g[[col, target]].dropna()
        if len(s) < 30 or s[target].nunique() < 2:
            continue
        ics.append(s[col].corr(s[target], method="spearman"))
    return float(np.mean(ics)) if ics else np.nan


def quintile(df: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    """按当日横截面五分档(避免跨日分布漂移)"""
    q = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("date"):
        s = g[score].dropna()
        if len(s) < 25:
            continue
        try:
            q.loc[s.index] = pd.qcut(s.rank(method="first"), N_Q,
                                     labels=range(1, N_Q + 1)).astype(float)
        except ValueError:
            continue
    return q


def four_gate(per: pd.Series, dist: pd.Series, label: str) -> list:
    top, bot = per.index.max(), per.index.min()
    lift = per.loc[top] / max(per.loc[bot], 1e-9)
    rho = float(pd.Series(range(len(per))).corr(
        per.reset_index(drop=True), method="spearman"))
    return [("G1 顶档<40%", dist.iloc[-1] < G1_TOP_MAX, f"{dist.iloc[-1]:.1%}"),
            (f"G2 档数≥{G2_MIN_B}且最大档<50%",
             len(dist) >= G2_MIN_B and dist.max() < G2_TOP_MAX,
             f"{len(dist)}档/{dist.max():.0%}"),
            (f"G3 lift>{G3_MIN_LIFT}", lift > G3_MIN_LIFT, f"{lift:.2f}x"),
            (f"G4 |ρ|>{G4_MIN_RHO}且方向正", rho > G4_MIN_RHO, f"ρ={rho:+.2f}")]


# ================= 加载 =================
say("# 研究33: 分位化龙头因子 — 候选方案并行回测")
f = pd.read_parquet(OUT / "22_features.parquet")
say(f"\n样本={len(f):,} (train={int((f['split']=='train').sum())}/"
    f"test={int((f['split']=='test').sum())}) "
    f"{f['date'].min()}~{f['date'].max()}")
say(f"目标锁定【封板率 y】, 基线均值={f['y'].mean():.1%}; "
    f"三段市况 偏多={int((f['regime']=='偏多').sum())}/"
    f"震荡={int((f['regime']=='震荡').sum())}/偏空={int((f['regime']=='偏空').sum())}")

# ================= 1. 单因子方向检验(train段) =================
say("\n## 1. 单因子 rank IC(train段, 逐日横截面均值)")
say("\n| 成分 | 方向假设 | rankIC(封板率) | 与假设一致 | rankIC(次日胜率) |")
say("|---|---|---|---|---|")
tr = f[f["split"] == "train"]
ic_y, ic_n = {}, {}
for c, sgn in COMPS.items():
    iy = rank_ic(tr, c, "y")
    inn = rank_ic(tr, c, "next_win")
    ic_y[c], ic_n[c] = iy, inn
    ok = "✅" if (iy > 0) == (sgn > 0) and abs(iy) >= IC_MIN else "❌"
    say(f"| {c} | {'↑越大越好' if sgn > 0 else '↓越小越好'} | {iy:+.4f} "
        f"| {ok} | {inn:+.4f} |")

keep = [c for c, sgn in COMPS.items()
        if abs(ic_y[c]) >= IC_MIN and (ic_y[c] > 0) == (sgn > 0)]
drop = [c for c in COMPS if c not in keep]
say(f"\nP2 减法保留 {len(keep)} 项: {keep}")
say(f"P2 减法剔除 {len(drop)} 项: {drop} "
    f"(|IC|<{IC_MIN} 或方向与假设反)")
wts = {c: abs(ic_y[c]) for c in keep}
tw = sum(wts.values()) or 1.0
wts = {c: v / tw for c, v in wts.items()}
say(f"P3 IC加权(train归一): " + " ".join(f"{c}:{v:.3f}" for c, v in wts.items()))

# ================= 2. 候选方案打分 =================
say("\n## 2. 候选方案并行回测(全样本, 目标=封板率)")
f["A0"] = ((f["zb_cnt20"] >= 0.5).astype(int) + (f["ind_ztdens"] >= 0.03).astype(int)
           + (f["ind_rank"] <= 3.5).astype(int) + (f["y_volr5"] < 2.5).astype(int)
           + (f["ind_breadth"] >= 0.65).astype(int))
f["P1"] = pct_score(f, COMPS)
f["P2"] = pct_score(f, {c: COMPS[c] for c in keep})
f["P3"] = pct_score(f, {c: COMPS[c] for c in keep}, wts)

CANDS = {"A0 基线(二值累加)": "A0", "P1 分位等权(5项)": "P1",
         f"P2 分位减法({len(keep)}项)": "P2", "P3 分位IC加权": "P3"}

say("\n### 2.1 test 段 5档封板率(Forward Return 分档)")
say("\n| 方案 | 档1 | 档2 | 档3 | 档4 | 档5 | 顶-底spread | lift | ρ |")
say("|---|---|---|---|---|---|---|---|---|")
res = {}
for lab, col in CANDS.items():
    t = f[f["split"] == "test"].dropna(subset=[col, "y"])
    if col == "A0":
        t = t.assign(qb=t["A0"])
    else:
        t = t.assign(qb=quintile(t, col, "y"))
    t = t.dropna(subset=["qb"])
    per = t.groupby("qb", observed=True)["y"].mean()
    dist = t["qb"].value_counts(normalize=True).sort_index()
    spread = (per.loc[per.index.max()] - per.loc[per.index.min()]) * 100
    lift = per.loc[per.index.max()] / max(per.loc[per.index.min()], 1e-9)
    rho = float(pd.Series(range(len(per))).corr(
        per.reset_index(drop=True), method="spearman"))
    res[lab] = {"per": per, "dist": dist, "spread": spread,
                "lift": lift, "rho": rho,
                "gates": four_gate(per, dist, lab)}
    say(f"| {lab} | " + " | ".join(f"{v:.1%}" for v in per.values)
        + f" | {spread:+.1f}pp | {lift:.2f}x | {rho:+.2f} |")

say("\n### 2.2 四关体检(test段)")
for lab, r in res.items():
    ok = all(x[1] for x in r["gates"])
    say(f"\n**{lab}** — {'✅ 过闸' if ok else '❌ 不过闸(' + ','.join(x[0].split()[0] for x in r['gates'] if not x[1]) + ')'}")
    for name, passed, val in r["gates"]:
        say(f"  - {'✅' if passed else '❌'} {name} = {val}")

# ================= 3. 三段市况复核 =================
say("\n## 3. 三段市况复核(test段, 顶档-底档封板率spread)")
say("\n| 方案 | 偏多 | 震荡 | 偏空 | 三段同向 |")
say("|---|---|---|---|---|")
regime_ok = {}
for lab, col in CANDS.items():
    t = f[f["split"] == "test"].dropna(subset=[col, "y"])
    t = t.assign(qb=t["A0"] if col == "A0" else quintile(t, col, "y"))
    t = t.dropna(subset=["qb"])
    row, sps = [], []
    for rg in ["偏多", "震荡", "偏空"]:
        g = t[t["regime"] == rg]
        per = g.groupby("qb", observed=True)["y"].mean()
        if len(per) < 2:
            row.append("-")
            sps.append(np.nan)
            continue
        sp = (per.loc[per.index.max()] - per.loc[per.index.min()]) * 100
        row.append(f"{sp:+.1f}pp(n={len(g)})")
        sps.append(sp)
    same = all(s > 0 for s in sps if not np.isnan(s)) and \
        not any(np.isnan(s) for s in sps)
    regime_ok[lab] = same
    say(f"| {lab} | " + " | ".join(row) + f" | {'✅' if same else '❌'} |")

# ================= 4. walk-forward =================
say("\n## 4. walk-forward(train定 → test复核)")
say("\n> P1/P2 的分位是当日横截面自适应, 无参数可过拟合;")
say("> P3 的权重在 train 段由 rankIC 定, 此处检验其 test 段是否保持。")
say("\n| 方案 | train spread | test spread | 衰减 | train ρ | test ρ |")
say("|---|---|---|---|---|---|")
for lab, col in CANDS.items():
    row = []
    for sp in ["train", "test"]:
        t = f[f["split"] == sp].dropna(subset=[col, "y"])
        t = t.assign(qb=t["A0"] if col == "A0" else quintile(t, col, "y"))
        t = t.dropna(subset=["qb"])
        per = t.groupby("qb", observed=True)["y"].mean()
        row.append(((per.loc[per.index.max()] - per.loc[per.index.min()]) * 100,
                    float(pd.Series(range(len(per))).corr(
                        per.reset_index(drop=True), method="spearman"))))
    dec = row[1][0] - row[0][0]
    say(f"| {lab} | {row[0][0]:+.1f}pp | {row[1][0]:+.1f}pp | {dec:+.1f}pp "
        f"| {row[0][1]:+.2f} | {row[1][1]:+.2f} |")

# ================= 5. 次日胜率维度复核(应反向) =================
say("\n## 5. 次日胜率维度复核(研究32 判反向, 此处确认不被分位化救回)")
say("\n| 方案 | test 档1→档5 次日胜率 | spread | ρ |")
say("|---|---|---|---|")
for lab, col in CANDS.items():
    t = f[f["split"] == "test"].dropna(subset=[col, "next_win"])
    t = t.assign(qb=t["A0"] if col == "A0" else quintile(t, col, "next_win"))
    t = t.dropna(subset=["qb"])
    per = t.groupby("qb", observed=True)["next_win"].mean()
    sp = (per.loc[per.index.max()] - per.loc[per.index.min()]) * 100
    rho = float(pd.Series(range(len(per))).corr(
        per.reset_index(drop=True), method="spearman"))
    say(f"| {lab} | " + " → ".join(f"{v:.1%}" for v in per.values)
        + f" | {sp:+.1f}pp | {rho:+.2f} |")

# ================= 6. 结论 =================
say("\n## 6. 结论")
best = max(res.items(), key=lambda kv: (kv[1]["spread"]
                                        if all(x[1] for x in kv[1]["gates"])
                                        and regime_ok.get(kv[0]) else -1e9))
say(f"\n**最优方案: {best[0]}** (test spread {best[1]['spread']:+.1f}pp, "
    f"lift {best[1]['lift']:.2f}x, ρ={best[1]['rho']:+.2f})")
say("\n判据: 四关全过 且 三段市况同向 且 test spread 最大。")
say("\n落地纪律:")
say("- 分位化只能用于【封板率排序】, 不得用于次日质量/仓位分层")
say("  (次日胜率维度所有候选均反向, 分位化救不回来);")
say("- 分位必须按【当日横截面】算, 跨日分位引入前视;")
say("- 若采纳, 须先在 core/longtou.py 增并列字段(不覆盖冻结的")
say("  qscore/sscore), 走影子输出观察 ≥5 个交易日再考虑替换;")
say("- 全市场人群下仍受目标变量定义域限制: 99% 当日不触板,")
say("  封板率排序的实际可用域仍是触板候选池。")

(OUT / "33_pct_factor.md").write_text("\n".join(R), encoding="utf-8")
say("\n报告已写入 research/out/33_pct_factor.md")
