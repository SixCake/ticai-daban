# -*- coding: utf-8 -*-
"""研究11: 提前抓涨停 — 触+2%决策时刻多日OOS(量×时序×题材×板型)

研究10发现雷达日志pct≥3门槛造成启动前盲区(已修, 明日积累)。
本研究不等日志: 用QMT 1m历史重建10个交易日的"+2%决策时刻",
全部特征当时可见, 回答: 启动初期(+2%)能否提前分离最终涨停者?

特征(四维):
  A 时序形态: r3/r5/r10(决策前3/5/10min涨幅)、加速度、凹凸性、
              轨迹波动、回撤深度
  B 量能: vr2(决策时刻量比)、量能趋势(近3min/前3min)
  C 题材共振: ±3min内同概念触+2%家数(con2stock静态成分)
  D 板型/时段: cm20、决策分钟
标签: 当日收盘是否封板(1m high/close 判定)
输出: research/out/11_early_oos.md + parquet
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.attribute import load_con2stock  # noqa: E402

OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()
sys.path.insert(0, str(BIGQMT_SRC))
os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
from bigqmt_signal_trader.xtquant_compat import configure, xtdata  # noqa: E402
configure(redis_config={"formula_server": {"failure_cooldown_seconds": 5}})

ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
DAYS = sorted(ev["trade_date"].unique())[-10:]
stock2con = {}
for k, cs in load_con2stock().items():
    for c in cs:
        stock2con.setdefault(c, set()).add(k)


def day_1m(codes, day):
    out = {}
    for i in range(0, len(codes), 80):
        try:
            res = xtdata.get_market_data_ex(
                field_list=["high", "close", "volume"],
                stock_list=codes[i:i + 80], period="1m",
                start_time=day + "091500", end_time=day + "150500",
                dividend_type="none", chunk_size=0, timeout_seconds=30)
            out.update(res or {})
        except Exception as e:
            print(f"1m失败 {day} {i}: {e}", flush=True)
    return out


def day_1d(codes, day, count=6):
    out = {}
    for i in range(0, len(codes), 400):
        try:
            res = xtdata.get_market_data_ex(
                field_list=["close", "volume"],
                stock_list=codes[i:i + 400], period="1d", end_time=day,
                count=count, dividend_type="none", chunk_size=0,
                timeout_seconds=30)
            out.update(res or {})
        except Exception as e:
            print(f"1d失败 {day} {i}: {e}", flush=True)
    return out


rows = []
for day in DAYS:
    t0 = time.time()
    uni = sorted({c for cs in load_con2stock().values() for c in cs
                  if c.endswith((".SH", ".SZ"))
                  and c[:2] in ("60", "68", "00", "30")})
    d1 = day_1d(uni, day)
    cand = {}
    for c, dfd in d1.items():
        try:
            if dfd is None or len(dfd) < 2 \
                    or str(dfd.index[-1])[:8] != day:
                continue
            pre = float(dfd["close"].iloc[-2])
            close = float(dfd["close"].iloc[-1])
            if pre <= 0 or close <= 0:
                continue
            cand[c] = {"pre": pre,
                       "close_pct": (close / pre - 1) * 100,
                       "avg5v": float(dfd["volume"].iloc[:-1].tail(5).mean())}
        except Exception:
            continue
    zt = ev[(ev["trade_date"] == day) & ~ev["is_yizi"] & ~ev["is_st"]]
    pos = {r.ts_code for r in zt.itertuples()}
    neg = set()
    for c, v in cand.items():
        r = 0.20 if c[:2] in ("30", "68") else 0.10
        lo, hi = (5.0, r * 100 - 0.3) if r == 0.10 else (8.0, r * 100 - 0.3)
        if lo <= v["close_pct"] < hi and c not in pos:
            neg.add(c)
    codes = sorted(pos | neg)
    m1 = day_1m(codes, day)
    touches = []
    for c in codes:
        df = m1.get(c)
        if df is None or len(df) < 15:
            continue
        df = df[[str(ix)[:8] == day for ix in df.index]]
        df = df[df["volume"] > 0].rename_axis("tm").reset_index(drop=False)
        df = df.reset_index(drop=True)
        if len(df) < 15:
            continue
        v = cand.get(c)
        if not v:
            continue
        thr2 = v["pre"] * 1.02
        hit = df["high"] >= thr2
        if not hit.any():
            continue
        j = int(hit.values.argmax())
        if j < 11:
            continue
        pct_j = (float(df["close"].iloc[j]) / v["pre"] - 1) * 100
        pct_at = lambda k: ((float(df["close"].iloc[j - k]) / v["pre"] - 1)
                            * 100 if j - k >= 0 else pct_j)
        p1, p3, p5, p10 = pct_at(1), pct_at(3), pct_at(5), pct_at(10)
        accel = (pct_j - p1) - (p1 - p3)
        win = np.array([(float(df["close"].iloc[i]) / v["pre"] - 1) * 100
                        for i in range(max(0, j - 10), j + 1)])
        pathvol = float(np.diff(win).std()) if len(win) > 2 else 0.0
        cummax = np.maximum.accumulate(win)
        dd = float((cummax - win).max())
        half = pct_at(5)
        convex = (pct_j - half) - (half - p10)
        emin = max(j + 1, 1)
        vr2 = (df["volume"].iloc[:j + 1].sum() / emin) / (v["avg5v"] / 240) \
            if v["avg5v"] > 0 else np.nan
        v_recent = float(df["volume"].iloc[max(0, j - 2):j + 1].sum())
        v_prior = float(df["volume"].iloc[max(0, j - 5):max(0, j - 2)].sum())
        vtrend = v_recent / v_prior if v_prior > 0 else np.nan
        lp = round(v["pre"] * (1 + (0.20 if c[:2] in ("30", "68")
                                    else 0.10)), 2)
        sealed = float(df["close"].iloc[-1]) >= lp * 0.995
        hhmm = int(str(df["tm"].iloc[j])[8:10]) * 60 \
            + int(str(df["tm"].iloc[j])[10:12])
        touches.append((j, c))
        rows.append({
            "date": day, "ts_code": c, "pos": c in pos, "y": int(sealed),
            "cm20": int(c[:2] in ("30", "68")), "t2": hhmm, "pct2": pct_j,
            "pre": v["pre"], "entry": float(df["close"].iloc[j]),
            "r3": pct_j - p3, "r5": pct_j - p5, "r10": pct_j - p10,
            "accel": accel, "pathvol": pathvol, "drawdown": dd,
            "convex": convex, "vr2": vr2, "vtrend": vtrend})
    # C 题材共振: ±3min内同概念触+2%家数
    touch_min = {c: j for j, c in touches}
    cons_at = {}
    for j, c in touches:
        cons_at[c] = stock2con.get(c, set())
    for r in rows[-len(touches):]:
        c = r["ts_code"]
        n = 0
        for j2, c2 in touches:
            if c2 == c or abs(j2 - touch_min[c]) > 3:
                continue
            if cons_at[c] & cons_at[c2]:
                n += 1
        r["co_con"] = n
    say(f"进度 {day}: 触+2%样本累计{len(rows)} 耗时{time.time()-t0:.0f}s")

df = pd.DataFrame(rows)
df.to_parquet(OUT / "11_early_oos.parquet", index=False)
say(f"\n# 研究11: 提前抓涨停 触+2%决策 ({DAYS[0]}~{DAYS[-1]})")
say(f"\n样本: {len(df)} = 涨停股{int(df['pos'].sum())} "
    f"负例{int((~df['pos']).sum())}; 触+2%后最终封板率 {df['y'].mean():.0%}")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

FAMS = {
    "A时序": ["r3", "r5", "r10", "accel", "pathvol", "drawdown", "convex"],
    "B量能": ["vr2", "vtrend"],
    "C题材": ["co_con"],
    "D结构": ["cm20", "t2"],
}
FEATS = [f for fs in FAMS.values() for f in fs]
X = df[FEATS].fillna(0).values
y = df["y"].values

sc = StandardScaler()
Xs = sc.fit_transform(X)
lr = LogisticRegression(max_iter=3000, C=0.2,
                        class_weight="balanced").fit(Xs, y)
say("\n## L1 logistic系数(balance)")
say("| 特征 | 族 | 系数 | 含义 |")
say("|---|---|---|---|")
fam_of = {f: k for k, fs in FAMS.items() for f in fs}
for f, c in sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1])):
    if abs(c) >= 0.08:
        say(f"| {f} | {fam_of[f]} | {c:+.2f} | "
            f"{'利封板' if c > 0 else '利回落'} |")

say("\n## L2 族消融")
say("| 特征集 | 准确率 | 召回 | 精准 |")
say("|---|---|---|---|")


def evalfam(Xf):
    scf = StandardScaler()
    Xfs = scf.fit_transform(Xf)
    m = LogisticRegression(max_iter=3000, C=0.2,
                           class_weight="balanced").fit(Xfs, y)
    p = m.predict(Xfs)
    return (m.score(Xfs, y), p[y == 1].mean() if y.sum() else 0,
            y[p == 1].mean() if p.sum() else 0)


for fam, fs in FAMS.items():
    acc, rec, prec = evalfam(df[fs].fillna(0).values)
    say(f"| {fam} | {acc:.2f} | {rec:.0%} | {prec:.0%} |")
acc, rec, prec = evalfam(X)
say(f"| 全量 | {acc:.2f} | {rec:.0%} | {prec:.0%} |")

say("\n## L3 决策树交互(depth=3)")
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
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


say("| 叶子 | n | 封板率 | 路径条件 |")
say("|---|---|---|---|")
for lf in sorted(set(leaves)):
    mask = leaves == lf
    say(f"| {lf} | {int(mask.sum())} | {y[mask].mean():.0%} "
        f"| {' & '.join(path_rules(X[mask][0]))} |")

# 分桶直观验证
say("\n## L4 关键因子分桶(直观验证)")


def bucket(name, factor, scope=None, bins=None):
    d = (scope if scope is not None else df).dropna(subset=[factor])
    if bins is None:
        try:
            d = d.assign(bin=pd.qcut(d[factor], 4, duplicates="drop"))
        except ValueError:
            return
    else:
        d = d.assign(bin=pd.cut(d[factor], bins))
    g = d.groupby("bin", observed=True).agg(n=("y", "size"),
                                            zt=("y", "mean"))
    say(f"\n{name}:")
    say("| 桶 | n | 封板率 |")
    say("|---|---|---|")
    for b, r in g.iterrows():
        say(f"| {b} | {int(r['n'])} | {r['zt']:.0%} |")


bucket("A 加速度accel(近1min增量-前1min)", "accel")
bucket("B 量能趋势vtrend(近3min/前3min)", "vtrend")
bucket("B 量比vr2", "vr2")
bucket("C 题材共振co_con", "co_con", bins=[-1, 0, 2, 5, 99])
bucket("A 回撤drawdown", "drawdown")

# ---------- 规则稳定性与可交易性 ----------
say("\n## L5 树规则分日稳定性")
rules = {
    "R1 r3>1.2 & vr2>1.1(主力规则)": (df["r3"] > 1.2) & (df["vr2"] > 1.1)
                                    & (df["r3"] <= 4.8),
    "R2 r3>4.8(暴拉)": df["r3"] > 4.8,
    "R3 r3≤1.2 & accel>0.3 & 10cm(蓄势加速)": (df["r3"] <= 1.2)
        & (df["accel"] > 0.3) & (df["cm20"] == 0),
}
say("| 规则 | 全期封板率(n) | 跑赢基准31%的天数 |")
say("|---|---|---|")
for name, cond in rules.items():
    sub = df[cond.fillna(False)]
    win = 0
    nd = 0
    for d in DAYS:
        sd = df[(df["date"] == d)]
        hd = sd[cond.fillna(False)]
        if len(hd) >= 3:
            nd += 1
            win += 1 if hd["y"].mean() > sd["y"].mean() else 0
    say(f"| {name} | {sub['y'].mean():.0%}(n={len(sub)}) | {win}/{nd} |")

say("\n## L6 +2%入场次日收益(可交易性)")
say("入场价=触+2%分钟bar收盘; 次日收益=次日收盘/入场-1")
all_days = sorted(set(ev["trade_date"].unique()) | {"20260826"})
next_of = dict(zip(all_days[:-1], all_days[1:]))
next_close_map = {}
for day in DAYS:
    nd = next_of.get(day)
    if not nd:
        continue
    codes_d = sorted(df[df["date"] == day]["ts_code"])
    res = day_1d(codes_d, nd, count=2)
    for c, d2 in res.items():
        try:
            for ix, cl in zip(d2.index, d2["close"]):
                if str(ix)[:8] == nd and float(cl) > 0:
                    next_close_map[(day, c)] = float(cl)
        except Exception:
            continue
df["next_ret"] = [
    (next_close_map[(r.date, r.ts_code)] / r.entry - 1) * 100
    if (r.date, r.ts_code) in next_close_map else np.nan
    for r in df.itertuples()]
say("| 规则 | n | 封板率 | 当日收益% | 次日收益% | 次日胜率 |")
say("|---|---|---|---|---|---|")
df["day_ret"] = np.where(df["y"] == 1,
                          (df["pre"] * np.where(df["cm20"] == 1, 1.20, 1.10)
                           / df["entry"] - 1) * 100,
                          np.nan)   # 封板者当日收在涨停价(近似)
for name, cond in [("全体触+2%",
                    pd.Series(True, index=df.index)), *rules.items()]:
    sub = df[cond.fillna(False)]
    nr = sub[sub["next_ret"].notna()]["next_ret"]
    wr = (nr > 0).mean() if len(nr) else float("nan")
    dr = sub[sub["day_ret"].notna()]["day_ret"]
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {dr.median():.2f}(封板者) | {nr.median():.2f}(n={len(nr)}) "
        f"| {wr:.0%} |")

report = "\n".join(R)
(OUT / "11_early_oos.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/11_early_oos.md")
