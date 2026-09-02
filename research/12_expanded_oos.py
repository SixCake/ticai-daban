# -*- coding: utf-8 -*-
"""研究12: 提前抓涨停因子 全窗口扩展验证(20250901-20260825, 238个交易日)

研究11在10日窗口发现 +2% 决策规则(近3min涨幅×量比×加速度×板型)。
本研究将验证集扩大24倍, 并按方法论要求做行情分段(偏多/偏空/震荡)检验。

新增:
  - 断点续跑: 每10日checkpoint落盘, 中断可续
  - 行情分段: 上证指数 MA20 位置+斜率 → 偏多/偏空/震荡, 规则分段稳定性
  - 月度稳定性表
数据: QMT FormulaServer 1m历史(2025-07-21起; 假bar volume=0 已过滤)
输出: research/out/12_expanded_oos.md + parquet
"""
import json
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
CKPT = OUT / "12_checkpoint.parquet"
START = "20250901"
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
ALL_DAYS = sorted(ev["trade_date"].unique())
DAYS = [d for d in ALL_DAYS if d >= START]
stock2con = {}
for k, cs in load_con2stock().items():
    for c in cs:
        stock2con.setdefault(c, set()).add(k)
UNI = sorted({c for cs in load_con2stock().values() for c in cs
              if c.endswith((".SH", ".SZ"))
              and c[:2] in ("60", "68", "00", "30")})


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


def build_day(day: str) -> list:
    rows = []
    d1 = day_1d(UNI, day)
    cand = {}
    for c, dfd in d1.items():
        try:
            if dfd is None or len(dfd) < 2 or str(dfd.index[-1])[:8] != day:
                continue
            pre = float(dfd["close"].iloc[-2])
            close = float(dfd["close"].iloc[-1])
            if pre <= 0 or close <= 0:
                continue
            cand[c] = {"pre": pre, "close_pct": (close / pre - 1) * 100,
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
    # 题材共振: ±3min内同概念触+2%家数
    touch_min = {c: j for j, c in touches}
    cons_at = {c: stock2con.get(c, set()) for j, c in touches}
    for r in rows:
        c = r["ts_code"]
        n = sum(1 for j2, c2 in touches
                if c2 != c and abs(j2 - touch_min[c]) <= 3
                and cons_at[c] & cons_at[c2])
        r["co_con"] = n
    return rows


# ---------- 主循环(断点续跑) ----------
done_days, rows = set(), []
if CKPT.exists():
    ck = pd.read_parquet(CKPT)
    done_days = set(ck["date"].unique())
    rows = ck.to_dict("records")
    print(f"断点续跑: 已完成{len(done_days)}天 {len(rows)}样本", flush=True)

t_start = time.time()
for di, day in enumerate(DAYS):
    if day in done_days:
        continue
    t0 = time.time()
    try:
        new = build_day(day)
    except Exception as e:
        print(f"{day} 构建失败: {e}", flush=True)
        continue
    rows.extend(new)
    done_days.add(day)
    if len(done_days) % 10 == 0:
        pd.DataFrame(rows).to_parquet(CKPT, index=False)
        print(f"进度 {day} {len(done_days)}/{len(DAYS)}天 "
              f"样本{len(rows)} 本日{time.time()-t0:.0f}s "
              f"累计{(time.time()-t_start)/60:.0f}min", flush=True)
    else:
        print(f"进度 {day} 样本{len(rows)} {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(OUT / "12_expanded_oos.parquet", index=False)
if CKPT.exists():
    CKPT.unlink()

# ---------- 行情分段 ----------
idx = xtdata.get_market_data_ex(
    field_list=["close"], stock_list=["000001.SH"], period="1d",
    start_time="20250701", end_time="20260825",
    dividend_type="none", chunk_size=0).get("000001.SH")
regime_by = {}
if idx is not None and len(idx) > 25:
    closes = pd.Series(idx["close"].values,
                       index=[str(ix)[:8] for ix in idx.index])
    ma20 = closes.rolling(20).mean()
    idx_list = list(closes.index)
    for pos, d in enumerate(idx_list):
        if pd.isna(ma20.iloc[pos]):
            continue
        above = closes.iloc[pos] > ma20.iloc[pos]
        slope = ma20.iloc[pos] - ma20.iloc[max(0, pos - 5)]
        if above and slope > 0:
            regime_by[d] = "偏多"
        elif (not above) and slope < 0:
            regime_by[d] = "偏空"
        else:
            regime_by[d] = "震荡"
df["regime"] = df["date"].map(regime_by).fillna("震荡")

say(f"# 研究12: 提前抓涨停 全窗口扩展验证 ({DAYS[0]}~{DAYS[-1]}, "
    f"{len(done_days)}个交易日)")
say(f"\n样本: {len(df)} = 涨停股{int(df['pos'].sum())} "
    f"负例{int((~df['pos']).sum())}; 触+2%后最终封板率 {df['y'].mean():.0%}")
say(f"行情分段天数: " + str(df.groupby('regime')['date'].nunique().to_dict()))

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
say("\n## L1 logistic系数(全窗口)")
say("| 特征 | 系数 | 含义 |")
say("|---|---|---|")
fam_of = {f: k for k, fs in FAMS.items() for f in fs}
for f, c in sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1])):
    if abs(c) >= 0.05:
        say(f"| {f} ({fam_of[f]}) | {c:+.2f} | "
            f"{'利封板' if c > 0 else '利回落'} |")

say("\n## L2 决策树交互(depth=3, 全窗口)")
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50,
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

# ---------- 规则验证(研究11的规则原样套用) ----------
RULES = {
    "基准(全体触+2%)": pd.Series(True, index=df.index),
    "R1 r3>1.2 & vr2>1.1 & r3≤4.8": (df["r3"] > 1.2) & (df["vr2"] > 1.1)
                                     & (df["r3"] <= 4.8),
    "R2 r3>4.8(暴拉)": df["r3"] > 4.8,
    "R3 r3≤1.2 & accel>0.3 & 10cm": (df["r3"] <= 1.2)
        & (df["accel"] > 0.3) & (df["cm20"] == 0),
}

say("\n## L3 规则全窗口表现 + 分日稳定性")
say("| 规则 | n | 封板率 | lift | 跑赢当日基准天数 |")
say("|---|---|---|---|---|")
base_rule = RULES["基准(全体触+2%)"]
for name, cond in RULES.items():
    sub = df[cond.fillna(False)]
    if name.startswith("基准"):
        say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} | 1.00x | - |")
        continue
    win, nd = 0, 0
    for d, g in df.groupby("date"):
        h = g[cond.reindex(g.index).fillna(False)]
        if len(h) >= 3:
            nd += 1
            win += 1 if h["y"].mean() > g["y"].mean() else 0
    lift = sub["y"].mean() / max(df["y"].mean(), 1e-9)
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} | {lift:.2f}x "
        f"| {win}/{nd} |")

say("\n## L4 行情分段检验(偏多/偏空/震荡)")
say("| 规则 | 偏多封板率(n) | 偏空封板率(n) | 震荡封板率(n) |")
say("|---|---|---|---|")
for name, cond in RULES.items():
    cells = []
    for rg in ["偏多", "偏空", "震荡"]:
        sub = df[(df["regime"] == rg) & cond.fillna(False)]
        cells.append(f"{sub['y'].mean():.0%}({len(sub)})" if len(sub) else "-")
    say(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")

say("\n## L5 月度稳定性(R1封板率 vs 当月基准)")
say("| 月份 | 基准 | R1 | lift | R1样本 |")
say("|---|---|---|---|---|")
df["month"] = df["date"].str[:6]
r1 = RULES["R1 r3>1.2 & vr2>1.1 & r3≤4.8"]
for m, g in df.groupby("month"):
    h = g[r1.reindex(g.index).fillna(False)]
    if len(h) < 5:
        continue
    say(f"| {m} | {g['y'].mean():.0%} | {h['y'].mean():.0%} "
        f"| {h['y'].mean()/max(g['y'].mean(),1e-9):.2f}x | {len(h)} |")

# ---------- 次日收益 ----------
say("\n## L6 +2%入场次日收益(全窗口)")
next_of = dict(zip(ALL_DAYS[:-1], ALL_DAYS[1:]))
next_of["20260825"] = "20260826"
next_close_map = {}
for day in sorted(df["date"].unique()):
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
say("| 规则 | n | 封板率 | 次日收益中位% | 次日胜率 |")
say("|---|---|---|---|---|")
for name, cond in RULES.items():
    sub = df[cond.fillna(False)]
    nr = sub["next_ret"].dropna()
    wr = (nr > 0).mean() if len(nr) else float("nan")
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {nr.median():.2f}(n={len(nr)}) | {wr:.0%} |")

df.to_parquet(OUT / "12_expanded_oos.parquet", index=False)
report = "\n".join(R)
(OUT / "12_expanded_oos.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/12_expanded_oos.md")
