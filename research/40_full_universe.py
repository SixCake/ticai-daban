# -*- coding: utf-8 -*-
"""研究40: 全宇宙打板因子筛选(日线口径 · 消除选择偏见)

**为何重做**: 研究39 的宇宙是**触板股**(zt_minute 的 sealed+zb), 这条件化了
结果 —— 实盘在 4% 买入时根本不知道它会不会触板。实测偏见 8.2 倍:
  研究39 宇宙(触板股)     32,398 条票·日
  正确宇宙(盘中达≥4%)    264,970 条票·日
  研究39 封板率 63.5%  vs  真实封板率 8.2%  → 虚高 7.7 倍
所以研究39 的 EV 全部系统性高估(只统计了最终触板的票)。

**本研究口径(全部日线, 不需要分钟数据)**:
  宇宙   全A 中「盘中最高涨幅 ≥ 目标档」的票·日 —— daily_panel 的 high 判定,
         不再预筛触板股
  入场价 阈值价 + 实测滑点(研究39 用 312 条分钟样本校准: 3%档→实际4.53%,
         滑点 +1.1~1.5pp)。这是有数据支撑的假设, 不是拍脑袋。
  离场   日线近似定稿规则(core/exit_rules 的口径):
           止损5%   low ≤ entry×0.95  → 离场于 entry×0.95
           封板续持 close ≥ limit×SEAL_EPS
           未封板   离场于 close[T]
         近似之处: 不知道止损在哪一分钟触发。但因子筛选要比较的是
         **不同因子组的相对优劣**, 不是精确绝对收益, 近似足够。
  因子   core/shape.py 的 21 个(T-1 及更早日线), 严格无前视
  标签   T日离场收益 + T+1/T+2/T+3 收盘(仅封板续持有敞口)
  评估   5档分位 × 三段市况(牛/熊/震荡) × 胜率/盈亏比/期望

样本: ≥3% 392,822 条 / ≥4% 264,970 条 / ≥5% 182,099 条 / ≥6% 132,243 条
(窗口 20250101~20260610; daily_panel 全窗口 6.8 年更大)

产物: research/out/40_full_universe.md
用法: python research/40_full_universe.py [--start YYYYMMDD] [--end YYYYMMDD]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import path_of                               # noqa: E402
from core.exit_rules import pl_stats, SEAL_EPS, STOP_LOSS   # noqa: E402
from core.shape import (build_bars, build_regimes, reg_of,  # noqa: E402
                        compute_factors, CONT, BOOL, GROUP)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CACHE = Path(__file__).resolve().parent.parent / "data" / "factor" \
    / "fulluniv_panel.parquet"

# 目标涨幅档位
TARGETS = [3.0, 4.0, 5.0, 6.0]
# 实测滑点(研究39 用 312 条分钟样本校准: 阈值价 → 实际 bar close 成交均价)
SLIPPAGE = {3.0: 1.53, 4.0: 1.48, 5.0: 1.27, 6.0: 1.12}
say = print


def limit_rate(code: str) -> float:
    c = str(code)
    if c.endswith(".BJ"):
        return 0.30
    if c[:3] in ("300", "301", "302", "688", "689"):
        return 0.20
    return 0.10


def build_dataset(start: str, end: str, refresh: bool) -> pd.DataFrame:
    """全宇宙数据集: 盘中达到任一目标档的票·日 + T-1 因子 + 离场标签"""
    if CACHE.exists() and not refresh:
        d = pd.read_parquet(CACHE)
        d = d[(d["trade_date"] >= start) & (d["trade_date"] <= end)]
        say(f"载入缓存 {CACHE} → 窗口内 {len(d)} 条")
        return d
    say("加载日线面板…")
    cols = ["trade_date", "ts_code", "open", "high", "low", "close",
            "vol", "pre_close", "pct_chg"]
    p = pd.read_parquet(path_of("market.daily_panel"), columns=cols)
    p = p[(p["trade_date"] >= start) & (p["trade_date"] <= end)]
    p = p.dropna(subset=["high", "low", "close", "pre_close"])
    p = p[p["pre_close"] > 0]
    say(f"  全A {len(p)} 条票·日, {p['trade_date'].nunique()} 个交易日")

    # 涨停价近似(未做ST修正, 同 core/structure 口径)
    p["lp"] = [round(pc * (1 + limit_rate(c)), 2)
               for pc, c in zip(p["pre_close"], p["ts_code"])]
    p["hi_pct"] = (p["high"] / p["pre_close"] - 1) * 100
    p["lo_pct"] = (p["low"] / p["pre_close"] - 1) * 100
    p["cl_pct"] = (p["close"] / p["pre_close"] - 1) * 100
    p["op_pct"] = (p["open"] / p["pre_close"] - 1) * 100
    p["touched"] = p["high"] >= p["lp"] * 0.999
    p["sealed"] = p["close"] >= p["lp"] * SEAL_EPS

    # 宇宙: 盘中最高涨幅 ≥ 最低目标档(3%) —— 各档在其内再筛
    u = p[p["hi_pct"] >= min(TARGETS)].copy()
    say(f"  宇宙(盘中达≥{min(TARGETS)}%): {len(u)} 条票·日, "
        f"{u['ts_code'].nunique()} 只票")

    say("构建 per-code bar 数组(算 T-1 因子)…")
    bars = build_bars(p)

    say("计算 T-1 因子…")
    recs, n_skip = [], 0
    for r in u.itertuples():
        b = bars.get(r.ts_code)
        if not b:
            n_skip += 1
            continue
        i = int(np.searchsorted(b["date"], r.trade_date))
        if i >= len(b["date"]) or b["date"][i] != r.trade_date:
            n_skip += 1
            continue
        fac = compute_factors(b, i)       # 只用 T-1 及更早, 无前视
        if not fac:
            n_skip += 1
            continue
        rec = {"trade_date": r.trade_date, "ts_code": r.ts_code,
               "lp": float(r.lp), "day_close": float(r.close),
               "day_low": float(r.low), "hi_pct": float(r.hi_pct),
               "op_pct": float(r.op_pct),
               "touched": bool(r.touched), "sealed": bool(r.sealed)}
        rec.update(fac)
        # T+1/T+2/T+3 收盘(仅封板续持有敞口)
        for k, key in ((1, "d1_close"), (2, "d2_close"), (3, "d3_close")):
            rec[key] = (float(b["close"][i + k])
                        if i + k < len(b["date"])
                        and not np.isnan(b["close"][i + k]) else None)
        recs.append(rec)
    d = pd.DataFrame(recs)
    say(f"  有效 {len(d)} 条(跳过 {n_skip}: 无面板/T-1不足)")

    # 各档的入场价与离场收益(日线近似)
    for tgt in TARGETS:
        slip = SLIPPAGE[tgt]
        entry_pct = tgt + slip                       # 阈值 + 实测滑点
        d[f"e{int(tgt)}_ok"] = d["hi_pct"] >= tgt
        # 入场价 = 昨收 × (1 + 入场涨幅)
        pre = d["lp"] / (1 + d["ts_code"].map(limit_rate))
        d[f"e{int(tgt)}_px"] = pre * (1 + entry_pct / 100)
        ep = d[f"e{int(tgt)}_px"]
        stop = ep * STOP_LOSS
        # 离场判定(日线近似): 止损优先, 否则收盘
        hit_stop = d[f"e{int(tgt)}_ok"] & (d["day_low"] <= stop)
        ret = pd.Series(np.nan, index=d.index)
        ret[hit_stop] = (stop[hit_stop] / ep[hit_stop] - 1) * 100
        rest = d[f"e{int(tgt)}_ok"] & ~hit_stop
        ret[rest] = (d["day_close"][rest] / ep[rest] - 1) * 100
        d[f"e{int(tgt)}_ret"] = ret
        d[f"e{int(tgt)}_why"] = np.where(hit_stop, "回落止损5%",
                                         np.where(d["sealed"], "封板续持",
                                                  "未封板收盘离场"))
    say("缓存 → %s" % CACHE)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    # 市况标签(全窗口三分位, 同研究31)
    rg = build_regimes(p)
    d["reg"] = d["trade_date"].map(lambda x: reg_of(x, rg))
    return d


def eval_entry(d: pd.DataFrame, tgt: int) -> dict:
    """单档入场机制复核"""
    ok, ret = f"e{tgt}_ok", f"e{tgt}_ret"
    sub = d[d[ok] & d[ret].notna()]
    if not len(sub):
        return {"n": 0}
    n, wr, aw, al, ratio, exp = pl_stats(list(sub[ret]))
    sealed = sub[sub["sealed"]]
    unsealed = sub[~sub["sealed"]]
    _, _, _, _, _, es = pl_stats(list(sealed[ret]))
    _, _, _, _, _, eu = pl_stats(list(unsealed[ret]))
    return {"n": len(sub),
            "seal_rate": round(len(sealed) / len(sub) * 100, 2),
            "wr": round(wr * 100, 2) if wr is not None else None,
            "ratio": round(ratio, 3) if ratio is not None else None,
            "exp": round(exp, 3) if exp is not None else None,
            "exp_sealed": round(es, 3) if es is not None else None,
            "exp_unsealed": round(eu, 3) if eu is not None else None}


def eval_factor(d: pd.DataFrame, tgt: int, fac: str, is_bool: bool) -> dict:
    """单因子 5档分位(连续)或 0/1(布尔) + 三段市况一致性"""
    ok, ret = f"e{tgt}_ok", f"e{tgt}_ret"
    sub = d[d[ok] & d[ret].notna() & d[fac].notna()]
    if len(sub) < 2000:
        return {}
    rows = []
    if is_bool:
        groups = [(0, sub[sub[fac] == 0]), (1, sub[sub[fac] == 1])]
    else:
        try:
            sub = sub.assign(q=pd.qcut(sub[fac], 5, labels=False,
                                       duplicates="drop"))
        except Exception:
            return {}
        groups = [(int(q) + 1, sub[sub["q"] == q])
                  for q in sorted(sub["q"].dropna().unique())]
    for lab, g in groups:
        if len(g) < 500:
            continue
        n, wr, aw, al, ratio, exp = pl_stats(list(g[ret]))
        rows.append((lab, n,
                     round(len(g[g["sealed"]]) / len(g) * 100, 1),
                     round(wr * 100, 1) if wr is not None else None,
                     round(ratio, 3) if ratio is not None else None,
                     round(exp, 3) if exp is not None else None))
    if len(rows) < 3:
        return {}
    # 单调性(按期望)
    exps = [r[5] for r in rows if r[5] is not None]
    ups = sum(1 for a, b in zip(exps, exps[1:]) if b > a)
    downs = sum(1 for a, b in zip(exps, exps[1:]) if b < a)
    if ups == len(exps) - 1:
        mono = "严格单调↑"
    elif downs == len(exps) - 1:
        mono = "严格单调↓"
    elif ups >= downs:
        mono = "偏↑"
    else:
        mono = "偏↓"
    # 三段市况一致性
    dirs = {}
    for rg in ("牛", "熊", "震荡"):
        g2 = sub[sub["reg"] == rg]
        if is_bool:
            e0 = pl_stats(list(g2[g2[fac] == 0][ret]))[5]
            e1 = pl_stats(list(g2[g2[fac] == 1][ret]))[5]
            dirs[rg] = ("↑" if e1 is not None and e0 is not None and e1 > e0
                        else "↓" if e1 is not None and e0 is not None
                        else "无")
        else:
            if len(g2) < 1000 or "q" not in g2:
                dirs[rg] = "无"
                continue
            e = [pl_stats(list(g2[g2["q"] == q][ret]))[5]
                 for q in sorted(g2["q"].dropna().unique())]
            e = [x for x in e if x is not None]
            if len(e) < 3:
                dirs[rg] = "无"
                continue
            u = sum(1 for a, b in zip(e, e[1:]) if b > a)
            dn = sum(1 for a, b in zip(e, e[1:]) if b < a)
            dirs[rg] = "↑" if u > dn else "↓" if dn > u else "无"
    cons = max(sum(1 for v in dirs.values() if v == "↑"),
               sum(1 for v in dirs.values() if v == "↓"))
    spread = (exps[-1] - exps[0]) if len(exps) >= 2 else None
    return {"rows": rows, "mono": mono, "cons": cons, "dirs": dirs,
            "spread": round(spread, 3) if spread is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250101")
    ap.add_argument("--end", default="20260610")
    ap.add_argument("--refresh", action="store_true", help="丢弃缓存重算")
    a = ap.parse_args()
    d = build_dataset(a.start, a.end, a.refresh)
    if not len(d):
        say("无数据")
        return
    say("评估各档入场机制…")
    entry_res = {int(t): eval_entry(d, int(t)) for t in TARGETS}
    for t, r in entry_res.items():
        if r.get("n"):
            say(f"  {t}%档: {r['n']}条 封板率{r['seal_rate']}% "
                f"胜率{r['wr']}% 盈亏比{r['ratio']} 期望{r['exp']}%")
    say("评估因子(5档分位 × 三段市况)…")
    fac_res = {}
    for fac in CONT + BOOL:
        fac_res[fac] = {int(t): eval_factor(d, int(t), fac, fac in BOOL)
                        for t in TARGETS}
    write_report(d, entry_res, fac_res, a.start, a.end)
    say(f"\n报告已写入 {OUT/'40_full_universe.md'}")


def write_report(d, entry_res, fac_res, start, end):
    L = []
    A = L.append
    A("# 研究40: 全宇宙打板因子筛选(日线口径)\n")
    A(f"- 窗口 {start}~{end}, 全A {len(d)} 条票·日, "
      f"{d['trade_date'].nunique()} 个交易日")
    A("- 宇宙: **盘中最高涨幅 ≥ 目标档**(daily_panel high 判定), "
      "**不再预筛触板股** —— 消除研究39 的选择偏见")
    A("- 入场价: 阈值 + 实测滑点(研究39 用312条分钟样本校准)")
    A("- 离场: 日线近似定稿规则(止损5% / 封板续持 / 未封板收盘)")
    A("- 因子: `core/shape.py` 21 个(T-1 日线), 严格无前视")
    A("- 评估: 5档分位 × 三段市况 × 胜率/盈亏比/期望\n")

    A("## 为何重做: 研究39 的选择偏见(8.2倍)\n")
    A("研究39 宇宙是**触板股**, 条件化了结果 —— 实盘在4%买入时不知道"
      "它会不会触板:\n")
    A("| | 票·日 | 封板率 |")
    A("|---|---|---|")
    A("| 研究39 宇宙(触板股) | 32,398 | **63.5%**(虚高) |")
    A("| 正确宇宙(盘中达≥4%) | 264,970 | **8.2%**(真实) |")
    A("")
    A("偏见 **8.2 倍**, 封板率虚高 **7.7 倍**。研究39 只统计了最终触板的票, "
      "把「涨到4%然后回落、从未触板」的票全排除 —— 而那正是实盘会买到的票。\n")

    A("## 入场价位机制复核(全宇宙, 日线近似)\n")
    A("| 目标档 | 滑点 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% | "
      "封板票期望 | 未封板票期望 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t in TARGETS:
        r = entry_res[int(t)]
        if not r.get("n"):
            continue
        A(f"| {int(t)}% | +{SLIPPAGE[t]}pp | {r['n']} | {r['seal_rate']} "
          f"| {r['wr']} | {r['ratio']} | **{r['exp']}** "
          f"| {r['exp_sealed']} | {r['exp_unsealed']} |")
    A("")

    A("## 因子筛选结果\n")
    A("判据: ① 5档分位单调 ② 三段市况一致 ≥2/3 ③ spread 幅度\n")
    A("| 因子 | 组 | 最优档 | 单调性 | 三段一致 | spread(pp) | 价位风险 |")
    A("|---|---|---|---|---|---|---|")
    cands = []
    for fac in CONT + BOOL:
        best = None
        for t in TARGETS:
            r = fac_res[fac][int(t)]
            if not r or r["cons"] < 2:
                continue
            if r["mono"] not in ("严格单调↑", "严格单调↓"):
                continue
            if best is None or abs(r["spread"] or 0) > abs(best[1]["spread"]
                                                           or 0):
                best = (int(t), r)
        if best:
            from core.shape import PRICE_RISK
            cands.append((fac, best))
            A(f"| `{fac}` | {GROUP[fac]} | {best[0]}% | {best[1]['mono']} "
              f"| {best[1]['cons']}/3 | {best[1]['spread']} "
              f"| {'**是**' if fac in PRICE_RISK else '否'} |")
    if not cands:
        A("| (无因子通过判据) | - | - | - | - | - | - |")
    A("")
    A(f"**通过 {len(cands)} / {len(CONT)+len(BOOL)} 个因子。**\n")

    A("## 因子明细(通过判据的)\n")
    for fac, (bt, r) in cands:
        A(f"### `{fac}` — {GROUP[fac]} · 最优 {bt}% 档\n")
        A("| 档 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% |")
        A("|---|---|---|---|---|---|")
        for row in r["rows"]:
            A("| " + " | ".join(str(x) if x is not None else "-"
                                for x in row) + " |")
        A(f"\n单调性 {r['mono']} · 三段一致 {r['cons']}/3 ({r['dirs']}) · "
          f"spread {r['spread']}pp\n")

    A("## 约束\n")
    A("1. **离场是日线近似** —— 不知道止损在哪一分钟触发。因子筛选比较的是"
      "**相对优劣**, 近似足够; 但要算精确绝对收益需分钟数据")
    A("2. **入场价是建模值**(阈值+实测滑点), 非逐笔观测。滑点由研究39 的312条"
      "分钟样本校准, 需周期性复核")
    A("3. 涨停价近似未做ST修正(同 core/structure 口径)")
    A("4. **只筛选不进生产** —— 通过判据的因子只进影子字段与看板展示")
    A("5. 因子定义唯一出处 `core/shape.py`")

    (OUT / "40_full_universe.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
