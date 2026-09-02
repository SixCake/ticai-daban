# -*- coding: utf-8 -*-
"""研究21: 分钟级回测(信号→执行→持仓→出场 全链模拟)

协议:
  - 决策无未来信息: G组第4根bar(09:34可见)决策; L组触+1%的bar决策;
    特征只用决策bar及之前
  - 入场: 市价=T+1bar收盘×1.001滑点; 挂单=决策价-depth, 之后10根bar内
    low触及即成交(分钟路径真实判定), 不触及放弃
  - 仓位: 最多3仓, 每仓1/3当前权益, 当日卖出不再买入(保守)
  - 出场: 次日收盘; V3加盘中回落5%止损(参考价=max(入场,日内最高))
  - 规则: S3稳封相/剧震(gap≤5.2)/L颠簸高; 接力×稳封相加权
  - 候选方案并行(方法论): V0无规则基准 / V1市价 / V2挂单等回踩 /
    V3挂单+止损
输出: research/out/21_minute_backtest.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
CACHE1M = OUT / "1m_cache"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ev_set = set(zip(ev["trade_date"], ev["ts_code"]))
df12 = pd.read_parquet(OUT / "12_expanded_oos.parquet")
regime_by = df12.groupby("date")["regime"].first().to_dict()
DAYS = sorted({f.stem.replace("1m_", "") for f in CACHE1M.glob("1m_*.parquet")})

day_cache = {}
_xt = None
close_memo = {}


def _xtdata():
    global _xt
    if _xt is None:
        from bigqmt_signal_trader.xtquant_compat import configure, xtdata
        configure(redis_config={"formula_server": {
            "failure_cooldown_seconds": 5}})
        _xt = xtdata
    return _xt


def get_close_px(code, day):
    """某票某日收盘价: 1m缓存优先, 缺失走QMT 1d补(带memo)"""
    bars = load_day(day).get(code)
    if bars is not None and len(bars):
        return float(bars["close"].iloc[-1])
    key = (code, day)
    if key in close_memo:
        return close_memo[key]
    px = None
    try:
        res = _xtdata().get_market_data_ex(
            field_list=["close"], stock_list=[code], period="1d",
            end_time=day, count=1, dividend_type="none",
            chunk_size=0, timeout_seconds=10)
        d = res.get(code)
        if d is not None and len(d):
            px = float(d["close"].iloc[-1])
    except Exception:
        px = None
    close_memo[key] = px
    return px


def load_day(day):
    if day in day_cache:
        return day_cache[day]
    d = {}
    try:
        ck = pd.read_parquet(CACHE1M / f"1m_{day}.parquet")
        for c, g in ck.groupby("code"):
            d[c] = g.reset_index(drop=True)
    except Exception:
        pass
    day_cache[day] = d
    return d


def limit_px_of(code, pre):
    r = 0.20 if code[:2] in ("30", "68") else 0.10
    return round(pre * (1 + r), 2)


def gen_signals(day, prev_day, all_touches=False):
    """当日全部信号(决策bar/入场bar/规则/决策价), 无未来信息。
    all_touches=True 时不做规则过滤(V0基准: 所有高开/触板)"""
    sigs = []
    for code, bars in load_day(day).items():
        if code[:2] in ("30", "68") or len(bars) < 15:
            continue
        pb = load_day(prev_day).get(code) if prev_day else None
        if pb is None or len(pb) < 10:
            continue
        pre = float(pb["close"].iloc[-1])
        if pre <= 0:
            continue
        close = bars["close"].values.astype(float)
        high = bars["high"].values.astype(float)
        low = bars["low"].values.astype(float)
        pct = (close / pre - 1) * 100
        lp = limit_px_of(code, pre)
        y_zt = int((prev_day, code) in ev_set)
        # ---- G组: 首bar≥+1%, 第4根bar决策 ----
        if pct[0] >= 1.0 and len(bars) >= 5:
            gap = pct[3]
            hi3 = float(high[:4].max()) / pre * 100
            lo3 = float(low[:4].min()) / pre * 100
            odip = hi3 - pct[3]
            amp3 = hi3 - lo3
            ph, pl = float(pb["high"].max()), float(pb["low"].min())
            pc = float(pb["close"].iloc[-1])
            y_cpos = (pc - pl) / (ph - pl) if ph > pl else 0.5
            why = None
            if gap > 5.2 and odip <= 0.05:
                why = "稳封相+昨收强" if y_cpos > 0.6 else "稳封相"
            elif gap <= 5.2 and amp3 > 4.3:
                why = "剧震"
            if why or all_touches:
                sigs.append({"code": code, "di": 3, "ei": 4,
                             "rule": why or "高开任意", "dprice": float(close[3]),
                             "lp": lp, "y_zt": y_zt, "cohort": "G"})
            continue
        # ---- L组: 触+1%首现(j≥6), 触板bar决策 ----
        hits = np.where(high >= pre * 1.01)[0]
        if len(hits) == 0:
            continue
        j = int(hits[0])
        if j < 6 or j + 1 >= len(bars):
            continue
        seg = pct[max(0, j - 10):j + 1]
        if len(seg) >= 8:
            diffs = np.diff(seg)
            pathvol = float(diffs.std())
            if pathvol > 0.93 or all_touches:
                sigs.append({"code": code, "di": j, "ei": j + 1,
                             "rule": "L颠簸高" if pathvol > 0.93 else "L触板任意",
                             "dprice": float(close[j]),
                             "lp": lp, "y_zt": y_zt, "cohort": "L"})
    sigs.sort(key=lambda s: s["di"])
    return sigs


def run_backtest(mode: str, use_rules: bool = True,
                 depth_g: float = 0.5, depth_l: float = 0.4,
                 stop: bool = False):
    """mode: market/limit; 返回 trades, equity曲线"""
    cash = 1_000_000.0
    MAXPOS = 3
    open_pos = []          # {code, entry, day, day_hi, bars_next}
    trades = []
    equity = {}
    for i, day in enumerate(DAYS[1:-1], start=1):
        prev, nxt = DAYS[i - 1], DAYS[i + 1]
        slots_free = MAXPOS - len(open_pos)
        # ---- 1. 处理隔夜持仓(当日盘中止损/收盘平仓) ----
        keep = []
        nb = load_day(day)
        for p in open_pos:
            bars = nb.get(p["code"])
            exit_px = None
            if bars is not None and len(bars):
                close = bars["close"].values.astype(float)
                high = bars["high"].values.astype(float)
                if stop:
                    ref = p["entry"]
                    for k in range(len(bars)):
                        ref = max(ref, float(high[k]))
                        if float(close[k]) <= ref * 0.95:
                            exit_px = float(close[k]) * 0.999
                            break
                if exit_px is None:
                    exit_px = float(close[-1]) * 0.999   # 次日收盘出场
            if exit_px is None:
                # 次日无1m缓存(非当日样本股): 1d补收盘价
                c2 = get_close_px(p["code"], day)
                exit_px = c2 * 0.999 if c2 else p["entry"]
            ret = exit_px / p["entry"] - 1
            cash += p["size"] * (1 + ret)
            trades.append({**p, "exit": exit_px, "ret": ret,
                           "exit_day": day})
        open_pos = keep
        # ---- 2. 当日新信号入场(卖出当日不再买入, 仓位用开盘时空槽) ----
        if slots_free > 0:
            sigs = gen_signals(day, prev, all_touches=not use_rules)
            for s in sigs:
                if slots_free <= 0:
                    break
                bars = load_day(day)[s["code"]]
                close = bars["close"].values.astype(float)
                low = bars["low"].values.astype(float)
                ei = s["ei"]
                if ei >= len(bars):
                    continue
                if mode == "market":
                    entry = float(close[ei]) * 1.001
                else:
                    depth = depth_g if s["cohort"] == "G" else depth_l
                    lmt = s["dprice"] * (1 - depth / 100)
                    win = range(ei, min(ei + 10, len(bars)))
                    if not any(float(low[k]) <= lmt for k in win):
                        continue                        # 未回踩, 放弃
                    entry = lmt
                equity_now = cash + sum(q["size"] for q in open_pos)
                size = min(equity_now / MAXPOS, cash)
                if size < 10000:
                    continue
                cash -= size
                open_pos.append({"code": s["code"], "entry": entry,
                                 "day": day, "size": size,
                                 "rule": s["rule"], "cohort": s["cohort"],
                                 "y_zt": s["y_zt"]})
                slots_free -= 1
        # ---- 3. 日终权益 ----
        eq = cash
        for p in open_pos:
            bars = load_day(day).get(p["code"])
            px = float(bars["close"].iloc[-1]) if bars is not None \
                and len(bars) else p["entry"]
            eq += p["size"] * px / p["entry"]
        equity[day] = eq
        if len(day_cache) > 4:          # 逐日淘汰防内存膨胀
            for old in list(day_cache)[:len(day_cache) - 4]:
                day_cache.pop(old, None)
    # 清算残余
    for p in open_pos:
        exit_px = get_close_px(p["code"], DAYS[-1])
        exit_px = exit_px * 0.999 if exit_px else p["entry"]
        ret = exit_px / p["entry"] - 1
        cash += p["size"] * (1 + ret)
        trades.append({**p, "exit": exit_px, "ret": ret,
                       "exit_day": DAYS[-1]})
    return pd.DataFrame(trades), pd.Series(equity)


def perf(eq: pd.Series, trades: pd.DataFrame, name: str):
    total = eq.iloc[-1] / eq.iloc[0] - 1
    n = len(eq)
    ann = (1 + total) ** (240 / max(n, 1)) - 1
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    wr = (trades["ret"] > 0).mean() if len(trades) else 0
    avg = trades["ret"].mean() * 100 if len(trades) else 0
    med = trades["ret"].median() * 100 if len(trades) else 0
    tpd = len(trades) / max(n, 1)
    say(f"| {name} | {total:+.1%} | {ann:+.1%} | {dd:.1%} | {wr:.0%} "
        f"| {med:+.2f}% | {tpd:.1f} | {len(trades)} |")
    return trades, eq


say("# 研究21: 分钟级回测(238日, 1m路径, 无未来信息)")
say(f"\n初始资金100万, 最多3仓×1/3, 出场=次日收盘, 滑点0.1%")

say("\n## 总体表现")
say("| 方案 | 总收益 | 年化 | 最大回撤 | 胜率 | 单笔中位 | 笔/日 | 总笔数 |")
say("|---|---|---|---|---|---|---|---|")
t0, e0 = run_backtest("market", use_rules=False)
perf(e0, t0, "V0 无规则市价(基准)")
t1, e1 = run_backtest("market")
perf(e1, t1, "V1 规则+市价")
t2, e2 = run_backtest("limit")
perf(e2, t2, "V2 规则+挂单等回踩")
t3, e3 = run_backtest("limit", stop=True)
perf(e3, t3, "V3 挂单+盘中5%止损")

say("\n## 分规则表现(V2 挂单)")
say("| 规则 | 笔数 | 胜率 | 收益中位% | 收益均值% |")
say("|---|---|---|---|---|")
for rule, g in t2.groupby("rule"):
    say(f"| {rule} | {len(g)} | {(g['ret'] > 0).mean():.0%} "
        f"| {g['ret'].median() * 100:+.2f} | {g['ret'].mean() * 100:+.2f} |")

say("\n## 三段行情分段(V2)")
t2r = t2.copy()
t2r["regime"] = t2r["day"].map(regime_by).fillna("震荡")
say("| 行情段 | 笔数 | 胜率 | 收益中位% |")
say("|---|---|---|---|")
for rg in ["偏多", "偏空", "震荡"]:
    g = t2r[t2r["regime"] == rg]
    if len(g):
        say(f"| {rg} | {len(g)} | {(g['ret'] > 0).mean():.0%} "
            f"| {g['ret'].median() * 100:+.2f} |")

say("\n## 月度收益(V2权益曲线)")
e2m = e2.copy()
e2m.index = pd.to_datetime(e2m.index)
mon = e2m.groupby(e2m.index.strftime("%Y%m")).last()
prev = 1_000_000
for m, v in mon.items():
    say(f"- {m}: {v / prev - 1:+.1%}")
    prev = v

report = "\n".join(R)
(OUT / "21_minute_backtest.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/21_minute_backtest.md")
