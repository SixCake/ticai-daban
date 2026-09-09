# -*- coding: utf-8 -*-
"""研究39: 半路板入场口径的形态因子验证(可执行 · 无前视)

**为何重做**: 研究37/38 犯了两个致命错误 ——
  ① 买入基准用 close[T](涨停价): 涨停股尾盘封死, 买单排队也不一定成交,
     那个价位买不到。数学恒等式暴露了问题: r1_open = open[T+1]/close[T]-1
     恰好等于「买入溢价」, 所谓"收益"就是为买入必须支付的溢价, 是成本。
  ② 宇宙用已涨停股: 在 T-1 无法知道 T 会涨停, 属前视。
修正后结论完全反转: 不可执行口径 M4 期望 +1.965%(最优), 可执行口径
-0.773%(最差), 7 个方案三段市况全部负期望。

**本研究的正确口径**:
  宇宙   触板股(zt_minute 的 sealed + zb 组) —— 盘中触板时已知, 非前视
  入场   **半路板**: 分钟线上首次触及目标涨幅时的 bar close 价
         —— 这是真实成交价, 且低于涨停价(未封死才买得进)
  排除   is_yizi(一字板全程贴死涨停, 买不进)
         入场 bar 已贴死涨停价(已封死, 买不进)
         当日数据不完整(盘中采集)
  因子   T-1 及更早日线算(core/shape.py), 严格无前视
  离场   core/exit_rules.simulate_exit 用**分钟线**逐点模拟定稿卖出规则
  多日   T+1/T+2/T+3 收益取 daily_panel(封板续持才有敞口)

**样本量诚实标注**: zt_minute 只有 4 个完整交易日(东财 API 仅 ~5 日深度,
历史日无法回填), 约 300 条样本。这**不足以做 5 档分位因子筛选**, 只能做
粗分组(2~3组)对比 + 入场价位机制复核。数据由 daily_update.sh 每日积累,
本研究应随数据增长周期性重跑。

产物: research/out/39_halfway_entry.md
用法: python research/39_halfway_entry.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from datastore import path_of                               # noqa: E402
from core.exit_rules import (pl_stats, simulate_exit,       # noqa: E402
                             SEAL_EPS, STOP_LOSS)
from core.shape import build_bars, compute_factors, limit_rate

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
MIN_DIR = DATA / "limitup" / "1m"

# 半路板目标涨幅档位(研究08 §4 的分档)
ENTRY_TARGETS = [3.0, 4.0, 5.0, 6.0]
say = print


def load_minute() -> pd.DataFrame:
    """载入全部分钟线, 剔除当日(盘中采集不完整)"""
    fs = sorted(glob.glob(str(MIN_DIR / "zt_minute_*.parquet")))
    if not fs:
        return pd.DataFrame()
    import datetime as _dt
    today = _dt.datetime.now().strftime("%Y%m%d")
    dfs = []
    for f in fs:
        dt = Path(f).stem.replace("zt_minute_", "")
        if dt >= today:                 # 当日盘中数据不完整, 剔除
            say(f"  剔除 {dt}(当日盘中, 数据不完整)")
            continue
        dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    d = pd.concat(dfs, ignore_index=True)
    say(f"  载入 {len(dfs)} 个交易日, {len(d)} 行分钟线, "
        f"{d.groupby('date')['ts_code'].nunique().sum()} 只票·日")
    return d


def find_entry(g: pd.DataFrame, target: float) -> tuple:
    """半路板入场: 分钟线上首次触及目标涨幅 → (入场价, 入场时刻, 可买)

    g: 单票单日分钟线, 按 t 升序, 含 open/high/low/close/limit_px
    可买判据:
      · 一字板(is_yizi) → 全程贴死涨停, 买不进
      · 入场 bar 的 close 已贴死涨停价 → 已封死, 买不进
      · 找不到触及目标涨幅的 bar → 无入场
    返回 (None, None, False) 表示不可买/无入场。
    """
    if not len(g):
        return None, None, False
    lp = float(g["limit_px"].iloc[0])
    if lp <= 0:
        return None, None, False
    # 昨收 = 涨停价 / (1+幅)
    code = str(g["ts_code"].iloc[0])
    pre = lp / (1 + limit_rate(code))
    if g["is_yizi"].iloc[0]:
        return None, None, False        # 一字板买不进
    thr = pre * (1 + target / 100)
    for r in g.itertuples():
        if r.close is None or np.isnan(r.close) or r.close <= 0:
            continue
        if r.close >= thr:
            if r.close >= lp * SEAL_EPS:
                return None, None, False    # 已封死, 买不进
            return float(r.close), str(r.t), True
    return None, None, False            # 全天未触及目标涨幅


def build_dataset() -> pd.DataFrame:
    """构建半路板数据集: 每票每日 × 每档目标涨幅"""
    m = load_minute()
    if not len(m):
        return pd.DataFrame()
    say("加载日线面板(算 T-1 形态因子 + T+N 收益)…")
    cols = ["trade_date", "ts_code", "open", "high", "low", "close",
            "vol", "pct_chg"]
    panel = pd.read_parquet(path_of("market.daily_panel"), columns=cols)
    panel = panel.dropna(subset=["close"])
    bars = build_bars(panel)

    recs = []
    for (dt, code), g in m.groupby(["date", "ts_code"], sort=False):
        g = g.sort_values("t")
        b = bars.get(code)
        if not b:
            continue
        idx = int(np.searchsorted(b["date"], dt))
        if idx >= len(b["date"]) or b["date"][idx] != dt:
            continue
        fac = compute_factors(b, idx)       # 只用 T-1 及更早, 无前视
        if not fac:
            continue
        # 分钟线转 simulate_exit 需要的 [[HHMMSS, price], ...]
        pts = [[f"{str(r.t).zfill(4)}00", float(r.close)]
               for r in g.itertuples()
               if r.close is not None and not np.isnan(r.close)
               and r.close > 0]
        # 对齐的最高价: 止损参考价随分时动态上移(无前视)。
        # **必须传**: 半路板入场与离场在同一日, 若用全天最高价则含未来
        # 信息, 且触板股全天高点≈涨停价 → 止损线高于入场价,
        # 第一根bar 就误触发止损。
        hi_pts = [[f"{str(r.t).zfill(4)}00", float(r.high)]
                  for r in g.itertuples()
                  if r.high is not None and not np.isnan(r.high)
                  and r.high > 0]
        lp = float(g["limit_px"].iloc[0])
        rec = {"date": dt, "ts_code": code,
               "name": g["name"].iloc[0],
               "grp": g["grp"].iloc[0],
               "height": int(g["height"].iloc[0] or 1),
               "limit_px": lp,
               "day_close": float(b["close"][idx]),
               "day_hi": float(b["high"][idx])}
        rec.update(fac)
        # 封板判定(分钟线): 收盘贴死涨停价
        rec["sealed"] = bool(rec["day_close"] >= lp * SEAL_EPS)
        # T+1/T+2/T+3 收盘(封板续持才有敞口)
        for k, key in ((1, "d1_close"), (2, "d2_close"), (3, "d3_close")):
            rec[key] = (float(b["close"][idx + k])
                        if idx + k < len(b["date"])
                        and not np.isnan(b["close"][idx + k]) else None)
        # 每档目标涨幅的入场与离场
        for tgt in ENTRY_TARGETS:
            px, tt, ok = find_entry(g, tgt)
            rec[f"e{int(tgt)}_px"] = px
            rec[f"e{int(tgt)}_t"] = tt
            rec[f"e{int(tgt)}_ok"] = ok
            if not ok or px is None:
                rec[f"e{int(tgt)}_ret"] = None
                rec[f"e{int(tgt)}_why"] = None
                rec[f"e{int(tgt)}_entry_pct"] = None
                continue
            pre = lp / (1 + limit_rate(code))
            rec[f"e{int(tgt)}_entry_pct"] = (px / pre - 1) * 100
            # 离场: 只用入场时刻之后的分钟线(不偷看入场前的走势),
            # 并传 hi_pts 让止损参考价动态上移(避免全天高点前视)
            cut = f"{str(tt).zfill(4)}00"
            after = [p for p in pts if p[0] >= cut]
            after_hi = [p for p in hi_pts if p[0] >= cut]
            ret, why = simulate_exit(after, px, rec["day_hi"], lp,
                                     rec["day_close"], hi_pts=after_hi)
            rec[f"e{int(tgt)}_ret"] = ret
            rec[f"e{int(tgt)}_why"] = why
        recs.append(rec)
    d = pd.DataFrame(recs)
    say(f"  有效票·日 {len(d)} 条")
    return d


def eval_entry(d: pd.DataFrame, tgt: int) -> dict:
    """单档入场评估 → 可买样本的封板率/离场收益统计"""
    col_ok = f"e{tgt}_ok"
    col_ret = f"e{tgt}_ret"
    col_pct = f"e{tgt}_entry_pct"
    sub = d[d[col_ok] & d[col_ret].notna()]
    if not len(sub):
        return {"n": 0}
    n, wr, aw, al, ratio, exp = pl_stats(list(sub[col_ret]))
    # 封板票 vs 未封板票的收益分解(研究08 的核心机制)
    sealed = sub[sub["sealed"]]
    unsealed = sub[~sub["sealed"]]
    _, _, _, _, _, exp_s = pl_stats(list(sealed[col_ret]))
    _, _, _, _, _, exp_u = pl_stats(list(unsealed[col_ret]))
    return {
        "n": len(sub),
        "n_all": int(d[col_ok].sum()),
        "entry_pct_mean": round(float(sub[col_pct].mean()), 2),
        "seal_rate": round(len(sealed) / len(sub) * 100, 2),
        "wr": round(wr * 100, 2) if wr is not None else None,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "exp": round(exp, 3) if exp is not None else None,
        "exp_sealed": round(exp_s, 3) if exp_s is not None else None,
        "exp_unsealed": round(exp_u, 3) if exp_u is not None else None,
    }


def eval_factor(d: pd.DataFrame, tgt: int, fac: str, bool_fac: bool) -> list:
    """粗分组因子对比(样本量小, 不做5档分位)"""
    col_ok, col_ret = f"e{tgt}_ok", f"e{tgt}_ret"
    sub = d[d[col_ok] & d[col_ret].notna() & d[fac].notna()]
    if len(sub) < 40:
        return []
    rows = []
    if bool_fac:
        groups = [(0, sub[sub[fac] == 0]), (1, sub[sub[fac] == 1])]
    else:
        med = sub[fac].median()
        groups = [(f"≤{med:.1f}", sub[sub[fac] <= med]),
                  (f">{med:.1f}", sub[sub[fac] > med])]
    for lab, g in groups:
        if len(g) < 15:
            continue
        n, wr, aw, al, ratio, exp = pl_stats(list(g[col_ret]))
        rows.append((lab, n,
                     round(len(g[g["sealed"]]) / len(g) * 100, 1),
                     round(wr * 100, 1) if wr is not None else None,
                     round(ratio, 3) if ratio is not None else None,
                     round(exp, 3) if exp is not None else None))
    return rows


def main():
    say("构建半路板数据集…")
    d = build_dataset()
    if not len(d):
        say("无数据, 退出")
        return
    say("评估各档入场价位…")
    entry_res = {t: eval_entry(d, int(t)) for t in ENTRY_TARGETS}
    for t, r in entry_res.items():
        if r.get("n"):
            say(f"  {t}%档: 可买{r['n']} 实际入场均价{r['entry_pct_mean']}% "
                f"封板率{r['seal_rate']}% 期望{r['exp']}%")
    say("评估形态因子(粗分组)…")
    from core.shape import CONT, BOOL, GROUP
    fac_res = {}
    for fac in CONT + BOOL:
        fac_res[fac] = {int(t): eval_factor(d, int(t), fac, fac in BOOL)
                        for t in ENTRY_TARGETS}
    write_report(d, entry_res, fac_res)
    say(f"\n报告已写入 {OUT/'39_halfway_entry.md'}")


def write_report(d, entry_res, fac_res):
    from core.shape import CONT, BOOL, GROUP
    L = []
    A = L.append
    days = sorted(d["date"].unique())
    A("# 研究39: 半路板入场口径的形态因子验证\n")
    A(f"- 宇宙: 触板股(zt_minute sealed+zb), **{len(d)}** 条票·日, "
      f"{len(days)} 个交易日({', '.join(days)})")
    A("- 入场: **半路板** —— 分钟线上首次触及目标涨幅时的 bar close 价"
      "(真实成交价, 低于涨停价)")
    A(f"- 入场档位: {'/'.join(str(int(t))+'%' for t in ENTRY_TARGETS)}")
    A("- 排除: 一字板(is_yizi) / 入场bar已贴死涨停 / 当日盘中数据不完整")
    A("- 因子: T-1 及更早日线(`core/shape.py`), 严格无前视")
    A("- 离场: `core/exit_rules.simulate_exit` 用分钟线逐点模拟定稿卖出规则")
    A("- 指标: `core/exit_rules.pl_stats`\n")

    A("## ⚠ 样本量限制(必读)\n")
    A(f"zt_minute 只有 **{len(days)} 个完整交易日**(东财 API 仅 ~5 日深度, "
      "历史日无法回填), 共 "
      f"{len(d)} 条票·日。这**不足以做 5 档分位因子筛选**, 只能做"
      "粗分组(中位数二分)对比。\n")
    A("数据由 `daily_update.sh` 每日积累(`fetch_zt_minute.py --max-days 2`), "
      "本研究应随数据增长周期性重跑。**当前所有因子结论都只是方向性提示, "
      "不得进生产。**\n")

    A("## 为何重做(研究37/38 的两个致命错误)\n")
    A("1. **买入基准用 `close[T]`(涨停价)** —— 涨停股尾盘封死, 买单排队也"
      "不一定成交, 那个价位买不到。数学恒等式: `r1_open = open[T+1]/close[T]-1`"
      " 恰好等于「买入溢价」, 所谓收益就是为买入必须支付的**成本**。")
    A("2. **宇宙用已涨停股** —— T-1 无法知道 T 会涨停, 属前视。\n")
    A("修正后结论完全反转: 不可执行口径 M4 期望 +1.965%(最优), 可执行口径 "
      "-0.773%(最差), 7 个方案三段市况全部负期望。\n")

    A("## 入场价位机制复核\n")
    A("研究08 §4 定稿: **打板 EV 由入场价位决定, 6%以上入场数学上不可行**"
      "(需81%+封板率打平)。本研究用真实分钟数据复核:\n")
    A("| 目标档 | 可买样本 | 实际入场均价 | 封板率% | 胜率% | 盈亏比 | 期望% | "
      "封板票期望 | 未封板票期望 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t in ENTRY_TARGETS:
        r = entry_res[int(t)]
        if not r.get("n"):
            A(f"| {int(t)}% | 0 | - | - | - | - | - | - | - |")
            continue
        A(f"| {int(t)}% | {r['n']}/{r['n_all']} | {r['entry_pct_mean']}% "
          f"| {r['seal_rate']} | {r['wr']} | {r['ratio']} | **{r['exp']}** "
          f"| {r['exp_sealed']} | {r['exp_unsealed']} |")
    A("")

    A("## 形态因子粗分组对比\n")
    A("样本量小, 按中位数二分(连续因子)或 0/1 分组(布尔因子)。"
      "**仅方向性提示, 不作结论。**\n")
    for grp in ("A 形态", "B 阻力", "C 突破", "D 量能", "E 价位", "F 量价"):
        facs = [f for f in CONT + BOOL if GROUP.get(f) == grp]
        if not facs:
            continue
        A(f"### {grp}组\n")
        for fac in facs:
            any_row = False
            for t in ENTRY_TARGETS:
                rows = fac_res[fac][int(t)]
                if not rows:
                    continue
                if not any_row:
                    A(f"**`{fac}`**\n")
                    A("| 入场档 | 分组 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% |")
                    A("|---|---|---|---|---|---|---|")
                    any_row = True
                for lab, n, sr, wr, ratio, exp in rows:
                    A(f"| {int(t)}% | {lab} | {n} | {sr} | {wr} | {ratio} "
                      f"| {exp} |")
            if not any_row:
                A(f"**`{fac}`** — 样本不足(<40), 无法评估\n")
            else:
                A("")

    A("## 约束与后续\n")
    A("1. **样本量不足是当前最大瓶颈** —— 4 个交易日约 300 条, 无法做"
      "统计显著的因子筛选。需靠 `daily_update.sh` 每日积累, 建议积累到 "
      "≥60 个交易日再下结论")
    A("2. **本研究结论只作方向性提示, 不进生产** —— 与 T/S 双轨体系一致"
      "(新维度须经历史验证且只走影子)")
    A("3. 东财分钟 API 仅 ~5 日深度, 历史日无法回填 —— 这是硬约束, "
      "只能靠持续积累")
    A("4. 离场口径是**部分模拟**: `simulate_exit` 只模拟三条(止损5% / "
      "P2 10:10 / 14:57强平), P0/P1/P3 未模拟")
    A("5. 因子定义唯一出处 `core/shape.py`; 可执行标签口径见 "
      "`compute_labels_exec`(接力场景)与本研究的 `find_entry`(半路场景)")

    (OUT / "39_halfway_entry.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
