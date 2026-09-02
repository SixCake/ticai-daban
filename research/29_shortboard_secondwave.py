# -*- coding: utf-8 -*-
"""研究29: 高位龙头短板 + 二波状态机(精简5态) 历史验证

目的(方案 龙头短板二波追踪层 Phase 1, 集成前必须通过):
  现有天梯/角色是"当日涨停快照", 前几日龙头今日未涨停即被丢弃。本研究验证
  "保留高位龙头短板并按二波状态机分类"是否有前向 alpha, 通过后才接入生产。

口径(与 core/shortboard.py 冻结口径一致):
  Cohort = 近N日题材龙头(theme.day.leader_code) ∪ 市场高板(events limit_times≥3)
           - 今日涨停池(events当日ts_code), 且高位守卫:
           peak_drawdown>-30% 且 (20日最大涨幅≥wave_th 或 窗口内曾连板≥hb_th)
  5态(自上而下首个命中):
    逻辑失效 peak_drawdown≤-30% 或 (neg_streak≥2 & neg_deep) 或 (ldlr_prev≥0.5 & pct≤-5%)
    二波候选 pct≥3% 且 cpos≥0.6 且 pressure_recovery≥0.35 且 ldlr_prev<0.5
    放量分歧 volr5≥1.5 且 cpos<0.5 且 peak_drawdown>-25% 且 neg_streak<2
    绕异动   volr5<0.8 且 |pct|<3% 且 cpos≥0.5 且 peak_drawdown>-15%
    断板观察 默认
  字段来源: pct/cpos=当日T实际bar(daily_panel); volr5/neg_streak/neg_deep/
    ldlr_prev=factor.longtou的T行(=T-1结构值); peak_drawdown/pressure_recovery
    =daily_panel近20日(峰/谷取≤T-1, 现价取T)。与盘中"报价+尾行"口径对齐。

环境三段(强制方法论): 牛/熊/震荡 由全A等权指数 vs 20日均线偏离(±2%)切分。
多方案: N∈{3,5,8} × wave_th∈{0.35,0.50} × hb_th∈{3,4}。
验收门槛(Gate): 二波候选/绕异动 前向收益 > 失效 且 > 0, 且至少2/3环境方向一致。

用法: python research/29_shortboard_secondwave.py
产物: research/out/29_shortboard_secondwave.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load, path_of  # noqa: E402

OUT = ROOT / "research" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 冻结阈值(与 core/shortboard.py 一致; 验证后可微调) ----
PEAK_WIN = 20            # 峰/谷回看窗口
GUARD_DD = -0.30         # 高位守卫: 距峰回撤下限(未深破位)
HB_LIMIT = 3             # 市场高板入池连板阈值(基线)
# 状态判据变体(方案第三维): 二波候选冲高阈bz_pct / 绕异动接近阈yd_near
STATE_VARIANTS = {
    "base":      {"bz_pct": 3.0, "yd_near": 0.8},
    "strict_bz": {"bz_pct": 5.0, "yd_near": 0.8},   # 二波候选更严(冲高≥5%)
    "loose_yd":  {"bz_pct": 3.0, "yd_near": 0.6},   # 绕异动更宽(10日≥60%)
}
SCHEMES = [
    {"label": "N5/w35/hb3/base", "N": 5, "wave_th": 0.35, "hb_th": 3, "sv": "base"},
    {"label": "N5/w35/hb3/strict_bz", "N": 5, "wave_th": 0.35, "hb_th": 3, "sv": "strict_bz"},
    {"label": "N5/w35/hb3/loose_yd", "N": 5, "wave_th": 0.35, "hb_th": 3, "sv": "loose_yd"},
    {"label": "N5/w50/hb3/base", "N": 5, "wave_th": 0.50, "hb_th": 3, "sv": "base"},
    {"label": "N3/w35/hb3/base", "N": 3, "wave_th": 0.35, "hb_th": 3, "sv": "base"},
    {"label": "N8/w35/hb3/base", "N": 8, "wave_th": 0.35, "hb_th": 3, "sv": "base"},
    {"label": "N5/w35/hb4/base", "N": 5, "wave_th": 0.35, "hb_th": 4, "sv": "base"},
]
STATE_ORDER = ["二波候选", "绕异动", "放量分歧", "断板观察", "逻辑失效"]


def shortboard_state(snap: dict, sv: dict = None) -> str:
    """精简5态(自上而下首个命中); 与 core/shortboard.shortboard_state_of 同口径。
    sv=状态判据变体{bz_pct,yd_near}, 缺省用基线阈值。"""
    sv = sv or STATE_VARIANTS["base"]
    bz_pct, yd_near = sv["bz_pct"], sv["yd_near"]
    dd = snap.get("peak_drawdown")
    rec = snap.get("pressure_recovery")
    ns = snap.get("neg_streak")
    nd = snap.get("neg_deep")
    ldlr = snap.get("ldlr_prev")
    pct = snap.get("pct")
    cpos = snap.get("cpos")
    volr5 = snap.get("volr5")
    if dd is None or pct is None or cpos is None:
        return "断板观察"
    # 1 逻辑失效
    if (dd <= GUARD_DD
            or (ns is not None and ns >= 2 and bool(nd))
            or (ldlr is not None and ldlr >= 0.5 and pct <= -5.0)):
        return "逻辑失效"
    # 2 二波候选(需第一波已死亡测试, 排除第一波延续误标)
    if (snap.get("death_test") and pct >= bz_pct and cpos >= 0.6
            and rec is not None and rec >= 0.35
            and (ldlr is None or ldlr < 0.5)):
        return "二波候选"
    # 3 放量分歧
    if (volr5 is not None and volr5 >= 1.5 and cpos < 0.5
            and dd > -0.25 and (ns is None or ns < 2)):
        return "放量分歧"
    # 4 绕异动(接近监管严重异常波动线: 10日累计近+100% / 30日近+200%)
    c10, c30 = snap.get("cum10"), snap.get("cum30")
    if ((c10 is not None and c10 == c10 and c10 >= 100.0 * yd_near)
            or (c30 is not None and c30 == c30 and c30 >= 200.0 * yd_near)):
        return "绕异动"
    # 5 断板观察
    return "断板观察"


def build_panel_features() -> pd.DataFrame:
    """全A面板 → 每股 (date,code) 的峰谷/回撤/修复/当日bar/前向收益"""
    print("加载 daily_panel ...")
    p = pd.read_parquet(path_of("market.daily_panel"))
    p = p.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = p.groupby("ts_code", sort=False)
    # 峰/谷取≤T-1(shift1): 第一波压力峰, 与盘中口径一致(不含当日未完成bar)
    p["peak_high"] = g["high"].transform(lambda s: s.rolling(PEAK_WIN, min_periods=10).max().shift(1))
    p["trough_low"] = g["low"].transform(lambda s: s.rolling(PEAK_WIN, min_periods=10).min().shift(1))
    p["wave_ret20"] = p["peak_high"] / p["trough_low"] - 1
    p["peak_dd"] = p["close"] / p["peak_high"] - 1
    rng = (p["peak_high"] - p["trough_low"]).replace(0, np.nan)
    p["pressure_rec"] = ((p["close"] - p["trough_low"]) / rng).clip(0, 1)
    # 当日T实际bar(offline口径, 对应盘中cur_pct/cur_cpos)
    p["pct_t"] = p["pct_chg"]
    p["cpos_t"] = np.where(p["high"] > p["low"],
                           (p["close"] - p["low"]) / (p["high"] - p["low"]), 0.5)
    # 前向收益(cohort非涨停, 无events next_open_ret, 自算)
    p["ret_open_t1"] = g["open"].shift(-1) / p["close"] - 1
    p["ret_close_t1"] = g["close"].shift(-1) / p["close"] - 1
    p["ret_close_t3"] = g["close"].shift(-3) / p["close"] - 1
    p["ret_close_t5"] = g["close"].shift(-5) / p["close"] - 1
    # 累计涨幅(监管异动口径): 10日/30日 rolling 对数收益和 → 累计涨幅%
    p["logret"] = np.log1p(p["pct_chg"] / 100)
    p["cum10"] = (np.exp(g["logret"].transform(
        lambda s: s.rolling(10, min_periods=10).sum())) - 1) * 100
    p["cum30"] = (np.exp(g["logret"].transform(
        lambda s: s.rolling(30, min_periods=30).sum())) - 1) * 100
    # 市场等权指数 vs 20日均线 → 环境三段
    mkt = p.groupby("trade_date")["pct_chg"].mean().sort_index()
    idx = (1 + mkt / 100).cumprod()
    dev = idx / idx.rolling(20).mean() - 1
    regime = pd.Series("震荡", index=idx.index)
    regime[dev > 0.02] = "牛"
    regime[dev < -0.02] = "熊"
    print(f"面板特征完成 {len(p):,}行; 环境分布 "
          f"{regime.value_counts().to_dict()}")
    cols = ["trade_date", "ts_code", "pct_t", "cpos_t", "peak_dd",
            "pressure_rec", "wave_ret20", "cum10", "cum30", "ret_open_t1",
            "ret_close_t1", "ret_close_t3", "ret_close_t5"]
    return p[cols], regime


def build_superset(regime: pd.Series) -> pd.DataFrame:
    """近8日(最大N)龙头∪高板候选超集(未排今日池/未守卫), 每行=(T,code,kind,gap,lt)"""
    td = load("theme.day", columns=["trade_date", "concept_code",
                                    "leader_code", "leader_name", "concept_name"])
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "name", "limit_times"])
    dates = sorted(regime.index)              # 全交易日历(升序)
    dpos = {d: i for i, d in enumerate(dates)}

    # 出现表: (date, code, kind, limit_times)
    lead = td[["trade_date", "leader_code"]].dropna().drop_duplicates()
    lead = lead.rename(columns={"leader_code": "ts_code"})
    lead["kind"] = "leader"
    lead["lt"] = 0
    hb = ev[ev["limit_times"] >= HB_LIMIT][["trade_date", "ts_code", "limit_times"]]
    hb = hb.rename(columns={"limit_times": "lt"})
    hb["kind"] = "hb"
    appear = pd.concat([lead[["trade_date", "ts_code", "kind", "lt"]],
                        hb[["trade_date", "ts_code", "kind", "lt"]]],
                       ignore_index=True).drop_duplicates(
        subset=["trade_date", "ts_code", "kind"])
    NMAX = max(s["N"] for s in SCHEMES)

    # 每个出现向前铺 NMAX 个交易日 → (T, code, gap, kind, lt)
    rows = []
    appear["pos"] = appear["trade_date"].map(dpos)
    for r in appear.itertuples():
        base = r.pos
        for gap in range(1, NMAX + 1):
            tp = base + gap
            if tp >= len(dates):
                break
            rows.append((dates[tp], r.ts_code, gap, r.kind, r.lt))
    sup = pd.DataFrame(rows, columns=["trade_date", "ts_code", "gap", "kind", "lt"])
    # 同(T,code)合并: 取最小gap, 最大lt, 是否leader/hb
    agg = sup.groupby(["trade_date", "ts_code"]).agg(
        gap=("gap", "min"), lt=("lt", "max"),
        is_leader=("kind", lambda s: (s == "leader").any()),
        is_hb=("kind", lambda s: (s == "hb").any())).reset_index()

    # 排除今日涨停池
    pool = ev.groupby("trade_date")["ts_code"].apply(set).to_dict()
    keep = [c not in pool.get(d, set())
            for d, c in zip(agg["trade_date"], agg["ts_code"])]
    agg = agg[np.array(keep)].reset_index(drop=True)
    # 名称
    names = ev.sort_values("trade_date").groupby("ts_code")["name"].last().to_dict()
    agg["name"] = agg["ts_code"].map(names)
    print(f"候选超集(排今日池后) {len(agg):,} 行, "
          f"{agg['ts_code'].nunique()} 只")
    return agg


def attach_features(sup: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """超集挂面板特征 + factor.longtou结构字段(T行=T-1值)"""
    df = sup.merge(feat, on=["trade_date", "ts_code"], how="inner")
    fl = load("factor.longtou",
              columns=["trade_date", "ts_code", "y_volr5", "neg_streak",
                       "neg_deep", "ldlr_prev"])
    df = df.merge(fl, on=["trade_date", "ts_code"], how="left")
    df = df.rename(columns={"peak_dd": "peak_drawdown",
                            "pressure_rec": "pressure_recovery",
                            "y_volr5": "volr5", "pct_t": "pct",
                            "cpos_t": "cpos"})
    # 有当日bar(peak_drawdown/pct非空)才纳入(停牌/无成交剔除)
    df = df.dropna(subset=["peak_drawdown", "pct"]).reset_index(drop=True)
    return df


def compute_death_test(df: pd.DataFrame) -> pd.DataFrame:
    """per(date,code)死亡测试: 用 core.shortboard.death_test_of(与生产同口径,
    尾21根≤T-1)。仅对潜在二波候选行(pct≥3&cpos≥0.6&修复≥0.35)计算,
    其余False(不影响状态判定)。"""
    from core.shortboard import death_test_of
    codes = df["ts_code"].unique().tolist()
    pn = pd.read_parquet(path_of("market.daily_panel"),
                         columns=["trade_date", "ts_code", "high", "low",
                                  "close", "vol", "pct_chg"])
    pn = pn[pn["ts_code"].isin(codes)].sort_values(["ts_code", "trade_date"])
    bars_by_code = {c: [(r.trade_date, r.high, r.low, r.close, r.vol, r.pct_chg)
                        for r in g.itertuples()]
                    for c, g in pn.groupby("ts_code")}
    idx_by_code = {c: {b[0]: i for i, b in enumerate(bl)}
                   for c, bl in bars_by_code.items()}
    dt = []
    for r in df.itertuples():
        if not (r.pct >= 3.0 and r.cpos >= 0.6 and r.pressure_recovery >= 0.35):
            dt.append(False)          # 非潜在二波候选, 无需死亡测试
            continue
        bl = bars_by_code.get(r.ts_code, [])
        i = idx_by_code.get(r.ts_code, {}).get(r.trade_date)
        if i is None:
            dt.append(False)
            continue
        dt.append(death_test_of(r.ts_code, bl[max(0, i - 21):i]))
    df["death_test"] = dt
    return df


def derive_states(df: pd.DataFrame, regime: pd.Series, sv: dict = None) -> pd.DataFrame:
    df = df.copy()
    df["state"] = [shortboard_state(s, sv) for s in df.to_dict("records")]
    df["regime"] = df["trade_date"].map(regime)
    return df


def apply_scheme(df: pd.DataFrame, sch: dict) -> pd.DataFrame:
    m = (df["gap"] <= sch["N"]) & (df["peak_drawdown"] > GUARD_DD)
    ident = (df["wave_ret20"] >= sch["wave_th"]) | (df["lt"] >= sch["hb_th"])
    return df[m & ident].copy()


RET_COLS = ["ret_open_t1", "ret_close_t1", "ret_close_t3", "ret_close_t5"]


def stat_block(sub: pd.DataFrame) -> dict:
    out = {"n": len(sub)}
    for c in RET_COLS:
        v = sub[c].dropna()
        out[c] = round(v.mean() * 100, 2) if len(v) else np.nan
    return out


def run_scheme(df: pd.DataFrame, sch: dict, regime: pd.Series) -> dict:
    coh = apply_scheme(df, sch)
    coh = derive_states(coh, regime, STATE_VARIANTS.get(sch.get("sv", "base")))
    res = {"label": sch["label"], "n": len(coh), "by_state": {}, "by_regime": {}}
    for st in STATE_ORDER:
        res["by_state"][st] = stat_block(coh[coh["state"] == st])
    for rg in ["牛", "熊", "震荡"]:
        sub = coh[coh["regime"] == rg]
        res["by_regime"][rg] = {"n": len(sub),
                                "by_state": {st: stat_block(sub[sub["state"] == st])
                                             for st in STATE_ORDER}}
    return res


def gate_check(res: dict) -> dict:
    """验收: 二波候选/绕异动 T+1开盘收益 > 失效 且 > 0, 且≥2/3环境方向一致"""
    bs = res["by_state"]
    active = ["二波候选", "绕异动"]
    fail_ret = bs["逻辑失效"]["ret_open_t1"]
    verdict = {}
    for st in active:
        r = bs[st]["ret_open_t1"]
        n = bs[st]["n"]
        beat_fail = (r is not np.nan and not np.isnan(r)
                     and not np.isnan(fail_ret) and r > fail_ret)
        positive = (not np.isnan(r)) and r > 0
        # 环境一致性: 该态在≥2个环境中 T+1开盘 >0
        env_ok = 0
        for rg in ["牛", "熊", "震荡"]:
            rr = res["by_regime"][rg]["by_state"][st]["ret_open_t1"]
            if not np.isnan(rr) and rr > 0:
                env_ok += 1
        verdict[st] = {"n": n, "ret_open_t1": r, "beat_fail": bool(beat_fail),
                       "positive": bool(positive), "env_ok": env_ok,
                       "pass": bool(beat_fail and positive and env_ok >= 2)}
    res["gate"] = verdict
    res["gate_pass"] = any(v["pass"] for v in verdict.values())
    return res


def fmt_table(res: dict) -> str:
    lines = [f"### 方案 {res['label']} (cohort n={res['n']})", "",
             "| 状态 | n | T+1开盘% | T+1收盘% | T+3% | T+5% |",
             "|---|---|---|---|---|---|"]
    for st in STATE_ORDER:
        s = res["by_state"][st]
        lines.append(f"| {st} | {s['n']} | {s['ret_open_t1']} | "
                     f"{s['ret_close_t1']} | {s['ret_close_t3']} | {s['ret_close_t5']} |")
    lines += ["", "**分环境 T+1开盘收益%**", "",
              "| 状态 | 牛(n) | 熊(n) | 震荡(n) |", "|---|---|---|---|"]
    for st in STATE_ORDER:
        cells = []
        for rg in ["牛", "熊", "震荡"]:
            b = res["by_regime"][rg]["by_state"][st]
            cells.append(f"{b['ret_open_t1']}({b['n']})")
        lines.append(f"| {st} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "**Gate**: " + "; ".join(
        f"{st}→{'通过' if v['pass'] else '未过'}"
        f"(n={v['n']}, T+1开盘={v['ret_open_t1']}%, 优于失效={v['beat_fail']}, "
        f">0={v['positive']}, 环境{v['env_ok']}/3)"
        for st, v in res["gate"].items()), ""]
    return "\n".join(lines)


def main():
    feat, regime = build_panel_features()
    sup = build_superset(regime)
    df = attach_features(sup, feat)
    df = compute_death_test(df)
    print(f"挂特征后 {len(df):,} 行")

    report = ["# 研究29: 高位龙头短板 + 二波状态机(精简5态) 历史验证", "",
              f"- 样本区间: {df['trade_date'].min()} ~ {df['trade_date'].max()}",
              f"- 环境分布(全A等权指数vs20日均线±2%): "
              f"{regime.value_counts().to_dict()}",
              f"- 多方案: {', '.join(s['label'] for s in SCHEMES)}", "",
              "验收门槛: 二波候选/绕异动 T+1开盘收益 > 逻辑失效 且 > 0, "
              "且≥2/3环境方向一致(同研究28口径)。", ""]
    any_pass = False
    frozen = None
    for sch in SCHEMES:
        res = gate_check(run_scheme(df, sch, regime))
        report.append(fmt_table(res))
        if res["gate_pass"]:
            any_pass = True
            if frozen is None:
                frozen = sch["label"]
    report += ["---", "",
               f"## 结论: Gate {'通过' if any_pass else '未通过'}"
               + (f" (推荐冻结方案: {frozen})" if any_pass else ""),
               "通过则进入 Phase 2 生产集成(core/shortboard.py + review/poller/dashboard);",
               "未通过则不集成(92 Kobe 新维度须历史验证)。"]
    txt = "\n".join(report)
    outp = OUT / "29_shortboard_secondwave.md"
    outp.write_text(txt, encoding="utf-8")
    print(f"\n报告 → {outp}")
    print(f"Gate: {'通过' if any_pass else '未通过'}"
          + (f" (冻结方案 {frozen})" if any_pass else ""))


if __name__ == "__main__":
    main()
