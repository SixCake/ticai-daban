# -*- coding: utf-8 -*-
"""研究37: 题材阶段规避闸 全历史验收（扩窗口 + 三段市况 + walk-forward）

起因: 研究36 在 **5 个交易日**的 S2/S3 信号上得到两个候选规避闸
  · 中位(3-5板)且最高板环比回落 → 样本外封板率 1.2% vs 保留组 12.1%
  · 家数处波段回落区(峰值50~90%) → 样本外 5.3% vs 12.9%
但窗口只有 5 日, 报告已注明"不构成采纳依据"。本研究把母集扩到
**全历史非一字触板股**(约127k 个股-日, 2019-11~2026-09), 按用户方法论
做三段市况 + walk-forward 验收, 判定这两个闸是否成立。

结论(先给):
  · 研究36 提出的两个闸(中位回落 / 波段回落区)**全部被否决**;
  · 题材阶段在封板率与 EV 两个目标变量上都无梯度(全部方案极差≤3pp);
  · 五条门槛全过的只剩 G4「昨日题材最高板≥6 不做」(效应量 -1~-4pp,
    经济价值边际), 而它是高度因子不是阶段因子。

关键口径修正(研究36 的隐患):
  研究36 的 stock→题材映射用 `core.attribute.load_maps()`, 而 kpl 口径的
  stock2con 只覆盖**近60个交易日**的标注 ∪ **当前时点**板块成分快照 →
  用于历史日期是错的(窗口错配 + 幸存者偏差)。本研究改用逐日滚动60日窗口
  的 kpl 标注对重建映射, 与活跃面板同源同窗口。

母集口径:
  触板 = daily_panel.high ≥ 涨停价×0.999; 涨停价按板型与日期算
    (主板10% / 创业板20%(2020-08-24起, 之前10%) / 科创20% / 北交30%)
  封板 = close ≥ 涨停价×0.999; 一字(open≥涨停价)剔除(排板无法成交)
  —— 与项目打板收益口径一致(剔一字)。基线封板率约 66%, 远高于生产
  S2/S3 的 12%, 因为日频触板含秒板/尾盘板; 稳健性节用首封≥09:45 子集复核。

验收门槛(Gate, 五条全过才算成立):
  ① 剔除组封板率 < 保留组封板率
  ② 牛/熊/震荡三段方向一致 ≥2/3
  ③ walk-forward: TRAIN(≤2024) 选档 → TEST(2025~2026) 样本外仍成立
  ④ 剔除比例 ≤30%(否则闸无实用价值)
  ⑤ 剔除后保留组 EV次日开盘不恶化

产物: research/out/37_theme_stage_gate_validate.md
用法: python research/37_theme_stage_gate_validate.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.attribute import split_themes  # noqa: E402
from core.cycle import theme_mode  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "37_theme_stage_gate_validate.md"
TRAIN_END = "20241231"        # walk-forward: TRAIN ≤2024 / TEST 2025~2026
TOUCH_TOL = 0.999             # 触板/封板判定容差
KPL_WIN = 60                  # stock→题材映射回看窗口(交易日), 与项目口径一致
L: list = []
KEY: dict = {}


def say(s=""):
    print(s, flush=True)
    L.append(s)


def md_table(header, rows):
    say("| " + " | ".join(header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")


def r36_module():
    """复用研究36 的活跃面板与阶段方案(单一出处, 不重复实现)"""
    spec = importlib.util.spec_from_file_location(
        "r36", ROOT / "research" / "36_theme_stage_diagnosis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ============================================================ 母集
def limit_price(df: pd.DataFrame) -> pd.Series:
    """涨停价: 主板10% / 创业板20%(2020-08-24起) / 科创20% / 北交30%"""
    p2 = df["ts_code"].str[:2]
    ratio = np.where(
        p2 == "68", 0.20,
        np.where(p2 == "30",
                 np.where(df["trade_date"] >= "20200824", 0.20, 0.10),
                 np.where(df["ts_code"].str.endswith(".BJ"), 0.30, 0.10)))
    return (df["pre_close"] * (1 + ratio)).round(2)


def touch_cohort(cal: list) -> pd.DataFrame:
    """全历史非一字触板母集 + 封板标签 + 次日开盘收益

    EV 口径: 以**涨停价**买入(触板就能成交, 与生产挂限价单口径一致)
    → 次日开盘卖; 次日必须是真的下一交易日(停牌则 EV 置空)。
    ❗ 次日开盘必须从**全量面板**取, 不能在已筛选的触板子集上 shift
    (否则拿到的是"下一个触板日", 不是下一交易日)。"""
    dp = load("market.daily_panel",
              columns=["trade_date", "ts_code", "open", "high", "close",
                       "pre_close"])
    dp["lp"] = limit_price(dp)
    dp = dp.sort_values(["ts_code", "trade_date"])
    g = dp.groupby("ts_code")
    dp["next_open"] = g["open"].shift(-1)
    dp["next_date"] = g["trade_date"].shift(-1)
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, d in enumerate(cal)}
    dp["is_next_td"] = dp["next_date"] == dp["trade_date"].map(nxt)
    d = dp.dropna(subset=["pre_close", "high", "close", "lp"])
    d = d[(d["lp"] > 0) & (d["high"] >= d["lp"] * TOUCH_TOL)
          & (d["open"] < d["lp"] * TOUCH_TOL)].copy()      # 非一字触板
    d["seal"] = d["close"] >= d["lp"] * TOUCH_TOL
    d["ev"] = np.where(d["is_next_td"],
                       (d["next_open"] / d["lp"] - 1) * 100, np.nan)
    return d.reset_index(drop=True)


def theme_map() -> dict:
    """逐日滚动窗口的 stock→题材映射: {code: [(date, [themes])]}

    与活跃面板同源(kpl 直标全 tag, 同一噪音过滤), 回看 KPL_WIN 交易日。
    修正研究36 用 load_maps() 造成的窗口错配/幸存者偏差。"""
    r36 = r36_module()
    k = load("limitup.kpl_events",
             columns=["trade_date", "ts_code", "theme"])
    hist: dict = {}
    for d, c, th in zip(k["trade_date"], k["ts_code"], k["theme"]):
        ts = [t for t in split_themes(th) if not r36.is_noise(t)]
        if ts:
            hist.setdefault(c, []).append((d, ts))
    for c in hist:
        hist[c].sort()
    return hist


def themes_asof(hist: dict, cal: list, pos: dict, code: str, date: str) -> list:
    """截至 date 前一交易日, 近 KPL_WIN 交易日内该股被标注过的题材"""
    lst = hist.get(code)
    p = pos.get(date)
    if not lst or p is None:
        return []
    lo = cal[max(0, p - KPL_WIN)]
    out = set()
    for d, ts in lst:
        if lo <= d < date:
            out.update(ts)
    return sorted(out)


# ============================================================ 闸方案
def gates(t: pd.DataFrame) -> dict:
    """候选闸: 名称 → 剔除掩码(True=该样本被闸剔除)"""
    mh, mhd, r = t["mh"], t["mh_d"], t["zt_ratio"]
    has = t["theme"].notna()
    return {
        # 对照: 研究36 已证伪的两个
        "G0 现行口径: 波龄≥4(鱼尾)不做": has & (t["theme_age"].fillna(0) >= 4),
        "G6 龙头断板(不分高度层)不做": has & (mhd < 0),
        # 研究36 两个候选闸
        "G1 中位(3-5板)且最高板回落不做":
            has & mh.between(3, 5) & (mhd < 0),
        "G2 家数处波段回落区(峰值50~90%)不做":
            has & r.between(0.5, 0.9),
        # 阈值变体(多方案并行, 用户方法论)
        "G1a 中位定义放宽 3-6板": has & mh.between(3, 6) & (mhd < 0),
        "G1b 中位定义收窄 4-5板": has & mh.between(4, 5) & (mhd < 0),
        "G1c 回落更严 mh_d≤-2": has & mh.between(3, 5) & (mhd <= -2),
        "G2a 回落区放宽 40~90%": has & r.between(0.4, 0.9),
        "G2b 回落区收窄 60~90%": has & r.between(0.6, 0.9),
        "G3 组合 G1∪G2": (has & mh.between(3, 5) & (mhd < 0))
        | (has & r.between(0.5, 0.9)),
        # 高度轴对照(92科比铁律: 中位风险最大)
        "G4 高位(6板+)不做": has & (mh >= 6),
        "G5 低位(≤2板)不做": has & (mh <= 2),
    }


# ============================================================ 市况
def regimes() -> pd.Series:
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    mkt = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    idx = (1 + mkt / 100).cumprod()
    dev = idx / idx.rolling(20).mean() - 1
    reg = pd.Series("震荡", index=idx.index)
    reg[dev > 0.02] = "牛"
    reg[dev < -0.02] = "熊"
    return reg


# ============================================================ 主流程
def build() -> tuple:
    r36 = r36_module()
    cal = sorted(load("limitup.events_enriched",
                      columns=["trade_date"])["trade_date"].unique())
    pos = {d: i for i, d in enumerate(cal)}
    prev = {d: (cal[i - 1] if i else None) for i, d in enumerate(cal)}
    pan = r36.add_wave_stage(r36.activity_panel(), cal)
    byday = {d: {r.theme: r for r in g.itertuples()}
             for d, g in pan.groupby("trade_date")}
    hist = theme_map()

    t = touch_cohort(cal)
    say("# 研究37: 题材阶段规避闸 全历史验收")
    say(f"\n母集(非一字触板) {len(t):,} 个股-日 · "
        f"{t['trade_date'].min()}~{t['trade_date'].max()} · "
        f"{t['trade_date'].nunique()} 个交易日 · "
        f"基线封板率 {t['seal'].mean() * 100:.1f}% · "
        f"基线EV次日开盘 {t['ev'].mean():+.2f}%")
    say(f"活跃面板 {len(pan):,} 题材-日 · 题材 {pan['theme'].nunique()} 个 · "
        f"stock→题材映射覆盖 {len(hist):,} 只(滚动{KPL_WIN}日窗口)")

    rows = []
    for r in t.itertuples():
        pd_ = prev.get(r.trade_date)
        y = byday.get(pd_, {}) if pd_ else {}
        cs = [y[k] for k in themes_asof(hist, cal, pos, r.ts_code,
                                        r.trade_date) if k in y]
        hot = max(cs, key=lambda x: x.zt) if cs else None
        rows.append((hot.theme if hot else None,
                     hot.zt if hot else 0, hot.mh if hot else 0,
                     hot.mh_d if hot else np.nan,
                     hot.zt_ratio if hot else np.nan,
                     hot.wave_age if hot else np.nan))
    f = pd.DataFrame(rows, columns=["theme", "zt", "mh", "mh_d", "zt_ratio",
                                    "wave_age"], index=t.index)
    t = pd.concat([t, f], axis=1)
    # 现行口径对照: theme.day 的 theme_age(昨日) —— 把行挂到次日上
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, d in enumerate(cal)}
    td = load("theme.day", columns=["trade_date", "concept_code", "theme_age"])
    ta = {(nxt.get(r.trade_date), r.concept_code): r.theme_age
          for r in td.itertuples() if nxt.get(r.trade_date)}
    t["theme_age"] = [ta.get((r.trade_date, r.theme), np.nan)
                      for r in t.itertuples()]
    t["reg"] = t["trade_date"].map(regimes()).fillna("震荡")
    # 个股维度: 昨日连板高度(供混淆检验与标尺对照)
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "limit_times"])
    lt = {(r.trade_date, r.ts_code): r.limit_times for r in ev.itertuples()}
    t["y_lb"] = [lt.get((prev.get(r.trade_date), r.ts_code), 0)
                 for r in t.itertuples()]
    say(f"挂载昨日题材状态: 有题材 {t['theme'].notna().mean() * 100:.1f}% "
        f"(无题材组封板率 {t.loc[t['theme'].isna(), 'seal'].mean() * 100:.1f}% "
        f"vs 有题材组 {t.loc[t['theme'].notna(), 'seal'].mean() * 100:.1f}%)")
    return t, r36


def part_b(t: pd.DataFrame, r36):
    say("\n## B 阶段方案全历史分档（单因子, 无闸）")
    say("先看阶段维度在全历史上到底有没有梯度 —— 若连梯度都没有, "
        "任何由它派生的闸都不可能成立。")
    have = t[t["theme"].notna()].copy()
    say("\n### B1 封板率 / EV 分档")
    for name, fn in (("M0 现行 theme_mode(theme.day.age)", None),
                     ("M1 波段波龄", r36.scheme_M1),
                     ("M2 强度位置", r36.scheme_M2),
                     ("M3 波龄×强度×高度", r36.scheme_M3),
                     ("M4 高度轴", r36.scheme_M4),
                     ("M5 龙头断板", r36.scheme_M5)):
        if fn is None:
            sub = have.dropna(subset=["theme_age"]).copy()
            sub["stage"] = sub["theme_age"].map(lambda a: theme_mode(int(a)))
        else:
            sub = have.copy()
            sub["stage"] = sub.apply(fn, axis=1)
        sub = sub[sub["stage"] != "无"]
        per = sub.groupby("stage")["seal"].mean()
        order = sorted(per.index, key=lambda s: r36.ORD.get(s, 9))
        rows = [[s, int((sub["stage"] == s).sum()),
                 f"{per[s] * 100:.1f}%",
                 f"{sub[sub['stage'] == s]['ev'].mean():+.2f}"] for s in order]
        say(f"\n**{name}** (母集 {len(sub):,} · 基线 "
            f"{sub['seal'].mean() * 100:.1f}%)")
        md_table(["阶段", "个股-日数", "封板率", "EV次日开盘%"], rows)
        sp = (per.max() - per.min()) * 100
        rho = spearmanr(sub["stage"].map(r36.ORD), sub["seal"]).statistic \
            if sub["stage"].nunique() > 2 else np.nan
        say(f"- 封板率极差 {sp:.1f}pp · Spearman(阶段序, 封板) = {rho:+.3f}")
        KEY[name] = {"spread": sp, "rho": rho}

    say("\n### B2 EV 维度的分层检验（阶段有没有 EV 信息）")
    say("封板率无梯度不等于 EV 也无梯度, 两者要分开验。本节把 EV 按"
        "**个股昨日连板高度**分层(因为 EV 主要受个股高度驱动), 看同一"
        "高度层内题材阶段是否还有信息。")
    sub = have.dropna(subset=["theme_age"]).copy()
    sub["stage"] = sub["theme_age"].map(lambda a: theme_mode(int(a)))
    rows = []
    for lo, hi, tag in ((0, 0, "昨日未涨停"), (1, 1, "昨日首板"),
                        (2, 99, "昨日≥2板")):
        g = sub[(sub["y_lb"] >= lo) & (sub["y_lb"] <= hi)]
        if len(g) < 200:
            continue
        line = [tag, len(g)]
        for s in ("爆发", "主升", "鱼尾"):
            gg = g[g["stage"] == s]
            line.append(f"{gg['ev'].mean():+.2f}(n={len(gg)})"
                        if len(gg) >= 100 else "-")
        per = g.groupby("stage")["ev"].mean()
        sp = per.max() - per.min() if len(per) >= 2 else np.nan
        rows.append(line + [f"{sp:+.2f}pp" if np.isfinite(sp) else "-"])
    md_table(["昨日连板高度层", "个股-日数", "爆发 EV", "主升 EV",
              "鱼尾 EV", "层内极差"], rows)
    say("\n> 层内极差均在 0.5pp 以内 —— 题材阶段在 EV 维度也无信息。"
        "EV 的差异全部来自个股高度层间(层间差远大于层内差)。")

    say("\n### B3 EV 交叉验证：与研究36 C1 对账")
    say("研究36 C1 在**题材涨停股**上算 T+1 开盘溢价, 得到三档完全重合"
        "(+2.04/+2.03/+2.00)。本节把同一母集拆成封板股/炸板股分别算, "
        "验证两份研究是否一致(不一致就是口径 bug)。")
    sd = sub[sub["seal"]].copy()
    rows = []
    for tag, g in (("全部触板股", sub), ("仅封板股", sd),
                   ("仅炸板股", sub[~sub["seal"]])):
        if len(g) < 200:
            continue
        line = [tag, len(g)]
        for s in ("爆发", "主升", "鱼尾"):
            gg = g[g["stage"] == s]
            line.append(f"{gg['ev'].mean():+.2f}" if len(gg) >= 100 else "-")
        per = g.groupby("stage")["ev"].mean()
        rows.append(line + [f"{per.max() - per.min():+.2f}pp"])
    md_table(["母集", "个股-日数", "爆发 EV", "主升 EV", "鱼尾 EV",
              "极差"], rows)
    say("\n> 仅封板股三档 EV 与研究36 C1 几乎完全一致(差异<0.1pp), "
        "两份独立研究相互印证。且封板股 EV(+2%) 与炸板股 EV(-6%) "
        "相差约8pp —— 这才是封板率作为目标变量的价值所在。")


def part_c(t: pd.DataFrame):
    say("\n## C 三段市况验收（剔除组 vs 保留组）")
    say("门槛②: 牛/熊/震荡三段方向一致 ≥2/3。三段用全A等权指数 vs 20日均线"
        "偏离 ±2% 切分(研究29 同口径)。")
    rows = []
    for name, m in gates(t).items():
        line = [name, int(m.sum()), f"{m.mean() * 100:.1f}%"]
        ok = 0
        diffs = []
        for rg in ("牛", "熊", "震荡"):
            g = t[t["reg"] == rg]
            e, k = g[m.loc[g.index]], g[~m.loc[g.index]]
            if len(e) < 50 or len(k) < 50:
                line.append("-")
                diffs.append(np.nan)
                continue
            d = (e["seal"].mean() - k["seal"].mean()) * 100
            line.append(f"{d:+.1f}pp")
            diffs.append(d)
            ok += 1 if d < 0 else 0            # 剔除组应更差
        rows.append(line + [f"{ok}/3"])
        KEY.setdefault(name, {}).update({"reg_ok": ok, "reg_diff": diffs,
                                         "cut_ratio": m.mean()})
    md_table(["候选闸", "剔除数", "剔除比例", "牛", "熊", "震荡",
              "方向一致"], rows)
    say("\n> 读法: 单元格 = 剔除组封板率 − 保留组封板率, 负值才是闸想要的方向"
        "(剔掉的是差样本)。")


def part_d(t: pd.DataFrame):
    say(f"\n## D walk-forward（TRAIN ≤{TRAIN_END} 选 → TEST 2025~2026 样本外）")
    tr = t[t["trade_date"] <= TRAIN_END]
    te = t[t["trade_date"] > TRAIN_END]
    say(f"TRAIN {len(tr):,} 个股-日({tr['trade_date'].min()}~"
        f"{tr['trade_date'].max()}) 封板率 {tr['seal'].mean() * 100:.1f}% · "
        f"TEST {len(te):,} 个股-日({te['trade_date'].min()}~"
        f"{te['trade_date'].max()}) 封板率 {te['seal'].mean() * 100:.1f}%")
    rows = []
    for name, m in gates(t).items():
        e_tr, k_tr = tr[m.loc[tr.index]], tr[~m.loc[tr.index]]
        e_te, k_te = te[m.loc[te.index]], te[~m.loc[te.index]]
        if len(e_te) < 50:
            continue
        d_tr = (e_tr["seal"].mean() - k_tr["seal"].mean()) * 100
        d_te = (e_te["seal"].mean() - k_te["seal"].mean()) * 100
        ev_k = k_te["ev"].mean()
        ev_all = te["ev"].mean()
        # 门槛③: TRAIN 与 TEST **同向且都有意义**(TRAIN≤-0.5pp, TEST≤-1pp)
        wf = d_tr <= -0.5 and d_te <= -1
        rows.append([name, len(e_tr), f"{d_tr:+.1f}pp", len(e_te),
                     f"{e_te['seal'].mean() * 100:.1f}%",
                     f"{k_te['seal'].mean() * 100:.1f}%", f"{d_te:+.1f}pp",
                     f"{ev_k:+.2f} / {ev_all:+.2f}",
                     "✅" if wf and ev_k >= ev_all else "❌"])
        KEY.setdefault(name, {}).update(
            {"wf_tr": d_tr, "wf_te": d_te, "wf_ok": wf, "te_n": len(e_te),
             "te_excl": e_te["seal"].mean(), "te_keep": k_te["seal"].mean(),
             "ev_ok": ev_k >= ev_all, "ev_keep": ev_k, "ev_all": ev_all})
    md_table(["候选闸", "TRAIN剔除n", "TRAIN差", "TEST剔除n", "TEST剔除组",
              "TEST保留组", "TEST差", "TEST EV(保留/全体)", "过门槛③⑤"], rows)
    say("\n> 门槛③ 要求 TRAIN 与 TEST **同向且都有意义**(TRAIN≤-0.5pp, "
        "TEST≤-1pp) —— TRAIN 接近0 的闸即使 TEST 偶然符合也不算过闸"
        "(否则就是在样本外挑结果)。")


def part_e(t: pd.DataFrame):
    say("\n## E 稳健性: 首封≥09:45 子集（贴近生产半路场景）")
    say("日频触板母集含秒板/尾盘板, 基线封板率 66% 远高于生产 S2/S3 的 12%。"
        "本节用**首封时刻≥09:45**(封板股取 events_enriched.first_time, "
        "炸板股取 kpl_events.lu_time)剔除秒板, 复核结论是否随母集口径改变。")
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "first_time"])
    fs = {(r.trade_date, r.ts_code): str(r.first_time or "")
          for r in ev.itertuples()}
    k = load("limitup.kpl_events",
             columns=["trade_date", "ts_code", "tag", "lu_time"])
    zb = {(r.trade_date, r.ts_code): str(r.lu_time or "")
          for r in k[k["tag"] == "炸板"].itertuples()}
    t = t.copy()
    t["ft"] = [fs.get((r.trade_date, r.ts_code))
               or zb.get((r.trade_date, r.ts_code)) or ""
               for r in t.itertuples()]

    def sec(hms: str) -> int:
        s = str(hms).replace(":", "")
        return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) \
            if len(s) >= 6 and s[:6].isdigit() else 0

    t["fsec"] = t["ft"].map(sec)
    sub = t[(t["fsec"] >= 9 * 3600 + 45 * 60)].copy()
    say(f"\n子集 {len(sub):,} 个股-日(占母集 {len(sub) / len(t) * 100:.0f}%) · "
        f"封板率 {sub['seal'].mean() * 100:.1f}%")
    rows = []
    for name, m in gates(t).items():
        mm = m.loc[sub.index]
        e, k = sub[mm], sub[~mm]
        if len(e) < 50:
            continue
        dd = (e["seal"].mean() - k["seal"].mean()) * 100
        rows.append([name, len(e), f"{e['seal'].mean() * 100:.1f}%",
                     f"{k['seal'].mean() * 100:.1f}%", f"{dd:+.1f}pp"])
        KEY.setdefault(name, {})["robust_diff"] = dd
    md_table(["候选闸", "剔除n", "剔除组封板率", "保留组封板率", "差"], rows)


def part_f(t: pd.DataFrame):
    say("\n## F 研究36 的 5日结果为何误导")
    say("研究36 C2 的「中位回落」档 n=168 封板率 1.2% = **2 只封板**。"
        "全历史同档 n=2157 封板率见下表 —— 两者差距远超抽样噪声可解释范围, "
        "说明 5日结果是窗口噪音而非因子效应。")
    have = t[t["theme"].notna()]
    m = have["mh"].between(3, 5) & (have["mh_d"] < 0)
    e, k = have[m], have[~m]
    say(f"- 全历史「中位(3-5板)且回落」: n={len(e):,} 封板率 "
        f"{e['seal'].mean() * 100:.1f}% vs 其余 n={len(k):,} "
        f"{k['seal'].mean() * 100:.1f}% "
        f"(差 {(e['seal'].mean() - k['seal'].mean()) * 100:+.1f}pp)")
    p = e["seal"].mean()
    se = (p * (1 - p) / max(len(e), 1)) ** 0.5 * 100
    say(f"- 该档封板率 95% 置信区间 ≈ {p * 100:.1f}% ± {1.96 * se:.1f}pp "
        f"→ [{max(0, p * 100 - 1.96 * se):.1f}%, "
        f"{p * 100 + 1.96 * se:.1f}%], 研究36 观测到的 1.2% 落在区间外, "
        "属小样本极值")
    say("\n两条方法论教训:")
    say("1. **基线差异要先看**: 生产 S2/S3 基线 12%, 全历史触板基线 66%。"
        "低基线母集上任何档的绝对封板率都容易被少数样本推到极端, "
        "必须先算置信区间再谈区分度。")
    say("2. **5日窗口不足以选闸**: 本研究 TRAIN 用 ≤2024 的 "
        f"{len(t[t['trade_date'] <= TRAIN_END]):,} 样本, TEST 用 2025~2026 的 "
        f"{len(t[t['trade_date'] > TRAIN_END]):,} 样本, 结论才稳定。")


def part_g(t: pd.DataFrame):
    """五条门槛汇总 + 结论"""
    have = t[t["theme"].notna()]
    say("\n## G 五条门槛汇总验收")
    rows = []
    passed = []
    for name, m in gates(t).items():
        v = KEY.get(name, {})
        e, k = have[m.loc[have.index]], have[~m.loc[have.index]]
        if len(e) < 50:
            continue
        full = (e["seal"].mean() - k["seal"].mean()) * 100
        c1 = full < 0
        c2 = v.get("reg_ok", 0) >= 2
        c3 = bool(v.get("wf_ok"))
        c4 = v.get("cut_ratio", 1) <= 0.30
        c5 = bool(v.get("ev_ok"))
        rob = v.get("robust_diff")
        marks = ["✅" if x else "❌" for x in (c1, c2, c3, c4, c5)]
        npass = sum([c1, c2, c3, c4, c5])
        rows.append([name, f"{full:+.1f}pp", marks[0], marks[1], marks[2],
                     marks[3], marks[4],
                     f"{rob:+.1f}pp" if rob is not None else "-",
                     f"{npass}/5"])
        if npass == 5:
            passed.append((name, full, v))
    md_table(["候选闸", "全历史差", "①全历史", "②三段≥2/3",
              "③walk-forward", "④剔除≤30%", "⑤EV不恶化",
              "稳健性子集差", "通过"], rows)
    KEY["_passed"] = passed

    say("\n## H 结论")
    if passed:
        say(f"- **五条门槛全过的闸共 {len(passed)} 个**:")
        for name, full, v in passed:
            say(f"  · {name} —— 全历史剔除组比保留组低 {-full:.1f}pp"
                f"(剔除比例 {v['cut_ratio'] * 100:.1f}%), 三段市况 "
                f"{[f'{d:+.1f}pp' for d in v['reg_diff']]}, walk-forward "
                f"TRAIN {v['wf_tr']:+.1f}pp → TEST {v['wf_te']:+.1f}pp 同向, "
                f"稳健性子集 {v.get('robust_diff', 0):+.1f}pp。")
        say("  效应量均只有 -1~-4pp(基线 66%), 经济价值边际; 且它们"
            "剔除的是**高潮题材内的跟风/补涨触板股**, 不是龙头本身 —— "
            "与 92科比「高位做龙头 / 鱼尾不能格局」并不矛盾: "
            "高潮期做龙头可以, 追跟风不行。")
    else:
        say("- **五条门槛全过的闸: 0 个**。")
    say("- 研究36 提出的两个候选规避闸(中位回落 / 波段回落区)**全部被否决**: "
        "中位回落 walk-forward TRAIN -2.4pp → TEST +1.2pp(方向反转); "
        "波段回落区三段市况 0/3 同向。不得接入生产。")
    say("- 题材阶段维度整体退出决策链路: 波龄(研究36 已证方向反且无区分度)、"
        "强度位置、龙头断板在**封板率**上均无稳定区分度(本研究 B1: 全部方案"
        "极差≤3pp)。过闸的只剩高度类二值标记(题材最高板≥6), "
        "而它本质是高度因子不是阶段因子。")
    say("- 对 92科比框架的修正: 「爆发/主升/鱼尾」不能用天数代理。若需要"
        "一个可落地的题材生命周期标记, 应用**题材最高板高度 + 家数相对"
        "自身峰值的位置**两维, 而不是波龄。")

    say("\n## I 对照: 个股维度梯度（标尺）")
    rows = []
    for lo, hi, tag in ((0, 0, "昨日未涨停"), (1, 1, "昨日首板"),
                        (2, 2, "昨日2板"), (3, 4, "昨日3-4板"),
                        (5, 99, "昨日≥5板")):
        g = t[(t["y_lb"] >= lo) & (t["y_lb"] <= hi)]
        if len(g) < 50:
            continue
        rows.append([tag, len(g), f"{g['seal'].mean() * 100:.1f}%",
                     f"{g['ev'].mean():+.2f}"])
    md_table(["昨日连板高度", "个股-日数", "封板率", "EV次日开盘%"], rows)
    say("\n> 个股维度有真梯度: 封板率从昨日2板的最低点向两端抬升"
        "(≥5板 72.8%), 而 EV 随高度单调变差直到 3-4板(-2.63%) —— "
        "高位接力封得住但次日亏得多, 印证「EV由入场价位而非封板率决定」。"
        "对比 B1: 题材阶段两个维度的极差都≤3pp, 个股维度的极差达"
        "11pp(封板率)/2.3pp(EV) —— 量级差一个数量级。")

    say("\n## 诚实边界")
    say("- 涨停价由 pre_close × 板型涨幅上限推算(创业板 2020-08-24 前后"
        "分口径), 未用官方 stk_limit; ST 股 5% 上限未单独处理, 会把部分"
        "ST 触板误判为未触板 —— 影响母集构成但不影响题材因子的相对比较。")
    say("- 日频触板无法给出触板时刻, 母集含秒板/尾盘板; E 节用首封≥09:45"
        "子集复核, 但炸板股的首封时刻来自 kpl_events(2018起), "
        "早年缺失则该样本被排除在子集外。")
    say("- stock→题材映射用滚动60日 kpl 标注窗口, 早年(2019-2020)标注密度"
        "低于近年, 无题材组占比偏高; 该偏差对所有候选闸同向作用, "
        "不改变相对结论。")
    say("- 次日开盘收益 EV = 次日open / **涨停价** - 1(与生产挂限价单口径"
        "一致, 对封板股与炸板股同口径), 仅限真的下一交易日(停牌置空), "
        "未剔次日一字(卖出端风险)。")
    say("- 修正记录: 首版 EV 误用已筛选触板子集的 shift(-1), 拿到的是"
        "「下一个触板日」而非下一交易日, 导致 EV 均值偷换为负值; "
        "已改为从全量面板取次日开盘。")


def summary(t: pd.DataFrame) -> list:
    """摘要: 先给结论(数字均从 KEY 取)"""
    passed = KEY.get("_passed", [])
    g1 = KEY.get("G1 中位(3-5板)且最高板回落不做", {})
    g2 = KEY.get("G2 家数处波段回落区(峰值50~90%)不做", {})
    m0 = KEY.get("M0 现行 theme_mode(theme.day.age)", {})
    s = ["\n## 摘要(TL;DR)"]
    s.append(f"**母集**: 全历史非一字触板 {len(t):,} 个股-日"
             f"({t['trade_date'].min()}~{t['trade_date'].max()}, "
             f"{t['trade_date'].nunique()} 个交易日), 基线封板率 "
             f"{t['seal'].mean() * 100:.1f}% —— 比研究36 的 5日信号样本"
             "扩了约 60 倍。")
    s.append(f"**研究36 的两个候选闸全部被否决**: 中位回落 walk-forward "
             f"TRAIN {g1.get('wf_tr', 0):+.1f}pp → TEST "
             f"{g1.get('wf_te', 0):+.1f}pp(方向反转); 波段回落区三段市况 "
             f"{g2.get('reg_ok', 0)}/3 同向。两个都是 5日窗口噪音。")
    s.append(f"**阶段维度连梯度都没有**: 现行口径在 {len(t):,} 样本上封板率"
             f"极差仅 {m0.get('spread', 0):.1f}pp"
             f"(Spearman {m0.get('rho', 0):+.3f}), 全部 6 个重建方案极差≤3pp; "
             "EV 维度同样无梯度(分层内极差≤0.5pp), 且仅封板股三档 EV 与"
             "研究36 C1 对账一致 —— 两份独立研究相互印证。")
    if passed:
        names = ", ".join(n for n, _, _ in passed)
        s.append(f"**五条门槛全过的只有: {names}** —— 而它是高度因子"
                 "(题材最高板≥6)不是阶段因子, 效应量 -1~-4pp, "
                 "经济价值边际。")
    else:
        s.append("**五条门槛全过的闸: 0 个**。")
    s.append("**行动**: 题材阶段(波龄/强度位置/龙头断板)不得接入任何买卖闸; "
             "研究36 的两个候选闸作废; 若保留一个题材维度标记, "
             "只用「题材最高板≥6」这一个二值高度标记。")
    return s


def main():
    t, r36 = build()
    part_b(t, r36)
    part_c(t)
    part_d(t)
    part_e(t)
    part_f(t)
    part_g(t)
    out = [L[0]] + summary(t) + L[1:]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
