# -*- coding: utf-8 -*-
"""研究41: 预期差独立因子 —— 心法⑬量化

心法⑬原文: "抓预期差, 找情绪锚定点, 是提高盈亏比的核心。" 心法㉒:
"上涨和下跌都是表象, 对未来的预期才是交易的本质。"

本研究把"预期差"从题材热度里**独立提炼**成白盒因子, 并做一个关键对照:
**预期差(跃升/超额) 是否比 绝对水平 更有信息?** —— 因为心法说的是"抓
预期差"(边际变化), 而不是"看热度"(存量水平)。若跃升组区分度 > 绝对组,
则证实"预期差"是独立的盈亏比来源, 值得单列成因子。

预期差因子（全部 T 日盘前/盘中可知, 预测 T 日涨停票 T+1 兑现, 无前视）:
  E0 题材绝对热度  zt_all(T)              ← 对照(存量水平, 非预期差)
  E1 题材热度跃升  zt_all(T) − zt_all(T−1) ← 预期差核心(边际升温)
       个股→题材用 theme.attribution 独占归属; 昨日不在场记 0(冷门→今日爆发)
  E2 个股开盘超额  open_ret − 当日涨停池 open_ret 中位数
       开盘就超预期 = 竞价资金给出的预期差(百分比口径)
  E3 封单超预期    fd_amount / amount(封单额/成交额)
       封单远超成交 = 一致预期强度(锚定点强度)
  E4 组合          E1+E2+E3 三分位等权打分

判据:
  · 单调性 —— 预期差越大, T+1 胜率/均值越高
  · 区分度 —— 高三分位胜率 − 低三分位胜率
  · 关键对照 —— E1(跃升) vs E0(绝对) 谁区分度更大
  · 三段市况(熊/震荡/牛)同向 ≥2/3(用户强制方法论)
  · 兑现口径: 非一字涨停票 T+1 开盘卖 next_open_ret(研究28/40 同口径)

产物: research/out/41_expectation_gap.md
用法: python research/41_expectation_gap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out" / "41_expectation_gap.md"
ENV_SEG = [("全样本", "20190101", "20261231"),
           ("熊市", "20220101", "20221231"),
           ("震荡市", "20230101", "20240930"),
           ("牛市", "20241001", "20261231")]
FACTORS = ["E0", "E1", "E2", "E3"]
FACTOR_NAME = {"E0": "题材绝对热度(对照)", "E1": "题材热度跃升(预期差)",
               "E2": "个股开盘超额", "E3": "封单超预期", "E4": "组合"}
L: list = []
KEY: dict = {}


def say(s=""):
    print(s, flush=True)
    L.append(s)


def md_table(header, rows):
    say("| " + " | ".join(str(x) for x in header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")


def tercile(s: pd.Series) -> pd.Series:
    """三分位分箱(低/中/高)。rank(method=first) 破 ties, 保证三箱样本均衡"""
    r = s.rank(method="first", pct=True)
    return pd.cut(r, [0, 1 / 3, 2 / 3, 1.0],
                  labels=["低", "中", "高"], include_lowest=True)


# ============================================================ 因子构建
def theme_expectation() -> pd.DataFrame:
    """题材级 (trade_date, concept_code) → zt_all(T), dzt(跃升)"""
    td = load("theme.day",
              columns=["trade_date", "concept_code", "zt_all"])
    cal = sorted(td["trade_date"].unique())
    prev = {cal[i]: cal[i - 1] for i in range(1, len(cal))}
    zt_by = {(r.trade_date, r.concept_code): r.zt_all
             for r in td.itertuples()}
    zt_prev, dzt = [], []
    for r in td.itertuples():
        p = zt_by.get((prev.get(r.trade_date), r.concept_code), 0)  # 昨日不在场=0
        zt_prev.append(p)
        dzt.append(r.zt_all - p)
    td["zt_prev"] = zt_prev
    td["dzt"] = dzt
    return td[["trade_date", "concept_code", "zt_all", "dzt"]]


def build_factors() -> pd.DataFrame:
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "is_yizi", "fd_amount",
                       "amount", "next_open_ret"])
    ev = ev[~ev["is_yizi"]].copy()               # 非一字涨停票(可买到)
    attr = load("theme.attribution",
                columns=["trade_date", "ts_code", "concept_code"])
    td = theme_expectation()
    dp = load("market.daily_panel",
              columns=["trade_date", "ts_code", "open_ret"])

    eg = ev.merge(attr, on=["trade_date", "ts_code"], how="left")
    eg = eg.merge(td, on=["trade_date", "concept_code"], how="left")
    eg = eg.merge(dp, on=["trade_date", "ts_code"], how="left")
    # 题材缺失(无独占归属/题材不在场) → 冷门记 0
    eg["zt_all"] = eg["zt_all"].fillna(0)
    eg["dzt"] = eg["dzt"].fillna(0)
    # E2 开盘超额: 个股 open_ret − 当日涨停池 open_ret 中位数
    med = eg.groupby("trade_date")["open_ret"].median()
    eg["open_excess"] = eg["open_ret"] - eg["trade_date"].map(med)
    # E3 封单超预期: 封单额/成交额(amount>0 才有意义)
    eg["seal_ratio"] = np.where(eg["amount"] > 0,
                                eg["fd_amount"] / eg["amount"], np.nan)
    eg = eg.rename(columns={"zt_all": "E0", "dzt": "E1",
                            "open_excess": "E2", "seal_ratio": "E3"})
    eg = eg.dropna(subset=["next_open_ret", "E2"])
    return eg


def combo_score(eg: pd.DataFrame) -> pd.Series:
    """E4 组合: E1+E2+E3 三分位等权打分(低0/中1/高2 求和)"""
    sc = pd.Series(0.0, index=eg.index)
    for f in ["E1", "E2", "E3"]:
        r = eg[f].rank(method="first", pct=True)
        pt = np.where(r <= 1 / 3, 0.0, np.where(r <= 2 / 3, 1.0, 2.0))
        sc = sc + pt
    return sc


# ============================================================ 验证
def stats(g: pd.DataFrame) -> dict:
    return {"n": len(g),
            "胜率%": round((g["next_open_ret"] > 0).mean() * 100, 1),
            "均值%": round(g["next_open_ret"].mean() * 100, 2)}


def part_box(eg: pd.DataFrame):
    say("# 研究41: 预期差独立因子（跃升/超额 vs 绝对水平）\n")
    say(f"母集=非一字涨停票 {len(eg):,} · {eg['trade_date'].min()}"
        f"~{eg['trade_date'].max()} · 兑现=T+1 开盘卖 next_open_ret · "
        f"个股→题材用 theme.attribution 独占归属 · "
        f"基线 T+1 胜率 {(eg['next_open_ret'] > 0).mean() * 100:.1f}%")
    eg = eg.copy()
    eg["E4"] = combo_score(eg)

    say("\n## A 各因子三分位分箱 → T+1 兑现（全样本）")
    for f in FACTORS + ["E4"]:
        eg["_bin"] = tercile(eg[f])
        rows = []
        for bn in ["低", "中", "高"]:
            g = eg[eg["_bin"] == bn]
            if len(g) >= 30:
                rows.append([bn] + list(stats(g).values()))
        hi = eg[eg["_bin"] == "高"]["next_open_ret"]
        lo = eg[eg["_bin"] == "低"]["next_open_ret"]
        gap = ((hi > 0).mean() - (lo > 0).mean()) * 100 \
            if len(hi) >= 30 and len(lo) >= 30 else np.nan
        KEY[f] = {"gap": gap}
        md_table([f"{f} {FACTOR_NAME[f]}", "样本", "胜率%", "均值%"], rows)
        say(f"> 高−低胜率差 **{gap:+.1f}pct**\n")


def part_env(eg: pd.DataFrame):
    say("\n## B 三段市况验收: 高三分位胜率 − 低三分位胜率（方向一致 ≥2/3）")
    eg = eg.copy()
    eg["E4"] = combo_score(eg)
    for f in FACTORS + ["E4"]:
        ok = 0
        rows = []
        for seg, a, b in ENV_SEG[1:]:
            g0 = eg[(eg["trade_date"] >= a) & (eg["trade_date"] <= b)].copy()
            if len(g0) < 90:
                rows.append([seg, "样本不足", "-", "-"])
                continue
            g0["_bin"] = tercile(g0[f])          # 段内三分位(避免跨段分位漂移)
            hi = g0[g0["_bin"] == "高"]["next_open_ret"]
            lo = g0[g0["_bin"] == "低"]["next_open_ret"]
            if len(hi) < 30 or len(lo) < 30:
                rows.append([seg, "样本不足", "-", "-"])
                continue
            gap = ((hi > 0).mean() - (lo > 0).mean()) * 100
            flag = "✓" if gap > 0 else "✗"
            ok += gap > 0
            rows.append([seg, f"{len(hi)}/{len(lo)}",
                         f"{gap:+.1f}pct", flag])
        say(f"\n### {f} {FACTOR_NAME[f]} —— 方向一致 {ok}/3 段")
        md_table(["市况", "高/低样本", "胜率差", "方向"], rows)
        if f in KEY:
            KEY[f]["reg_ok"] = ok


def summary():
    s = ["\n## 摘要(TL;DR)"]
    if not KEY:
        s.append("(无因子通过样本门槛)")
        return s
    s.append("**全样本区分度(高−低胜率差)**: " + " · ".join(
        f"{f} {KEY[f]['gap']:+.1f}pct" for f in KEY if "gap" in KEY[f]))
    e0 = KEY.get("E0", {}).get("gap", np.nan)
    e1 = KEY.get("E1", {}).get("gap", np.nan)
    if np.isfinite(e0) and np.isfinite(e1):
        verdict = ("**预期差(跃升 E1) 优于 绝对热度(E0)** —— 证实心法⑬"
                   "「抓预期差」是独立盈亏比来源, 边际升温比存量水平更有信息"
                   if e1 > e0 else
                   "**绝对热度(E0) 不劣于 预期差(跃升 E1)** —— 心法⑬的"
                   "「预期差」在此口径下未跑赢存量热度, 题材强度本身已含大部分信息")
        s.append(f"**关键对照**: E0 绝对 {e0:+.1f}pct vs E1 跃升 {e1:+.1f}pct。"
                 + verdict)
    best = max((f for f in KEY if "reg_ok" in KEY[f]),
               key=lambda f: (KEY[f]["reg_ok"], KEY[f]["gap"]), default=None)
    if best:
        s.append(f"**最优因子: {best} {FACTOR_NAME[best]}** —— 三段同向 "
                 f"{KEY[best]['reg_ok']}/3, 全样本区分度 {KEY[best]['gap']:+.1f}pct。")
    s.append("**边界**: 与 core/cycle 同原则 —— 通过验收的因子可作排序/展示"
             "叠加参照; 接入买卖闸前需与 V5 融合分做增量消融(研究24 口径), "
             "确认相对现有因子有独立增量再入闸。")
    return s


def main():
    eg = build_factors()
    part_box(eg)
    part_env(eg)
    out = [L[0]] + summary() + L[1:]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
