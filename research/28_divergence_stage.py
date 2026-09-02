# -*- coding: utf-8 -*-
"""研究28: 四阶段仓位化引入分歧维度验证（买在分歧卖在一致）

研究27发现: 四阶段对T+1无稳定梯度, 但"冬"(高炸板=分歧释放)胜率反而高
→ 假设: 真正驱动T+1赚钱效应的是分歧/一致维度, 阶段需与之交互才有仓位意义

日频分歧/一致指标(涨停池内):
  br    = 炸板率 mean(open_times≥1)      分歧释放强度
  yizi  = 一字率 mean(is_yizi)           纯一致(买不到的共识)
  accel = 缩量加速率 mean(炸板0且首封≤09:45)  一致加速(追高危险区)
  cons  = yizi + accel                   一致强度

方案并行(用户方法论):
  V1 阶段B × br中位数切分: 同阶段内分歧日T+1应优于一致日
  V2 三分位组合: 高分歧(br上三分位) vs 高一致(cons上三分位且br下三分位)
     T+1胜率梯度应显著, 高一致=追高危险区(胜率最低)
  V3 阶段×分歧交互: 夏+分歧=最佳买点; 任何阶段+高一致=降仓区

验收: 三环境(熊2022/震荡2023-24Q3/牛2024-10~)方向一致≥2段,
      买点区胜率>50%且显著高于危险区
赚钱效应口径: 非一字涨停票 T+1 开盘卖 next_open_ret
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load  # noqa: E402

ENV_SEG = [("全样本", "20190101", "20261231"),
           ("熊市", "20220101", "20221231"),
           ("震荡市", "20230101", "20240930"),
           ("牛市", "20241001", "20260617")]
OUT = Path(__file__).resolve().parent / "out" / "28_divergence_stage.md"


def daily_indicators(ev):
    g = ev.groupby("trade_date")
    sent = pd.DataFrame({
        "zt": g.size(),
        "br": g["open_times"].apply(lambda s: (s >= 1).mean()),
        "mh": g["limit_times"].max(),
        "yizi": g["is_yizi"].mean(),
        "accel": g.apply(
            lambda d: ((d["open_times"] == 0)
                       & (d["first_time"].astype(str) <= "094500")).mean(),
            include_groups=False)})
    sent["cons"] = sent["yizi"] + sent["accel"]
    sent["zt_ma5"] = sent["zt"].rolling(5, min_periods=3).mean()
    br_q30, br_q70 = sent["br"].quantile(.3), sent["br"].quantile(.7)
    sb = []
    for d, r in sent.iterrows():     # 阶段B(分位阈值, 研究27方案B)
        if r["br"] > br_q70 or (r["zt"] < 0.8 * r["zt_ma5"]
                                and r["mh"] <= 3):
            sb.append("冬")
        elif r["br"] < br_q30 and r["mh"] >= 5 and r["zt"] > r["zt_ma5"]:
            sb.append("夏")
        elif r["mh"] >= 5:
            sb.append("秋")
        else:
            sb.append("春")
    sent["stage"] = sb
    return sent


def stats(g):
    return {"n": len(g),
            "均值%": round(g["next_open_ret"].mean() * 100, 2),
            "胜率%": round((g["next_open_ret"] > 0).mean() * 100, 1)}


def main():
    ev = load("limitup.events_enriched")
    sent = daily_indicators(ev)
    base = ev[~ev["is_yizi"]].merge(
        sent, left_on="trade_date", right_index=True)
    br_med = sent["br"].median()
    br_hi, br_lo = sent["br"].quantile(2 / 3), sent["br"].quantile(1 / 3)
    cons_hi = sent["cons"].quantile(2 / 3)
    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    pr("# 研究28: 四阶段×分歧维度 仓位化验证\n")
    pr(f"全期切分: br中位{br_med:.2f} br上三分位{br_hi:.2f} "
       f"cons上三分位{cons_hi:.2f}\n")

    # ---------- V1 阶段内分歧切分 ----------
    pr("## V1 同阶段内: 分歧日(br≥中位) vs 一致日 T+1")
    for (seg, a, b) in ENV_SEG:
        g0 = base[(base["trade_date"] >= a) & (base["trade_date"] <= b)]
        if not len(g0):
            continue
        pr(f"\n### {seg}")
        rows = []
        for st in ["春", "夏", "秋", "冬"]:
            for nm, mask in [("分歧", g0["br"] >= br_med),
                             ("一致", g0["br"] < br_med)]:
                g = g0[(g0["stage"] == st) & mask]
                if len(g) >= 30:
                    rows.append({"阶段": st, "日类型": nm, **stats(g)})
        if rows:
            pr(pd.DataFrame(rows).to_string(index=False))

    # ---------- V2 分歧/一致三分位组合 ----------
    pr("\n## V2 高分歧 vs 高一致(追高危险区) T+1")
    for (seg, a, b) in ENV_SEG:
        g0 = base[(base["trade_date"] >= a) & (base["trade_date"] <= b)]
        if not len(g0):
            continue
        divg = g0[g0["br"] >= br_hi]
        cons = g0[(g0["cons"] >= cons_hi) & (g0["br"] < br_lo)]
        mid = g0[(g0["br"] < br_hi) & ~((g0["cons"] >= cons_hi)
                                        & (g0["br"] < br_lo))]
        pr(f"\n### {seg}")
        pr(pd.DataFrame([{"组": "高分歧(买在分歧)", **stats(divg)},
                         {"组": "中性", **stats(mid)},
                         {"组": "高一致(追高危险)", **stats(cons)}])
           .to_string(index=False))

    # ---------- V3 阶段×分歧交互 ----------
    pr("\n## V3 交互: 夏+分歧最佳买点 / 高一致降仓区")
    for (seg, a, b) in ENV_SEG:
        g0 = base[(base["trade_date"] >= a) & (base["trade_date"] <= b)]
        if not len(g0):
            continue
        buy = g0[(g0["stage"].isin(["夏", "春"])) & (g0["br"] >= br_med)]
        hold = g0[(g0["stage"].isin(["夏", "春"])) & (g0["br"] < br_med)]
        rest = g0[(g0["stage"].isin(["秋", "冬"])) & (g0["br"] >= br_med)]
        danger = g0[(g0["cons"] >= cons_hi)]
        pr(f"\n### {seg}")
        pr(pd.DataFrame([
            {"区": "买点(春夏+分歧)", **stats(buy)},
            {"区": "持有(春夏+一致)", **stats(hold)},
            {"区": "观察(秋冬+分歧)", **stats(rest)},
            {"区": "降仓(高一致加速)", **stats(danger)}])
           .to_string(index=False))

    # ---------- 验收 ----------
    pr("\n## 验收: V2梯度(高分歧胜率-高一致胜率) 三环境方向")
    ok = 0
    for (seg, a, b) in ENV_SEG[1:]:
        g0 = base[(base["trade_date"] >= a) & (base["trade_date"] <= b)]
        divg = g0[g0["br"] >= br_hi]
        cons = g0[(g0["cons"] >= cons_hi) & (g0["br"] < br_lo)]
        if len(divg) < 30 or len(cons) < 30:
            pr(f"{seg}: 样本不足")
            continue
        w_d = (divg["next_open_ret"] > 0).mean()
        w_c = (cons["next_open_ret"] > 0).mean()
        flag = "✓" if w_d > w_c else "✗"
        ok += w_d > w_c
        pr(f"{seg}: 高分歧胜率{w_d * 100:.1f}% vs 高一致{w_c * 100:.1f}% "
           f"差{(w_d - w_c) * 100:+.1f}pct {flag}")
    pr(f"\n方向一致 {ok}/3 段 (验收≥2)")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {OUT}")


if __name__ == "__main__":
    main()
