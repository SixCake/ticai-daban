# -*- coding: utf-8 -*-
"""研究50: E3 封单续持判据增量消融（决策链路 P0-b · 盈亏比优先）

研究41 证明 E3 封单超预期(seal_ratio = fd_amount/amount)对封板票次日开盘
**胜率**有 +23.9pct 区分度(三段3/3)。本研究决定**盈亏比**维度(用户定盈亏比
优先): E3 高的封板票「续持到次日开盘卖」的盈亏比是否 > E3 低的 → 若是,
则 clear_unsealed 可用 E3 筛「哪些封板票值得续持」(放大盈利单, 环节5)。

母集: 非一字封板票(events_enriched) · 兑现: 次日开盘卖 next_open_ret
判据: **盈亏比优先** —— E3 高档盈亏比 > 低档, 牛熊震荡三段方向一致 ≥2/3

⚠ 时序: E3(封单额)是封板后才有的 → 只能用于续持/卖出判据, 不作买入因子。
用法: python research/50_e3_hold.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load                                  # noqa: E402
from core.exit_rules import pl_stats                        # noqa: E402

ENV_SEG = [("熊市", "20220101", "20221231"),
           ("震荡市", "20230101", "20240930"),
           ("牛市", "20241001", "20261231")]
HDR = f"{'档':<6}{'样本':>8}{'胜率%':>8}{'盈亏比':>8}{'期望%':>8}"


def tercile(s: pd.Series) -> pd.Series:
    r = s.rank(method="first", pct=True)
    return pd.cut(r, [0, 1 / 3, 2 / 3, 1.0],
                  labels=["低", "中", "高"], include_lowest=True)


def main():
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "is_yizi", "fd_amount",
                       "amount", "next_open_ret"])
    ev = ev[~ev["is_yizi"]].copy()                 # 非一字(可买到)
    ev["E3"] = np.where(ev["amount"] > 0, ev["fd_amount"] / ev["amount"],
                        np.nan)
    ev = ev.dropna(subset=["E3", "next_open_ret"])
    ev["ret"] = ev["next_open_ret"] * 100          # 百分比口径(同 pl_stats)

    print("# 研究50: E3 封单续持判据增量消融（盈亏比优先）")
    print(f"\n母集 非一字封板票 {len(ev):,} · "
          f"{ev['trade_date'].min()}~{ev['trade_date'].max()}")
    n, wr, aw, al, ratio, exp = pl_stats(list(ev["ret"]))
    print(f"基线(全部续持·次日开盘卖): {n} 样本 胜率{wr*100:.1f}% "
          f"盈亏比{ratio:.2f} 期望{exp:+.2f}")

    print(f"\n## A 全样本 E3 三分位 → 次日开盘卖盈亏比")
    print(HDR)
    ev["_bin"] = tercile(ev["E3"])
    ratios = {}
    for bn in ["低", "中", "高"]:
        g = ev[ev["_bin"] == bn]
        n, wr, aw, al, ratio, exp = pl_stats(list(g["ret"]))
        ratios[bn] = ratio
        print(f"{bn:<6}{n:>8}{wr*100:>8.1f}{ratio:>8.2f}{exp:>8.2f}")
    print(f"  → 高−低盈亏比差 {ratios['高'] - ratios['低']:+.2f} "
          f"({'E3续持判据有增量' if ratios['高'] > ratios['低'] else '无增量'})")

    print(f"\n## B 三段市况: E3高档盈亏比 vs 低档（方向一致 ≥2/3）")
    ok = 0
    for seg, a, b in ENV_SEG:
        g0 = ev[(ev["trade_date"] >= a) & (ev["trade_date"] <= b)].copy()
        if len(g0) < 90:
            print(f"  {seg}: 样本不足")
            continue
        g0["_bin"] = tercile(g0["E3"])
        hi = g0[g0["_bin"] == "高"]["ret"]
        lo = g0[g0["_bin"] == "低"]["ret"]
        _, _, _, _, rh, eh = pl_stats(list(hi))
        _, _, _, _, rl, el = pl_stats(list(lo))
        if rh is None or rl is None:
            print(f"  {seg}: 样本不足")
            continue
        flag = "✓" if rh > rl else "✗"
        ok += rh > rl
        print(f"  {seg}: 高档盈亏比{rh:.2f}(期望{eh:+.2f}) vs "
              f"低档{rl:.2f}(期望{el:+.2f}) {flag}")
    print(f"\n方向一致性: {ok}/3 段「E3高档盈亏比 > 低档」 "
          f"→ {'通过' if ok >= 2 else '不通过'}")

    # 落地判据: E3 低档若期望显著为负 → 这些封板票应在尾盘卖(不续持)
    lo_all = ev[ev["_bin"] == "低"]["ret"]
    _, _, _, _, _, el = pl_stats(list(lo_all))
    print(f"\n## C 落地判据")
    print(f"  E3 低档(全样本)次日开盘卖期望 {el:+.2f}% → "
          + ("应在尾盘卖(不续持), 避免次日低溢价" if el < 0
             else "续持仍正, E3判据仅用于排序"))


if __name__ == "__main__":
    main()
