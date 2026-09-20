# -*- coding: utf-8 -*-
"""研究52: 止损宽度多方案验证（整体盈亏比最大化, 回撤≤20%）

靶子2(研究46b): 回落止损盈亏比0.33最低, 反事实证明止损误杀(止损-3.83 vs
不止损-2.24)。本研究决定最优止损宽度 —— 多方案并行(用户方法论)。

口径: presig 低位(pct<4)票, 用 px_hist 逐点模拟不同止损宽度 w 的离场:
  · 封板票: 续持(次日开盘卖近似, 用当日收盘 proxy)
  · 不封板票: 止损宽度 w 离场(触发止损) 或 14:55 清仓
判据: 整体盈亏比最大化 + 最大单笔亏损≤20%(用户回撤容忍)

用法: python research/52_stoploss.py
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.exit_rules import SEAL_EPS, FLAT_SEC, _sec, pl_stats  # noqa: E402


def sim_exit(pts, pb, lp, close, w):
    """逐点模拟止损宽度 w 的离场 → 收益%。封板续持到收盘, 不封板止损/清仓"""
    ref = pb
    stop = ref * (1 - w)
    sealed = False
    for t, px in pts:
        if px > ref:
            ref = px
            stop = ref * (1 - w)
        if px <= stop:
            return (px / pb - 1) * 100
        if lp > 0 and px >= lp * SEAL_EPS:
            sealed = True
        if _sec(t) >= FLAT_SEC:
            return (px / pb - 1) * 100
    return (close / pb - 1) * 100


def main():
    schemes = [("止损5%", 0.05), ("止损8%", 0.08),
               ("止损10%", 0.10), ("止损15%", 0.15), ("不止损", 1.0)]
    print("# 研究52: 止损宽度多方案(低位pct<4, 整体盈亏比+回撤)")
    print(f"{'方案':<10}{'样本':>7}{'胜率%':>8}{'盈亏比':>8}"
          f"{'期望%':>8}{'最大亏损%':>10}")
    for name, w in schemes:
        rets = []
        for f in glob.glob(str(ROOT / "data" / "live" /
                               "presig_state_*.json")):
            d = json.load(open(f))
            for s in d["signals"]:
                if s.get("stage") not in ("S2", "S3") or not s.get("pb"):
                    continue
                if (s.get("pct") or 0) >= 4:
                    continue
                ph = s.get("px_hist") or []
                pts = [(str(p[0]), float(p[1]))
                       for p in ph if len(p) >= 2 and p[1]]
                if len(pts) < 2:
                    continue
                close = pts[-1][1]
                r = sim_exit(pts, s["pb"], s.get("limit_px") or 0, close, w)
                rets.append(r)
        n, wr, aw, al, ratio, exp = pl_stats(rets)
        mx = min(rets) if rets else 0
        print(f"{name:<10}{n:>7}{wr*100:>8.1f}{ratio:>8.2f}"
              f"{exp:>8.2f}{mx:>10.1f}")


if __name__ == "__main__":
    main()
