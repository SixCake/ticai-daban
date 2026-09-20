# -*- coding: utf-8 -*-
"""研究49: 入场价位闸门回测（决策链路 P0-a · 盈亏比优先）

靶子1（研究46b 诊断）: 盈亏比随入场价位单调递减 <4%=1.34 → ≥10%=0.14。
本回测验证"砍高位买入"能否把整体盈亏比拉高, 且方向三段稳健。

**双口径**（各有取舍, 互补）:
  A. presig 实盘信号(23日) —— 实际触发 pct + S2/S3 筛选, 最真实,
     但仅一段市况。方案对比: A0 无闸 / A1 pct≤4% / A2 pct≤6%。
  B. fulluniv_panel(345日) —— 无差别全宇宙, 各入场档(e3/e4/e5/e6)
     盈亏比 × 牛熊震荡三段, 验证"盈亏比随价位递减"方向稳健性。
     ⚠ 无差别宇宙幅度偏小(研究40 各档 0.77~0.84), 只验方向不验幅度。

判据: **盈亏比优先**（用户定）。离场口径 A 用 research/46 exit_full,
B 用 fulluniv_panel 内置 e{t}_ret（研究40 日线近似, 止损优先/封板续持）。

用法: python research/49_entry_gate.py
"""
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA                                     # noqa: E402
from core.exit_rules import pl_stats                        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "r46", Path(__file__).resolve().parent / "46_entry_exit.py")
_r46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r46)
exit_full = _r46.exit_full

LIVE = DATA / "live"
PANEL = DATA / "factor" / "fulluniv_panel.parquet"
HDR = f"{'方案/档':<20}{'样本':>7}{'胜率%':>8}{'盈亏比':>8}{'期望%':>8}{'封板率%':>9}"


def stat(name, rets, sealed=None):
    n, wr, aw, al, ratio, exp = pl_stats(rets)
    if not n:
        return f"{name:<20}{0:>7}{'-':>8}{'-':>8}{'-':>8}{'-':>9}"
    sr = (sum(sealed) / n * 100) if sealed is not None else 0.0
    return (f"{name:<20}{n:>7}{wr*100:>8.1f}"
            f"{(ratio if ratio else 0):>8.2f}{exp:>8.2f}{sr:>9.1f}")


def build_regimes() -> dict:
    """按月均全A中位涨幅三分位切牛/熊/震荡（同研究31/40）"""
    dp = pd.read_parquet(DATA / "market" / "1d" / "daily_panel.parquet",
                         columns=["trade_date", "pct_chg"])
    dp["date"] = dp["trade_date"].astype(str)
    mr = dp.groupby("date")["pct_chg"].median()
    mon = mr.groupby(mr.index.str[:6]).mean().sort_values()
    n = len(mon)
    bear = set(mon.index[:n // 3])
    bull = set(mon.index[-n // 3:])
    return {dt: ("熊" if dt[:6] in bear else "牛" if dt[:6] in bull else "震荡")
            for dt in mr.index}


def part_a():
    """A. presig 实盘信号闸门方案对比（最真实口径）"""
    print("=" * 62)
    print("A. presig 实盘信号(23日) —— 闸门方案对比（实际触发 pct）")
    print("=" * 62)
    dates = sorted(re.search(r"(\d{8})", f.stem).group(1)
                   for f in LIVE.glob("presig_state_*.json"))
    rows = []
    for d in dates:
        data = json.load(open(LIVE / f"presig_state_{d}.json", encoding="utf-8"))
        for s in data["signals"]:
            if s.get("stage") not in ("S2", "S3") or not s.get("pb"):
                continue
            r, why, _ = exit_full(s)
            if r is None:
                continue
            rows.append((s.get("pct"), r, bool(s.get("zt_shape"))))
    print(f"\n窗口 {dates[0]}~{dates[-1]} · 成交样本 {len(rows)}")
    print(HDR)
    schemes = [("A0 无闸(现状)", lambda p: True),
               ("A1 pct≤4%闸", lambda p: p is not None and p < 4),
               ("A2 pct≤6%闸", lambda p: p is not None and p < 6)]
    for name, fn in schemes:
        sub = [r for (p, r, sl) in rows if fn(p)]
        sl = [s for (p, r, s) in rows if fn(p)]
        print(stat(name, sub, sl))


def part_b():
    """B. fulluniv_panel 各入场档盈亏比 × 三段（方向稳健性）"""
    print("\n" + "=" * 62)
    print("B. fulluniv_panel(345日) —— 各入场档盈亏比 × 三段市况")
    print("=" * 62)
    d = pd.read_parquet(PANEL)
    reg = build_regimes()
    d["reg"] = d["trade_date"].astype(str).map(reg)
    print(f"\n窗口 {d['trade_date'].min()}~{d['trade_date'].max()} · "
          f"{len(d)} 票·日")
    print(f"\n{'档':<6}{'市况':<6}{'样本':>8}{'胜率%':>8}{'盈亏比':>8}"
          f"{'期望%':>8}{'封板率%':>9}")
    # 方向一致性: 每段内 e3盈亏比 是否 > e6盈亏比（低位优于高位）
    ok_env = 0
    for rg in ("牛", "熊", "震荡"):
        g = d[d["reg"] == rg]
        r3 = r6 = None
        for tgt in (3, 6):
            sub = g[g[f"e{tgt}_ok"] & g[f"e{tgt}_ret"].notna()]
            n, wr, aw, al, ratio, exp = pl_stats(list(sub[f"e{tgt}_ret"]))
            sr = sub["sealed"].mean() * 100
            print(f"e{tgt:<5}{rg:<6}{n:>8}{wr*100:>8.1f}"
                  f"{(ratio if ratio else 0):>8.2f}{exp:>8.2f}{sr:>9.1f}")
            if tgt == 3:
                r3 = ratio
            else:
                r6 = ratio
        if r3 is not None and r6 is not None and r3 > r6:
            ok_env += 1
            print(f"  → {rg}: e3盈亏比 {r3:.2f} > e6 {r6:.2f} ✓ 低位优于高位")
        else:
            print(f"  → {rg}: e3盈亏比 {r3} ≤ e6 {r6} ✗")
    print(f"\n方向一致性: {ok_env}/3 段「低位档盈亏比 > 高位档」 "
          f"→ {'通过' if ok_env >= 2 else '不通过'}")


if __name__ == "__main__":
    part_a()
    part_b()
