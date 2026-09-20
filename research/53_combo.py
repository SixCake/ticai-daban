# -*- coding: utf-8 -*-
"""研究53: 决策链路三杠杆组合的整体盈亏比验证

三杠杆(均已落地 strategy.py):
  ① 入场价位闸 pct<4 (研究49: 整体0.99→1.34)
  ② E3续持闸: 封板票E3≥0.20续持(研究51: 盈亏比3.74), E3低尾盘卖
  ③ 不止损: 不封板票持到收盘(研究52: 盈亏比1.31, 回撤-12.2%)

本研究验证三杠杆叠加后的**策略整体盈亏比**(presig 23日)。
判据: 盈亏比优先, 对比 baseline(无任何闸)。

用法: python research/53_combo.py
"""
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.exit_rules import pl_stats                        # noqa: E402

e3 = {(r.trade_date, r.ts_code): (r.E3, r.next_open_ret)
      for r in pd.read_csv("/tmp/e3.csv",
                           dtype={"trade_date": str}).itertuples()}
spec = importlib.util.spec_from_file_location(
    "r46", ROOT / "research" / "46_entry_exit.py")
r46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r46)
exit_full = r46.exit_full

base, combo = [], []
for f in glob.glob(str(ROOT / "data" / "live" / "presig_state_*.json")):
    date = re.search(r"(\d{8})", f).group(1)
    d = json.load(open(f))
    for s in d["signals"]:
        if s.get("stage") not in ("S2", "S3") or not s.get("pb"):
            continue
        ret, why, hold = exit_full(s)
        if ret is None:
            continue
        base.append(ret)                     # baseline: 无闸, 止损5%离场
        if (s.get("pct") or 0) >= 4:         # ① 入场价位闸
            continue
        if s.get("zt_shape"):                # 封板票
            t = e3.get((date, s.get("ts_code")))
            if t and t[0] >= 0.2 and t[1] == t[1]:   # ② E3高续持
                combo.append(t[1] * 100)
                continue
            combo.append(hold if hold is not None else ret)  # E3低尾盘卖
            continue
        combo.append(hold if hold is not None else ret)      # ③ 不止损持到收盘

n, wr, aw, al, ratio, exp = pl_stats(base)
print(f"baseline(无闸):        {n}样本 胜率{wr*100:.1f}% "
      f"盈亏比{ratio:.2f} 期望{exp:+.2f}")
n2, wr2, aw2, al2, ratio2, exp2 = pl_stats(combo)
print(f"三杠杆组合(决策链路):  {n2}样本 胜率{wr2*100:.1f}% "
      f"盈亏比{ratio2:.2f} 期望{exp2:+.2f}")
print(f"  → 整体盈亏比提升: {ratio:.2f} → {ratio2:.2f} "
      f"(+{(ratio2/ratio-1)*100:.0f}%)")
