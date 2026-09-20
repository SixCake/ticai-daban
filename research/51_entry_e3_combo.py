# -*- coding: utf-8 -*-
"""研究51: 低位入场 + E3高续持 组合的整体盈亏比（盈亏比3+路径验证）

核心问题: 盈亏比3+唯一达标口径是E3高档封板续持(研究50 3.19), 但那是"封板票
子集"。本研究决定实盘可执行组合「低位入场(pct<4) + 封板E3高续持 + 不封板
当日离场」的策略整体盈亏比能否达3。

用法: python research/51_entry_e3_combo.py
"""
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load                                  # noqa: E402
from core.exit_rules import pl_stats                        # noqa: E402

ev = load("limitup.events_enriched",
          columns=["trade_date", "ts_code", "fd_amount", "amount",
                   "next_open_ret"])
ev["E3"] = np.where(ev["amount"] > 0, ev["fd_amount"] / ev["amount"], np.nan)
evmap = {(str(r.trade_date), r.ts_code): (r.E3, r.next_open_ret)
         for r in ev.itertuples()}
spec = importlib.util.spec_from_file_location(
    "r46", Path(__file__).resolve().parent / "46_entry_exit.py")
r46 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r46)
exit_full = r46.exit_full

base, combo = [], []
hold_n = 0
for f in glob.glob(str(ROOT / "data" / "live" / "presig_state_*.json")):
    date = re.search(r"(\d{8})", f).group(1)
    d = json.load(open(f))
    for s in d["signals"]:
        if s.get("stage") not in ("S2", "S3") or not s.get("pb"):
            continue
        if (s.get("pct") or 0) >= 4:
            continue
        r, why, h = exit_full(s)
        if r is None:
            continue
        base.append(r)
        if bool(s.get("zt_shape")):
            e3, nor = evmap.get((date, s.get("ts_code")), (None, None))
            if e3 is not None and e3 >= 0.20 and nor is not None:
                combo.append(nor * 100)
                hold_n += 1
                continue
        combo.append(r)

n, wr, aw, al, ratio, exp = pl_stats(base)
print(f"低位(pct<4)基线(当日离场): {n}样本 胜率{wr*100:.1f}% "
      f"平均盈利{aw:+.2f} 平均亏损{al:+.2f} 盈亏比{ratio:.2f} 期望{exp:+.2f}")
n2, wr2, aw2, al2, ratio2, exp2 = pl_stats(combo)
print(f"低位+E3高续持组合:        {n2}样本 胜率{wr2*100:.1f}% "
      f"平均盈利{aw2:+.2f} 平均亏损{al2:+.2f} 盈亏比{ratio2:.2f} 期望{exp2:+.2f}")
print(f"  (其中 E3高续持 {hold_n} 条)")
print(f"  -> 盈亏比3+目标: {'达标' if ratio2 >= 3 else '未达标'}")
