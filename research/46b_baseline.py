# -*- coding: utf-8 -*-
"""研究46b: 盈亏比 baseline 多日汇总（决策链路 P1-a）

循环全部 presig_state_*.json，复用 research/46 的 exit_full 口径汇总：
  · 全体成交样本盈亏比（决策链路 baseline）
  · 按入场价位档 / 离场原因 的盈亏比 —— 找盈亏比最低环节 = 优化靶子
  · 止损反事实（止损离场 vs 不止损持到收盘）多日 diff
  · P2 反事实 多日 diff
判据用**盈亏比优先**（用户定）。口径同 research/46（core/exit_rules），
复用其 exit_full 避免口径分叉。

⚠ 局限: px_hist 只录 pct≥1%或prob≥0.2 的票, 部分票轨迹 14:00 前就断 →
  离场含"轨迹不足回退收盘", 非完整策略回测(同研究46)。

用法: python research/46b_baseline.py
"""
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA                                     # noqa: E402
from core.exit_rules import pl_stats                        # noqa: E402

# 复用 research/46 的 exit_full（文件名数字开头, 用 importlib 加载）
_spec = importlib.util.spec_from_file_location(
    "r46", Path(__file__).resolve().parent / "46_entry_exit.py")
_r46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r46)
exit_full = _r46.exit_full

LIVE = DATA / "live"
HDR = f"{'组':<18}{'样本':>6}{'胜率%':>8}{'盈亏比':>8}{'期望%':>8}"


def pct_tier(pct):
    if pct is None:
        return None
    if pct < 4:
        return "1_<4%"
    if pct < 6:
        return "2_4-6%"
    if pct < 10:
        return "3_6-10%"
    return "4_≥10%"


def line(name, rets):
    n, wr, aw, al, ratio, exp = pl_stats(rets)
    if not n:
        return f"{name:<18}{0:>6}{'-':>8}{'-':>8}{'-':>8}"
    return (f"{name:<18}{n:>6}{wr*100:>8.1f}"
            f"{(ratio if ratio else 0):>8.2f}{exp:>8.2f}")


def main():
    dates = sorted(re.search(r"(\d{8})", f.stem).group(1)
                   for f in LIVE.glob("presig_state_*.json"))
    all_rets, by_tier, by_why = [], defaultdict(list), defaultdict(list)
    sl_rule, sl_hold, p2_rule, p2_hold = [], [], [], []
    for d in dates:
        data = json.load(open(LIVE / f"presig_state_{d}.json", encoding="utf-8"))
        for s in data["signals"]:
            if s.get("stage") not in ("S2", "S3") or not s.get("pb"):
                continue
            r, why, hold = exit_full(s)
            if r is None:
                continue
            all_rets.append(r)
            by_tier[pct_tier(s.get("pct"))].append(r)
            by_why[why].append(r)
            if why == "回落止损5%":
                sl_rule.append(r)
                sl_hold.append(hold)
            elif why == "P2 10:10未封板":
                p2_rule.append(r)
                p2_hold.append(hold)

    print(f"# 盈亏比 baseline 多日汇总（决策链路 P1-a）")
    print(f"\n窗口 {dates[0]}~{dates[-1]} · {len(dates)} 日 · "
          f"成交样本 {len(all_rets)} 条")

    print(f"\n## ① 全体 baseline（盈亏比优先判据）")
    print(HDR)
    print(line("全体成交", all_rets))

    print(f"\n## ② 按入场价位档（找靶子: 研究08 低位入场盈亏比应更高）")
    print(HDR)
    for k in sorted(by_tier):
        print(line(k, by_tier[k]))

    print(f"\n## ③ 按离场原因（找靶子: 盈亏比最低环节）")
    print(HDR)
    for why in sorted(by_why, key=lambda w: -len(by_why[w])):
        print(line(why, by_why[why]))

    print(f"\n## ④ 止损反事实（止损离场 vs 不止损持到收盘, 多日）")
    print(line("止损离场", sl_rule))
    print(line("不止损持到收盘", sl_hold))
    _, _, _, _, _, er = pl_stats(sl_rule)
    _, _, _, _, _, eh = pl_stats(sl_hold)
    if er is not None and eh is not None:
        print(f"  → 期望差 {er-eh:+.2f}pp: "
              + ("止损有效(避免更大亏损)" if er - eh > 0
                 else "止损误杀(持到收盘更好)"))

    print(f"\n## ⑤ P2 反事实（P2离场 vs 不止损持到收盘, 多日）")
    print(line("P2 10:10离场", p2_rule))
    print(line("不止损持到收盘", p2_hold))
    _, _, _, _, _, pr = pl_stats(p2_rule)
    _, _, _, _, _, ph = pl_stats(p2_hold)
    if pr is not None and ph is not None:
        print(f"  → 期望差 {pr-ph:+.2f}pp: "
              + ("P2早砍有效" if pr - ph > 0 else "P2早砍错失"))


if __name__ == "__main__":
    main()
