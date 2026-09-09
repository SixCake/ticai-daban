# -*- coding: utf-8 -*-
"""研究46: 买卖点系统研究

买点: 触发时机(时刻) × 触发价位(pct) → 胜率/期望矩阵, 回答"何时何价买入"
卖点: 定稿离场规则(止损5%/P2 10:10/14:57强平/封板续持)的实际效果 ——
      各离场原因分布与收益, 并验证 **P2 10:10 砍未封板是否有效**
      (对比"P2离场" vs "不止损持有到收盘")

核心结论方向(项目定稿):
  · 打板EV由入场价位决定(研究08) → 买点看 pct 档
  · 定稿卖出端: 止损5% / P2 10:10未封板即卖 / 封板续持无上限
  · 本研究验证这套卖出规则在今日实盘的实际贡献

用法: python research/46_entry_exit.py [date YYYYMMDD]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from core.exit_rules import pl_stats, simulate_exit          # noqa: E402

LIVE = DATA / "live"


def cm20(code):
    return str(code)[:2] in ("30", "68")


def exit_full(s):
    """→ (规则离场收益%, 离场原因, 持有到收盘收益%)"""
    ph = s.get("px_hist") or []
    pts = [[p[0], p[1]] for p in ph if len(p) >= 2 and p[1]]
    if not pts or not s.get("pb"):
        return None, None, None
    pb = s["pb"]
    prices = [p[1] for p in pts]
    close_px = prices[-1]
    ret, why = simulate_exit(pts, pb, max(prices), s.get("limit_px") or 0,
                             close_px, hi_pts=[[p[0], p[1]] for p in pts])
    hold = (close_px / pb - 1) * 100          # 不止损, 持有到收盘
    return ret, why, hold


def stat_line(name, rets):
    nn, wr, aw, al, ratio, exp = pl_stats(rets)
    if not nn:
        return f"{name:<16}{0:>5}{'-':>7}{'-':>7}{'-':>7}"
    return (f"{name:<16}{nn:>5}{wr*100:>7.1f}"
            f"{(ratio if ratio else 0):>7.2f}{exp:>7.2f}")


HDR = f"{'组':<16}{'样本':>5}{'胜率%':>7}{'盈亏比':>7}{'期望%':>7}"


def analyze(date):
    d = json.load(open(LIVE / f"presig_state_{date}.json", encoding="utf-8"))
    sig = d["signals"]
    buyp = [s for s in sig if s.get("stage") in ("S2", "S3") and s.get("pb")]

    print(f"# 今日({date})买卖点研究")
    print(f"\n买点(S2/S3有买价) {len(buyp)} 条")

    # ---------- 买点: 触发时机 × 价位 ----------
    print(f"\n## 买点① 触发时机(信号触发时刻)")
    print(HDR)
    tg = defaultdict(list)
    for s in buyp:
        r, _, _ = exit_full(s)
        if r is None:
            continue
        t = str(s.get("t", "")).replace(":", "")
        lab = ("1_早盘≤10:00" if t <= "100000" else
               "2_10:00-11:30" if t <= "113000" else
               "3_午后13-14" if t < "140000" else "4_尾盘≥14:00")
        tg[lab].append(r)
    for k in sorted(tg):
        print(stat_line(k, tg[k]))

    print(f"\n## 买点② 触发价位(pct, 研究08: EV由入场价位决定)")
    print(HDR)
    pg = defaultdict(list)
    for s in buyp:
        r, _, _ = exit_full(s)
        if r is None:
            continue
        pct = s.get("pct") or 0
        lab = ("1_<4%" if pct < 4 else "2_4-6%" if pct < 6 else
               "3_6-10%" if pct < 10 else "4_≥10%(高位/20cm)")
        pg[lab].append(r)
    for k in sorted(pg):
        print(stat_line(k, pg[k]))

    # ---------- 卖点: 离场原因 ----------
    print(f"\n## 卖点① 定稿离场规则分布(各原因收益)")
    print(HDR)
    wg = defaultdict(list)
    for s in buyp:
        r, why, _ = exit_full(s)
        if r is not None:
            wg[why].append(r)
    for why in sorted(wg, key=lambda w: -len(wg[w])):
        print(stat_line(why, wg[why]))

    # ---------- 卖点: P2 有效性验证 ----------
    print(f"\n## 卖点② P2 10:10 砍未封板 有效性验证")
    print("  对比[P2离场] vs [不止损持有到收盘] —— 若P2更优, 证明早砍有效")
    p2 = [s for s in buyp if exit_full(s)[1] == "P2 10:10未封板"]
    p2_rule = [exit_full(s)[0] for s in p2]
    p2_hold = [exit_full(s)[2] for s in p2]
    print(f"  P2离场样本 {len(p2)} 条:")
    print(f"    [P2 10:10离场]  " + stat_line("", p2_rule).strip())
    print(f"    [持有到收盘]    " + stat_line("", p2_hold).strip())
    n, wr, aw, al, ratio, exp_r = pl_stats(p2_rule)
    _, _, _, _, _, exp_h = pl_stats(p2_hold)
    if exp_r is not None and exp_h is not None:
        diff = exp_r - exp_h
        verdict = "P2早砍有效(避免后续下跌)" if diff > 0 \
            else "P2早砍反而错失(持有到收盘更好)"
        print(f"    → 期望差 {diff:+.2f}pp: {verdict}")

    # 止损有效性
    print(f"\n## 卖点③ 止损5% 有效性验证")
    sl = [s for s in buyp if exit_full(s)[1] == "回落止损5%"]
    sl_rule = [exit_full(s)[0] for s in sl]
    sl_hold = [exit_full(s)[2] for s in sl]
    print(f"  止损样本 {len(sl)} 条:")
    print(f"    [止损离场]      " + stat_line("", sl_rule).strip())
    print(f"    [不止损持到收盘]" + stat_line("", sl_hold).strip())
    _, _, _, _, _, er = pl_stats(sl_rule)
    _, _, _, _, _, eh = pl_stats(sl_hold)
    if er is not None and eh is not None:
        diff = er - eh
        print(f"    → 期望差 {diff:+.2f}pp: "
              + ("止损有效(避免更大亏损)" if diff > 0
                 else "止损反而错失(持到收盘更好)"))


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        import re
        ds = [re.search(r"(\d{8})", f.stem).group(1)
              for f in sorted(LIVE.glob("presig_state_*.json"))]
        date = max(ds) if ds else None
    analyze(date)
