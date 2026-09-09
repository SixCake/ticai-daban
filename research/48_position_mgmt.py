# -*- coding: utf-8 -*-
"""研究48: 分时持仓管理规则 —— alpha在持仓管理的具体落地

基于研究47发现(触发后5min MA5偏离/RSI强区分度, 8日稳定)与研究46(定稿卖出
规则行情依赖), 设计**分时持仓管理规则**: 在定稿规则基础上, 增加"触发后
分时弱势提前卖", 用个股级分时状态替代P2 10:10的一刀切。

持仓管理决策树(触发买入后, 逐分钟评估):
  1. 封板(贴死涨停价)          → 续持(无上限, 定稿)
  2. 跌破止损线                → 止损卖(定稿兜底, 参考价=max(买价,动态最高))
  3. 触发后≥EVAL_MIN 且 未封板:
       价<5minMA5 且 RSI<RSI_WEAK (弱势) → **提前卖**(新增)
       价>5minMA5 且 RSI>RSI_STRONG(强势) → 持有(等封板/尾盘)
  4. P2 10:10 未封板           → 卖(定稿兜底)
  5. 14:57                     → 强平(定稿兜底)

对比定稿规则(止损5%/P2 10:10/14:57强平)的期望, 验证持仓管理改善。

用法: python research/48_position_mgmt.py [--multi]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from core.exit_rules import (FLAT_SEC, P2_SEC, SEAL_EPS,     # noqa: E402
                             STOP_LOSS, _sec, pl_stats, simulate_exit)

LIVE = DATA / "live"
EVAL_MIN = 10           # 触发后10min开始评估(研究48: T=10~15最稳健)
WIN = 15                # 5min窗口点数(每20秒一点)
RSI_WEAK = 45           # RSI<此=超卖弱势
RSI_STRONG = 55         # RSI>此=强势


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    g, l = [], []
    for i in range(1, len(prices)):
        ch = prices[i] - prices[i - 1]
        g.append(max(ch, 0))
        l.append(max(-ch, 0))
    ag = sum(g[-period:]) / period
    al = sum(l[-period:]) / period
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def simulate_exit_mgmt(pts, pb, limit_px, close_px, hi_pts=None):
    """分时持仓管理离场: 定稿规则 + 触发后分时弱势提前卖

    返回 (收益%, 离场原因)。止损参考价随分时动态上移(同定稿, 无前视)。
    """
    if not pb or pb <= 0:
        return None, "无买价"
    hi_map = {}
    if hi_pts:
        for e in hi_pts:
            if len(e) >= 2 and e[1] and e[1] > 0:
                hi_map[_sec(e[0])] = float(e[1])
    ref = pb
    stop = ref * STOP_LOSS
    sealed = False
    strong = False                    # 强势标记(触发后评估, 延长持有)
    prices = []
    t0 = None
    sorted_pts = sorted(pts or [], key=lambda x: str(x[0]))
    for e in sorted_pts:
        if len(e) < 2 or not e[1] or e[1] <= 0:
            continue
        px, t = float(e[1]), _sec(e[0])
        if t0 is None:
            t0 = t                        # 首点=触发时刻
        prices.append(px)
        h = hi_map.get(t)
        if h and h > ref:                 # 动态上移止损参考价(无前视)
            ref = h
            stop = ref * STOP_LOSS
        if px <= stop:
            return (px / pb - 1) * 100, "止损5%"
        if limit_px > 0 and px >= limit_px * SEAL_EPS:
            sealed = True
        # 触发后≥EVAL_MIN 评估强势(价>5minMA5且RSI>RSI_STRONG) → 延长持有
        # v1(弱势砍仓)实测无效(卖飞44-53%), 改为强势延长持有(跳过P2)
        if (not sealed and not strong and t0 is not None
                and t >= t0 + EVAL_MIN * 60 and len(prices) >= WIN):
            w = prices[-WIN:]
            ma5 = sum(w) / WIN
            rsi = calc_rsi(w)
            if px > ma5 and rsi is not None and rsi > RSI_STRONG:
                strong = True
        # P2 10:10: 强势票跳过(延长持有到尾盘/封板), 弱势票按定稿砍
        if t >= P2_SEC and not sealed and not strong:
            return (px / pb - 1) * 100, "P2 10:10未封板"
        if t >= FLAT_SEC:
            return ((px / pb - 1) * 100,
                    "14:57强平(封板续持)" if sealed else "14:57强平")
    if close_px and close_px > 0:
        return ((close_px / pb - 1) * 100,
                "封板续持·回退收盘" if sealed else "轨迹不足回退收盘")
    return None, "无数据"


def both_exits(s):
    """→ (定稿收益, 定稿原因, 分时管理收益, 分时管理原因)"""
    ph = s.get("px_hist") or []
    pts = [[p[0], p[1]] for p in ph if len(p) >= 2 and p[1]]
    if not pts or not s.get("pb"):
        return None, None, None, None
    pb = s["pb"]
    prices = [p[1] for p in pts]
    close_px = prices[-1]
    limit_px = s.get("limit_px") or 0
    hi_pts = [[p[0], p[1]] for p in pts]
    r1, w1 = simulate_exit(pts, pb, max(prices), limit_px, close_px,
                           hi_pts=hi_pts)
    r2, w2 = simulate_exit_mgmt(pts, pb, limit_px, close_px, hi_pts=hi_pts)
    return r1, w1, r2, w2


def analyze_date(date, verbose=False):
    d = json.load(open(LIVE / f"presig_state_{date}.json", encoding="utf-8"))
    buyp = [s for s in d["signals"]
            if s.get("stage") in ("S2", "S3") and s.get("pb")]
    rule, mgmt = [], []
    mgmt_why = {}
    for s in buyp:
        r1, w1, r2, w2 = both_exits(s)
        if r1 is not None:
            rule.append(r1)
        if r2 is not None:
            mgmt.append(r2)
            mgmt_why[w2] = mgmt_why.get(w2, 0) + 1
    n1, wr1, aw1, al1, ra1, ex1 = pl_stats(rule)
    n2, wr2, aw2, al2, ra2, ex2 = pl_stats(mgmt)
    return {"date": date, "n": len(buyp),
            "rule": (n1, wr1, ra1, ex1), "mgmt": (n2, wr2, ra2, ex2),
            "diff": (ex2 - ex1) if ex1 is not None and ex2 is not None else None,
            "mgmt_why": mgmt_why}


def main():
    multi = "--multi" in sys.argv
    dates = []
    for f in sorted(LIVE.glob("presig_state_*.json")):
        m = re.match(r"presig_state_(\d{8})\.json$", f.name)
        if m:
            dates.append(m.group(1))
    if not multi:
        dates = dates[-1:]

    print(f"# 分时持仓管理规则 回测({'多日' if multi else '单日'})")
    print(f"\n规则: 定稿(止损5%/P2 10:10/14:57强平/封板续持)")
    print(f"      + 触发后{EVAL_MIN}min 价>5minMA5且RSI>{RSI_STRONG}(强势) → 跳过P2延长持有")
    print(f"\n{'日期':<10}{'买点':>5} | {'定稿期望':>8}{'管理期望':>8}{'改善':>7}"
          f" | 管理离场分布")
    tot_rule, tot_mgmt = [], []
    for date in dates:
        r = analyze_date(date)
        ex1 = r["rule"][3]
        ex2 = r["mgmt"][3]
        f2 = lambda x: f"{x:.2f}" if x is not None else "-"
        diff = f"{r['diff']:+.2f}" if r["diff"] is not None else "-"
        why = r["mgmt_why"]
        why_s = " ".join(f"{k}{v}" for k, v in sorted(
            why.items(), key=lambda x: -x[1])[:3])
        print(f"{date:<10}{r['n']:>5} | {f2(ex1):>8}{f2(ex2):>8}{diff:>7}"
              f" | {why_s}")
    # 汇总(合并所有日的收益)
    if multi:
        all_rule, all_mgmt = [], []
        for date in dates:
            d = json.load(open(LIVE / f"presig_state_{date}.json",
                               encoding="utf-8"))
            buyp = [s for s in d["signals"]
                    if s.get("stage") in ("S2", "S3") and s.get("pb")]
            for s in buyp:
                r1, _, r2, _ = both_exits(s)
                if r1 is not None:
                    all_rule.append(r1)
                if r2 is not None:
                    all_mgmt.append(r2)
        _, wr1, _, _, ra1, ex1 = pl_stats(all_rule)
        _, wr2, _, _, ra2, ex2 = pl_stats(all_mgmt)
        print(f"\n## 汇总({len(dates)}日合并)")
        print(f"  定稿规则:   样本{len(all_rule)} 胜率{wr1*100:.1f}% "
              f"盈亏比{ra1:.2f} 期望{ex1:.2f}%")
        print(f"  分时管理:   样本{len(all_mgmt)} 胜率{wr2*100:.1f}% "
              f"盈亏比{ra2:.2f} 期望{ex2:.2f}%")
        print(f"  → 期望改善 {ex2-ex1:+.2f}pp")


if __name__ == "__main__":
    main()
