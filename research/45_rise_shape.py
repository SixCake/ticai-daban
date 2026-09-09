# -*- coding: utf-8 -*-
"""研究45: 今日所有信号的上涨形态与模式 + 买入胜率

为何做: zt_shape_of 只对**已封板**票分类(判据要求 price≥涨停价), 但
px_hist 分时轨迹**所有信号都有**(实测轨迹完整100%)。本研究基于 px_hist
研究全部信号的上涨形态, 不受"封板"限制, 回答"股票是怎么涨的、哪种涨法
的买点胜率高"。

上涨形态特征(从 px_hist 提取, 涨幅以 limit_px 反推昨收归一):
  first 开盘涨幅 / peak 峰值涨幅 / peak_t 达峰时刻 / end 收盘涨幅
  pull 回落(peak-end) / pathvol 路径波动 / r3 涨速 / t 触发时刻

两部分:
  ① 上涨模式全景(互斥分类): 秒板/早盘封板/盘中封板/尾盘封板/冲高回落/
     触板未封/弱势未触板 —— 描述"怎么涨的"+各模式结局
  ② 决策可知形态 vs 买入胜率: 开盘涨幅/触发时段/pathvol 分档(触发时可知,
     无前视) —— 回答"哪种涨法值得买"

用法: python research/45_rise_shape.py [date YYYYMMDD]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from core.exit_rules import pl_stats, simulate_exit          # noqa: E402

LIVE = DATA / "live"


def cm20(code):
    return str(code)[:2] in ("30", "68")


def rise_feat(s):
    """从 px_hist 提取上涨形态特征(涨幅归一到昨收)"""
    ph = s.get("px_hist") or []
    pts = [(str(p[0]), float(p[1])) for p in ph if len(p) >= 2 and p[1]]
    lp = s.get("limit_px") or 0
    if len(pts) < 3 or lp <= 0:
        return None
    ratio = 0.20 if cm20(s["ts_code"]) else 0.10
    pc = lp / (1 + ratio)                       # 昨收
    pcts = [(t, (px / pc - 1) * 100) for t, px in pts]
    first = pcts[0][1]
    pi = max(range(len(pcts)), key=lambda i: pcts[i][1])
    peak, peak_t = pcts[pi][1], pcts[pi][0]
    end = pcts[-1][1]
    return {"first": first, "peak": peak, "peak_t": peak_t.replace(":", ""),
            "end": end, "pull": peak - end, "limit_pct": ratio * 100,
            "pv": s.get("pathvol"), "r3": s.get("r3"),
            "t": str(s.get("t", "")).replace(":", ""),
            "sealed": bool(s.get("zt_shape"))}


def rise_pattern(f):
    """上涨模式互斥分类(达峰时段 × 结局)"""
    if f is None:
        return None
    near_limit = f["peak"] >= f["limit_pct"] * 0.97
    if f["sealed"]:
        if f["peak_t"] <= "093500":
            return "1_秒板/一字"
        if f["peak_t"] <= "100000":
            return "2_早盘封板"
        if f["peak_t"] < "140000":
            return "3_盘中封板"
        return "4_尾盘封板"
    # 未封板
    if f["pull"] >= 3:
        return "5_冲高回落"
    if near_limit:
        return "6_触板未封"
    return "7_弱势未触板"


def exit_ret(s):
    ph = s.get("px_hist") or []
    pts = [[p[0], p[1]] for p in ph if len(p) >= 2 and p[1]]
    if not pts or not s.get("pb"):
        return None
    pb = s["pb"]
    prices = [p[1] for p in pts]
    ret, _ = simulate_exit(pts, pb, max(prices), s.get("limit_px") or 0,
                           prices[-1], hi_pts=[[p[0], p[1]] for p in pts])
    return ret


def stat_line(name, g):
    n = len(g)
    sealed = sum(1 for s in g if s.get("zt_shape"))
    rets = [r for r in (exit_ret(s) for s in g) if r is not None]
    nn, wr, aw, al, ratio, exp = pl_stats(rets)
    return (f"{name:<14}{n:>5}{sealed:>5}{sealed/n*100:>7.1f}"
            f"{nn:>6}{(wr*100 if wr else 0):>7.1f}"
            f"{(ratio if ratio else 0):>7.2f}{(exp if exp else 0):>7.2f}")


HDR = (f"{'组':<14}{'信号':>5}{'封板':>5}{'封板率%':>7}{'买点':>6}"
       f"{'胜率%':>7}{'盈亏比':>7}{'期望%':>7}")


def analyze(date):
    d = json.load(open(LIVE / f"presig_state_{date}.json", encoding="utf-8"))
    sig = d["signals"]
    feats = {id(s): rise_feat(s) for s in sig}

    print(f"# 今日({date})所有信号上涨形态与模式")
    print(f"\n信号 {len(sig)} 条 | 形态可提取 "
          f"{sum(1 for f in feats.values() if f)} 条")

    # ① 上涨模式全景(所有信号)
    print(f"\n## ① 上涨模式全景(全部信号, 互斥分类)")
    print(HDR)
    groups = {}
    for s in sig:
        p = rise_pattern(feats[id(s)])
        if p:
            groups.setdefault(p, []).append(s)
    for p in sorted(groups):
        print(stat_line(p, groups[p]))

    # ② 决策可知形态 vs 买入胜率(仅买点 S2/S3, 触发时可知, 无前视)
    buyp = [s for s in sig if s.get("stage") in ("S2", "S3") and s.get("pb")]
    print(f"\n## ② 决策可知形态 vs 买入胜率(买点{len(buyp)}条, 无前视)")

    def band(val, cuts, labels):
        for c, lab in zip(cuts, labels):
            if val < c:
                return lab
        return labels[-1]

    print("\n-- 按触发时段(信号触发时刻, 盘中可知) --")
    print(HDR)
    tg = {}
    for s in buyp:
        f = feats[id(s)]
        if not f:
            continue
        t = f["t"]
        lab = ("1_早盘≤10:00" if t <= "100000" else
               "2_盘中10-14" if t < "140000" else "3_尾盘≥14:00")
        tg.setdefault(lab, []).append(s)
    for k in sorted(tg):
        print(stat_line(k, tg[k]))

    print("\n-- 按路径波动 pathvol(平稳/中/颠簸, 触发时可知) --")
    print(HDR)
    pg = {}
    for s in buyp:
        pv = s.get("pathvol")
        if pv is None:
            continue
        lab = ("1_平稳<0.4" if pv < 0.4 else
               "2_中0.4-0.8" if pv < 0.8 else "3_颠簸≥0.8")
        pg.setdefault(lab, []).append(s)
    for k in sorted(pg):
        print(stat_line(k, pg[k]))

    print("\n-- 按开盘涨幅(竞价/开盘, 决策可知) --")
    print(HDR)
    fg = {}
    for s in buyp:
        f = feats[id(s)]
        if not f:
            continue
        lab = band(f["first"], [0, 1, 3, 6],
                   ["1_低开<0", "2_平开0-1", "3_小高开1-3",
                    "4_中高开3-6", "5_大高开≥6"])
        fg.setdefault(lab, []).append(s)
    for k in sorted(fg):
        print(stat_line(k, fg[k]))


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        import re
        ds = [re.search(r"(\d{8})", f.stem).group(1)
              for f in sorted(LIVE.glob("presig_state_*.json"))]
        date = max(ds) if ds else None
    analyze(date)
