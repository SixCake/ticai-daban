# -*- coding: utf-8 -*-
"""研究47: 盘中分时因子(5min MA5/波动率/RSI) 叠加买卖点

⚠ 数据硬约束(已验证): px_hist 从**信号触发时刻**才开始记录(首点==触发时刻
  100%), 触发前无分时点 → **分时5min因子在买点决策时刻算不出来**, 不能作
  买点判据(否则前视)。

因此本研究口径:
  · 买点: 触发时刻无分时数据 → 分时因子**不可用于买点**(改用日线/竞价因子)
  · 卖点/持仓: 触发后 px_hist 有 466+点(每20秒一点, 5min≈15点) →
    用**触发后5min窗口**的 MA5偏离/波动率/RSI 分析持仓管理(触发后可知)

分时因子(触发后5min窗口, 白盒公式):
  ma5_dev = 5min末价 / 5min均价 - 1   (价格相对5分钟均线, 强势/弱势)
  vol5    = 5min收益率标准差          (分时波动率)
  rsi     = 5min窗口RSI(14)           (超买/超卖)

回答: 触发后5min的分时状态能否预测离场收益 → 可否作持仓管理/卖点依据。

用法: python research/47_intraday_factors.py [date YYYYMMDD]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from core.exit_rules import pl_stats, simulate_exit          # noqa: E402

LIVE = DATA / "live"
WIN = 15                    # 触发后5min窗口点数(每20秒一点, 5min≈15点)


def calc_rsi(prices, period=14):
    """RSI(白盒): 100 - 100/(1+RS), RS=均涨/均跌"""
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        ch = prices[i] - prices[i - 1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def intraday_factors(s):
    """触发后5min窗口的分时因子(持仓期间可知, 非买点判据)"""
    ph = s.get("px_hist") or []
    pts = [float(p[1]) for p in ph if len(p) >= 2 and p[1]]
    if len(pts) < WIN:
        return None
    win = pts[:WIN]                       # 触发后5min
    ma5 = sum(win) / len(win)
    ma5_dev = (win[-1] / ma5 - 1) * 100   # 5min末价相对均价
    rets = [(win[i] / win[i - 1] - 1) for i in range(1, len(win))]
    mr = sum(rets) / len(rets)
    vol5 = (sum((r - mr) ** 2 for r in rets) / len(rets)) ** 0.5 * 100
    return {"ma5_dev": ma5_dev, "vol5": vol5, "rsi": calc_rsi(win)}


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

    print(f"# 今日({date})盘中分时因子(触发后5min) 叠加卖点")
    print(f"\n⚠ 买点触发时刻无分时数据(px_hist从触发才记录) → 分时因子")
    print(f"  **不可作买点判据**, 仅用于触发后持仓管理/卖点分析")
    print(f"\n买点 {len(buyp)} 条")

    # MA5偏离(价格相对5分钟均线)
    print(f"\n## 分时因子① MA5偏离(5min末价相对5min均价)")
    print(HDR)
    mg = {}
    for s in buyp:
        f = intraday_factors(s)
        r = exit_ret(s)
        if not f or r is None:
            continue
        v = f["ma5_dev"]
        lab = ("1_价<MA5(弱势)" if v < -0.3 else
               "2_价≈MA5(±0.3)" if v <= 0.3 else "3_价>MA5(强势)")
        mg.setdefault(lab, []).append(r)
    for k in sorted(mg):
        print(stat_line(k, mg[k]))

    # 波动率
    print(f"\n## 分时因子② 5min波动率(分时收益率标准差)")
    print(HDR)
    vg = {}
    for s in buyp:
        f = intraday_factors(s)
        r = exit_ret(s)
        if not f or r is None:
            continue
        v = f["vol5"]
        lab = ("1_低波动<0.3" if v < 0.3 else
               "2_中0.3-0.6" if v < 0.6 else "3_高波动≥0.6")
        vg.setdefault(lab, []).append(r)
    for k in sorted(vg):
        print(stat_line(k, vg[k]))

    # RSI
    print(f"\n## 分时因子③ 5min RSI(14)")
    print(HDR)
    rg = {}
    for s in buyp:
        f = intraday_factors(s)
        r = exit_ret(s)
        if not f or f["rsi"] is None or r is None:
            continue
        v = f["rsi"]
        lab = ("1_超卖<40" if v < 40 else
               "2_中性40-60" if v <= 60 else
               "3_偏强60-75" if v <= 75 else "4_超买>75")
        rg.setdefault(lab, []).append(r)
    for k in sorted(rg):
        print(stat_line(k, rg[k]))


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        import re
        ds = [re.search(r"(\d{8})", f.stem).group(1)
              for f in sorted(LIVE.glob("presig_state_*.json"))]
        date = max(ds) if ds else None
    analyze(date)
