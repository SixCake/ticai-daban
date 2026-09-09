# -*- coding: utf-8 -*-
"""K线形态/阻力点/突破点因子 —— 唯一出处

研究37 定稿的 18 个白盒几何因子。现有生产因子(core/structure 的
g_chip/g_eco/g_intra + 研究30 增量)全是**统计聚合量**(触板未封天数/量比/
连跌天数/涨速), 没有几何/形态维度 —— 本模块补这个缺口。

调用方:
  research/37_shape_factors.py   单因子筛选
  research/38_shape_scheme_matrix.py  方案矩阵(M0~M6)
两处共用本模块避免口径分叉(同 core/exit_rules.py 的做法)。

前视防护: compute_factors 只用 i-1 及更早的 bar(i = 信号日索引)。
调用方必须保证这一点, 不得传入含当日数据的窗口。

缺失不补缺: 一字板(high==low)时 body_ratio/upper_shadow 记 None 而非填 0;
bar 不足时 trapped_ratio/dist_high_120 记 None。统计时按可用样本算。

阈值(1.2/0.8倍量, 3%/2%涨幅, 20/60/120窗口)取业内常用值作初筛,
未做二次拟合 —— 研究37 目的是看有无区分度, 不是定阈值。
"""
import numpy as np

# ---- 因子清单 ----
CONT = ["body_ratio", "upper_shadow", "dist_high_20", "dist_high_60",
        "dist_high_120", "trapped_ratio", "int_prox", "box_width_20",
        "flat_days", "ma_align", "vol_price_sync", "rise_20d",
        # F 量价组(日线级) —— 只看形态不看量价是不完整的,
        # A股打板的常识是「量在价先」, 缩量洗盘/放量突破/量价背离
        # 都必须用成交量判定
        "volr5", "vol_trend5", "vol_conc", "shrink_wash", "vol_break",
        "obv_slope20", "pv_diverge", "amt_ratio"]
BOOL = ["fake_yin", "n_shape", "doji_shrink", "new_high_20", "new_high_60",
        "new_high_120", "gap_up"]
ALL_FACTORS = CONT + BOOL

GROUP = {
    "body_ratio": "A 形态", "upper_shadow": "A 形态", "fake_yin": "A 形态",
    "n_shape": "A 形态", "doji_shrink": "A 形态",
    "dist_high_20": "B 阻力", "dist_high_60": "B 阻力",
    "dist_high_120": "B 阻力", "trapped_ratio": "B 阻力",
    "int_prox": "B 阻力",
    "box_width_20": "C 突破", "flat_days": "C 突破",
    "new_high_20": "C 突破", "new_high_60": "C 突破",
    "new_high_120": "C 突破", "gap_up": "C 突破", "ma_align": "C 突破",
    "vol_price_sync": "D 量能", "rise_20d": "E 价位",
    "volr5": "F 量价", "vol_trend5": "F 量价", "vol_conc": "F 量价",
    "shrink_wash": "F 量价", "vol_break": "F 量价",
    "obv_slope20": "F 量价", "pv_diverge": "F 量价", "amt_ratio": "F 量价",
}

# 天然推高入场价位的因子: 研究08 已证 6%以上入场数学上不可行
# (需81%+封板率才打平), 这类因子必须与入场价位闸门联合评估才有意义
PRICE_RISK = {"new_high_20", "new_high_60", "new_high_120", "gap_up",
              "dist_high_20", "dist_high_60", "dist_high_120"}

LABELS = ["r1_open", "r1_close", "r2_close", "r3_close"]
NLABEL = {"r1_open": "T+1开盘", "r1_close": "T+1收盘",
          "r2_close": "T+2收盘", "r3_close": "T+3收盘"}

# 可执行口径标签(基准 = open[T+1] 次日开盘买入)
XLABELS = ["x1_close", "x2_close", "x3_close"]
XNLABEL = {"x1_close": "T+1收盘", "x2_close": "T+2收盘",
           "x3_close": "T+3收盘"}
SEAL_EPS = 0.9995          # 封死判据, 同 core/theme_signal / core/exit_rules


def limit_rate(code: str) -> float:
    """涨停幅近似(未做ST修正, 同 core/structure 口径)"""
    c = str(code)
    if c.endswith(".BJ"):
        return 0.30
    if c[:3] in ("300", "301", "302", "688", "689"):
        return 0.20
    return 0.10


def compute_factors(b: dict, i: int) -> dict:
    """信号日 T = b['date'][i]。**只用 i-1 及更早的 bar**(前视防护)。

    b: {date/open/high/low/close/vol/pct 的 numpy 数组}, 按 date 升序
    返回 {} 表示数据不足(i<1)。缺失项记 None, 不填 0 伪装。
    """
    o, h, l, c, v, p = (b["open"], b["high"], b["low"], b["close"],
                        b["vol"], b["pct"])
    if i < 1:
        return {}
    j = i - 1                      # T-1
    out = {}
    # ---- A 组 K线形态 (T-1 bar) ----
    rng = h[j] - l[j]
    if rng and rng > 0:
        out["body_ratio"] = (c[j] - o[j]) / rng
        out["upper_shadow"] = (h[j] - max(o[j], c[j])) / rng
    else:
        out["body_ratio"] = None           # 一字板: 无实体可言, 不填0
        out["upper_shadow"] = None
    out["doji_shrink"] = None              # 需 vma5, 下面算
    if j >= 1 and c[j - 1] > 0:
        out["fake_yin"] = int(c[j] < o[j] and c[j] > c[j - 1])
    else:
        out["fake_yin"] = None
    # vma5 = T-1 之前5日均量(不含 T-1)
    vs = v[max(0, j - 5):j]
    vs = vs[~np.isnan(vs)]
    vma5 = vs.mean() if len(vs) >= 3 else None
    if vma5 and vma5 > 0 and v[j] and not np.isnan(v[j]) and rng and rng > 0:
        shrink = v[j] < 0.7 * vma5
        out["doji_shrink"] = int(abs(c[j] - o[j]) / rng < 0.2 and shrink)
    # N字板: 近5日内 ∃ 放量阳→缩量阴→放量阳
    ns = None
    if j >= 4 and vma5:
        ns = 0
        for k in range(max(1, j - 4), j - 1):
            if (v[k] > 1.2 * vma5 and p[k] > 3.0
                    and v[k + 1] < 0.8 * v[k] and p[k + 1] < 0
                    and v[k + 2] > v[k + 1] and p[k + 2] > 0):
                ns = 1
                break
    out["n_shape"] = ns
    # ---- B 组 阻力点 ----
    for n, key in ((20, "dist_high_20"), (60, "dist_high_60"),
                   (120, "dist_high_120")):
        seg = h[max(0, j - n + 1):j + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= min(n, 20) and c[j] > 0:
            mx = seg.max()
            out[key] = (c[j] - mx) / mx * 100
        else:
            out[key] = None
    seg = c[max(0, j - 119):j + 1]
    seg = seg[~np.isnan(seg)]
    out["trapped_ratio"] = ((seg > c[j]).mean() * 100
                            if len(seg) >= 60 and c[j] > 0 else None)
    out["int_prox"] = (abs(c[j] - round(c[j] / 5) * 5) / c[j] * 100
                       if c[j] > 0 else None)
    # ---- C 组 突破点 ----
    hs = h[max(0, j - 19):j + 1]
    ls = l[max(0, j - 19):j + 1]
    hs, ls = hs[~np.isnan(hs)], ls[~np.isnan(ls)]
    out["box_width_20"] = ((hs.max() - ls.min()) / ls.min() * 100
                           if len(hs) >= 15 and ls.min() > 0 else None)
    fd = 0
    for k in range(j, 0, -1):
        if abs(p[k]) < 2.0:
            fd += 1
        else:
            break
    out["flat_days"] = fd
    for n, key in ((20, "new_high_20"), (60, "new_high_60"),
                   (120, "new_high_120")):
        seg = h[max(0, j - n + 1):j]        # 不含 T-1 自身
        seg = seg[~np.isnan(seg)]
        out[key] = (int(h[j] >= seg.max())
                    if len(seg) >= min(n, 20) and not np.isnan(h[j]) else None)
    out["gap_up"] = (int(o[j] > h[j - 1])
                     if j >= 1 and not np.isnan(o[j])
                     and not np.isnan(h[j - 1]) else None)
    mas = {}
    for n in (5, 10, 20, 60):
        seg = c[max(0, j - n + 1):j + 1]
        seg = seg[~np.isnan(seg)]
        mas[n] = seg.mean() if len(seg) >= n else None
    if all(mas[n] is not None for n in (5, 10, 20, 60)):
        out["ma_align"] = (int(mas[5] > mas[10]) + int(mas[10] > mas[20])
                           + int(mas[20] > mas[60]))
    else:
        out["ma_align"] = None
    # ---- D 组 量能 ----
    ps = p[max(0, j - 19):j + 1]
    vs2 = v[max(0, j - 19):j + 1]
    ok = ~(np.isnan(ps) | np.isnan(vs2))
    if ok.sum() >= 15 and ps[ok].std() > 0 and vs2[ok].std() > 0:
        out["vol_price_sync"] = float(np.corrcoef(ps[ok], vs2[ok])[0, 1])
    else:
        out["vol_price_sync"] = None
    # ---- E 组 价位(防接飞刀) ----
    # 近20日累计涨幅。用户定稿买入模型 MAX_RISE_20D=20%: 首板超此剔除。
    if j >= 20 and c[j - 20] > 0 and c[j] > 0:
        out["rise_20d"] = (c[j] / c[j - 20] - 1) * 100
    else:
        out["rise_20d"] = None
    # ---- F 组 量价(日线级) ----
    # 只看形态不看量价是不完整的: A股打板常识是「量在价先」,
    # 缩量洗盘/放量突破/量价背离都必须用成交量判定。
    v5 = v[max(0, j - 4):j + 1]
    v5 = v5[~np.isnan(v5)]
    v20 = v[max(0, j - 19):j + 1]
    v20 = v20[~np.isnan(v20)]
    # volr5: 昨日量 / 前5日均量(同 core/structure 的 y_volr5 口径)
    out["volr5"] = (v[j] / vma5 if vma5 and vma5 > 0
                    and not np.isnan(v[j]) else None)
    # vol_trend5: 近5日量的线性斜率, 用均量归一(可跳股票比较)
    if len(v5) >= 4 and v5.mean() > 0:
        x = np.arange(len(v5), dtype=float)
        out["vol_trend5"] = float(np.polyfit(x, v5, 1)[0] / v5.mean())
    else:
        out["vol_trend5"] = None
    # vol_conc: 近5日量和 / 近20日量和 —— 量能集中度(近期是否在放量)
    out["vol_conc"] = (v5.sum() / v20.sum() * 100
                       if len(v20) >= 15 and v20.sum() > 0 else None)
    # shrink_wash: 近5日内「价涨量缩」天数占比 —— 缩量洗盘=主力控盘
    if j >= 5:
        cnt = tot = 0
        for k in range(j - 4, j + 1):
            if k < 1 or np.isnan(p[k]) or np.isnan(v[k]) or np.isnan(v[k - 1]):
                continue
            tot += 1
            if p[k] > 0 and v[k] < v[k - 1]:
                cnt += 1
        out["shrink_wash"] = cnt / tot * 100 if tot >= 3 else None
    else:
        out["shrink_wash"] = None
    # vol_break: 放量突破 = 昨日量>1.5×前5日均量 且 收盘创近20日新高
    h20 = h[max(0, j - 19):j]          # 不含 T-1 自身
    h20 = h20[~np.isnan(h20)]
    out["vol_break"] = (int(vma5 and vma5 > 0 and not np.isnan(v[j])
                            and v[j] > 1.5 * vma5
                            and len(h20) >= 15 and c[j] >= h20.max())
                        if vma5 and len(h20) >= 15 else None)
    # obv_slope20: OBV(能量潮)近20日斜率, 用量和归一
    if j >= 20:
        obv = [0.0]
        for k in range(j - 19, j + 1):
            if np.isnan(p[k]) or np.isnan(v[k]):
                obv.append(obv[-1])
            else:
                obv.append(obv[-1] + (v[k] if p[k] > 0
                                      else -v[k] if p[k] < 0 else 0.0))
        y = np.array(obv[1:], dtype=float)
        base = v20.sum() if len(v20) and v20.sum() > 0 else None
        if base:
            x = np.arange(len(y), dtype=float)
            out["obv_slope20"] = float(np.polyfit(x, y, 1)[0] / base)
        else:
            out["obv_slope20"] = None
    else:
        out["obv_slope20"] = None
    # pv_diverge: 量价背离 = 收盘创20日新高 但 量未创20日新高(危险)
    vmax20 = v20[:-1].max() if len(v20) >= 15 else None
    out["pv_diverge"] = (int(len(h20) >= 15 and c[j] >= h20.max()
                             and vmax20 is not None and not np.isnan(v[j])
                             and v[j] < vmax20)
                         if len(h20) >= 15 and vmax20 is not None else None)
    # amt_ratio: 近5日均量 / 近**60日**均量 —— 中期资金关注度变化。
    # 窗口必须与 vol_conc(近5/近20) 错开: 两者同窗口时数学上等价
    # (v5.sum()/v20.sum() ≡ 0.25×v5.mean()/v20.mean()), 实测秩相关=1.00。
    v60 = v[max(0, j - 59):j + 1]
    v60 = v60[~np.isnan(v60)]
    out["amt_ratio"] = (v5.mean() / v60.mean()
                        if len(v5) >= 4 and len(v60) >= 40
                        and v60.mean() > 0 else None)
    return out


def compute_labels(b: dict, i: int) -> dict:
    """标签只用 T+1 及之后(前视防护的另一侧)。基准 = close[T]

    ⚠ **这个口径不可执行**: close[T] 是涨停价, 封死的票尾盘买不进去
    (买单排队也不一定成交)。保留仅供与研究37 已发布报告对照,
    **不得当作可实现收益**。可执行口径用 compute_labels_exec。
    """
    c, o = b["close"], b["open"]
    base = c[i]
    if not base or base <= 0 or np.isnan(base):
        return {}
    out = {}
    out["r1_open"] = ((o[i + 1] / base - 1) * 100
                      if i + 1 < len(c) and not np.isnan(o[i + 1]) else None)
    for k, key in ((1, "r1_close"), (2, "r2_close"), (3, "r3_close")):
        out[key] = ((c[i + k] / base - 1) * 100
                    if i + k < len(c) and not np.isnan(c[i + k]) else None)
    return out


def compute_labels_exec(b: dict, i: int, code: str) -> dict:
    """**可执行口径**标签: 基准 = open[T+1](次日开盘买入)

    为何不用 close[T]: T 日已涨停封死, 尾盘买单排队也不一定能成交 ——
    以涨停价为买入基准算出的收益是**纸面数字, 实现不了**。

    buyable 标记: T+1 开盘价贴死涨停价(一字板)时仍然买不进, 记 False。
    调用方必须用 buyable 过滤, 否则会把不可成交的票算进收益。
    """
    c, o = b["close"], b["open"]
    if i + 1 >= len(c):
        return {}
    buy = o[i + 1]
    if not buy or buy <= 0 or np.isnan(buy):
        return {}
    # T+1 涨停价 = T日收盘 × (1+幅)
    lp = round(c[i] * (1 + limit_rate(code)), 2) if c[i] > 0 else 0.0
    out = {"buy_px": float(buy),
           "buyable": bool(lp <= 0 or buy < lp * SEAL_EPS),
           "gap_open": (buy / c[i] - 1) * 100 if c[i] > 0 else None}
    for k, key in ((1, "x1_close"), (2, "x2_close"), (3, "x3_close")):
        out[key] = ((c[i + k] / buy - 1) * 100
                    if i + k < len(c) and not np.isnan(c[i + k]) else None)
    return out


def build_bars(panel) -> dict:
    """{code: {date/open/high/low/close/vol/pct 的 numpy 数组}} 升序"""
    out = {}
    for code, g in panel.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date")
        out[code] = {
            "date": g["trade_date"].values,
            "open": g["open"].to_numpy(dtype=float),
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
            "vol": g["vol"].to_numpy(dtype=float),
            "pct": g["pct_chg"].to_numpy(dtype=float),
        }
    return out


def build_regimes(panel) -> dict:
    """按月均"全A中位涨幅"三分位切牛/熊/震荡(同 research/31, 数据驱动)"""
    mr = panel.groupby("trade_date")["pct_chg"].median().rename("mret")
    mon = mr.groupby(mr.index.str[:6]).mean().sort_values()
    n = len(mon)
    return {"bull": set(mon.index[-n // 3:]),
            "bear": set(mon.index[:n // 3])}


def reg_of(date: str, rg: dict) -> str:
    m = str(date)[:6]
    return "熊" if m in rg["bear"] else "牛" if m in rg["bull"] else "震荡"
