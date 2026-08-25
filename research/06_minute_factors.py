# -*- coding: utf-8 -*-
"""研究06: 涨停/触板分钟路径因子 —— "半路上车"能否提前判封板

问题: 股票大涨触及涨停价但未封(或尚未封)时, 哪些分钟级特征能预测
  A) 今日最终能否封死 (封板组 vs 炸板组) —— 该不该上车
  B) 封板组内: 首封后是否炸板 (open_times==0 硬封 vs ≥1) —— 上车后风险

数据: data/minutes/zt_minute_*.parquet (collect/fetch_zt_minute.py)
样本: 触板分钟 = 首根 high>=limit_px*0.998 的K线; 特征用截至该分钟(含)信息
     排除: 一字板(开盘即封, 半路无机会)、ST、首触在14:45后(观察窗过短)

输出: 控制台分层统计 + research/out/06_minute_factors.csv
用法: python research/06_minute_factors.py [--days 9]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

MIN_DIR = DATA / "minutes"
OUT = ROOT / "research" / "out"
TOUCH_TOL = 0.998          # high >= limit_px*0.998 记为触板
LATE_CUTOFF = "1445"       # 首触晚于此点不进主样本


def limit_rate(ts_code: str) -> float:
    c = ts_code[:3]
    return 0.2 if c in ("300", "301", "688", "689") else 0.1


def feats_one(g: pd.DataFrame) -> tuple[dict | None, str]:
    """单只stock-day: 定位首触分钟并计算截至该分钟的特征"""
    g = g.sort_values("t").reset_index(drop=True)
    lp = float(g["limit_px"].iloc[0])
    if not np.isfinite(lp) or lp <= 0:
        return None, "no_lp"
    rate = limit_rate(g["ts_code"].iloc[0])
    pre = lp / (1 + rate)
    hi = g["high"].values
    touch = np.where(hi >= lp * TOUCH_TOL)[0]
    if not len(touch):
        return None, "no_touch"
    i = int(touch[0])
    t_touch = g["t"].iloc[i]
    if t_touch > LATE_CUTOFF:
        return None, "late"
    if g["open"].iloc[0] >= lp * TOUCH_TOL:     # 一字开盘(含一字后炸)
        return None, "yizi"
    c = g["close"].values
    o = g["open"].values
    v = g["vol"].values
    amt = g["amount"].values
    lo = g["low"].values
    n_prev = max(i, 1)
    avg_v_prev = v[:i].sum() / n_prev if i > 0 else max(v[i], 1.0)
    s5 = c[i] / c[max(i - 5, 0)] - 1 if i >= 1 else np.nan
    s15 = c[i] / c[max(i - 15, 0)] - 1 if i >= 1 else np.nan
    vwap = amt[:i + 1].sum() / max(v[:i + 1].sum() * 100, 1e-9)
    float_mv_yuan = g["float_mv"].iloc[0] if np.isfinite(
        g["float_mv"].iloc[0]) else np.nan  # parquet中已是元单位
    return {
        "date": g["date"].iloc[0], "ts_code": g["ts_code"].iloc[0],
        "name": g["name"].iloc[0], "grp": g["grp"].iloc[0],
        "height": g["height"].iloc[0], "open_times": g["open_times"].iloc[0],
        "zb_times": g["zb_times"].iloc[0],
        "t_touch": t_touch, "mins_from_open": i,
        "open_ret": o[0] / pre - 1,
        "pct_touch": c[i] / pre - 1,
        "close_ret": c[-1] / pre - 1,
        "s5": s5, "s15": s15,
        "accel": (s5 - s15) if np.isfinite(s5) and np.isfinite(s15) else np.nan,
        "vol_burst5": v[max(i - 4, 0):i + 1].sum() / max(avg_v_prev, 1.0),
        "vol_touch": v[i] / max(avg_v_prev, 1.0),
        "above7_min": int((c[:i] >= pre * 1.07).sum()),
        "pullback_pre": 1 - lo[:i].min() / hi[:i].max() if i >= 1 else 0.0,
        "vwap_dev": c[i] / vwap - 1,
        "turnover_pre": amt[:i].sum() / float_mv_yuan
        if np.isfinite(float_mv_yuan) and float_mv_yuan > 0 else np.nan,
        "log_float_mv": np.log(g["float_mv"].iloc[0])
        if np.isfinite(g["float_mv"].iloc[0]) and g["float_mv"].iloc[0] > 0
        else np.nan,
    }, "ok"


def auc(y: pd.Series, x: pd.Series) -> float:
    """单特征rank AUC (Mann-Whitney), 取与|0.5|偏离大的方向"""
    m = x.notna() & y.notna()
    if m.sum() < 30:
        return np.nan
    r = x[m].rank()
    a = (r[y[m] == 1].sum() - (y[m] == 1).sum() * ((y[m] == 1).sum() + 1) / 2) \
        / ((y[m] == 1).sum() * (y[m] == 0).sum())
    return max(a, 1 - a)


def qlift(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """上下四分位条件封板率: (P(y|Q4), P(y|Q1))"""
    m = x.notna() & y.notna()
    if m.sum() < 40:
        return np.nan, np.nan
    q1, q4 = x[m].quantile(0.25), x[m].quantile(0.75)
    lo, hi_ = y[m][x[m] <= q1], y[m][x[m] >= q4]
    return (hi_.mean() if len(hi_) else np.nan,
            lo.mean() if len(lo) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=99)
    args = ap.parse_args()

    files = sorted(MIN_DIR.glob("zt_minute_*.parquet"))[-args.days:]
    if not files:
        print("无数据, 先跑 collect/fetch_zt_minute.py")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"载入 {len(files)} 日 {df.groupby('date')['ts_code'].nunique().sum()} "
          f"stock-day bars={len(df)}")

    rows = []
    skipped = {"no_touch": 0, "yizi": 0, "late": 0, "no_lp": 0}
    for (d, c), g in df.groupby(["date", "ts_code"]):
        f, reason = feats_one(g)
        if f is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        rows.append(f)
    panel = pd.DataFrame(rows)
    panel["y_seal"] = (panel["grp"] == "sealed").astype(int)
    print(f"\n[样本] 有效 {len(panel)} (排除: 一字{skipped['yizi']} "
          f"晚触{skipped['late']} 未触板{skipped['no_touch']} "
          f"缺涨停价{skipped['no_lp']})")
    print("分组:", panel["grp"].value_counts().to_dict())
    print("按日:", panel.groupby(["date", "grp"]).size().unstack(
        fill_value=0).to_string())

    y = panel["y_seal"]
    FEATS = ["mins_from_open", "open_ret", "s5", "s15", "accel",
             "vol_burst5", "vol_touch", "above7_min", "pullback_pre",
             "vwap_dev", "turnover_pre", "log_float_mv", "height"]
    print(f"\n=== 任务A: 触板时刻预测最终封板 (n={len(panel)}, "
          f"封板率{y.mean():.1%}) ===")
    tab = []
    for f in FEATS:
        a = auc(y, panel[f])
        hi_, lo_ = qlift(y, panel[f])
        s1 = panel[y == 1][f].median()
        s0 = panel[y == 0][f].median()
        tab.append({"factor": f, "AUC": a, "封板组中位": s1, "炸板组中位": s0,
                    "P(封|Q4)": hi_, "P(封|Q1)": lo_})
    t = pd.DataFrame(tab).sort_values("AUC", ascending=False, na_position="last")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    sub = panel[panel["grp"] == "sealed"].copy()
    if len(sub) >= 60:
        sub["y_hold"] = (sub["open_times"] == 0).astype(int)
        print(f"\n=== 任务B: 封板组内预测硬封不炸 (n={len(sub)}, "
              f"硬封率{sub['y_hold'].mean():.1%}) ===")
        tab = []
        for f in FEATS:
            a = auc(sub["y_hold"], sub[f])
            hi_, lo_ = qlift(sub["y_hold"], sub[f])
            tab.append({"factor": f, "AUC": a,
                        "P(硬封|Q4)": hi_, "P(硬封|Q1)": lo_})
        tb = pd.DataFrame(tab).sort_values("AUC", ascending=False,
                                           na_position="last")
        print(tb.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 描述: 触板→封死时滞 & 封板时间分布 & 炸板回踩深度
    se = df[df["grp"] == "sealed"].copy()
    se["at_limit"] = se["close"] >= se["limit_px"] * TOUCH_TOL
    print("\n=== 描述统计 ===")
    lag = []
    for (d, c), g in se.groupby(["date", "ts_code"]):
        g = g.sort_values("t")
        to = g["first_time"].iloc[0]
        ft = f"{str(to)[:2]}:{str(to)[2:4]}" if pd.notna(to) else None
        tt = g["t"].iloc[(g["high"] >= g["limit_px"].iloc[0] * TOUCH_TOL)
                         .values.argmax()] if (
            g["high"] >= g["limit_px"].iloc[0] * TOUCH_TOL).any() else None
        if ft and tt:
            fts = int(ft[:2]) * 60 + int(ft[3:5])
            tts = int(tt[:2]) * 60 + int(tt[2:4])
            lag.append(fts - tts)
    if lag:
        lag = pd.Series(lag)
        print(f"首触→首封时滞(分钟): 中位{lag.median():.0f} "
              f"均值{lag.mean():.1f} p25{lag.quantile(.25):.0f} "
              f"p75{lag.quantile(.75):.0f} (负=事件表首封早于K线首触, 口径差)")
    zb = panel[panel["grp"] == "zb"]
    if len(zb):
        print(f"炸板组收盘涨幅: 中位{zb['close_ret'].median():.2%} "
              f"均值{zb['close_ret'].mean():.2%} "
              f"收绿比例{(zb['close_ret'] < 0).mean():.1%}")
    tt = panel["t_touch"].astype(str).str[:2]
    print("首触时刻分布(小时):", tt.value_counts().sort_index().to_dict())
    OUT.mkdir(exist_ok=True)
    panel.to_csv(OUT / "06_minute_panel.csv", index=False)
    print(f"\n面板已存 research/out/06_minute_panel.csv ({len(panel)}行)")


if __name__ == "__main__":
    main()
