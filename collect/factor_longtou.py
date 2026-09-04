# -*- coding: utf-8 -*-
"""龙头因子每日构建（研究22/23落地, P1展示层数据源）

口径与 research/22_longtou_factors.py 完全一致（近似口径亦沿用）:
- 决策日T的个股特征 = T-1日行按股票shift(1)（只用≤T-1数据）;
- 市场级(zt_prev/ld_prev/ldlr_prev/adv_prev/cycle_prev) = T-1全市场统计;
- 行业统计(ind_breadth/ind_ztdens/ind_score) = T-1日行业截面;
- 涨跌停价 = pre_close×(1±rate), 30/68→20% 其余10%, 未做ST修正;
- 量比用 vol 口径（daily_panel 无成交额）;
- 行业映射取 events 最新 industry（点时漂移忽略）。

qscore/sscore 由 core/longtou.py 统一计算（阈值冻结自研究23）。
产物: factor.longtou（trade_date×ts_code 宽表, 增量追加新交易日）。

尾行口径: research/22 的 shift 语义要求决策日行以当日 panel 行为载体,
盘中尚无当日 bar。故额外产出「尾行」——把 panel 末日 M 的累计特征/
市场/行业统计挂到 M 的下一交易日 T, 使 T 盘中即可使用 T-1 数据
(与研究口径"决策日T只用≤T-1数据"一致); 历史在样行不受影响。

用法:
  python collect/factor_longtou.py           # 增量(panel有新日期才计算)
  python collect/factor_longtou.py --force   # 全量重算
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.longtou import FACTOR_CONTRACT, qscore_of, sscore_of  # noqa: E402
from datastore import load, path_of, save  # noqa: E402

OUT_COLS = ["trade_date", "ts_code", "industry",
            "zb_cnt20", "zb_cnt5", "y_volr5", "neg_streak", "neg_deep",
            "ind_rank", "ind_gap", "ind_n", "y_cpos2", "y_pct",
            "ind_breadth", "ind_ztdens", "ind_score",
            "zt_prev", "ld_prev", "ldlr_prev", "adv_prev", "cycle_prev",
            "mvol_prev", "qscore", "sscore"]


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def next_trade_date(d: str) -> str | None:
    """d(YYYYMMDD)的下一交易日。

    meta.trade_cal 由 poller 以「当日」为 end_date 缓存, horizon 往往只到 d
    本身(收盘后跑本脚本时日历末日=panel末日) → 查不到下一日时必须退化为
    跳过周末, 否则尾行被静默丢弃、盘中无当日因子可用(因子滞后一决策日)。
    退化分支不校节假日: 宁可多挂一行非交易日因子(数值仍是 T-1 口径,
    无害), 也不能让整日因子缺失。
    """
    cal: list = []
    try:
        cal = sorted(load("meta.trade_cal")["cal_date"].tolist())
    except Exception:
        cal = []
    nxt = next((x for x in cal if x > d), None)
    if nxt:
        return nxt
    from datetime import datetime, timedelta
    dt0 = datetime.strptime(d, "%Y%m%d")
    for _ in range(10):
        dt0 += timedelta(days=1)
        if dt0.weekday() < 5:
            fd = dt0.strftime("%Y%m%d")
            print(f"[warn] 交易日历 horizon 止于 {cal[-1] if cal else '(空)'}, "
                  f"退化取 {fd} 作 {d} 的下一交易日(未校节假日)")
            return fd
    return None


CUM_COLS = ["zb_cnt20", "zb_cnt5", "y_volr5", "neg_streak", "neg_deep",
            "ind_rank", "ind_gap", "ind_n", "cpos", "pct_raw"]


def compute(panel: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """全量panel → 决策日因子宽表（行=决策日T, 特征=T-1可见值）"""
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # 涨跌停价近似（与research/22一致, 未做ST修正）
    rate = np.where(panel["ts_code"].str[:2].isin(["30", "68"]), 0.20, 0.10)
    limit_px = np.round(panel["pre_close"] * (1 + rate), 2)
    low_px = np.round(panel["pre_close"] * (1 - rate), 2)
    panel["sealed"] = panel["close"] >= limit_px * 0.999
    panel["touched"] = panel["high"] >= limit_px * 0.999
    panel["sealed_dn"] = panel["close"] <= low_px * 1.001
    panel["zb"] = panel["touched"] & ~panel["sealed"]      # 触板未封=炸板
    panel["cpos"] = np.where(
        panel["high"] > panel["low"],
        (panel["close"] - panel["low"]) / (panel["high"] - panel["low"]), 0.5)
    panel["neg"] = panel["pct_chg"] < 0

    # 行业映射（events最新口径）
    ind_map = (ev.sort_values("trade_date").groupby("ts_code")["industry"]
               .last().to_dict())
    panel["industry"] = panel["ts_code"].map(ind_map)

    t0 = time.time()
    g = panel.groupby("ts_code", sort=False)
    panel["zb_cnt20"] = g["zb"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    panel["zb_cnt5"] = g["zb"].transform(lambda s: s.rolling(5, min_periods=3).sum())
    panel["vol_ma5p"] = g["vol"].transform(lambda s: s.rolling(5, min_periods=3).mean().shift(1))
    panel["y_volr5"] = panel["vol"] / panel["vol_ma5p"]
    neg_id = (~panel["neg"]).groupby(panel["ts_code"]).cumsum()
    panel["neg_streak"] = np.where(
        panel["neg"], panel.groupby([panel["ts_code"], neg_id]).cumcount() + 1, 0)
    panel["neg_deep"] = g["pct_chg"].transform(
        lambda s: s.rolling(3, min_periods=1).min() <= -5.0)

    gi = panel.dropna(subset=["industry"]).groupby(["trade_date", "industry"])
    panel["ind_rank"] = gi["pct_chg"].rank(ascending=False, method="min")
    panel["ind_gap"] = gi["pct_chg"].transform("max") - panel["pct_chg"]
    panel["ind_n"] = gi["pct_chg"].transform("size")
    print(f"个股特征完成, 耗时{time.time() - t0:.0f}s")

    # shift(1): 行日D特征 → 决策日D可见（只用≤D-1数据）
    panel["pct_raw"] = panel["pct_chg"]
    # 每股最后一行(截至panel末日M的累计值, shift前捕获, 供尾行)
    last_rows = panel.groupby("ts_code", sort=False).tail(1).copy()
    shift_cols = ["zb_cnt20", "zb_cnt5", "y_volr5", "neg_streak", "neg_deep",
                  "ind_rank", "ind_gap", "ind_n", "cpos", "pct_raw"]
    panel[shift_cols] = g[shift_cols].shift(1)
    panel = panel.rename(columns={"cpos": "y_cpos2", "pct_raw": "y_pct"})

    # 市场/行业日频统计（决策日T取T-1值）
    panel["up"] = ~panel["neg"]
    md = panel.groupby("trade_date").agg(
        mkt_n=("ts_code", "size"), adv_n=("up", "sum"),
        zt_n=("sealed", "sum"), ld_n=("sealed_dn", "sum"),
        vol_tot=("vol", "sum")).sort_index()
    md["adv_ratio"] = md["adv_n"] / md["mkt_n"]
    md["vol_ratio"] = md["vol_tot"] / md["vol_tot"].shift(1)
    md["cycle"] = (45 * md["adv_ratio"] + 25 * clip01(md["zt_n"] / 80)
                   + 15 * clip01(1 - md["ld_n"] / 20)
                   + 15 * clip01(md["vol_ratio"] / 1.1))
    md["ldlr"] = md["ld_n"] / md["zt_n"].clip(lower=1)

    panel["ind_up"] = panel["up"].astype(float)
    idat = panel.dropna(subset=["industry"]).groupby(["trade_date", "industry"]).agg(
        ind_breadth=("ind_up", "mean"),
        ind_ztdens=("sealed", "mean"), ind_med=("pct_chg", "median"))
    idat["ind_score"] = (45 * idat["ind_breadth"]
                         + 25 * clip01((idat["ind_med"] + 2) / 4)
                         + 30 * clip01(idat["ind_ztdens"] / 0.08))
    idat = idat.reset_index()

    dates = sorted(md.index)
    mkt_lookup = md.shift(1).rename(columns={
        "zt_n": "zt_prev", "ld_n": "ld_prev", "adv_ratio": "adv_prev",
        "cycle": "cycle_prev", "ldlr": "ldlr_prev", "vol_ratio": "mvol_prev"})
    mkt_lookup = mkt_lookup[["zt_prev", "ld_prev", "adv_prev", "cycle_prev",
                             "ldlr_prev", "mvol_prev"]]
    date_to_next = dict(zip(dates[:-1], dates[1:]))
    idat_prev = idat.copy()
    idat_prev["trade_date"] = idat_prev["trade_date"].map(date_to_next)
    idat_prev = idat_prev.dropna(subset=["trade_date"])

    out = panel[["trade_date", "ts_code", "industry", "zb_cnt20", "zb_cnt5",
                 "y_volr5", "neg_streak", "neg_deep", "ind_rank", "ind_gap",
                 "ind_n", "y_cpos2", "y_pct"]].copy()
    out = out.merge(mkt_lookup, left_on="trade_date", right_index=True, how="left")
    out = out.merge(idat_prev, on=["trade_date", "industry"], how="left")

    # 尾行: panel末日M的累计值挂到下一交易日T, 使T盘中可用(数据≤M=T-1)
    m_date = panel["trade_date"].max()
    t_next = next_trade_date(m_date)
    if not t_next:
        print(f"[warn] 无法定出 {m_date} 的下一交易日, 尾行缺失 → "
              f"下一交易日盘中无因子可用")
    if t_next:
        tail = last_rows[["ts_code", "industry"] + CUM_COLS].copy()
        tail["trade_date"] = t_next
        tail = tail.rename(columns={"cpos": "y_cpos2", "pct_raw": "y_pct"})
        mrow = md.loc[m_date]
        for src, dst in [("zt_n", "zt_prev"), ("ld_n", "ld_prev"),
                         ("adv_ratio", "adv_prev"), ("cycle", "cycle_prev"),
                         ("ldlr", "ldlr_prev"), ("vol_ratio", "mvol_prev")]:
            tail[dst] = mrow[src]
        tail = tail.merge(
            idat[idat["trade_date"] == m_date]
            [["industry", "ind_breadth", "ind_ztdens", "ind_score"]],
            on="industry", how="left")
        out = pd.concat([out, tail], ignore_index=True)

    out["qscore"] = out.apply(qscore_of, axis=1).astype("Int8")
    out["sscore"] = out.apply(sscore_of, axis=1).astype("Int8")
    return out[OUT_COLS]


def main():
    ap = argparse.ArgumentParser(description="龙头因子每日构建")
    ap.add_argument("--force", action="store_true", help="全量重算(默认增量追加)")
    args = ap.parse_args()

    p = path_of("market.daily_panel")
    if not p.exists():
        sys.exit("market.daily_panel 不存在, 先跑 collect/fetch_daily_panel.py")
    panel = pd.read_parquet(p)
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "industry"])
    panel_max = panel["trade_date"].max()
    print(f"panel={len(panel):,}行 ~{panel_max}; events={len(ev):,}")

    out_p = path_of("factor.longtou")
    old = pd.read_parquet(out_p) if out_p.exists() else None
    # 需覆盖到 panel 末日的下一交易日(尾行), 否则盘中无当日因子可用
    need_date = next_trade_date(panel_max) or panel_max
    if old is not None and not args.force:
        old_max = old["trade_date"].max()
        if old_max >= need_date:
            print(f"因子已覆盖至 {old_max} (panel末日{panel_max}), 无新交易日, 跳过")
            return
        print(f"增量: 追加 {old_max} 之后的行")

    feat = compute(panel, ev)
    if old is not None and not args.force:
        feat = pd.concat([old, feat[feat["trade_date"] > old["trade_date"].max()]],
                         ignore_index=True)
    save("factor.longtou", feat)
    last = feat["trade_date"].max()
    day = feat[feat["trade_date"] == last]
    print(f"产物 {len(feat):,}行 {feat['trade_date'].min()}~{last} "
          f"(合同{FACTOR_CONTRACT})")
    print(f"最新决策日{last}: {len(day):,}只 "
          f"qscore覆盖{day['qscore'].notna().mean():.0%} "
          f"sscore覆盖{day['sscore'].notna().mean():.0%} "
          f"zt_prev={day['zt_prev'].iloc[0]:.0f} "
          f"ld_prev={day['ld_prev'].iloc[0]:.0f} "
          f"ldlr_prev={day['ldlr_prev'].iloc[0]:.2f}")


if __name__ == "__main__":
    main()
