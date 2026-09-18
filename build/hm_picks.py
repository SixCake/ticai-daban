# -*- coding: utf-8 -*-
"""龙虎榜席位TOP1每日选股 + 实际收益回填(hmlist.picks 唯一写出口)

两步(可分别执行):
  ① generate: 对 hmlist.detail 最新交易日(或--date), 每个游资席位按当日
     买入额取TOP1买入股, 映射最新月度滚动评级(hmlist.seat_rating)得到
     评级快照与预测收益, 追加进 picks(src=daily)。已有该日记录则跳过。
  ② backfill: 对所有 act_close_ret 缺失的行, 用 market.daily_panel 计算
     T+1开盘/收盘收益与T+2收盘收益(相对基准日收盘, %), 并判 verdict
     (次日收盘>0=win)。面板未覆盖的行保持 pending。

口径与技能 predict_next_day.py 一致:
  act_open_ret  = (T+1 open / base_close - 1) * 100
  act_close_ret = (T+1 close / base_close - 1) * 100
  act_2d_ret    = (T+2 close / base_close - 1) * 100

CLI:
  python build/hm_picks.py                    # 最新明细日生成 + 全量回填
  python build/hm_picks.py --date 20260917    # 指定基准日
  python build/hm_picks.py --backfill-only    # 只回填不生成
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load, path_of, save  # noqa: E402

RATING_JOIN = {"游资名称": "hm_name", "原始评级": "orig_grade",
               "原始评分": "orig_score", "混合评级": "grade",
               "混合评分": "score", "验证信号数": "hist_signals",
               "胜率%": "hist_win_rate", "次日收益%": "pred_ret1",
               "2日收益%": "pred_ret2"}


def _open_dates() -> list:
    cal = load("meta.trade_cal")
    return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].astype(str))


def _latest_rating() -> pd.DataFrame:
    """最新月度滚动评级快照(中文列→统一英文列)"""
    p = path_of("hmlist.seat_rating")
    if not p.exists():
        return pd.DataFrame(columns=list(RATING_JOIN.values()))
    rt = load("hmlist.seat_rating")
    month = rt["rating_month"].max()
    rt = rt[rt["rating_month"] == month]
    cols = [c for c in RATING_JOIN if c in rt.columns]
    return rt[cols].rename(columns=RATING_JOIN)


def generate(date: str | None) -> pd.DataFrame:
    """基准日席位TOP1选股行(src=daily); detail/rating缺失返回空表"""
    dp = path_of("hmlist.detail")
    if not dp.exists():
        print("hmlist.detail 不存在, 先跑 collect/fetch_hm_detail.py")
        return pd.DataFrame()
    det = load("hmlist.detail")
    if date is None:
        date = str(det["trade_date"].max())
    day = det[det["trade_date"].astype(str) == date]
    if day.empty:
        print(f"{date} 无游资明细(当晚数据可能尚未更新)")
        return pd.DataFrame()

    # 席位×个股聚合 → 每席位买入额TOP1
    g = day.groupby(["hm_name", "ts_code", "ts_name"], as_index=False).agg(
        buy_amount=("buy_amount", "sum"), sell_amount=("sell_amount", "sum"),
        net_amount=("net_amount", "sum"))
    # 只保留净买入信号: 原技能口径仅要求买入额>0, 会把席位出货日
    # (买1万卖3千万, 净额<0)也选成TOP1看涨信号 —— 语义相反, 必须剔除
    g = g[(g["net_amount"] > 0) & (g["buy_amount"] >= 1e5)]
    idx = g.groupby("hm_name")["buy_amount"].idxmax()
    top1 = g.loc[idx].copy()

    top1 = top1.merge(_latest_rating(), on="hm_name", how="left")
    top1["grade"] = top1["grade"].fillna("NA")

    # 基准日行情(daily_panel)
    pn = load("market.daily_panel",
              columns=["trade_date", "ts_code", "close", "pct_chg"])
    px = pn[pn["trade_date"] == date].set_index("ts_code")
    top1["base_close"] = top1["ts_code"].map(px["close"])
    top1["base_pct"] = top1["ts_code"].map(px["pct_chg"])

    dates = _open_dates()
    nxt = next((d for d in dates if d > date), None)
    top1["trade_date"] = date
    top1["next_date"] = nxt
    top1["verdict"] = "pending"
    top1["src"] = "daily"
    top1["hit_top"] = (top1["grade"].isin(["A", "B"])
                       | (pd.to_numeric(top1["score"], errors="coerce") >= 60))
    print(f"{date} TOP1选股: {len(top1)} 席位 "
          f"(A/B级 {int(top1['grade'].isin(['A', 'B']).sum())})")
    return top1


def backfill(picks: pd.DataFrame) -> pd.DataFrame:
    """用 daily_panel 回填实际收益与 verdict(只补缺失, 不覆盖已有值)"""
    pend = picks["act_close_ret"].isna() & picks["next_date"].notna()
    if not pend.any():
        return picks
    pn = load("market.daily_panel",
              columns=["trade_date", "ts_code", "open", "close"])
    dates = _open_dates()
    dset = set(dates)
    didx = {d: i for i, d in enumerate(dates)}

    need = picks.loc[pend]
    want_days = set()
    for nd in need["next_date"].astype(str):
        want_days.add(nd)
        i = didx.get(nd)
        if i is not None and i + 1 < len(dates):
            want_days.add(dates[i + 1])
    sub = pn[pn["trade_date"].isin(want_days)]
    px = {(r.trade_date, r.ts_code): (r.open, r.close) for r in sub.itertuples()}

    n_fill = 0
    for i in picks.index[pend]:
        r = picks.loc[i]
        nd = str(r["next_date"])
        if nd not in dset or pd.isna(r["base_close"]) or not r["base_close"]:
            continue
        base = float(r["base_close"])
        p1 = px.get((nd, r["ts_code"]))
        if p1 is None or pd.isna(p1[1]):     # 面板未覆盖(停牌/未更新)
            continue
        picks.at[i, "act_open_ret"] = (
            round((p1[0] / base - 1) * 100, 3) if p1[0] else None)
        picks.at[i, "act_close_ret"] = round((p1[1] / base - 1) * 100, 3)
        picks.at[i, "verdict"] = "win" if p1[1] > base else "lose"
        j = didx.get(nd)
        if j is not None and j + 1 < len(dates):
            p2 = px.get((dates[j + 1], r["ts_code"]))
            if p2 is not None and p2[1] is not None:
                picks.at[i, "act_2d_ret"] = round((p2[1] / base - 1) * 100, 3)
        n_fill += 1
    print(f"回填实际收益: {n_fill} 行")
    return picks


def main():
    ap = argparse.ArgumentParser(description="席位TOP1选股生成/回填")
    ap.add_argument("--date", help="生成基准日YYYYMMDD(默认明细最新日)")
    ap.add_argument("--backfill-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="该日已有daily记录时仍重新生成(覆盖)")
    args = ap.parse_args()

    picks = load("hmlist.picks") if path_of("hmlist.picks").exists() \
        else pd.DataFrame()

    if not args.backfill_only:
        new = generate(args.date)
        if len(new):
            date = str(new["trade_date"].iloc[0])
            if len(picks):
                has = picks[picks["trade_date"].astype(str) == date]
                if len(has) and not args.force:
                    print(f"{date} 已有 picks {len(has)} 行(src={has['src'].iloc[0]}), "
                          f"跳过生成(--force 覆盖)")
                    new = pd.DataFrame()
                elif len(has):
                    picks = picks[~picks.index.isin(has.index)]
            if len(new):
                picks = pd.concat([picks, new], ignore_index=True)

    if len(picks):
        picks = backfill(picks)
        picks = picks.drop_duplicates(
            subset=["trade_date", "hm_name", "ts_code", "src"], keep="last")
        picks = picks.sort_values(
            ["trade_date", "hm_name"]).reset_index(drop=True)
        out = save("hmlist.picks", picks)
        n_win = int((picks["verdict"] == "win").sum())
        n_done = int(picks["verdict"].isin(["win", "lose"]).sum())
        print(f"picks {len(picks)} 行, 已验证 {n_done} (胜率 "
              f"{n_win / n_done * 100:.1f}%) → {out}" if n_done
              else f"picks {len(picks)} 行 → {out}")


if __name__ == "__main__":
    main()
