# -*- coding: utf-8 -*-
"""一次性导入: 龙虎榜游资技能历史数据(workspace) → 项目 hmlist 域

来源(/Users/mcfell/.qoderwork/workspace/mtih4dp8bfd0wbvf/outputs/):
  hm_predict_{T}_{N}.csv ×14        → hmlist.picks (20260831~20260917, 待回填实际收益)
  hm_seat_signals_202608.csv        → hmlist.picks (20260803~20260831, 自带已验证收益)
  hm_seat_rolling3m_rating_202608.csv → hmlist.seat_rating (rating_month=202608)
  hm_detail_20260831.csv            → hmlist.detail 种子

幂等: picks 按 (trade_date,hm_name,ts_code,src) 去重, 重复导入不产生重复行。
实际收益回填不在本脚本: 由 build/hm_picks.py --backfill 统一用 daily_panel 计算。

CLI:
  python collect/import_hm_history.py [--src WORKSPACE_OUTPUTS_DIR]
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load, path_of, save  # noqa: E402

DEFAULT_SRC = Path("/Users/mcfell/.qoderwork/workspace/mtih4dp8bfd0wbvf/outputs")

PICK_COLS = ["trade_date", "next_date", "hm_name", "ts_code", "ts_name",
             "buy_amount", "sell_amount", "net_amount",
             "orig_grade", "orig_score", "grade", "score",
             "hist_signals", "hist_win_rate", "pred_ret1", "pred_ret2",
             "base_close", "base_pct",
             "act_open_ret", "act_close_ret", "act_2d_ret",
             "verdict", "hit_top", "src"]


def _hit_top(row) -> bool:
    return (row.get("grade") in ("A", "B")) or \
        (pd.notna(row.get("score")) and float(row["score"]) >= 60)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    # 只保留净买入信号(与 build/hm_picks.py 同口径): 原技能CSV把席位
    # 出货日(净额<0)也记为TOP1看涨预测, 语义相反, 剔除
    if len(df):
        buy = pd.to_numeric(df.get("buy_amount"), errors="coerce").fillna(0)
        net = pd.to_numeric(df.get("net_amount"), errors="coerce").fillna(0)
        drop = ~((net > 0) & (buy >= 1e5))
        if drop.any():
            print(f"    剔除净卖出/微量买入行: {int(drop.sum())}")
        df = df[~drop]
    for c in PICK_COLS:
        if c not in df.columns:
            df[c] = None
    df["hit_top"] = df.apply(_hit_top, axis=1)
    df["verdict"] = df["verdict"].fillna("pending")
    return df[PICK_COLS]


def import_predicts(src: Path) -> pd.DataFrame:
    """hm_predict_{T}_{N}.csv: 列名为中文, 映射到统一schema; 无实际收益(待回填)"""
    rows = []
    rename = {"游资名称": "hm_name", "原始评级": "orig_grade",
              "原始评分": "orig_score", "混合评级": "grade",
              "混合评分": "score", "历史信号数": "hist_signals",
              "历史胜率%": "hist_win_rate", "基准日收盘": "base_close",
              "基准日涨跌幅%": "base_pct", "预测次日收盘收益%": "pred_ret1",
              "预测次日2日收益%": "pred_ret2"}
    for f in sorted(src.glob("hm_predict_*_*.csv")):
        m = re.match(r"hm_predict_(\d{8})_(\d{8})\.csv", f.name)
        if not m:
            continue
        t, n = m.groups()
        df = pd.read_csv(f, encoding="utf-8-sig").rename(columns=rename)
        df["trade_date"], df["next_date"], df["src"] = t, n, "import_predict"
        rows.append(df)
        print(f"  {f.name}: {len(df)} 行")
    if not rows:
        return pd.DataFrame(columns=PICK_COLS)
    return _norm(pd.concat(rows, ignore_index=True))


def import_signals(src: Path) -> pd.DataFrame:
    """hm_seat_signals_202608.csv: 只取选股事实, 实际收益不导入。

    源表的 close_return_pct/2day_return_pct 不可信:
      ① 2day口径是 T-1收盘→T+1收盘(含T日涨幅), 与本项目 act_2d_ret
        (T收盘→T+2收盘)定义不同, 同列混用会污染均值;
      ② 新股 sig_pre_close=发行价, 2日收益算出660%级垃圾值(超纯应材)。
    统一由 build/hm_picks.py --backfill 用 daily_panel 重算(单一事实源)。
    """
    f = src / "hm_seat_signals_202608.csv"
    if not f.exists():
        print("  signals 文件缺失, 跳过")
        return pd.DataFrame(columns=PICK_COLS)
    df = pd.read_csv(f, encoding="utf-8-sig")
    out = pd.DataFrame({
        "trade_date": df["_trade_date"].astype(str),
        "next_date": df["_next_date"].astype(str),
        "hm_name": df["hm_name"], "ts_code": df["ts_code"],
        "ts_name": df["ts_name"],
        "buy_amount": df["buy_amount"], "sell_amount": df["sell_amount"],
        "net_amount": df["net_amount"],
        "base_close": df["sig_close"],
        "src": "import_signals"})
    print(f"  {f.name}: {len(out)} 行")
    return _norm(out)


def import_rating(src: Path):
    cands = sorted(src.glob("hm_seat_rolling3m_rating_*.csv"))
    if not cands:
        print("  评级表缺失, 跳过")
        return
    f = cands[-1]
    month = re.search(r"(\d{6})\.csv", f.name).group(1)
    df = pd.read_csv(f, encoding="utf-8-sig")
    df.insert(0, "rating_month", month)
    p = path_of("hmlist.seat_rating")
    if p.exists():
        old = load("hmlist.seat_rating")
        df = pd.concat([old[old["rating_month"] != month], df],
                       ignore_index=True)
    save("hmlist.seat_rating", df.sort_values(["rating_month", "游资名称"]))
    print(f"  评级表 {f.name}: {len(df)} 席位 → {p}")


def import_detail_seed(src: Path):
    f = src / "hm_detail_20260831.csv"
    p = path_of("hmlist.detail")
    if not f.exists() or p.exists():
        return
    df = pd.read_csv(f, encoding="utf-8-sig")
    for c in ["buy_amount", "sell_amount", "net_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["trade_date"] = df["trade_date"].astype(str)
    save("hmlist.detail", df)
    print(f"  明细种子 {f.name}: {len(df)} 行 → {p}")


def main():
    ap = argparse.ArgumentParser(description="游资席位历史数据导入")
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="workspace outputs 目录")
    args = ap.parse_args()
    src = Path(args.src)

    print("==> 导入 picks")
    picks = pd.concat([import_predicts(src), import_signals(src)],
                      ignore_index=True)
    p = path_of("hmlist.picks")
    if p.exists():
        old = load("hmlist.picks")
        picks = pd.concat([old, picks], ignore_index=True)
    picks = picks.drop_duplicates(
        subset=["trade_date", "hm_name", "ts_code", "src"], keep="last")
    # 同票同席位跨src重叠(20260831: signals已验证 vs predict待回填) →
    # 优先保留 signals(自带实际收益)
    picks = picks.sort_values("src", key=lambda s: s.map(
        {"import_signals": 1, "import_predict": 0, "daily": 2}).fillna(3))
    picks = picks.drop_duplicates(
        subset=["trade_date", "hm_name", "ts_code"], keep="last")
    picks = picks.sort_values(["trade_date", "hm_name"]).reset_index(drop=True)
    save("hmlist.picks", picks)
    n_pend = int((picks["verdict"] == "pending").sum())
    print(f"picks {len(picks)} 行 ({picks['trade_date'].min()}~"
          f"{picks['trade_date'].max()}), 待回填 {n_pend} → {p}")

    print("==> 导入 seat_rating / detail 种子")
    import_rating(src)
    import_detail_seed(src)


if __name__ == "__main__":
    main()
