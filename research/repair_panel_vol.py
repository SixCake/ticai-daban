# -*- coding: utf-8 -*-
"""修复 market.daily_panel 的 vol 单位断档(lab_333=股 / tushare补尾=手)

背景(实测):
  面板历史段由 lab_333 前复权日线构建, 其 volume 单位是「股」——与 tushare
  daily.vol(手)恰好100倍(2020~2026抽样比值恒为100.000); 2026-07-16 起改由
  tushare 补尾(手), 于是量能列在断档日整体跳变~100倍: 日K量柱断层、
  量比因子 y_volr5 与市场 vol_tot 在断档窗口失真。
  另 lab_333 文件末尾数日的 volume 还有额外个股畸变(实测0.92~1.76倍),
  需用 tushare 对应交易日重刷。

动作(先备份, 可重复执行):
  1. 探测断档日: 全市场 vol 中位数环比 < 0.2 的首日(该日起为手口径)
  2. 断档日之前 vol /= 100 (股→手), 统一到 tushare 手口径
  3. 断档日前 --tail-fix 个交易日用 tushare daily 重刷 vol(修lab_333尾段畸变)
  4. 校验: 断档前后中位量连续 + 抽样个股与 tushare 逐日比对≈1.0

CLI:
  python research/repair_panel_vol.py --dry-run     # 只诊断不落盘
  python research/repair_panel_vol.py               # 备份+修复
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_pro  # noqa: E402
from datastore import path_of  # noqa: E402

pro = get_pro()
OUT = path_of("market.daily_panel")
BAK = OUT.with_name(OUT.stem + ".pre_volrepair.parquet")

SAMPLE_CODES = ["000001.SZ", "600519.SH", "300750.SZ", "688256.SH",
                "601086.SH"]


def detect_break(panel: pd.DataFrame) -> str | None:
    """断档日(含): 全市场vol中位数环比骤降<0.2的首日; 无则None"""
    med = panel.groupby("trade_date")["vol"].median().sort_index()
    ratio = med / med.shift(1)
    cand = ratio[ratio < 0.2]
    return str(cand.index[0]) if len(cand) else None


def fetch_vol(days: list[str]) -> dict:
    """{(trade_date, ts_code): vol手}, tushare官方口径"""
    out = {}
    for d in days:
        df = None
        for attempt in range(3):
            try:
                df = pro.daily(trade_date=d)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [FAIL] {d}: {e}")
                time.sleep(2 * (attempt + 1))
        if df is not None and len(df):
            for r in df.itertuples():
                out[(d, r.ts_code)] = float(r.vol)
        time.sleep(0.2)
    return out


def verify(panel: pd.DataFrame, break_date: str):
    """断档前后中位量连续性 + 抽样个股与tushare逐日比值"""
    med = panel.groupby("trade_date")["vol"].median().sort_index()
    ratio = med / med.shift(1)
    win = ratio[(ratio.index >= _shift(med.index, break_date, -3))
                & (ratio.index <= _shift(med.index, break_date, 2))]
    print(f"[校验] 断档前后中位量环比: {win.round(3).to_dict()}")
    days = sorted(panel.loc[panel["trade_date"] < break_date,
                            "trade_date"].unique())[-2:]
    ts = fetch_vol(days)
    for c in SAMPLE_CODES:
        s = panel[panel["ts_code"] == c].set_index("trade_date")
        for d in days:
            if d in s.index and (d, c) in ts:
                print(f"[校验] {c} {d} panel/tushare = "
                      f"{s.loc[d, 'vol'] / ts[(d, c)]:.4f}")


def _shift(idx, date: str, k: int) -> str:
    """交易日列表内偏移k位(越界夹边), 供校验窗口取邻日"""
    i = list(idx).index(date) if date in idx else 0
    return str(idx[max(0, min(len(idx) - 1, i + k))])


def main():
    ap = argparse.ArgumentParser(description="面板vol单位断档修复")
    ap.add_argument("--dry-run", action="store_true", help="只诊断不落盘")
    ap.add_argument("--tail-fix", type=int, default=5,
                    help="断档日前用tushare重刷的交易日数(修lab_333尾段畸变)")
    args = ap.parse_args()

    panel = pd.read_parquet(OUT)
    print(f"面板 {len(panel)} 行, {panel['trade_date'].min()}~"
          f"{panel['trade_date'].max()}")
    break_date = detect_break(panel)
    if not break_date:
        print("未发现单位断档(已是统一手口径, 无需修复)")
        return
    n_pre = int((panel["trade_date"] < break_date).sum())
    print(f"断档日 {break_date}: 之前 {n_pre} 行为股口径(lab_333), "
          f"之后为手口径(tushare补尾)")

    dates = sorted(panel.loc[panel["trade_date"] < break_date,
                             "trade_date"].unique())
    fix_days = dates[-args.tail_fix:]
    print(f"重刷 lab_333 尾段 {len(fix_days)} 日: {fix_days[0]}~{fix_days[-1]}")
    ts = fetch_vol(fix_days)
    print(f"  tushare 取到 {len(ts)} 行")

    mask = panel["trade_date"] < break_date
    panel.loc[mask, "vol"] = panel.loc[mask, "vol"] / 100.0     # 股→手
    fresh = pd.DataFrame([(d, c, v) for (d, c), v in ts.items()],
                         columns=["trade_date", "ts_code", "vol_new"])
    panel = panel.merge(fresh, on=["trade_date", "ts_code"], how="left")
    hit = int(panel["vol_new"].notna().sum())
    m = panel["vol_new"].notna()
    panel.loc[m, "vol"] = panel.loc[m, "vol_new"]
    panel = panel.drop(columns=["vol_new"])
    print(f"  尾段重刷命中 {hit} 行")

    verify(panel, break_date)
    if args.dry_run:
        print("[dry-run] 不落盘")
        return
    if not BAK.exists():
        shutil.copy2(OUT, BAK)
        print(f"备份原面板 → {BAK}")
    panel.to_parquet(OUT, index=False)
    print(f"修复完成 → {OUT}")


if __name__ == "__main__":
    main()
