# -*- coding: utf-8 -*-
"""采集开盘啦题材板块列表与成分（tushare kpl_concept / kpl_concept_cons, 当前快照）

产物:
  theme.kpl_concepts  题材板块列表(.KP代码, 含is_theme过滤标记/member_count)
  theme.kpl_members   板块成分快照(含入选原因desc/人气值hot_num)

注意: 成分接口单页上限3000行, 须offset翻页拉全; kpl_concept_cons
按trade_date查历史快照(2024-09起可查), 本脚本只拉最近交易日快照,
历史时点回看供后续研究需要时再回填。
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_NOISE_KEYWORDS, MAX_MEMBER_COUNT, get_pro  # noqa: E402
from datastore import save  # noqa: E402

pro = get_pro()

PAGE = 3000


def latest_trade_date() -> str:
    """最近一个可拉到成分数据的交易日(kpl_concept_cons当日T+0有, 兜底往前找)"""
    today = datetime.now().strftime("%Y%m%d")
    end = today if datetime.now().hour >= 16 else \
        (datetime.fromordinal(datetime.now().toordinal() - 1)
         .strftime("%Y%m%d"))
    cal = pro.trade_cal(exchange="SSE", start_date="20260101", end_date=today,
                        is_open="1")
    for d in sorted(cal["cal_date"].tolist(), reverse=True):
        if d <= end:
            return d
    return end


def fetch_cons_all(date: str) -> pd.DataFrame:
    """成分全量(offset翻页, 单页上限3000)"""
    rows = []
    offset = 0
    while True:
        df = pro.kpl_concept_cons(trade_date=date, offset=offset)
        if df is None or not len(df):
            break
        rows.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
        time.sleep(0.35)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    date = latest_trade_date()
    for _ in range(3):          # 当日快照未出时往前回退(最多3个候选日)
        print(f"拉取 {date} 开盘啦题材板块与成分…")
        idx = pro.kpl_concept(trade_date=date)
        time.sleep(0.35)
        mem = fetch_cons_all(date)
        if len(idx) and len(mem):
            break
        cal = pro.trade_cal(exchange="SSE", start_date="20260101",
                            end_date=date, is_open="1")
        days = sorted(cal["cal_date"].tolist())
        if len(days) < 2:
            break
        date = days[-2]
        time.sleep(0.35)
    if not len(idx) or not len(mem):
        print(f"[FAIL] 板块 {len(idx)} 行 / 成分 {len(mem)} 行, 数据为空")
        return

    idx["member_count"] = idx["ts_code"].map(
        mem.groupby("ts_code")["con_code"].nunique()).fillna(0).astype(int)
    idx["is_noise_kw"] = idx["name"].apply(
        lambda n: any(k in str(n) for k in CONCEPT_NOISE_KEYWORDS))
    idx["is_theme"] = ((~idx["is_noise_kw"])
                       & (idx["member_count"] <= MAX_MEMBER_COUNT)
                       & (idx["member_count"] > 0))
    idx["snapshot_date"] = date

    cols = ["ts_code", "name", "member_count", "is_noise_kw", "is_theme",
            "z_t_num", "up_num", "snapshot_date"]
    for c in cols:
        if c not in idx.columns:
            idx[c] = None
    save("theme.kpl_concepts", idx[cols])

    mem = mem.rename(columns={"ts_code": "concept_code", "name": "concept_name"})
    mem["snapshot_date"] = date
    mcols = ["concept_code", "concept_name", "con_code", "con_name",
             "snapshot_date", "desc", "hot_num"]
    for c in mcols:
        if c not in mem.columns:
            mem[c] = None
    save("theme.kpl_members", mem[mcols])

    print(f"题材板块 {len(idx)} 个(其中题材 {idx['is_theme'].sum()} 个), "
          f"成分明细 {len(mem)} 行, 覆盖股票 {mem['con_code'].nunique()} 只")
    print(f"板块成分数分布: "
          f"{idx[idx['is_theme']]['member_count'].describe().round(1).to_dict()}")


if __name__ == "__main__":
    main()
