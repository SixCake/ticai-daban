# -*- coding: utf-8 -*-
"""题材日度快照: 独占归属结果 → 每日每题材的涨停家数/高度/龙头

龙头判定（游资世界观）: 连板高度 > 封单额 > 首封时间早 > 炸板次数少

产物: theme.day
  trade_date, concept_code, concept_name,
  zt_cnt(独占后家数), zt_cnt_raw(归属前触及家数),
  max_height(最高连板), leader_code, leader_name, leader_height, leader_fd_amount,
  theme_age(该题材连续有涨停的天数)

kpl段(2024-01起): concept_code即题材名(name恒等), 触及家数用当日
kpl涨停事件theme统计; ths历史段沿用同花顺成分触及口径。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_SOURCE  # noqa: E402
from core.attribute import touch_map_kpl  # noqa: E402
from datastore import load, save  # noqa: E402


def main():
    att = load("theme.attribution")
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "name", "limit_times",
                       "fd_amount", "first_min", "open_times", "is_st"])

    # kpl段起点(无kpl事件库时全程同花顺口径)
    kpl_start = None
    if CONCEPT_SOURCE == "kpl":
        kpl_start = load("limitup.kpl_events",
                         columns=["trade_date"])["trade_date"].min()

    # 归属前每概念触及家数
    mem = load("theme.members")
    concepts = load("theme.concepts")
    theme_codes = set(concepts[concepts["is_theme"]]["ts_code"])
    mem_t = mem[mem["concept_code"].isin(theme_codes)]
    stock2con = mem_t.groupby("con_code")["concept_code"].apply(set).to_dict()

    m = att.merge(ev, on=["trade_date", "ts_code"], how="left")

    rows = []
    for (d, k), grp in m.groupby(["trade_date", "concept_code"]):
        if k == "UNASSIGNED":
            continue
        # 龙头: 连板高度降序 → 封单额降序 → 首封早 → 炸板少
        g = grp.sort_values(["limit_times", "fd_amount", "first_min", "open_times"],
                            ascending=[False, False, True, True])
        leader = g.iloc[0]
        rows.append((d, k, len(grp), g["limit_times"].max(),
                     leader["ts_code"], leader["name"], leader["limit_times"],
                     leader["fd_amount"]))

    td = pd.DataFrame(rows, columns=["trade_date", "concept_code", "zt_cnt",
                                     "max_height", "leader_code", "leader_name",
                                     "leader_height", "leader_fd_amount"])
    # kpl段题材名即代码(name恒等), ths段用同花顺代码→名映射
    td["concept_name"] = td["concept_code"].map(
        concepts.set_index("ts_code")["name"]).fillna(td["concept_code"])

    # 归属前触及家数: kpl段当日theme直标统计 / ths段成分交集
    raw_cnt = []
    for d, grp in ev.groupby("trade_date"):
        if kpl_start and d >= kpl_start:
            cnt = touch_map_kpl(d, grp["ts_code"].tolist())[0]
        else:
            cnt = {}
            for c in grp["ts_code"]:
                for k in stock2con.get(c, ()):
                    cnt[k] = cnt.get(k, 0) + 1
        for k, v in cnt.items():
            raw_cnt.append((d, k, v))
    rc = pd.DataFrame(raw_cnt, columns=["trade_date", "concept_code", "zt_cnt_raw"])
    td = td.merge(rc, on=["trade_date", "concept_code"], how="left")
    td["zt_cnt_raw"] = td["zt_cnt_raw"].fillna(0).astype(int)

    # theme_age: 题材连续有独占涨停的天数
    td = td.sort_values(["concept_code", "trade_date"])
    dates_all = np.array(sorted(ev["trade_date"].unique()))
    dpos = {d: i for i, d in enumerate(dates_all)}
    age = []
    for k, grp in td.groupby("concept_code"):
        pos = grp["trade_date"].map(dpos).values
        streak = np.ones(len(pos), dtype=int)
        for i in range(1, len(pos)):
            if pos[i] == pos[i - 1] + 1:
                streak[i] = streak[i - 1] + 1
        age.append(pd.Series(streak, index=grp.index))
    td["theme_age"] = pd.concat(age).sort_index()

    cols = ["trade_date", "concept_code", "concept_name", "zt_cnt", "zt_cnt_raw",
            "max_height", "theme_age", "leader_code", "leader_name",
            "leader_height", "leader_fd_amount"]
    td = td[cols].sort_values(["trade_date", "zt_cnt"], ascending=[True, False])
    p = save("theme.day", td)
    print(f"题材日度快照 {len(td)} 行 → {p}")
    print(f"  覆盖日期 {td['trade_date'].min()}~{td['trade_date'].max()}, "
          f"题材数 {td['concept_code'].nunique()}")


if __name__ == "__main__":
    main()
