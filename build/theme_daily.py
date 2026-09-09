# -*- coding: utf-8 -*-
"""题材日度快照: 独占归属结果 → 每日每题材的涨停家数/高度/龙头

龙头判定（游资世界观）: 连板高度 > 封单额 > 首封时间早 > 炸板次数少

产物: theme.day
  trade_date, concept_code, concept_name,
  zt_cnt(独占后家数), zt_cnt_raw(归属前关联家数),
  max_height(最高连板), leader_code, leader_name, leader_height, leader_fd_amount,
  theme_age(该题材连续有独占涨停的天数 —— 持续性, 不是阶段),
  zt_all(归因自由关联家数), wave_no(题材历史第几波)

kpl段: concept_code即题材名(name恒等), 关联家数用当日kpl涨停事件theme统计;
ths历史段沿用同花顺成分关联口径。

zt_all / wave_no 为何不能用 zt_cnt / theme_age 代替（研究36~39）:
  · theme.day 的行存在条件是「该题材当日有≥1只**独占**涨停股」, 而 kpl
    独占归属只取该股 theme 标注的第一个题材 → 题材强度被标签排序随机性
    污染, 且题材在场但独占为0的日子会从表里消失;
  · theme_age 数的是连续出现天数 = 持续性, 与阶段正交
    (Spearman(age,家数)=+0.25), 用它切「爆发/主升/鱼尾」方向会反置;
  · 所以波次必须在**归因自由**面板(kpl 直标全 tag, 不经独占投票)上算,
    口径见 core.cycle.segment_waves / theme_stage。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_SOURCE  # noqa: E402
from core.attribute import touch_map_kpl  # noqa: E402
from core.cycle import WAVE_MIN_ZT  # noqa: E402
from core.theme_wave import activity_panel, wave_table  # noqa: E402
from datastore import load, save  # noqa: E402


def add_wave_no(td: pd.DataFrame, act: pd.DataFrame,
                dates_all: list) -> pd.DataFrame:
    """把归因自由关联家数 + 波次挂到 theme.day 行上

    口径全部走 core.theme_wave（与盘中 poller 共用，避免两处算出不同
    的 wave_no）。两个关键点: 波次历史用全量活跃面板（包含独占为0
    因而不在 theme.day 里的日子）; 切波在所有有涨停的日子上算。

    kpl T+1 兜底: activity_panel 直读 kpl_events, 而 kpl 是 T+1 拉取 ——
    当晚复盘时最新交易日尚未入库, act 缺该日 → 当日题材 zt_all/wave_no
    全空(阶段判不出, 天梯 heat3 失真)。用 td 已算好的 zt_cnt_raw
    (touch_map_kpl 同口径: 今日涨停股全tag关联家数, 含kpl缺失兜底)补
    该日 panel 行。只补 theme.day 已有的题材(今日有独占涨停者), 「在场
    但独占为0」的题材明日 kpl 入库自然修复。不改 activity_panel 本身
    (盘中 poller 用 wave_state 历史基线, 加今日行会破坏其空档判断)。
    """
    if len(act):
        act_dates = set(act["trade_date"].unique())
        miss = td[~td["trade_date"].isin(act_dates)]
        if len(miss):
            extra = miss[["trade_date", "concept_code", "zt_cnt_raw"]].rename(
                columns={"zt_cnt_raw": "zt_all"})
            extra = extra[extra["zt_all"] > 0]
            act = pd.concat([act, extra], ignore_index=True)
    wt = wave_table(act, dates_all)
    td = td.merge(wt[["trade_date", "concept_code", "zt_all", "wave_no"]],
                  on=["trade_date", "concept_code"], how="left")
    td["zt_all"] = td["zt_all"].fillna(0).astype(int)
    td["wave_no"] = td["wave_no"].fillna(0).astype(int)
    return td


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

    # 归属前每概念关联家数
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

    # 归属前关联家数: kpl段当日theme直标统计 / ths段成分交集
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

    # 归因自由关联家数 + 波次（研究39 定稿题材阶段的输入）
    td = add_wave_no(td, activity_panel(), list(dates_all))

    cols = ["trade_date", "concept_code", "concept_name", "zt_cnt", "zt_cnt_raw",
            "zt_all", "wave_no", "max_height", "theme_age", "leader_code",
            "leader_name", "leader_height", "leader_fd_amount"]
    td = td[cols].sort_values(["trade_date", "zt_cnt"], ascending=[True, False])
    p = save("theme.day", td)
    print(f"题材日度快照 {len(td)} 行 → {p}")
    print(f"  覆盖日期 {td['trade_date'].min()}~{td['trade_date'].max()}, "
          f"题材数 {td['concept_code'].nunique()}")
    print(f"  波次挂载率 {(td['wave_no'] > 0).mean() * 100:.1f}% · "
          f"最大波次 {td['wave_no'].max()} · "
          f"zt_all 均值 {td['zt_all'].mean():.1f} · "
          f"在场率 {(td['zt_all'] >= WAVE_MIN_ZT).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
