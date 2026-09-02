# -*- coding: utf-8 -*-
"""离线归属流水线: 全历史事件 → theme.attribution

归属算法本体在 core/attribute.py（盘中poller/雷达同样调用该出处）；
本文件只负责全量批处理与落盘。

kpl源(默认): 2024-01(kpl事件库起点)起用开盘啦theme直标, 之前的历史
段保留同花顺投票归属(研究可复现; 两段concept_code编码空间天然隔离:
885xxx.TI vs 题材名)。最近1-2日kpl数据T+1未到时自动延续法近似,
次日重跑被直标覆盖。
ths源: 全历史迭代投票(需--rebuild-ths强制重算历史段, 否则沿用旧结果)。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_SOURCE  # noqa: E402
from core.attribute import attribute_day, attribute_day_kpl, load_maps  # noqa: E402
from datastore import load, path_of, save  # noqa: E402


def build_kpl(ev: pd.DataFrame) -> pd.DataFrame:
    """kpl段: kpl事件库起点日期起逐日直标(缺失日延续近似)"""
    kpl = load("limitup.kpl_events", columns=["trade_date"])
    kpl_start = kpl["trade_date"].min()
    dates = sorted(d for d in ev["trade_date"].unique() if d >= kpl_start)
    rows = []
    src_cnt = {"direct": 0, "carry": 0}
    for d in dates:
        codes = ev[ev["trade_date"] == d]["ts_code"].tolist()
        attr, src = attribute_day_kpl(d, codes)
        src_cnt[src] = src_cnt.get(src, 0) + 1
        for c, k in attr.items():
            rows.append((d, c, k))
    att = pd.DataFrame(rows, columns=["trade_date", "ts_code", "concept_code"])
    att["concept_name"] = att["concept_code"].where(
        att["concept_code"] == "UNASSIGNED", att["concept_code"])
    att.loc[att["concept_code"] == "UNASSIGNED", "concept_name"] = "未分组"
    print(f"kpl段 {len(dates)} 日({dates[0]}~{dates[-1]}): "
          f"直标{src_cnt['direct']}日/延续{src_cnt['carry']}日, "
          f"{len(att)}行")
    return att


def build_ths(ev: pd.DataFrame, att_old: pd.DataFrame | None) -> pd.DataFrame:
    """同花顺段: 沿用旧结果(非--rebuild-ths)或全量重算投票
    (ths源时load_maps本身就是同花顺分支)"""
    if att_old is not None and "--rebuild-ths" not in sys.argv:
        return att_old
    stock2con, msize, cname = load_maps()
    print(f"事件 {len(ev)} 行, 题材概念 {len(msize)} 个, "
          f"有题材归属的股票 {len(stock2con)} 只")
    rows = []
    stats = []
    for d, grp in ev.groupby("trade_date"):
        codes = grp["ts_code"].tolist()
        attr, rnd = attribute_day(codes, stock2con, msize)
        for c, k in attr.items():
            rows.append((d, c, k))
        stats.append((d, len(codes), rnd,
                      sum(1 for k in attr.values() if k == "UNASSIGNED")))
    att = pd.DataFrame(rows, columns=["trade_date", "ts_code", "concept_code"])
    att["concept_name"] = att["concept_code"].map(cname).fillna("未分组")
    st = pd.DataFrame(stats, columns=["trade_date", "zt_cnt", "rounds",
                                      "unassigned"])
    print(f"ths段投票重算 {len(att)} 行, 平均迭代 {st['rounds'].mean():.2f}轮, "
          f"未归属 {st['unassigned'].sum() / st['zt_cnt'].sum():.2%}")
    return att


def _ths_maps():
    """直接读同花顺源映射(kpl源下重建历史段时用, 绕过CONCEPT_SOURCE分发)"""
    concepts = load("theme.concepts")
    members = load("theme.members")
    theme = concepts[concepts["is_theme"]]
    theme_codes = set(theme["ts_code"])
    msize = theme.set_index("ts_code")["member_count"].to_dict()
    cname = theme.set_index("ts_code")["name"].to_dict()
    mem = members[members["concept_code"].isin(theme_codes)]
    stock2con = (mem.groupby("con_code")["concept_code"]
                 .apply(lambda s: sorted(set(s))).to_dict())
    return stock2con, msize, cname


def main():
    ev = load("limitup.events_enriched", columns=["trade_date", "ts_code"])
    p_old = path_of("theme.attribution")
    att_old = load("theme.attribution") if p_old.exists() else None

    if CONCEPT_SOURCE == "kpl":
        kpl = load("limitup.kpl_events", columns=["trade_date"])
        kpl_start = kpl["trade_date"].min()
        # ths历史段: kpl起点前的旧归属直接沿用(不重算, 保研究可复现)
        ths_part = (att_old[att_old["trade_date"] < kpl_start]
                    if att_old is not None else None)
        kpl_part = build_kpl(ev)
        if ths_part is not None and len(ths_part):
            print(f"ths历史段沿用旧归属 {len(ths_part)} 行"
                  f"({ths_part['trade_date'].min()}~{ths_part['trade_date'].max()})")
            att = pd.concat([ths_part, kpl_part], ignore_index=True)
        else:
            att = kpl_part
    else:
        att = build_ths(ev, att_old)

    att = att.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    p = save("theme.attribution", att)
    un = (att["concept_code"] == "UNASSIGNED").mean()
    print(f"归属完成 {len(att)} 行, 未归属 {un:.2%} → {p}")


if __name__ == "__main__":
    main()
