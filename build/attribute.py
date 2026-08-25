# -*- coding: utf-8 -*-
"""概念独占归属（iterative greedy）

规则:
  1. 只用 is_theme=True 的概念（过滤指数样本/属性类/超大杂烩）
  2. 每股候选 = 其成分概念 ∩ 当日有涨停股出现的概念
  3. 迭代: 每概念统计当前归属家数 → 每股归到候选中家数最大者
     平票取当日涨停密度(raw/成分数)更高者(真热点优先, v2),
     再平取成分数更小者(更聚焦), 再平取概念代码小者
  4. 收敛到不动点（上限20轮）；无候选者归 UNASSIGNED

v2密度tie-break背景(研究03): v1"成分数小"会把深中华A锁进两轮车(77成分,raw1)
而丢掉黄金概念(82成分,raw3)、金健米业锁进乳业(35,raw1)丢掉粮食概念(47,raw4)。
密度tie-break使大热点漏标 48.8%→21.7%, 现实格 n 984→4237、日聚类t 22.6→36.0。

产物:
  data/attribution.parquet   trade_date, ts_code, concept_code, concept_name, rounds
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA


def load_maps():
    concepts = pd.read_parquet(DATA / "concepts.parquet")
    members = pd.read_parquet(DATA / "concept_members.parquet")
    theme = concepts[concepts["is_theme"]]
    theme_codes = set(theme["ts_code"])
    msize = theme.set_index("ts_code")["member_count"].to_dict()
    cname = theme.set_index("ts_code")["name"].to_dict()

    # 股票 → [题材概念]
    mem = members[members["concept_code"].isin(theme_codes)]
    stock2con = (mem.groupby("con_code")["concept_code"]
                 .apply(lambda s: sorted(set(s))).to_dict())
    return stock2con, msize, cname


def load_con2stock() -> dict:
    """概念 → 成分股代码列表 (仅is_theme概念)"""
    concepts = pd.read_parquet(DATA / "concepts.parquet")
    members = pd.read_parquet(DATA / "concept_members.parquet")
    theme_codes = set(concepts[concepts["is_theme"]]["ts_code"])
    mem = members[members["concept_code"].isin(theme_codes)]
    return (mem.groupby("concept_code")["con_code"]
            .apply(lambda s: sorted(set(s))).to_dict())


def attribute_day(codes: list[str], stock2con: dict, msize: dict) -> dict:
    """单日独占归属, 返回 {ts_code: concept_code}"""
    raw_cnt = defaultdict(int)          # 归属前每概念触及家数
    cand = {}
    for c in codes:
        cons = stock2con.get(c, [])
        cand[c] = cons
        for k in cons:
            raw_cnt[k] += 1
    # 候选只保留当日有涨停出现的概念
    active = set(raw_cnt)
    for c in codes:
        cand[c] = [k for k in cand[c] if k in active]

    attr = {}
    dens = {k: raw_cnt[k] / msize.get(k, 10**9) for k in active}
    for rnd in range(1, 21):
        cnt = defaultdict(int)
        for c, k in attr.items():
            cnt[k] += 1
        new_attr = {}
        for c in codes:
            ks = cand[c]
            if not ks:
                new_attr[c] = "UNASSIGNED"
                continue
            # cnt最大; 平票取当日涨停密度高(真热点优先); 再平取成分数小; 再平取代码升序
            new_attr[c] = sorted(
                ks, key=lambda k: (-cnt.get(k, 0), -dens.get(k, 0),
                                   msize.get(k, 10**9), k))[0]
        if new_attr == attr:
            return new_attr, rnd
        attr = new_attr
    return attr, 20


def touch_map(codes: list[str], stock2con: dict, msize: dict):
    """多概念触及层(展示用, 不参与独占统计): 每股 → 当日有涨停出现的全部
    成分题材概念。返回 (raw_cnt: {概念: 触及家数}, touches: {股票: [概念,...]}),
    touches 按触及家数降序 → 成分数小 → 代码升序。"""
    raw_cnt = defaultdict(int)
    cons_of = {}
    for c in codes:
        cons = stock2con.get(c, [])
        cons_of[c] = cons
        for k in cons:
            raw_cnt[k] += 1
    touches = {c: sorted(cons_of[c],
                         key=lambda k: (-raw_cnt[k], msize.get(k, 10**9), k))
               for c in codes}
    return dict(raw_cnt), touches


def main():
    ev = pd.read_parquet(DATA / "events_enriched.parquet",
                         columns=["trade_date", "ts_code"])
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
    att.to_parquet(DATA / "attribution.parquet", index=False)

    st = pd.DataFrame(stats, columns=["trade_date", "zt_cnt", "rounds", "unassigned"])
    print(f"归属完成 {len(att)} 行 → attribution.parquet")
    print(f"  平均迭代轮数 {st['rounds'].mean():.2f}, "
          f"未归属占比 {st['unassigned'].sum() / st['zt_cnt'].sum():.2%}")


if __name__ == "__main__":
    main()
