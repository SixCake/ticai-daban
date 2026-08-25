# -*- coding: utf-8 -*-
"""离线归属流水线: 全历史事件 → data/attribution.parquet

归属算法本体在 core/attribute.py（盘中poller/雷达同样调用该出处）；
本文件只负责全量批处理与落盘。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA
from core.attribute import attribute_day, load_maps


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
