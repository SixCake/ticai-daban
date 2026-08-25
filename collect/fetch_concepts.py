# -*- coding: utf-8 -*-
"""采集概念列表与成分（tushare 同花顺概念, 当前快照）

产物:
  theme.concepts  概念列表(含is_theme过滤标记/member_count)
  theme.members   概念成分快照
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_NOISE_KEYWORDS, MAX_MEMBER_COUNT, get_pro
from datastore import save

pro = get_pro()


def main():
    idx = pro.ths_index(exchange="A", type="N")
    print(f"概念总数: {len(idx)}")

    member_rows = []
    counts = {}
    for i, row in enumerate(idx.itertuples()):
        code = row.ts_code
        for attempt in range(3):
            try:
                m = pro.ths_member(ts_code=code)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [FAIL] {code} {row.name}: {e}")
                    m = None
                time.sleep(2)
        time.sleep(0.15)
        n = 0 if m is None else len(m)
        counts[code] = n
        if m is not None and len(m):
            m = m.assign(concept_code=code, concept_name=row.name)
            member_rows.append(m)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(idx)}")

    members = pd.concat(member_rows, ignore_index=True)
    members["snapshot_date"] = datetime.now().strftime("%Y%m%d")

    idx["member_count"] = idx["ts_code"].map(counts).fillna(0).astype(int)
    idx["is_noise_kw"] = idx["name"].apply(
        lambda n: any(k in str(n) for k in CONCEPT_NOISE_KEYWORDS))
    idx["is_theme"] = (~idx["is_noise_kw"]) & (idx["member_count"] <= MAX_MEMBER_COUNT) & (idx["member_count"] > 0)

    save("theme.concepts", idx)
    cols = ["concept_code", "concept_name", "con_code", "con_name", "snapshot_date"]
    save("theme.members", members[cols])

    print(f"\n概念列表 {len(idx)} 个, 其中题材 {idx['is_theme'].sum()} 个")
    print(f"成分明细 {len(members)} 行, 覆盖股票 {members['con_code'].nunique()} 只")
    print(f"题材概念成分数分布: {idx[idx['is_theme']]['member_count'].describe().round(1).to_dict()}")


if __name__ == "__main__":
    main()
