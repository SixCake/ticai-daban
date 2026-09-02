# -*- coding: utf-8 -*-
"""采集申万2021行业分类(一级/二级)成分映射

产物: data/meta/sw_map.json  {ts_code: {l1, l2, l1_code, l2_code}}
源:
  1. tushare index_classify(L1/L2 指数树) + 逐 L2 index_member(index_code=)
     取 is_new=='Y' 当前成分(全市场覆盖主路径)
  2. index_member_all 补缺(硬上限3000, 自带 l1/l2 名)
覆盖约92%(缺失=综合/问题股/B股, 下游视为"未分类"排除出申万聚合)

注: index_member 参数是 index_code 不是 ts_code; index_member_all 单独用
    仅覆盖~2891只(3000行上限), 故以逐L2拉取为主、all补缺。

用法: python collect/fetch_sw.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA, get_pro  # noqa: E402

pro = get_pro()
META = DATA / "meta"


def _retry(fn, **kw):
    """3次重试退避; 全失败返回None"""
    for att in range(3):
        try:
            return fn(**kw)
        except Exception as e:
            if att == 2:
                print(f"  [FAIL] {fn.__name__} {kw}: {e}")
                return None
            time.sleep(1.5 * (att + 1))
    return None


def main():
    l1 = _retry(pro.index_classify, level="L1", src="SW2021")
    l2 = _retry(pro.index_classify, level="L2", src="SW2021")
    if l1 is None or l2 is None or not len(l1) or not len(l2):
        print("申万指数树拉取失败, 保留旧 sw_map(不覆盖)")
        return
    # L1: industry_code -> industry_name (L2.parent_code 指向 L1.industry_code)
    l1map = dict(zip(l1["industry_code"], l1["industry_name"]))
    sw: dict = {}
    for i, r in enumerate(l2.itertuples()):
        m = _retry(pro.index_member, index_code=r.index_code)
        if m is not None and len(m):
            l1name = l1map.get(r.parent_code, "")
            for c in m[m["is_new"] == "Y"]["con_code"]:
                sw[c] = {"l1": l1name, "l2": r.industry_name,
                         "l1_code": r.parent_code, "l2_code": r.index_code}
        time.sleep(0.12)
        if (i + 1) % 30 == 0:
            print(f"  L2进度 {i + 1}/{len(l2)} 累计{len(sw)}只")
    # index_member_all 补缺(上限3000, 自带l1/l2名; 仅填逐L2未覆盖票)
    allm = _retry(pro.index_member_all)
    add = 0
    if allm is not None and len(allm):
        for r in allm.itertuples():
            if r.ts_code not in sw:
                sw[r.ts_code] = {"l1": r.l1_name, "l2": r.l2_name,
                                 "l1_code": "", "l2_code": ""}
                add += 1
    META.mkdir(exist_ok=True)
    (META / "sw_map.json").write_text(
        json.dumps(sw, ensure_ascii=False), encoding="utf-8")
    n_l1 = len({v["l1"] for v in sw.values() if v["l1"]})
    n_l2 = len({v["l2"] for v in sw.values() if v["l2"]})
    print(f"申万映射落盘 {len(sw)}只 (index_member_all补缺{add}) "
          f"L1={n_l1} L2={n_l2} → {META / 'sw_map.json'}")


if __name__ == "__main__":
    main()
