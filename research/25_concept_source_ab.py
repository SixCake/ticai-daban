# -*- coding: utf-8 -*-
"""研究25: 概念数据源 A/B —— 东财概念板块 vs 同花顺概念(现用)

回答: 盘中概念关联用哪个数据源更好?
维度:
  1. 规模与覆盖: 概念数/成分数/涨停股覆盖率/每股平均候选数
  2. 归属质量: 近30个交易日用两源分别跑 attribute_day,
     对比未归属率/成簇题材数/最大簇占比(大筐风险)/独苗率
  3. 案例检验: 星网锐捷(算力网络缺失案)/锦龙股份(券商概念缺失案)
产物:
  data/theme/static/em_boards.parquet   东财板块快照(供后续研究复用)
  data/theme/static/em_members.parquet  东财成分快照
"""
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_NOISE_KEYWORDS, MAX_MEMBER_COUNT  # noqa: E402
from core.attribute import attribute_day, load_maps  # noqa: E402
from datastore import save  # noqa: E402

EM_HOSTS = ["https://push2.eastmoney.com", "https://push2delay.eastmoney.com"]
UA = {"User-Agent": "Mozilla/5.0"}
# 东财独有的行情/人气/属性类伪概念(非叙事题材); 两源共有名单的过滤交给
# CONCEPT_NOISE_KEYWORDS, 保证两源过滤口径对等
EM_NOISE_KEYWORDS = [
    "昨日", "今日", "最近", "历史新高", "热股", "题材股", "百元股",
    "标准普尔", "富时罗素", "茅指数", "宁组合", "转债标的",
]


def em_get(path: str) -> dict:
    for host in EM_HOSTS:
        try:
            req = urllib.request.Request(host + path, headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=8).read())
        except Exception:
            continue
    return {}


def to_ts(code: str) -> str | None:
    """东财6位代码 → tushare ts_code"""
    if code.startswith("6"):
        return code + ".SH"
    if code.startswith(("0", "3")):
        return code + ".SZ"
    if code.startswith(("4", "8", "9")):
        return code + ".BJ"
    return None


def fetch_em_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """板块列表(分页拉全) + 全量成分"""
    rows_b, pn = [], 1
    while True:
        d = em_get(f"/api/qt/clist/get?pn={pn}&pz=100&po=1&np=1&fltt=2&invt=2"
                   "&fid=f12&fs=m:90+t:3&fields=f12,f14")
        diff = (d.get("data") or {}).get("diff") or []
        if not diff:
            break
        rows_b.extend({"code": r["f12"], "name": r["f14"]} for r in diff)
        total = (d.get("data") or {}).get("total", 0)
        if pn * 100 >= total:
            break
        pn += 1
    boards = pd.DataFrame(rows_b)
    print(f"东财概念板块 {len(boards)} 个, 逐一拉成分…")
    rows = []
    for i, b in enumerate(boards.itertuples()):
        d2 = em_get(f"/api/qt/clist/get?pn=1&pz=5000&po=1&np=1&fltt=2&invt=2"
                    f"&fid=f12&fs=b:{b.code}&fields=f12")
        diff = (d2.get("data") or {}).get("diff") or []
        for r in diff:
            c = to_ts(str(r["f12"]))
            if c:
                rows.append({"concept_code": b.code,
                             "concept_name": b.name, "con_code": c})
        if (i + 1) % 100 == 0:
            print(f"  进度 {i + 1}/{len(boards)}")
        time.sleep(0.08)
    mem = pd.DataFrame(rows)
    boards["member_count"] = (mem.groupby("concept_code")["con_code"]
                              .nunique().reindex(boards["code"]).fillna(0)
                              .astype(int).values)
    return boards, mem


def build_maps(boards: pd.DataFrame, mem: pd.DataFrame):
    """按现用口径过滤后构建 (stock2con, msize, cname)"""
    boards = boards.assign(is_noise=boards["name"].apply(
        lambda n: any(k in str(n)
                      for k in CONCEPT_NOISE_KEYWORDS + EM_NOISE_KEYWORDS)))
    theme = boards[(~boards["is_noise"])
                   & (boards["member_count"] <= MAX_MEMBER_COUNT)
                   & (boards["member_count"] > 0)]
    codes = set(theme["code"])
    msize = theme.set_index("code")["member_count"].to_dict()
    cname = theme.set_index("code")["name"].to_dict()
    m2 = mem[mem["concept_code"].isin(codes)]
    stock2con = (m2.groupby("con_code")["concept_code"]
                 .apply(lambda s: sorted(set(s))).to_dict())
    return stock2con, msize, cname, theme


def attr_metrics(dates_codes: list, stock2con: dict, msize: dict) -> dict:
    """归属质量指标(与研究03同族): 未归属率/成簇数/最大簇占比/独苗率"""
    un, cl_sizes, n_days = 0, [], 0
    for codes in dates_codes:
        attr, _ = attribute_day(codes, stock2con, msize)
        vals = list(attr.values())
        un += sum(1 for k in vals if k == "UNASSIGNED")
        cnt = defaultdict(int)
        for k in vals:
            if k != "UNASSIGNED":
                cnt[k] += 1
        if cnt:
            sizes = sorted(cnt.values(), reverse=True)
            cl_sizes.append(sizes)
        n_days += 1
    total = sum(len(c) for c in dates_codes)
    clusters = [s for day in cl_sizes for s in day]
    multi = [s for s in clusters if s >= 2]
    return {
        "未归属率": round(un / total, 4),
        "日均成簇题材(≥2家)": round(len(multi) / n_days, 1),
        "日均独苗题材(1家)": round((len(clusters) - len(multi)) / n_days, 1),
        "平均簇规模": round(sum(clusters) / len(clusters), 2) if clusters else 0,
        "最大簇占比均值": round(sum(s[0] / sum(s) for s in cl_sizes) / n_days, 3),
    }


def main():
    # ---- A. 同花顺(现用) ----
    ths2con, ths_size, ths_name = load_maps()
    # ---- B. 东财 ----
    boards, mem = fetch_em_all()
    save("theme.em_boards", boards)
    save("theme.em_members", mem)
    em2con, em_size, em_name, em_theme = build_maps(boards, mem)

    ev = pd.read_parquet("data/limitup/1d/events_enriched.parquet")
    dates = sorted(set(ev["trade_date"]))[-30:]
    dates_codes = [ev[ev["trade_date"] == d]["ts_code"].tolist() for d in dates]
    universe = sorted({c for cs in dates_codes for c in cs})

    def coverage(s2c):
        hit = [c for c in universe if s2c.get(c)]
        cands = [len(s2c[c]) for c in hit]
        return {"涨停股覆盖率": round(len(hit) / len(universe), 4),
                "每股平均候选概念": round(sum(cands) / len(cands), 2),
                "候选≤2的稀疏票占比": round(
                    sum(1 for x in cands if x <= 2) / len(cands), 4)}

    print("\n===== 维度1: 规模与覆盖 =====")
    print(f"同花顺: 题材概念 {len(ths_size)} 个 | " +
          " | ".join(f"{k} {v}" for k, v in coverage(ths2con).items()))
    print(f"东  财: 题材概念 {len(em_size)} 个 | " +
          " | ".join(f"{k} {v}" for k, v in coverage(em2con).items()))

    print("\n===== 维度2: 归属质量(近30个交易日) =====")
    a, b = attr_metrics(dates_codes, ths2con, ths_size), \
        attr_metrics(dates_codes, em2con, em_size)
    print(f"{'指标':<16}{'同花顺':>10}{'东财':>10}")
    for k in a:
        print(f"{k:<16}{a[k]:>10}{b[k]:>10}")

    print("\n===== 维度3: 案例检验 =====")
    for code, tag in [("002396.SZ", "星网锐捷(算力网络案)"),
                      ("000712.SZ", "锦龙股份(券商案)")]:
        ths_cs = [ths_name.get(k, k) for k in ths2con.get(code, [])]
        em_cs = [em_name.get(k, k) for k in em2con.get(code, [])]
        print(f"{tag}")
        print(f"  同花顺候选({len(ths_cs)}): {', '.join(ths_cs[:15])}")
        print(f"  东财候选({len(em_cs)}): {', '.join(em_cs)}")
        for kw in ("算力", "券商", "证券"):
            if any(kw in n for n in em_cs):
                print(f"  → 东财含'{kw}'相关概念 ✓")


if __name__ == "__main__":
    main()
