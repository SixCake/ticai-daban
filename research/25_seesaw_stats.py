# -*- coding: utf-8 -*-
"""研究25: 龙头拐头→板块跟跌+跷跷板 事件统计（下跌定义选优）

数据源: data/live/seesaw_YYYYMMDD.jsonl (core/seesaw.py 盘中积累)
  kind=trigger 触发快照 | kind=outcome 结局回填(m=5/10/20分钟)

统计内容:
  1. 各下跌定义(及组合)的事件数 / 跟跌命中率(+10min概念均跌恶化) / 平均跟跌幅度
  2. 跷跷板对手概念持续性(+10/+20min热度增量仍为正的比例)
  3. 按市场环境分段(强制方法论): 牛市/熊市/震荡市独立统计

市场环境划分(市场级): 全A等权日收益20日均值 ma20
  ma20 > +0.10% → 牛市; ma20 < -0.10% → 熊市; 其余 → 震荡市
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load  # noqa: E402

LIVE = Path(__file__).resolve().parent.parent / "data" / "live"
DEFS = ["D1", "D2", "D3", "D4"]
FOLLOW_EPS = 0.3       # 均跌恶化超过0.3个点算"跟跌成立"
MIN_N = 30             # 样本不足提示阈值


def load_events():
    """合并trigger与outcome: (concept, leader, te)为主键"""
    trig, outs = [], {}
    for f in sorted(LIVE.glob("seesaw_*.jsonl")):
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "trigger":
                trig.append(d)
            elif d.get("kind") == "outcome":
                outs.setdefault((d["concept_code"], d["leader_code"],
                                 d["te"]), {})[str(d["m"])] = d
    rows = []
    for t in trig:
        o = outs.get((t["concept_code"], t["leader_code"], t["te"]), {})
        rows.append({**t, "o5": o.get("5"), "o10": o.get("10"),
                     "o20": o.get("20")})
    return rows


def market_env():
    """{trade_date: 牛/熊/震荡} 全A等权20日趋势"""
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    day = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    ma20 = day.rolling(20).mean()
    env = {}
    for d, v in ma20.items():
        if pd.isna(v):
            continue
        env[str(d)] = "牛市" if v > 0.10 else ("熊市" if v < -0.10
                                               else "震荡市")
    return env


def follow_delta(t, key="o10"):
    """+10min概念均跌变化(负=跟跌)"""
    o, m = t.get(key), t.get("members")
    if not o or not o.get("members") or not m:
        return None
    return o["members"]["avg_pct"] - m["avg_pct"]


def opp_persist(t, key="o10"):
    """对手概念热度仍为正的比例"""
    o = t.get(key)
    if not o or not o.get("opp"):
        return None, 0
    ds = [x["dheat"] for x in o["opp"] if x.get("dheat") is not None]
    if not ds:
        return None, 0
    return sum(1 for d in ds if d > 0) / len(ds), len(ds)


def sect_persist(t, key="o10"):
    """板块级跷跷板: 对手板块均涨幅在结局时点仍高于触发时点的比例"""
    o = t.get(key)
    if not o:
        return None
    t0 = {x["concept_code"]: x.get("avg_pct") for x in t.get("opp", [])}
    ds = [x["avg_pct"] - t0[x["concept_code"]] for x in o.get("opp", [])
          if x.get("avg_pct") is not None
          and t0.get(x["concept_code"]) is not None]
    if not ds:
        return None
    return sum(1 for d in ds if d > 0) / len(ds)


def report(rows, title):
    print(f"\n===== {title} (事件n={len(rows)}) =====")
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["fd10"] = df.apply(lambda t: follow_delta(t), axis=1)
    df["fd20"] = df.apply(lambda t: follow_delta(t, "o20"), axis=1)
    p10 = [opp_persist(t) for t in rows]
    p20 = [opp_persist(t, "o20") for t in rows]
    df["op10"] = [p[0] for p in p10]
    df["op20"] = [p[0] for p in p20]
    df["sp10"] = [sect_persist(t) for t in rows]
    df["sp20"] = [sect_persist(t, "o20") for t in rows]

    groups = []
    for r in range(1, len(DEFS) + 1):
        groups += ["".join(c) for c in combinations(DEFS, r)]
    out = []
    for g in groups:
        sub = df[df["defs"].apply(lambda s: set(g) <= set(s))]
        n = len(sub)
        if n == 0:
            continue
        hit = (sub["fd10"] < -FOLLOW_EPS).mean()
        out.append({"定义": g, "n": n,
                    "跟跌命中%": round(100 * hit, 1),
                    "跟跌幅度%": round(sub["fd10"].mean(), 2),
                    "20m跟跌%": round(sub["fd20"].mean(), 2),
                    "板块跷跷板10m%": round(100 * sub["sp10"].mean(), 1),
                    "板块跷跷板20m%": round(100 * sub["sp20"].mean(), 1),
                    "个股对手10m%": round(100 * sub["op10"].mean(), 1)})
    print(pd.DataFrame(out).to_string(index=False))
    if len(df) < MIN_N:
        print(f"  ⚠ 样本{len(df)}<{MIN_N}, 继续积累数据, 不作选优结论")


def main():
    rows = load_events()
    print(f"累计事件 {len(rows)} 条")
    if not rows:
        print("暂无数据: 盘中监测启动后运行本脚本")
        return
    env = market_env()
    for t in rows:
        t["env"] = env.get(t["date"], "震荡市")
    n_done = sum(1 for t in rows if t.get("o10"))
    print(f"已回填+10m结局 {n_done}/{len(rows)}")
    report(rows, "全样本")
    for e in ["牛市", "熊市", "震荡市"]:
        report([t for t in rows if t["env"] == e], e)


if __name__ == "__main__":
    main()
