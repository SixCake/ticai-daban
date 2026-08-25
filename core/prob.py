# -*- coding: utf-8 -*-
"""涨停概率 v0（启发式）— 唯一出处

z = -6 + 0.55×涨幅 + 0.5×min(max(涨速3m,0),3) + 0.35×min(量比,5)
      + 0.08×min(题材heat,15) − 0.30×距涨停%   ;  prob = sigmoid(z)
临近涨停(价格≥涨停价×0.995)时 z = max(z, 4.0)

研究05裁决(docs/research_05_涨停概率校准复盘.md): 排名AUC .92-.95成立、
10cm≥90%桶校准诚实；20cm过自信、中段二元化；tover/trank为遗漏的强反向
因子。v1(分板型+dist主导+封板意愿)冻结, 待20s日志OOS验证后替换本模块。
"""
import math

from core.momentum import window_diff


def stock_prob(quotes: dict, heat_by: dict, stock2con: dict,
               cname: dict, hist: dict, t: float) -> list[dict]:
    """未板股涨停概率排名（降序）。仅排 涨幅≥2% 且非ST/北交所 的未板股；
    已板/触板(near)者保留标记供看板单列。heat_by: {concept_code: heat}；
    hist: {code: (ts,pct)时序}供1/3/5分钟涨速。"""
    rows = []
    for c, q in quotes.items():
        if "ST" in q["name"] or c.endswith(".BJ"):
            continue
        lp = q["limit_px"]
        if lp <= 0 or q["pct"] < 2:
            continue
        if q["price"] >= lp * 0.995:      # 已板/触板: 概率记1档单列
            near = True
        else:
            near = False
        dist = (lp - q["price"]) / q["price"] * 100
        s1 = window_diff(hist.get(c), 60, t)
        s3 = window_diff(hist.get(c), 180, t)
        s5 = window_diff(hist.get(c), 300, t)
        cons = stock2con.get(c, [])
        hk = max(cons, key=lambda k: heat_by.get(k, -1e9), default=None)
        heat = heat_by.get(hk, 0.0)
        z = (-6.0 + 0.55 * q["pct"] + 0.5 * min(max(s3, 0), 3)
             + 0.35 * min(q["vr"], 5) + 0.08 * min(heat, 15)
             - 0.30 * dist)
        if near:
            z = max(z, 4.0)
        prob = 1 / (1 + math.exp(-z))
        rows.append({"ts_code": c, "name": q["name"],
                     "prob": round(prob, 3), "pct": round(q["pct"], 2),
                     "s1": s1, "s3": s3, "s5": s5,
                     "vr": round(q["vr"], 2), "dist": round(dist, 2),
                     "tover": round(q["tover"], 2),
                     "heat": round(heat, 1),
                     "hk": hk,
                     "theme": cname.get(hk, "-") if hk else "-",
                     "near": near})
    rows.sort(key=lambda x: -x["prob"])
    return rows
