# -*- coding: utf-8 -*-
"""题材热度 v2（自算口径, 尺寸中性）— 唯一出处

heat = headx + 30×dens5 + 50×dens7 + 1.5×头部20只3分钟涨速
       + 0.6×放量均量比(≥3%成员) + 90×zdens

headx = top10均涨 − 同尺寸抽样基线（全市场涨幅降序中 rank
  N/(n+1)..10N/(n+1) 段均值, N=全市场 n=题材成分数, 次序统计量期望）
dens5/dens7/zdens = 涨>5%/>7%/涨停家数占成分数比例

由v0升级背景: v0原始家数项随成分数线性放大, 600-1400成分大筐霸榜、
真同动小簇被埋；v2用密度+头部超额消除尺寸偏差（实测新能源汽车由
榜1降至86名）。热点阈值 HOT_THRESHOLD≈全市场p95。
"""
from core.momentum import window_diff

HOT_THRESHOLD = 12   # v2刻度≈p95


def _speed3_top(members: list, quotes: dict, hist: dict, t: float) -> float:
    """题材头部20只(按当前涨幅)的3分钟平均涨速"""
    tops = sorted((c for c in members if c in quotes),
                  key=lambda c: -quotes[c]["pct"])[:20]
    if not tops:
        return 0.0
    return sum(window_diff(hist.get(c), 180, t) for c in tops) / len(tops)


def theme_heat(con2stock: dict, cname: dict, quotes: dict,
               hist: dict, t: float) -> list[dict]:
    """全题材热度排名（降序）。quotes: {code: 行情dict}（需含pct/vr/
    price/limit_px/name）；hist: {code: (ts,pct)时序}供涨速。剔除ST与
    无涨停价新股；成分有效样本<5的题材不参与排名。"""
    # 基线=同尺寸随机抽样的top10期望值(次序统计量 rank k1..k2)
    mkt = sorted((qq["pct"] for qq in quotes.values()
                  if qq["limit_px"] > 0), reverse=True)
    N = len(mkt)
    rows = []
    for k, members in con2stock.items():
        qs = [quotes[c] for c in members
              if c in quotes and "ST" not in quotes[c]["name"]
              and quotes[c]["limit_px"] > 0]
        n = len(qs)
        if n < 5:
            continue
        qs.sort(key=lambda x: -x["pct"])
        top10 = sum(q["pct"] for q in qs[:10]) / min(10, n)
        k1 = min(N - 1, N // (n + 1))
        k2 = min(N, max(k1 + 1, 10 * N // (n + 1)))
        base = sum(mkt[k1:k2]) / (k2 - k1)
        headx = top10 - base
        n5 = sum(1 for q in qs if q["pct"] >= 5)
        n7 = sum(1 for q in qs if q["pct"] >= 7)
        zt = sum(1 for q in qs
                 if q["limit_px"] > 0 and q["price"] >= q["limit_px"] * 0.995)
        s3 = _speed3_top(members, quotes, hist, t)
        vrs = [q["vr"] for q in qs if q["pct"] >= 3]
        vr = min(sum(vrs) / len(vrs), 8) if vrs else 0.0
        dens5, dens7 = n5 / n, n7 / n
        zdens = zt / n
        heat = (headx + 30 * dens5 + 50 * dens7 + 1.5 * s3
                + 0.6 * min(vr, 4) + 90 * zdens)
        rows.append({"concept_code": k, "name": cname.get(k, k),
                     "heat": round(heat, 2), "top10": round(top10, 2),
                     "headx": round(headx, 2),
                     "n5": n5, "n7": n7, "s3": round(s3, 2),
                     "vr": round(vr, 2), "zt": zt, "nmem": n,
                     "dens5": round(dens5, 3)})
    rows.sort(key=lambda x: -x["heat"])
    return rows
