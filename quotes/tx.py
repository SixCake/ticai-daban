# -*- coding: utf-8 -*-
"""腾讯实时行情 qt.gtimg.cn (60只/批, GBK)

字段: f[1]名称 f[3]现价 f[4]昨收 f[32]涨幅% f[33]最高 f[34]最低
      f[36]成交量 f[37]成交额(万) f[38]换手% f[44]流通市值(亿)
      f[47]涨停价 f[49]量比
用途: 雷达全量扫描、盘中题材中军识别、龙头短板盘中收盘位置
量纲陷阱: f[36]成交量分板块——主板/创业板返手、科创板返股
      (实测额/价反推比值 1.00 vs 99.9, 与腾讯日K接口同一陷阱),
      归一为股后才能与 QMT 源 volume 同口径算竞价量比。
"""
import urllib.request

from core.codes import to_sym, to_ts_code


def fetch_quotes(codes: list[str]) -> dict:
    """{ts_code: {name, price, pct, amount(元), float_mv(元),
    vr(量比), limit_px(涨停价), tover(换手率%)}}; 失败批次跳过"""
    out = {}
    for i in range(0, len(codes), 60):
        batch = codes[i:i + 60]
        url = "http://qt.gtimg.cn/q=" + ",".join(to_sym(c) for c in batch)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=5).read().decode("gbk", "ignore")
        except Exception:
            continue
        for line in raw.strip().split(";"):
            line = line.strip()
            if "=" not in line:
                continue
            sym, val = line.split("=", 1)
            f = val.strip('"').split("~")
            if len(f) < 50 or not f[3]:
                continue
            try:
                code = to_ts_code(sym.replace("v_", ""))
                v36 = float(f[36]) if f[36] else 0.0
                out[code] = {
                    "name": f[1], "price": float(f[3]),
                    "open": float(f[5]) if f[5] else 0.0,
                    "high": float(f[33]) if f[33] else 0.0,
                    "low": float(f[34]) if f[34] else 0.0,
                    "pct": float(f[32]), "amount": float(f[37]) * 1e4,
                    # 科创板返股, 其余板块返手 → 统一归为股
                    "volume": v36 if code.startswith("68") else v36 * 100,
                    "float_mv": float(f[44]) * 1e8,
                    "vr": float(f[49]) if f[49] else 0.0,
                    "limit_px": float(f[47]) if f[47] else 0.0,
                    "tover": float(f[38]) if f[38] else 0.0}
            except (ValueError, IndexError):
                continue
    return out
