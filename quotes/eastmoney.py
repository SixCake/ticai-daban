# -*- coding: utf-8 -*-
"""东财概念板块涨幅榜（雷达外部对照源）

接口: push2.eastmoney.com, 被封锁时自动切 push2delay 备源
字段: f12代码 f14名称 f3涨幅 f22涨速 f104涨家数 f105跌家数 f128领涨股
"""
import json
import urllib.request

EM_PATH = ("/api/qt/clist/get?pn=1&pz=100&po=1&np=1"
           "&fltt=2&invt=2&fid=f3&fs=m:90+t:3"
           "&fields=f12,f14,f3,f22,f104,f105,f128")
EM_HOSTS = ["https://push2.eastmoney.com", "https://push2delay.eastmoney.com"]


def fetch_em_boards() -> list[dict]:
    """东财概念板块涨幅榜(外部对照): name/pct/speed/up/down/leader; 主备双源"""
    for host in EM_HOSTS:
        try:
            req = urllib.request.Request(host + EM_PATH,
                                         headers={"User-Agent": "Mozilla/5.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=6).read())
            rows = [{"name": r["f14"], "pct": float(r["f3"]),
                     "speed": float(r["f22"]), "up": int(r["f104"]),
                     "down": int(r["f105"]), "leader": r.get("f128", "")}
                    for r in d["data"]["diff"]]
            if rows:
                return rows
        except Exception:
            continue
    return []
