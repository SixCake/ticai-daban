# -*- coding: utf-8 -*-
"""盘前/盘中预警雷达: 全量成分股扫描(20s) → 题材热度自算排名 + 东财板块对照
 → 个股涨停概率启发式v0排名

题材热度(自算口径, 与归属同宇宙): 成分股聚合
  heat = top10均涨 + 0.5*涨>5%家数 + 0.8*涨>7%家数 + 1.5*头部涨速(3min)
         + 0.6*放量均量比 + 1.2*涨停家数
个股涨停概率v0(启发式, 日志校准见research 05):
  z = -6 + 0.55*涨幅 + 0.5*min(涨速3,3) + 0.35*min(量比,5) + 0.08*min(题材heat,15)
      - 0.30*距涨停%   ;  prob = sigmoid(z)
涨速由扫描序列差分(1/3/5min); 量比/涨停价直接取腾讯f49/f47。

输出 data/live/radar.json; 校准日志 data/live/radar_log_YYYYMMDD.jsonl(每分钟, 涨幅≥3%)
"""
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from build.attribute import load_con2stock, load_maps  # noqa: E402
from tx_quote import fetch_quotes  # noqa: E402

LIVE = DATA / "live"
LIVE.mkdir(exist_ok=True)
INTERVAL = 20
EM_PATH = ("/api/qt/clist/get?pn=1&pz=100&po=1&np=1"
           "&fltt=2&invt=2&fid=f3&fs=m:90+t:3"
           "&fields=f12,f14,f3,f22,f104,f105,f128")
EM_HOSTS = ["https://push2.eastmoney.com", "https://push2delay.eastmoney.com"]


def is_trading_now(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H%M")
    return "0925" <= hm <= "1130" or "1300" <= hm <= "1500"


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


class Radar:
    def __init__(self):
        self.con2stock = load_con2stock()
        _, _, self.cname = load_maps()
        self.stock2con = defaultdict(list)
        for k, cs in self.con2stock.items():
            for c in cs:
                self.stock2con[c].append(k)
        self.codes = sorted({c for cs in self.con2stock.values() for c in cs
                             if not c.endswith(".BJ")})
        self.hist: dict = {}          # code -> deque[(t, pct)]
        self.cycle = 0
        self.em_cache: list = []
        self.em_ts = 0.0
        self.em_skip = 0              # 东财失败退避(剩余跳过cycle数)
        self.bad_sweep = 0            # 腾讯扫描完整性退避计数
        self._log_prob: dict = {}     # 上轮日志概率, 供dp(概率变化)计算
        self._heat_hist: dict = {}    # 题材热度历史, 供dheat(热度趋势)计算
        print(f"雷达初始化: {len(self.con2stock)}概念 {len(self.codes)}成分股")

    def sweep(self) -> dict:
        t0 = time.time()
        batches = [self.codes[i:i + 60] for i in range(0, len(self.codes), 60)]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(fetch_quotes, batches))
        quotes = {}
        for r in res:
            quotes.update(r)
        t = time.time()
        for c, q in quotes.items():
            if "ST" in q["name"]:
                continue
            h = self.hist.setdefault(c, deque(maxlen=64))
            h.append((t, q["pct"]))
        return quotes

    def speed_of(self, c: str, secs: int, t: float) -> float:
        h = self.hist.get(c)
        if not h:
            return 0.0
        now_pct = h[-1][1]
        target = t - secs
        for ts, pct in h:           # deque按时间升序, 取首个≤target的样本
            if ts >= target:
                return round(now_pct - pct, 2)
        return round(now_pct - h[0][1], 2)

    def theme_dheat(self, k: str, secs: int, t: float) -> float:
        """题材热度secs窗口变化(正=升温, 负=退潮)"""
        h = self._heat_hist.get(k)
        if not h:
            return 0.0
        now_v = h[-1][1]
        target = t - secs
        for ts, v in h:
            if ts >= target:
                return round(now_v - v, 2)
        return round(now_v - h[0][1], 2)

    def theme_heat(self, quotes: dict, t: float) -> list[dict]:
        # heat v2(尺寸中性): 头部超额+密度项, 消除大成分筐尺寸偏差
        # 基线=同尺寸随机抽样的top10期望值(次序统计量 rank k1..k2)
        mkt = sorted((qq["pct"] for qq in quotes.values()
                      if qq["limit_px"] > 0), reverse=True)
        N = len(mkt)
        rows = []
        for k, members in self.con2stock.items():
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
            s3 = self._speed3_top(members, quotes, t)
            vrs = [q["vr"] for q in qs if q["pct"] >= 3]
            vr = min(sum(vrs) / len(vrs), 8) if vrs else 0.0
            dens5, dens7 = n5 / n, n7 / n
            zdens = zt / n
            heat = (headx + 30 * dens5 + 50 * dens7 + 1.5 * s3
                    + 0.6 * min(vr, 4) + 90 * zdens)
            rows.append({"concept_code": k, "name": self.cname.get(k, k),
                         "heat": round(heat, 2), "top10": round(top10, 2),
                         "headx": round(headx, 2),
                         "n5": n5, "n7": n7, "s3": round(s3, 2),
                         "vr": round(vr, 2), "zt": zt, "nmem": n,
                         "dens5": round(dens5, 3)})
        rows.sort(key=lambda x: -x["heat"])
        return rows

    def _speed3_top(self, members, quotes, t) -> float:
        tops = sorted((c for c in members if c in quotes),
                      key=lambda c: -quotes[c]["pct"])[:20]
        if not tops:
            return 0.0
        return sum(self.speed_of(c, 180, t) for c in tops) / len(tops)

    def stock_prob(self, quotes: dict, heat_by: dict, t: float) -> list[dict]:
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
            s1 = self.speed_of(c, 60, t)
            s3 = self.speed_of(c, 180, t)
            s5 = self.speed_of(c, 300, t)
            cons = self.stock2con.get(c, [])
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
                         "theme": self.cname.get(hk, "-") if hk else "-",
                         "near": near})
        rows.sort(key=lambda x: -x["prob"])
        return rows

    def once(self) -> float:
        """执行一轮扫描并写出radar.json, 返回耗时(秒)"""
        now = datetime.now()
        t = time.time()
        quotes = self.sweep()
        # 限流保护: 腾讯扫描完整性<90% → 指数退避(20s→40s→80s)
        if len(quotes) < 0.9 * len(self.codes):
            self.bad_sweep += 1
            print(f"[{now:%H:%M:%S}] 扫描不完整 {len(quotes)}/{len(self.codes)}, 退避")
        else:
            self.bad_sweep = 0
        themes = self.theme_heat(quotes, t)
        heat_by = {r["concept_code"]: r["heat"] for r in themes}
        rank_by = {r["concept_code"]: i + 1 for i, r in enumerate(themes)}
        for r in themes:            # 题材热度历史, 供dheat趋势因子
            hh = self._heat_hist.setdefault(r["concept_code"],
                                            deque(maxlen=48))
            hh.append((t, r["heat"]))
        # 限流保护: 东财每分钟一次, 失败退避5分钟, 期间用缓存
        if self.em_skip > 0:
            self.em_skip -= 1
            em = self.em_cache
        elif self.cycle % 3 == 0:
            em = fetch_em_boards()
            if em:
                self.em_cache, self.em_ts = em, time.time()
            elif time.time() - self.em_ts < 300:
                em = self.em_cache
                self.em_skip = 3
            else:
                self.em_skip = 15
        else:
            em = self.em_cache
        em_by = {b["name"]: b for b in em}
        our_names = {r["name"] for r in themes}
        external = [b for b in em if b["name"] not in our_names
                    and not b["name"].startswith("昨日")][:15]
        for r in themes:
            b = em_by.get(r["name"]) or next(
                (b for b in em
                 if len(b["name"]) >= 3 and
                 (b["name"] in r["name"] or r["name"] in b["name"])), None)
            r["em"] = {"pct": b["pct"], "speed": b["speed"],
                       "leader": b["leader"]} if b else None
        stocks_all = self.stock_prob(quotes, heat_by, t)
        prob_by = {s["ts_code"]: s for s in stocks_all}
        # 快照题材附带领涨成分股top5(供看板点击展开)
        for r in themes[:40]:
            mem = sorted((c for c in self.con2stock[r["concept_code"]]
                          if c in quotes and "ST" not in quotes[c]["name"]
                          and quotes[c]["limit_px"] > 0),
                         key=lambda c: -quotes[c]["pct"])
            tops = []
            for c in mem[:5]:
                q = quotes[c]
                s = prob_by.get(c)
                tops.append({"name": q["name"],
                             "pct": round(q["pct"], 2),
                             "prob": s["prob"] if s else None,
                             "near": bool(s and s["near"])})
            r["top"] = tops
        near_cnt = sum(1 for s in stocks_all if s["near"])
        n_hot = sum(1 for r in themes if r["heat"] >= 12)   # v2刻度≈p95
        self.cycle += 1
        snap = {"ts": now.strftime("%H:%M:%S"), "trading": True,
                "interval": INTERVAL, "themes": themes[:40],
                "external": external, "near_cnt": near_cnt,
                "n_hot": n_hot,
                "stocks": [s for s in stocks_all
                           if not s["near"] and s["prob"] >= 0.05][:80]}
        (LIVE / "radar.json").write_text(json.dumps(snap, ensure_ascii=False),
                                         encoding="utf-8")
        # 校准日志: 每cycle(20s)一次, 涨幅≥3%或概率≥0.2全量(含负例),
        # 研究05发现1分钟粒度丢失ramp轨迹(赢家首条≥4%日志已在+9%),
        # 20s全量保留真实起涨轨迹供研究06; dp=较上轮(20s前)概率变化
        with open(LIVE / f"radar_log_{now:%Y%m%d}.jsonl", "a",
                  encoding="utf-8") as f:
            for c, q in quotes.items():
                if "ST" in q["name"] or c.endswith(".BJ"):
                    continue
                s = prob_by.get(c)
                prob = s["prob"] if s else 0.0
                if q["limit_px"] <= 0 or (q["pct"] < 3 and prob < 0.2):
                    continue
                f.write(json.dumps({
                    "t": now.strftime("%H%M%S"), "code": c,
                    "name": q["name"], "pct": round(q["pct"], 2),
                    "s1": s["s1"] if s else 0.0,
                    "s3": s["s3"] if s else 0.0,
                    "s5": s["s5"] if s else 0.0,
                    "vr": q["vr"], "tover": q["tover"],
                    "dist": s["dist"] if s else 0.0,
                    "prob": prob,
                    "dp": round(prob - self._log_prob.get(c, prob), 3),
                    "heat": s["heat"] if s else 0.0,
                    "trank": rank_by.get(s["hk"], 99) if s else 99,
                    "dheat": self.theme_dheat(s["hk"], 300, t) if s else 0.0,
                    "theme": s["theme"] if s else "-",
                    "near": bool(s and s["near"])},
                    ensure_ascii=False) + "\n")
        self._log_prob = {c: s["prob"] for c, s in prob_by.items()}
        top = snap["stocks"]
        print(f"[{now:%H:%M:%S}] 雷达 扫描{len(quotes)} "
              f"热题:{[r['name'] for r in themes[:3]]} "
              f"概率TOP:{[(s['name'], s['prob']) for s in top[:3]]} "
              f"耗时{time.time() - t:.1f}s")
        return time.time() - t

    def run(self):
        while True:
            if not is_trading_now(datetime.now()):
                time.sleep(120)
                continue
            elapsed = self.once()
            time.sleep(max(1.0, INTERVAL * min(4, 2 ** self.bad_sweep)
                           - elapsed))


if __name__ == "__main__":
    Radar().run()
