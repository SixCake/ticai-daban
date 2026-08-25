# -*- coding: utf-8 -*-
"""盘前/盘中预警雷达: 全量成分股扫描(20s) → 题材热度自算排名 + 东财板块对照
 → 个股涨停概率v0排名

题材热度与概率公式见 core/heat.py 与 core/prob.py（领域逻辑唯一出处）；
本文件只负责: 扫描调度、时序状态维护、限流退避、快照与校准日志落盘。

输出 data/live/radar.json; 校准日志 data/live/radar_log_YYYYMMDD.jsonl
(每cycle 20s, 涨幅≥3%或概率≥0.2全量含负例)
"""
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.calendar import is_trading_hours  # noqa: E402
from core.heat import HOT_THRESHOLD, theme_heat  # noqa: E402
from core.momentum import window_diff  # noqa: E402
from core.prob import stock_prob  # noqa: E402
from quotes.eastmoney import fetch_em_boards  # noqa: E402
from quotes.tx import fetch_quotes  # noqa: E402

LIVE = DATA / "live"
LIVE.mkdir(exist_ok=True)
INTERVAL = 20


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
        themes = theme_heat(self.con2stock, self.cname, quotes, self.hist, t)
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
        stocks_all = stock_prob(quotes, heat_by, self.stock2con, self.cname,
                                self.hist, t)
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
        n_hot = sum(1 for r in themes if r["heat"] >= HOT_THRESHOLD)
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
                    "dheat": window_diff(self._heat_hist.get(s["hk"]), 300, t) if s else 0.0,
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
            if not is_trading_hours(datetime.now()):
                time.sleep(120)
                continue
            elapsed = self.once()
            time.sleep(max(1.0, INTERVAL * min(4, 2 ** self.bad_sweep)
                           - elapsed))


if __name__ == "__main__":
    Radar().run()
