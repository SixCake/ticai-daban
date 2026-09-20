# -*- coding: utf-8 -*-
"""研究55: 龙头拐头·跷跷板 开盘缺失段(09:30-11:00) PIT-safe 回放补全

背景: 某交易日 radar 进程 11:00 才启动 → seesaw 监测只覆盖 11:00-11:30,
开盘 09:30-11:00(龙头拐头最高发的开盘波动窗口)完全缺失。本脚本用东财
1分钟K线(与 collect/fetch_zt_minute.py 同源: push2his/push2delay, 当日
全天可得)逐分钟回放 core/seesaw.SeesawTracker, 补全缺失段。

无前视(PIT)设计 —— 每个 seesaw 输入在分钟 T 只用 ≤T 的数据:
  pct/price     ← 东财分钟 bar close(T), 绝不读未来分钟
  hist/s3(D1)   ← 分钟累积 deque + window_diff(180s), 天然 trailing-only
  amount(D4)    ← 东财分钟额 cumsum 差分, 只用 ≤T 累计额
  heat_rows     ← theme_heat(横截面_T): 每分钟从当日横截面重算涨停
                  (price≥limit_px*0.995), 不用收盘涨停表(否则=前视)
  口径A昨龙头    ← theme.day T-1, 上一交易日数据开盘前已知
  con2stock归属  ← 沿用 live 所用 kpl 延续近似, 无 T+1 直标
  vr量比        ← 东财分钟 vol(手)cumsum/elapsed_min vs avg5vol(手,历史缓存)

保真度损失(须标注, 回放≠实时逐tick):
  1. 粒度 1min vs 实时 20-40s: 触发时点归整到分钟边界; max_pct 用分钟
     close 采样(1/min vs 实时 2-3/min), D2 回落判定可能略偏
  2. 行情源: live 11:00-11:30 来自 tx 轮询, 回放 09:31-11:00 来自东财K线
  3. 触发不可逐tick复现: 个别事件与实时会有出入(任何回放固有属性)
  → 事件打 src:"replay" 标签, 落 seesaw_replay_YYYYMMDD.jsonl(独立文件),
    供研究25/27 与实时事件区分, 不污染 PIT 统计。

用法:
  python research/55_seesaw_open_replay.py --fetch    # 抓全宇宙当日分钟线(缓存)
  python research/55_seesaw_open_replay.py --replay   # 回放(需先 --fetch)
  python research/55_seesaw_open_replay.py            # fetch 后 replay
  python research/55_seesaw_open_replay.py --date 20260914
"""
import argparse
import json
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.heat import theme_heat  # noqa: E402
from core.seesaw import SeesawTracker  # noqa: E402
from datastore import load  # noqa: E402
from quotes.qmt import _limit_ratio, _valid  # noqa: E402

LIVE = ROOT / "data" / "live"
META = ROOT / "data" / "meta"
EM_HOSTS = ["push2delay.eastmoney.com", "push2his.eastmoney.com",
            "92.push2his.eastmoney.com"]
_host_idx = 0


# ---------- 东财当日分钟线抓取(全宇宙, 并行, 缓存) ----------

def _secid(ts: str) -> str:
    c, e = ts.split(".")
    return ("1." if e == "SH" else "0.") + c


def fetch_minute_one(ts: str, key: str, retries: int = 2):
    """东财1分钟K线, 只取当日 09:31~11:30 段; 返回 [(HHMM,o,h,l,c,vol,amt)]
    vol 单位手、amt 单位元(与东财原始一致)。失败/无当日数据返回 None。"""
    global _host_idx
    params = {"secid": _secid(ts), "klt": "1", "fqt": "0", "lmt": "300",
              "end": "20500101", "fields1": "f1,f2,f3",
              "fields2": "f51,f52,f53,f54,f55,f56,f57"}
    base = _host_idx
    for attempt in range(retries + 1):
        for k in range(len(EM_HOSTS)):
            host = EM_HOSTS[(base + k) % len(EM_HOSTS)]
            try:
                r = requests.get(f"https://{host}/api/qt/stock/kline/get",
                                 params=params, timeout=12,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    continue
                d = (r.json().get("data") or {}).get("klines")
                if d:
                    _host_idx = (base + k) % len(EM_HOSTS)
                    rows = []
                    for b in d:
                        if not b.startswith(key):
                            continue
                        hhmm = b[11:16].replace(":", "")
                        if hhmm < "0931" or hhmm > "1130":
                            continue
                        tm, o, c, h, l, v, a = b.split(",")
                        rows.append([hhmm, float(o), float(h), float(l),
                                     float(c), float(v), float(a)])
                    return rows or None
                return None          # 200但空(停牌/无当日数据)
            except Exception:
                continue
        time.sleep(0.8 + attempt + random.random() * 0.4)
    return None


def build_minute_universe(date: str, codes: list, workers: int = 8) -> dict:
    """并行抓取全宇宙当日分钟线, 增量缓存到 data/live/minute_universe_{date}.json"""
    cache_p = LIVE / f"minute_universe_{date}.json"
    cache = {}
    if cache_p.exists():
        try:
            c = json.loads(cache_p.read_text(encoding="utf-8"))
            if c.get("date") == date:
                cache = c.get("data", {})
        except Exception:
            cache = {}
    todo = [c for c in codes if c not in cache]
    print(f"分钟线抓取: 全宇宙{len(codes)}只, 已缓存{len(cache)}, 待抓{len(todo)}")
    key = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_minute_one, c, key): c for c in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            try:
                rows = fu.result()
            except Exception:
                rows = None
            if rows:
                cache[c] = rows
                ok += 1
            else:
                fail += 1
            if n % 200 == 0:
                print(f"  {n}/{len(todo)} ok={ok} fail={fail} "
                      f"{time.time()-t0:.0f}s", flush=True)
                cache_p.write_text(json.dumps({"date": date, "data": cache},
                                              ensure_ascii=False),
                                   encoding="utf-8")
    cache_p.write_text(json.dumps({"date": date, "data": cache},
                                  ensure_ascii=False), encoding="utf-8")
    print(f"分钟线抓取完成: ok={ok} fail={fail} 缓存{len(cache)}只 "
          f"{time.time()-t0:.0f}s → {cache_p}")
    return cache


# ---------- 昨收(权威口径) ----------

def build_prev_close(date: str, codes: list) -> dict:
    """昨收(权威): tushare daily(上一交易日)全市场一次 + daily_panel
    最后非NaN close 兼底(停牌复牌股, tushare当日无记录)。
    权威口径与行情源 tick lastClose 一致。
    取代纯daily_panel口径: 后者对部分活跃股近日 close 为NaN,
    groupby.last()会取陈旧非NaN值(实测五洲医疗 301234 近日NaN→~33旧值
    →今日~100→伪+202%); tushare上一交易日直接给出真实收盘99.6。"""
    cache_p = LIVE / f"prevclose_{date}.json"
    if cache_p.exists():
        try:
            c = json.loads(cache_p.read_text(encoding="utf-8"))
            if c.get("date") == date and c.get("data"):
                print(f"昨收缓存命中: {len(c['data'])}只")
                return c["data"]
        except Exception:
            pass
    dp = load("market.daily_panel",
              columns=["trade_date", "ts_code", "close"])
    prev = max(d for d in dp["trade_date"].unique() if d < date)
    import config
    import tushare as ts
    pro = ts.pro_api(config.get_token())
    d = pro.daily(trade_date=prev)
    pc = {c: float(v) for c, v in zip(d["ts_code"], d["close"])
          if v == v and v > 0}
    print(f"昨收: tushare {prev} 全市场{len(pc)}只")
    # 兼底: 停牌复牌股(tushare上一交易日无记录) → daily_panel最后非NaN close
    miss = [c for c in codes if c not in pc]
    if miss:
        dpp = dp[dp["trade_date"] < date].sort_values("trade_date")
        fb = dpp.groupby("ts_code")["close"].last().to_dict()
        n_fb = 0
        for c in miss:
            v = fb.get(c)
            if v is not None and v == v and v > 0:   # 非NaN且>0
                pc[c] = float(v)
                n_fb += 1
        print(f"昨收兼底: {len(miss)}只停牌股缺上一交易日, "
              f"daily_panel回填{n_fb}只")
    cache_p.write_text(json.dumps({"date": date, "data": pc},
                                  ensure_ascii=False), encoding="utf-8")
    print(f"昨收完成: {len(pc)}只 → {cache_p}")
    return pc


def _epoch(date: str, hhmm: str) -> float:
    return datetime.strptime(date + hhmm, "%Y%m%d%H%M").timestamp()


def _elapsed_min(hhmm: str) -> float:
    h, m = int(hhmm[:2]), int(hhmm[2:4])
    hm = h * 60 + m
    return max(1.0, min(hm, 11 * 60 + 30) - (9 * 60 + 30))


# ---------- 回放引擎 ----------

def replay(date: str, gap_end: str = "1100") -> Path:
    c2s = load_con2stock()
    _, _, cname = load_maps()
    minute = json.loads(
        (LIVE / f"minute_universe_{date}.json").read_text(encoding="utf-8"))
    if minute.get("date") != date:
        raise SystemExit(f"分钟缓存非当日: {minute.get('date')} != {date}")
    bars = minute["data"]
    pc_p = LIVE / f"prevclose_{date}.json"
    pc = json.loads(pc_p.read_text(encoding="utf-8")).get("data", {}) \
        if pc_p.exists() else {}
    if not pc:
        raise SystemExit(f"昨收缓存缺失, 先跑 --fetch: {pc_p}")
    names = json.loads((META / "qmt_names.json").read_text(
        encoding="utf-8")).get("data", {})
    avg5 = json.loads((META / "qmt_avg5vol.json").read_text(
        encoding="utf-8")).get("data", {})

    # 每股按分钟索引 + 当日开盘价
    idx = {}
    day_open = {}
    skipped = 0
    for c, rows in bars.items():
        pre = pc.get(c, 0)
        if not rows or pre <= 0:
            skipped += 1
            continue
        idx[c] = {r[0]: r for r in rows}
        day_open[c] = rows[0][1] if rows else 0.0
    codes = sorted(idx)
    print(f"回放宇宙: {len(codes)}只(有当日分钟线+有效昨收), "
          f"剔除{skipped}只昨收缺失")

    # 分钟轴(全宇宙并集, 09:31~11:30; 触发限 <gap_end, 结局可延展)
    all_hm = sorted({hm for c in codes for hm in idx[c]})
    all_hm = [hm for hm in all_hm if "0931" <= hm <= "1130"]
    print(f"分钟轴: {all_hm[0]}~{all_hm[-1]} 共{len(all_hm)}分钟")

    tracker = SeesawTracker(c2s, cname, date, LIVE)
    # 清空 __init__ 从 live 文件回载的状态, 只保留 con2stock/cname/昨龙头口径A
    tracker.log_path = LIVE / f"seesaw_replay_{date}.jsonl"
    tracker.events, tracker.cooldown = [], {}
    tracker.lead_state, tracker.amt_hist, tracker.heat_hist = {}, {}, {}
    tracker.con_series, tracker.con_day, tracker._zj_codes = {}, {}, set()

    hist: dict = {}                 # code -> deque[(t,pct)] (trailing)
    cum: dict = {c: [0.0, 0.0] for c in codes}   # code -> [cum_vol手, cum_amt元]
    n_trig = 0
    for hm in all_hm:
        t = _epoch(date, hm)
        ts_str = f"{hm[:2]}:{hm[2:4]}:00"
        emin = _elapsed_min(hm)
        quotes = {}
        for c in codes:
            r = idx[c].get(hm)
            if r is None:
                continue
            _, o, h, l, cl, v, a = r
            cum[c][0] += v
            cum[c][1] += a
            pre = pc[c]
            nm = names.get(c, "")
            a5 = avg5.get(c, 0.0)
            vr = (cum[c][0] / emin) / (a5 / 240) if emin > 0 and a5 > 0 else 0.0
            quotes[c] = {
                "name": nm, "price": cl, "open": day_open[c],
                "volume": cum[c][0] * 100, "pct": (cl - pre) / pre * 100,
                "amount": cum[c][1], "float_mv": 0.0, "vr": round(vr, 3),
                "limit_px": round(pre * (1 + _limit_ratio(c, nm)), 2),
                "tover": 0.0}
            hist.setdefault(c, deque(maxlen=64)).append((t, quotes[c]["pct"]))
        heat_rows = theme_heat(c2s, cname, quotes, hist, t)
        before = len(tracker.events)
        tracker.update(quotes, hist, heat_rows, t, ts_str)
        n_trig += len(tracker.events) - before

    # 只保留缺口段触发(te < gap_end), 打 src 标签, 重写独立文件
    cutoff = _epoch(date, gap_end)
    out_p = tracker.log_path
    with open(out_p, "w", encoding="utf-8") as f:
        n_ev = n_out = 0
        for ev in tracker.events:
            if ev["te"] >= cutoff:
                continue
            trig = {k: v for k, v in ev.items() if k != "outcomes"}
            trig["kind"] = "trigger"
            trig["src"] = "replay"
            f.write(json.dumps(trig, ensure_ascii=False) + "\n")
            n_ev += 1
            for m, o in (ev.get("outcomes") or {}).items():
                rec = {"kind": "outcome", "src": "replay", "date": date,
                       "te": ev["te"], "concept_code": ev["concept_code"],
                       "leader_code": ev["leader_code"], "m": int(m), **o}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"回放完成: 缺口段(<{gap_end})触发{n_ev}条 结局{n_out}条 "
          f"(全程检测触发{n_trig}) → {out_p}")
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gap-end", default="1100")
    args = ap.parse_args()

    c2s = load_con2stock()
    codes = sorted({c for cs in c2s.values() for c in cs if _valid(c)})
    if args.fetch or not args.replay:
        build_minute_universe(args.date, codes, workers=args.workers)
        build_prev_close(args.date, codes)
    if args.replay or not args.fetch:
        replay(args.date, gap_end=args.gap_end)


if __name__ == "__main__":
    main()
