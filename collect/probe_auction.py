# -*- coding: utf-8 -*-
"""竞价时段 tick 原始字段勘探（纯采集，不做任何判定）

目的: 搞清 QMT 在 09:15~09:25 集合竞价期间到底给什么字段, 为后续定义
「竞价抢筹」因子提供数据能力依据。此前从未验证过——`quotes/qmt._row()`
只取 lastPrice/lastClose/volume/amount/open 五个字段, 且雷达 09:25 才开始
扫描(radar_log 最早记录 09:26:44), 竞价阶段零采样。

A股竞价机制(要验的正是这个):
  09:15~09:20  可撤单阶段   ← 申报量可以是假的(诱多后撤单)
  09:20~09:25  不可撤单阶段 ← 申报量才真实
  09:25:00     集中撮合, 产生开盘价与成交量
撮合前 tick 的 volume 是累计申报量还是 0? 9:20 分界是否可见?
9:24:50~9:25:00 末段是否有逐秒抢筹跳变? —— 本脚本回答这三个问题。

采集: 默认 09:14:30~09:25:20 每 5 秒一轮, 落
  data/live/auction_probe_YYYYMMDD.jsonl, 每行一个采样轮:
    {"t":"09:20:03", "phase":"不可撤单", "n_total":5210, "n_active":3120,
     "vol_sum":..., "amt_sum":...,
     "series": {code: [price, vol, amt]},   # 仅 vol>0 的票, 供分段量比
     "raw":    {code: {完整tick全字段}}}    # 固定样本票, 供字段勘探

独立进程: 不接雷达主循环, 不改 core/early_signal.py。与雷达各自持有
独立 QMT 会话(同为裸订阅, 口径一致)——但两会话共用同一 account_id,
故采集窗口刻意在 09:25:20 结束(详见下方会话冲突风险注释)。

用法:
  python collect/probe_auction.py                 # 等到 09:14:30 自动开采
  python collect/probe_auction.py --now           # 立即采一轮(测管道用)
  python collect/probe_auction.py --interval 3 --end 09:26:00
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, QUOTE_SOURCE  # noqa: E402
from quotes import qmt  # noqa: E402

LIVE = DATA / "live"
LIVE.mkdir(exist_ok=True)
START = "09:14:30"        # 早于 09:15 开盘竞价, 留出推送预热
END = "09:25:20"          # 略过 09:25:00 撮合即退 —— 避开与雷达抢订阅(见下)
INTERVAL = 5              # 秒
RAW_N = 40                # 每轮 dump 完整字段的样本票数(覆盖各板块)
WARMUP = 90               # 推送订阅预热等待上限(秒)

# 会话冲突风险: subscribe_whole_quote 按 account_id 排队
# (request_queue=bigqmt:rpc:queue:{account_id}), 本脚本与雷达是
# **两个进程共用同一账号**。雷达 09:25:00 起扫描并订阅, 故本脚本
# 必须在 09:25:20 前退出, 把重叠窗口压到 ~20s。若明日雷达日志出现
# 订阅失败/超时, 说明两会话确实冲突 —— 退路是把竞价采样折进
# apps/radar.py 作盘前阶段(单会话), 而不是继续双进程。


def phase_of(hms: str) -> str:
    """竞价阶段标注(要验的正是这个分界是否可见)"""
    if hms < "09:15:00":
        return "竞价前"
    if hms < "09:20:00":
        return "可撤单"
    if hms < "09:25:00":
        return "不可撤单"
    if hms < "09:30:00":
        return "撮合后"
    return "盘中"


def wait_until(hms: str) -> None:
    """阻塞到指定时刻(已过则立即返回)"""
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        if now >= hms:
            return
        time.sleep(1)


def pick_raw_codes(ticks: dict) -> list:
    """选样本票做全字段 dump: 各板块均衡覆盖 + 当前有量的优先"""
    boards = {"60": [], "00": [], "30": [], "68": [], "8": [], "4": []}
    for c, t in ticks.items():
        pre = c[:2] if c[:2] in ("60", "00", "30", "68") else c[:1]
        key = pre if pre in boards else "4"
        boards[key].append((c, float(t.get("volume") or 0)))
    out = []
    per = max(1, RAW_N // max(1, sum(1 for v in boards.values() if v)))
    for k, lst in boards.items():
        if not lst:
            continue
        lst.sort(key=lambda x: -x[1])       # 有量的优先(字段更可能齐全)
        out.extend(c for c, _ in lst[:per])
    return out[:RAW_N]


def sample(ticks: dict, raw_codes: list) -> dict:
    """一轮采样 → 可 JSON 序列化的 dict"""
    now = datetime.now()
    hms = now.strftime("%H:%M:%S")
    series, vol_sum, amt_sum = {}, 0.0, 0.0
    for c, t in ticks.items():
        try:
            vol = float(t.get("volume") or 0)
            amt = float(t.get("amount") or 0)
            px = float(t.get("lastPrice") or 0)
        except Exception:
            continue
        if vol > 0 or amt > 0:
            series[c] = [round(px, 3), vol, round(amt, 1)]
            vol_sum += vol
            amt_sum += amt
    raw = {}
    for c in raw_codes:
        t = ticks.get(c)
        if t:
            raw[c] = {k: v for k, v in t.items()
                      if isinstance(v, (int, float, str, bool)) or v is None}
    return {"t": hms, "phase": phase_of(hms),
            "n_total": len(ticks), "n_active": len(series),
            "vol_sum": round(vol_sum, 1), "amt_sum": round(amt_sum, 1),
            "push_age": round(time.time() - qmt._push["ts"], 1),
            "series": series, "raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true",
                    help="立即采一轮并退出(测管道, 不等竞价窗口)")
    ap.add_argument("--interval", type=int, default=INTERVAL)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()

    if QUOTE_SOURCE != "qmt":
        print(f"QUOTE_SOURCE={QUOTE_SOURCE}, 竞价勘探需要 qmt 源"
              f"(腾讯接口不给竞价分段申报量)")
        return 1

    date = datetime.now().strftime("%Y%m%d")
    out = LIVE / f"auction_probe_{date}.jsonl"

    print(f"[probe] 启动竞价 tick 推送订阅(与雷达各自独立会话)")
    qmt._ensure_push()

    # 预热: 等推送有数据, 上限 WARMUP 秒
    t0 = time.time()
    while not qmt._push["ticks"] and time.time() - t0 < WARMUP:
        time.sleep(2)
    n = len(qmt._push["ticks"])
    print(f"[probe] 推送预热 {'成功' if n else '超时'}: {n} 只 tick "
          f"耗时 {time.time() - t0:.0f}s")
    if not n:
        print("[probe] 无 tick 数据, 退出(不写空文件)")
        return 1

    raw_codes = pick_raw_codes(qmt._push["ticks"])
    print(f"[probe] 全字段样本票 {len(raw_codes)} 只: {raw_codes[:8]}...")

    if a.now:
        s = sample(qmt._push["ticks"], raw_codes)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[probe] --now 采样一轮 → {out}")
        if s["raw"]:
            k0 = next(iter(s["raw"].values()))
            print(f"[probe] tick 字段全集({len(k0)}): {sorted(k0.keys())}")
            print(f"[probe] 样本 tick: {json.dumps(k0, ensure_ascii=False)}")
        else:
            print("[probe] raw 为空(无样本票 tick)")
        print(f"[probe] 有量票数 {s['n_active']}/{s['n_total']} "
              f"phase={s['phase']}")
        return 0

    if not a.start <= datetime.now().strftime("%H:%M:%S"):
        print(f"[probe] 等待至 {a.start}")
        wait_until(a.start)

    print(f"[probe] 开始采集 {a.start}~{a.end} 每 {a.interval}s → {out}")
    keys_seen: set = set()
    n_round = 0
    first_active: dict = {}
    with open(out, "a", encoding="utf-8") as f:
        while datetime.now().strftime("%H:%M:%S") <= a.end:
            ticks = qmt._push["ticks"]
            s = sample(ticks, raw_codes)
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            f.flush()
            n_round += 1
            for v in s["raw"].values():
                keys_seen.update(v.keys())
            for c in s["series"]:
                first_active.setdefault(c, s["t"])
            print(f"[probe] {s['t']} {s['phase']} tick{s['n_total']} "
                  f"有量{s['n_active']} vol_sum{s['vol_sum']:.0f} "
                  f"推送龄{s['push_age']}s")
            time.sleep(a.interval)

    print(f"\n[probe] 采集完成 {n_round} 轮 → {out}")
    print(f"[probe] tick 字段全集({len(keys_seen)}): {sorted(keys_seen)}")
    if first_active:
        earliest = sorted(first_active.items(), key=lambda x: x[1])[:5]
        print(f"[probe] 最早出现量的票: {earliest}")
        print(f"[probe] 有量票数 {len(first_active)}")
    print("\n[probe] 三个待答问题(人工看 jsonl):")
    print("  1. 竞价期间 volume 是累计申报量还是 0? → 看 09:15~09:25 的 vol_sum")
    print("  2. 09:20 不可撤单分界是否可见? → 对比 09:19:55 与 09:20:05 的量")
    print("  3. 09:24:50~09:25:00 末段是否有抢筹跳变? → 看末段逐轮 vol_sum")
    return 0


if __name__ == "__main__":
    sys.exit(main())
