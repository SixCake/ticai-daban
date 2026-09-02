# -*- coding: utf-8 -*-
"""修复污染分时数据并固化: 腾讯m1历史分钟K × server清洗逻辑。

背景: qmt 推送断连期间横截面返回陈旧快照价, 雷达多次重启/并行导致
intraday_px 与 presig px_hist 混入异基准平行价格线(与真实价差
0.6%~4.7%, 双相位时刻交错), 分时图锯齿成块。server 端已做当日
腾讯校验(_merge_with_tx), 但腾讯当日分时接口无历史数据——当日一过,
历史日走本地清洗会残留漏网污染(间隙<1.2%的平行线)。

本脚本用腾讯 m1 分钟K(800根≈最近4个交易日)作历史真值, 一次性:
  1. 清洗并固化 intraday_px_DATE.json (复用 server._merge_with_tx,
     本地高频点保留 + 污染分钟换腾讯点 + 缺失分钟腾讯补);
  2. 剔除 presig_state_DATE.json 各信号 px_hist 中的污染点
     (与所在分钟腾讯收盘价偏差>1%, 集合竞价段与开盘价差>10%)。

用法: python research/repair_ipx.py [YYYYMMDD ...]  # 默认最近4个交易日
"""
import json
import shutil
import sys
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from apps.server import _merge_with_tx  # noqa: E402

LIVE = DATA / "live"
T_LO, T_HI = "09:15:00", "15:00:59"
M1_URL = "https://ifzq.gtimg.cn/appstock/app/kline/mkline?param={sym},m1,,800"


def _sym(code: str) -> str:
    return ("sh" if code.endswith(".SH") else
            "bj" if code.endswith(".BJ") else "sz") + code[:6]


def fetch_m1(code: str) -> dict:
    """{date: [[HH:MM:SS, close, cumvol, cumamt], ...]} 分钟累计口径。"""
    try:
        req = urllib.request.Request(
            M1_URL.format(sym=_sym(code)),
            headers={"User-Agent": "Mozilla/5.0"})
        j = json.loads(urllib.request.urlopen(req, timeout=6).read())
        rows = j["data"][_sym(code)]["m1"]
    except Exception:
        return {}
    by_date: dict[str, list] = defaultdict(list)
    cum_v = cum_a = 0.0
    for r in rows:
        d, close, vol = r[0][:8], float(r[2]), float(r[5])
        if vol < 0:
            vol = 0.0
        cum_v += vol
        cum_a += vol * close * 100     # 额≈量×价(手→股), 供量柱比例参考
        by_date[d].append([f"{r[0][8:10]}:{r[0][10:12]}:00", close,
                           cum_v, cum_a])
    return by_date


def _fmt_t(t: str) -> str:
    t = t.strip()
    return f"{t[:2]}:{t[2:4]}:{t[4:6]}" if len(t) == 6 and ":" not in t else t


def main():
    dates = sys.argv[1:] or [
        (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(4)]
    dates = sorted(d for d in dates
                   if (LIVE / f"intraday_px_{d}.json").exists()
                   or (LIVE / f"presig_state_{d}.json").exists())
    if not dates:
        print("无可修复日期")
        return

    # 需拉 m1 的票: 各日期 intraday_px ∪ presig px_hist 有数据的票
    need: set[str] = set()
    presig: dict[str, dict] = {}       # date -> presig json(已载入)
    ipx: dict[str, dict] = {}          # date -> intraday_px json(已载入)
    for d in dates:
        f_i = LIVE / f"intraday_px_{d}.json"
        ipx[d] = (json.loads(f_i.read_text(encoding="utf-8"))
                  if f_i.exists() else {})
        need |= {c for c, ps in ipx[d].items() if ps}
        f_p = LIVE / f"presig_state_{d}.json"
        presig[d] = (json.loads(f_p.read_text(encoding="utf-8"))
                     if f_p.exists() else {})
        for s in presig[d].get("signals", []):
            if s.get("px_hist"):
                need.add(s["ts_code"])
    need = {c for c in need if not c.endswith(".BJ")}
    print(f"修复日期: {dates}\n待拉腾讯m1: {len(need)}只")

    m1: dict[str, dict] = {}
    lock = threading.Lock()
    done = [0]

    def work(c: str):
        r = fetch_m1(c)
        with lock:
            m1[c] = r
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  m1进度 {done[0]}/{len(need)}")

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, sorted(need)))
    ok = sum(1 for v in m1.values() if v)
    print(f"m1 拉取成功 {ok}/{len(need)}")

    for d in dates:
        print(f"===== {d} =====")
        # ---- 1. 清洗固化 intraday_px ----
        src = ipx[d]
        # local 合并: intraday_px + presig px_hist (同时刻vol>0优先)
        sig_pts: dict[str, list] = defaultdict(list)
        for s in presig[d].get("signals", []):
            sig_pts[s["ts_code"]].extend(s.get("px_hist") or [])
        fixed: dict = {}
        n_pt = n_kept = n_tx = 0
        for c in sorted(set(src) | set(sig_pts)):
            pts = src.get(c, [])
            m = {}
            for e in pts:
                if T_LO <= _fmt_t(e[0]) <= T_HI:
                    m[_fmt_t(e[0])] = [e[1], e[2] if len(e) > 2 else 0,
                                       e[3] if len(e) > 3 else 0]
            for e in sig_pts.get(c, []):
                t = _fmt_t(e[0])
                if not (T_LO <= t <= T_HI):
                    continue
                v = [e[1], e[2] if len(e) > 2 else 0,
                     e[3] if len(e) > 3 else 0]
                old = m.get(t)
                if old is None or (v[1] > 0 and old[1] <= 0):
                    m[t] = v
            local = [[t] + v for t, v in sorted(m.items())]
            tx = m1.get(c, {}).get(d, [])
            out = _merge_with_tx(local, tx) if tx else local
            n_pt += len(local)
            n_kept += sum(1 for p in out if not p[0].endswith(":00"))
            n_tx += sum(1 for p in out if p[0].endswith(":00"))
            fixed[c] = [[p[0].replace(":", ""), p[1], p[2], p[3]]
                        for p in out]
        if fixed:
            f_i = LIVE / f"intraday_px_{d}.json"
            bak = LIVE / f"intraday_px_{d}.pre_repair.json"
            if f_i.exists() and not bak.exists():   # 保护最初原始备份
                shutil.copy(f_i, bak)
            f_i.write_text(json.dumps(fixed), encoding="utf-8")
            print(f"  intraday_px: {len(fixed)}只 入参{n_pt}点 → "
                  f"保留高频{n_kept} + 腾讯分钟{n_tx} (备份 .pre_repair.json)")

        # ---- 2. 清洗 presig px_hist 污染点 ----
        n_rm_sig = n_rm_pt = 0
        for s in presig[d].get("signals", []):
            ph = s.get("px_hist") or []
            if not ph:
                continue
            tx = m1.get(s["ts_code"], {}).get(d, [])
            if not tx:
                continue
            tx_by_min = {p[0][:5]: p for p in tx}
            minutes = sorted(tx_by_min)
            min_idx = {mm: i for i, mm in enumerate(minutes)}
            open_ref = tx[0][1]
            keep = []
            removed = 0
            for p in ph:
                t = _fmt_t(p[0])
                if not (T_LO <= t <= T_HI):
                    continue               # 盘后点直接弃
                mm = t[:5]
                i = min_idx.get(mm)
                if i is None:              # 集合竞价段: 撮合后恒等于开盘价
                    if abs(p[1] - open_ref) / open_ref <= 0.005:
                        keep.append([t, p[1]] + list(p[2:]))
                    else:
                        removed += 1
                    continue
                refs = [tx_by_min[mm][1]]
                if i > 0:
                    refs.append(tx_by_min[minutes[i - 1]][1])
                if min(abs(p[1] - r) / r for r in refs) <= 0.010:
                    keep.append([t, p[1]] + list(p[2:]))
                else:
                    removed += 1
            if removed:
                s["px_hist"] = keep
                n_rm_sig += 1
                n_rm_pt += removed
        if n_rm_pt:
            f_p = LIVE / f"presig_state_{d}.json"
            bak = LIVE / f"presig_state_{d}.pre_repair.json"
            if not bak.exists():
                shutil.copy(f_p, bak)
            f_p.write_text(json.dumps(presig[d], ensure_ascii=False),
                           encoding="utf-8")
            print(f"  presig px_hist: {n_rm_sig}条信号剔除{n_rm_pt}个污染点"
                  f" (备份 .pre_repair.json)")
        else:
            print("  presig px_hist: 无需清洗")


if __name__ == "__main__":
    main()
