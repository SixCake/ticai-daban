# -*- coding: utf-8 -*-
"""雷达轨迹标注: 分钟级概率日志 × 涨停结果 → 每stock-day轨迹汇总

结果标签源(优先级): events_enriched.parquet(收盘权威, 含首封/末封/炸板/连板)
 → data/live/latest.json盘中池(当日 provisional, daily_update后重跑覆盖)。

输出 data/live/radar_labeled_YYYYMMDD.jsonl, 每stock-day一行:
  轨迹: n/t_first/t_last/pct_max/pct_last/prob_max/t_prob_max/t_c30/t_c50/
        heat_max/theme
  结果: zt/height/first_time/last_time/open_times/src
  领先: lead30/lead50 = 首封时间 - 首次概率过30%/50% 的分钟数
        (正=雷达提前于封板, 负=雷达晚于封板)

用法:
  python label_radar.py            # 标注最新日志日
  python label_radar.py 20260825   # 指定日期
  python label_radar.py 20260825 000017.SZ   # 追溯单只分钟级轨迹
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

LIVE = DATA / "live"


def _sec(t) -> int:
    s = str(t).zfill(6)
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def _load_log(date: str) -> list[dict]:
    f = LIVE / f"radar_log_{date}.jsonl"
    if not f.exists():
        return []
    return [json.loads(x) for x in f.open(encoding="utf-8") if x.strip()]


def _outcomes(date: str) -> tuple[dict, str]:
    """返回 {code: {height,first_time,last_time,open_times}}, 标签源名"""
    ev = pd.read_parquet(DATA / "events_enriched.parquet",
                         columns=["trade_date", "ts_code", "limit_times",
                                  "open_times", "first_time", "last_time"])
    ev = ev[ev["trade_date"] == date]
    if len(ev):
        return {r.ts_code: {"height": int(r.limit_times),
                            "first_time": str(r.first_time).zfill(6),
                            "last_time": str(r.last_time).zfill(6),
                            "open_times": int(r.open_times)}
                for r in ev.itertuples()}, "events"
    lf = LIVE / "latest.json"
    if lf.exists():
        d = json.loads(lf.read_text(encoding="utf-8"))
        if d.get("date") == date:
            return {p["ts_code"]: {"height": int(p["height"]),
                                   "first_time": str(p["first_time"]).zfill(6),
                                   "last_time": str(p["last_time"]).zfill(6),
                                   "open_times": int(p["open_times"])}
                    for p in d.get("pool", [])}, "live"
    return {}, "none"


def label(date: str) -> list[dict]:
    recs = _load_log(date)
    if not recs:
        print(f"标注 {date}: 无雷达日志")
        return []
    oc, src = _outcomes(date)
    g: dict = {}
    for r in recs:
        c = r["code"]
        s = g.setdefault(c, {"code": c, "name": r.get("name", ""), "n": 0,
                             "t_first": r["t"], "pct_max": -99.0,
                             "prob_max": -1.0, "t_prob_max": None,
                             "t_c30": None, "t_c50": None,
                             "heat_max": 0.0, "trank_min": 99,
                             "dheat_c50": 0.0,
                             "theme": r.get("theme", "-")})
        s["n"] += 1
        s["t_last"] = r["t"]
        s["pct_last"] = r["pct"]
        if r.get("name") and not s["name"]:
            s["name"] = r["name"]
        if r.get("trank", 99) < s["trank_min"]:
            s["trank_min"] = r["trank"]
        if r["pct"] > s["pct_max"]:
            s["pct_max"] = r["pct"]
        if r["prob"] > s["prob_max"]:
            s["prob_max"] = r["prob"]
            s["t_prob_max"] = r["t"]
        if s["t_c30"] is None and r["prob"] >= 0.3:
            s["t_c30"] = r["t"]
        if s["t_c50"] is None and r["prob"] >= 0.5:
            s["t_c50"] = r["t"]
            s["dheat_c50"] = r.get("dheat", 0.0)
        if r.get("heat", 0) > s["heat_max"]:
            s["heat_max"] = r["heat"]
            s["theme"] = r.get("theme", s["theme"])
    rows = []
    for c, s in g.items():
        o = oc.get(c)
        zt = o is not None
        row = {**s, "pct_max": round(s["pct_max"], 2),
               "prob_max": round(s["prob_max"], 3),
               "heat_max": round(s["heat_max"], 1),
               "zt": zt, "src": src,
               "height": o["height"] if o else 0,
               "first_time": o["first_time"] if o else None,
               "last_time": o["last_time"] if o else None,
               "open_times": o["open_times"] if o else 0,
               "lead30": None, "lead50": None}
        if o:
            ft = _sec(o["first_time"])
            if s["t_c30"]:
                row["lead30"] = round((ft - _sec(s["t_c30"])) / 60, 1)
            if s["t_c50"]:
                row["lead50"] = round((ft - _sec(s["t_c50"])) / 60, 1)
        rows.append(row)
    rows.sort(key=lambda x: -x["prob_max"])
    out = LIVE / f"radar_labeled_{date}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    nz = sum(1 for x in rows if x["zt"])
    hi = [x for x in rows if x["prob_max"] >= 0.5]
    hit = sum(1 for x in hi if x["zt"])
    leads = [x["lead50"] for x in rows if x["lead50"] is not None
             and x["lead50"] >= 0]
    print(f"标注 {date} [{src}]: stock-day {len(rows)} 涨停{nz} "
          f"| prob≥0.5样本{len(hi)} 命中{hit} "
          f"| 雷达领先首封(lead50≥0) {len(leads)}只 "
          f"中位{sorted(leads)[len(leads)//2] if leads else '-'}min → {out}")
    return rows


def trace(date: str, code: str):
    recs = [r for r in _load_log(date) if r["code"] == code]
    if not recs:
        print(f"{date} 无 {code} 轨迹")
        return
    nm = next((r.get("name") for r in recs if r.get("name")), "")
    print(f"== {code} {nm} {date} 分钟轨迹 {len(recs)}点 ==")
    for r in recs:
        print(f"{r['t']} 涨{r['pct']:>6.2f} prob{r['prob']:.3f} "
              f"dp{float(r.get('dp', 0)):+.3f} s3{float(r.get('s3', 0)):+.2f} "
              f"vr{r['vr']:>5.2f} 距板{r['dist']:>5.2f} "
              f"热{float(r.get('heat', 0)):>5.1f} {r.get('theme', '-')} "
              f"{'[触板]' if r['near'] else ''}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        trace(sys.argv[1], sys.argv[2])
    else:
        date = sys.argv[1] if len(sys.argv) > 1 else None
        if not date:
            logs = sorted(LIVE.glob("radar_log_*.jsonl"))
            date = logs[-1].stem.split("_")[-1] if logs else None
        if date:
            label(date)
