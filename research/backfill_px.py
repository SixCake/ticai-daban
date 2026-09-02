# -*- coding: utf-8 -*-
"""回补当日分时价格轨迹: radar_log(20s全量pct) × 昨收 → intraday_px_DATE.json

背景: 雷达多次重启导致信号 px_hist 缺失上午段; 但 radar_log 从 09:25 起
完整记录了所有 ≥1% 票的 20s pct 轨迹, 可无损还原上午价格曲线。
产物:
  data/live/intraday_px_DATE.json  {code: [[HH:MM:SS, price], ...]}
  并把早于 px_hist 的点前插回 presig_state 的信号时间线
昨收来源: QMT 1d 横截面(最后一根非今日bar), 失败票用 limit_px/涨幅上限 反推
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

LIVE = DATA / "live"
DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")

# ---------- 1. radar_log 轨迹 ----------
log_f = LIVE / f"radar_log_{DATE}.jsonl"
traj = {}
with open(log_f, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        traj.setdefault(r["code"], []).append(
            [r["t"], r["pct"], r.get("vol", 0), r.get("amt", 0)])
for c in traj:
    traj[c].sort(key=lambda x: x[0])
print(f"日志轨迹: {len(traj)}只")

# ---------- 2. 昨收 ----------
BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()
sys.path.insert(0, str(BIGQMT_SRC))
os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
from bigqmt_signal_trader.xtquant_compat import configure, xtdata  # noqa: E402
configure(redis_config={"formula_server": {"failure_cooldown_seconds": 5}})

codes = [c for c in traj if c.endswith((".SH", ".SZ"))
         and c[:2] in ("60", "68", "00", "30")]
pre_map = {}
for i in range(0, len(codes), 400):
    batch = codes[i:i + 400]
    try:
        res = xtdata.get_market_data_ex(
            field_list=["close"], stock_list=batch, period="1d",
            end_time=DATE, count=2, dividend_type="none", chunk_size=0,
            timeout_seconds=30)
        for c, df in (res or {}).items():
            try:
                rows = [(str(ix)[:8], float(v))
                        for ix, v in zip(df.index, df["close"])]
                prev = [v for d, v in rows if d != DATE]
                if prev and prev[-1] > 0:
                    pre_map[c] = prev[-1]
            except Exception:
                continue
    except Exception as e:
        print(f"1d批次失败 {i}: {e}")
print(f"昨收覆盖: {len(pre_map)}/{len(codes)}")

# ---------- 2b. 昨收(真实口径): presig信号 price0/(1+pct/100) 反推 ----------
# radar 的 pct 基于 tick lastClose, 无滞后; QMT 1d 横截面可能滞后, 仅作兜底
sig_pre = {}
ps_f0 = LIVE / f"presig_state_{DATE}.json"
if ps_f0.exists():
    for s in json.loads(ps_f0.read_text(encoding="utf-8")).get("signals", []):
        c, p0, pc = s.get("ts_code"), s.get("price0"), s.get("pct")
        if c and p0 and pc is not None and (1 + pc / 100) > 0.5:
            sig_pre.setdefault(c, p0 / (1 + pc / 100))
print(f"信号反推昨收: {len(sig_pre)}只 (优先于1d横截面)")

# ---------- 3. 还原价格轨迹 ----------
intraday = {}
for c, pts in traj.items():
    pre = sig_pre.get(c) or pre_map.get(c)
    if not pre:
        continue
    intraday[c] = [[t, round(pre * (1 + p / 100), 3), v, a]
                   for t, p, v, a in pts]
out_f = LIVE / f"intraday_px_{DATE}.json"
out_f.write_text(json.dumps(intraday), encoding="utf-8")
print(f"intraday_px 写出: {len(intraday)}只 → {out_f.name} "
      f"({out_f.stat().st_size // 1024}KB)")

# ---------- 4. 前插回信号时间线 ----------
ps_f = LIVE / f"presig_state_{DATE}.json"
if ps_f.exists():
    ps = json.loads(ps_f.read_text(encoding="utf-8"))
    n_fix = 0
    for s in ps.get("signals", []):
        c = s.get("ts_code")
        if c not in intraday:
            continue
        px = s.get("px_hist", [])
        first_t = px[0][0] if px else "99:99:99"
        pre_pts = [e for e in intraday[c] if e[0] < first_t]
        if pre_pts:
            s["px_hist"] = pre_pts + px
            n_fix += 1
    ps_f.write_text(json.dumps(ps, ensure_ascii=False), encoding="utf-8")
    print(f"信号时间线前插: {n_fix}条")
