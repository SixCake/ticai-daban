# -*- coding: utf-8 -*-
"""修复竞价时段行情源降级造成的假信号（2026-09-03 事故）

根因: QMT 推送订阅未就绪时 quotes/qmt.fetch_quotes 在 09:25~09:30 走日 bar
横截面降级, 而服务端日 bar 停在 8/27 → 全市场把"上一交易日的收盘价/涨幅/
涨停价/全天成交额"当成当日竞价行情:
  · 1917 条伪 S1（昨日涨幅 ≥1% 即触发, 昨日涨停被标成"一字板·不可捕捉"）
  · 155 条伪 S2 + 29 条伪 S3（陈旧→真实的价格断层被当成颠簸加速/高开）
  · 涨停价错配 → 封板事件流与形态分类全线误报
代码侧已修（qmt 竞价时段禁降级 + radar 陈旧闸 + 重启封口 _alerted）,
本脚本只负责把当日落盘数据洗干净。

修复内容:
  1) 自动定位陈旧窗口末端（雷达日志中 pct 全量冻结的最后一轮的下一轮）
  2) presig_state: 删陈旧窗口内信号 + 删轨迹仍被污染的 S2/S3（窗口后 10min,
     pathvol 窗口 600s）; 用真实昨收重算涨停价; 剥离陈旧 px_hist 前缀并补
     真实竞价首点; 重放封板事件流; 用真实竞价开盘价重算涨停形态
  3) open_traj / intraday_px / radar_log: 剥离陈旧前缀, 补真实竞价首点

用法: python research/repair_stale_auction.py 20260903 [--cut 09:31:07] [--dry]
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, get_pro  # noqa: E402
from core.early_signal import zt_shape_of  # noqa: E402
from quotes.tx import fetch_quotes as fetch_tx  # noqa: E402

LIVE = DATA / "live"
TOUCH_EPS, LOCK_EPS, LOCK_HOLD = 0.995, 0.9995, 60   # 与 apps/radar.py 同口径
POLLUTE_SEC = 600             # 陈旧样本滞留 pathvol 窗口(10min)内的污染时长
AUCTION = "09:25:00"          # 竞价首点时刻


def sec(hms: str) -> int:
    p = hms.replace(":", "")
    return int(p[:2]) * 3600 + int(p[2:4]) * 60 + int(p[4:6])


def epoch(date: str, hms: str) -> float:
    return datetime.strptime(date + hms.replace(":", ""),
                             "%Y%m%d%H%M%S").timestamp()


def bak(p: Path) -> Path:
    """备份原件; 已存在则不覆盖(修复可重跑, 首次备份才是真原件)"""
    b = p.with_name(p.stem + ".pre_repair" + p.suffix)
    if not b.exists():
        shutil.copy2(p, b)
    return b


def limit_ratio(code: str, name: str) -> float:
    if code.startswith(("30", "68")):
        return 0.20
    if "ST" in (name or "").upper() or "退" in (name or ""):
        return 0.05
    return 0.10


def detect_cut(date: str) -> str | None:
    """雷达日志里 pct 全量冻结(整轮快照一字不差)的最后一轮 → 下一轮即真实
    行情起点。返回 'HH:MM:SS'; 未发现冻结轮返回 None。"""
    rows: dict = {}
    f = LIVE / f"radar_log_{date}.jsonl"
    if not f.exists():
        return None
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        rows.setdefault(o["t"], {})[o["code"]] = o["pct"]
    ts = sorted(rows)
    last_frozen = None
    for a, b in zip(ts, ts[1:]):
        if rows[a] == rows[b]:
            last_frozen = b
        elif last_frozen:
            t = str(b)
            return f"{t[:2]}:{t[2:4]}:{t[4:6]}"
    return None


def prev_closes(date: str) -> tuple:
    """(上一交易日, {code: 该日收盘价}) — 该收盘价即当日昨收"""
    pro = get_pro()
    cal = pro.trade_cal(exchange="SSE", start_date="20250101",
                        end_date=date, is_open="1")
    days = sorted(d for d in cal["cal_date"] if d < date)
    prev = days[-1]
    df = pro.daily(trade_date=prev)
    return prev, dict(zip(df["ts_code"], df["close"]))


def open_prices(codes: list) -> dict:
    """当日真实开盘价(腾讯 f[5]); 竞价首点重建与形态分类用"""
    out: dict = {}
    batches = [codes[i:i + 60] for i in range(0, len(codes), 60)]
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(fetch_tx, batches):
            out.update({c: q.get("open", 0.0) for c, q in r.items()})
    return out


def replay_seal(s: dict, ph: list) -> None:
    """按修复后的价格时间线重放触板/封死/炸板事件流(与 radar 同判据)"""
    s["touch_t"], s["sealed_t"], s["zt_ev"], s["zb_cnt"] = None, None, [], 0
    lp = s.get("limit_px") or 0
    if lp <= 0 or not ph:
        return
    lock_since, ev = None, s["zt_ev"]
    for tt, px, *_ in ph:
        t = sec(tt)
        if px >= lp * TOUCH_EPS and not s["touch_t"]:
            s["touch_t"] = tt
        # 贴死涨停价只记首次时刻(等价 radar 的 setdefault), 否则保持时长永为0
        if px >= lp * LOCK_EPS:
            if lock_since is None:
                lock_since = t
        else:
            lock_since = None
        cur = bool(lock_since is not None and t - lock_since >= LOCK_HOLD)
        last = ev[-1][1] if ev else None
        if last is None and cur:
            ev.append([tt, 1])
            s["sealed_t"] = tt
        elif last == 1 and not cur:
            ev.append([tt, 0])
            s["zb_cnt"] += 1
        elif last == 0 and cur:
            ev.append([tt, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--cut", help="陈旧窗口末端 HH:MM:SS(缺省自动探测)")
    ap.add_argument("--dry", action="store_true", help="只报告不落盘")
    a = ap.parse_args()
    date, dry = a.date, a.dry

    cut = a.cut or detect_cut(date)
    if not cut:
        print("未探测到陈旧窗口, 请用 --cut 指定")
        return
    cut_s, cut_hms = sec(cut), cut.replace(":", "")
    pol_s = cut_s + POLLUTE_SEC
    auc_s, auc_ep = sec(AUCTION), epoch(date, AUCTION)
    print(f"陈旧窗口末端(真实行情起点) = {cut} · 轨迹污染窗至 "
          f"{pol_s // 3600:02d}:{pol_s % 3600 // 60:02d}")

    f_ps = LIVE / f"presig_state_{date}.json"
    st = json.loads(f_ps.read_text(encoding="utf-8"))
    sigs = st.get("signals", [])
    f_ipx = LIVE / f"intraday_px_{date}.json"
    ipx = json.loads(f_ipx.read_text(encoding="utf-8")) \
        if f_ipx.exists() else {}
    codes = sorted({s["ts_code"] for s in sigs} | set(ipx))
    prev, pre_close = prev_closes(date)
    opx = open_prices(codes)
    print(f"昨收基准 {prev} 覆盖{len(pre_close)}只 · 当日开盘价"
          f"覆盖{len(opx)}/{len(codes)}只")

    # ---------- 开盘轨迹: 剥陈旧前缀 + 补真实竞价首点 ----------
    f_ot = LIVE / f"open_traj_{date}.json"
    ot = json.loads(f_ot.read_text(encoding="utf-8")) if f_ot.exists() else {}
    ot2 = {}
    cut_ep = epoch(date, cut)
    for c, traj in ot.items():
        tr = [[float(ts), float(p)] for ts, p in traj if float(ts) >= cut_ep]
        pre, op = pre_close.get(c), opx.get(c, 0.0)
        if pre and op > 0 and (not tr or tr[0][0] > auc_ep):
            tr.insert(0, [auc_ep, round((op / pre - 1) * 100, 2)])
        if tr:
            ot2[c] = tr

    # ---------- 信号 ----------
    kept, d_stale, d_poll, n_fix_lp, n_shape = [], 0, 0, 0, 0
    for s in sigs:
        t_s = sec(s["t"])
        if t_s < cut_s:
            d_stale += 1
            continue
        if s["stage"] != "S1" and t_s < pol_s:
            d_poll += 1
            continue
        c, nm = s["ts_code"], s.get("name", "")
        pre, op = pre_close.get(c), opx.get(c, 0.0)
        if pre:
            lp = round(pre * (1 + limit_ratio(c, nm)), 2)
            if abs(lp - (s.get("limit_px") or 0)) > 0.001:
                n_fix_lp += 1
            s["limit_px"] = lp
        ph = [e for e in (s.pop("px_hist", []) or []) if sec(e[0]) >= cut_s]
        if pre and op > 0 and (not ph or sec(ph[0][0]) > auc_s):
            ph.insert(0, [AUCTION, op, 0, 0])
        replay_seal(s, ph)
        q = {"price": ph[-1][1] if ph else 0.0, "limit_px": s["limit_px"]}
        r = zt_shape_of(c, None, q, op, ot2.get(c))
        if r:
            s["zt_shape"], s["mode"] = r
            n_shape += 1
        else:
            s.pop("zt_shape", None)
            s.pop("mode", None)
        s["_px"] = ph
        kept.append(s)
    kept.sort(key=lambda s: (s["stage"] != "S2", s["stage"] != "S3",
                             -s["pct"]))
    print(f"信号 {len(sigs)} → {len(kept)} (删陈旧窗内{d_stale} "
          f"删轨迹污染S2/S3 {d_poll}) · 涨停价重算{n_fix_lp} "
          f"形态重算{n_shape}")

    # ---------- 分时 / 校准日志 ----------
    ipx2, n_ipx = {}, 0
    for c, tr in ipx.items():
        tr2 = [e for e in tr if str(e[0]) >= cut_hms]
        n_ipx += len(tr) - len(tr2)
        pre, op = pre_close.get(c), opx.get(c, 0.0)
        if pre and op > 0 and (not tr2 or str(tr2[0][0]) > "092500"):
            tr2.insert(0, ["092500", op, 0, 0])
        if tr2:
            ipx2[c] = tr2
    f_log = LIVE / f"radar_log_{date}.jsonl"
    log_lines = [ln for ln in f_log.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    log2 = [ln for ln in log_lines
            if json.loads(ln)["t"] >= cut_hms]
    print(f"分时剔陈旧点{n_ipx} · 校准日志 {len(log_lines)} → {len(log2)}行")

    if dry:
        print("[dry] 未落盘")
        return
    for s in kept:
        s["px_hist"] = s.pop("_px")
    names = [bak(f).name for f in (f_ps, f_ot, f_ipx, f_log) if f.exists()]
    f_ps.write_text(json.dumps({"date": date, "signals": kept},
                               ensure_ascii=False), encoding="utf-8")
    f_ot.write_text(json.dumps(ot2), encoding="utf-8")
    f_ipx.write_text(json.dumps(ipx2), encoding="utf-8")
    f_log.write_text("\n".join(log2) + "\n", encoding="utf-8")
    print(f"已落盘 · 备份 {', '.join(names)}")


if __name__ == "__main__":
    main()
