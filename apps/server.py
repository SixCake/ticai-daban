# -*- coding: utf-8 -*-
"""轻量本地服务: 看板静态页 + JSON API

GET /                   → web/dashboard.html
GET /api/live           → data/live/latest.json
GET /api/radar          → data/live/radar.json (预警雷达)
GET  /api/focus          → data/live/focus.json (专注板块集合)
POST /api/focus          → 覆写 focus.json (body: {items:[{type,code,name}]})
GET /api/factors?date=  → factor.longtou 市场生态+个股因子(默认最新决策日)
GET /api/review?date=   → data/review/review_DATE.json (缺失则现场构建)
GET /api/dates          → 最近60个交易日列表
GET /api/intraday?code=&date=  → 全天分时合并数据 {code,pts,pb,pt,why}
GET /api/dailyk?code=&n=  → 日K线(不复权) {code,bars,src} 供个股详情弹窗
GET /api/intradaypx_batch?codes=a,b&date=  → 多票原始分时(跷跷板图表)
GET /api/simlist        → 策略启用清单(strategies/strategies.yaml)
GET /api/runs?kind=&strategy= → run 清单(kind=live 策略模拟 / backtest 策略回测)
GET /api/run?id=&asof=  → run 详情(指标/曲线/交易/持仓/日志; live 含盘中实时点)
POST /api/backtest/run  → 发起回测 {strategy,start,end,freq,capital}
POST /api/sim/start     → 发起模拟 {strategy,seed_run?,capital?} seed_run=以某次回测为起点
POST /api/sim/stop      → 关闭/取消 run {id}

启动: python apps/server.py [port]  默认8765
"""
import json
import os
import signal
import subprocess
import sys
from collections import defaultdict
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from apps.review import build_review  # noqa: E402
from core.longtou import QSCORE_NEXT_WIN, SSCORE_SEAL, env_status  # noqa: E402
from datastore import load, path_of  # noqa: E402
from core.heat import sw_aggregate  # noqa: E402

WEB = ROOT / "web"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

# 进程级缓存：intraday_px 文件较大，按 mtime 失效
_ipx_cache: dict = {}
_con_px_cache: dict = {}
# presig_state 精简缓存(每次全量解析数百MB耗时秒级, 按 mtime 失效):
# date -> (mt, {code: [(px_hist, pb, pt, why), ...]})
_ps_cache: dict = {}
# 腾讯官方分钟分时缓存(真实价): (date, code) -> (fetch_ts, res)
# 盘中 60s TTL 刷新, 盘后数据不变自然命中
_tx_min_cache: dict = {}
# 腾讯快照昨收(分时图涨跌幅/涨跌停基准): (date, code) -> (fetch_ts, pc)
_tx_pc_cache: dict = {}
# 腾讯 m1 分钟K缓存: code -> rows(800根, 历史不变; 当日盘中变但不用于当日)
_m1_rows_cache: dict = {}
# m1 vol→股倍率缓存(量纲分板块且跨日不变): code -> mult
_m1_mult_cache: dict = {}
# 因子表缓存：每日收盘后才变化，按 mtime 失效；{date: payload_json}
_fac_cache: dict = {"mt": None, "dates": [], "payloads": {}}
# 日K缓存(个股详情弹窗): code -> (fetch_ts, bars), 盘中60s TTL
_daily_k_cache: dict = {}
# 申万某日重算缓存(日期维度回看): date -> {date, sw}
_swday_cache: dict = {}

_FAC_MARKET = ("zt_prev", "ld_prev", "ldlr_prev", "adv_prev", "cycle_prev",
               "mvol_prev")


def _num(v, nd=3):
    """NaN/NA→None, 其余转纯Python数值(保证JSON合法)"""
    try:
        if v is None or pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return int(v) if nd < 0 else round(float(v), nd)


def _factor_payload(date: str | None) -> dict:
    p = path_of("factor.longtou")
    if not p.exists():
        return {"error": "factor.longtou 不存在, 先跑 collect/factor_longtou.py"}
    mt = p.stat().st_mtime
    if _fac_cache["mt"] != mt:
        dates = sorted(pd.read_parquet(p, columns=["trade_date"])
                       ["trade_date"].unique().tolist())
        _fac_cache.update({"mt": mt, "dates": dates, "payloads": {}})
    if not _fac_cache["dates"]:
        return {"error": "factor.longtou 为空"}
    date = date or _fac_cache["dates"][-1]
    if date in _fac_cache["payloads"]:
        return json.loads(_fac_cache["payloads"][date])
    df = pd.read_parquet(p, filters=[("trade_date", "=", date)])
    if df.empty:
        return {"error": f"因子表无 {date}"}
    # stat_date = 因子数据的截止交易日(决策日T的统计来自T-1)
    idx = _fac_cache["dates"].index(date)
    stat_date = _fac_cache["dates"][idx - 1] if idx > 0 else None
    mrow = df.iloc[0]
    market = {k: _num(mrow[k]) for k in _FAC_MARKET}
    market["env"] = env_status(market)
    stocks = {}
    cols = ["ts_code", "industry", "qscore", "sscore", "zb_cnt20", "ind_rank",
            "y_volr5", "neg_streak", "ind_breadth", "ind_ztdens"]
    for r in df[cols].itertuples(index=False):
        q, s = _num(r[2], -1), _num(r[3], -1)
        stocks[r[0]] = {
            "ind": None if pd.isna(r[1]) else str(r[1]),
            "q": q, "s": s,
            # 两位小数校准概率(研究23 test段): qw=次日胜率 sp=封板率
            "qw": QSCORE_NEXT_WIN.get(q), "sp": SSCORE_SEAL.get(s),
            "zb": _num(r[4], 0), "rk": _num(r[5], 0), "vr": _num(r[6], 2),
            "ng": _num(r[7], -1), "brd": _num(r[8], 3), "ztd": _num(r[9], 3),
        }
    payload = {"date": date, "stat_date": stat_date,
               "market": market, "stocks": stocks}
    _fac_cache["payloads"][date] = json.dumps(payload, ensure_ascii=False)
    return payload


def _tx_daily_k(code: str, n: int = 120) -> list:
    """腾讯日K(不复权): [["YYYYMMDD", o, h, l, c, vol手], ...] 升序。

    接口行序为 [date, open, close, high, low, volume], 此处统一转 ohlc;
    盘中会多一根当日未收定 bar(本地面板盘后才补尾), 供日K图及时展示。
    vol 量纲分板块(与_m1_mult同口径): 科创板/北交所返股, 其余返手,
    统一归一为手以与 market.daily_panel 拼接不断层。
    """
    import time as _time
    import urllib.request
    now = _time.time()
    ck = _daily_k_cache.get(code)
    if ck and now - ck[0] < 60:
        return ck[1]
    sym = ("sh" if code.endswith(".SH") else
           "bj" if code.endswith(".BJ") else "sz") + code[:6]
    url = ("https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param="
           + f"{sym},day,,,{int(n)}")
    vmult = 0.01 if code[:3] in ("688", "689") or code.endswith(".BJ") \
        else 1.0
    bars: list = []
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            j = json.loads(r.read().decode("utf-8"))
        data = j["data"][sym]
        for row in (data.get("day") or data.get("qfqday") or []):
            try:
                bars.append([str(row[0]).replace("-", "")[:8],
                             float(row[1]), float(row[3]), float(row[4]),
                             float(row[2]),
                             float(row[5]) * vmult if len(row) > 5 else 0.0])
            except (ValueError, IndexError, TypeError):
                continue
    except Exception:
        bars = []
    if bars:
        _daily_k_cache[code] = (now, bars)
    return bars


def _panel_daily_k(code: str, n: int = 120) -> list:
    """本地面板日K(market.daily_panel, 收盘定稿口径)。

    pyarrow 行组统计可裁剪, 单票查询实测~0.1s, 不依赖外部行情。
    """
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path_of("market.daily_panel"),
                          columns=["trade_date", "open", "high", "low",
                                   "close", "vol"],
                          filters=[("ts_code", "=", code)])
        df = t.to_pandas().sort_values("trade_date").tail(n)
        return [[str(r.trade_date), float(r.open), float(r.high),
                 float(r.low), float(r.close), float(r.vol)]
                for r in df.itertuples()]
    except Exception:
        return []


def _daily_k(code: str, n: int = 120) -> tuple:
    """日K合并(不复权): 腾讯补当日盘中bar + 本地面板覆盖已收定交易日。

    两源 vol 均已归一为手, 拼接不会出现量纲断层。
    返回 (bars 升序尾n根, 源标记); 任一源失败则单源降级。
    """
    by: dict = {}
    src = []
    tx = _tx_daily_k(code, n)
    if tx:
        src.append("tx")
        for b in tx:
            by[b[0]] = b
    pn = _panel_daily_k(code, n)
    if pn:
        src.append("panel")
        for b in pn:
            by[b[0]] = b          # 面板为收盘定稿口径, 覆盖腾讯
    return [by[k] for k in sorted(by)][-n:], ("+".join(src) or "none")


def _fmt_t(t: str) -> str:
    """统一时间格式为 HH:MM:SS。支持 HHMMSS 和 HH:MM:SS 两种输入。"""
    t = t.strip()
    if len(t) == 6 and ":" not in t:
        return f"{t[:2]}:{t[2:4]}:{t[4:6]}"
    return t


def _presig_index(date: str) -> dict:
    """presig_state 精简索引: {code: [(px_hist, pb, pt, why), ...]}。

    全量文件数百MB(含全部信号px_hist), 每请求解析一次耗时秒级,
    按 mtime 失效缓存解析结果(内存中仅保留精简字段)。
    """
    f = DATA / "live" / f"presig_state_{date}.json"
    if not f.exists():
        return {}
    mt = f.stat().st_mtime
    ck = _ps_cache.get(date)
    if ck is not None and ck[0] == mt:
        return ck[1]
    by_code: dict = {}
    try:
        ps = json.loads(f.read_text(encoding="utf-8"))
        for s in ps.get("signals", []):
            by_code.setdefault(s.get("ts_code", ""), []).append(
                (s.get("px_hist", []), s.get("pb"), s.get("pt"),
                 s.get("why")))
    except Exception:
        by_code = {}
    _ps_cache.clear()          # 只保留最近访问日期(单槽, 防内存膨胀)
    _ps_cache[date] = (mt, by_code)
    return by_code


def _tx_minute(date: str, code: str):
    """腾讯官方分钟分时(真实价+量能), 当日数据。

    返回 (pts, vols, vwap) 或 None:
      pts  [[HH:MM:SS, px, cumvol原始, cumamt], ...] 价格点(供价格合并)
      vols [[HH:MM, vol股, amt元], ...] 分钟差分量能(量纲归一为股)
      vwap [[HH:MM:SS, 均价], ...] 官方累计amt/累计vol(股)
    背景: qmt 断连期间本地 px_hist/intraday_px 的 vol/amt 是脏字段
    (字段错位/多源累计基数不连续, 实测混出 118亿假成交额), 分时图
    量柱与 VWAP 一律以腾讯官方为准。
    vol 量纲分板块(实测科创板返股、主板/创业板返手): 首行 amt/vol
    ≈px(股) 或 ≈px×100(手) 自适应判别, 不依赖板块硬规则。
    """
    import time as _time
    import urllib.request
    now = _time.time()
    ck = _tx_min_cache.get((date, code))
    if ck and now - ck[0] < 60:
        return ck[1]
    sym = ("sh" if code.endswith(".SH") else
           "bj" if code.endswith(".BJ") else "sz") + code[:6]
    url = ("https://web.ifzq.gtimg.cn/appstock/app/minute/query?code="
           + sym)
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            j = json.loads(r.read().decode("utf-8"))
        raw = j["data"][sym]["data"]["data"]
    except Exception:
        return None
    # qt 快照顺带提取昨收(qt[4]): 分时图涨跌幅/涨跌停线基准
    try:
        q = j["data"][sym]["qt"][sym]
        _tx_pc_cache[(date, code)] = (
            now, float(q[4]) if q and len(q) > 4 and q[4] else None)
    except Exception:
        _tx_pc_cache[(date, code)] = (now, None)
    rows = [p for p in (it.split() for it in raw) if len(p) >= 3]
    # 量纲判别: 首几行 amt/vol 与分钟价 px 对照(股≈1倍, 手≈100倍)
    mult = None
    for p in rows[:3]:
        try:
            px, v0, a0 = float(p[1]), float(p[2]), float(p[3])
        except (ValueError, IndexError):
            continue
        if v0 > 0 and a0 > 0 and px > 0:
            r0 = a0 / v0 / px
            if 0.8 <= r0 <= 1.25:
                mult = 1.0
                break
            if 80 <= r0 <= 125:
                mult = 100.0
                break
    if mult is None:     # 兜底: 科创板/北交所=股, 其余=手(实测口径)
        mult = (1.0 if code[:3] in ("688", "689")
                or code.endswith(".BJ") else 100.0)
    pts: list = []
    vols: list = []
    vwap: list = []
    pv = pa = 0.0
    for p in rows:
        hhmm, px = p[0], float(p[1])
        vol = float(p[2])
        amt = float(p[3]) if len(p) > 3 else 0.0
        t = f"{hhmm[:2]}:{hhmm[2:]}:00"
        if not ("09:15:00" <= t <= "15:00:59"):   # 滤盘后固定点
            continue
        pts.append([t, px, vol, amt])
        vols.append([t[:5], max(0.0, vol - pv) * mult,
                     max(0.0, amt - pa)])
        vs = vol * mult
        if vs > 0:
            vwap.append([t, amt / vs])
        pv, pa = vol, amt
    res = (pts, vols, vwap) if pts else None
    _tx_min_cache[(date, code)] = (now, res)
    return res


def _m1_rows(code: str) -> list:
    """腾讯 m1 分钟K(800根≈最近4个交易日), 进程级缓存。
    行格式 [YYYYMMDDHHMM, o, c, h, l, vol, ...]; 供昨收与历史日量能。"""
    ck = _m1_rows_cache.get(code)
    if ck is not None:
        return ck
    import urllib.request
    sym = ("sh" if code.endswith(".SH") else
           "bj" if code.endswith(".BJ") else "sz") + code[:6]
    url = ("https://ifzq.gtimg.cn/appstock/app/kline/mkline?param="
           + f"{sym},m1,,800")
    rows: list = []
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            j = json.loads(r.read().decode("utf-8"))
        data = j["data"][sym]
        key = next(k for k in data if str(k).startswith("m1"))
        rows = data[key]
    except Exception:
        rows = []
    _m1_rows_cache[code] = rows
    return rows


def _m1_mult(code: str) -> float:
    """m1 vol → 股 倍率。当日 minute/query 全天总量(股)锚定;
    失败降级板块规则(科创板=股, 其余=手)。接口行为跨日不变。"""
    ck = _m1_mult_cache.get(code)
    if ck is not None:
        return ck
    mult = None
    from datetime import datetime as _dt
    today = _dt.now().strftime("%Y%m%d")
    txr = _tx_minute(today, code)
    if txr:
        tot_shares = sum(v[1] for v in txr[1])
        tot_m1 = sum(float(r[5]) for r in _m1_rows(code)
                     if r[0][:8] == today)
        if tot_shares > 0 and tot_m1 > 0:
            r0 = tot_shares / tot_m1
            if 0.8 <= r0 <= 1.25:
                mult = 1.0
            elif 80 <= r0 <= 125:
                mult = 100.0
    if mult is None:
        mult = (1.0 if code[:3] in ("688", "689")
                or code.endswith(".BJ") else 100.0)
    _m1_mult_cache[code] = mult
    return mult


def _m1_day_vols(date: str, code: str):
    """历史日分钟量能(m1分钟K近似, 额≈量×收盘):
    ([[HH:MM, vol股, amt], ...], [[HH:MM:SS, vwap], ...])。"""
    rows = [r for r in _m1_rows(code) if r[0][:8] == date]
    if not rows:
        return [], []
    mult = _m1_mult(code)
    vols: list = []
    vwap: list = []
    cv = ca = 0.0
    for r in rows:
        try:
            hhmm = r[0][8:12]
            close, vol = float(r[2]), float(r[5])
        except (ValueError, IndexError):
            continue
        if vol < 0:
            vol = 0.0
        v = vol * mult
        t = f"{hhmm[:2]}:{hhmm[2:]}"
        vols.append([t, v, v * close])
        cv += v
        ca += v * close
        if cv > 0:
            vwap.append([t + ":00", ca / cv])
    return vols, vwap


def _tx_m1_prev_close(date: str, code: str) -> float | None:
    """腾讯 m1 分钟K取昨收: date 前一交易日的最后一根分钟 close。"""
    prev = None
    for row in _m1_rows(code):       # [YYYYMMDDHHMM, o, c, h, l, v, ...]
        if row[0][:8] < date:
            prev = float(row[2])
        else:
            break
    return prev


def _log_prev_close(date: str, code: str, m: dict) -> float | None:
    """radar_log 的 pct × 本地分时价配对反推昨收(离线兜底)。
    px = 昨收×(1+pct/100) → 昨收 = px/(1+pct/100), 取中位数抗个别污染点。"""
    f = DATA / "live" / f"radar_log_{date}.jsonl"
    if not f.exists() or not m:
        return None
    sec_px: dict[int, float] = {}
    for t, v in m.items():
        p = t.split(":")
        if len(p) == 3:
            sec_px[int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])] = v[0]
    bases: list = []
    try:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("code") != code or r.get("pct") is None:
                    continue
                tt = _fmt_t(str(r.get("t", "")))
                if ":" not in tt:
                    continue
                q = tt.split(":")
                sec = int(q[0]) * 3600 + int(q[1]) * 60 + int(q[2])
                for d in (0, -2, 2, -5, 5):   # ±5s 内就近配对
                    px = sec_px.get(sec + d)
                    if px:
                        bases.append(px / (1 + r["pct"] / 100.0))
                        break
    except Exception:
        return None
    if len(bases) < 3:
        return None
    bases.sort()
    return round(bases[len(bases) // 2], 3)


def _merge_with_tx(local_pts: list, tx_pts: list) -> list:
    """本地高频点优先 + 腾讯官方分钟线真值校验补缺。

    qmt 推送(局域网2s级)与雷达 sweep(20s)写本地 px_hist/intraday_px,
    是主力高频源; 但断连/重启期间会混入异基准平行价(实测与真实价
    偏差1.2%~2.4%不等, 陈旧昨收/多源混写), 本地无法自证真实。
    以腾讯官方分钟价为真值逐分钟裁决:
    - 单层分钟(无平行线): 层内中位价偏差≤1% → 整分钟保留(含分钟内
      真实波动); 否则整分钟污染由腾讯分钟点补;
    - 多层分钟(同分钟平行线, 间隙>0.8%): 与真值最近的层为主层;
      最近两层偏差差<0.3%(等距夹逼不可分辨) → 整分钟换腾讯点;
      主层保留, 其余层仅当偏差≤1%且与主层时间先后不交错(真实急拉)
      才保留, 时刻交错的平行线层剔除;
    - 集合竞价段(腾讯无分钟线)与开盘价偏差>0.5%(撮合后应静止)剔除。
    """
    if not tx_pts:
        return local_pts
    tx_by_min = {p[0][:5]: p for p in tx_pts}
    minutes = sorted(tx_by_min)
    min_idx = {m: i for i, m in enumerate(minutes)}
    open_ref = tx_pts[0][1]        # 开盘价, 校验集合竞价段

    by_min: dict[str, list] = defaultdict(list)
    pre: list = []
    for p in local_pts:
        if p[0][:5] in tx_by_min:
            by_min[p[0][:5]].append(p)
        else:
            pre.append(p)

    # 集合竞价段(腾讯无分钟线): 9:25撮合后价格恒等于开盘价,
    # 偏差>0.5%(超出撮合价静态特性)视为污染剔除
    out: list = [p for p in pre
                 if abs(p[1] - open_ref) / open_ref <= 0.005]
    covered: set = set()
    for m, items in by_min.items():
        i = min_idx[m]
        refs = [tx_by_min[m][1]]
        if i > 0:                  # 分钟开头点对齐前一分钟收盘更公平
            refs.append(tx_by_min[minutes[i - 1]][1])

        def dev(px: float) -> float:      # 价格对真值参考的最小相对偏差
            return min(abs(px - r) / r for r in refs)

        layers = _split_layers(sorted(items, key=lambda p: p[1]), 0.004)
        if len(layers) == 1:
            # 单层: 层内中位价校验(分钟内真实波动全保留)
            ly = layers[0]
            if dev(ly[len(ly) // 2][1]) <= 0.010:
                out.extend(items)
                covered.add(m)
            continue                  # 整层污染 → 腾讯补
        # 多层(同分钟平行线): 按层代表偏差裁决
        stats = sorted(((dev(ly[len(ly) // 2][1]), ly) for ly in layers),
                       key=lambda x: x[0])
        d1, main = stats[0]
        if d1 > 0.010:
            continue                  # 最近层也污染 → 腾讯补
        if len(stats) > 1 and stats[1][0] - d1 < 0.003:
            continue                  # 双层等距夹逼不可分辨 → 腾讯补
        # 主层与各非主层在时间序上层间交替([B,A,B,A]) → 平行线剔除;
        # 单调先后([B,B,A,A]真实急拉)且偏差≤1% → 保留
        seq = sorted((p[0], k) for k, (_, ly) in enumerate(stats)
                     for p in ly)
        order = [k for _, k in seq]
        switches = sum(1 for i in range(len(order) - 1)
                       if order[i] != order[i + 1])
        ok_layers = {0}               # 主层恒保留
        if switches <= 2:             # 单调先先后后(至多一次切换)
            ok_layers = {k for _, k in seq}
        m_ts = [t for t, k in seq if k == 0]
        lo, hi = (min(m_ts), max(m_ts)) if m_ts else ("", "")
        for k, (d, ly) in enumerate(stats):
            if k == 0 or k not in ok_layers:
                if k == 0:
                    out.extend(ly)
                continue
            ts = [p[0] for p in ly]    # 与主层时间不交错(平行线则剔除)
            if d <= 0.010 and (all(t < lo for t in ts)
                               or all(t > hi for t in ts)):
                out.extend(ly)
        covered.add(m)
    for m in minutes:
        if m not in covered:
            out.append(list(tx_by_min[m]))
    out.sort(key=lambda r: r[0])
    return out


def _split_layers(items: list, thr: float) -> list:
    """按最大价格间隙递归分层(间隙>thr处切开)。"""
    if len(items) < 2:
        return [items]
    best_gap, best_i = 0.0, -1
    for i in range(len(items) - 1):
        g = (items[i + 1][1] - items[i][1]) / items[i][1]
        if g > best_gap:
            best_gap, best_i = g, i
    if best_gap < thr:
        return [items]
    return (_split_layers(items[:best_i + 1], thr)
            + _split_layers(items[best_i + 1:], thr))


def _clean_series(pts: list) -> list:
    """剔除多进程/多行情源混写造成的异基准平行价格层。

    背景: qmt横截面的昨收字段陈旧(停在前一交易日), 与腾讯源真实昨收
    差~2.4%, 同一票两套平行价格; 雷达进程反复重启/并行时 px_hist
    同时记录两套, 按时刻排序后逐点交替 → 分时图锯齿成块。
    区分异基准层与真实急拉/回落层: 异基准层与主层的时刻交错
    (同分钟内 A@:00 B@:04 A@:20 B@:24), 真实分层的时间先后聚集
    (如拉尾盘高层点全在后半分钟)。时刻交错的少数层剔除。
    """
    if len(pts) < 6:
        return pts
    buckets: dict[str, list] = defaultdict(list)
    for row in pts:
        buckets[row[0][:5]].append(row)
    out: list = []
    for minute in sorted(buckets):
        items = sorted(buckets[minute], key=lambda r: r[1])
        layers = _split_layers(items, 0.012)
        if len(layers) == 1:
            out.extend(items)
            continue
        main = max(layers, key=len)
        m_ts = [r[0] for r in main]
        lo, hi = min(m_ts), max(m_ts)
        out.extend(main)
        for ly in layers:
            if ly is main:
                continue
            ts = [r[0] for r in ly]
            # 整层在主层时间跨度之前/之后 → 真实先后形态, 保留;
            # 与主层交错 → 异基准平行线, 剔除
            if all(t < lo for t in ts) or all(t > hi for t in ts):
                out.extend(ly)
    out.sort(key=lambda r: r[0])
    return out


def _build_intraday(code: str, date: str) -> dict:
    """合并 intraday_px + presig px_hist，返回 {code, pts, pb, pt, why}。

    当日: 本地高频点(qmt推送/sweep)为主, 腾讯官方分钟线真值校验
    (污染点剔除)与缺失分钟补缺; 拉不到腾讯时退化为本地分层清洗。
    历史日: 本地合并 + 分层清洗。
    """
    m: dict[str, list] = {}  # HH:MM:SS -> [price, cumvol, cumamt]

    # 1. intraday_px 价格骨架（时间无冒号；昨收×pct重建, 昨收陈旧时整段偏移）
    global _ipx_cache
    ipx_file = DATA / "live" / f"intraday_px_{date}.json"
    raw_ipx: list = []
    if ipx_file.exists():
        mt = ipx_file.stat().st_mtime
        ck = _ipx_cache.get(date)
        if ck is None or ck[0] != mt:
            ck = (mt, json.loads(ipx_file.read_text(encoding="utf-8")))
            _ipx_cache[date] = ck
        raw_ipx = ck[1].get(code, [])

    # 2. presig px_hist（真实tick口径；mtime缓存解析）
    pb = pt = why = None
    px_entries: list = []
    for hist, s_pb, s_pt, s_why in _presig_index(date).get(code, []):
        px_entries.extend(hist)
        if s_pb is not None:
            pb, pt, why = s_pb, s_pt, s_why

    # 校准: px_hist最早点(真实价) / intraday_px最近邻点(重建价) → 比例校正重建段
    if raw_ipx and px_entries:
        def _sec(t: str) -> int:
            p = _fmt_t(t).split(":")
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        ref = min(px_entries, key=lambda e: _sec(e[0]))
        near = min(raw_ipx, key=lambda e: abs(_sec(e[0]) - _sec(ref[0])))
        if near[1] > 0 and abs(ref[1] / near[1] - 1) > 0.002:
            ratio = ref[1] / near[1]
            raw_ipx = [[e[0], round(e[1] * ratio, 3)] + list(e[2:])
                       for e in raw_ipx]

    for e in raw_ipx:
        m[_fmt_t(e[0])] = [e[1], e[2] if len(e) > 2 else 0,
                            e[3] if len(e) > 3 else 0]
    for entry in px_entries:
        t_key = _fmt_t(entry[0])
        v = [entry[1], entry[2] if len(entry) > 2 else 0,
             entry[3] if len(entry) > 3 else 0]
        old = m.get(t_key)
        # 同时刻多源: 优先保留带分时量的点(腾讯真实口径);
        # qmt陈旧昨收盘(vol=0)不覆盖已有真实点
        if old is None or (v[1] > 0 and old[1] <= 0):
            m[t_key] = v

    # 时段过滤: 非连续竞价时段的点剔除(盘后手动补跑雷达会追加
    # 收盘后的重复平台点, 污染"最新价/最新时刻"与图形右端)
    pts = [[t] + v for t, v in sorted(m.items())
           if "09:15:00" <= t <= "15:00:59"]
    from datetime import datetime as _dt
    is_today = date == _dt.now().strftime("%Y%m%d")
    vols: list = []
    vwap: list = []
    if is_today:
        txr = _tx_minute(date, code)
        if txr:
            pts = _merge_with_tx(pts, txr[0])
            vols, vwap = txr[1], txr[2]
        else:
            pts = _clean_series(pts)
    else:
        pts = _clean_series(pts)
        vols, vwap = _m1_day_vols(date, code)
    # 昨收(涨跌幅/涨跌停基准): 当日腾讯快照qt > m1前一交易日收盘 > log配对
    pc = None
    if is_today:
        ck_pc = _tx_pc_cache.get((date, code))
        if ck_pc and ck_pc[1]:
            pc = ck_pc[1]
    if pc is None:
        pc = _tx_m1_prev_close(date, code)
    if pc is None:
        pc = _log_prev_close(date, code, m)
    return {"code": code, "pts": pts, "pb": pb, "pt": pt, "why": why,
            "pc": pc, "vols": vols, "vwap": vwap}


def _sw_day(date: str) -> dict:
    """从 daily_panel 重算某日申万一级/二级聚合(收盘口径), 供看板日期回看。
    pct=pct_chg等权; amt=Σ(vol手×close×100)元; net=Σ涨家amt−Σ跌家amt;
    历史日无分时→s3/vr=0, 涨停按板别幅度判定。"""
    ck = _swday_cache.get(date)
    if ck is not None:
        return ck
    import pandas as pd
    p = path_of("market.daily_panel")
    if not p.exists():
        return {"date": date, "sw": []}
    dp = pd.read_parquet(p, filters=[("trade_date", "=", date)])
    dp = dp.dropna(subset=["close", "pct_chg", "vol", "pre_close"])
    names: dict = {}
    fn = DATA / "meta" / "qmt_names.json"
    if fn.exists():
        names = json.loads(fn.read_text(encoding="utf-8")).get("data", {})
    fsw = DATA / "meta" / "sw_map.json"
    sw_map = json.loads(fsw.read_text(encoding="utf-8")) if fsw.exists() else {}

    def _lim(pc: float, code: str, nm: str) -> float:
        if code.endswith(".BJ"):
            r = 0.30
        elif code[:3] in ("300", "301", "302", "688", "689"):
            r = 0.20
        elif nm and ("ST" in nm or "退" in nm):
            r = 0.05
        else:
            r = 0.10
        return round(pc * (1 + r), 2)

    quotes: dict = {}
    for row in dp.itertuples():
        nm = names.get(row.ts_code, row.ts_code)
        quotes[row.ts_code] = {
            "name": nm, "pct": float(row.pct_chg),
            "amount": float(row.vol * row.close * 100),
            "price": float(row.close),
            "limit_px": _lim(float(row.pre_close), row.ts_code, nm),
            "vr": 0.0}
    sw = sw_aggregate(sw_map, quotes, {}, {}, 0.0)
    out = {"date": date, "sw": sw}
    _swday_cache[date] = out
    return out


# ---------- 策略模拟 / 策略回测(run 目录模型) ----------
# 回测与模拟同构: 一次运行一个 data/sim/runs/{run_id}/ 目录(meta.json +
# equity/trades/positions/state/run.log), 见 apps/sim.py 头部注释。

SIM_ROOT = DATA / "sim"
RUNS_DIR = SIM_ROOT / "runs"
STRAT_DIR = ROOT / "strategies"
_simlist_cache: dict = {}


def _sim_registry() -> list:
    """strategies/strategies.yaml 的启用清单(mtime 守护缓存)"""
    f = STRAT_DIR / "strategies.yaml"
    if not f.exists():
        return []
    mt = f.stat().st_mtime
    if _simlist_cache.get("mt") == mt:
        return _simlist_cache["items"]
    try:
        import yaml
        d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        items = d.get("strategies") or []
    except Exception:
        items = []
    _simlist_cache.update({"mt": mt, "items": items})
    return items


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except Exception:
        return False
    return True


def _run_meta(rid: str) -> dict:
    f = RUNS_DIR / rid / "meta.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_status(meta: dict) -> str:
    """展示态状态。live 模拟是跨日的: 子进程收盘退出 ≠ 模拟关闭(次日
    继续), 故 live 只看显式 closed/failed; backtest 子进程异常消失且无
    收尾写 → failed。"""
    st = meta.get("status")
    if meta.get("kind") == "live":
        return st if st in ("closed", "failed") else "running"
    if st == "running" and not _pid_alive(meta.get("pid")):
        return "failed"
    return st


def _latest_state(rd: Path) -> dict:
    sd = Path(rd) / "state"
    if not sd.exists():
        return {}
    fs = sorted(sd.glob("*.json"))
    if not fs:
        return {}
    try:
        return json.loads(fs[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_quick(meta: dict) -> dict:
    """列表页摘要指标(含 live 盘中实时点); 不足样本一律 None 不伪造"""
    out = {"days": 0, "total_return": None, "annualized": None,
           "max_drawdown": None, "equity_now": None, "today_return": None,
           "benchmark_return": None}
    eqf = RUNS_DIR / meta["id"] / "equity.parquet"
    if not eqf.exists():
        return out
    try:
        eq = pd.read_parquet(eqf).sort_values("trade_date")
    except Exception:
        return out
    if not len(eq):
        return out
    nav = eq["equity"].astype(float)
    intraday = None
    if meta.get("kind") == "live":
        st = _latest_state(RUNS_DIR / meta["id"])
        if st and str(st.get("date")) > str(eq["trade_date"].astype(str).iloc[-1]):
            intraday = st
    last_eq = float(intraday["equity"]) if intraday else float(nav.iloc[-1])
    first_eq = float(nav.iloc[0])
    out["days"] = int(len(eq)) + (1 if intraday else 0)
    out["equity_now"] = round(last_eq, 2)
    out["total_return"] = round(last_eq / first_eq - 1, 6) if first_eq else None
    if out["days"] >= 2 and first_eq > 0 and last_eq > 0:
        out["annualized"] = round((last_eq / first_eq) ** (242 / out["days"]) - 1, 6)
    series = [float(v) for v in nav.values] + ([last_eq] if intraday else [])
    peak, mdd = series[0], 0.0
    for v in series:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1)
    out["max_drawdown"] = round(mdd, 6)
    if intraday and float(nav.iloc[-1]) > 0:
        out["today_return"] = round(last_eq / float(nav.iloc[-1]) - 1, 6)
    bm = eq["benchmark"].astype(float).dropna()
    if len(bm) >= 1 and float(bm.iloc[0]) > 0:
        b_last = (intraday.get("benchmark") if intraday
                  and intraday.get("benchmark") else None)
        b_last = float(b_last) if b_last else float(bm.iloc[-1])
        out["benchmark_return"] = round(b_last / float(bm.iloc[0]) - 1, 6)
    return out


def _runs_list(kind: str | None = None, strategy: str | None = None) -> list:
    from apps import sim as simmod
    try:
        simmod.migrate_legacy()   # 幂等兑底: legacy 环境首次打开看板即迁移
    except Exception:
        pass
    out = []
    if not RUNS_DIR.exists():
        return out
    for rd in sorted(RUNS_DIR.iterdir()):
        if not rd.is_dir():
            continue
        meta = _run_meta(rd.name)
        if not meta:
            continue
        if kind and meta.get("kind") != kind:
            continue
        if strategy and meta.get("strategy") != strategy:
            continue
        out.append({**meta, "status": _run_status(meta),
                    "alive": _pid_alive(meta.get("pid")),
                    **_run_quick(meta)})
    out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return out


def _to_records(df) -> list:
    """DataFrame → list[dict], 规整两类坑(均为 sys_analyser 输出实测):
    ① index 名字与某列同名(trades 的 index 叫 datetime, 又有一列 datetime)
       → reset_index 报 "cannot insert datetime, already exists"
    ② 两列同名(trades 有 datetime,datetime 两列) → itertuples 会把重名
       列变成 _1/_2, getattr 取不到
    规整: 先改 index 名避免撞列, reset 后给重名列加后缀。"""
    if df is None or not len(df):
        return []
    df = df.copy()
    if df.index.name and df.index.name in df.columns:
        df.index.name = "_idx_" + str(df.index.name)
    df = df.reset_index()
    seen, newcols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            newcols.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            newcols.append(c)
    df.columns = newcols
    return [r._asdict() for r in df.itertuples()]


def _cross_metrics(eq: "pd.DataFrame", risk_free: float = 0.015) -> dict:
    """跨日(模拟开始→当日)全套绩效指标, 用 rqrisk 对日收益序列算。

    为何不用 pkl 的 summary: pkl 是单次 run 的结果, live 模式每天跑单日
    会覆盖它; 用户要的是"从模拟时间到当日"的整体指标, 故对跨日净值序列
    重算。基准取 equity parquet 的 benchmark 列(自建打板基准或指数)。
    不足 2 个交易日的指标返回 None(不伪造)。"""
    import math
    import numpy as np
    out = {}
    if eq is None or len(eq) < 2:
        return out
    eq = eq.sort_values("trade_date").reset_index(drop=True)
    nav = eq["equity"].astype(float).values
    strat_ret = np.diff(nav) / nav[:-1]
    bm = eq["benchmark"].astype(float)
    if bm.notna().sum() >= 2:
        bmv = bm.fillna(method="ffill").values if hasattr(bm, "fillna") \
            else bm.ffill().values
        bench_ret = np.diff(bmv) / np.where(bmv[:-1] == 0, 1, bmv[:-1])
    else:
        bench_ret = np.zeros_like(strat_ret)
    try:
        from rqrisk.risk import Risk, DAILY
        k = Risk(strat_ret, bench_ret, risk_free, period=DAILY,
                 trading_days_a_year=244)
        def _g(fn):
            try:
                v = getattr(k, fn)
                v = v() if callable(v) else v   # rqrisk 指标多为属性而非方法
                v = float(v)
                return None if math.isnan(v) else v
            except Exception:
                return None
        out.update({
            "total_returns": _g("return_rate"),
            "annualized_returns": _g("annual_return"),
            "excess_returns": _g("excess_return_rate"),
            "excess_annual_returns": _g("excess_annual_return"),
            "alpha": _g("alpha"), "beta": _g("beta"),
            "sharpe": _g("sharpe"), "sortino": _g("sortino"),
            "information_ratio": _g("information_ratio"),
            "max_drawdown": _g("max_drawdown"),
            "excess_max_drawdown": _g("excess_max_drawdown"),
            "excess_sharpe": _g("excess_sharpe"),
            "volatility": _g("annual_volatility"),
            "excess_volatility": _g("excess_annual_volatility"),
            "benchmark_volatility": _g("benchmark_annual_volatility"),
            "downside_risk": _g("annual_downside_risk"),
            "tracking_error": _g("annual_tracking_error"),
            "win_rate": _g("win_rate"),
            "var": _g("var"), "calmar": _g("calmar"),
        })
    except Exception:
        out = {}
    # 最大回撤区间(自算: 净值峰值→谷值)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    if len(dd) and dd.min() < 0:
        i_end = int(np.argmin(dd))
        i_start = int(np.argmax(nav[:i_end + 1])) if i_end > 0 else 0
        out["max_drawdown_duration_start_date"] = str(eq["trade_date"][i_start])
        out["max_drawdown_duration_end_date"] = str(eq["trade_date"][i_end])
    out["days"] = int(len(eq))
    out["start_date"] = str(eq["trade_date"].iloc[0])
    out["end_date"] = str(eq["trade_date"].iloc[-1])
    return out


def _log_progress(log_lines) -> str | None:
    """日志里最新模拟日期([YYYY-MM-DD ...] 前缀) = 回测/补跑进度"""
    import re as _re
    last = None
    for ln in log_lines:
        m = _re.match(r"\[(\d{4}-\d{2}-\d{2}) ", ln)
        if m:
            last = m.group(1).replace("-", "")
    return last


def _live_state_block(st: dict) -> dict:
    """详情 payload 的实时快照块(资金/持仓/当日盈亏)"""
    if not st:
        return {}
    return ({k: st.get(k) for k in
             ("date", "ts", "equity", "cash", "frozen_cash",
              "market_value", "start_cash", "day_pnl",
              "total_pnl", "total_pnl_pct")}
            | {"positions": st.get("positions") or []})


def _run_payload(rid: str, asof: str | None = None) -> dict:
    """run 详情数据(回测与模拟同构, 对齐聚宽回测详情的信息结构)。

    asof: 截至某日(YYYYMMDD)。缺省=最新。给定则把 equity/trades/
    positions 截断到 <= asof 并重算指标 —— 供模拟详情日期导航
    (默认最新/前后交易日跳转/一键最新) 展示"截至某日"的整体表现。

    live 实时: 最新 state 的日期新于 equity 末日时, 把盘中快照拼成
    当日净值点并入序列 → 指标/曲线盘中实时产出(与回测同一套口径)。

    跨日口径(模拟开始 → 当日):
      指标/收益曲线   ← run_dir/equity.parquet + rqrisk 重算
      交易/回合盈亏   ← run_dir/trades.parquet (sim.py 逐日累积)
      每日持仓        ← run_dir/positions.parquet (逐日累积)
    实时/单次口径:
      当前持仓/资金   ← run_dir/state/ 最新快照(live)
      被抽掉的订单    ← 最新 state 的 fill_sim_skipped
      日志            ← run_dir/run.log 末尾
    """
    import math
    rd = RUNS_DIR / rid
    meta = _run_meta(rid)
    if not meta:
        return {"error": f"run {rid} 不存在"}
    eqf = rd / "equity.parquet"
    if not eqf.exists():
        # 尚无 equity(回测运行中/首日模拟未结算): 仍返回骨架 payload ——
        # 日志流+进度日实时跟进, 不能只给一句 error 让页面干等
        st0 = _latest_state(rd) if meta.get("kind") == "live" else {}
        lf0 = rd / "run.log"
        log0 = (lf0.read_text(encoding="utf-8", errors="replace")
                .splitlines()[-400:] if lf0.exists() else [])
        return {
            "run_id": rid,
            "meta": {**meta, "status": _run_status(meta),
                     "alive": _pid_alive(meta.get("pid")),
                     "progress_date": _log_progress(log0)},
            "intraday_ts": st0.get("ts") if st0 else None,
            "live_state": _live_state_block(st0),
            "metrics": {}, "portfolio": [], "benchmark": [], "trades": [],
            "round_trips": [], "n_win": 0, "n_loss": 0, "positions": [],
            "skipped": (st0.get("fill_sim_skipped") or []) if st0 else [],
            "log": log0,
        }
    try:
        eq = pd.read_parquet(eqf)
    except Exception as e:
        return {"error": f"读取 equity 失败: {e}"}
    # ---- live 盘中实时点 ----
    st = _latest_state(rd) if meta.get("kind") == "live" else {}
    intraday_ts = None
    if st and str(st.get("date")) > str(eq["trade_date"].astype(str).max()):
        intraday_ts = st.get("ts")
        eq = pd.concat([eq, pd.DataFrame([{
            "trade_date": st["date"], "equity": st["equity"],
            "cash": st.get("cash"), "position_value": st.get("market_value"),
            "benchmark": st.get("benchmark"),
        }])], ignore_index=True)
    if asof:
        eq = eq[eq["trade_date"].astype(str) <= str(asof)]
        if eq.empty:
            return {"error": f"asof={asof} 之前无记录"}

    def _f(v):
        if v is None:
            return None
        try:
            v = float(v)
        except Exception:
            return None
        return None if math.isnan(v) else round(v, 6)

    metrics = _cross_metrics(eq)

    # ---- 收益曲线(策略/基准 归一净值) ----
    eq = eq.sort_values("trade_date").reset_index(drop=True)
    base_eq = float(eq["equity"].iloc[0]) or 1.0
    portfolio, benchmark = [], []
    bm_base = None
    for r in eq.itertuples():
        d = str(r.trade_date)
        portfolio.append({
            "date": d,
            "unit_net_value": _f(float(r.equity) / base_eq),
            "total_value": _f(r.equity),
            "cash": _f(getattr(r, "cash", None)),
            "market_value": _f(getattr(r, "position_value", None)),
        })
        b = getattr(r, "benchmark", None)
        if b is not None and b == b:
            if bm_base is None:
                bm_base = float(b)
            benchmark.append({"date": d,
                              "unit_net_value": round(float(b) / bm_base, 6)
                              if bm_base else None})
        else:
            benchmark.append({"date": d, "unit_net_value": None})
    metrics["starting_cash"] = base_eq
    metrics["run_type"] = "PAPER(累积)"

    # ---- 交易明细(跨日累积, 按 asof 截断) ----
    trades = []
    tf = rd / "trades.parquet"
    if tf.exists():
        try:
            tr = pd.read_parquet(tf).sort_values("datetime")
            if asof:
                tr = tr[tr["datetime"].astype(str).str[:10].str.replace(
                    "-", "") <= str(asof)]
            for rec in _to_records(tr):
                qty = float(rec.get("last_quantity") or 0)
                px = float(rec.get("last_price") or 0)
                trades.append({
                    "datetime": str(rec.get("datetime") or ""),
                    "code": rec.get("order_book_id", ""),
                    "symbol": rec.get("symbol", ""),
                    "side": str(rec.get("side", "")),
                    "effect": str(rec.get("position_effect", "")),
                    "price": round(px, 3), "quantity": int(qty),
                    "amount": round(px * qty, 2),
                    "commission": _f(rec.get("commission")),
                    "tax": _f(rec.get("tax")),
                })
        except Exception:
            trades = []

    # ---- 回合盈亏(FIFO 买→卖) ----
    round_trips, lots = [], {}
    for t in trades:
        code = t["code"]
        if t["side"] == "BUY":
            lots.setdefault(code, []).append([t["price"], t["quantity"]])
        elif t["side"] == "SELL":
            remain, cost, q = t["quantity"], 0.0, lots.get(code, [])
            while remain > 0 and q:
                bp, bq = q[0]
                take = min(remain, bq)
                cost += bp * take
                q[0][1] -= take
                remain -= take
                if q[0][1] <= 0:
                    q.pop(0)
            sold = t["quantity"] - remain
            if sold > 0:
                avg_cost = cost / sold
                round_trips.append({
                    "code": code, "symbol": t["symbol"],
                    "sell_datetime": t["datetime"],
                    "avg_cost": round(avg_cost, 3),
                    "sell_price": t["price"], "quantity": sold,
                    "pnl": round((t["price"] - avg_cost) * sold, 2),
                    "pnl_pct": round((t["price"] / avg_cost - 1) * 100, 3)
                    if avg_cost else None,
                })
    n_win = sum(1 for r in round_trips if r["pnl"] > 0)
    n_loss = len(round_trips) - n_win
    metrics["trade_win_rate"] = round(n_win / len(round_trips), 4) \
        if round_trips else None

    # ---- 每日持仓(跨日累积, 按 asof 截断) ----
    positions = []
    pf_ = rd / "positions.parquet"
    if pf_.exists():
        try:
            sp = pd.read_parquet(pf_).sort_values(["date", "order_book_id"])
            if asof:
                sp = sp[sp["date"].astype(str).str[:10].str.replace(
                    "-", "") <= str(asof)]
            for rec in _to_records(sp):
                positions.append({
                    "date": str(rec.get("date") or ""),
                    "code": rec.get("order_book_id", ""),
                    "symbol": rec.get("symbol", ""),
                    "quantity": int(float(rec.get("quantity") or 0)),
                    "avg_price": _f(rec.get("avg_price")),
                    "last_price": _f(rec.get("last_price")),
                    "market_value": _f(rec.get("market_value")),
                })
        except Exception:
            positions = []

    # ---- 被成交概率闸门抽掉的订单(最新 state) ----
    skipped = st.get("fill_sim_skipped") or [] if st else []

    # ---- 日志(run 目录) ----
    log_lines = []
    lf = rd / "run.log"
    if lf.exists():
        try:
            log_lines = lf.read_text(
                encoding="utf-8", errors="replace").splitlines()[-400:]
        except Exception:
            log_lines = []

    return {
        "run_id": rid,
        "meta": {**meta, "status": _run_status(meta),
                 "alive": _pid_alive(meta.get("pid")),
                 "progress_date": _log_progress(log_lines)},
        "intraday_ts": intraday_ts,
        "live_state": _live_state_block(st),
        "metrics": metrics,
        "portfolio": portfolio,
        "benchmark": benchmark,
        "trades": trades,
        "round_trips": round_trips,
        "n_win": n_win,
        "n_loss": n_loss,
        "positions": positions,
        "skipped": skipped,
        "log": log_lines,
    }


def _spawn_sim_cmd(extra: list) -> list:
    return [sys.executable, "-u", str(ROOT / "apps" / "sim.py")] + extra


def _spawn_run(run_id: str, extra: list) -> dict:
    """建 run 目录 + meta(running) 并拉起子进程, 日志进 run_dir/run.log"""
    from datetime import datetime as _dt
    rd = RUNS_DIR / run_id
    rd.mkdir(parents=True, exist_ok=True)
    cmd = _spawn_sim_cmd(extra + ["--run-id", run_id])
    fh = open(rd / "run.log", "a", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                         cwd=str(ROOT))
    from apps.sim import write_meta
    write_meta(rd, pid=p.pid, status="running",
               created_at=(_run_meta(run_id).get("created_at")
                           or _dt.now().strftime("%Y-%m-%d %H:%M:%S")))
    return {"run_id": run_id, "pid": p.pid}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/dashboard.html"):
            # 看板页强制 no-store: 启发式缓存会让旧 JS 驻留浏览器
            # (实测旧缓存页没有 btRun 等新函数, 点「运行回测」静默无反应)
            body = (WEB / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/live":
            f = DATA / "live" / "latest.json"
            return self._send_json(f.read_text(encoding="utf-8")
                                   if f.exists() else '{"error":"no live data"}')
        if parsed.path == "/api/radar":
            f = DATA / "live" / "radar.json"
            return self._send_json(f.read_text(encoding="utf-8")
                                   if f.exists() else '{"error":"no radar data"}')
        if parsed.path == "/api/focus":
            f = DATA / "live" / "focus.json"
            return self._send_json(f.read_text(encoding="utf-8")
                                   if f.exists() else '{"items":[]}')
        if parsed.path == "/api/factors":
            date = parse_qs(parsed.query).get("date", [None])[0]
            try:
                payload = _factor_payload(date)
                return self._send_json(
                    json.dumps(payload, ensure_ascii=False),
                    200 if "error" not in payload else 404)
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        if parsed.path == "/api/presig":
            from datetime import datetime as _dt
            date = parse_qs(parsed.query).get(
                "date", [_dt.now().strftime("%Y%m%d")])[0]
            f = DATA / "live" / f"presig_state_{date}.json"
            return self._send_json(f.read_text(encoding="utf-8")
                                   if f.exists()
                                   else '{"date":"%s","signals":[]}' % date)
        if parsed.path == "/api/intraday":
            from datetime import datetime as _dt
            q = parse_qs(parsed.query)
            date = q.get("date", [_dt.now().strftime("%Y%m%d")])[0]
            code = q.get("code", [""])[0]
            if not code:
                return self._send_json('{"error":"code required"}', 400)
            try:
                result = _build_intraday(code, date)
                return self._send_json(json.dumps(result, ensure_ascii=False))
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        if parsed.path == "/api/swday":
            q = parse_qs(parsed.query)
            date = q.get("date", [""])[0]
            if not date:
                return self._send_json('{"error":"date required"}', 400)
            try:
                return self._send_json(json.dumps(_sw_day(date),
                                                  ensure_ascii=False))
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        if parsed.path == "/api/dailyk":
            q = parse_qs(parsed.query)
            code = q.get("code", [""])[0]
            if not code:
                return self._send_json('{"error":"code required"}', 400)
            try:
                n = min(500, int(q.get("n", ["120"])[0]))
                bars, src = _daily_k(code, n)
                return self._send_json(json.dumps(
                    {"code": code, "src": src, "bars": bars},
                    ensure_ascii=False))
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        if parsed.path == "/api/intradaypx":
            from datetime import datetime as _dt
            q = parse_qs(parsed.query)
            date = q.get("date", [_dt.now().strftime("%Y%m%d")])[0]
            code = q.get("code", [""])[0]
            f = DATA / "live" / f"intraday_px_{date}.json"
            if not f.exists():
                return self._send_json('{"code":"%s","pts":[]}' % code)
            mt = f.stat().st_mtime
            ck = _ipx_cache.get(date)
            if ck is None or ck[0] != mt:
                ck = (mt, json.loads(f.read_text(encoding="utf-8")))
                _ipx_cache[date] = ck
            return self._send_json(json.dumps(
                {"code": code, "pts": ck[1].get(code, [])},
                ensure_ascii=False))
        if parsed.path == "/api/intradaypx_batch":
            from datetime import datetime as _dt
            q = parse_qs(parsed.query)
            date = q.get("date", [_dt.now().strftime("%Y%m%d")])[0]
            codes = [c for c in q.get("codes", [""])[0].split(",")
                     if c][:20]
            f = DATA / "live" / f"intraday_px_{date}.json"
            if not f.exists():   # 请求日无分时(非交易时段/当日未跑)→回退最近一日
                import re as _re
                cands = [c for c in (DATA / "live").glob("intraday_px_*.json")
                         if _re.fullmatch(r"intraday_px_\d{8}", c.stem)]
                if cands:
                    f = max(cands, key=lambda c: c.stem)
                    date = f.stem.rsplit("_", 1)[-1]
            if not f.exists() or not codes:
                return self._send_json('{"date":"%s","series":{}}' % date)
            mt = f.stat().st_mtime
            ck = _ipx_cache.get(date)
            if ck is None or ck[0] != mt:
                ck = (mt, json.loads(f.read_text(encoding="utf-8")))
                _ipx_cache[date] = ck
            series = {c: [[p[0], p[1]] for p in ck[1].get(c, [])]
                      for c in codes}      # 只留 [HHMMSS, 价] 减载荷载
            cons = [c for c in q.get("cons", [""])[0].split(",")
                    if c][:10]
            cseries = {}
            if cons:
                fc = DATA / "live" / f"concept_px_{date}.json"
                if fc.exists():
                    mt2 = fc.stat().st_mtime
                    ck2 = _con_px_cache.get(date)
                    if ck2 is None or ck2[0] != mt2:
                        ck2 = (mt2, json.loads(fc.read_text(
                            encoding="utf-8")))
                        _con_px_cache[date] = ck2
                    cseries = {k2: ck2[1].get(k2, []) for k2 in cons}
            return self._send_json(json.dumps(
                {"date": date, "series": series, "cons": cseries},
                ensure_ascii=False))
        if parsed.path == "/api/review":
            date = parse_qs(parsed.query).get("date", [None])[0]
            if not date:
                return self._send_json('{"error":"date required"}', 400)
            f = DATA / "review" / f"review_{date}.json"
            if not f.exists():
                try:
                    snap = build_review(date)
                    if "error" not in snap:
                        f.parent.mkdir(exist_ok=True)
                        f.write_text(json.dumps(snap, ensure_ascii=False),
                                     encoding="utf-8")
                    else:
                        return self._send_json(
                            json.dumps(snap, ensure_ascii=False), 404)
                except Exception as e:
                    return self._send_json(
                            json.dumps({"error": str(e)}, ensure_ascii=False), 500)
            return self._send_json(f.read_text(encoding="utf-8"))
        if parsed.path == "/api/simlist":
            return self._send_json(json.dumps(
                {"strategies": _sim_registry()}, ensure_ascii=False))
        if parsed.path == "/api/runs":
            q = parse_qs(parsed.query)
            kind = q.get("kind", [None])[0]
            strat = q.get("strategy", [None])[0]
            try:
                return self._send_json(json.dumps(
                    {"runs": _runs_list(kind, strat)},
                    ensure_ascii=False, default=str))
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        if parsed.path == "/api/run":
            q = parse_qs(parsed.query)
            rid = q.get("id", [None])[0]
            asof = q.get("asof", [None])[0]
            if not rid:
                return self._send_json('{"error":"id required"}', 400)
            try:
                return self._send_json(json.dumps(
                    _run_payload(rid, asof=asof),
                    ensure_ascii=False, default=str))
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        if parsed.path == "/api/dates":
            ev = load("limitup.events_enriched", columns=["trade_date"])
            dates = sorted(ev["trade_date"].unique())[-60:][::-1]
            return self._send_json(json.dumps({"dates": dates}))
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/backtest/run", "/api/sim/start",
                           "/api/sim/stop"):
            return self._post_sim(parsed.path)
        if parsed.path == "/api/focus":
            from datetime import datetime as _dt
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n).decode("utf-8") if n else "{}"
                items = json.loads(body).get("items", [])
                # 规整: 只留 type/code/name 且 type+name 非空的项
                clean = {"items": [
                    {"type": it.get("type"), "code": it.get("code", ""),
                     "name": it.get("name", "")}
                    for it in items if it.get("type") and it.get("name")],
                    "updated": _dt.now().strftime("%Y%m%d %H:%M:%S")}
                f = DATA / "live" / "focus.json"
                f.parent.mkdir(exist_ok=True)
                f.write_text(json.dumps(clean, ensure_ascii=False),
                             encoding="utf-8")
                return self._send_json(json.dumps(
                    {"ok": True, "n": len(clean["items"])}))
            except Exception as e:
                return self._send_json(
                    json.dumps({"error": str(e)}, ensure_ascii=False), 500)
        self.send_response(404)
        self.end_headers()

    def _post_sim(self, path: str):
        """回测运行 / 模拟启停(异步子进程, 看板轮询状态)"""
        from datetime import datetime as _dt
        from apps.sim import write_meta
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8")
                              if n else "{}")
        except Exception as e:
            return self._send_json(
                json.dumps({"error": f"body 解析失败: {e}"},
                           ensure_ascii=False), 400)
        strat = str(body.get("strategy") or "")
        now = _dt.now()
        try:
            if path == "/api/backtest/run":
                if not (STRAT_DIR / strat / "strategy.py").exists():
                    return self._send_json(
                        json.dumps({"error": f"策略 {strat} 不存在"},
                                   ensure_ascii=False), 400)
                start = str(body.get("start") or "").replace("-", "")
                end = str(body.get("end") or "").replace("-", "")
                freq = "1d" if body.get("freq") == "1d" else "1m"
                if len(start) != 8 or len(end) != 8:
                    return self._send_json(
                        '{"error":"start/end 需 YYYYMMDD"}', 400)
                run_id = f"{strat}__bt_{now.strftime('%Y%m%d_%H%M%S')}"
                rd = RUNS_DIR / run_id
                write_meta(rd, id=run_id, kind="backtest", strategy=strat,
                           mode="replay", start=start, end=end, freq=freq,
                           capital=int(body.get("capital") or 0) or None,
                           seed_run=None,
                           created_at=now.strftime("%Y-%m-%d %H:%M:%S"),
                           status="running")
                extra = ["--run-one", strat, "--mode", "replay",
                         "--start", start, "--end", end, "--freq", freq]
                if body.get("capital"):
                    extra += ["--capital", str(int(body["capital"]))]
                return self._send_json(json.dumps(
                    _spawn_run(run_id, extra), ensure_ascii=False))
            if path == "/api/sim/start":
                if not (STRAT_DIR / strat / "strategy.py").exists():
                    return self._send_json(
                        json.dumps({"error": f"策略 {strat} 不存在"},
                                   ensure_ascii=False), 400)
                seed = body.get("seed_run") or None
                if seed and not (RUNS_DIR / seed / "meta.json").exists():
                    return self._send_json(
                        json.dumps({"error": f"起点回测 {seed} 不存在"},
                                   ensure_ascii=False), 400)
                today = now.strftime("%Y%m%d")
                run_id = f"{strat}__sim_{now.strftime('%Y%m%d_%H%M%S')}"
                rd = RUNS_DIR / run_id
                write_meta(rd, id=run_id, kind="live", strategy=strat,
                           mode="live", start=today, end=today, freq="1m",
                           capital=int(body.get("capital") or 0) or None,
                           seed_run=seed,
                           created_at=now.strftime("%Y-%m-%d %H:%M:%S"),
                           status="running")
                extra = ["--run-one", strat, "--mode", "live",
                         "--start", today, "--end", today, "--freq", "1m"]
                if seed:
                    extra += ["--seed-run", seed]
                if body.get("capital"):
                    extra += ["--capital", str(int(body["capital"]))]
                return self._send_json(json.dumps(
                    _spawn_run(run_id, extra), ensure_ascii=False))
            # /api/sim/stop
            rid = str(body.get("id") or "")
            meta = _run_meta(rid)
            if not meta:
                return self._send_json(
                    json.dumps({"error": f"run {rid} 不存在"},
                               ensure_ascii=False), 404)
            if _pid_alive(meta.get("pid")):
                os.kill(int(meta["pid"]), signal.SIGTERM)
            final = "closed" if meta.get("kind") == "live" else "cancelled"
            write_meta(RUNS_DIR / rid, status=final,
                       closed_at=now.strftime("%Y-%m-%d %H:%M:%S"))
            return self._send_json(json.dumps(
                {"ok": True, "id": rid, "status": final}))
        except Exception as e:
            return self._send_json(
                json.dumps({"error": str(e)}, ensure_ascii=False), 500)

    def _send_json(self, text: str, code: int = 200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # 静默


if __name__ == "__main__":
    print(f"看板服务: http://localhost:{PORT}")
    # 多线程: 分时接口解析大文件耗时秒级, 不能阻塞其他API/静态页
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

