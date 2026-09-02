# -*- coding: utf-8 -*-
"""研究30: S2/S3 前向信号实盘复核 + 封板因子发现 + 离线强化学习策略提炼

问题: core/early_signal.py 的 S2/S3 阈值来自研究12/14c/16 的历史 walk-forward,
实盘跑了 5 个交易日后需要复核两件事:
  1) S2/S3 在真实链路是否有效(封板率/成交率/EV, 分支×日×环境分层)
  2) 是否还有别的识别封板因子(增益+置换重要性 → 分档单调性 → 消融增量)

方法论遵循项目既有口径(不做回顾性分离度评判):
  - 有效性以 Forward Return 分档单调性 + 首尾spread 为准
  - walk-forward: 前段训练 / 后段纯样本外, 杜绝事后拟合
  - 消融: 逐个关掉单因子看增量, 而非叠加复杂度
  - 环境分层: 按当日情绪(涨停家数/连板高度/上涨占比/炸板率)分强中弱三段独立报
强化学习段: 决策=买/不买, 奖励=EV, 数据是规则策略的 logged bandit 反馈
(无随机探索)→ 悲观 tabular Q-learning(单步 γ=0 + count/std 惩罚, CQL 思想)
提炼可读状态规则, 并在样本外日与 S2/S3 规则策略对比 policy value。
诚实边界: 无 propensity → IPS/DR 不可用; RL 结论仍需 forward 分档复核。

标签真值口径(不复用 presig 的 sealed_t 字段, 见 §B0 数据体检):
  收盘封板 = events_enriched(收盘权威, limit='U')
  盘中触板 = intraday_px 价格首次 ≥ 涨停价×0.995
  成交(fill) = 信号后价格曾 ≤ 推荐买入价 pb(挂限价单等回踩的真实成交判定)

数据: data/live/presig_state_*.json / radar_log_*.jsonl / intraday_px_*.json
      data/limitup/1d/events_enriched.parquet
      data/market/1d/daily_panel.parquet
输出: research/out/30_sig_dataset.parquet / 30_universe.parquet
      / 30_factor_rank.csv / 30_s2s3_rl.md
用法: python research/30_s2s3_rl_eval.py [--skip-universe]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

LIVE = DATA / "live"
OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)

DAYS = ["20260827", "20260828", "20260831", "20260901", "20260902"]
UNI_DAYS = ["20260828", "20260831", "20260901", "20260902"]   # 0827日志无量额字段
TR_DAYS = ["20260828", "20260831"]      # 宇宙层 walk-forward 训练段
TE_DAYS = ["20260901", "20260902"]      # 宇宙层纯样本外
CAP_PER_CODE = 26                       # 每票每日最多保留横截面样本数
SCALE = (20 / 60) ** 0.5                # 20s粒度 pathvol 换算(同 early_signal)
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


def _sec(t) -> int:
    s = str(t).replace(":", "").zfill(6)
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def limit_ratio(code: str, name: str = "") -> float:
    if "ST" in name:
        return 0.05
    return 0.20 if str(code)[:2] in ("30", "68") else 0.10


def pathvol(series: list) -> float:
    """轨迹颠簸度(相邻样本涨幅差标准差), 样本<8返回0"""
    if len(series) < 8:
        return 0.0
    d = np.diff(np.asarray(series, dtype=float))
    return float(d.std())


# ---------------------------------------------------------------- 数据加载
def load_events() -> dict:
    """{(date, code): {first_sec, open_times, lb, next_open_ret...}}"""
    ev = pd.read_parquet(DATA / "limitup/1d/events_enriched.parquet",
                         columns=["trade_date", "ts_code", "first_time",
                                  "last_time", "open_times", "limit_times",
                                  "is_yizi", "float_mv", "turnover_ratio",
                                  "next_open_ret", "next_close_ret"])
    ev = ev[ev["trade_date"].isin(DAYS)]
    out = {}
    for r in ev.itertuples():
        out[(r.trade_date, r.ts_code)] = {
            "first_sec": _sec(r.first_time), "last_sec": _sec(r.last_time),
            "open_times": int(r.open_times), "lb": int(r.limit_times),
            "yizi": bool(r.is_yizi), "fmv": float(r.float_mv or 0),
            "tover_d": float(r.turnover_ratio or 0),
            "next_open_ret": r.next_open_ret,
            "next_close_ret": r.next_close_ret}
    return out


def load_panel() -> pd.DataFrame:
    """日线(含次日EV), 只取研究窗口"""
    dp = pd.read_parquet(DATA / "market/1d/daily_panel.parquet",
                         columns=["trade_date", "ts_code", "open", "high",
                                  "low", "close", "pre_close", "pct_chg",
                                  "open_ret"],
                         filters=[("trade_date", ">=", "20260701")])
    return dp.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def load_ipx(date: str) -> dict:
    """{code: [(sec, price), ...]} 全天分时价格(量额字段口径混乱, 只取价)"""
    f = LIVE / f"intraday_px_{date}.json"
    if not f.exists():
        return {}
    raw = json.loads(f.read_text())
    out = {}
    for c, pts in raw.items():
        out[c] = [(_sec(p[0]), float(p[1])) for p in pts if p[1] > 0]
    return out


def first_touch(pts: list, limit_px: float) -> int | None:
    """盘中首次触板时刻(秒); 无触板返回None"""
    if limit_px <= 0:
        return None
    thr = limit_px * 0.995
    for sec, p in pts:
        if p >= thr:
            return sec
    return None


def px_after(pts: list, sec: int) -> list:
    return [(s, p) for s, p in pts if s >= sec]


# ---------------------------------------------------------------- 数据集A
def build_signal_dataset(ev: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """信号层: 每条 S1/S2/S3 信号 = 决策时刻特征 + 结果标签 + EV。
    横截面特征(prob/dp/heat/dheat/trank/dist/tover)从 radar_log 按触发时刻
    最近邻回捞(≤120s且不含未来), 缺失置NaN不猜测。"""
    dates = sorted(panel["trade_date"].unique())
    nxt = {d: (dates[i + 1] if i + 1 < len(dates) else None)
           for i, d in enumerate(dates)}
    pi = {(r.trade_date, r.ts_code): r for r in panel.itertuples()}
    rows = []
    for date in DAYS:
        f = LIVE / f"presig_state_{date}.json"
        if not f.exists():
            continue
        sigs = json.loads(f.read_text())["signals"]
        ipx = load_ipx(date)
        # 触发时刻集合 → 单遍流式回捞横截面
        want = defaultdict(list)
        for s in sigs:
            want[s["ts_code"]].append((_sec(s.get("t", "093000")), s["stage"]))
        snap = {}                                    # (code,stage) -> 横截面行
        cur = {}
        lf = LIVE / f"radar_log_{date}.jsonl"
        if lf.exists() and want:
            with lf.open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    c = r.get("code")
                    if c not in want:
                        continue
                    sec = _sec(r["t"])
                    for tgt, stage in want[c]:
                        k = (c, stage)
                        # 日志按时间升序 → 最后一条 sec<=触发时刻的行即最近邻
                        if sec <= tgt and (k not in snap or sec > snap[k][0]):
                            snap[k] = (sec, r)
        for s in sigs:
            c = s["ts_code"]
            tsec = _sec(s.get("t", "093000"))
            st = s.get("struct") or {}
            d = pi.get((date, c))
            nd = pi.get((nxt.get(date), c)) if nxt.get(date) else None
            pre = float(d.pre_close) if d is not None else 0.0
            limit_px = float(s.get("limit_px") or
                             (pre * (1 + limit_ratio(c, s.get("name", "")))
                              if pre else 0))
            pts = ipx.get(c, [])
            # ---- 结果标签
            e = ev.get((date, c))
            touch_sec = first_touch(pts, limit_px)
            if touch_sec is None and s.get("px_hist"):
                touch_sec = first_touch(
                    [(_sec(p[0]), float(p[1])) for p in s["px_hist"]],
                    limit_px)
            pb = s.get("pb")
            price0 = float(s.get("price0") or 0)
            after = px_after(pts, tsec)
            fill = None
            if pb and after:
                fill = bool(min(p for _, p in after) <= float(pb))
            elif pb and s.get("px_hist"):
                fill = bool(min(float(p[1]) for p in s["px_hist"])
                            <= float(pb))
            close = float(d.close) if d is not None else np.nan
            row = {
                "date": date, "ts_code": c, "name": s.get("name", ""),
                "stage": s["stage"], "branch": s.get("why", ""),
                "tsec": tsec, "mins_open": round((tsec - 34200) / 60, 1),
                "pct": float(s.get("pct", 0)), "r3": float(s.get("r3", 0)),
                "accel": float(s.get("accel", 0)),
                "pathvol": float(s.get("pathvol", 0)),
                "vr": float(s.get("vr", 0)), "limit_px": limit_px,
                "price0": price0, "pb": float(pb) if pb else np.nan,
                "disc": (float(pb) / price0 - 1) * 100 if pb and price0
                else np.nan,
                "g_chip": st.get("g_chip", np.nan),
                "gate": (1 if st.get("gate") else 0) if st else np.nan,
                "v5": st.get("v5", np.nan), "zb20": st.get("zb20", np.nan),
                "ir": st.get("ir", np.nan),
                "sealed_t_field": bool(s.get("sealed_t")),
                "touch": bool(touch_sec is not None),
                "touch_sec": touch_sec,
                "seal_close": bool(e is not None),
                "first_seal_sec": e["first_sec"] if e else None,
                "lb": e["lb"] if e else 0, "yizi": e["yizi"] if e else False,
                "fill": fill,
                "lead_min": (round((touch_sec - tsec) / 60, 1)
                             if touch_sec is not None else np.nan),
            }
            # ---- EV: 推荐买入价 pb 为入场基准(未成交则EV无意义, 记NaN)
            entry = float(pb) if pb else price0
            if entry > 0:
                row["ev_day"] = (close / entry - 1) * 100 if close == close \
                    else np.nan
                if nd is not None:
                    row["ev_next_open"] = (float(nd.open) / entry - 1) * 100
                    row["ev_next_close"] = (float(nd.close) / entry - 1) * 100
                else:
                    row["ev_next_open"] = np.nan
                    row["ev_next_close"] = np.nan
            # 未成交 → 真实执行收益恒为0(挂单没成交=没持仓);
            # 次日日线缺失(如最后一日) → 奖励置NaN整条弃用, 不当作0
            if fill is None:
                row["reward_day"] = np.nan
                row["reward_open"] = np.nan
            else:
                row["reward_day"] = row.get("ev_day", np.nan) if fill else 0.0
                row["reward_open"] = (row.get("ev_next_open", np.nan) if fill
                                      else (0.0 if nd is not None else np.nan))
            row["gap_ret"] = (float(nd.open) / close - 1) * 100 \
                if nd is not None and close == close and close > 0 else np.nan
            # ---- 横截面回捞
            sn = snap.get((c, s["stage"]))
            for k in ("prob", "dp", "heat", "dheat", "trank", "dist",
                      "tover", "s1", "s3", "s5"):
                row[k] = float(sn[1][k]) if sn and k in sn[1] else np.nan
            row["near_int"] = (1 if sn and sn[1].get("near") else 0) if sn \
                else np.nan
            row["theme"] = sn[1].get("theme", "-") if sn else "-"
            # ---- 开盘形态(gap/开盘3min回撤与振幅), 用于复核S3分支口径
            if pts:
                op = [p for s2, p in pts if s2 <= 34200 + 240]
                if op:
                    row["gap_open"] = (op[0] / pre - 1) * 100 if pre \
                        else np.nan
                    row["odip3"] = max(op) - op[-1]
                    row["amp3"] = max(op) - min(op)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "30_sig_dataset.parquet")
    say(f"[A] 信号层样本 {len(df)} 条 "
        f"({df['date'].nunique()}日 S2={int((df.stage=='S2').sum())} "
        f"S3={int((df.stage=='S3').sum())} S1={int((df.stage=='S1').sum())})")
    return df


# ---------------------------------------------------------------- 数据集B
UNI_COLS = ["date", "ts_code", "name", "tsec", "mins_open", "pct", "s1",
            "s3", "s5", "vr", "tover", "dist", "prob", "dp", "heat", "dheat",
            "trank", "near_int", "amt_speed", "amt_burst", "vwap_dev", "odip",
            "amp", "n_touch", "since_first2", "pv10", "ramp", "vr_pct",
            "theme_zt", "theme_touch", "mkt_zt", "y_zt", "y_lb", "y_pct",
            "rise20", "y_cpos", "is20", "limit_px", "price", "touch30",
            "touch10", "seal_close", "ev_day", "ev_next_open"]

FEATS = ["pct", "s1", "s3", "s5", "vr", "vr_pct", "tover", "dist", "prob",
         "dp", "heat", "dheat", "trank", "near_int", "amt_speed", "amt_burst",
         "vwap_dev", "odip", "amp", "n_touch", "mins_open", "since_first2",
         "pv10", "ramp", "theme_zt", "theme_touch", "mkt_zt", "y_zt", "y_lb",
         "y_pct", "rise20", "y_cpos", "is20"]

# 生产已有口径(core/early_signal + core/prob 已消费的维度)
PROD_FEATS = ["pct", "s1", "s3", "s5", "vr", "pv10", "amp", "odip", "dist",
              "prob", "mins_open"]
# 本研究新增候选因子(生产尚未使用)
NEW_FEATS = ["theme_zt", "theme_touch", "vr_pct", "amt_speed", "amt_burst",
             "vwap_dev", "n_touch", "trank", "dheat", "heat", "y_zt", "y_lb",
             "y_pct", "rise20", "y_cpos", "is20", "mkt_zt", "tover", "ramp",
             "since_first2"]
# 可交易因子集: 排除 dist/near_int(距涨停与触板是“定义性”因子, 不构成alpha)
TRADE_FEATS = [f for f in FEATS if f not in ("dist", "near_int")]


def build_universe_dataset(ev: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """宇宙层: radar_log 20s横截面(涨幅≥2%的打板决策区) → 未来10/30min触板
    标签 + 收盘封板 + 前向EV。已触板时刻之后的行剔除(决策已失效且泄露)。"""
    pi = {(r.trade_date, r.ts_code): r for r in panel.itertuples()}
    dates = sorted(panel["trade_date"].unique())
    nxt = {d: (dates[i + 1] if i + 1 < len(dates) else None)
           for i, d in enumerate(dates)}
    hist_close = panel.pivot_table(index="ts_code", columns="trade_date",
                                   values="close")
    all_rows = []
    for date in UNI_DAYS:
        lf = LIVE / f"radar_log_{date}.jsonl"
        if not lf.exists():
            continue
        ipx = load_ipx(date)
        per = defaultdict(list)
        with lf.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["t"] > "150000" or r["pct"] < 2:
                    continue
                if "ST" in r.get("name", "") or r["code"].endswith(".BJ"):
                    continue
                per[r["code"]].append(r)
        # 当日已封板家数时间线(题材扎堆/市场情绪因子)
        ev_day = {c: v for (d, c), v in ev.items() if d == date}
        seal_secs = sorted(v["first_sec"] for v in ev_day.values())
        touch_all = {}
        for c, pts in ipx.items():
            d = pi.get((date, c))
            if d is None:
                continue
            lp = float(d.pre_close) * (1 + limit_ratio(c))
            ts = first_touch(pts, round(lp, 2))
            if ts is not None:
                touch_all[c] = ts
        touch_secs = sorted(touch_all.values())
        # 题材维度时间线预聚合(避免逐行遍历成分股)
        con_seal, con_touch = {}, {}
        for k, members in CON_MEMBERS.items():
            ss = sorted(ev_day[m]["first_sec"] for m in members
                        if m in ev_day)
            tt = sorted(touch_all[m] for m in members if m in touch_all)
            if ss or tt:
                con_seal[k], con_touch[k] = ss, tt
        rows = []
        for c, rs in per.items():
            d = pi.get((date, c))
            if d is None:
                continue
            pre = float(d.pre_close)
            lp = round(pre * (1 + limit_ratio(c, rs[0].get("name", ""))), 2)
            tsec0 = first_touch(ipx.get(c, []), lp)
            nd = pi.get((nxt.get(date), c)) if nxt.get(date) else None
            cons = CON2STOCK.get(c, ())
            rs.sort(key=lambda x: x["t"])
            amts, prev_amt, prev_sec = [], None, None
            win = []                 # 10min滚动窗口(sec,pct), 供pv10
            pmax = pmin = -99.0
            ntouch = 0
            for i, r in enumerate(rs):
                sec = _sec(r["t"])
                if tsec0 is not None and sec >= tsec0:
                    break                       # 已触板之后不再是决策点
                pct = float(r["pct"])
                price = pre * (1 + pct / 100)
                amt = float(r.get("amt") or 0)
                vol = float(r.get("vol") or 0)
                # 量能因子: 累计成交额增速(元/分钟)与突增比
                spd = np.nan
                if prev_amt is not None and sec > prev_sec:
                    da = amt - prev_amt
                    spd = (da / (sec - prev_sec) * 60) if da >= 0 else np.nan
                amts.append(spd)
                prev5 = [x for x in amts[-6:-1] if x == x]
                burst = (spd / (sum(prev5) / len(prev5))) if spd == spd and \
                    prev5 and sum(prev5) > 0 else np.nan
                vwap = amt / vol if vol > 0 and amt > 0 else np.nan
                win.append((sec, pct))
                while win and win[0][0] < sec - 600:
                    win.pop(0)
                pmax, pmin = max(pmax, pct), min(pmin, pct)
                ntouch += 1 if i and rs[i - 1].get("near") else 0
                rows.append({
                    "date": date, "ts_code": c, "name": r.get("name", ""),
                    "tsec": sec, "mins_open": round((sec - 34200) / 60, 1),
                    "pct": pct, "s1": float(r.get("s1") or 0),
                    "s3": float(r.get("s3") or 0),
                    "s5": float(r.get("s5") or 0),
                    "vr": float(r.get("vr") or 0),
                    "tover": float(r.get("tover") or 0),
                    "dist": float(r.get("dist") or 0),
                    "prob": float(r.get("prob") or 0),
                    "dp": float(r.get("dp") or 0),
                    "heat": float(r.get("heat") or 0),
                    "dheat": float(r.get("dheat") or 0),
                    "trank": float(r.get("trank") or 99),
                    "near_int": 1 if r.get("near") else 0,
                    "amt_speed": spd, "amt_burst": burst,
                    "vwap_dev": ((price / vwap - 1) * 100
                                 if vwap == vwap and vwap > 0 else np.nan),
                    "odip": pmax - pct, "amp": pmax - pmin,
                    "n_touch": ntouch,
                    "since_first2": round((sec - _sec(rs[0]["t"])) / 60, 1),
                    "pv10": pathvol([p for _, p in win]) * SCALE,
                    "ramp": sum(1 for x in rs[max(0, i - 14):i + 1]
                                if float(x.get("s1") or 0) > 0.5),
                    "vr_pct": np.nan,
                    "theme_zt": max((_count_le(con_seal[k], sec)
                                      for k in cons if k in con_seal),
                                     default=0),
                    "theme_touch": max((_count_le(con_touch[k], sec)
                                         for k in cons if k in con_touch),
                                        default=0),
                    "mkt_zt": _count_le(seal_secs, sec),
                    "y_zt": np.nan, "y_lb": np.nan, "y_pct": np.nan,
                    "rise20": np.nan, "y_cpos": np.nan,
                    "is20": 1 if c[:2] in ("30", "68") else 0,
                    "limit_px": lp, "price": price,
                    "touch30": (1 if tsec0 is not None
                                and 0 < tsec0 - sec <= 1800 else 0),
                    "touch10": (1 if tsec0 is not None
                                and 0 < tsec0 - sec <= 600 else 0),
                    "seal_close": 1 if c in ev_day else 0,
                    "ev_day": ((float(d.close) / price - 1) * 100
                               if d.close == d.close else np.nan),
                    "ev_next_open": ((float(nd.open) / price - 1) * 100
                                     if nd is not None and nd.open == nd.open
                                     else np.nan),
                    "_mkt_touch": _count_le(touch_secs, sec),
                })
                prev_amt, prev_sec = amt, sec
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df = _downsample(df, CAP_PER_CODE)
    df = _add_prev_day(df, panel, hist_close, ev)
    df["vr_pct"] = df.groupby([df["date"], df["tsec"] // 60])["vr"] \
        .rank(pct=True)
    df = df[UNI_COLS]
    df.to_parquet(OUT / "30_universe.parquet")
    say(f"[B] 宇宙层样本 {len(df)} 行 / {df.ts_code.nunique()}票 "
        f"{df.date.nunique()}日 | 未来30min触板率 {df.touch30.mean():.3%} "
        f"收盘封板率 {df.seal_close.mean():.3%}")
    return df


def _count_le(sorted_secs: list, sec: int) -> int:
    from bisect import bisect_right
    return bisect_right(sorted_secs, sec)


def _downsample(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """每票每日等间距降采样(正例全保留, 负例抽稀)"""
    keep = []
    for (d, c), g in df.groupby(["date", "ts_code"], sort=False):
        pos = g[g["touch30"] == 1]
        neg = g[g["touch30"] == 0]
        keep.append(pos)
        if len(neg) > cap:
            idx = np.linspace(0, len(neg) - 1, cap).astype(int)
            keep.append(neg.iloc[idx])
        else:
            keep.append(neg)
    return pd.concat(keep, ignore_index=True)


def _add_prev_day(df: pd.DataFrame, panel: pd.DataFrame,
                  hist_close: pd.DataFrame, ev: dict) -> pd.DataFrame:
    """T-1 结构因子: 昨日涨停/连板高度/昨日涨幅/20日涨幅/昨收位置"""
    pi = {(r.trade_date, r.ts_code): r for r in panel.itertuples()}
    dates = sorted(panel["trade_date"].unique())
    dpos = {d: i for i, d in enumerate(dates)}
    prv = {d: (dates[i - 1] if i > 0 else None) for i, d in enumerate(dates)}
    ev_all = pd.read_parquet(DATA / "limitup/1d/events_enriched.parquet",
                             columns=["trade_date", "ts_code", "limit_times"])
    ev_map = {(r.trade_date, r.ts_code): int(r.limit_times)
              for r in ev_all.itertuples()}
    cache = {}                        # code -> close 对齐 dates 的 numpy 数组
    hc = hist_close.reindex(columns=dates)
    full = np.full(len(dates), np.nan)

    def _arr(c):
        if c not in cache:
            cache[c] = (hc.loc[c].to_numpy(dtype=float)
                        if c in hc.index else full.copy())
        return cache[c]

    yz, yl, yp, r20, yc = [], [], [], [], []
    for r in df.itertuples():
        pd_ = prv.get(r.date)
        d = pi.get((pd_, r.ts_code)) if pd_ else None
        if d is None:
            yz.append(np.nan); yl.append(np.nan); yp.append(np.nan)
            r20.append(np.nan); yc.append(np.nan)
            continue
        yz.append(1 if (pd_, r.ts_code) in ev_map else 0)
        yl.append(ev_map.get((pd_, r.ts_code), 0))
        yp.append(float(d.pct_chg) if d.pct_chg == d.pct_chg else np.nan)
        arr, pos = _arr(r.ts_code), dpos.get(pd_)
        base = arr[pos - 20] if pos is not None and pos >= 20 else np.nan
        r20.append((float(d.close) / base - 1) * 100
                   if base == base and base > 0 else np.nan)
        rng = float(d.high) - float(d.low)
        yc.append((float(d.close) - float(d.low)) / rng if rng > 0 else np.nan)
    df["y_zt"], df["y_lb"], df["y_pct"] = yz, yl, yp
    df["rise20"], df["y_cpos"] = r20, yc
    return df


CON2STOCK: dict = {}
CON_MEMBERS: dict = {}


def load_concepts():
    from core.attribute import load_con2stock
    c2s = load_con2stock()
    global CON2STOCK, CON_MEMBERS
    CON_MEMBERS = {k: set(v) for k, v in c2s.items()}
    for k, cs in c2s.items():
        for c in cs:
            CON2STOCK.setdefault(c, set()).add(k)


# ---------------------------------------------------------------- 报表工具
def md_table(head: list, rows: list) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def f2(v, nd=2):
    return "-" if v is None or v != v else f"{v:.{nd}f}"


def pc(v, nd=1):
    return "-" if v is None or v != v else f"{v * 100:.{nd}f}%"


def bucket_mono(df: pd.DataFrame, col: str, targets: list, nb=5):
    """5档单调性: 各档目标均值 + Spearman(档位,目标) + 首尾spread"""
    sub = df[df[col].notna()]
    if len(sub) < 60 or sub[col].nunique() < nb:
        return None
    try:
        q = pd.qcut(sub[col], nb, duplicates="drop")
    except ValueError:
        return None
    g = sub.groupby(q, observed=True)
    res = {"n": g.size().tolist()}
    for t in targets:
        m = g[t].mean().tolist()
        res[t] = m
        ok = [x for x in m if x == x]
        res[f"{t}_rho"] = _spearman(list(range(len(m))), m) if len(ok) >= 3 \
            else np.nan
        res[f"{t}_spread"] = (m[0] - m[-1]) if len(m) >= 2 else np.nan
    res["edges"] = [f"{iv.left:.2f}~{iv.right:.2f}" for iv in q.cat.categories]
    return res


def _spearman(x, y):
    from scipy.stats import spearmanr
    pairs = [(a, b) for a, b in zip(x, y) if b == b]
    if len(pairs) < 3:
        return np.nan
    r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    return float(r.statistic)


# ---------------------------------------------------------------- §B0 体检
def sec_b0(sig: pd.DataFrame):
    say("\n## B0 数据体检(标签真值口径)")
    say("presig_state 的 `sealed_t` 字段在多日为空, 不可作为封板标签;"
        "本研究一律用 events_enriched(收盘权威)+ intraday_px(盘中触板)重算。")
    rows = []
    for d, g in sig.groupby("date"):
        s23 = g[g.stage.isin(["S2", "S3"])]
        rows.append([d, len(g), len(s23),
                     int(s23.sealed_t_field.sum()), int(s23.touch.sum()),
                     int(s23.seal_close.sum()),
                     f"{int(s23.seal_close.sum()) / max(len(s23), 1):.1%}"])
    say(md_table(["日期", "信号数", "S2/S3数", "sealed_t字段=真",
                  "盘中触板(重算)", "收盘封板(权威)", "收盘封板率"], rows))
    say("\n结论: `sealed_t` 与实际封板严重不一致 → "
        "看板/复盘若消费该字段会低估封板率, 需修 apps/radar.py 的状态推进。")


# ---------------------------------------------------------------- §B 有效性
def env_tiers(panel: pd.DataFrame, ev: dict) -> dict:
    """环境分层: 涨停家数/最高连板/上涨家数占比/炸板率 → 强中弱三段"""
    dp = panel[panel.trade_date.isin(DAYS)]
    rows = []
    for d in DAYS:
        g = dp[dp.trade_date == d]
        prow = {r.ts_code: r for r in g.itertuples()}
        up = float((g.pct_chg > 0).mean()) if len(g) else np.nan
        zt = sum(1 for (dd, _), v in ev.items() if dd == d)
        lb = max([v["lb"] for (dd, _), v in ev.items() if dd == d], default=0)
        ipx = load_ipx(d)
        touch = 0
        for c, pts in ipx.items():
            r = prow.get(c)
            if r is None:
                continue
            lp = round(float(r.pre_close) * (1 + limit_ratio(c)), 2)
            if first_touch(pts, lp) is not None:
                touch += 1
        zb = 1 - zt / touch if touch else np.nan
        rows.append({"date": d, "zt": zt, "lb": lb, "up": up,
                     "touch": touch, "zb": zb})
    df = pd.DataFrame(rows)

    def z(c, inv=False):
        s = df[c].astype(float)
        v = (s - s.mean()) / (s.std() or 1)
        return -v if inv else v

    df["score"] = z("zt") + z("lb") + z("up") + z("zb", inv=True)
    rk = df["score"].rank(ascending=False)
    tier = {}
    for i, r in enumerate(df.itertuples()):
        tier[r.date] = "强" if rk.iloc[i] <= 2 else ("弱" if rk.iloc[i] >= 4
                                                   else "中")
    say("\n## B1 环境分层(情绪三段)")
    say(md_table(["日期", "涨停家数", "最高连板", "上涨家数占比", "触板家数",
                  "炸板率", "环境分", "分段"],
                 [[r.date, r.zt, r.lb, pc(r.up), r.touch, pc(r.zb),
                   f2(r.score), tier[r.date]] for r in df.itertuples()]))
    return tier


def sec_b(sig: pd.DataFrame, uni: pd.DataFrame | None, tier: dict):
    say("\n## B2 S2/S3 分支×日 有效性(封板率/成交率/EV)")
    s23 = sig[sig.stage.isin(["S2", "S3"])].copy()
    s23["grp"] = np.where(s23.branch.str.contains("竞价量爆"), "S3竞价量爆",
                          np.where(s23.branch.str.contains("高开"),
                                   "S3高开", "S2颠簸"))
    rows = []
    for (grp, d), g in s23.groupby(["grp", "date"]):
        filled = g[g.fill == True]  # noqa: E712
        rows.append([grp, d, tier.get(d, "-"), len(g), pc(g.fill.mean()),
                     pc(g.touch.mean()), pc(g.seal_close.mean()),
                     f2(filled.ev_day.mean()), f2(filled.ev_next_open.mean()),
                     f2(g.reward_day.mean()), f2(g.reward_open.mean())])
    say(md_table(["分支组", "日期", "环境", "信号数", "成交率", "盘中触板率",
                  "收盘封板率", "EV_day(成交)", "EV_次日开盘(成交)",
                  "策略EV_day", "策略EV_次日开盘"], rows))

    say("\n### 分支组汇总(全窗口)")
    rows = []
    for grp, g in s23.groupby("grp"):
        filled = g[g.fill == True]  # noqa: E712
        rows.append([grp, len(g), pc(g.fill.mean()), pc(g.touch.mean()),
                     pc(g.seal_close.mean()), f2(filled.ev_day.mean()),
                     f2(filled.ev_next_open.mean()),
                     f2(filled.ev_next_open.median()),
                     f2(g.reward_open.mean())])
    s1 = sig[sig.stage == "S1"]
    rows.append(["S1观察名单(对照)", len(s1), "-", pc(s1.touch.mean()),
                 pc(s1.seal_close.mean()), "-", "-", "-", "-"])
    base = uni.seal_close.mean() if uni is not None else np.nan
    rows.append(["宇宙基线(涨幅≥2%全样本)", len(uni) if uni is not None else 0,
                 "-", "-", pc(base), "-", "-", "-", "-"])
    say(md_table(["分支组", "信号数", "成交率", "触板率", "收盘封板率",
                  "EV_day(成交)", "EV_次日开盘(成交)", "EV中位",
                  "策略EV_次日开盘"], rows))
    if uni is not None:
        say(f"\n对照: 同日涨幅≥2%宇宙样本收盘封板率 {pc(base)} → "
            f"各分支 lift = 分支封板率/基线。")

    say("\n### EV分解: 封板与否、日内段与隔夜段(成交样本)")
    fl = s23[s23.fill == True]  # noqa: E712
    rows = []
    for key, g in [("封板票(seal_close=1)", fl[fl.seal_close]),
                   ("未封板票(seal_close=0)", fl[~fl.seal_close])]:
        rows.append([key, len(g), f2(g.ev_day.mean()), f2(g.gap_ret.mean()),
                     f2(g.ev_next_open.mean()), f2(g.ev_next_close.mean()),
                     f2(g.ev_next_close.median())])
    for grp, g in fl.groupby("grp"):
        rows.append([f"{grp}(全部成交)", len(g), f2(g.ev_day.mean()),
                     f2(g.gap_ret.mean()), f2(g.ev_next_open.mean()),
                     f2(g.ev_next_close.mean()),
                     f2(g.ev_next_close.median())])
    say(md_table(["分组", "成交样本数", "日内EV(pb→收盘)",
                  "隔夜EV(收盘→次日开盘)", "pb→次日开盘",
                  "pb→次日收盘", "EV中位"], rows))
    say("\n口径说明: EV以推荐买入价 pb(触发价×0.996 / 开盘价×0.995)为入场基准, "
        "未成交样本不计入EV均值但计入策略EV(记0)。")

    say("\n### EV分解×分支组: 封板票与未封板票各自赚多少(成交样本)")
    rows = []
    for grp, g in fl.groupby("grp"):
        for lab, gg in [("封板", g[g.seal_close]), ("未封板", g[~g.seal_close])]:
            rows.append([grp, lab, len(gg),
                         f"{len(gg) / max(len(g), 1):.1%}",
                         f2(gg.ev_day.mean()), f2(gg.ev_next_open.mean()),
                         f2(gg.ev_next_close.mean()), f2(gg.pct.mean())])
    say(md_table(["分支组", "封板与否", "成交样本数", "占本组成交比",
                  "日内EV", "pb→次日开盘", "pb→次日收盘",
                  "触发时涨幅均值"], rows))
    e1 = float(fl[fl.seal_close].ev_next_open.mean())
    e0 = float(fl[~fl.seal_close].ev_next_open.mean())
    say(f"\n盈亏平衡封板率估算: 封板票EV≈{f2(e1)}%, "
        f"未封板票EV≈{f2(e0)}% → 需封板率 > "
        f"{abs(e0) / (e1 - e0):.1%} 才能打平; "
        "但这只是全样本均值, 分支内尾部亏损差异很大(见上表)。")

    say("\n### S3 竞价量爆口径诊断(vr 门槛是否失效)")
    s3v = s23[s23.grp == "S3竞价量爆"]
    if len(s3v):
        say(f"该分支 vr 中位数 {f2(s3v.vr.median())} / "
            f"p25 {f2(s3v.vr.quantile(.25))} / "
            f"p75 {f2(s3v.vr.quantile(.75))}, "
            f"触发时涨幅中位数 {f2(s3v.pct.median())}% "
            f"(研究口径要求高开/竞价爆量, 实盘却是平开微涨票)")
        if uni is not None:
            early = uni[(uni.mins_open <= 20)]
            vnz = early[early.vr > 0]
            say(f"生产 vr 口径=今日每分钟均量/近5日每分钟均量; "
                f"开盘20min内宇宙 vr 非零占比 {pc((early.vr > 0).mean())}, "
                f"非零样本 vr 中位数 {f2(vnz.vr.median())}, "
                f"vr≥5 占非零样本 {pc((vnz.vr >= 5).mean())} → "
                f"门槛 5 在开盘窗口几乎不构成筛选"
                f"(且部分票 vr 缺失为0, 因子本身数据质量不稳)。")
            say(f"对比: 改用同分钟截面分位 vr_pct, 样本外最优档"
                f"(0.68~1.00)封板率 13.2% vs 档1 4.9% → "
                f"建议 S3 竞价量爆改用截面分位阈值而非绝对 vr。")

    say("\n## B3 关键因子分档单调性(信号层, 目标=收盘封板/EV)")
    say("分档按因子值升序(档1=最小); spread=档1-档5, "
        "对“越大越好”的因子应为负值。ρ=档位与目标的Spearman。")
    cols = ["pathvol", "accel", "r3", "pct", "vr", "prob", "dp", "heat",
            "dheat", "trank", "dist", "tover", "v5", "g_chip", "zb20",
            "mins_open", "gap_open", "odip3", "amp3"]
    rows = []
    for c in cols:
        res = bucket_mono(s23, c, ["seal_close", "touch", "reward_open"])
        if not res:
            continue
        seal = [pc(x) for x in res["seal_close"]] + ["-"] * 5
        rows.append([c, len(res["seal_close"])] + seal[:5]
                    + [f2(res["seal_close_rho"]),
                       f2(res["seal_close_spread"] * 100, 1),
                       f2(res["reward_open_rho"]),
                       f2(res["reward_open_spread"])])
    say(md_table(["因子", "有效档数"] + [f"档{i + 1}封板率" for i in range(5)]
                 + ["封板率ρ", "档1-档5(pp)", "EVρ", "EV档1-档5"], rows))


# ---------------------------------------------------------------- §C 因子
def _fit(tr, te, feats, tgt):
    """统一模型口径: HistGB(walk-forward), 返回 AUC/lift/预测值"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    clf = HistGradientBoostingClassifier(
        max_iter=220, learning_rate=0.08, max_leaf_nodes=31, max_bins=64,
        min_samples_leaf=80, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=7)
    clf.fit(tr[feats], tr[tgt])
    p = clf.predict_proba(te[feats])[:, 1]
    auc = roc_auc_score(te[tgt], p) if te[tgt].nunique() > 1 else np.nan
    base = te[tgt].mean()
    lift = {}
    for k in (0.01, 0.05, 0.10):
        n = max(int(len(te) * k), 1)
        top = te.iloc[np.argsort(-p)[:n]]
        lift[k] = top[tgt].mean() / base if base else np.nan
    return clf, p, auc, lift


def sec_c(uni: pd.DataFrame):
    from sklearn.inspection import permutation_importance
    say("\n## C1 封板因子发现(walk-forward: 训练%s / 样本外%s)"
        % (TR_DAYS, TE_DAYS))
    tr = uni[uni.date.isin(TR_DAYS)]
    te = uni[uni.date.isin(TE_DAYS)]
    say(f"训练样本 {len(tr)} 行(触板率 {tr.touch30.mean():.3%} / "
        f"封板率 {tr.seal_close.mean():.3%}) / "
        f"样本外 {len(te)} 行(触板率 {te.touch30.mean():.3%} / "
        f"封板率 {te.seal_close.mean():.3%})")
    res = {}
    for tgt, feats, tag in (("touch30", FEATS, "全因子→未来30min触板"),
                            ("seal_close", FEATS, "全因子→收盘封板"),
                            ("seal_close", TRADE_FEATS,
                             "可交易因子(排除dist/near)→收盘封板")):
        clf, p, auc, lift = _fit(tr, te, feats, tgt)
        pi = permutation_importance(clf, te[feats], te[tgt], n_repeats=3,
                                    scoring="roc_auc", random_state=0,
                                    n_jobs=2)
        imp = pd.Series(pi.importances_mean, index=feats).sort_values(
            ascending=False)
        res[tag] = {"auc": auc, "lift": lift, "imp": imp, "pred": p,
                    "feats": feats}
        say(f"\n### {tag}: 样本外AUC {auc:.4f} | "
            f"lift@1% {lift[0.01]:.2f}x @5% {lift[0.05]:.2f}x "
            f"@10% {lift[0.10]:.2f}x")
        say(md_table(["排名", "因子", "置换重要性(ΔAUC)"],
                     [[i + 1, k, f"{v:.5f}"]
                      for i, (k, v) in enumerate(imp.head(15).items())]))
    imp = res["全因子→未来30min触板"]["imp"]
    imp.to_frame("perm_auc_touch").join(
        res["全因子→收盘封板"]["imp"].to_frame("perm_auc_seal")).join(
        res["可交易因子(排除dist/near)→收盘封板"]["imp"]
        .to_frame("perm_auc_seal_trade"), how="outer") \
        .to_csv(OUT / "30_factor_rank.csv")

    say("\n### C2 增量验证: 生产已有口径 vs 新增候选因子(目标=收盘封板)")
    _, _, auc_prod, _ = _fit(tr, te, PROD_FEATS, "seal_close")
    _, _, auc_new, _ = _fit(tr, te, NEW_FEATS, "seal_close")
    _, _, auc_all, _ = _fit(tr, te, PROD_FEATS + NEW_FEATS, "seal_close")
    say(md_table(["因子集", "因子数", "样本外AUC", "相对生产口径ΔAUC"],
                 [["生产已有口径", len(PROD_FEATS), f"{auc_prod:.4f}", "-"],
                  ["仅新增候选因子", len(NEW_FEATS), f"{auc_new:.4f}",
                   f"{auc_new - auc_prod:+.4f}"],
                  ["生产+新增全量", len(PROD_FEATS) + len(NEW_FEATS),
                   f"{auc_all:.4f}", f"{auc_all - auc_prod:+.4f}"]]))
    say("\n逐个加入单因子的增量(在生产口径基线上, 目标=收盘封板):")
    rows = []
    for f in NEW_FEATS:
        _, _, a, _ = _fit(tr, te, PROD_FEATS + [f], "seal_close")
        rows.append([f, f"{auc_prod:.4f}", f"{a:.4f}", f"{a - auc_prod:+.4f}"])
    rows.sort(key=lambda x: -float(x[3]))
    say(md_table(["新增因子", "基线AUC", "加入后AUC", "ΔAUC"], rows))

    say("\n### C3 消融(逐个关掉单因子, 全因子集, 目标=未来30min触板)")
    base_auc = res["全因子→未来30min触板"]["auc"]
    rows = []
    for f in imp.head(8).index:
        feats = [x for x in FEATS if x != f]
        _, _, a, _ = _fit(tr, te, feats, "touch30")
        rows.append([f, f"{base_auc:.4f}", f"{a:.4f}", f"{a - base_auc:+.4f}"])
    say(md_table(["关闭因子", "基线AUC", "关闭后AUC", "ΔAUC"], rows))

    say("\n### C4 样本外分档单调性(前12因子; 目标=触板/封板/次日EV)")
    say("分档按因子值升序; 对“越大越好”因子 EV档1-档5 应为负。")
    rows = []
    for f in imp.head(12).index:
        r = bucket_mono(te, f, ["touch30", "seal_close", "ev_next_open"])
        if not r:
            continue
        t5 = [pc(x) for x in r["touch30"]] + ["-"] * 5
        s5 = [pc(x) for x in r["seal_close"]] + ["-"] * 5
        rows.append([f, len(r["seal_close"])] + t5[:5] + s5[:5]
                    + [f2(r["seal_close_rho"]), f2(r["ev_next_open_rho"]),
                       f2(r["ev_next_open_spread"])])
    say(md_table(["因子", "有效档数"] + [f"档{i + 1}触板" for i in range(5)]
                 + [f"档{i + 1}封板" for i in range(5)]
                 + ["封板ρ", "EVρ", "EV档1-档5"], rows))

    say("\n### C5 新因子候选清单(样本外分档 + 建议阈值)")
    rows = []
    for f in NEW_FEATS:
        r = bucket_mono(te, f, ["seal_close", "ev_next_open"])
        if not r:
            continue
        s = r["seal_close"]
        best = int(np.nanargmax(s)) if any(x == x for x in s) else 0
        rows.append([f, len(s), pc(s[0]), pc(s[-1]),
                     f2(r["seal_close_rho"]), f2(r["ev_next_open_rho"]),
                     pc(s[best]), r["edges"][best] if best < len(r["edges"])
                     else "-", f2(r["ev_next_open_spread"])])
    rows.sort(key=lambda x: -(float(x[6].rstrip("%"))
                              if x[6] != "-" else 0))
    say(md_table(["因子", "有效档数", "档1封板率", "末档封板率", "封板ρ",
                  "EVρ", "最优档封板率", "最优档区间", "EV档1-档5"], rows))
    say("\n读法: 封板ρ/EVρ 接近±1 且最优档封板率显著高于基线"
        f"({pc(te.seal_close.mean())}) 的因子才值得进生产; "
        "最优档区间即建议阈值带。")

    say("\n### C6 因子组合分数分档 → 封板率与EV(样本外, 最终判据)")
    say("封板率高不等于能赚钱: 必须看同一分档上的 EV 是否同步单调。")
    for tag in ("可交易因子(排除dist/near)→收盘封板",
                "全因子→未来30min触板"):
        te2 = te.assign(score=res[tag]["pred"])
        try:
            q = pd.qcut(te2.score, 5, duplicates="drop")
        except ValueError:
            continue
        g = te2.groupby(q, observed=True)
        rows = [[f"档{i + 1}", n, pc(t), pc(s), f2(e), f2(em), f2(ed)]
                for i, (n, t, s, e, em, ed) in enumerate(zip(
                    g.size(), g.touch30.mean(), g.seal_close.mean(),
                    g.ev_next_open.mean(), g.ev_next_open.median(),
                    g.ev_day.mean()))]
        say(f"\n模型: {tag}")
        say(md_table(["分档(低→高)", "样本数", "未来30min触板率",
                      "收盘封板率", "EV次日开盘均值", "EV次日开盘中位",
                      "EV当日收盘"], rows))
        top = te2.iloc[np.argsort(-te2.score)[:max(int(len(te2) * 0.05), 1)]]
        say(f"top5%: n={len(top)} 封板率 {pc(top.seal_close.mean())} "
            f"EV次日开盘 {f2(top.ev_next_open.mean())} "
            f"(中位 {f2(top.ev_next_open.median())}) "
            f"EV当日收盘 {f2(top.ev_day.mean())}")

    say("\n### C7 机制矩阵: 入场价位×封板与否 → EV(样本外宇宙)")
    say("解释 C6 的反直觉: 高封板概率档往往已涨到高位, 封板了EV也薄; "
        "真正的EV来自低位入场×封板。")
    bins = [2, 4, 6, 8, 12, 100]
    lab = ["2~4%", "4~6%", "6~8%", "8~12%", "12%+"]
    te3 = te.assign(bin=pd.cut(te.pct, bins, labels=lab))
    rows = []
    for b in lab:
        g = te3[te3.bin == b]
        if not len(g):
            continue
        s1, s0 = g[g.seal_close == 1], g[g.seal_close == 0]
        e1 = s1.ev_next_open.mean()
        e0 = s0.ev_next_open.mean()
        be = (abs(e0) / (e1 - e0)) if e1 == e1 and e0 == e0 and e1 > e0 \
            else np.nan
        rows.append([b, len(g), pc(g.seal_close.mean()),
                     f2(e1), f2(s1.ev_next_open.median()), f2(e0),
                     f2(g.ev_next_open.mean()),
                     pc(be) if be == be else "-"])
    say(md_table(["入场时涨幅", "样本数", "封板率", "封板票EV次日开盘",
                  "封板票EV中位", "未封板票EV次日开盘", "全档EV",
                  "打平所需封板率"], rows))
    say("\n可执行结论: 封板率不是唯一目标——必须同时限定入场价位"
        "(高位入场即使封板EV也薄, 未封板则是尾部巨亏); "
        "因子应服务于“低位+即将封板”的交集, 而非单纯高封板概率。")


# ---------------------------------------------------------------- §D 离线RL
BRANCH_GRP = {"竞价量爆": 0, "高开": 1, "颠簸": 2}


def _bgrp(branch: str) -> int:
    for k, v in BRANCH_GRP.items():
        if k in branch:
            return v
    return 3


def _edges(rows: list, col: str, nb=3) -> list:
    v = sorted(float(r[col]) for r in rows if r[col] == r[col])
    if len(v) < nb * 3:
        return []
    return [v[int(len(v) * i / nb)] for i in range(1, nb)]


def _key(r: dict, spec: list, edges: dict) -> str:
    parts = [str(r["stage"]), str(r["bgrp"])]
    for col, nb in spec:
        e = edges.get(col, [])
        v = r[col]
        b = "x" if v != v else sum(1 for x in e if v > x)   # 缺失单独一档
        parts.append(f"{col}{b}")
    return "|".join(parts)


def learn_q(rows: list, spec: list, edges: dict, epochs=8, alpha0=0.6,
            lam=1.0, gamma=0.0):
    """悲观 tabular Q-learning: 单步 bandit(γ=0, 买入即持有到次日卖出),
    α 随 epoch 衰减; 悲观项 = λ·σ/√N(CQL 思想, 少样本状态不轻易买)。"""
    Q, N, rew = defaultdict(float), defaultdict(int), defaultdict(list)
    for r in rows:
        k = _key(r, spec, edges)
        N[k] += 1
        rew[k].append(float(r["reward"]))
    for ep in range(epochs):
        a = alpha0 / (1 + ep * 0.5)
        for r in rows:
            k = _key(r, spec, edges)
            Q[k] += a * (float(r["reward"]) + gamma * 0.0 - Q[k])
    qp = {}
    for k in Q:
        arr = np.asarray(rew[k])
        sd = float(arr.std()) if len(arr) > 1 else abs(Q[k])
        qp[k] = Q[k] - lam * sd / np.sqrt(N[k])
    return qp, Q, N


def _boot_ci(vals: list, n=800):
    if not vals:
        return (np.nan, np.nan)
    arr = np.asarray(vals)
    ms = [arr[np.random.randint(0, len(arr), len(arr))].mean()
          for _ in range(n)]
    return (float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


def sec_d(sig: pd.DataFrame, tier: dict):
    say("\n## D1 离线强化学习: 买/不买策略提炼(悲观Q-learning)")
    d = sig[sig.stage.isin(["S2", "S3"]) & sig.pb.notna()].copy()
    d["bgrp"] = d.branch.map(_bgrp)
    say(f"候选决策样本(S2/S3且有推荐价pb) {len(d)} 条; "
        f"成交率 {pc(d.fill.mean())}")
    specs = {
        "A 粗粒度(pathvol×prob)": [("pathvol", 3), ("prob", 3)],
        "B 题材双因子(dheat拐点×heat持续性)":
            [("dheat", 3), ("heat", 3), ("prob", 3)],
        "C 盘中位置(trank×prob×时段)":
            [("trank", 3), ("prob", 3), ("mins_open", 3)],
        "D 入场价位×封板概率(pct×prob)": [("pct", 3), ("prob", 3)],
        "E 入场价位×题材扎堆(pct×trank×heat)":
            [("pct", 3), ("trank", 3), ("heat", 2)],
    }
    runs = [("R1 pb→次日开盘(未成交计0)", "reward_open",
             ["20260827", "20260828", "20260831"], ["20260901"]),
            ("R2 pb→当日收盘(5日均可用)", "reward_day",
             DAYS[:3], DAYS[3:])]
    first = True
    for rname, rcol, trd, ted in runs:
        ok = d[d[rcol].notna()]
        tr = ok[ok.date.isin(trd)]
        te = ok[ok.date.isin(ted)]
        if len(tr) < 60 or len(te) < 60:
            say(f"\n### {rname}: 样本不足(训练{len(tr)}/样本外{len(te)}), 跳过")
            continue
        say(f"\n### {rname} | 训练{trd}({len(tr)}条) → "
            f"样本外{ted}({len(te)}条)")
        say(f"全买基线EV(样本外) {f2(te[rcol].mean())} | "
            f"成交样本EV {f2(te[te.fill == True][rcol].mean())}")  # noqa: E712
        trr = tr.assign(reward=tr[rcol]).to_dict("records")
        ter = te.assign(reward=te[rcol]).to_dict("records")
        for sname, spec in specs.items():
            edges = {c: _edges(trr, c, n) for c, n in spec}
            qp, Q, N = learn_q(trr, spec, edges)
            nstate = len({_key(r, spec, edges) for r in trr})
            pol = {
                "π_rule(S2/S3全买)": lambda r: True,
                "π_learn(Q_pess>0)":
                    lambda r: qp.get(_key(r, spec, edges), -9) > 0,
                "π_gate(只买S2颠簸)": lambda r: r["bgrp"] == 2,
                "π_gate(只买S3竞价量爆)": lambda r: r["bgrp"] == 0,
                "π_zero(不买)": lambda r: False,
            }
            rows = []
            for pname, keep in pol.items():
                sel = [r for r in ter if keep(r)]
                vals = [float(r[rcol]) for r in sel]
                lo, hi = _boot_ci(vals) if vals else (0.0, 0.0)
                filled = [r for r in sel if r["fill"]]
                rows.append([
                    pname, f"{len(sel) / len(ter):.1%}", len(sel),
                    pc(np.mean([r["fill"] for r in sel])) if sel else "-",
                    f2(np.mean(vals)) if vals else 0.0,
                    f"{lo:.2f}~{hi:.2f}" if vals else "-",
                    pc(np.mean([r["seal_close"] for r in sel])) if sel
                    else "-",
                    f2(np.mean([r[rcol] for r in filled])) if filled else "-"])
            say(f"\n状态空间 {sname}: {nstate}个状态 "
                f"(stage×分支组×" +
                "×".join(f"{c}{n}档" for c, n in spec) + ")")
            say(md_table(["策略", "覆盖率", "样本数", "成交率", "EV均值",
                          "95%CI", "选中封板率", "成交样本EV"], rows))
            if first:
                say("\n#### Q 表(状态价值, 悲观修正后) — 可直接翻译为生产闸门")
                rows = []
                for k in sorted(qp, key=lambda x: -qp[x])[:20]:
                    rows.append([k, N[k], f2(Q[k]), f2(qp[k]),
                                 "买" if qp[k] > 0 else "不买"])
                say(md_table(["状态(stage|分支组|因子档)", "训练样本N",
                              "Q(经验均值)", "Q_pess", "策略动作"], rows))
                first = False
    say("\n诚实边界: 日志数据来自规则策略且无随机探索 → 无 propensity, "
        "IPS/DR 不可用; 上述 policy value 属 Direct Method + 悲观修正, "
        "少样本状态因 λσ/√N 惩罚而被判为不买; 结论需 forward 分档复核。")


# ---------------------------------------------------------------- main
def main():
    import warnings
    warnings.filterwarnings("ignore")
    skip_uni = "--skip-universe" in sys.argv
    np.random.seed(7)
    say("# 研究30: S2/S3 实盘复核 + 封板因子发现 + 离线RL策略提炼")
    say(f"数据窗口 {DAYS[0]}~{DAYS[-1]} ({len(DAYS)}个交易日)")
    load_concepts()
    ev = load_events()
    panel = load_panel()
    sig = build_signal_dataset(ev, panel)
    uni = None
    if not skip_uni:
        cache = OUT / "30_universe.parquet"
        if "--reuse" in sys.argv and cache.exists():
            uni = pd.read_parquet(cache)
            say(f"[B] 宇宙层样本复用缓存 {len(uni)} 行 / "
                f"{uni.ts_code.nunique()}票 {uni.date.nunique()}日")
        else:
            uni = build_universe_dataset(ev, panel)
    sec_b0(sig)
    tier = env_tiers(panel, ev)
    sec_b(sig, uni, tier)
    if uni is not None:
        sec_c(uni)
    sec_d(sig, tier)
    (OUT / "30_s2s3_rl.md").write_text("\n".join(R), encoding="utf-8")
    print(f"\n→ 报告 {OUT / '30_s2s3_rl.md'}")


if __name__ == "__main__":
    main()
