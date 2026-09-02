# -*- coding: utf-8 -*-
"""研究14: 双队列前向验证(修复高开盲区+补次日收益+时段效应)

研究13的两个缺陷修复:
  1. 高开盲区: +1%首触决策点漏掉高开≥+1%的强势股。拆双队列:
     - G组(高开): 首bar收盘≥+1% → 决策=开盘, 入场=开盘后第2分钟收盘
       特征: gap/竞价量比open_vr/开盘3min动量om3/开盘回撤odip
     - L组(低拉): +1%首触(j≥6) → 决策=触板bar, 入场=T+1min收盘
       特征: r3/accel/pathvol/vr1/vtrend + 蓄势段tight/drift/volramp/base_hi
  2. 时间段问题: 补次日收益next_ret; 复核"10点前封板次日胜率更高";
     决策时刻分桶×次日收益
walk-forward: 训练≤20260430 / 测试≥20260501
输出: research/out/14_dual_oos.parquet
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
CKPT = OUT / "14_checkpoint.parquet"
START = "20250901"

BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()
sys.path.insert(0, str(BIGQMT_SRC))
os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
from bigqmt_signal_trader.xtquant_compat import configure, xtdata  # noqa: E402
configure(redis_config={"formula_server": {"failure_cooldown_seconds": 5}})

ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ALL_DAYS = sorted(ev["trade_date"].unique())
DAYS = [d for d in ALL_DAYS if d >= START]
from core.attribute import load_con2stock  # noqa: E402
c2s = load_con2stock()
UNI = sorted({c for cs in c2s.values() for c in cs
              if c.endswith((".SH", ".SZ")) and c[:2] in ("60", "68", "00", "30")})


def fetch_1m(codes, day):
    out = {}
    for i in range(0, len(codes), 80):
        try:
            res = xtdata.get_market_data_ex(
                field_list=["high", "low", "close", "volume"],
                stock_list=codes[i:i + 80], period="1m",
                start_time=day + "091500", end_time=day + "150500",
                dividend_type="none", chunk_size=0, timeout_seconds=30)
            out.update(res or {})
        except Exception as e:
            print(f"1m失败 {day} {i}: {e}", flush=True)
    return out


def fetch_1d(codes, day, count=6):
    out = {}
    for i in range(0, len(codes), 400):
        try:
            res = xtdata.get_market_data_ex(
                field_list=["close", "volume"],
                stock_list=codes[i:i + 400], period="1d", end_time=day,
                count=count, dividend_type="none", chunk_size=0,
                timeout_seconds=30)
            out.update(res or {})
        except Exception as e:
            print(f"1d失败 {day} {i}: {e}", flush=True)
    return out


def clean_day_df(df, day):
    """清洗当日1m; 兼容层可能返回残缺/异常结构, 一律防御"""
    empty = pd.DataFrame(columns=["tm", "high", "low", "close", "volume"])
    try:
        if df is None or len(df) == 0:
            return empty
        if not {"volume", "close"}.issubset(set(list(df.columns))):
            return empty
        df = pd.DataFrame(df)            # 强制转原生DataFrame
        df = df[[str(ix)[:8] == day for ix in df.index]]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        df = df[df["volume"] > 0].rename_axis("tm").reset_index(drop=False)
        return df.reset_index(drop=True)
    except Exception:
        return empty


# ---------- 1m原始序列缓存: 拉过直接读盘, 后续研究全部复用 ----------
CACHE1M = OUT / "1m_cache"
CACHE1M.mkdir(exist_ok=True)


def get_1m(codes, day):
    """{code: 当日已清洗1m DataFrame}; 磁盘缓存优先, 缺失才拉QMT"""
    cf = CACHE1M / f"1m_{day}.parquet"
    cached = {}
    if cf.exists():
        try:
            ck = pd.read_parquet(cf)
            cached = {c2: g.reset_index(drop=True)
                      for c2, g in ck.groupby("code")}
        except Exception:
            cached = {}
    miss = [c for c in codes if c not in cached]
    if miss:
        raw = fetch_1m(miss, day)
        for c2, d2 in raw.items():
            d3 = clean_day_df(d2, day)
            if len(d3):
                d3.insert(0, "code", c2)
                cached[c2] = d3
        if cached:
            pd.concat(cached.values()).to_parquet(cf, index=False)
    return {c: cached[c] for c in codes if c in cached}


def build_day(day: str) -> list:
    rows = []
    d1 = fetch_1d(UNI, day)
    cand = {}
    for c, dfd in d1.items():
        try:
            if dfd is None or len(dfd) < 2 or str(dfd.index[-1])[:8] != day:
                continue
            pre = float(dfd["close"].iloc[-2])
            close = float(dfd["close"].iloc[-1])
            if pre <= 0 or close <= 0:
                continue
            cand[c] = {"pre": pre, "close_pct": (close / pre - 1) * 100,
                       "avg5v": float(dfd["volume"].iloc[:-1].tail(5).mean())}
        except Exception:
            continue
    zt = ev[(ev["trade_date"] == day) & ~ev["is_yizi"] & ~ev["is_st"]]
    pos = {r.ts_code for r in zt.itertuples()}
    neg = set()
    for c, v in cand.items():
        r = 0.20 if c[:2] in ("30", "68") else 0.10
        lo, hi = (5.0, r * 100 - 0.3) if r == 0.10 else (8.0, r * 100 - 0.3)
        if lo <= v["close_pct"] < hi and c not in pos:
            neg.add(c)
    codes = sorted(pos | neg)
    m1 = get_1m(codes, day)
    for c in codes:
        df = m1.get(c)
        if df is None or len(df) < 20:
            continue
        v = cand.get(c)
        if not v:
            continue
        pre, avg5v = v["pre"], v["avg5v"]
        pct_s = (df["close"] / pre - 1) * 100
        lp = round(pre * (1 + (0.20 if c[:2] in ("30", "68") else 0.10)), 2)
        sealed = float(df["close"].iloc[-1]) >= lp * 0.995
        close_pct = float(pct_s.iloc[-1])
        cm20 = int(c[:2] in ("30", "68"))
        gap0 = float(pct_s.iloc[0])
        base = {"date": day, "ts_code": c, "pos": c in pos,
                "y": int(sealed), "cm20": cm20, "pre": pre,
                "close_pct": close_pct, "gap": gap0}
        if gap0 >= 1.0 and len(df) >= 5:
            # ---- G组 高开队列: 决策=开盘(09:31可见首bar), 入场=第2min收盘 ----
            entry = float(df["close"].iloc[1]) * 1.001
            after = df.iloc[2:]
            om3 = float(pct_s.iloc[3]) - gap0 if len(df) > 3 else 0.0
            hi3 = float((df["high"].iloc[:4] / pre - 1).max() * 100)
            lo3 = float((df["low"].iloc[:4] / pre - 1).min() * 100)
            odip = hi3 - float(pct_s.iloc[3]) if len(df) > 3 else 0.0
            open_vr = (float(df["volume"].iloc[0]) / (avg5v / 240)
                       if avg5v > 0 else np.nan)
            rows.append({**base, "cohort": "G", "td": 571,   # 09:31
                         "pct_d": gap0, "entry": entry,
                         "open_vr": open_vr, "om3": om3,
                         "odip": odip, "amp3": hi3 - lo3,
                         "r3": np.nan, "accel": np.nan, "pathvol": np.nan,
                         "vr1": np.nan, "vtrend": np.nan, "tight": np.nan,
                         "drift": np.nan, "volramp": np.nan,
                         "base_hi": np.nan,
                         "max_after": float((after["high"] / pre - 1).max()
                                            * 100) if len(after) else gap0,
                         "entry_ret": (float(df["close"].iloc[-1]) / entry
                                       - 1) * 100})
            continue
        # ---- L组 低拉队列: +1%首触 ----
        hit1 = df["high"] >= pre * 1.01
        if not hit1.any():
            continue
        j = int(hit1.values.argmax())
        if j < 6 or j + 1 >= len(df):
            continue
        pct_j = float(pct_s.iloc[j])
        pct_at = lambda k: float(pct_s.iloc[j - k]) if j - k >= 0 else pct_j
        p1, p3 = pct_at(1), pct_at(3)
        r3 = pct_j - p3
        accel = (pct_j - p1) - (p1 - p3)
        win = pct_s.iloc[max(0, j - 10):j + 1].values
        pathvol = float(np.diff(win).std()) if len(win) > 2 else 0.0
        emin = max(j + 1, 1)
        vr1 = (df["volume"].iloc[:j + 1].sum() / emin) / (avg5v / 240) \
            if avg5v > 0 else np.nan
        vr_ = float(df["volume"].iloc[max(0, j - 2):j + 1].sum())
        vp_ = float(df["volume"].iloc[max(0, j - 5):max(0, j - 2)].sum())
        vtrend = vr_ / vp_ if vp_ > 0 else np.nan
        w0 = max(0, j - 30)
        seg = pct_s.iloc[w0:j]
        if len(seg) >= 10:
            tight = float(seg.std())
            drift = pct_j - float(seg.iloc[0])
            vb = df["volume"].iloc[w0:j]
            ramp = (float(vb.iloc[-5:].mean()) / float(vb.iloc[:10].mean())
                    if len(vb) >= 15 and float(vb.iloc[:10].mean()) > 0
                    else np.nan)
            base_hi = float((df["high"].iloc[w0:j] / pre - 1).max() * 100)
        else:
            tight, drift, ramp, base_hi = (np.nan,) * 4
        hhmm = int(str(df["tm"].iloc[j])[8:10]) * 60 \
            + int(str(df["tm"].iloc[j])[10:12])
        entry = float(df["close"].iloc[j + 1]) * 1.001
        after = df.iloc[j + 1:]
        rows.append({**base, "cohort": "L", "td": hhmm, "pct_d": pct_j,
                     "entry": entry, "open_vr": np.nan, "om3": np.nan,
                     "odip": np.nan, "amp3": np.nan,
                     "r3": r3, "accel": accel, "pathvol": pathvol,
                     "vr1": vr1, "vtrend": vtrend, "tight": tight,
                     "drift": drift, "volramp": ramp, "base_hi": base_hi,
                     "max_after": float((after["high"] / pre - 1).max()
                                        * 100) if len(after) else pct_j,
                     "entry_ret": (float(df["close"].iloc[-1]) / entry
                                   - 1) * 100})
    return rows


# ---------- 主循环(断点续跑) ----------
done_days, rows = set(), []
if CKPT.exists():
    ck = pd.read_parquet(CKPT)
    done_days = set(ck["date"].unique())
    rows = ck.to_dict("records")
    print(f"断点续跑: 已完成{len(done_days)}天 {len(rows)}样本", flush=True)

t_start = time.time()
for di, day in enumerate(DAYS):
    if day in done_days:
        continue
    t0 = time.time()
    try:
        new = build_day(day)
    except Exception as e:
        print(f"{day} 构建失败: {e}", flush=True)
        continue
    rows.extend(new)
    done_days.add(day)
    if len(done_days) % 10 == 0:
        pd.DataFrame(rows).to_parquet(CKPT, index=False)
        print(f"进度 {day} {len(done_days)}/{len(DAYS)}天 样本{len(rows)} "
              f"累计{(time.time()-t_start)/60:.0f}min", flush=True)
    else:
        print(f"进度 {day} 样本{len(rows)} {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)

# ---------- 次日收益 ----------
print("拉取次日收盘...", flush=True)
next_of = dict(zip(ALL_DAYS[:-1], ALL_DAYS[1:]))
next_of["20260825"] = "20260826"
next_close_map = {}
for day in sorted(df["date"].unique()):
    nd = next_of.get(day)
    if not nd:
        continue
    codes_d = sorted(df[df["date"] == day]["ts_code"])
    res = fetch_1d(codes_d, nd, count=2)
    for c, d2 in res.items():
        try:
            for ix, cl in zip(d2.index, d2["close"]):
                if str(ix)[:8] == nd and float(cl) > 0:
                    next_close_map[(day, c)] = float(cl)
        except Exception:
            continue
df["next_ret"] = [
    (next_close_map[(r.date, r.ts_code)] / r.entry - 1) * 100
    if (r.date, r.ts_code) in next_close_map else np.nan
    for r in df.itertuples()]

# 封板时刻(事件库first_time), 供时段效应复核
ft = ev.set_index(["trade_date", "ts_code"])["first_time"]
df["ft"] = [ft.get((r.date, r.ts_code), None) for r in df.itertuples()]

df.to_parquet(OUT / "14_dual_oos.parquet", index=False)
if CKPT.exists():
    CKPT.unlink()
print(f"完成: {len(df)}样本 (G组{(df['cohort']=='G').sum()} "
      f"L组{(df['cohort']=='L').sum()})", flush=True)
