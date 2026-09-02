# -*- coding: utf-8 -*-
"""研究13: 前向验证 + 提前感知(触+1%决策, T+1min可执行入场)

研究12的批评回应:
  1. 事后性 → walk-forward: 训练期≤20260430拟合规则, 测试期≥20260501
     纯样本外评估; 入场价=T+1分钟bar收盘×1.001滑点(给1分钟反应)
  2. 提前感知 → 决策点前移至+1%首触; 新增启动前蓄势段特征
     (触+1%前30min: 平台紧度tight/漂移drift/量能爬坡volramp)
  3. 新因子 → 形态(平台结构)、时序(量能节奏)、题材(同题材已启动家数
     lead_cnt/当下均涨 peer_move, 抽样30个测试期交易日重建)

数据: QMT 1m全窗口(238日) + 题材状态抽样重建
输出: research/out/13_forward_{oos,theme}.parquet + 13_forward.md
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.attribute import load_con2stock  # noqa: E402

OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
CKPT = OUT / "13_checkpoint.parquet"
TRAIN_END = "20260430"      # walk-forward 分割
START = "20250901"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()
sys.path.insert(0, str(BIGQMT_SRC))
os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
from bigqmt_signal_trader.xtquant_compat import configure, xtdata  # noqa: E402
configure(redis_config={"formula_server": {"failure_cooldown_seconds": 5}})

ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
ALL_DAYS = sorted(ev["trade_date"].unique())
DAYS = [d for d in ALL_DAYS if d >= START]
c2s = load_con2stock()
stock2con = {}
for k, cs in c2s.items():
    for c in cs:
        stock2con.setdefault(c, set()).add(k)
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
    df = df[[str(ix)[:8] == day for ix in df.index]]
    df = df[df["volume"] > 0].rename_axis("tm").reset_index(drop=False)
    return df.reset_index(drop=True)


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
    m1 = fetch_1m(codes, day)
    for c in codes:
        df = m1.get(c)
        if df is None or len(df) < 20:
            continue
        df = clean_day_df(df, day)
        if len(df) < 20:
            continue
        v = cand.get(c)
        if not v:
            continue
        pre = v["pre"]
        pct_s = (df["close"] / pre - 1) * 100
        # 决策点: +1%首触
        hit1 = df["high"] >= pre * 1.01
        if not hit1.any():
            continue
        j = int(hit1.values.argmax())
        if j < 6 or j + 1 >= len(df):
            continue
        pct_j = float(pct_s.iloc[j])
        pct_at = lambda k: float(pct_s.iloc[j - k]) if j - k >= 0 else pct_j
        # --- 决策时刻特征(全部截至bar j, 无未来信息) ---
        p1, p3, p5 = pct_at(1), pct_at(3), pct_at(5)
        r3 = pct_j - p3
        accel = (pct_j - p1) - (p1 - p3)
        win = pct_s.iloc[max(0, j - 10):j + 1].values
        pathvol = float(np.diff(win).std()) if len(win) > 2 else 0.0
        emin = max(j + 1, 1)
        vr1 = (df["volume"].iloc[:j + 1].sum() / emin) / (v["avg5v"] / 240) \
            if v["avg5v"] > 0 else np.nan
        vr_ = float(df["volume"].iloc[max(0, j - 2):j + 1].sum())
        vp_ = float(df["volume"].iloc[max(0, j - 5):max(0, j - 2)].sum())
        vtrend = vr_ / vp_ if vp_ > 0 else np.nan
        hhmm = int(str(df["tm"].iloc[j])[8:10]) * 60 \
            + int(str(df["tm"].iloc[j])[10:12])
        # --- 启动前蓄势段特征(bar j 之前的30min, 完全前向可见) ---
        w0 = max(0, j - 30)
        base = pct_s.iloc[w0:j]              # 不含当前bar
        if len(base) >= 10:
            tight = float(base.std())        # 平台紧度(越小越稳)
            drift = pct_j - float(base.iloc[0])   # 蓄势段漂移
            # 量能爬坡: 决策前5bar量 / 蓄势段前段均量
            vb = df["volume"].iloc[w0:j]
            ramp = (float(vb.iloc[-5:].mean()) / float(vb.iloc[:10].mean())
                    if len(vb) >= 15 and float(vb.iloc[:10].mean()) > 0
                    else np.nan)
            base_hi = float((df["high"].iloc[w0:j] / pre - 1).max() * 100)
        else:
            tight, drift, ramp, base_hi = np.nan, np.nan, np.nan, np.nan
        # 竞价强度: 首根bar相对昨收
        gap = float(pct_s.iloc[0])
        # --- 入场: T+1分钟bar收盘(前向可执行) + 0.1%滑点 ---
        entry = float(df["close"].iloc[j + 1]) * 1.001
        after = df.iloc[j + 1:]
        lp = round(pre * (1 + (0.20 if c[:2] in ("30", "68") else 0.10)), 2)
        sealed = float(df["close"].iloc[-1]) >= lp * 0.995
        max_after = float((after["high"] / pre - 1).max() * 100) \
            if len(after) else pct_j
        close_pct = float(pct_s.iloc[-1])
        rows.append({
            "date": day, "ts_code": c, "pos": c in pos, "y": int(sealed),
            "cm20": int(c[:2] in ("30", "68")), "t1": hhmm, "pct1": pct_j,
            "pre": pre, "entry": entry,
            "r3": r3, "accel": accel, "pathvol": pathvol,
            "vr1": vr1, "vtrend": vtrend,
            "tight": tight, "drift": drift, "volramp": ramp,
            "base_hi": base_hi, "gap": gap,
            "max_after": max_after, "close_pct": close_pct,
            "entry_ret": (float(df["close"].iloc[-1]) / entry - 1) * 100,
        })
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
df.to_parquet(OUT / "13_forward_oos.parquet", index=False)
if CKPT.exists():
    CKPT.unlink()
print(f"主数据完成: {len(df)}样本", flush=True)

# ---------- 题材前向因子(抽样测试期30日重建) ----------
test_days = [d for d in sorted(df["date"].unique()) if d > TRAIN_END]
sample_days = test_days[::3][:30]
theme_rows = []
for day in sample_days:
    t0 = time.time()
    sub = df[df["date"] == day]
    touched = list(sub["ts_code"])
    if not touched:
        continue
    peers_uni = set()
    for c in touched:
        for k in stock2con.get(c, set()):
            peers_uni.update(c2s.get(k, []))
    peers_uni = sorted(c for c in peers_uni
                       if c.endswith((".SH", ".SZ"))
                       and c[:2] in ("60", "68", "00", "30"))
    pm = fetch_1m(peers_uni, day)
    # 每只peer的当日序列: pct(相对首bar收盘)与累计high
    peer_data = {}
    for c2, d2 in pm.items():
        d2 = clean_day_df(d2, day)
        if len(d2) < 20:
            continue
        base_px = float(d2["close"].iloc[0])
        if base_px <= 0:
            continue
        peer_data[c2] = {
            "tm": [str(x)[8:12] for x in d2["tm"]],
            "pct": ((d2["close"] / base_px - 1) * 100).values,
            "hip": ((d2["high"].cummax() / base_px - 1) * 100).values,
        }
    for _, r in sub.iterrows():
        c = r["ts_code"]
        # 决策分钟(首触+1%的bar时刻): 用entry前一根 — 近似用t1换算
        hh, mm = divmod(int(r["t1"]), 60)
        tm_key = f"{hh:02d}{mm:02d}"
        cons = stock2con.get(c, set())
        peers = [c2 for c2 in peer_data
                 if c2 != c and stock2con.get(c2, set()) & cons]
        if not peers:
            continue
        up_cnt, lead_cnt, moves = 0, 0, []
        for c2 in peers:
            pd2 = peer_data[c2]
            if tm_key not in pd2["tm"]:
                continue
            k = pd2["tm"].index(tm_key)
            if pd2["pct"][k] > 1.0:
                up_cnt += 1
            led = bool(pd2["hip"][k - 1] > 2.0) if k > 0 else False
            if led:
                lead_cnt += 1       # 决策前同题材已有冲过+2%的
            moves.append(pd2["pct"][k])
        if moves:
            theme_rows.append({
                "date": day, "ts_code": c,
                "peer_up": up_cnt, "peer_lead": lead_cnt,
                "peer_move": float(np.mean(moves))})
    print(f"题材 {day} peers={len(peer_data)} "
          f"{time.time()-t0:.0f}s", flush=True)

tdf = pd.DataFrame(theme_rows)
tdf.to_parquet(OUT / "13_forward_theme.parquet", index=False)
print("题材因子数据完成", flush=True)
