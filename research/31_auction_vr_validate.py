# -*- coding: utf-8 -*-
"""研究31: 竞价量比闸的历史验证(stk_auction 窗口, 起点2025-01-20)

验证对象: apps/radar 的竞价质量闸 —— 竞价涨幅≥1%(S1候选域)内, 竞价量比
横截面分位≥0.90 过闸。闸当前只作用于展示层不拦截触发, 但既然长期挂在
信号上, 必须先证明它筛出来的票确实更好。

方法论(用户定稿口径):
  · 三段市况独立验证(牛/熊/震荡), 方向一致性 ≥2/3 环境同梯度才算通过
  · 多方案并行对比, 不做单方案调参
  · Forward Return 分档单调性(5档) + spread, 不用回顾性分离度

市况分段用数据驱动三分位(按月均"全A中位涨幅"排序取上/中/下三段),
不设人为阈值。

产物: data/factor/auction_panel.parquet(逐日缓存, 可续跑)
      research/out/auction_vr_validate.txt
用法: python research/31_auction_vr_validate.py [--refresh]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, get_pro  # noqa: E402

OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
CACHE = DATA / "factor" / "auction_panel.parquet"
START = "20250120"          # stk_auction 历史起点(实测)

# 待对比方案(多方案并行, 用户方法论要求)
SCHEMES = [
    ("A 分位≥0.90(现设计)", lambda d: d["vrp"] >= 0.90),
    ("B 分位≥0.95", lambda d: d["vrp"] >= 0.95),
    ("C 绝对vr≥2", lambda d: d["vr"] >= 2.0),
    ("D 涨幅≥2%×分位≥0.90", lambda d: (d["gap"] >= 2.0) & (d["vrp"] >= 0.90)),
]


def build_panel(pro, refresh: bool) -> pd.DataFrame:
    """逐日拉 stk_auction 缓存成 parquet; 已缓存日期跳过(可续跑)"""
    old = pd.read_parquet(CACHE) if CACHE.exists() and not refresh else None
    have = set(old["date"].astype(str)) if old is not None else set()
    cal = pro.trade_cal(exchange="SSE", start_date=START,
                        end_date=pd.Timestamp.now().strftime("%Y%m%d"),
                        is_open="1")
    days = [d for d in sorted(cal["cal_date"]) if d not in have]
    print(f"竞价面板: 已缓存{len(have)}日, 待拉{len(days)}日")
    rows = []
    for i, date in enumerate(days):
        try:
            df = pro.stk_auction(trade_date=date)
        except Exception as e:
            print(f"  {date} 拉取失败: {e}")
            time.sleep(1)
            continue
        if df is None or df.empty:
            continue
        df = df[~df["ts_code"].str.endswith(".BJ")].copy()
        df["date"] = date
        df = df.rename(columns={"volume_ratio": "vr", "vol": "vol",
                                "turnover_rate": "tover"})
        rows.append(df[["date", "ts_code", "price", "pre_close", "vr",
                        "vol", "amount", "tover"]])
        if i % 40 == 0:
            print(f"  {i + 1}/{len(days)} {date}")
        time.sleep(0.15)
    new = pd.concat(rows, ignore_index=True) if rows else \
        pd.DataFrame(columns=["date", "ts_code", "price", "pre_close", "vr",
                              "vol", "amount", "tover"])
    out = pd.concat([old, new], ignore_index=True) if old is not None else new
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    print(f"竞价面板落盘 {len(out)}行 {out['date'].nunique()}日")
    return out


def limit_ratio(code: str) -> float:
    return 0.20 if code.startswith(("30", "68")) else 0.10


def join_panel(auc: pd.DataFrame) -> pd.DataFrame:
    """挂当日封板结果与次日 Forward Return"""
    dp = pd.read_parquet(DATA / "market" / "1d" / "daily_panel.parquet")
    dp["date"] = dp["trade_date"].astype(str)
    dp = dp[dp["date"] >= START]
    dp = dp.sort_values(["ts_code", "date"])
    dp["nx_open"] = dp.groupby("ts_code")["open"].shift(-1)
    dp["nx_close"] = dp.groupby("ts_code")["close"].shift(-1)
    d = auc.merge(dp[["date", "ts_code", "open", "high", "low", "close",
                      "pre_close", "pct_chg", "nx_open", "nx_close"]],
                  on=["date", "ts_code"], how="inner", suffixes=("_a", ""))
    d = d[d["pre_close"] > 0].copy()
    d["gap"] = (d["price"] / d["pre_close"] - 1) * 100
    d["ratio"] = d["ts_code"].map(limit_ratio)
    d["sealed"] = d["pct_chg"] >= d["ratio"] * 100 * 0.98
    # Forward Return: 次日开盘溢价 / 次日收盘收益(均以今日收盘为基准)
    d["fr_open"] = (d["nx_open"] / d["close"] - 1) * 100
    d["fr_close"] = (d["nx_close"] / d["close"] - 1) * 100
    # 竞价量比横截面分位: 每日在 gap≥1% 且 vr>0 域内排名(与生产同口径)
    dom = d[(d["gap"] >= 1.0) & (d["vr"] > 0)]
    d["vrp"] = dom.groupby("date")["vr"].rank(pct=True)
    return d.dropna(subset=["fr_close"])


def regimes(d: pd.DataFrame) -> dict:
    """按月均"全A中位涨幅"三分位切牛/熊/震荡(数据驱动, 不设人为阈值)"""
    mr = d.groupby("date")["pct_chg"].median().rename("mret")
    mon = mr.groupby(mr.index.str[:6]).mean().sort_values()
    n = len(mon)
    bear = set(mon.index[:n // 3])
    bull = set(mon.index[-n // 3:])
    return {dt: ("熊" if dt[:6] in bear else "牛" if dt[:6] in bull else "震荡")
            for dt in d["date"]}


def cohort_stat(x: pd.DataFrame) -> tuple:
    """(样本数, 封板率%, 次日开盘胜率%, 次日开盘溢价均值%, 次日收盘收益均值%)"""
    if not len(x):
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (len(x), x["sealed"].mean() * 100,
            (x["fr_open"] > 0).mean() * 100,
            x["fr_open"].mean(), x["fr_close"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="丢弃缓存重拉")
    a = ap.parse_args()
    pro = get_pro()
    auc = build_panel(pro, a.refresh)
    d = join_panel(auc)
    d["reg"] = d["date"].map(regimes(d))
    base = d[d["gap"] >= 1.0].copy()      # S1候选域=竞价涨幅≥1%
    print(f"\n验证窗口 {base['date'].min()}~{base['date'].max()} "
          f"{base['date'].nunique()}日 · S1候选域样本 {len(base)}")

    lines = []

    def put(s=""):
        print(s)
        lines.append(s)

    put("=" * 78)
    put("一、5档分位单调性(竞价量比分位 → 封板率/次日收益)")
    put("-" * 78)
    b = base[base["vr"] > 0].copy()
    b["q"] = pd.qcut(b["vrp"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    put(f"{'档':<4}{'样本':>8}{'封板率%':>9}{'次日开胜率%':>12}"
        f"{'次日开溢价%':>12}{'次日收盘%':>11}")
    for q, g in b.groupby("q", observed=True):
        n, sl, w, fo, fc = cohort_stat(g)
        put(f"{str(q):<4}{n:>8}{sl:>9.2f}{w:>12.2f}{fo:>12.2f}{fc:>11.2f}")
    g5, g1 = b[b["q"] == 5], b[b["q"] == 1]
    put(f"档5-档1 spread: 封板率 {cohort_stat(g5)[1] - cohort_stat(g1)[1]:+.2f}pp"
        f" · 次日收盘 {cohort_stat(g5)[4] - cohort_stat(g1)[4]:+.2f}pp")

    put("=" * 78)
    put("二、多方案并行 × 三段市况(过闸组 vs 未过闸组)")
    put("-" * 78)
    for name, fn in SCHEMES:
        put(f"\n【{name}】")
        put(f"{'市况':<6}{'过闸n':>8}{'封板率%':>9}{'次日收盘%':>11}"
            f"{'未过闸n':>9}{'封板率%':>9}{'次日收盘%':>11}{'差':>9}")
        ok_env = 0
        for reg in ("牛", "熊", "震荡"):
            sub = base[(base["reg"] == reg) & (base["vr"] > 0)].copy()
            if not len(sub):
                continue
            m = fn(sub)
            n1, s1, _, _, f1 = cohort_stat(sub[m])
            n0, s0, _, _, f0 = cohort_stat(sub[~m])
            diff = f1 - f0
            ok_env += 1 if diff > 0 else 0
            put(f"{reg:<6}{n1:>8}{s1:>9.2f}{f1:>11.2f}"
                f"{n0:>9}{s0:>9.2f}{f0:>11.2f}{diff:>+9.2f}")
        put(f"方向一致性: {ok_env}/3 环境次日收益同向为正 "
            f"→ {'通过' if ok_env >= 2 else '不通过'}")

    put("=" * 78)
    put("三、全窗口汇总(不分市况)")
    put("-" * 78)
    put(f"{'方案':<24}{'过闸n':>8}{'封板率%':>9}{'次日开胜率%':>12}"
        f"{'次日收盘%':>11}")
    bv = base[base["vr"] > 0].copy()
    n, sl, w, fo, fc = cohort_stat(bv[bv["vrp"] < 0.90])
    put(f"{'未过闸(分位<0.90)':<24}{n:>8}{sl:>9.2f}{w:>12.2f}{fc:>11.2f}")
    for name, fn in SCHEMES:
        m = fn(bv)
        n, sl, w, fo, fc = cohort_stat(bv[m])
        put(f"{name:<24}{n:>8}{sl:>9.2f}{w:>12.2f}{fc:>11.2f}")

    (OUT / "auction_vr_validate.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\n落盘 {OUT / 'auction_vr_validate.txt'}")


if __name__ == "__main__":
    main()
