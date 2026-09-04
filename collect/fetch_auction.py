# -*- coding: utf-8 -*-
"""竞价数据 T+1 校正: tushare stk_auction 官方口径 → auction_YYYYMMDD.json

盘中竞价快照(apps/radar._auc_capture)在 09:25~09:30 采集并冻结, 本脚本只
补 official 子键作对照, **绝不覆盖盘中值**——影子字段的全部价值在于记录
"系统在决策时刻相信的是什么", 覆盖就等于用未来信息验证当时的决策
(违反 Forward Return 方法论)。两者差异本身就是行情源质量的体检指标。

官方口径(实测反推验证): volume_ratio = 竞价量 / 近5日每分钟均量, 与
quotes 层 emin=1 的算法同式。量纲: vol=股, amount=元, turnover_rate=%。
分位排名域与闸阈值与盘中一致(竞价涨幅≥1% 域内, 分位≥0.90 过闸)。

历史起点 2025-01-20(更早日期返回空, 竞价量比因子只有约1.6年可回溯)。

用法: python collect/fetch_auction.py [--days N] [--date YYYYMMDD]
"""
import argparse
import json
import math
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, get_pro  # noqa: E402

LIVE = DATA / "live"
GATE_PCT = 0.90            # 与 apps/radar.AUC_GATE_PCT 同值
DOMAIN_PCT = 1.0           # 与 apps/radar.AUC_DOMAIN_PCT 同值(S1候选域)
MIN_N = 20                 # 截面样本不足则分位不可算


def _f(v, nd: int | None = None) -> float:
    """安全转float: None/NaN/inf → 0.0。
    NaN 能通过 `x <= 0` 这类守卫(比较恒 False), 而 json.dump 默认
    allow_nan=True 会把 NaN 写进文件 → 浏览器 JSON.parse 报错。"""
    try:
        x = float(v or 0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return round(x, nd) if nd is not None else x


def trade_days(pro, n: int) -> list:
    """最近 n 个交易日(含今日)"""
    today = datetime.now().strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date="20250101",
                        end_date=today, is_open="1")
    return sorted(cal["cal_date"])[-n:]


def official_of(pro, date: str) -> dict:
    """官方竞价截面 {code: {gap,px,amt,vol,vr,tover,vrp,gate}}"""
    df = pro.stk_auction(trade_date=date)
    if df is None or df.empty:
        return {}
    df = df[~df["ts_code"].str.endswith(".BJ")]
    out = {}
    for r in df.itertuples():
        pc = _f(getattr(r, "pre_close", 0))
        px = _f(getattr(r, "price", 0))
        if pc <= 0 or px <= 0:
            continue
        out[r.ts_code] = {
            "gap": round((px / pc - 1) * 100, 2), "px": px,
            "amt": _f(getattr(r, "amount", 0)),
            "vol": _f(getattr(r, "vol", 0)),
            "vr": _f(getattr(r, "volume_ratio", 0), 3),
            "tover": _f(getattr(r, "turnover_rate", 0), 4)}
    # 与盘中同域同闸: 竞价涨幅≥1% 域内排名, 分位≥0.90 过闸
    vals = sorted(v["vr"] for v in out.values()
                  if v["gap"] >= DOMAIN_PCT and v["vr"] > 0)
    ok = len(vals) >= MIN_N
    for v in out.values():
        p = bisect_left(vals, v["vr"]) / len(vals) \
            if ok and v["vr"] > 0 else None
        v["vrp"] = round(p, 3) if p is not None else None
        v["gate"] = (p >= GATE_PCT) if p is not None else None
    return out


def merge(date: str, off: dict) -> None:
    """写 official 子键, 保留盘中 stocks 不动"""
    f = LIVE / f"auction_{date}.json"
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.exists() \
            else {"date": date, "stocks": {}}
    except Exception:
        d = {"date": date, "stocks": {}}
    d["date"] = date
    d.setdefault("stocks", {})
    d["official"] = off
    d["official_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st = d["stocks"]
    both = [c for c in st if c in off and st[c].get("vr") and off[c].get("vr")]
    if both:
        rel = sorted(abs(st[c]["vr"] / off[c]["vr"] - 1) for c in both)
        flip = sum(1 for c in both if st[c].get("gate") != off[c].get("gate"))
        print(f"  盘中/官方对照 {len(both)}只 量比中位偏差"
              f"{rel[len(rel) // 2] * 100:.1f}% 闸标记翻转{flip}只")
    else:
        print("  无盘中快照可对照(雷达当日未采到竞价量)")
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1, help="回填最近N个交易日")
    ap.add_argument("--date", help="指定单日 YYYYMMDD")
    a = ap.parse_args()
    pro = get_pro()
    dates = [a.date] if a.date else trade_days(pro, a.days)
    for date in dates:
        off = official_of(pro, date)
        if not off:
            print(f"{date} stk_auction 无数据(早于2025-01-20或非交易日)")
            continue
        merge(date, off)
        print(f"{date} 官方竞价 {len(off)}只 过闸"
              f"{sum(1 for v in off.values() if v['gate'])}只")


if __name__ == "__main__":
    main()
