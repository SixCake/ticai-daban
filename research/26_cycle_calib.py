# -*- coding: utf-8 -*-
"""研究26: 情绪四阶段v0校准（仅当样本≥30天时跑）

数据源: data/live/intraday_YYYYMMDD.jsonl 的 stage/sentiment 字段
(poller 每分钟积累, core/cycle.stage_of v0 输出)

方法论(用户强制: 牛/熊/震荡三段独立验证, 同研究25环境划分):
  1. 各阶段(春/夏/秋/冬)收盘快照: 炸板率/最高板/涨停家数分布
  2. 阶段→次日兑现: 次日全A上涨家数比与涨停家数, 验证
     "夏次日应明显强于冬"——若不成立则v0阈值需调
  3. 阶段切换频率: 单日内stage翻转次数(过多=规则抖动, 需平滑)

样本<30天仅打印积累进度, 不作结论。
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load  # noqa: E402

LIVE = Path(__file__).resolve().parent.parent / "data" / "live"
MIN_DAYS = 30


def load_days():
    """{date: {stage, open_sent, close_sent, flips}}"""
    out = {}
    for f in sorted(LIVE.glob("intraday_*.jsonl")):
        date = f.stem.replace("intraday_", "")
        rows = []
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "stage" in d:
                rows.append(d)
        if not rows:
            continue
        stages = [r["stage"] for r in rows]
        flips = sum(1 for i in range(1, len(stages))
                    if stages[i] != stages[i - 1])
        out[date] = {"stage": stages[-1], "open": rows[0]["sentiment"],
                     "close": rows[-1]["sentiment"], "flips": flips}
    return out


def next_day_metrics():
    """{date: {advance, zt}} 次日全A上涨比+涨停家数"""
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    adv = dp.groupby("trade_date").apply(
        lambda g: (g["pct_chg"] > 0).mean(), include_groups=False)
    ev = load("limitup.events", columns=["trade_date"])
    zt = ev.groupby("trade_date").size()
    dates = sorted(adv.index)
    nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    return {d: {"advance": float(adv[nxt[d]]),
                "zt": int(zt.get(nxt[d], 0))}
            for d in dates[:-1]}


def market_env():
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    day = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    ma20 = day.rolling(20).mean()
    return {str(d): ("牛市" if v > 0.10 else "熊市" if v < -0.10
                     else "震荡市")
            for d, v in ma20.items() if pd.notna(v)}


def main():
    days = load_days()
    print(f"积累天数: {len(days)} (阈值{MIN_DAYS})")
    if len(days) < MIN_DAYS:
        for d, v in list(days.items())[-5:]:
            print(f"  {d}: 收盘阶段={v['stage']} 翻转{v['flips']}次 "
                  f"涨停{v['close']['zt_count']} 炸板率{v['close']['broken_rate']:.0%}")
        print("样本不足, 继续积累, 不作校准结论")
        return
    env = market_env()
    nxt = next_day_metrics()
    rows = []
    for stage in ["春", "夏", "秋", "冬"]:
        sub = [d for d, v in days.items() if v["stage"] == stage]
        if not sub:
            continue
        nx = [nxt[d] for d in sub if d in nxt]
        rows.append({
            "阶段": stage, "天数": len(sub),
            "日内翻转均值": round(sum(days[d]["flips"] for d in sub)
                                / len(sub), 1),
            "收盘炸板率%": round(100 * sum(
                days[d]["close"]["broken_rate"] for d in sub) / len(sub), 1),
            "收盘最高板": round(sum(
                days[d]["close"]["max_height"] for d in sub) / len(sub), 1),
            "次日上涨比%": round(100 * sum(
                nxt[d]["advance"] for d in nx) / max(1, len(nx)), 1),
            "次日涨停家数": round(sum(
                nxt[d]["zt"] for d in nx) / max(1, len(nx)), 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    for e in ["牛市", "熊市", "震荡市"]:
        sub = [d for d in days if env.get(d) == e]
        if len(sub) < 5:
            continue
        print(f"\n--- {e} (n={len(sub)}) ---")
        tbl = []
        for stage in ["春", "夏", "秋", "冬"]:
            ss = [d for d in sub if days[d]["stage"] == stage]
            if not ss:
                continue
            nx = [nxt[d] for d in ss if d in nxt]
            tbl.append({"阶段": stage, "天数": len(ss),
                        "次日上涨比%": round(100 * sum(
                            nxt[d]["advance"] for d in nx)
                            / max(1, len(nx)), 1)})
        print(pd.DataFrame(tbl).to_string(index=False))
    # 校准判据: 夏次日上涨比应>冬+5pct, 否则提示调阈值
    sx = [d for d in days if days[d]["stage"] == "夏" and d in nxt]
    wt = [d for d in days if days[d]["stage"] == "冬" and d in nxt]
    if sx and wt:
        a_s = sum(nxt[d]["advance"] for d in sx) / len(sx)
        a_w = sum(nxt[d]["advance"] for d in wt) / len(wt)
        print(f"\n校准判据: 夏次日{a_s:.1%} vs 冬次日{a_w:.1%} "
              + ("区分度成立" if a_s > a_w + 0.05
                 else "⚠ 区分度不足, v0阈值需调"))


if __name__ == "__main__":
    main()
