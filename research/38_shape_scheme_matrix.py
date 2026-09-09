# -*- coding: utf-8 -*-
"""研究38: 形态因子方案矩阵(M0~M6)

研究37 单因子筛选出 3 个通过判据且**无价位风险**的因子:
  trapped_ratio  上方套牢盘密度   档1期望3.001 vs 档5 2.036, 三段一致3/3
  int_prox       距整数关口       档1期望2.385 → 档5 2.670, 三段一致3/3
  box_width_20   箱体宽度         盈亏比 档1 2.946 → 档5 1.828, 三段一致3/3

本研究把三者组合成方案矩阵并行回测, 选出最优形态。

方法论(沿用项目定稿口径):
  · 主判据是 T+1/T+2/T+3 胜率与盈亏比, **不是封板率**(研究30/31/08
    三次证明封板率与 EV 系统性反向)
  · 三段市况独立验证(牛/熊/震荡), 方向一致性 ≥2/3 才算通过
  · 分位阈值在全窗口上算一次, **不按市况分别拟合**(避免前视与过拟合)
  · 盈亏比/期望复用 core/exit_rules.pl_stats

入场价位闸门来自研究08 §4: 全档 EV 均为负, 2~4% 档需21.2%封板率打平
(T3实测25.3%可达), 6%以上需81%+ **数学上不可行**。故 M5/M6 加价位闸。

因子定义唯一出处: core/shape.py(与研究37 共用, 避免口径分叉)

产物: data/factor/shape_panel.parquet(因子数据集缓存, 可续跑)
      research/out/38_shape_scheme_matrix.md
用法: python research/38_shape_scheme_matrix.py [--refresh] [--quick]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from datastore import load, path_of, save                   # noqa: E402
from core.exit_rules import pl_stats                        # noqa: E402
from core.shape import (build_bars, build_regimes, reg_of,  # noqa: E402
                        compute_factors, compute_labels,
                        compute_labels_exec, XLABELS, XNLABEL)

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
CACHE = DATA / "factor" / "shape_panel.parquet"

# 方案矩阵。阈值用全窗口分位(算一次, 不按市况分别拟合)
# trapped_ratio 越低越好(研究37: 档1期望3.001 > 档5 2.036)
# int_prox     越高越好(研究37: 档5期望2.670 > 档1 2.385)
# box_width_20 越低盈亏比越好(研究37: 档1 2.946 > 档5 1.828)
Q_LOW = 0.40          # 低分位阈值(40分位以下)
Q_HIGH = 0.60         # 高分位阈值(60分位以上)
MAX_RISE_20D = 20.0   # 用户定稿买入模型: 首板近20日累计涨幅超此剔除(防接飞刀)
FIRST_BOARD = 1       # 用户定稿 MAX_LIANBAN=1: 只做首板/一进二

# ⚠ 口径重要差异(必读):
# 本研究的宇宙是 events_enriched 的**已涨停股**。研究37 用的基准
# close[T] 是涨停价 —— **尾盘买不进去**(买单排队也不一定成交),
# 那个口径的收益是纸面数字。
# 本研究改用**可执行口径**: 基准 = open[T+1](次日开盘买入),
# 并用 buyable 排除 T+1 开盘就涨停的一字板(仍买不进)。
# 所以本研究测的是「昨日涨停股 → 今日开盘接力」策略。

LABELS = ["r1_open", "r1_close", "r2_close", "r3_close"]
NLABEL = {"r1_open": "T+1开盘", "r1_close": "T+1收盘",
          "r2_close": "T+2收盘", "r3_close": "T+3收盘"}
say = print


def build_dataset(refresh: bool, quick: bool) -> pd.DataFrame:
    """构建因子数据集(带缓存)。因子只用 T-1 及更早 bar, 标签只用 T+1 及之后"""
    if CACHE.exists() and not refresh and not quick:
        say(f"载入缓存 {CACHE}")
        return pd.read_parquet(CACHE)
    say("加载触板股宇宙(limitup.events_enriched)…")
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "limit_times"])
    ev = ev.drop_duplicates(subset=["trade_date", "ts_code"])
    say(f"  {len(ev)} 条, {ev['trade_date'].nunique()} 个交易日")

    say("加载日线面板…")
    cols = ["trade_date", "ts_code", "open", "high", "low", "close",
            "vol", "pct_chg"]
    panel = pd.read_parquet(path_of("market.daily_panel"), columns=cols)
    if quick:
        cut = str(int(panel["trade_date"].max()) - 20000)
        panel = panel[panel["trade_date"] >= cut]
        say(f"  --quick: 截断到 {cut} 起")
    panel = panel.dropna(subset=["close"])

    say("构建 per-code bar 数组…")
    bars = build_bars(panel)

    say("计算因子与标签…")
    recs, n_skip = [], 0
    for r in ev.itertuples():
        b = bars.get(r.ts_code)
        if not b:
            n_skip += 1
            continue
        idx = int(np.searchsorted(b["date"], r.trade_date))
        if idx >= len(b["date"]) or b["date"][idx] != r.trade_date:
            n_skip += 1
            continue
        f = compute_factors(b, idx)
        lb = compute_labels(b, idx)
        ex = compute_labels_exec(b, idx, r.ts_code)
        if not f or not ex:
            n_skip += 1
            continue
        rec = {"trade_date": r.trade_date, "ts_code": r.ts_code,
               "limit_times": int(r.limit_times or 1),
               "entry_pct": (float(b["pct"][idx])
                             if not np.isnan(b["pct"][idx]) else None)}
        rec.update(f)
        rec.update(lb)          # 不可执行口径, 仅供与研究37 对照
        rec.update(ex)          # 可执行口径(基准 open[T+1])
        recs.append(rec)
    d = pd.DataFrame(recs)
    say(f"  有效 {len(d)} 条(跳过 {n_skip})")
    if not quick and len(d):
        save("factor.shape_panel", d)
        say(f"  已缓存 → {CACHE}")
    # 市况标签(全窗口三分位, 同研究31)
    rg = build_regimes(panel)
    d["reg"] = d["trade_date"].map(lambda x: reg_of(x, rg))
    return d


def thresholds(d: pd.DataFrame) -> dict:
    """分位阈值(全窗口算一次, 不按市况分别拟合 → 避免前视与过拟合)"""
    return {
        "trapped_ratio": d["trapped_ratio"].quantile(Q_LOW),
        "int_prox": d["int_prox"].quantile(Q_HIGH),
        "box_width_20": d["box_width_20"].quantile(Q_LOW),
    }


def schemes(q: dict) -> list:
    """方案矩阵 → [(名称, 过滤函数, 说明)]

    **所有方案都先过 buyable 闸** —— T+1 开盘就涨停的一字板买不进,
    不排除会把不可成交的票算进收益。
    """
    tr, ip, bw = q["trapped_ratio"], q["int_prox"], q["box_width_20"]
    BUY = d0 = "buyable"
    return [
        ("M0 基线", lambda d: d[d[BUY]],
         "全部可成交触板股, 无形态筛选"),
        ("M1 低套牢盘", lambda d: d[d[BUY]
                                    & (d["trapped_ratio"] <= tr)],
         f"trapped_ratio ≤ {tr:.1f}%(上方套牢盘少=无抛压)"),
        ("M2 远离整数关口", lambda d: d[d[BUY] & (d["int_prox"] >= ip)],
         f"int_prox ≥ {ip:.2f}%(避开整数关口心理阻力)"),
        ("M3 窄箱体", lambda d: d[d[BUY] & (d["box_width_20"] <= bw)],
         f"box_width_20 ≤ {bw:.1f}%(横盘充分, 亏损可控)"),
        ("M4 三者组合",
         lambda d: d[d[BUY] & (d["trapped_ratio"] <= tr)
                     & (d["int_prox"] >= ip) & (d["box_width_20"] <= bw)],
         "M1+M2+M3 三个条件同时满足"),
        ("M5 组合+首板",
         lambda d: d[d[BUY] & (d["trapped_ratio"] <= tr)
                     & (d["int_prox"] >= ip) & (d["box_width_20"] <= bw)
                     & (d["limit_times"] == FIRST_BOARD)],
         f"M4 + limit_times=={FIRST_BOARD}(用户定稿 MAX_LIANBAN=1, "
         "只做首板/一进二)"),
        ("M6 组合+首板+防接飞刀",
         lambda d: d[d[BUY] & (d["trapped_ratio"] <= tr)
                     & (d["int_prox"] >= ip) & (d["box_width_20"] <= bw)
                     & (d["limit_times"] == FIRST_BOARD)
                     & (d["rise_20d"] < MAX_RISE_20D)],
         f"M5 + rise_20d < {MAX_RISE_20D}%(用户定稿 MAX_RISE_20D, "
         "首板近20日超此剔除)"),
    ]


def eval_scheme(d: pd.DataFrame) -> dict:
    """单方案评估(**可执行口径** XLABELS, 基准 open[T+1])"""
    out = {}
    for lab in XLABELS:
        vals = [x for x in d[lab] if x is not None and not pd.isna(x)]
        n, wr, aw, al, ratio, exp = pl_stats(vals)
        out[lab] = (n,
                    round(wr * 100, 2) if wr is not None else None,
                    round(aw, 3) if aw is not None else None,
                    round(al, 3) if al is not None else None,
                    round(ratio, 3) if ratio is not None else None,
                    round(exp, 3) if exp is not None else None)
    # 不可执行口径(r1_open, 基准 close[T])仅供对照
    vals = [x for x in d["r1_open"] if x is not None and not pd.isna(x)]
    n, wr, aw, al, ratio, exp = pl_stats(vals)
    out["r1_open"] = (n, round(wr * 100, 2) if wr is not None else None,
                      round(aw, 3) if aw is not None else None,
                      round(al, 3) if al is not None else None,
                      round(ratio, 3) if ratio is not None else None,
                      round(exp, 3) if exp is not None else None)
    # 买入价统计(次日开盘相对昨收的溢价)
    g = [x for x in d["gap_open"] if x is not None and not pd.isna(x)]
    out["gap_open"] = (round(float(np.mean(g)), 3) if g else None,
                       round(float(np.median(g)), 3) if g else None)
    # 三段市况的 T+1收盘期望(可执行口径)
    regs = {}
    for r in ("牛", "熊", "震荡"):
        sub = d[d["reg"] == r]
        vals = [x for x in sub["x1_close"]
                if x is not None and not pd.isna(x)]
        n, wr, aw, al, ratio, exp = pl_stats(vals)
        regs[r] = (n, round(exp, 3) if exp is not None else None)
    out["regs"] = regs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="丢弃缓存重算")
    ap.add_argument("--quick", action="store_true", help="只用最近2年(调试)")
    a = ap.parse_args()

    d = build_dataset(a.refresh, a.quick)
    if not len(d):
        say("无有效样本")
        return
    q = thresholds(d)
    say(f"分位阈值: trapped_ratio ≤ {q['trapped_ratio']:.1f}% · "
        f"int_prox ≥ {q['int_prox']:.2f}% · "
        f"box_width_20 ≤ {q['box_width_20']:.1f}%")

    say("评估方案矩阵…")
    res = {}
    for name, fn, desc in schemes(q):
        try:
            sub = fn(d)
        except Exception as e:
            say(f"  {name} 过滤失败: {e}")
            continue
        res[name] = {"desc": desc, "n": len(sub),
                     "stat": eval_scheme(sub) if len(sub) else None}
        say(f"  {name}: {len(sub)} 条 ({len(sub)/len(d)*100:.1f}%)")

    write_report(d, q, res)
    say(f"\n报告已写入 {OUT/'38_shape_scheme_matrix.md'}")


def write_report(d, q, res):
    L = []
    A = L.append
    A("# 研究38: 形态因子方案矩阵(M0~M6)\n")
    A(f"- 宇宙: 触板股 **{len(d)}** 条, {d['trade_date'].nunique()} 个交易日, "
      f"{d['trade_date'].min()}~{d['trade_date'].max()}")
    A(f"- 分位阈值(全窗口算一次, 不按市况分别拟合): "
      f"`trapped_ratio ≤ {q['trapped_ratio']:.1f}%` · "
      f"`int_prox ≥ {q['int_prox']:.2f}%` · "
      f"`box_width_20 ≤ {q['box_width_20']:.1f}%`")
    A(f"- 入场闸(用户定稿买入模型): M5 `limit_times=={FIRST_BOARD}`(首板) / "
      f"M6 再加 `rise_20d < {MAX_RISE_20D}%`(防接飞刀)")
    A("- 指标: 胜率 / 平均盈利 / 平均亏损 / **盈亏比** / 期望"
      "(复用 `core/exit_rules.pl_stats`)\n")

    A("## ⚠ 口径重要差异(必读)\n")
    A("**研究37 的口径不可执行**: 它用 `close[T]` 作买入基准, 但 T 日已涨停"
      "封死 —— **尾盘买不进去**(买单排队也不一定成交), 算出的收益是"
      "纸面数字。\n")
    A("本研究改用**可执行口径**:\n")
    A("- 买入基准 = **`open[T+1]`**(次日开盘买入) —— 这是真能成交的价格")
    A("- `buyable` 闸排除 **T+1 开盘就涨停的一字板**(仍买不进), "
      f"全宇宙 {len(d)} 条中可成交 **{int(d['buyable'].sum())}** 条"
      f"({d['buyable'].mean()*100:.1f}%)")
    A(f"- 买入溢价(T+1开盘 相对 T日收盘): 均值 "
      f"{d.loc[d['buyable'], 'gap_open'].mean():.2f}% · 中位 "
      f"{d.loc[d['buyable'], 'gap_open'].median():.2f}%")
    A("")
    A("所以本研究测的是「**昨日涨停股 → 今日开盘接力**」策略:\n")
    A("| | 研究08(盘中打板) | 研究37(不可执行) | **研究38(接力)** |")
    A("|---|---|---|---|")
    A("| 宇宙 | 盘中触发 S2/S3 | 已涨停股 | 已涨停股 |")
    A("| 买入价 | 触发时刻(~4.4%) | ~~收盘涨停价~~ | **T+1开盘** |")
    A("| 可成交 | 是 | **否** | 是(排除一字板) |")
    A("")
    A("**外推注意**: 若要用于盘中打板, 需在**盘中触板宇宙**(带触发时刻"
      "涨幅)上重跑验证 —— 当前无历史分时数据支持(intraday_px 只有近7天)。\n")

    A("## 结论\n")
    # 找最优: 按可执行口径 T+1收盘期望, 样本量充足
    best = None
    for name, r in res.items():
        if not r["stat"]:
            continue
        s = r["stat"]["x1_close"]
        if s[0] < 500:               # 样本不足不参选
            continue
        if best is None or (s[5] or -99) > (best[1]["stat"]["x1_close"][5]
                                            or -99):
            best = (name, r)
    if best:
        s = best[1]["stat"]["x1_close"]
        A(f"**最优方案: {best[0]}** — {best[1]['desc']}\n")
        A(f"- 样本 {s[0]} 条({s[0]/len(d)*100:.1f}% of 宇宙)")
        A(f"- **可执行口径** T+1收盘(基准 open[T+1]): 胜率 {s[1]}% · "
          f"盈亏比 **{s[4]}** · 期望 **{s[5]}%**")
        s3 = best[1]["stat"]["x3_close"]
        A(f"- T+3收盘: 胜率 {s3[1]}% · 盈亏比 {s3[4]} · 期望 {s3[5]}%")
        s0 = best[1]["stat"]["r1_open"]
        A(f"- (对照)研究37 不可执行口径 T+1开盘: 期望 {s0[5]}% —— "
          "以涨停价为基准, 实际买不到")
        rg = best[1]["stat"]["regs"]
        A(f"- 三段市况 T+1收盘期望: 牛 {rg['牛'][1]} / 熊 {rg['熊'][1]} / "
          f"震荡 {rg['震荡'][1]}\n")
    else:
        A("(无方案样本量 ≥500, 无法选优)\n")

    A("## 方案矩阵总表(可执行口径, 基准 open[T+1])\n")
    A("| 方案 | 说明 | 样本 | T+1收盘胜率% | T+1收盘盈亏比 | "
      "**T+1收盘期望** | T+3收盘期望 | (对照)不可执行T+1开盘 |")
    A("|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        if not r["stat"]:
            A(f"| {name} | {r['desc']} | {r['n']} | - | - | - | - | - |")
            continue
        s1, s3, s0 = (r["stat"]["x1_close"], r["stat"]["x3_close"],
                      r["stat"]["r1_open"])
        A(f"| **{name}** | {r['desc']} | {s1[0]} "
          f"| {s1[1]} | {s1[4]} | {s1[5]} | {s3[5]} | {s0[5]} |")
    A("")
    A("最后一列是研究37 的不可执行口径(以涨停价为买入基准), 仅作对照 —— "
      "它系统性高于可执行口径, 差额就是「涨停价买不到」的虚高。\n")

    A("## 三段市况一致性(T+1收盘期望, 可执行口径)\n")
    A("| 方案 | 牛(样本/期望) | 熊(样本/期望) | 震荡(样本/期望) | 全为正 |")
    A("|---|---|---|---|---|")
    for name, r in res.items():
        if not r["stat"]:
            continue
        rg = r["stat"]["regs"]
        cells = []
        allpos = True
        for k in ("牛", "熊", "震荡"):
            n, e = rg[k]
            cells.append(f"{n} / {e}")
            if e is None or e <= 0:
                allpos = False
        A(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} "
          f"| {'**是**' if allpos else '否'} |")
    A("")

    A("## 各持有期明细\n")
    for name, r in res.items():
        if not r["stat"]:
            continue
        A(f"### {name}\n")
        A(f"{r['desc']}\n")
        A("| 持有期 | 样本 | 胜率% | 平均盈利 | 平均亏损 | 盈亏比 | 期望 |")
        A("|---|---|---|---|---|---|---|")
        for lab in XLABELS:
            s = r["stat"][lab]
            A(f"| {XNLABEL[lab]} | {s[0]} | {s[1]} | {s[2]} | {s[3]} "
              f"| {s[4]} | {s[5]} |")
        s0 = r["stat"]["r1_open"]
        A(f"| ~~T+1开盘(不可执行)~~ | {s0[0]} | {s0[1]} | {s0[2]} | {s0[3]} "
          f"| {s0[4]} | {s0[5]} |")
        gp = r["stat"]["gap_open"]
        A(f"\n买入溢价(T+1开盘 vs T日收盘): 均值 {gp[0]}% · 中位 {gp[1]}%\n")

    A("## 方法论与约束\n")
    A("1. **主判据是胜率与盈亏比, 不是封板率** —— 研究30/31/08 三次证明"
      "封板率与 EV 系统性反向")
    A("2. **分位阈值全窗口算一次**, 不按市况分别拟合 —— 避免前视与过拟合")
    A("3. **入场闸改用用户定稿买入模型**而非研究08 价位闸 —— 本宇宙是"
      "已涨停股(entry_pct~10%), 研究08 的盘中触发价闸不适用")
    A("4. **主口径是无条件持有**(今日收盘→T+N), 不是打板策略实际 EV —— "
      "研究37 副口径(定稿卖出规则)显示封板续持才是收益来源"
      "(22.3%的票 T+2 +11.6% / T+3 +12.9%)")
    A("5. **只筛选不进生产** —— 最优方案只进影子字段与看板展示, "
      "不改 S1/S2/S3 触发路径")
    A("6. 因子定义唯一出处 `core/shape.py`(与研究37 共用, 避免口径分叉)")

    (OUT / "38_shape_scheme_matrix.md").write_text("\n".join(L),
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
