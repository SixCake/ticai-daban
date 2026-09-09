# -*- coding: utf-8 -*-
"""研究41: 题材阶段 × 角色 的打板研究(全宇宙口径)

**为何做**: 研究40 把所有票当同质处理, 得出「打板全档负期望」的结论。但
实盘不是这样做的 —— 92科比框架的核心是**行情不同阶段做不同题材**:
  爆发(第1波) → 做龙头
  主升(第2波) → 龙头/中军
  鱼尾(第3波+) → 做补涨或切换
研究40 忽略了这个维度, 等于把「爆发期龙头」和「鱼尾期跟风」混在一起算
平均, 结论必然被稀释。本研究把题材阶段与角色拆开看。

**口径(沿用研究40, 全部日线)**:
  宇宙   盘中最高涨幅 ≥ 目标档(复用 data/factor/fulluniv_panel.parquet)
  入场价 阈值 + 实测滑点(研究39 校准)
  离场   日线近似定稿规则(止损5% / 封板续持 / 未封板收盘)
  因子   无 —— 本研究只拆维度, 不做因子筛选

**新增维度**:
  theme_stage  爆发/主升/鱼尾/无 —— `core/cycle.theme_stage(wave_no, zt_all)`
               定稿纯波次语义: 爆发=第1波 / 主升=第2波 / 鱼尾=第3波+
               数据源 theme.day(6.8年全覆盖)
  role         龙头 / 题材内涨停 / 题材内未涨停 / 无题材
               龙头 = 该股 == theme.day.leader_code(其所属题材当日龙头)
  height       连板高度(events_enriched.limit_times, 非涨停股记0)

**归属近似必须说明**: `con2stock` 是**静态**映射(当前概念成分), 用于历史
是近似 —— 概念成分会随时间变化。这是数据限制, 不是设计选择。

产物: research/out/41_theme_stage_role.md
用法: python research/41_theme_stage_role.py [--start YYYYMMDD] [--end YYYYMMDD]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load                                  # noqa: E402
from core.cycle import theme_stage                          # noqa: E402
from core.attribute import load_con2stock                   # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
UNIV = Path(__file__).resolve().parent.parent / "data" / "factor" \
    / "fulluniv_panel.parquet"

TARGETS = [3.0, 4.0, 5.0, 6.0]
SLIPPAGE = {3.0: 1.53, 4.0: 1.48, 5.0: 1.27, 6.0: 1.12}
STAGES = ["爆发", "主升", "鱼尾", "无"]
ROLES = ["昨日龙头", "昨日题材内", "无题材"]
say = print


def build_dims(d: pd.DataFrame) -> pd.DataFrame:
    """给全宇宙数据集加 theme_stage / role 两个维度

    **必须锚定 T-1**(同 T级 信号定稿口径), 不能用当日:
      ① 当日 theme_stage 含未来信息 —— 阶段由当日涨停家数/波次算出
      ② 当日 leader_code 由当日涨停数据选出 —— 入场时根本不知道
      ③ 用当日 sealed 定义角色会造成**循环论证**: 「题材内涨停」
        定义就是 sealed → 封板率必然100%, 「题材内未涨停」必然0%
    入场时刻真正可知的是: 昨日题材阶段 + 昨日龙头是谁。
    """
    say("加载 theme.day(题材阶段与龙头)…")
    td = load("theme.day")
    dates = sorted(td["trade_date"].unique())
    # (date, concept) → (wave_no, zt_all, leader_code)
    td_key = {(r.trade_date, r.concept_code):
              (int(r.wave_no or 0), int(r.zt_all or 0), r.leader_code)
              for r in td.itertuples()}
    import bisect
    say(f"  theme.day {len(td)} 行, {len(dates)} 个交易日")

    say("加载概念成分映射(静态, 历史归属为近似)…")
    c2s = load_con2stock()
    s2c = {}
    for con, codes in c2s.items():
        for c in codes:
            s2c.setdefault(c, []).append(con)
    say(f"  {len(c2s)} 个概念, {len(s2c)} 只票有归属")

    say("计算维度(锚定 T-1)…")
    stages, roles, themes = [], [], []
    for r in d.itertuples():
        # T-1 = 最后一个严格早于当日的交易日
        pos = bisect.bisect_left(dates, r.trade_date) - 1
        if pos < 0:
            stages.append("无")
            roles.append("无题材")
            themes.append(None)
            continue
        prev = dates[pos]
        cons = s2c.get(r.ts_code, [])
        best_stage, best_role, best_theme = "无", "无题材", None
        best_zt = -1
        for con in cons:
            k = (prev, con)
            if k not in td_key:
                continue
            wave_no, zt_all, leader = td_key[k]
            st = theme_stage(wave_no, zt_all)
            if st == "无":
                continue
            # 取昨日关联家数最多的在场题材作为主题材
            if zt_all > best_zt:
                best_zt = zt_all
                best_stage = st
                best_theme = con
                # 角色 = **昨日龙头**(入场时可知), 不用当日 sealed
                best_role = "昨日龙头" if (leader and leader == r.ts_code) \
                    else "昨日题材内"
        stages.append(best_stage)
        roles.append(best_role)
        themes.append(best_theme)
    d["theme_stage"] = stages
    d["role"] = roles
    d["theme"] = themes
    say(f"  阶段分布: {dict(pd.Series(stages).value_counts())}")
    say(f"  角色分布: {dict(pd.Series(roles).value_counts())}")
    return d


def stat(sub: pd.DataFrame, tgt: int) -> tuple:
    """单组统计 → (样本, 封板率%, 胜率%, 盈亏比, 期望%)"""
    from core.exit_rules import pl_stats
    ret = f"e{tgt}_ret"
    ok = f"e{tgt}_ok"
    s = sub[sub[ok] & sub[ret].notna()]
    if len(s) < 200:
        return (len(s), None, None, None, None)
    n, wr, aw, al, ratio, exp = pl_stats(list(s[ret]))
    return (n, round(s["sealed"].mean() * 100, 2),
            round(wr * 100, 2) if wr is not None else None,
            round(ratio, 3) if ratio is not None else None,
            round(exp, 3) if exp is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250101")
    ap.add_argument("--end", default="20260610")
    a = ap.parse_args()
    if not UNIV.exists():
        say(f"缺少 {UNIV}, 先跑 research/40_full_universe.py")
        return
    d = pd.read_parquet(UNIV)
    d = d[(d["trade_date"] >= a.start) & (d["trade_date"] <= a.end)]
    say(f"全宇宙 {len(d)} 条票·日, {d['trade_date'].nunique()} 个交易日")
    d = build_dims(d)

    say("交叉分析: 题材阶段 × 角色 × 入场档…")
    res = {}
    for tgt in TARGETS:
        t = int(tgt)
        for st in STAGES:
            for ro in ROLES:
                sub = d[(d["theme_stage"] == st) & (d["role"] == ro)]
                res[(t, st, ro)] = stat(sub, t)
    write_report(d, res, a.start, a.end)
    say(f"\n报告已写入 {OUT/'41_theme_stage_role.md'}")


def write_report(d, res, start, end):
    L = []
    A = L.append
    A("# 研究41: 题材阶段 × 角色 的打板研究\n")
    A(f"- 窗口 {start}~{end}, 全宇宙 **{len(d)}** 条票·日, "
      f"{d['trade_date'].nunique()} 个交易日")
    A("- 口径沿用研究40: 宇宙=盘中达≥目标档, 入场=阈值+实测滑点, "
      "离场=日线近似定稿规则")
    A("- 阶段: `core/cycle.theme_stage` 定稿纯波次语义"
      "(爆发=第1波/主升=第2波/鱼尾=第3波+), **锚定 T-1** 的 theme.day")
    A("- 角色: **昨日龙头**(==T-1 的 theme.day.leader_code) / 昨日题材内 / "
      "无题材\n")

    A("## ⚠ 为何必须锚定 T-1(重要教训)\n")
    A("第一版用了**当日** theme.day 与当日 `sealed` 定义角色, 造成两个致命问题:\n")
    A("1. **循环论证**: 「题材内涨停」定义就是 sealed → 封板率必然 100%, "
      "「题材内未涨停」必然 0%。实测确实如此, 结果毫无意义。")
    A("2. **前视**: 当日 leader_code 由当日涨停数据选出, 入场时根本不知道; "
      "当日阶段由当日涨停家数算出, 同样含未来信息。\n")
    A("修正后锚定 T-1 —— 入场时刻真正可知的是「昨日题材阶段 + 昨日龙头是谁」, "
      "与 T级 信号的定稿口径一致。\n")

    A("## 为何做\n")
    A("研究40 把所有票当同质处理, 得出「打板全档负期望」。但实盘按92科比"
      "框架是**行情不同阶段做不同题材**:\n")
    A("- 爆发(第1波) → 做龙头")
    A("- 主升(第2波) → 龙头/中军")
    A("- 鱼尾(第3波+) → 做补涨或切换\n")
    A("研究40 把「爆发期龙头」和「鱼尾期跟风」混在一起算平均, 结论必然被"
      "稀释。本研究拆开看。\n")

    A("## 维度分布\n")
    A("| 题材阶段 | 样本 | 占比 |")
    A("|---|---|---|")
    for st in STAGES:
        n = int((d["theme_stage"] == st).sum())
        A(f"| {st} | {n} | {n/len(d)*100:.1f}% |")
    A("")
    A("| 角色 | 样本 | 占比 |")
    A("|---|---|---|")
    for ro in ROLES:
        n = int((d["role"] == ro).sum())
        A(f"| {ro} | {n} | {n/len(d)*100:.1f}% |")
    A("")

    A("## 归属近似(必须说明)\n")
    A("`con2stock` 是**静态**映射(当前概念成分), 用于历史是近似 —— "
      "概念成分会随时间变化。这是数据限制, 不是设计选择。"
      "归属缺失的票归入「无题材」组。\n")

    for tgt in TARGETS:
        t = int(tgt)
        A(f"## {t}% 入场档: 题材阶段 × 角色\n")
        A("| 题材阶段 | 角色 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% |")
        A("|---|---|---|---|---|---|---|")
        for st in STAGES:
            for ro in ROLES:
                n, sr, wr, ra, ex = res[(t, st, ro)]
                if n < 200:
                    A(f"| {st} | {ro} | {n} | 样本不足 | - | - | - |")
                    continue
                A(f"| {st} | {ro} | {n} | {sr} | {wr} | {ra} | **{ex}** |")
        A("")
        # 找该档最优组合
        best = None
        for st in STAGES:
            for ro in ROLES:
                n, sr, wr, ra, ex = res[(t, st, ro)]
                if n < 500 or ex is None:
                    continue
                if best is None or ex > best[2]:
                    best = (st, ro, ex, n, ra)
        if best:
            A(f"**{t}%档最优组合: {best[0]} × {best[1]}** — "
              f"期望 **{best[2]}%**, 盈亏比 {best[4]}, 样本 {best[3]}\n")

    A("## 结论\n")
    # 汇总各档最优
    A("| 入场档 | 最优阶段 | 最优角色 | 期望% | 盈亏比 | 样本 | "
      "vs 全宇宙基线 |")
    A("|---|---|---|---|---|---|---|")
    for tgt in TARGETS:
        t = int(tgt)
        base = None
        for st in STAGES:
            for ro in ROLES:
                pass
        # 基线 = 该档全体
        from core.exit_rules import pl_stats
        sub = d[d[f"e{t}_ok"] & d[f"e{t}_ret"].notna()]
        _, _, _, _, _, bexp = pl_stats(list(sub[f"e{t}_ret"]))
        best = None
        for st in STAGES:
            for ro in ROLES:
                n, sr, wr, ra, ex = res[(t, st, ro)]
                if n < 500 or ex is None:
                    continue
                if best is None or ex > best[2]:
                    best = (st, ro, ex, n, ra)
        if best and bexp is not None:
            A(f"| {t}% | {best[0]} | {best[1]} | **{best[2]}** | {best[4]} "
              f"| {best[3]} | {best[2]-bexp:+.3f}pp (基线{round(bexp,3)}) |")
    A("")

    A("## 约束\n")
    A("1. **归属是静态近似** —— con2stock 为当前概念成分, 历史归属会漂移")
    A("2. **角色只区分龙头/非龙头** —— 补涨/共振需历史归属集合(att_set), "
      "当前数据不支持精确判定")
    A("3. **离场是日线近似** —— 不知道止损在哪一分钟触发")
    A("4. **入场价是建模值**(阈值+实测滑点)")
    A("5. 样本 <500 的组合不参选最优(避免小样本偶然)")
    A("6. **只筛选不进生产** —— 与 T/S 双轨体系一致")

    (OUT / "41_theme_stage_role.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
