# -*- coding: utf-8 -*-
"""研究42: 题材可持续性分类 × 阶段 × 角色(全宇宙口径)

**为何做**: 研究41 发现「爆发 × 昨日龙头」是最优组合(+1.7~2.3pp), 但
「爆发」= 第1波 = 题材近1年内**首次被炒**。这里面混了两种完全不同的东西:
  ① 真爆发 —— 产业趋势启动, 有多家涨停/打出高度/持续多日, 可延续
  ② 消息面催化 —— 单一事件驱动(中标/政策/传闻), 1~2家首板, 次日即死
研究41 把两者混在一起算平均, 稀释了结论。本研究拆开。

**分类口径(全部 T-1, 无前视)**:
  可持续型  zt_all≥5 且 (max_height≥2 或 theme_age≥3)
            —— 广度(多家涨停) + 强度(打出连板) 或 持续性(连续活跃≥3天)
  消息面型  zt_all≤2 且 max_height==1 且 theme_age==1
            —— 窄(1~2家) + 全是首板 + 第一天出现
  中间型    其余
  已验证    wave_no≥2(题材被炒过第2轮以上, 单独作为一个维度)

实测分布(20250101~20260610, theme.day 11,883 行):
  可持续型 961 行(8.1%) / 消息面型 5,418 行(45.6%) / 已验证 6,603 行(55.6%)

**口径沿用研究40/41**: 宇宙=盘中达≥目标档, 入场=阈值+实测滑点,
离场=日线近似定稿规则, 阶段与角色锚定 T-1。

⚠ 不修改阶段语义: 记忆定稿「阶段是语义概念, 不得为了提区分度而扭曲语义」。
本研究**新增独立的题材类型维度**, 不动 theme_stage 的定义。

产物: research/out/42_theme_sustainability.md
用法: python research/42_theme_sustainability.py [--start YYYYMMDD] [--end YYYYMMDD]
"""
import argparse
import bisect
import sys
from pathlib import Path

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
STAGES = ["爆发", "主升", "鱼尾", "无"]
ROLES = ["昨日龙头", "昨日题材内", "无题材"]
TTYPES = ["可持续型", "中间型", "消息面型", "无题材"]
# 可持续型阈值(实测分布: 8.1%)
SUS_MIN_ZT = 5          # 广度: 昨日关联涨停家数
SUS_MIN_HT = 2          # 强度: 昨日最高连板
SUS_MIN_AGE = 3         # 持续性: 昨日连续活跃天数
# 消息面型阈值(实测分布: 45.6%)
NEWS_MAX_ZT = 2
say = print


def build_dims(d: pd.DataFrame) -> pd.DataFrame:
    """加 theme_stage / role / theme_type / proven 四个维度(全部锚定 T-1)"""
    say("加载 theme.day…")
    td = load("theme.day")
    dates = sorted(td["trade_date"].unique())
    td_key = {(r.trade_date, r.concept_code):
              (int(r.wave_no or 0), int(r.zt_all or 0), int(r.max_height or 0),
               int(r.theme_age or 0), r.leader_code)
              for r in td.itertuples()}
    say(f"  {len(td)} 行, {len(dates)} 个交易日")

    say("加载概念成分映射(静态, 历史归属为近似)…")
    c2s = load_con2stock()
    s2c = {}
    for con, codes in c2s.items():
        for c in codes:
            s2c.setdefault(c, []).append(con)
    say(f"  {len(c2s)} 个概念, {len(s2c)} 只票有归属")

    say("计算维度(锚定 T-1)…")
    stages, roles, ttypes, provens, themes = [], [], [], [], []
    for r in d.itertuples():
        pos = bisect.bisect_left(dates, r.trade_date) - 1
        if pos < 0:
            stages.append("无"); roles.append("无题材")
            ttypes.append("无题材"); provens.append(False); themes.append(None)
            continue
        prev = dates[pos]
        cons = s2c.get(r.ts_code, [])
        best = None          # (zt_all, stage, theme, role, ttype, proven)
        for con in cons:
            k = (prev, con)
            if k not in td_key:
                continue
            wave_no, zt_all, mx_ht, age, leader = td_key[k]
            st = theme_stage(wave_no, zt_all)
            if st == "无":
                continue
            # 题材类型(可持续 vs 消息面催化)
            if zt_all >= SUS_MIN_ZT and (mx_ht >= SUS_MIN_HT
                                         or age >= SUS_MIN_AGE):
                tt = "可持续型"
            elif zt_all <= NEWS_MAX_ZT and mx_ht == 1 and age == 1:
                tt = "消息面型"
            else:
                tt = "中间型"
            if best is None or zt_all > best[0]:
                best = (zt_all, st, con,
                        "昨日龙头" if (leader and leader == r.ts_code)
                        else "昨日题材内",
                        tt, wave_no >= 2)
        if best is None:
            stages.append("无"); roles.append("无题材")
            ttypes.append("无题材"); provens.append(False); themes.append(None)
        else:
            stages.append(best[1]); themes.append(best[2])
            roles.append(best[3]); ttypes.append(best[4]); provens.append(best[5])
    d["theme_stage"] = stages
    d["role"] = roles
    d["theme"] = themes
    d["theme_type"] = ttypes
    d["proven"] = provens
    say(f"  阶段: {dict(pd.Series(stages).value_counts())}")
    say(f"  角色: {dict(pd.Series(roles).value_counts())}")
    say(f"  题材类型: {dict(pd.Series(ttypes).value_counts())}")
    say(f"  已验证(wave≥2): {sum(provens)} ({sum(provens)/len(provens)*100:.1f}%)")
    # 存题材维度缓存(供研究43 显著性检验复用, 避免重算)
    DIMS = Path(__file__).resolve().parent.parent / "data" / "factor" \
        / "themedims_panel.parquet"
    d[["trade_date", "ts_code", "theme_stage", "role", "theme",
       "theme_type", "proven"]].to_parquet(DIMS, index=False)
    say(f"  题材维度缓存 → {DIMS}")
    return d


def stat(sub: pd.DataFrame, tgt: int) -> tuple:
    from core.exit_rules import pl_stats
    ret, ok = f"e{tgt}_ret", f"e{tgt}_ok"
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

    say("交叉分析…")
    # 主表: 题材类型 × 角色 × 入场档
    res_tt = {}
    for tgt in TARGETS:
        t = int(tgt)
        for tt in TTYPES:
            for ro in ROLES:
                res_tt[(t, tt, ro)] = stat(
                    d[(d["theme_type"] == tt) & (d["role"] == ro)], t)
    # 三维: 阶段 × 题材类型 × 角色(只看爆发期, 样本最足)
    res_3d = {}
    for tgt in TARGETS:
        t = int(tgt)
        for st in ("爆发", "鱼尾"):
            for tt in ("可持续型", "消息面型"):
                for ro in ("昨日龙头", "昨日题材内"):
                    res_3d[(t, st, tt, ro)] = stat(
                        d[(d["theme_stage"] == st) & (d["theme_type"] == tt)
                          & (d["role"] == ro)], t)
    # proven 维度
    res_pv = {}
    for tgt in TARGETS:
        t = int(tgt)
        for pv in (True, False):
            for ro in ROLES:
                res_pv[(t, pv, ro)] = stat(
                    d[(d["proven"] == pv) & (d["role"] == ro)], t)
    write_report(d, res_tt, res_3d, res_pv, a.start, a.end)
    say(f"\n报告已写入 {OUT/'42_theme_sustainability.md'}")


def write_report(d, res_tt, res_3d, res_pv, start, end):
    L = []
    A = L.append
    A("# 研究42: 题材可持续性分类 × 阶段 × 角色\n")
    A(f"- 窗口 {start}~{end}, 全宇宙 **{len(d)}** 条票·日")
    A("- 口径沿用研究40/41: 宇宙=盘中达≥目标档, 入场=阈值+实测滑点, "
      "离场=日线近似, 阶段/角色/题材类型全部**锚定 T-1**\n")

    A("## 为何做\n")
    A("研究41 发现「爆发 × 昨日龙头」最优(+1.7~2.3pp)。但「爆发」=第1波="
      "题材近1年内**首次被炒**, 里面混了两种完全不同的东西:\n")
    A("1. **真爆发** —— 产业趋势启动, 多家涨停/打出高度/持续多日, 可延续")
    A("2. **消息面催化** —— 单一事件驱动(中标/政策/传闻), 1~2家首板, 次日即死\n")
    A("研究41 把两者混算, 稀释了结论。本研究拆开。\n")

    A("## 分类口径(全部 T-1)\n")
    A("| 类型 | 判据 | 占比 |")
    A("|---|---|---|")
    for tt in TTYPES:
        n = int((d["theme_type"] == tt).sum())
        A(f"| {tt} | " + {
            "可持续型": f"zt_all≥{SUS_MIN_ZT} 且 (max_height≥{SUS_MIN_HT} "
                        f"或 theme_age≥{SUS_MIN_AGE})",
            "消息面型": f"zt_all≤{NEWS_MAX_ZT} 且 max_height==1 且 theme_age==1",
            "中间型": "其余",
            "无题材": "昨日无在场题材",
        }[tt] + f" | {n/len(d)*100:.1f}% |")
    A("")
    A("⚠ **不修改阶段语义**(记忆定稿: 阶段是语义概念, 不得为提区分度而扭曲)。"
      "本研究新增**独立的题材类型维度**, 不动 `theme_stage` 定义。\n")

    A("## 题材类型 × 角色\n")
    for tgt in TARGETS:
        t = int(tgt)
        A(f"### {t}% 入场档\n")
        A("| 题材类型 | 角色 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% |")
        A("|---|---|---|---|---|---|---|")
        for tt in TTYPES:
            for ro in ROLES:
                n, sr, wr, ra, ex = res_tt[(t, tt, ro)]
                if n < 200:
                    A(f"| {tt} | {ro} | {n} | 样本不足 | - | - | - |")
                    continue
                A(f"| {tt} | {ro} | {n} | {sr} | {wr} | {ra} | **{ex}** |")
        A("")

    A("## 三维交叉: 阶段 × 题材类型 × 角色\n")
    A("验证「爆发期」内部是否真的需要区分可持续 vs 消息面:\n")
    for tgt in TARGETS:
        t = int(tgt)
        A(f"### {t}% 入场档\n")
        A("| 阶段 | 题材类型 | 角色 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% |")
        A("|---|---|---|---|---|---|---|---|")
        for st in ("爆发", "鱼尾"):
            for tt in ("可持续型", "消息面型"):
                for ro in ("昨日龙头", "昨日题材内"):
                    n, sr, wr, ra, ex = res_3d[(t, st, tt, ro)]
                    if n < 200:
                        A(f"| {st} | {tt} | {ro} | {n} | 样本不足 | - | - | - |")
                        continue
                    A(f"| {st} | {tt} | {ro} | {n} | {sr} | {wr} | {ra} "
                      f"| **{ex}** |")
        A("")

    A("## 已验证维度(wave_no≥2)\n")
    A("题材被炒过第2轮以上 = 历史已验证过, 不是首次出现的消息面:\n")
    for tgt in TARGETS:
        t = int(tgt)
        A(f"### {t}% 入场档\n")
        A("| 是否已验证 | 角色 | 样本 | 封板率% | 胜率% | 盈亏比 | 期望% |")
        A("|---|---|---|---|---|---|---|")
        for pv in (True, False):
            for ro in ROLES:
                n, sr, wr, ra, ex = res_pv[(t, pv, ro)]
                if n < 200:
                    A(f"| {'已验证' if pv else '首次'} | {ro} | {n} "
                      f"| 样本不足 | - | - | - |")
                    continue
                A(f"| {'已验证' if pv else '首次'} | {ro} | {n} | {sr} | {wr} "
                  f"| {ra} | **{ex}** |")
        A("")

    A("## 约束\n")
    A("1. **归属是静态近似** —— con2stock 为当前概念成分, 历史归属会漂移")
    A("2. **题材类型阈值是实测分布定的**(可持续型8.1%/消息面型45.6%), "
      "未做二次拟合")
    A("3. **角色只区分龙头/非龙头** —— 补涨/共振需历史归属集合, 数据不支持")
    A("4. **离场是日线近似**, **入场价是建模值**(阈值+实测滑点)")
    A("5. 样本 <200 的组合不列入(避免小样本偶然)")
    A("6. **只筛选不进生产**")

    (OUT / "42_theme_sustainability.md").write_text("\n".join(L),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
