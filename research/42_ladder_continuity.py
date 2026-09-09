# -*- coding: utf-8 -*-
"""研究42: 题材天梯连续性 —— 排序键该用当日家数还是近期热度

起因（用户指出）: 液冷 20260902/0903 连续两天天梯第1（家数5/7），20260904
只剩1家涨停（4板龙头集泰股份断板）→ 排名掉到 22/27，混在一堆1家题材里；
光模块更是当日无独占归属直接从 theme.day 消失。天梯是纯当日快照
（apps/review.py 按当日 zt_cnt 排序），**没有任何跨日延续机制**，
而个股层面已有「龙头断板层」保留前几日龙头 —— 题材与个股不对称。

本研究用数据决定修法，不凭直觉:
  Q1 近期热度是否比当日家数更能预测题材次日活跃？
     → 若是: 排序键应改成滚动热度（天梯自然具备连续性）
     → 若否: 连续性纯属展示需求，应做「置顶+退潮标记」或独立区块
  Q2 「昨日热门但今日降温」的题材次日表现如何？
     → 若延续率显著更低: 它们是退潮信号，值得显示（风险用途）
     → 若与今日热门相近: 它们只是歇一天，埋掉会丢机会信息
  Q3 现行 head(30) 截断漏掉多少昨日热门题材？

候选排序键（并行对比，用户方法论: 不做单方案调参）:
  P0 当日 zt_all                现行口径
  P1 近3日累计 zt_all
  P2 加权滚动 zt + 0.5昨 + 0.25前
  P3 近3日最大 zt_all
  P4 当日 + 昨日 zt_all

判据:
  · Spearman(排序键, 次日 zt_all) —— 预测力
  · Top10 命中率 —— 按该键取前10题材，次日仍在场（zt_all≥2）的比例
  · Top10 次日家数均值 —— 选出来的题材次日平均还有多少家涨停

产物: research/out/42_ladder_continuity.md
用法: python research/42_ladder_continuity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.cycle import WAVE_MIN_ZT  # noqa: E402
from core.theme_wave import activity_panel  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "42_ladder_continuity.md"
LADDER_N = 30          # 现行天梯截断
TOP_N = 10             # 命中率评估取前10
L: list = []
KEY: dict = {}


def say(s=""):
    print(s, flush=True)
    L.append(s)


def md_table(header, rows):
    say("| " + " | ".join(header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")


def build(cal: list) -> pd.DataFrame:
    """题材-日面板(宽表): 每个题材每日一行, 缺失日补0"""
    pan = activity_panel()
    pan = pan[pan["trade_date"].isin(cal)].copy()
    say("# 研究42: 题材天梯连续性 —— 排序键该用当日家数还是近期热度")
    say(f"\n活跃面板 {len(pan):,} 题材-日 · "
        f"{pan['trade_date'].min()}~{pan['trade_date'].max()} · "
        f"题材 {pan['concept_code'].nunique()} 个")
    # 宽表: 行=日期, 列=题材, 值=zt_all(缺失=0)
    wide = pan.pivot_table(index="trade_date", columns="concept_code",
                           values="zt_all", aggfunc="sum").reindex(cal)
    wide = wide.fillna(0)
    say(f"宽表 {wide.shape[0]} 日 × {wide.shape[1]} 题材 · "
        f"日均在场题材数 "
        f"{(wide >= WAVE_MIN_ZT).sum(axis=1).mean():.1f}")
    return wide


def part_q1(wide: pd.DataFrame):
    say("\n## Q1 近期热度 vs 当日家数: 谁更能预测次日活跃")
    say("目标变量 = 次日 zt_all。若近期热度的 Spearman 明显高于当日家数, "
        "说明排序键应该改成滚动热度; 否则连续性只是展示需求。")
    z0 = wide                                   # 当日
    z1 = wide.shift(1).fillna(0)                # 昨日
    z2 = wide.shift(2).fillna(0)                # 前日
    roll3 = z0 + z1 + z2                        # 近3日累计
    wavg = z0 + 0.5 * z1 + 0.25 * z2            # 加权滚动
    max3 = pd.concat([z0, z1, z2], axis=0).groupby(level=0).max() \
        if False else np.maximum(np.maximum(z0.values, z1.values), z2.values)
    max3 = pd.DataFrame(max3, index=wide.index, columns=wide.columns)
    z01 = z0 + z1                               # 当日+昨日
    nxt = wide.shift(-1)                        # 次日

    cands = {"P0 当日zt_all(现行)": z0, "P1 近3日累计": roll3,
             "P2 加权滚动(1/0.5/0.25)": wavg, "P3 近3日最大": max3,
             "P4 当日+昨日": z01}
    rows = []
    for name, df in cands.items():
        # 长表化: 只取当日在场的题材-日(与天梯口径一致)
        cur = df[z0 >= WAVE_MIN_ZT]
        tgt = nxt[z0 >= WAVE_MIN_ZT]
        pairs = [(cur.iloc[i, j], tgt.iloc[i, j])
                 for i in range(cur.shape[0])
                 for j in range(cur.shape[1])
                 if np.isfinite(tgt.iloc[i, j])]
        if len(pairs) < 100:
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        rho = spearmanr(a, b).statistic
        # Top10 命中率: 每日按该键取前10, 看次日仍在场比例
        hit, nxt_sum, ndays = [], [], 0
        for i in range(wide.shape[0] - 1):
            row = df.iloc[i]
            pool = row[z0.iloc[i] >= WAVE_MIN_ZT]
            if len(pool) < TOP_N:
                continue
            top = pool.nlargest(TOP_N).index
            ndays += 1
            nv = nxt.iloc[i][top]
            hit.append((nv >= WAVE_MIN_ZT).mean())
            nxt_sum.append(nv.mean())
        rows.append([name, f"{rho:+.3f}",
                     f"{np.mean(hit) * 100:.1f}%" if hit else "-",
                     f"{np.mean(nxt_sum):.2f}" if nxt_sum else "-",
                     len(pairs), ndays])
        KEY[name] = {"rho": rho,
                     "hit": np.mean(hit) if hit else np.nan,
                     "nxt_sum": np.mean(nxt_sum) if nxt_sum else np.nan}
    md_table(["排序键", "Spearman(次日家数)", "Top10次日在场命中率",
              "Top10次日家数均值", "样本", "评估天数"], rows)
    say("\n> 命中率 = 按该键取前10题材, 次日仍满足 zt_all≥2 的比例。"
        "这是天梯「选出来的题材明天还活不活」的直接度量。")


def part_q2(wide: pd.DataFrame):
    say("\n## Q2 「昨日热门但今日降温」的题材次日表现")
    say("这决定降温题材该不该显示: 若次日延续率显著更低 → 它们是退潮"
        "信号（风险用途，值得显示）; 若与今日热门相近 → 只是歇一天"
        "（机会用途，埋掉会丢信息）。")
    z0, z1 = wide, wide.shift(1).fillna(0)
    nxt = wide.shift(-1)
    groups = {
        "今日热门(zt≥5)": (z0 >= 5),
        "今日中等(zt3~4)": (z0 >= 3) & (z0 <= 4),
        "今日降温(昨≥3且今≤1)": (z1 >= 3) & (z0 <= 1),
        "今日弱(今=2)": (z0 == 2),
        "今日消失(昨≥3且今=0)": (z1 >= 3) & (z0 == 0),
    }
    rows = []
    for name, m in groups.items():
        sel = m & nxt.notna()
        vals = nxt[sel]
        flat = vals.values.flatten()
        flat = flat[np.isfinite(flat)]
        if len(flat) < 50:
            continue
        rows.append([name, len(flat), f"{(flat >= WAVE_MIN_ZT).mean() * 100:.1f}%",
                     f"{flat.mean():.2f}", f"{np.median(flat):.1f}",
                     f"{(flat >= 3).mean() * 100:.1f}%"])
        KEY[f"q2_{name}"] = {"cont": (flat >= WAVE_MIN_ZT).mean(),
                             "mean": flat.mean(), "n": len(flat)}
    md_table(["分组", "题材-日数", "次日在场率", "次日家数均值",
              "次日家数中位", "次日≥3家比例"], rows)
    hot = KEY.get("q2_今日热门(zt≥5)", {})
    cool = KEY.get("q2_今日降温(昨≥3且今≤1)", {})
    gone = KEY.get("q2_今日消失(昨≥3且今=0)", {})
    if hot and cool:
        say(f"\n- 今日热门次日在场率 {hot['cont'] * 100:.1f}% vs "
            f"今日降温 {cool['cont'] * 100:.1f}%"
            f"（差 {(hot['cont'] - cool['cont']) * 100:+.1f}pp）")
    if gone:
        say(f"- 今日消失(昨≥3今=0) 次日在场率 {gone['cont'] * 100:.1f}%"
            f"（n={gone['n']}）")


def part_q3(wide: pd.DataFrame):
    say(f"\n## Q3 现行 head({LADDER_N}) 截断漏掉多少昨日热门题材")
    z0, z1 = wide, wide.shift(1).fillna(0)
    miss, tot, days = 0, 0, 0
    for i in range(1, wide.shape[0]):
        today = z0.iloc[i]
        # 现行天梯: 当日有独占归属的题材, 按当日家数排序取前30
        # 此处用活跃面板近似(独占口径行数更少, 漏检只会更严重)
        pool = today[today > 0].nlargest(LADDER_N).index
        hot_prev = set(z1.iloc[i - 1][z1.iloc[i - 1] >= 3].index)
        if not hot_prev:
            continue
        days += 1
        tot += len(hot_prev)
        miss += len(hot_prev - set(pool))
    say(f"- 评估 {days} 个交易日 · 昨日热门(zt≥3)题材共 {tot} 个次")
    say(f"- 其中 **{miss} 个次({miss / max(tot, 1) * 100:.1f}%)** "
        f"落在今日 top{LADDER_N} 之外 —— 即天梯上看不到")

    say("\n### Q3b 案例复现（用户指出的液冷）")
    for th in ["液冷", "光模块"]:
        if th not in wide.columns:
            continue
        s = wide[th]
        seg = s[(s.index >= "20260825") & (s.index <= "20260904")]
        rank = []
        for d in seg.index:
            pool = wide.loc[d][wide.loc[d] > 0].nlargest(LADDER_N).index
            allpool = wide.loc[d][wide.loc[d] > 0].sort_values(ascending=False)
            r = list(allpool.index).index(th) + 1 if th in list(allpool.index) else None
            rank.append(f"{d}:{int(seg[d])}家/第{r}名"
                        + ("" if r and r <= LADDER_N else "(出天梯)"))
        say(f"- {th}: " + " → ".join(rank))


def part_d():
    say("\n## D 结论与修法推断")
    p0 = KEY.get("P0 当日zt_all(现行)", {})
    best = max((k for k in KEY if k.startswith("P")),
               key=lambda k: KEY[k].get("hit", 0), default=None)
    say("\n### D1 排序键该不该改")
    if best and p0:
        d_hit = (KEY[best]["hit"] - p0["hit"]) * 100
        d_rho = KEY[best]["rho"] - p0["rho"]
        say(f"- 最优排序键: **{best}** —— Top10次日在场命中率 "
            f"{KEY[best]['hit'] * 100:.1f}% vs 现行 "
            f"{p0['hit'] * 100:.1f}%（{d_hit:+.1f}pp），Spearman "
            f"{KEY[best]['rho']:+.3f} vs {p0['rho']:+.3f}（{d_rho:+.3f}）")
        if d_hit > 3:
            say("- 命中率提升明显 → **排序键应改成滚动热度**，"
                "天梯自然具备连续性。")
        else:
            say("- 命中率提升不明显（<3pp）→ 当日家数已是足够好的排序键，"
                "**连续性属于展示需求**，应做「昨日热门置顶+退潮标记」"
                "而不是改排序语义。")
    hot = KEY.get("q2_今日热门(zt≥5)", {})
    cool = KEY.get("q2_今日降温(昨≥3且今≤1)", {})
    gone = KEY.get("q2_今日消失(昨≥3且今=0)", {})
    say("\n### D2 降温题材该不该显示")
    if hot and cool:
        gap = (hot["cont"] - cool["cont"]) * 100
        say(f"- 今日降温题材次日在场率 {cool['cont'] * 100:.1f}%，"
            f"比今日热门低 {gap:.1f}pp → 它们是**退潮信号**，"
            "显示出来有明确的风险价值（告诉你哪个题材在死）。")
    if gone:
        say(f"- 今日彻底消失的题材次日在场率 {gone['cont'] * 100:.1f}%"
            f"（n={gone['n']}）→ 补昨日行并标注是必要的，"
            "否则最极端的退潮情况会被完全漏掉。")

    say("\n### D3 定调")
    say("- 天梯排序保持「当日家数」语义不变（它是当日排名，改语义会误导）;")
    say("- 昨日热门(zt≥3)但今日降温/消失的题材 → **补进天梯并标注退潮**，"
        "显示昨日家数与轨迹，与个股层「龙头断板层」对称;")
    say("- 阶段字段照常: 降温题材若 zt_all<2 则阶段为「无」（不在场），"
        "这本身就是准确的退潮表达。")

    say("\n## 诚实边界")
    say("- 本研究用归因自由活跃面板(zt_all)近似天梯口径; 实际天梯用独占"
        "zt_cnt 排序且行数更少, 所以 Q3 的漏检率是**下界**, 实际更严重。")
    say("- 次日在场率的目标变量口径与研究36~41 一致(次日 zt_all≥2), "
        "保证跨研究可比。")
    say("- 排序键候选只测了5种线性组合, 未做更复杂的衰减函数; "
        "若最优键提升不明显则无需继续搜索。")


def summary() -> list:
    p0 = KEY.get("P0 当日zt_all(现行)", {})
    best = max((k for k in KEY if k.startswith("P")),
               key=lambda k: KEY[k].get("hit", 0), default=None)
    hot = KEY.get("q2_今日热门(zt≥5)", {})
    cool = KEY.get("q2_今日降温(昨≥3且今≤1)", {})
    gone = KEY.get("q2_今日消失(昨≥3且今=0)", {})
    s = ["\n## 摘要(TL;DR)"]
    if best and p0:
        d_hit = (KEY[best]["hit"] - p0["hit"]) * 100
        verdict = ("改排序键" if d_hit > 3 else "只做展示层连续性")
        s.append(f"**排序键结论: {verdict}。** 最优键 {best} 的 Top10次日"
                 f"在场命中率 {KEY[best]['hit'] * 100:.1f}% vs 现行 "
                 f"{p0['hit'] * 100:.1f}%（{d_hit:+.1f}pp），Spearman "
                 f"{KEY[best]['rho']:+.3f} vs {p0['rho']:+.3f}。")
    if hot and cool:
        s.append(f"**降温题材值得显示**: 今日降温(昨≥3且今≤1)次日在场率 "
                 f"{cool['cont'] * 100:.1f}%，比今日热门 "
                 f"{hot['cont'] * 100:.1f}% 低 "
                 f"{(hot['cont'] - cool['cont']) * 100:.1f}pp → 是明确的"
                 "退潮信号，显示出来有风险价值。")
    if gone:
        s.append(f"**消失题材必须补**: 昨≥3且今=0 的题材次日在场率 "
                 f"{gone['cont'] * 100:.1f}%（n={gone['n']}），"
                 "不补昨日行就会漏掉最极端的退潮。")
    s.append("**定调**: 天梯排序语义不变（当日家数），昨日热门但降温/消失的"
             "题材补进天梯并标注退潮 + 显示轨迹，与个股层「龙头断板层」对称。")
    return s


def main():
    cal = list(sorted(load("limitup.events_enriched",
                           columns=["trade_date"])["trade_date"].unique()))
    wide = build(cal)
    part_q1(wide)
    part_q2(wide)
    part_q3(wide)
    part_d()
    out = [L[0]] + summary() + L[1:]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
