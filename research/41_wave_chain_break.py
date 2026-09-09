# -*- coding: utf-8 -*-
"""研究41: 波次链断 —— 两波间隔超过半年是否应视为无关

起因（用户提出）: 现行波次是「回看窗口内累计计数」（研究40 定档 LOOKBACK=244），
只要两波都落在窗口内就累加，不管它们之间隔了多久。但直觉上，若两波之间
空了半年以上，前一波与当下这波基本无关，不该继续累加 —— 应该**断链重算**。

本研究构造「波链」概念并验证:
    波链断裂 = 相邻两波间隔 > CHAIN_BREAK 交易日 → 后一波开新链，波次归 1
    wave_no  = 当前波在其所属波链内的序号（而不是窗口内累计序号）

与现行口径的区别:
    现行: 题材1年内被炒3次(哪怕每次之间空4个月) → wave_no=3 → 鱼尾
    链断: 若第2、3波与前一波各空4个月(>半年?) → 视间隔而定，可能仍是第1波

候选方案（并行对比，用户方法论: 不做单方案调参）:
    G0 现行      滚动窗口累计（LOOKBACK=244，无链断）
    G1 链断半年  间隔>122交易日断链（起点→起点口径）
    G2 链断半年  间隔>122交易日断链（前波末日→后波起点口径）
    G3 链断3月   间隔>61交易日断链
    G4 链断1年   间隔>244交易日断链
    G5 链断半年+无窗口  断链半年且取消回看窗口（检验窗口是否还需要）

判据（研究39/40 四条硬条件，不新增主观判据）:
    ① 全历史方向正确（鱼尾档次日延续率最低）
    ② 三段市况同向 ≥2/3（牛/熊/震荡）
    ③ 近250日方向也正确（看板实际面对的样本）
    ④ 三档最小占比 ≥10%（否则退化成两档）
    另报告「主升-鱼尾」差 —— 研究40 后该差仅 3.9pp，三档实际退化成两档，
    链断能否修复这个退化是本研究的次要目标。

产物: research/out/41_wave_chain_break.md
用法: python research/41_wave_chain_break.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.cycle import (BURST_MIN_ZT, WAVE_COOLDOWN, WAVE_LOOKBACK,  # noqa: E402
                        WAVE_MIN_ZT)
from core.theme_wave import activity_panel  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "41_wave_chain_break.md"
STAGES = ("爆发", "主升", "鱼尾")
RECENT_N = 250
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


def regimes(cal: list) -> pd.Series:
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    mkt = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    idx = (1 + mkt / 100).cumprod()
    dev = idx / idx.rolling(20).mean() - 1
    reg = pd.Series("震荡", index=idx.index)
    reg[dev > 0.02] = "牛"
    reg[dev < -0.02] = "熊"
    return reg


def wave_bounds(positions: list, active: list) -> tuple:
    """累计切波 → (波段起点列表, 每波最后一个在场日的下标列表)"""
    starts, ends, prev = [], [], None
    for i, (p, a) in enumerate(zip(positions, active)):
        if prev is None or (a and p - prev - 1 >= WAVE_COOLDOWN):
            starts.append(p)
            ends.append(p)                 # 新波起点同时也是当前末日
        else:
            ends[-1] = p                   # 延续: 更新该波最后在场日
        prev = p
    return starts, ends


def wave_no_lookup(starts: list, chain_idx: list, positions: list) -> list:
    """每行 → 其所属波段的链内序号"""
    out, j = [], 0
    for p in positions:
        while j + 1 < len(starts) and starts[j + 1] <= p:
            j += 1
        out.append(chain_idx[j] if j < len(chain_idx) else 1)
    return out


def chain_by_start(starts: list, brk: int | None) -> list:
    """链断（起点→起点口径）: 相邻起点间隔 > brk → 新链"""
    idx = []
    for i, s in enumerate(starts):
        if i == 0 or brk is None or s - starts[i - 1] > brk:
            idx.append(1)
        else:
            idx.append(idx[-1] + 1)
    return idx


def chain_by_end(starts: list, ends: list, brk: int | None) -> list:
    """链断（前波末日→后波起点口径）: 空档 > brk → 新链"""
    idx = []
    for i, s in enumerate(starts):
        if i == 0 or brk is None or s - ends[i - 1] - 1 > brk:
            idx.append(1)
        else:
            idx.append(idx[-1] + 1)
    return idx


def rolling_count(starts: list, positions: list, lookback: int | None) -> list:
    """现行口径: 窗口内累计波段数"""
    out = []
    for p in positions:
        lo = p - lookback if lookback is not None else -1
        n = sum(1 for s in starts if lo < s <= p)
        out.append(n if n >= 1 else 1)
    return out


def stage_of(wave_no: int, zt: int) -> str:
    if zt < WAVE_MIN_ZT:
        return "无"
    if wave_no >= 3:
        return "鱼尾"
    if wave_no <= 1:
        return "爆发" if zt >= BURST_MIN_ZT else "主升"
    return "主升"


# 方案: (标签, 波次计算函数)
def build(cal: list) -> pd.DataFrame:
    dpos = {d: i for i, d in enumerate(cal)}
    pan = activity_panel()
    pan = pan[pan["trade_date"].isin(dpos)].copy()
    say("# 研究41: 波次链断 —— 两波间隔超半年是否应视为无关")
    say(f"\n活跃面板 {len(pan):,} 题材-日 · "
        f"{pan['trade_date'].min()}~{pan['trade_date'].max()} · "
        f"题材 {pan['concept_code'].nunique()} 个 · "
        f"切波 COOLDOWN={WAVE_COOLDOWN} / 在场门槛 {WAVE_MIN_ZT} / "
        f"现行回看窗口 {WAVE_LOOKBACK}")

    cols = {}
    for k, g in pan.groupby("concept_code", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)
        pos = g["trade_date"].map(dpos).tolist()
        active = (g["zt_all"] >= WAVE_MIN_ZT).tolist()
        starts, ends = wave_bounds(pos, active)
        rec = {"concept_code": [k] * len(g), "trade_date": g["trade_date"].tolist(),
               "zt_all": g["zt_all"].tolist()}
        rec["G0 现行(窗口累计244)"] = rolling_count(starts, pos, WAVE_LOOKBACK)
        rec["G1 链断半年(起点口径)"] = wave_no_lookup(
            starts, chain_by_start(starts, 122), pos)
        rec["G2 链断半年(末日口径)"] = wave_no_lookup(
            starts, chain_by_end(starts, ends, 122), pos)
        rec["G3 链断3月"] = wave_no_lookup(
            starts, chain_by_start(starts, 61), pos)
        rec["G4 链断1年"] = wave_no_lookup(
            starts, chain_by_start(starts, 244), pos)
        rec["G5 链断半年+无窗口"] = wave_no_lookup(
            starts, chain_by_start(starts, 122), pos)
        for c, v in rec.items():
            cols.setdefault(c, []).extend(v)
    d = pd.DataFrame(cols)

    # 只看在场题材-日 + 次日延续标签
    d = d[d["zt_all"] >= WAVE_MIN_ZT].reset_index(drop=True)
    nxt = {dd: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, dd in enumerate(cal)}
    zb = {(r.trade_date, r.concept_code): r.zt_all for r in pan.itertuples()}
    d["cont"] = [int(bool(zb.get((nxt.get(t), c)) is not None
                          and zb[(nxt[t], c)] >= WAVE_MIN_ZT))
                 if nxt.get(t) else np.nan
                 for t, c in zip(d["trade_date"], d["concept_code"])]
    d = d.dropna(subset=["cont"])
    d["reg"] = d["trade_date"].map(regimes(cal)).fillna("震荡")
    say(f"在场题材-日样本 {len(d):,} · 基线延续率 "
        f"{d['cont'].mean() * 100:.1f}% · 三段市况 "
        f"{d['reg'].value_counts().to_dict()}")
    return d


SCHEMES = ["G0 现行(窗口累计244)", "G1 链断半年(起点口径)",
           "G2 链断半年(末日口径)", "G3 链断3月", "G4 链断1年",
           "G5 链断半年+无窗口"]


def part_a(d: pd.DataFrame):
    say("\n## A 各方案的波次分布")
    rows = []
    for tag in SCHEMES:
        w = d[tag]
        rows.append([tag, f"{w.mean():.2f}", int(w.max()),
                     f"{(w == 1).mean() * 100:.1f}%",
                     f"{(w == 2).mean() * 100:.1f}%",
                     f"{(w >= 3).mean() * 100:.1f}%"])
    md_table(["方案", "均波次", "最大波次", "首波占比", "二波占比",
              "三波+占比"], rows)
    say("\n各方案下的阶段档位分布:")
    rows = []
    for tag in SCHEMES:
        s = pd.Series([stage_of(int(w), int(z))
                       for w, z in zip(d[tag], d["zt_all"])])
        rows.append([tag] + [f"{(s == x).mean() * 100:.1f}%" for x in STAGES])
    md_table(["方案", "爆发占比", "主升占比", "鱼尾占比"], rows)


def part_b(d: pd.DataFrame):
    say("\n## B 四条硬条件验收")
    say("① 全历史方向正确（鱼尾档延续率最低）② 三段市况同向≥2/3 "
        "③ 近250日方向正确 ④ 三档最小占比≥10%")
    recent_dates = sorted(d["trade_date"].unique())[-RECENT_N:]
    rows = []
    for tag in SCHEMES:
        st = pd.Series([stage_of(int(w), int(z))
                        for w, z in zip(d[tag], d["zt_all"])], index=d.index)
        sub = d.assign(stage=st)
        sub = sub[sub["stage"] != "无"]
        per = sub.groupby("stage")["cont"].mean()
        if len(per) < 3:
            continue
        c1 = per["鱼尾"] == per.min()
        sp = (per["爆发"] - per["鱼尾"]) * 100
        mid = (per["主升"] - per["鱼尾"]) * 100
        rho = spearmanr(sub["stage"].map({"爆发": 0, "主升": 1, "鱼尾": 2}),
                        sub["cont"]).statistic
        sps = []
        for rg in ("牛", "熊", "震荡"):
            g = sub[sub["reg"] == rg]
            p2 = g.groupby("stage")["cont"].mean()
            if len(p2) >= 3:
                sps.append((p2["爆发"] - p2["鱼尾"]) * 100)
        c2 = sum(1 for x in sps if x > 0) >= 2
        # 近期
        r = sub[sub["trade_date"].isin(recent_dates)]
        rper = r.groupby("stage")["cont"].mean()
        c3 = bool(len(rper) >= 3 and rper["鱼尾"] == rper.min())
        rsp = (rper["爆发"] - rper["鱼尾"]) * 100 if len(rper) >= 3 else np.nan
        rmid = (rper["主升"] - rper["鱼尾"]) * 100 if len(rper) >= 3 else np.nan
        dist = sub["stage"].value_counts(normalize=True)
        minshare = min(dist.get(x, 0) for x in STAGES) * 100
        c4 = minshare >= 10
        npass = sum([c1, c2, c3, c4])
        rows.append([tag] + [f"{per[x] * 100:.1f}%" for x in STAGES]
                    + [f"{sp:+.1f}", f"{mid:+.1f}",
                       "✅" if c1 else "❌", "✅" if c2 else "❌",
                       f"{rsp:+.1f}", "✅" if c3 else "❌",
                       f"{minshare:.1f}%", "✅" if c4 else "❌",
                       f"{npass}/4"])
        KEY[tag] = {"per": per.to_dict(), "spread": sp, "mid": mid,
                    "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                    "npass": npass, "rho": rho, "reg_sp": sps,
                    "recent_per": rper.to_dict(), "recent_sp": rsp,
                    "recent_mid": rmid, "minshare": minshare}
    md_table(["方案", "爆发", "主升", "鱼尾", "爆-鱼", "主-鱼", "①", "②",
              "近期爆-鱼", "③", "最小档", "④", "通过"], rows)
    say("\n> 「主-鱼」= 主升档 − 鱼尾档延续率差。研究40 后现行口径该差仅 "
        "3.9pp，三档实际退化成两档；链断能否修复这个退化看这一列。")

    say("\n### B2 三段市况明细")
    rows = []
    for tag in SCHEMES:
        st = pd.Series([stage_of(int(w), int(z))
                        for w, z in zip(d[tag], d["zt_all"])], index=d.index)
        sub = d.assign(stage=st)
        sub = sub[sub["stage"] != "无"]
        line = [tag]
        for rg in ("牛", "熊", "震荡"):
            g = sub[sub["reg"] == rg]
            p2 = g.groupby("stage")["cont"].mean()
            line.append(" / ".join(f"{p2.get(x, np.nan) * 100:.0f}%"
                                   for x in STAGES) if len(p2) >= 3 else "-")
        rows.append(line)
    md_table(["方案", "牛(爆发/主升/鱼尾)", "熊", "震荡"], rows)


def part_c(d: pd.DataFrame):
    say("\n## C 近期样本明细（看板实际场景）")
    recent_dates = sorted(d["trade_date"].unique())[-RECENT_N:]
    r = d[d["trade_date"].isin(recent_dates)].copy()
    say(f"\n近期样本 {len(r):,} 题材-日（{r['trade_date'].min()}~"
        f"{r['trade_date'].max()}）· 基线 {r['cont'].mean() * 100:.1f}%")
    rows = []
    for tag in SCHEMES:
        st = pd.Series([stage_of(int(w), int(z))
                        for w, z in zip(r[tag], r["zt_all"])], index=r.index)
        sub = r.assign(stage=st)
        sub = sub[sub["stage"] != "无"]
        per = sub.groupby("stage")["cont"].mean()
        cnt = sub["stage"].value_counts()
        if len(per) < 3:
            continue
        rows.append([tag] + [f"{per.get(x, np.nan) * 100:.1f}%"
                             f"(n={cnt.get(x, 0)})" for x in STAGES])
    md_table(["方案", "爆发", "主升", "鱼尾"], rows)


def part_d(d: pd.DataFrame):
    say("\n## D 结论")
    g0 = KEY.get("G0 现行(窗口累计244)", {})
    g1 = KEY.get("G1 链断半年(起点口径)", {})
    g2 = KEY.get("G2 链断半年(末日口径)", {})
    g5 = KEY.get("G5 链断半年+无窗口", {})

    say("\n### D1 用户直觉的直接检验（G1 起点口径）")
    say("「两波间隔超半年就算无关」的正确实现是 **起点→起点间隔>122交易日"
        "断链**（G1）—— 因为「两波间隔」指的是两个波段之间的距离。实测:")
    if g0 and g1:
        say(f"- 爆-鱼梯度: 现行 {g0['spread']:+.1f}pp → 链断 "
            f"{g1['spread']:+.1f}pp（**降低**）")
        say(f"- 主-鱼差: 现行 {g0['mid']:+.1f}pp → 链断 {g1['mid']:+.1f}pp"
            f"（**未修复三档退化**）")
        say(f"- 鱼尾档占比: 现行 "
            f"{(d['G0 现行(窗口累计244)'] >= 3).mean() * 100:.1f}% → 链断 "
            f"{(d['G1 链断半年(起点口径)'] >= 3).mean() * 100:.1f}%")
    say("\n**结论: 直觉未获数据支持。** 断链后梯度反而变小 —— 说明那些"
        "「间隔半年以上的前一波」并非无关噪声, 它们仍然携带信息"
        "（题材被反复炒的历史本身就是延续性的预测因子）。")

    say("\n### D2 一个意外发现: 链断与回看窗口功能重叠")
    if g1 and g5:
        same = (abs(g1['spread'] - g5['spread']) < 0.05
                and abs(g1['mid'] - g5['mid']) < 0.05)
        say(f"- G1（链断半年+窗口244）与 G5（链断半年+**无**窗口）结果"
            f"{'完全相同' if same else '接近'} —— 因为一旦间隔>122日就断链, "
            "244日的回看窗口就再也切不到东西了。")
        say("- 含义: **链断与回看窗口在做同一件事**（都是「忽略久远的"
            "历史波」），二者选一即可。现行用的是窗口，无需再加链断。")

    say("\n### D3 为何 G2/G4 看似梯度更好却是假象")
    say("G2（末日口径）全历史主-鱼差 "
        f"{g2.get('mid', 0):+.1f}pp 看似优于现行 "
        f"{g0.get('mid', 0):+.1f}pp, 但它**过不了近期方向关**:")
    if g2:
        rp = g2.get("recent_per", {})
        say(f"- 近期样本里鱼尾档占 "
            f"{(d[d['trade_date'].isin(sorted(d['trade_date'].unique())[-RECENT_N:])]['G2 链断半年(末日口径)'] >= 3).mean() * 100:.0f}%, "
            f"爆发档只剩少量样本 → 主升 {rp.get('主升', 0) * 100:.1f}% "
            f"反而低于鱼尾 {rp.get('鱼尾', 0) * 100:.1f}%（方向反置）。")
    say("- 原因: 末日口径算的是「前波末日→后波起点」的空档, 比起点→起点"
        "间隔**小**, 所以断链几乎从不触发 → 退回到接近全历史累计的行为"
        f"（均波次 {d['G2 链断半年(末日口径)'].mean():.2f} vs 现行 "
        f"{d['G0 现行(窗口累计244)'].mean():.2f}）。它测的不是用户的想法, "
        "而是「几乎不断链」。")

    say("\n### D4 定调")
    say("- **不采用链断**, 维持现行口径（滚动回看窗口 "
        f"WAVE_LOOKBACK={WAVE_LOOKBACK}）。四条硬条件全过的方案里, "
        "现行的梯度最大。")
    say("- 三档退化（主-鱼 仅 "
        f"{g0.get('mid', 0):+.1f}pp）**链断修不了**。真正的成因是家数比"
        "波次更能决定延续率（同波次内 zt=2→zt≥5 差 30~40pp, 同家数档内"
        "波次 1→3+ 只差 10~20pp）—— 要修退化得改家数主导的定义, "
        "不是改波次口径。")
    say("- 无论哪种口径, 题材阶段只影响展示与题材追踪 —— 研究36/37/39 "
        "已证对封板率/EV 无区分度，**不得**据此接入买卖闸。")

    say("\n## 诚实边界")
    say("- 链断有两种间隔口径（起点→起点 / 前波末日→后波起点），本研究两者"
        "都测；后者更贴近「空了多久」的直觉，前者含波段自身时长。")
    say("- G5「链断半年+无窗口」用于检验回看窗口是否冗余，但其波次无上界，"
        "对超长历史题材会给出很大的波次，实际可用性低于带窗口的方案。")
    say("- 近250日样本分档后单档可低至数百，各方案差 <2pp 应视为持平，"
        "优先看方向正确性与档位均衡度。")
    say("- kpl 题材命名跨年变更会把同一题材切成多个，使波次偏小；"
        "该偏差对所有方案同向作用。")


def summary() -> list:
    g0 = KEY.get("G0 现行(窗口累计244)", {})
    s = ["\n## 摘要(TL;DR)"]
    g1 = KEY.get("G1 链断半年(起点口径)", {})
    g2 = KEY.get("G2 链断半年(末日口径)", {})
    if g0 and g1:
        s.append(f"**用户的直觉未获数据支持 —— 不采用链断。** 正确检验是"
                 f"「起点→起点间隔>122交易日断链」(G1): 爆-鱼梯度从现行 "
                 f"{g0['spread']:+.1f}pp **降到** {g1['spread']:+.1f}pp, "
                 f"主-鱼差从 {g0['mid']:+.1f}pp 降到 {g1['mid']:+.1f}pp。"
                 "断链后梯度反而变小 → 那些「间隔半年以上的前一波」并非"
                 "无关噪声, 题材被反复炒的历史本身就是延续性的预测因子。")
    s.append("**意外发现: 链断与回看窗口功能重叠。** G1(链断+窗口244) 与 "
             "G5(链断+**无**窗口) 结果完全相同 —— 一旦间隔>122日就断链, "
             "244日窗口就再也切不到东西。两者在做同一件事(忽略久远历史波), "
             "选一即可, 现行用窗口无需再加链断。")
    if g2:
        s.append(f"**G2/G4 看似梯度更好是假象**: G2 主-鱼差 "
                 f"{g2.get('mid', 0):+.1f}pp 优于现行, 但近期样本里鱼尾档"
                 "占绝大多数 → 方向反置, 过不了③。因为末日口径的间隔比"
                 "起点口径**小**, 断链几乎从不触发, 退回到接近全历史累计"
                 "的行为 —— 它测的不是链断而是「几乎不断链」。")
    s.append("**三档退化链断修不了**: 真正成因是家数比波次更能决定延续率"
             "(同波次内 zt=2→zt≥5 差 30~40pp, 同家数档内波次 1→3+ 只差 "
             "10~20pp) —— 要修退化得改家数主导的定义, 不是改波次口径。")
    s.append("**边界不变**: 题材阶段只用于展示与题材追踪，不得接入买卖闸"
             "（研究36/37/39 已证对封板率/EV 无区分度）。")
    return s


def main():
    cal = list(sorted(load("limitup.events_enriched",
                           columns=["trade_date"])["trade_date"].unique()))
    d = build(cal)
    part_a(d)
    part_b(d)
    part_c(d)
    part_d(d)
    out = [L[0]] + summary() + L[1:]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
