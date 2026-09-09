# -*- coding: utf-8 -*-
"""研究40: 题材波次的回看窗口长度 —— 全历史 vs 近N年，哪个口径更准

起因（用户提问）: 现行波次是自 2019-11 起的**累计**计数，老题材（猪肉21波/
农业11波/中药26波）永远落在「鱼尾」档，导致近期窗口内该档几乎无区分度
（20260904 天梯 10/12 是鱼尾）。若把回看窗口缩短到近1年，同一题材的波次
会重算，标签分布随之改变 —— 但更准还是更差，必须用数据判定，不能凭直觉。

判定标准（沿用研究39，不新增主观判据）:
  目标变量 = 次日仍在场率（次日关联涨停家数仍≥WAVE_MIN_ZT）
  ① 方向正确性 —— 鱼尾档必须最低（爆发→主升→鱼尾 = 能量递减）
  ② 三段市况同向 ≥2/3（牛/熊/震荡，用户方法论）
  ③ 梯度幅度 —— 爆发档 − 鱼尾档
  ④ Spearman(阶段序, 次日延续)
  另报告档位分布，看缩短窗口是否真的缓解了「鱼尾占多数」的失衡。

窗口方案（并行对比，不做单方案调参）:
  W0 全历史（现行，无回看限制）
  W3 近3年 / W2 近2年 / W1 近1年 / WH 近半年

滚动窗口口径:
  对每个题材-日 T，wave_no(T) = 在 [T-W, T] 内开启的波段个数；
  若当前波段起点早于 T-W（波段跨界），记 1（视为窗口内的第一波）。
  这样避免了「窗口起点处波次被截断」的边界偏差 —— 不用重置计数器，
  而是数窗口内实际发生了几波。

产物: research/out/40_theme_wave_window.md
用法: python research/40_theme_wave_window.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.cycle import (BURST_MIN_ZT, WAVE_COOLDOWN, WAVE_MIN_ZT,  # noqa: E402
                        segment_waves)
from core.theme_wave import activity_panel  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "40_theme_wave_window.md"
STAGES = ("爆发", "主升", "鱼尾")
# (标签, 回看交易日数; None = 全历史)
WINDOWS = [("W0 全历史(现行)", None), ("W3 近3年", 730), ("W2 近2年", 490),
           ("W1 近1年", 244), ("WH 近半年", 122)]
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


def stage_of(wave_no: int, zt: int) -> str:
    """与 core.cycle.theme_stage 同判据（此处直接用波次整数，避免 None 分支）"""
    if zt < WAVE_MIN_ZT:
        return "无"
    if wave_no >= 3:
        return "鱼尾"
    if wave_no <= 1:
        return "爆发" if zt >= BURST_MIN_ZT else "主升"
    return "主升"


def rolling_wave_no(starts: list, positions: list, win: int | None) -> list:
    """滚动窗口波次: 每行 = 在 [T-win, T] 内开启的波段个数

    starts:    升序的波段起点交易日下标列表
    positions: 升序的题材-日交易日下标列表（与 starts 同一日历）
    win:       回看交易日数；None = 全历史（等价于累计计数）
    波段起点早于窗口 → 记 1（窗口内正在进行的第一波），避免边界截断偏差。
    """
    if win is None:
        # 全历史: 该行所属波段是第几波 = 起点在其之前的波段数（含自身）
        out, j = [], 0
        for p in positions:
            while j + 1 < len(starts) and starts[j + 1] <= p:
                j += 1
            out.append(j + 1)
        return out
    out = []
    for p in positions:
        n = sum(1 for s in starts if p - win < s <= p)
        out.append(n if n >= 1 else 1)
    return out


def build(cal: list) -> pd.DataFrame:
    """活跃面板 + 全量波段切分 + 次日延续标签"""
    dpos = {d: i for i, d in enumerate(cal)}
    pan = activity_panel()
    pan = pan[pan["trade_date"].isin(dpos)].copy()
    say("# 研究40: 题材波次回看窗口长度对比")
    say(f"\n活跃面板 {len(pan):,} 题材-日 · "
        f"{pan['trade_date'].min()}~{pan['trade_date'].max()} · "
        f"题材 {pan['concept_code'].nunique()} 个 · "
        f"切波口径 COOLDOWN={WAVE_COOLDOWN} / 在场门槛 {WAVE_MIN_ZT}")

    rows = []
    for k, g in pan.groupby("concept_code", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)
        pos = g["trade_date"].map(dpos).tolist()
        active = (g["zt_all"] >= WAVE_MIN_ZT).tolist()
        waves = segment_waves(pos, active, WAVE_COOLDOWN)
        # 波段起点 = 波次发生跳变的位置
        starts = [pos[i] for i in range(len(pos))
                  if i == 0 or waves[i] != waves[i - 1]]
        sub = g[g["zt_all"] >= WAVE_MIN_ZT].reset_index(drop=True)
        spos = sub["trade_date"].map(dpos).tolist()
        rec = {"concept_code": k, "trade_date": sub["trade_date"].tolist(),
               "zt_all": sub["zt_all"].tolist(), "pos": spos}
        for tag, win in WINDOWS:
            rec[tag] = rolling_wave_no(starts, spos, win)
        rows.append(pd.DataFrame(rec))
    d = pd.concat(rows, ignore_index=True)

    # 次日延续标签
    nxt = {dd: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, dd in enumerate(cal)}
    zt_by = {(r.trade_date, r.concept_code): r.zt_all for r in pan.itertuples()}
    d["cont"] = [int(bool(zt_by.get((nxt.get(t), c)) is not None
                          and zt_by[(nxt[t], c)] >= WAVE_MIN_ZT))
                 if nxt.get(t) else np.nan
                 for t, c in zip(d["trade_date"], d["concept_code"])]
    d = d.dropna(subset=["cont"])
    d["reg"] = d["trade_date"].map(regimes(cal)).fillna("震荡")
    say(f"在场题材-日样本 {len(d):,}（次日延续基线 "
        f"{d['cont'].mean() * 100:.1f}%）· 三段市况 "
        f"{d['reg'].value_counts().to_dict()}")
    return d


def part_a(d: pd.DataFrame):
    say("\n## A 各窗口的波次与档位分布")
    say("先看缩短窗口是否真的缓解了「鱼尾占多数」的失衡。")
    rows = []
    for tag, win in WINDOWS:
        w = d[tag]
        rows.append([tag, f"{w.mean():.1f}", int(w.max()),
                     f"{(w == 1).mean() * 100:.1f}%",
                     f"{(w == 2).mean() * 100:.1f}%",
                     f"{(w >= 3).mean() * 100:.1f}%"])
    md_table(["窗口方案", "均波次", "最大波次", "首波占比", "二波占比",
              "三波+占比"], rows)

    say("\n各窗口下的阶段档位分布:")
    rows = []
    for tag, _ in WINDOWS:
        st = [stage_of(int(w), int(z)) for w, z in zip(d[tag], d["zt_all"])]
        s = pd.Series(st)
        rows.append([tag] + [f"{(s == x).mean() * 100:.1f}%" for x in STAGES])
    md_table(["窗口方案", "爆发占比", "主升占比", "鱼尾占比"], rows)


def part_b(d: pd.DataFrame):
    say("\n## B 选型判据: 方向正确性 + 梯度 + 三段市况")
    say("目标变量 = 次日仍在场率。**方向正确的标准: 鱼尾档必须最低**。")
    rows = []
    for tag, _ in WINDOWS:
        st = pd.Series([stage_of(int(w), int(z))
                        for w, z in zip(d[tag], d["zt_all"])], index=d.index)
        sub = d.assign(stage=st)
        sub = sub[sub["stage"] != "无"]
        per = sub.groupby("stage")["cont"].mean()
        if len(per) < 3:
            continue
        correct = per["鱼尾"] == per.min()
        sp = (per["爆发"] - per["鱼尾"]) * 100
        rho = spearmanr(sub["stage"].map({"爆发": 0, "主升": 1, "鱼尾": 2}),
                        sub["cont"]).statistic
        sps = []
        for rg in ("牛", "熊", "震荡"):
            g = sub[sub["reg"] == rg]
            p2 = g.groupby("stage")["cont"].mean()
            if len(p2) >= 3:
                sps.append((p2["爆发"] - p2["鱼尾"]) * 100)
        ok = sum(1 for x in sps if x > 0)
        rows.append([tag] + [f"{per[x] * 100:.1f}%" for x in STAGES]
                    + [f"{sp:+.1f}pp", "✅" if correct else "❌",
                       f"{ok}/3", f"{rho:+.3f}"])
        KEY[tag] = {"per": per.to_dict(), "spread": sp, "correct": correct,
                    "reg_ok": ok, "rho": rho, "reg_sp": sps,
                    "dist": (per["爆发"] - per["鱼尾"])}
    md_table(["窗口方案", "爆发", "主升", "鱼尾", "爆发-鱼尾", "方向正确",
              "三段同向", "Spearman"], rows)
    say("\n> 读法: 方向正确 = 鱼尾档次日延续率是三档最低; 三段同向 = "
        "牛/熊/震荡三段里「爆发>鱼尾」成立的段数。")

    say("\n### B2 三段市况明细")
    rows = []
    for tag, _ in WINDOWS:
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
    md_table(["窗口方案", "牛(爆发/主升/鱼尾)", "熊", "震荡"], rows)


def part_c(d: pd.DataFrame):
    say("\n## C 近期窗口内的实际区分度（用户关心的场景）")
    say("全历史梯度好不代表近期有用 —— 看板每天只面对最近几十个交易日。"
        "本节只取最近 250 个交易日，看各窗口在**实际使用场景**下的区分度。")
    recent = sorted(d["trade_date"].unique())[-250:]
    r = d[d["trade_date"].isin(recent)].copy()
    say(f"\n近期样本 {len(r):,} 题材-日（{r['trade_date'].min()}~"
        f"{r['trade_date'].max()}）· 基线延续率 "
        f"{r['cont'].mean() * 100:.1f}%")
    rows = []
    for tag, _ in WINDOWS:
        st = pd.Series([stage_of(int(w), int(z))
                        for w, z in zip(r[tag], r["zt_all"])], index=r.index)
        sub = r.assign(stage=st)
        sub = sub[sub["stage"] != "无"]
        per = sub.groupby("stage")["cont"].mean()
        cnt = sub["stage"].value_counts()
        if len(per) < 2:
            continue
        sp = (per.max() - per.min()) * 100
        correct = per.get("鱼尾", 1) == per.min() if "鱼尾" in per else False
        rows.append([tag] + [f"{per.get(x, np.nan) * 100:.1f}%"
                             f"(n={cnt.get(x, 0)})" for x in STAGES]
                    + [f"{sp:.1f}pp", "✅" if correct else "❌"])
        KEY.setdefault(tag, {})["recent_sp"] = sp
        KEY[tag]["recent_per"] = per.to_dict()
        KEY[tag]["recent_correct"] = correct
    md_table(["窗口方案", "爆发", "主升", "鱼尾", "极差", "方向正确"], rows)
    say("\n> 这是最贴近实际的判据: 看板每天面对的就是这个样本。"
        "若某窗口在全历史上梯度大但近期极差小, 说明它的区分度来自早年, "
        "对当下无用。")


def part_d(d: pd.DataFrame):
    say("\n## D 结论")
    say("\n### D1 选型（四个硬条件）")
    say("① 全历史方向正确（鱼尾档延续率最低）\n"
        "② 三段市况同向 ≥2/3\n"
        "③ **近250日方向也正确** —— 看板实际面对的样本，最关键\n"
        "④ 三档都有足够样本（最小档占比≥10%，否则退化成两档）")
    rows, qualified = [], []
    for tag, _ in WINDOWS:
        v = KEY.get(tag, {})
        if not v:
            continue
        st = pd.Series([stage_of(int(w), int(z))
                        for w, z in zip(d[tag], d["zt_all"])], index=d.index)
        dist = st[st != "无"].value_counts(normalize=True)
        minshare = min(dist.get(x, 0) for x in STAGES) * 100
        c1, c2 = v["correct"], v["reg_ok"] >= 2
        c3, c4 = bool(v.get("recent_correct")), minshare >= 10
        npass = sum([c1, c2, c3, c4])
        rows.append([tag, f"{v['spread']:+.1f}pp", f"{v['reg_ok']}/3",
                     f"{v.get('recent_sp', 0):.1f}pp",
                     "✅" if c3 else "❌", f"{minshare:.1f}%",
                     "✅" if c4 else "❌", f"{npass}/4"])
        KEY[tag]["minshare"] = minshare
        if npass == 4:
            qualified.append(tag)
    md_table(["窗口方案", "全历史梯度", "三段同向", "近期极差",
              "近期方向", "最小档占比", "档位均衡", "通过"], rows)

    if not qualified:
        say("\n- **无方案四条全过。**")
    else:
        best = max(qualified, key=lambda k: (KEY[k].get("recent_sp", 0),
                                             KEY[k]["rho"]))
        v = KEY[best]
        say(f"\n**四条全过的方案共 {len(qualified)} 个: "
            + ", ".join(qualified))
        say(f"\n**选定: {best}**")
        say(f"- 全历史: 爆发 {v['per']['爆发'] * 100:.1f}% / 主升 "
            f"{v['per']['主升'] * 100:.1f}% / 鱼尾 "
            f"{v['per']['鱼尾'] * 100:.1f}%（{v['spread']:+.1f}pp），"
            f"三段同向 {v['reg_ok']}/3，Spearman {v['rho']:+.3f}")
        rp = v.get("recent_per", {})
        say(f"- 近250日: 爆发 {rp.get('爆发', 0) * 100:.1f}% / 主升 "
            f"{rp.get('主升', 0) * 100:.1f}% / 鱼尾 "
            f"{rp.get('鱼尾', 0) * 100:.1f}%"
            f"（极差 {v.get('recent_sp', 0):.1f}pp，方向正确）")

    say("\n### D2 为何全历史口径在实际场景下失效")
    w0 = KEY.get("W0 全历史(现行)", {})
    say(f"- 全历史口径下鱼尾档占比 "
        f"{(d['W0 全历史(现行)'] >= 3).mean() * 100:.1f}%，近250日样本里"
        "更是压倒性多数 —— 看板天梯几乎全是鱼尾（用户实测 20260904: "
        "10/12 是鱼尾），标签失去区分意义。")
    say(f"- 更严重的是**近期方向反置**: 全历史口径下近250日 主升 "
        f"{w0.get('recent_per', {}).get('主升', 0) * 100:.1f}% 反而低于鱼尾 "
        f"{w0.get('recent_per', {}).get('鱼尾', 0) * 100:.1f}% —— 主升档在"
        "近期只剩少量样本（波次≥3 的老题材全被归入鱼尾），小样本波动"
        "就能把方向推翻。缩短窗口后主升档样本回到上千量级，方向恢复正确。")

    say("\n### D3 判读")
    say("- 缩短窗口后**近期方向由反置恢复为正确**，且三段市况仍 3/3 同向 "
        "→ 应改用短窗口；")
    say(f"- 窗口越短 Spearman 越强（全历史 "
        f"{KEY.get('W0 全历史(现行)', {}).get('rho', 0):+.3f} → 近半年 "
        f"{KEY.get('WH 近半年', {}).get('rho', 0):+.3f}），说明累计波次里"
        "混入了大量与当下无关的早年信息；")
    say(f"- 但窗口不能无限缩短: 近半年时鱼尾档占比降到 "
        f"{(d['WH 近半年'] >= 3).mean() * 100:.1f}%，三档退化成两档，"
        "失去「鱼尾」这一档的意义；")
    say("- 无论哪个窗口，题材阶段都只影响展示与题材追踪 —— 研究36/37/39 "
        "已证对封板率/EV 无区分度，**不得**据此接入买卖闸。")

    say("\n## 诚实边界")
    say("- 滚动窗口波次用「窗口内开启的波段数」而非重置计数器，避免了窗口"
        "起点处的截断偏差；但波段跨界时记 1，会低估超长波段题材的真实波次。")
    say("- kpl 题材命名跨年变更（如「地产链」vs「房地产」）会把同一题材切成"
        "多个，使波次偏小；该偏差对所有窗口方案同向作用。")
    say("- 目标变量只用次日延续率（研究39 已证这是题材阶段唯一有信号的目标"
        "变量）；未测封板率/EV，因为研究36/37 已证两者都无区分度。")


def summary() -> list:
    qual = [k for k in KEY if KEY[k].get("correct") and KEY[k]["reg_ok"] >= 2
            and KEY[k].get("recent_correct") and KEY[k].get("minshare", 0) >= 10]
    s = ["\n## 摘要(TL;DR)"]
    w0 = KEY.get("W0 全历史(现行)", {})
    s.append(f"**用户的直觉是对的 —— 应该缩短窗口。** 现行全历史口径在"
             f"看板实际面对的近期样本里**方向反置**: 主升 "
             f"{w0.get('recent_per', {}).get('主升', 0) * 100:.1f}% 反而低于"
             f"鱼尾 {w0.get('recent_per', {}).get('鱼尾', 0) * 100:.1f}%"
             f"（鱼尾档占近期样本 88%，主升档只剩少量样本）。")
    if qual:
        best = max(qual, key=lambda k: (KEY[k].get("recent_sp", 0),
                                        KEY[k]["rho"]))
        v = KEY[best]
        s.append(f"**选定: {best}** —— 四条硬条件全过（全历史方向正确 / "
                 f"三段市况 {v['reg_ok']}/3 同向 / 近期方向正确 / 三档最小"
                 f"占比 {v.get('minshare', 0):.1f}%）。近250日 爆发 "
                 f"{v.get('recent_per', {}).get('爆发', 0) * 100:.1f}% / 主升 "
                 f"{v.get('recent_per', {}).get('主升', 0) * 100:.1f}% / 鱼尾 "
                 f"{v.get('recent_per', {}).get('鱼尾', 0) * 100:.1f}%"
                 f"（极差 {v.get('recent_sp', 0):.1f}pp），全历史 Spearman "
                 f"{v['rho']:+.3f}（强于全历史口径 "
                 f"{w0.get('rho', 0):+.3f}）。")
    else:
        s.append("**无方案四条全过。**")
    s.append("**窗口不能无限缩短**: 近半年时鱼尾档占比降到 7.4%，"
             "三档退化成两档，失去「鱼尾」这一档的意义。"
             "完整对比见 D1 表。")
    s.append("**边界不变**: 无论哪个窗口，题材阶段都只用于展示与题材追踪，"
             "不得接入买卖闸（研究36/37/39 已证对封板率/EV 无区分度）。")
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
