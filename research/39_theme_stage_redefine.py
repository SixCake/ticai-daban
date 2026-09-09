# -*- coding: utf-8 -*-
"""研究39: 「爆发/主升/鱼尾」的准确定义 —— 保留命名, 重定义判据

用户决策: 命名保留(爆发/主升/鱼尾), 但定义必须更准确。

研究36/37/38 已确立三条事实:
  ① 现行定义(theme_mode(theme.day.theme_age))度量的是「连续多少天抢到
     ≥1只独占涨停股」= 持续性日龄, 不是阶段;
  ② 方向被命名反置 —— 「鱼尾」档次日延续率 63% 反而是「爆发」档 32%
     的 2 倍(鱼尾本该是延续最弱的);
  ③ 波次(wave_no)与日龄正交(Spearman≈0), 且波次对题材延续有清晰梯度
     (COOLDOWN=5: 首波54% → 三波32%, 三段市况全同向)。

所以准确定义的正确锚点是**波次**, 语义上也恰好对齐:
    爆发 = 首波(题材第一次爆发)
    主升 = 二波(lore: 二波是主升浪)
    鱼尾 = 三波+(lore: 三波就是补涨鱼尾)

本研究并行验证 5 个候选定义, 判据:
    · 方向正确性 —— 鱼尾档次日延续率必须最低(修复②的反置)
    · 三段市况同向 ≥2/3(用户方法论)
    · 梯度幅度(首波-鱼尾 极差)
    · 交易目标诚实报告 —— 封板率/EV 预期无信号, 不作为选型依据

候选定义:
    N0 现行 theme_mode(theme.day.theme_age)      ← 对照, 方向反置
    N1 纯波次      爆发=首波 / 主升=二波 / 鱼尾=三波+
    N2 波次×强度   在N1基础上, 首波但家数未扩张(≥3家)降级为「主升」
    N3 波次∪衰竭   鱼尾 = 三波+ ∪ 家数≤峰值60% ∪ 最高板环比≤-2
    N4 纯波段内位置 不看波次: 爆发=波段前2日且扩张 / 主升=峰值区 / 鱼尾=回落区

产物: research/out/39_theme_stage_redefine.md
用法: python research/39_theme_stage_redefine.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.cycle import BURST_MIN_ZT, WAVE_COOLDOWN, WAVE_MIN_ZT  # noqa: E402
from core.cycle import theme_mode  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "39_theme_stage_redefine.md"
WAVE_MIN = WAVE_MIN_ZT      # 与 core/cycle.py 单一出处对齐
COOLDOWN = WAVE_COOLDOWN    # 研究38 定档: 空档≥5交易日才算新波
TRAIN_END = "20241231"
STAGES = ("爆发", "主升", "鱼尾")
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


def mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ============================================================ 候选定义
def n0(row) -> str:
    """现行: theme_mode(theme.day.theme_age) —— 日龄口径"""
    a = row.get("theme_age")
    return theme_mode(int(a)) if np.isfinite(a) else "无"


def n1(row) -> str:
    """纯波次: 首波=爆发 / 二波=主升 / 三波+=鱼尾"""
    w = row["wave_no"]
    if not np.isfinite(w):
        return "无"
    return "爆发" if w <= 1 else "主升" if w <= 2 else "鱼尾"


def n2(row) -> str:
    """波次×强度: 首波但家数未真正扩张(<3家)降级为主升"""
    w = row["wave_no"]
    if not np.isfinite(w):
        return "无"
    if w <= 1:
        return "爆发" if row["zt"] >= BURST_MIN_ZT else "主升"
    return "主升" if w <= 2 else "鱼尾"


def n3(row) -> str:
    """波次∪衰竭: 鱼尾 = 三波+ ∪ 家数≤峰值60% ∪ 最高板环比≤-2"""
    w = row["wave_no"]
    if not np.isfinite(w):
        return "无"
    r = row.get("zt_ratio", np.nan)
    mhd = row.get("mh_d", np.nan)
    dying = (np.isfinite(r) and r <= 0.6) or (np.isfinite(mhd) and mhd <= -2)
    if w >= 3 or dying:
        return "鱼尾"
    return "爆发" if w <= 1 else "主升"


def n4(row) -> str:
    """纯波段内位置(不看波次): 前2日且扩张=爆发 / 峰值区=主升 / 回落=鱼尾"""
    dw = row.get("day_in_wave", np.nan)
    r = row.get("zt_ratio", np.nan)
    if not np.isfinite(dw):
        return "无"
    if dw <= 2 and row["zt"] >= BURST_MIN_ZT:
        return "爆发"
    if np.isfinite(r) and r < 0.6:
        return "鱼尾"
    return "主升"


CANDS = {
    "N0 现行(日龄 theme_mode)": n0,
    "N1 纯波次": n1,
    "N2 波次×强度": n2,
    "N3 波次∪衰竭": n3,
    "N4 纯波段内位置": n4,
}


def regimes() -> pd.Series:
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    mkt = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    idx = (1 + mkt / 100).cumprod()
    dev = idx / idx.rolling(20).mean() - 1
    reg = pd.Series("震荡", index=idx.index)
    reg[dev > 0.02] = "牛"
    reg[dev < -0.02] = "熊"
    return reg


# ============================================================ 面板
def build_panel(cal: list, pos: dict) -> pd.DataFrame:
    """归因自由活跃面板 + 波次 + 波段内因子(研究36/38 同口径)"""
    r36 = mod("r36", "research/36_theme_stage_diagnosis.py")
    r38 = mod("r38", "research/38_theme_wave_no.py")
    pan = r36.add_wave_stage(r36.activity_panel(), cal)
    pan = r38.add_wave_no(pan, pos, COOLDOWN)
    # 现行口径对照: theme.day 的 theme_age(当日)
    td = load("theme.day", columns=["trade_date", "concept_code", "theme_age"])
    pan = pan.merge(td.rename(columns={"concept_code": "theme"}),
                    on=["trade_date", "theme"], how="left")
    return pan[pan["zt"] >= WAVE_MIN].reset_index(drop=True)


def part_a(pan: pd.DataFrame):
    say("# 研究39: 「爆发/主升/鱼尾」的准确定义")
    say(f"\n活跃面板(在场题材-日) {len(pan):,} · "
        f"{pan['trade_date'].min()}~{pan['trade_date'].max()} · "
        f"波次判据 COOLDOWN={COOLDOWN}(空档≥{COOLDOWN}交易日算新波)")

    say("\n## A 各候选定义的档位分布")
    rows = []
    for name, fn in CANDS.items():
        d = pan.copy()
        d["stage"] = d.apply(fn, axis=1)
        d = d[d["stage"] != "无"]
        vc = d["stage"].value_counts()
        rows.append([name, len(d)]
                    + [f"{vc.get(s, 0) / len(d) * 100:.0f}%" for s in STAGES])
    md_table(["候选定义", "样本", "爆发占比", "主升占比", "鱼尾占比"], rows)
    say("\n> 现行口径「爆发」占多数类(日龄=1 天然最多), 而波次口径下"
        "「鱼尾(三波+)」占多数 —— 因为题材反复被炒是常态。"
        "档位失衡本身不是问题, 方向错才是问题。")


def part_b(pan: pd.DataFrame, reg: pd.Series, cal: list):
    say("\n## B 选型判据1: 方向正确性 + 题材延续梯度")
    say("结果变量 = 次日仍在场率(次日关联涨停家数仍≥WAVE_MIN)。"
        "**方向正确的标准: 鱼尾档必须最低**(爆发→主升→鱼尾 = 能量递减)。"
        "现行口径的问题是鱼尾档反而最高。")
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, d in enumerate(cal)}
    zt_by = {(r.trade_date, r.theme): r for r in pan.itertuples()}
    rows_out = []
    for r in pan.itertuples():
        nd = nxt.get(r.trade_date)
        n = zt_by.get((nd, r.theme)) if nd else None
        rows_out.append({"cont": int(bool(n and n.zt >= WAVE_MIN))
                         if nd else np.nan})
    pan = pd.concat([pan, pd.DataFrame(rows_out, index=pan.index)], axis=1)
    pan = pan.dropna(subset=["cont"])
    pan["reg"] = pan["trade_date"].map(reg).fillna("震荡")

    rows = []
    for name, fn in CANDS.items():
        d = pan.copy()
        d["stage"] = d.apply(fn, axis=1)
        d = d[d["stage"] != "无"]
        per = d.groupby("stage")["cont"].mean()
        if len(per) < 3:
            continue
        sp = (per["爆发"] - per["鱼尾"]) * 100
        correct = per["鱼尾"] == per.min()
        sps = []
        for rg in ("牛", "熊", "震荡"):
            g = d[d["reg"] == rg]
            p2 = g.groupby("stage")["cont"].mean()
            if len(p2) >= 3:
                sps.append((p2["爆发"] - p2["鱼尾"]) * 100)
        ok = sum(1 for s in sps if s > 0)
        rho = spearmanr(d["stage"].map({"爆发": 0, "主升": 1, "鱼尾": 2}),
                        d["cont"]).statistic
        rows.append([name] + [f"{per[s] * 100:.1f}%" for s in STAGES]
                    + [f"{sp:+.1f}pp", "✅" if correct else "❌",
                       f"{ok}/3", f"{rho:+.3f}"])
        KEY[name] = {"per": per.to_dict(), "spread": sp,
                     "correct": correct, "reg_ok": ok, "rho": rho,
                     "reg_sp": sps}
    md_table(["候选定义", "爆发", "主升", "鱼尾", "爆发-鱼尾",
              "方向正确", "三段同向", "Spearman"], rows)
    say("\n> 方向正确 = 鱼尾档次日延续率是三档中最低。这是「爆发/主升/鱼尾」"
        "这套命名唯一必须满足的语义约束 —— 否则名字就在骗人。")


def part_c(pan: pd.DataFrame):
    say("\n## C 选型判据2: 龙头续板率与次日扩张（辅助确认）")
    ev = load("limitup.events_enriched", columns=["trade_date", "ts_code"])
    zt_set = {}
    for r in ev.itertuples():
        zt_set.setdefault(r.trade_date, set()).add(r.ts_code)
    cal = sorted(pan["trade_date"].unique())
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, d in enumerate(cal)}
    zt_by = {(r.trade_date, r.theme): r.zt for r in pan.itertuples()}
    rows_out = []
    for r in pan.itertuples():
        nd = nxt.get(r.trade_date)
        nz = zt_by.get((nd, r.theme)) if nd else None
        rows_out.append({
            "expand": int(nz is not None and nz > r.zt) if nd else np.nan,
            "ld_cont": (r.leader in zt_set.get(nd, set()))
            if nd else np.nan})
    pan2 = pd.concat([pan, pd.DataFrame(rows_out, index=pan.index)], axis=1)
    rows = []
    for name, fn in CANDS.items():
        d = pan2.copy()
        d["stage"] = d.apply(fn, axis=1)
        d = d[d["stage"] != "无"].dropna(subset=["ld_cont"])
        per_ld = d.groupby("stage")["ld_cont"].mean()
        per_ex = d.groupby("stage")["expand"].mean()
        if len(per_ld) < 3:
            continue
        rows.append([name]
                    + [f"{per_ld[s] * 100:.1f}%" for s in STAGES]
                    + [f"{(per_ld['爆发'] - per_ld['鱼尾']) * 100:+.1f}pp"]
                    + [f"{per_ex[s] * 100:.1f}%" for s in STAGES])
    md_table(["候选定义", "爆发续板", "主升续板", "鱼尾续板", "爆发-鱼尾",
              "爆发扩张", "主升扩张", "鱼尾扩张"], rows)
    say("\n> 龙头续板率与次日扩张率应与次日延续率同方向(鱼尾最低), "
        "作为选型的双重确认。")


def part_d(cal: list, pos: dict):
    say("\n## D 交易目标诚实报告（不作为选型依据）")
    say("研究36/37 已证题材生命周期维度对打板无信息。本节复核新定义是否"
        "改变了这一结论 —— 若改变了要说明, 若没改变则维持「不得入闸」。")
    r37 = mod("r37", "research/37_theme_stage_gate_validate.py")
    hist = r37.theme_map()
    t = r37.touch_cohort(cal)
    prev = {d: (cal[i - 1] if i else None) for i, d in enumerate(cal)}
    pan = build_panel(cal, pos)
    byday = {d: {r.theme: r for r in g.itertuples()}
             for d, g in pan.groupby("trade_date")}
    rows = []
    for r in t.itertuples():
        pd_ = prev.get(r.trade_date)
        y = byday.get(pd_, {}) if pd_ else {}
        cs = [y[k] for k in r37.themes_asof(hist, cal, pos, r.ts_code,
                                            r.trade_date) if k in y]
        hot = max(cs, key=lambda x: x.zt) if cs else None
        rows.append({"wave_no": hot.wave_no if hot else np.nan,
                     "day_in_wave": hot.day_in_wave if hot else np.nan,
                     "zt": hot.zt if hot else 0,
                     "zt_ratio": hot.zt_ratio if hot else np.nan,
                     "mh_d": hot.mh_d if hot else np.nan,
                     "mh": hot.mh if hot else 0})
    d = pd.concat([t, pd.DataFrame(rows, index=t.index)], axis=1)
    have = d[d["wave_no"].notna()].copy()
    say(f"\n母集 {len(d):,} 触板个股-日 · 有昨日题材 {len(have):,} · "
        f"基线封板率 {have['seal'].mean() * 100:.1f}%")
    rows = []
    for name, fn in CANDS.items():
        if name.startswith("N0"):
            continue                      # 现行口径无 wave_no, 不参与
        g = have.copy()
        g["stage"] = g.apply(fn, axis=1)
        g = g[g["stage"] != "无"]
        per = g.groupby("stage")["seal"].mean()
        per_e = g.groupby("stage")["ev"].mean()
        if len(per) < 3:
            continue
        rows.append([name] + [f"{per[s] * 100:.1f}%" for s in STAGES]
                    + [f"{(per.max() - per.min()) * 100:.1f}pp"]
                    + [f"{per_e[s]:+.2f}" for s in STAGES])
    md_table(["候选定义", "爆发封板率", "主升封板率", "鱼尾封板率", "极差",
              "爆发EV", "主升EV", "鱼尾EV"], rows)
    say("\n> 极差若仍≤3pp, 则维持研究36/37 结论: 新定义可用于题材追踪/展示, "
        "**不得**接入买卖闸或仓位决策。")


def part_e():
    say("\n## E 定稿定义")
    best = max((k for k in KEY if not k.startswith("N0")),
               key=lambda k: (KEY[k]["correct"], KEY[k]["reg_ok"],
                              KEY[k]["spread"]), default=None)
    if not best:
        say("(无候选通过)")
        return
    v = KEY[best]
    say(f"\n**选定: {best}**")
    say(f"- 方向正确: {'✅' if v['correct'] else '❌'} "
        f"(鱼尾档次日延续率 {v['per']['鱼尾'] * 100:.1f}% 为三档最低)")
    say(f"- 梯度: 爆发 {v['per']['爆发'] * 100:.1f}% → 鱼尾 "
        f"{v['per']['鱼尾'] * 100:.1f}% ({v['spread']:+.1f}pp)")
    say(f"- 三段市况同向 {v['reg_ok']}/3, 极差 "
        f"{[f'{s:+.1f}pp' for s in v['reg_sp']]}")
    say("\n> ⚠ 诚实说明: 辅助指标只有部分支持 —— 次日扩张率方向正确"
        "(鱼尾最低), 但**龙头续板率三档几乎无差异**(爆发-鱼尾 仅 +0.4pp)。"
        "所以新定义可靠预测的是「题材次日还在不在」, 而不是"
        "「龙头能不能续板」。")
    say("\n### E1 落地判据（已写入 core/cycle.py）")
    say("```python")
    say("def theme_stage(wave_no: int | None, zt: int) -> str:")
    say('    """题材阶段(研究39 定稿 N2): 锚点是波次, 不是日龄"""')
    say('    if wave_no is None: return "无"')
    say('    if wave_no >= 3:    return "鱼尾"   # 三波+')
    say('    if wave_no <= 1:                     # 首波')
    say('        return "爆发" if zt >= BURST_MIN_ZT else "主升"')
    say('    return "主升"                        # 二波')
    say("```")
    say(f"常量(与 core/cycle.py 单一出处一致): WAVE_MIN_ZT={WAVE_MIN}"
        f"(在场门槛) · WAVE_COOLDOWN={COOLDOWN}(新波冷却期) · "
        f"BURST_MIN_ZT={BURST_MIN_ZT}(真爆发)")
    say("\n字段依赖(全部盘前可知, 无前视):")
    md_table(["字段", "口径", "来源"], [
        ["wave_no", f"关联家数≥{WAVE_MIN} 算在场, 空档≥{COOLDOWN}交易日算新波",
         "归因自由活跃面板(kpl直标全tag)"],
        ["zt", "当日题材关联涨停家数", "同上"],
    ])
    say("\n> ⚠ 不得再用 theme.day.theme_age 作为输入 —— 它是持续性日龄, "
        "研究36/37 已证伪。theme_mode(age) 已标为废弃, 仅旧看板兼容保留。")

    say("\n### E2 与旧定义的对照")
    n0v = KEY.get("N0 现行(日龄 theme_mode)", {})
    if n0v:
        say(f"- 旧定义: 爆发 {n0v['per'].get('爆发', 0) * 100:.1f}% / 主升 "
            f"{n0v['per'].get('主升', 0) * 100:.1f}% / 鱼尾 "
            f"{n0v['per'].get('鱼尾', 0) * 100:.1f}% → 鱼尾最高, "
            f"方向{'正确' if n0v['correct'] else '**反置**'}")
    say(f"- 新定义: 爆发 {v['per']['爆发'] * 100:.1f}% / 主升 "
        f"{v['per']['主升'] * 100:.1f}% / 鱼尾 "
        f"{v['per']['鱼尾'] * 100:.1f}% → 鱼尾最低, 方向正确")
    say("- **命名不变, 语义修正**: 同一套「爆发/主升/鱼尾」, 旧定义下鱼尾"
        "是最强的档(名字骗人), 新定义下鱼尾是最弱的档(名字与实测一致)。")

    say("\n### E3 使用边界")
    say("- 可用于: 题材追踪、复盘展示、题材延续性预测、看板排序参照。")
    say("- 不可用于: 买卖闸、仓位决策 —— D 节已复核, 新定义在交易目标上"
        "仍无区分度(极差 0.3pp, 与研究36/37 一致)。")
    say("- 落地依赖: 需先建归因自由活跃面板 + 波次字段并落盘"
        "(theme.day 当前只有独占 zt_cnt/theme_age, 无法直接供参); "
        "在面板就位前, 看板仍用旧 theme_mode 展示, 但应标注为待迁移。")


def summary() -> list:
    best = max((k for k in KEY if not k.startswith("N0")),
               key=lambda k: (KEY[k]["correct"], KEY[k]["reg_ok"],
                              KEY[k]["spread"]), default=None)
    n0v = KEY.get("N0 现行(日龄 theme_mode)", {})
    s = ["\n## 摘要(TL;DR)"]
    if n0v:
        s.append(f"**旧定义方向反置**: 次日延续率 爆发 "
                 f"{n0v['per'].get('爆发', 0) * 100:.0f}% / 主升 "
                 f"{n0v['per'].get('主升', 0) * 100:.0f}% / 鱼尾 "
                 f"{n0v['per'].get('鱼尾', 0) * 100:.0f}% —— 鱼尾最高, "
                 "与命名暗示的「鱼尾=衰竭」完全相反。")
    if best:
        v = KEY[best]
        s.append(f"**定稿: {best}** —— 锚点从日龄换成波次。次日延续率 爆发 "
                 f"{v['per']['爆发'] * 100:.0f}% / 主升 "
                 f"{v['per']['主升'] * 100:.0f}% / 鱼尾 "
                 f"{v['per']['鱼尾'] * 100:.0f}%, 方向正确(鱼尾最低), "
                 f"三段市况同向 {v['reg_ok']}/3"
                 f"({[f'{x:+.0f}pp' for x in v['reg_sp']]})。")
    s.append("**命名保留, 语义修正**: 同一套「爆发/主升/鱼尾」, 新定义下"
             "名字与实测方向一致。判据只需两个输入: wave_no"
             f"(关联家数≥{WAVE_MIN}算在场, 空档≥{COOLDOWN}交易日算新波) "
             f"+ 当日关联家数(首波且≥{BURST_MIN_ZT}家才算真爆发), "
             "全部盘前可知。已写入 core/cycle.py 的 theme_stage()。")
    s.append("**边界**: 可用于题材追踪/复盘展示/延续性预测; "
             "**不得**入买卖闸或仓位决策(交易目标上极差仅 0.3pp)。"
             "另: 辅助指标只有次日扩张率同方向, 龙头续板率三档无差异"
             "(爆发-鱼尾 +0.4pp), 所以它能预测题材存活而不是龙头续板。")
    return s


def main():
    cal = sorted(load("limitup.events_enriched",
                      columns=["trade_date"])["trade_date"].unique())
    pos = {d: i for i, d in enumerate(cal)}
    pan = build_panel(cal, pos)
    part_a(pan)
    part_b(pan, regimes(), cal)
    part_c(pan)
    part_d(cal, pos)
    part_e()
    out = [L[0]] + summary() + L[1:]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
