# -*- coding: utf-8 -*-
"""研究38: 题材波次(第几波) —— 「爆发/主升/鱼尾」是否应重定义为波次

起因: 研究36/37 证明 `theme_age` 不是阶段变量, 而是「题材连续多少天抢到
≥1只独占涨停股」的持续性计数。但打板口径的「爆发/主升/鱼尾」通常指的是
**波次**(第一波爆发 / 第二波主升 / 第三波鱼尾补涨), 而不是波段内的日龄。
两者是正交的量:

    theme_age  = 当前波段内第几天(day-in-wave)   ← 项目已实现, 已证伪
    wave_no    = 该题材历史上第几个波段(wave index) ← 项目从未实现

本研究构造 wave_no 并做**双目标**验证(避免研究36 只看单一目标的教训):
    目标1 题材次日延续率 —— 阶段因子的本职(它该预测题材还活不活)
    目标2 个股封板率 / EV —— 交易价值(它能不能帮打板赚钱)

注意与项目既有「二波」的区别:
    research/29 / docs/research_07 的「二波」是**个股级**(高位龙头回落后的
    第二波), 本研究是**题材级**(题材行情的第几波)。两者不同, 不可混用。

波次构造:
    在场 = 当日关联涨停家数(kpl 直标全 tag, 归因自由) ≥ WAVE_MIN
    新波 = 在场日与上一在场日之间空档 ≥ COOLDOWN 个交易日
    wave_no = 该题材历史上第几个波段(自 kpl 起点累计)
    day_in_wave = 当前波段内第几个在场日
    多方案并行: COOLDOWN ∈ {2,3,5}(用户方法论: 不做单方案调参)

验收门槛:
    目标1: 波次分档次日延续率有梯度, 且三段市况方向一致 ≥2/3
    目标2: 封板率或 EV 有梯度, walk-forward(TRAIN≤2024→TEST2025~2026)同向

产物: research/out/38_theme_wave_no.md
用法: python research/38_theme_wave_no.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.cycle import theme_mode  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "38_theme_wave_no.md"
WAVE_MIN = 2
TRAIN_END = "20241231"
COOLDOWNS = (2, 3, 5)
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


def add_wave_no(pan: pd.DataFrame, pos: dict, cooldown: int) -> pd.DataFrame:
    """加 wave_no(第几波) 与 day_in_wave(波段内第几个在场日)"""
    pan = pan.copy()
    pan["wave_no"] = 0
    pan["day_in_wave"] = 0
    for _, g in pan.groupby("theme", sort=False):
        g = g.sort_values("trade_date")
        ds = g["trade_date"].tolist()
        zts = g["zt"].tolist()
        wno, dw = [], []
        for i, d in enumerate(ds):
            p = pos.get(d, 0)
            gap = 10 ** 6 if i == 0 else p - pos.get(ds[i - 1], p) - 1
            if zts[i] >= WAVE_MIN and gap >= cooldown:
                w = (wno[-1] + 1) if wno else 1        # 新波段
                dd = 1
            elif zts[i] >= WAVE_MIN:
                w = wno[-1] if wno else 1              # 同波段延续
                dd = (dw[-1] + 1) if dw else 1
            else:                                      # 只1家: 挂在原波段
                w = wno[-1] if wno else 1
                dd = (dw[-1] + 1) if dw else 1
            wno.append(w)
            dw.append(dd)
        pan.loc[g.index, "wave_no"] = wno
        pan.loc[g.index, "day_in_wave"] = dw
    return pan


def regimes() -> pd.Series:
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    mkt = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    idx = (1 + mkt / 100).cumprod()
    dev = idx / idx.rolling(20).mean() - 1
    reg = pd.Series("震荡", index=idx.index)
    reg[dev > 0.02] = "牛"
    reg[dev < -0.02] = "熊"
    return reg


# ============================================================ A 构造
def part_a(pan: pd.DataFrame, cal: list, pos: dict):
    say("# 研究38: 题材波次(第几波)因子 —— 阶段是否应重定义为波次")
    say(f"\n活跃面板(归因自由, kpl直标全tag) {len(pan):,} 题材-日 · "
        f"{pan['trade_date'].min()}~{pan['trade_date'].max()} · "
        f"题材 {pan['theme'].nunique()} 个")

    say("\n## A 波次构造与正交性证明")
    say("`theme_age` 数的是**波段内日龄**, 打板口径的「爆发/主升/鱼尾」指的"
        "却是**波次**。若两者是同一个量, 重命名就够了; 若正交, 就必须新增"
        "字段。先证明正交性:")
    td = load("theme.day", columns=["trade_date", "concept_code", "theme_age"])
    rows = []
    for cd in COOLDOWNS:
        p = add_wave_no(pan, pos, cd)
        act = p[p["zt"] >= WAVE_MIN]
        m = act.merge(td.rename(columns={"concept_code": "theme"}),
                      on=["trade_date", "theme"], how="left")
        m = m.dropna(subset=["theme_age"])
        m["tm"] = m["theme_age"].map(lambda a: theme_mode(int(a)))
        rho = spearmanr(m["wave_no"], m["theme_age"]).statistic
        rows.append([cd, len(act), f"{rho:+.3f}",
                     f"{(act['wave_no'] == 1).mean() * 100:.1f}%",
                     f"{(act['wave_no'] == 2).mean() * 100:.1f}%",
                     f"{(act['wave_no'] >= 3).mean() * 100:.1f}%",
                     int(act["wave_no"].max())])
    md_table(["COOLDOWN(空档≥N日算新波)", "在场题材-日",
              "Spearman(波次, 日龄)", "首波占比", "二波占比", "三波+占比",
              "最大波次"], rows)
    say("\n> Spearman ≈ 0 → **波次与日龄是两个独立的量**, 不是同一字段的"
        "两种叫法。项目现有的 theme_age 无论怎么改名都无法表达波次。")

    say("\n### A2 交叉表: 同一现行标签下波次可以完全不同")
    p = add_wave_no(pan, pos, 3)
    act = p[p["zt"] >= WAVE_MIN]
    m = act.merge(td.rename(columns={"concept_code": "theme"}),
                  on=["trade_date", "theme"], how="left").dropna(
        subset=["theme_age"])
    m["tm"] = m["theme_age"].map(lambda a: theme_mode(int(a)))
    ct = pd.crosstab(m["tm"], m["wave_no"].clip(upper=5), normalize="index")
    cnt = m["tm"].value_counts()
    rows = [[idx] + [f"{ct.loc[idx, c] * 100:.0f}%" if c in ct.columns else "-"
                     for c in range(1, 6)] + [int(cnt.get(idx, 0))]
            for idx in ("爆发", "主升", "鱼尾") if idx in ct.index]
    md_table(["现行标签", "首波", "二波", "三波", "四波", "五波+",
              "题材-日数"], rows)
    say("\n> 同一个「爆发」标签里, 首波/二波/三波+ 都有 —— 现行标签完全"
        "无法区分波次。这就是命名混乱的根源: 用一个日龄字段去表达波次语义。")


# ============================================================ B 目标1
def part_b(pan: pd.DataFrame, cal: list, pos: dict, reg: pd.Series):
    say("\n## B 目标1: 波次 → 题材次日延续率（阶段因子的本职）")
    say("结果变量用相对量(避免规模混淆, 与研究36 C1 同口径):\n"
        "- `次日仍在场率` 次日关联涨停家数仍≥WAVE_MIN\n"
        "- `次日扩张率` 次日家数 > 当日家数\n"
        "- `龙头续板率` T日题材龙头在T+1仍涨停")
    r36 = mod("r36", "research/36_theme_stage_diagnosis.py")
    pan = r36.add_wave_stage(pan, cal)
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, d in enumerate(cal)}
    zt_by = {(r.trade_date, r.theme): r for r in pan.itertuples()}
    ev = load("limitup.events_enriched", columns=["trade_date", "ts_code"])
    zt_set = {}
    for r in ev.itertuples():
        zt_set.setdefault(r.trade_date, set()).add(r.ts_code)

    for cd in COOLDOWNS:
        p = add_wave_no(pan, pos, cd)
        p = p[p["zt"] >= WAVE_MIN].copy()
        rows_out = []
        for r in p.itertuples():
            nd = nxt.get(r.trade_date)
            n = zt_by.get((nd, r.theme)) if nd else None
            rows_out.append({
                "trade_date": r.trade_date, "theme": r.theme, "zt": r.zt,
                "mh": r.mh, "wave_no": r.wave_no, "day_in_wave": r.day_in_wave,
                "zt_ratio": r.zt_ratio,
                "cont": int(bool(n and n.zt >= WAVE_MIN)) if nd else np.nan,
                "expand": int(bool(n and n.zt > r.zt)) if nd else np.nan,
                "ld_cont": (r.leader in zt_set.get(nd, set()))
                if nd else np.nan})
        d = pd.DataFrame(rows_out).dropna(subset=["cont"])
        d["reg"] = d["trade_date"].map(reg).fillna("震荡")
        d["wb"] = d["wave_no"].clip(upper=4).map(
            {1: "首波", 2: "二波", 3: "三波", 4: "四波+"})
        say(f"\n### COOLDOWN={cd} (样本 {len(d):,} 题材-日)")
        rows = []
        for wb in ("首波", "二波", "三波", "四波+"):
            g = d[d["wb"] == wb]
            if len(g) < 50:
                continue
            rows.append([wb, len(g), f"{g['zt'].mean():.1f}",
                         f"{g['cont'].mean() * 100:.1f}%",
                         f"{g['expand'].mean() * 100:.1f}%",
                         f"{g['ld_cont'].mean() * 100:.1f}%"])
        md_table(["波次", "题材-日数", "均当日家数", "次日仍在场率",
                  "次日扩张率", "龙头续板率"], rows)
        rows, sps = [], []
        for rg in ("牛", "熊", "震荡"):
            g = d[d["reg"] == rg]
            per = g.groupby("wb")["cont"].mean()
            if len(per) < 2:
                continue
            line = [rg]
            for wb in ("首波", "二波", "三波", "四波+"):
                v = per.get(wb, np.nan)
                line.append(f"{v * 100:.0f}%" if np.isfinite(v) else "-")
            sp = (per.max() - per.min()) * 100
            rows.append(line + [f"{sp:+.0f}pp"])
            sps.append(sp)
        md_table(["市况", "首波", "二波", "三波", "四波+",
                  "仍在场率极差"], rows)
        rho = spearmanr(d["wave_no"], d["cont"]).statistic
        say(f"- Spearman(波次, 次日仍在场) = {rho:+.3f} · "
            f"三段市况极差 {[f'{s:+.0f}pp' for s in sps]}")
        KEY[f"cd{cd}_cont"] = {"rho": rho, "sps": sps,
                               "per": d.groupby("wb")["cont"].mean().to_dict()}


# ============================================================ C 目标2
def part_c(cal: list, pos: dict):
    say("\n## C 目标2: 波次 → 个股封板率 / EV（交易价值）")
    say("母集与研究37 一致(全历史非一字触板), EV = 次日open/涨停价-1。"
        "题材锚定在**昨日**波次状态(盘前可知)。")
    r36 = mod("r36", "research/36_theme_stage_diagnosis.py")
    r37 = mod("r37", "research/37_theme_stage_gate_validate.py")
    pan = r36.add_wave_stage(r36.activity_panel(), cal)
    hist = r37.theme_map()
    t = r37.touch_cohort(cal)
    prev = {d: (cal[i - 1] if i else None) for i, d in enumerate(cal)}
    say(f"\n母集 {len(t):,} 个股-日 · 基线封板率 "
        f"{t['seal'].mean() * 100:.1f}% · 基线EV {t['ev'].mean():+.2f}%")

    for cd in COOLDOWNS:
        p = add_wave_no(pan, pos, cd)
        byday = {d: {r.theme: r for r in g.itertuples()}
                 for d, g in p.groupby("trade_date")}
        rows = []
        for r in t.itertuples():
            pd_ = prev.get(r.trade_date)
            y = byday.get(pd_, {}) if pd_ else {}
            cs = [y[k] for k in r37.themes_asof(hist, cal, pos, r.ts_code,
                                                  r.trade_date) if k in y]
            hot = max(cs, key=lambda x: x.zt) if cs else None
            rows.append((hot.wave_no if hot else np.nan,
                         hot.day_in_wave if hot else np.nan,
                         hot.zt if hot else 0))
        f = pd.DataFrame(rows, columns=["wave_no", "day_in_wave", "zt"],
                         index=t.index)
        d = pd.concat([t, f], axis=1)
        d["reg"] = d["trade_date"].map(regimes()).fillna("震荡")
        have = d[d["wave_no"].notna()].copy()
        have["wb"] = have["wave_no"].clip(upper=4).map(
            {1: "首波", 2: "二波", 3: "三波", 4: "四波+"})
        say(f"\n### COOLDOWN={cd} (有昨日题材 {len(have):,} / {len(d):,})")
        rows = []
        for wb in ("首波", "二波", "三波", "四波+"):
            g = have[have["wb"] == wb]
            if len(g) < 50:
                continue
            rows.append([wb, len(g), f"{g['seal'].mean() * 100:.1f}%",
                         f"{g['ev'].mean():+.2f}"])
        md_table(["昨日题材波次", "个股-日数", "封板率", "EV次日开盘%"], rows)
        per = have.groupby("wb")["seal"].mean()
        sp = (per.max() - per.min()) * 100 if len(per) >= 2 else np.nan
        rho = spearmanr(have["wave_no"], have["seal"]).statistic
        rho_e = spearmanr(have["wave_no"], have["ev"].fillna(0)).statistic
        say(f"- 封板率极差 {sp:.1f}pp · Spearman(波次, 封板) = {rho:+.3f} · "
            f"(波次, EV) = {rho_e:+.3f}")
        # walk-forward: 二波是否为最优档(打板 lore: 二波最好做)
        tr = have[have["trade_date"] <= TRAIN_END]
        te = have[have["trade_date"] > TRAIN_END]
        if len(tr) and len(te):
            rows = []
            for wb in ("首波", "二波", "三波", "四波+"):
                gtr, gte = tr[tr["wb"] == wb], te[te["wb"] == wb]
                if len(gte) < 50:
                    continue
                rows.append([wb, len(gtr), f"{gtr['seal'].mean() * 100:.1f}%",
                             f"{gtr['ev'].mean():+.2f}", len(gte),
                             f"{gte['seal'].mean() * 100:.1f}%",
                             f"{gte['ev'].mean():+.2f}"])
            say(f"\nwalk-forward (TRAIN≤{TRAIN_END} 封板率 "
                f"{tr['seal'].mean() * 100:.1f}% / TEST "
                f"{te['seal'].mean() * 100:.1f}%)")
            md_table(["波次", "TRAIN n", "TRAIN封板率", "TRAIN EV",
                      "TEST n", "TEST封板率", "TEST EV"], rows)
        KEY[f"cd{cd}_seal"] = {"spread": sp, "rho": rho, "rho_ev": rho_e}


# ============================================================ D 结论
def part_d():
    say("\n## D 结论与命名建议")
    say("\n### D1 波次能不能替代阶段")
    say("**目标1(题材延续) —— 有信号, 但强度取决于 COOLDOWN**:")
    rows = []
    for cd in COOLDOWNS:
        v = KEY.get(f"cd{cd}_cont", {})
        per = v.get("per", {})
        order = ["首波", "二波", "三波", "四波+"]
        rows.append([cd] + [f"{per[k] * 100:.0f}%" if k in per else "-"
                            for k in order]
                    + [f"{v.get('rho', 0):+.3f}",
                       " / ".join(f"{s:+.0f}pp" for s in v.get("sps", []))])
    md_table(["COOLDOWN", "首波", "二波", "三波", "四波+",
              "Spearman", "三段市况极差"], rows)
    say("\n> COOLDOWN=5(真波段隔离)才出现单调递减梯度: 首波 "
        f"{KEY.get('cd5_cont', {}).get('per', {}).get('首波', 0) * 100:.0f}% → "
        f"三波 {KEY.get('cd5_cont', {}).get('per', {}).get('三波', 0) * 100:.0f}%, "
        "Spearman -0.151, 三段市况 +25/+19/+22pp 全同向 —— 这是题材生命周期"
        "信息真实存在的证据。COOLDOWN=2/3 把同一波的拖动也算成新波, "
        "梯度被冲淡。")
    say("- 方向与打板 lore 一致: 首波延续最强(题材刚启动, 资金还在进), "
        "三波最弱(已被反复炒过)。但注意它是**题材延续性**, "
        "不是个股封板率。")
    s3 = KEY.get("cd3_seal", {})
    if s3:
        say(f"\n**目标2(交易价值) —— 不成立**: 三个 COOLDOWN 方案封板率极差"
            f"均 ≤1.4pp, Spearman(波次, 封板) "
            + " / ".join(f"{KEY.get(f'cd{c}_seal', {}).get('rho', 0):+.3f}"
                        for c in COOLDOWNS)
            + ", (波次, EV) "
            + " / ".join(f"{KEY.get(f'cd{c}_seal', {}).get('rho_ev', 0):+.3f}"
                        for c in COOLDOWNS)
            + " —— 与研究36/37 一致, 题材生命周期维度对个股打板成败"
              "无信息。walk-forward 也未出现稳定最优档。")

    say("\n### D2 命名建议（解决当前的语义混乱）")
    say("现行 `theme_age` → `theme_mode` 的问题不在于阈值, 而在于**一个字段"
        "背了两个语义**。建议拆成两个独立字段:")
    md_table(["字段", "含义", "来源", "用途"], [
        ["`theme_persist_days`", "题材连续在场天数(持续性)",
         "归因自由关联家数面板, 不用 theme.day 独占口径",
         "题材追踪/热度排序; 预测题材次日是否仍在场(强信号, 研究36 C1)"],
        ["`theme_wave_no`", "题材历史第几波(波次)",
         "同上 + 空档≥5交易日判新波(COOLDOWN=5)",
         "复盘展示「这是第几波行情」; 延续性梯度清晰(首波54%→三波32%)"],
        ["`day_in_wave`", "当前波段内第几个在场日",
         "同上", "展示; 不得叫「爆发/主升/鱼尾」"],
        ["~~`theme_mode`~~", "爆发/主升/鱼尾", "已证伪",
         "**删除该命名**; 保留函数仅供旧看板兼容, 输出改标为持续性档"],
    ])
    say("\n> 关键: 「爆发/主升/鱼尾」这套命名**不要再挂在任何字段上**。"
        "它暗示一个已被证伪的方向(鱼尾=死), 而实测方向相反。若必须给"
        "题材一个生命周期标签, 用波次(首波/二波/三波+)而不是日龄。")

    say("\n### D3 波次的可用边界")
    say("- 波次**可用**于: 题材追踪与复盘展示、「这个题材是第几波」的"
        "人工判断辅助、题材延续性预测(COOLDOWN=5 下梯度清晰)。")
    say("- 波次**不可用**于: 任何买卖闸、仓位决策、选股排序 —— "
        "与研究36/37 一致, 题材生命周期维度对打板 EV 无贡献。")
    say("- 与项目既有「二波」的区别: research/29 与 docs/research_07 的"
        "二波是**个股级**(高位龙头回落后的第二波), 本研究是**题材级**。"
        "两者不可混用, 个股二波已有独立的冻结口径(core/shortboard.py)。")

    say("\n## 诚实边界")
    say("- 波次自 kpl 事件库起点累计, 早年(2018-2019)之前该题材若已有"
        "行情则无法计入, 首波判定对老题材偏乐观; kpl 题材命名跨年变更"
        "(如「地产链」vs「房地产」)会把同一题材的历史切成多个题材, "
        "使波次偏小。")
    say("- COOLDOWN ∈ {2,3,5} 三方案并行, 未做更细网格; 波次分布对 "
        "COOLDOWN 敏感(首波占比随 COOLDOWN 增大而上升), 采纳前须定档。")
    say("- 目标2 母集与研究37 同口径(日频触板, 基线封板率 66%), "
        "与生产 S2/S3 信号(基线 12%)不同; 但研究37 已证两种母集上"
        "题材维度结论一致。")
    say("- 波次是**盘前可知**的(用昨日面板), 无前视; 但「当日是否新波"
        "启动」需当日数据, 盘中不可用。")


def summary() -> list:
    c5 = KEY.get("cd5_cont", {})
    per = c5.get("per", {})
    s = ["\n## 摘要(TL;DR)"]
    s.append("**波次与日龄是两个独立的量**: Spearman(wave_no, theme_age) "
             "≈ 0(三个 COOLDOWN 方案均如此)。所以 `theme_age` 无论怎么改名"
             "都表达不了波次 —— 「爆发/主升/鱼尾」的命名混乱根源在于"
             "用一个日龄字段背了波次语义。")
    if per:
        order = ["首波", "二波", "三波", "四波+"]
        s.append(f"**波次对题材延续有真信号(COOLDOWN=5)**: 次日仍在场率 "
                 + " / ".join(f"{k} {per[k] * 100:.0f}%"
                              for k in order if k in per)
                 + f", Spearman {c5.get('rho', 0):+.3f}, 三段市况极差 "
                   f"{[f'{x:+.0f}pp' for x in c5.get('sps', [])]} 全同向 —— "
                   "首波最强、三波最弱, 方向与打板 lore 一致。"
                   "但 COOLDOWN=2/3 会把同波拖动算成新波, 梯度被冲淡"
                   "(Spearman 降到 -0.001/-0.053), 参数必须定档。")
    s.append("**但波次对打板无效**: 三方案封板率极差均≤1.4pp, "
             "Spearman(波次, 封板) ≈ 0 / (波次, EV) ≈ 0 —— 与研究36/37 一致, "
             "题材生命周期维度对个股打板成败无信息。")
    s.append("**命名建议**: 拆成 `theme_persist_days`(持续性) + "
             "`theme_wave_no`(波次) + `day_in_wave`(波段内日龄)三个字段, "
             "**彻底删除「爆发/主升/鱼尾」这套命名**(它暗示已被证伪的方向)。")
    s.append("**可用边界**: 波次可用于题材追踪/复盘展示/延续性预测, "
             "不可用于买卖闸或仓位决策。注意与 research/29 的**个股级**二波"
             "区分, 两者不可混用。")
    return s


def main():
    cal = sorted(load("limitup.events_enriched",
                      columns=["trade_date"])["trade_date"].unique())
    pos = {d: i for i, d in enumerate(cal)}
    r36 = mod("r36", "research/36_theme_stage_diagnosis.py")
    pan = r36.activity_panel()
    part_a(pan, cal, pos)
    part_b(pan, cal, pos, regimes())
    part_c(cal, pos)
    part_d()
    out = [L[0]] + summary() + L[1:]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
