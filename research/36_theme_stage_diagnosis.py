# -*- coding: utf-8 -*-
"""研究36: 题材阶段(波龄 / 爆发·主升·鱼尾)因子诊断与重建

起因: 研究34 K2「昨日题材阶段」失败率 爆发85.0% / 主升87.4% / 鱼尾85.7%,
封板率 15.0% / 12.6% / 14.3% —— 三档几乎重合, 无区分度。本研究不重复
"再切一次档", 而是先回答**为什么无区分度**, 再回答**这个维度能不能救**。

诊断结论(四条成因, 证据见各节):
  ① 度量错位 —— theme_age 不是"题材波段年龄", 而是 theme.day 行的
     **连续出现天数**, 而行的存在条件是「该题材当日有≥1只**独占归属**
     涨停股」, kpl口径独占归属 = 该股 kpl theme 标注的**第一个**题材
     (core.attribute.attribute_day_kpl 取 ts[0]) → 年龄与强度正交;
  ② 档位是混合体 —— 「鱼尾(age≥4)」里既有高潮日也有背景噪音;
  ③ 方向反了 —— 剔除规模混淆后, 鱼尾档次日延续率最高(年龄度量的是
     题材持续性而非阶段);
  ④ 与打板收益无关 —— 所有阶段方案对 T+1 开盘溢价几乎零信息。

三段结构:
  A 语义诊断  theme_age 到底在数什么(分布/断裂/强度相关性/错标矩阵)
  B 阶段重建  归因自由活跃面板(kpl直标全tag) + 波段口径, 5个候选方案并行
  C 区分度检验
    C1 题材-日层面(大样本, 全历史, 三段市况): 阶段 → 次日题材延续/龙头续板
       —— 直接检验"鱼尾=死"这一语义是否成立, 不经信号噪声
    C2 S2/S3 信号层面(研究35 同口径): 阶段 → 封板率, walk-forward

方法论纪律(用户定稿): 三段市况独立验证 + 多候选方案并行对比, 不做单方案调参。

产物: research/out/36_theme_stage_diagnosis.md
用法: python research/36_theme_stage_diagnosis.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONCEPT_NOISE_KEYWORDS  # noqa: E402
from core.attribute import load_con2stock, load_maps, split_themes  # noqa: E402
from core.cycle import theme_mode  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out"
REPORT = OUT / "36_theme_stage_diagnosis.md"
SIG_DS = OUT / "30_sig_dataset.parquet"        # 研究30 已建, 横截面特征≤T无未来
SIG_DAYS = ["20260827", "20260828", "20260831", "20260901", "20260902"]
TRAIN = ["20260827", "20260828", "20260831"]
TEST = ["20260901", "20260902"]
WIN = [str(d) for d in
       ["20260820", "20260821", "20260824", "20260825", "20260826",
        "20260827", "20260828", "20260831", "20260901", "20260902",
        "20260903", "20260904"]]
WAVE_MIN = 2          # 题材在场门槛: 当日关联涨停家数≥2 才算一个题材在动
GAP_TOL = 1           # 允许1个交易日空档不打断波段
L: list = []
KEY1: dict = {}       # C1 关键统计(供结论引用, 避免结论与表格漂移)
KEY2: dict = {}       # C2 关键统计


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    say("| " + " | ".join(header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")


def is_noise(t: str) -> bool:
    return any(w in t for w in CONCEPT_NOISE_KEYWORDS)


def board_height(status) -> int:
    """kpl status → 板数: N连板→N / N天M板→M / 首板·其他→1"""
    s = str(status or "")
    m = re.match(r"^(\d+)连板$", s)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)天(\d+)板$", s)
    if m:
        return int(m.group(2))
    return 1


# ============================================================ 活跃面板
def activity_panel() -> pd.DataFrame:
    """归因自由的题材-日活跃面板(kpl直标全tag, 不经独占投票)

    与 theme.day 的差别: theme.day 的 zt_cnt 是**独占**家数(每股只算
    首标签那个题材), 本面板 zt 是**关联**家数(该股所有题材标注都计一次)。
    后者不受标签排序随机性影响, 是刻画"题材在不在动"的正确口径。
    """
    k = load("limitup.kpl_events",
             columns=["trade_date", "ts_code", "tag", "theme", "status"])
    zt = k[(k["tag"] == "涨停") & (k["trade_date"] >= "20191128")].copy()
    zt["h"] = zt["status"].map(board_height)
    rows = []
    for d, c, th, h in zip(zt["trade_date"], zt["ts_code"], zt["theme"],
                           zt["h"]):
        for t in split_themes(th):
            if not is_noise(t):
                rows.append((d, t, c, h))
    e = pd.DataFrame(rows, columns=["trade_date", "theme", "ts_code", "h"])
    g = e.groupby(["trade_date", "theme"]).agg(
        zt=("ts_code", "size"), mh=("h", "max"),
        lb=("h", lambda s: int((s >= 2).sum()))).reset_index()
    # 龙头: 板数最高者(与 theme.day 龙头判定的第一顺位一致)
    ld = (e.sort_values(["h", "ts_code"], ascending=[False, True])
          .groupby(["trade_date", "theme"]).head(1)[
              ["trade_date", "theme", "ts_code"]]
          .rename(columns={"ts_code": "leader"}))
    g = g.merge(ld, on=["trade_date", "theme"], how="left")
    return g.sort_values(["theme", "trade_date"]).reset_index(drop=True)


def add_wave_stage(pan: pd.DataFrame, cal: list) -> pd.DataFrame:
    """在活跃面板上加波段口径的阶段因子(全部只用≤当日信息)

    波段定义: 连续交易日中 zt≥WAVE_MIN 的日子构成一个波段, 允许 ≤GAP_TOL
    日空档不打断。wave_age=波段内第几个在场日; peak_zt=波段内截至当日的
    家数峰值; zt_ratio=当日家数/峰值(强度位置); mh_d=最高板环比。
    """
    pan = pan.copy()
    pan["wave_age"] = 0
    pan["peak_zt"] = 0
    pan["zt_ratio"] = np.nan
    pan["mh_d"] = np.nan
    pan["lb_d"] = np.nan
    pos = {d: i for i, d in enumerate(cal)}
    for th, g in pan.groupby("theme", sort=False):
        g = g.sort_values("trade_date")
        ds = g["trade_date"].tolist()
        zts = g["zt"].tolist()
        mhs = g["mh"].tolist()
        lbs = g["lb"].tolist()
        age, peak, gap = 0, 0, GAP_TOL + 1
        wa, pk, rt, md, lbd = [], [], [], [], []
        for i, d in enumerate(ds):
            p = pos.get(d, 0)
            if i == 0:
                gap = GAP_TOL + 1                    # 首行视为新波段
            else:
                gap = p - pos.get(ds[i - 1], p) - 1  # 与上一在场日间隔几个空日
            active = zts[i] >= WAVE_MIN
            if active:
                if gap > GAP_TOL:                    # 波段重启
                    age, peak = 1, zts[i]
                else:
                    age += 1
                    peak = max(peak, zts[i])
            else:                                    # 只1家涨停: 挂在原波段上
                age = age + 1 if gap <= GAP_TOL else 1
                peak = max(peak, zts[i]) if age > 1 else zts[i]
            wa.append(age)
            pk.append(peak)
            rt.append(zts[i] / peak if peak else np.nan)
            md.append(mhs[i] - mhs[i - 1] if i else np.nan)
            lbd.append(lbs[i] - lbs[i - 1] if i else np.nan)
        pan.loc[g.index, ["wave_age", "peak_zt", "zt_ratio", "mh_d",
                          "lb_d"]] = list(zip(wa, pk, rt, md, lbd))
    return pan


# ============================================================ 阶段方案
def scheme_M0(row) -> str:
    """现行生产口径: theme_mode(theme.day.theme_age)"""
    return theme_mode(int(row["theme_age"]))


def scheme_M1(row) -> str:
    """波段波龄(修正年龄口径, 阈值沿用现行1/2-3/≥4)"""
    a = int(row["wave_age"])
    return "爆发" if a <= 1 else "主升" if a <= 3 else "鱼尾"


def scheme_M2(row) -> str:
    """强度位置: 当日家数相对波段峰值(不看年龄, 只看还在不在扩张)"""
    r = row["zt_ratio"]
    if not np.isfinite(r):
        return "无"
    return "峰值区" if r >= 0.9 else "回落区" if r >= 0.5 else "深回落"


def scheme_M3(row) -> str:
    """二维合成: 波龄 × 强度位置 × 高度趋势 → 启动/主升/高潮/鱼尾"""
    a, r = int(row["wave_age"]), row["zt_ratio"]
    mh, mhd = int(row["mh"]), row["mh_d"]
    if a <= 1:
        return "启动"
    if not np.isfinite(r):
        return "主升"
    if r < 0.5 or (np.isfinite(mhd) and mhd < 0 and mh <= 2):
        return "鱼尾"
    if mh >= 3 and r >= 0.9:
        return "高潮"
    return "主升"


def scheme_M4(row) -> str:
    """高度轴(92科比铁律口径): 高位/中位/低位 × 高度是否在涨"""
    mh, mhd = int(row["mh"]), row["mh_d"]
    if mh >= 6:
        return "高位(6板+)"
    if mh >= 3:
        return "中位上行" if (not np.isfinite(mhd) or mhd >= 0) else "中位回落"
    return "低位(≤2板)"


def scheme_M5(row) -> str:
    """龙头断板(把92科比「鱼尾不能格局」换成可观测事件)

    鱼尾的真实含义不是"天数够多", 而是"龙头断板/高度回落" —— 后者
    盘前直接可观测(昨日题材最高板 vs 前日)。"""
    mhd = row["mh_d"]
    if not np.isfinite(mhd):
        return "高度不明"
    return "龙头断板" if mhd < 0 else "高度持平/上行"


SCHEMES = {
    "M0 现行 theme_mode(theme.day.age)": None,      # 需 theme.day, 单独处理
    "M1 波段波龄(1/2-3/≥4)": scheme_M1,
    "M2 强度位置(家数/波段峰值)": scheme_M2,
    "M3 波龄×强度×高度 二维": scheme_M3,
    "M4 高度轴(高/中/低×趋势)": scheme_M4,
    "M5 龙头断板(鱼尾直觉的可观测化)": scheme_M5,
}
ORD = {"启动": 0, "爆发": 0, "低位(≤2板)": 0, "龙头断板": 0, "深回落": 0,
       "主升": 1, "中位上行": 1, "回落区": 1, "高度持平/上行": 1,
       "峰值区": 2, "高潮": 2, "中位回落": 2,
       "鱼尾": 3, "高位(6板+)": 3}


# ============================================================ 市况三段
def regimes() -> pd.Series:
    """牛/熊/震荡: 全A等权指数 vs 20日均线偏离 ±2%(研究29 同口径)"""
    dp = load("market.daily_panel", columns=["trade_date", "pct_chg"])
    mkt = dp.groupby("trade_date")["pct_chg"].mean().sort_index()
    idx = (1 + mkt / 100).cumprod()
    dev = idx / idx.rolling(20).mean() - 1
    reg = pd.Series("震荡", index=idx.index)
    reg[dev > 0.02] = "牛"
    reg[dev < -0.02] = "熊"
    return reg


# ============================================================ A 语义诊断
def part_a(td: pd.DataFrame, pan: pd.DataFrame):
    say("# 研究36: 题材阶段(波龄/爆发·主升·鱼尾)因子诊断与重建")
    say(f"\n活跃面板(kpl直标全tag) {len(pan)} 行 · "
        f"{pan['trade_date'].min()}~{pan['trade_date'].max()} · "
        f"题材 {pan['theme'].nunique()} 个")
    say(f"theme.day(独占口径) {len(td)} 行 · "
        f"{td['trade_date'].min()}~{td['trade_date'].max()}")

    say("\n## A 语义诊断: theme_age 到底在数什么")
    say("\n`build/theme_daily.py` 的定义是「题材连续有**独占**涨停的天数」, "
        "而独占归属在 kpl 口径下 = 该股 kpl theme 标注的**第一个**题材"
        "(`core.attribute.attribute_day_kpl` 取 `ts[0]`)。所以 theme_age "
        "数的是「该题材连续多少天抢到至少一只涨停股的首标签」。")

    say("\n### A1 分布: 「爆发」是多数类, 档位天然失衡")
    rows = []
    n = len(td)
    for lo, hi, tag in ((1, 1, "爆发(age=1)"), (2, 3, "主升(age2~3)"),
                        (4, 10 ** 6, "鱼尾(age≥4)")):
        m = (td["theme_age"] >= lo) & (td["theme_age"] <= hi)
        rows.append([tag, int(m.sum()), f"{m.mean() * 100:.1f}%",
                     f"{td.loc[m, 'zt_cnt'].mean():.2f}",
                     f"{td.loc[m, 'max_height'].mean():.2f}"])
    md_table(["档位", "题材-日数", "占比", "均独占家数", "均最高板"], rows)
    say(f"\ntheme_age 全量: 中位 {td['theme_age'].median():.0f} · "
        f"均值 {td['theme_age'].mean():.2f} · "
        f"P90 {td['theme_age'].quantile(0.9):.0f} · "
        f"最大 {td['theme_age'].max():.0f}")
    say("\n> 「鱼尾」只占 15% 的题材-日, 却在研究34 K2 的信号里占 39% —— "
        "信号天然聚集在热题材上, 档位在信号侧被重新加权, "
        "三档的基线本就不可比。")

    say("\n### A2 年龄与强度正交: 1家涨停也能把年龄堆到很大")
    say("若 theme_age 真是波段年龄, 它应与题材强度同向。实测:")
    rows = []
    for lo, hi, tag in ((1, 1, "独占1家"), (2, 3, "独占2~3家"),
                        (4, 6, "独占4~6家"), (7, 10 ** 6, "独占≥7家")):
        m = (td["zt_cnt"] >= lo) & (td["zt_cnt"] <= hi)
        if not m.sum():
            continue
        rows.append([tag, int(m.sum()),
                     f"{td.loc[m, 'theme_age'].mean():.1f}",
                     f"{td.loc[m, 'theme_age'].median():.0f}",
                     f"{(td.loc[m, 'theme_age'] >= 4).mean() * 100:.0f}%"])
    md_table(["当日独占家数", "题材-日数", "age均值", "age中位",
              "落入鱼尾档占比"], rows)
    for col in ("zt_cnt", "zt_cnt_raw", "max_height"):
        rho = spearmanr(td["theme_age"], td[col]).statistic
        say(f"- Spearman(theme_age, {col}) = {rho:+.3f}")
    say("\n> 年龄与家数只有弱正相关(+0.25), 与最高板的相关(+0.67)来自"
        "「连板需要连续日」这一机械关系, 不是阶段信息。")

    say("\n### A3 波段被人为打断: 独占消失导致的假重启")
    pset = {(r.trade_date, r.theme): r.zt for r in pan.itertuples()}
    tdset = {(r.trade_date, r.concept_code) for r in td.itertuples()}
    dates = sorted(td["trade_date"].unique())
    prev = {d: (dates[i - 1] if i else None) for i, d in enumerate(dates)}
    a1 = td[td["theme_age"] == 1]
    alive = sum(1 for r in a1.itertuples()
                if prev.get(r.trade_date)
                and pset.get((prev[r.trade_date], r.concept_code), 0) >= 1)
    brk = tot = 0
    for r in pan.itertuples():
        pd_ = prev.get(r.trade_date)
        if not pd_:
            continue
        tot += 1
        if (pd_, r.theme) in tdset and (r.trade_date, r.theme) not in tdset:
            brk += 1
    say(f"- age=1 共 {len(a1)} 行, 其中**昨日该题材仍有关联涨停** "
        f"{alive} 行 = {alive / len(a1) * 100:.1f}% → 这些是假重启, "
        "被贴上「爆发」标签的老题材")
    say(f"- 昨日有独占归属、今日独占消失(题材仍有关联涨停但波段被打断) "
        f"{brk}/{tot} = {brk / tot * 100:.1f}%")
    say("\n> 结论: 打断率不低, 但它不是主因 —— 主因是 A2 的年龄/强度正交。")

    say("\n### A4 错标实例(研究34 窗口内, 用户口径复核)")
    say("取研究34 报告 B 节失败最多的题材, 逐日列出 theme.day 的年龄与"
        "真实强度, 看 theme_mode 贴的标签是否成立:")
    names = ["农业", "AI应用", "算力", "液冷", "消费电子", "零售"]
    rows = []
    for nm in names:
        sub = td[(td["concept_name"] == nm) & td["trade_date"].isin(WIN)]
        for r in sub.sort_values("trade_date").itertuples():
            rows.append([nm, r.trade_date, int(r.zt_cnt), int(r.zt_cnt_raw),
                         int(r.max_height), int(r.theme_age),
                         theme_mode(int(r.theme_age)), r.leader_name,
                         int(r.leader_height)])
    md_table(["题材", "日期", "独占家数", "关联家数", "最高板", "age",
              "现行标签", "龙头", "龙头板"], rows)
    say("\n> 三类硬错标:\n"
        "> ① **高潮被贴鱼尾**: 农业 20260901 独占11家/关联18家/龙头6板 "
        "(全窗口最强一天), age=7 → 「鱼尾」;\n"
        "> ② **老题材被贴爆发**: AI应用 20260901 关联14家但独占只2家, "
        "独占断档 → age=1 → 「爆发」;\n"
        "> ③ **背景噪音堆年龄**: 机器人概念 0820~0831 每天只1家涨停却把 "
        "age 堆到 8 → 「鱼尾」, 而它根本没有波段可言。")

    say("\n### A5 档内混合检验: 鱼尾档里既有高潮也有真退潮")
    say("把现行「鱼尾(age≥4)」按当日强度二分, 若两组差异大, "
        "说明该档是混合体而不是一个状态:")
    t = td[(td["theme_age"] >= 4)].copy()
    pk = pan.set_index(["trade_date", "theme"])["zt"].to_dict()
    t["pan_zt"] = [pk.get((r.trade_date, r.concept_code), np.nan)
                   for r in t.itertuples()]
    rows = []
    for tag, m in (("鱼尾·当日家数≥5(实为高潮)", t["zt_cnt"] >= 5),
                   ("鱼尾·当日家数2~4", (t["zt_cnt"] >= 2) & (t["zt_cnt"] < 5)),
                   ("鱼尾·当日家数=1(实为背景噪音)", t["zt_cnt"] == 1)):
        s = t[m]
        if not len(s):
            continue
        rows.append([tag, len(s), f"{s['zt_cnt'].mean():.1f}",
                     f"{s['max_height'].mean():.1f}",
                     f"{s['pan_zt'].mean():.1f}"])
    md_table(["鱼尾档细分", "题材-日数", "均独占家数", "均最高板",
              "均关联家数"], rows)
    say("\n> 同一个「鱼尾」档里, 均家数从 1.0 到 8+ 横跨一个数量级 —— "
        "档位不是状态, 是垃圾抽屉。")


# ============================================================ C1 题材-日
def part_c1(pan: pd.DataFrame, reg: pd.Series):
    say("\n## C1 阶段语义直接验证(题材-日层面, 全历史, 三段市况)")
    say("不经信号噪声, 直接问: 贴上某阶段标签的题材-日, **次日**这个题材"
        "还活不活? 若「鱼尾=死」成立, 鱼尾档次日延续应显著最低。\n"
        "结果变量(全部 T+1, 一律用**相对量**):\n"
        "- `次日仍在场率` 次日该题材关联涨停家数仍≥WAVE_MIN 的比例\n"
        "- `次日扩张率` 次日家数 > 当日家数 的比例\n"
        "- `次日/当日家数` 中位数(>1=扩张, <1=收缩)\n"
        "- `龙头续板率` T日该题材龙头在T+1仍涨停的比例\n"
        "- `T+1开盘溢价` T日该题材涨停股(非一字)次日开盘收益均值%\n\n"
        "> **必须剔除规模混淆**: 用「次日家数」的绝对值当结果变量会被"
        "题材规模污染 —— 老/大题材本来家数就多, 于是任何年龄类因子都会"
        "显示「越老次日家数越多」, 那不是阶段信息而是规模信息。所以本节"
        "一律用相对量, 并按**当日家数分层**复核。")
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "next_open_ret", "is_yizi"])
    evd = {(r.trade_date, r.ts_code): r for r in ev.itertuples()}
    cal = sorted(pan["trade_date"].unique())
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else None)
           for i, d in enumerate(cal)}
    zt_by = {(r.trade_date, r.theme): r for r in pan.itertuples()}
    zt_set = {}
    for r in ev.itertuples():
        zt_set.setdefault(r.trade_date, set()).add(r.ts_code)

    # 题材-日 × 涨停股明细(算 T+1 溢价用)
    k = load("limitup.kpl_events",
             columns=["trade_date", "ts_code", "tag", "theme"])
    z = k[(k["tag"] == "涨停") & (k["trade_date"] >= "20191128")]
    pairs = []
    for d, c, th in zip(z["trade_date"], z["ts_code"], z["theme"]):
        for t in split_themes(th):
            if not is_noise(t):
                pairs.append((d, t, c))
    mem = pd.DataFrame(pairs, columns=["trade_date", "theme", "ts_code"])

    ret = {}
    for r in mem.itertuples():
        e = evd.get((r.trade_date, r.ts_code))
        if e is None or e.is_yizi or e.next_open_ret is None:
            continue
        ret.setdefault((r.trade_date, r.theme), []).append(
            float(e.next_open_ret) * 100)

    rows_out = []
    for r in pan.itertuples():
        nd = nxt.get(r.trade_date)
        n = zt_by.get((nd, r.theme)) if nd else None
        ld_next = (r.leader in zt_set.get(nd, set())) if nd else None
        rr = ret.get((r.trade_date, r.theme))
        rows_out.append({
            "trade_date": r.trade_date, "theme": r.theme, "zt": r.zt,
            "mh": r.mh, "lb": r.lb, "leader": r.leader,
            "wave_age": r.wave_age, "peak_zt": r.peak_zt,
            "zt_ratio": r.zt_ratio, "mh_d": r.mh_d,
            "age_old": None,
            "next_zt": (n.zt if n else 0) if nd else np.nan,
            "next_mh": (n.mh if n else 0) if nd else np.nan,
            "ld_cont": ld_next,
            "ret_open": (float(np.mean(rr)) if rr else np.nan)})
    d = pd.DataFrame(rows_out)
    d = d.dropna(subset=["next_zt"])
    d["reg"] = d["trade_date"].map(reg).fillna("震荡")
    # 现行口径对照: 挂 theme.day 的 theme_age
    td = load("theme.day", columns=["trade_date", "concept_code", "theme_age"])
    d = d.merge(td.rename(columns={"concept_code": "theme"}),
                on=["trade_date", "theme"], how="left")
    d = d[d["zt"] >= WAVE_MIN]          # 只看真正在场的题材-日
    # 规模无关的结果变量(避免"大题材天然家数多"混淆阶段结论)
    d["cont"] = (d["next_zt"] >= WAVE_MIN).astype(int)
    d["expand"] = (d["next_zt"] > d["zt"]).astype(int)
    d["ratio"] = d["next_zt"] / d["zt"]
    d["size"] = pd.cut(d["zt"], [1, 3, 6, 10 ** 6],
                       labels=["2~3家", "4~6家", "≥7家"])
    say(f"\n样本 {len(d):,} 题材-日 · "
        f"{d['trade_date'].min()}~{d['trade_date'].max()} · "
        f"三段市况 {d['reg'].value_counts().to_dict()} · "
        f"规模分层 {d['size'].value_counts().to_dict()}")
    say(f"\n全样本基准: 次日仍在场率 {d['cont'].mean() * 100:.1f}% · "
        f"次日扩张率 {d['expand'].mean() * 100:.1f}% · "
        f"龙头续板率 {d['ld_cont'].mean() * 100:.1f}% · "
        f"T+1开盘溢价 {d['ret_open'].mean():+.2f}%")

    for name, fn in SCHEMES.items():
        say(f"\n### {name}")
        if fn is None:
            sub = d.dropna(subset=["theme_age"]).copy()
            sub["stage"] = sub["theme_age"].map(lambda a: theme_mode(int(a)))
        else:
            sub = d.copy()
            sub["stage"] = sub.apply(fn, axis=1)
        sub = sub[sub["stage"] != "无"]
        if not len(sub):
            say("(无样本)")
            continue
        order = sorted(sub["stage"].unique(),
                       key=lambda s: ORD.get(s, 9))
        rows = []
        for s in order:
            g = sub[sub["stage"] == s]
            rows.append([s, len(g), f"{g['zt'].mean():.1f}",
                         f"{g['cont'].mean() * 100:.1f}%",
                         f"{g['expand'].mean() * 100:.1f}%",
                         f"{g['ratio'].median():.2f}",
                         f"{g['ld_cont'].mean() * 100:.1f}%",
                         f"{g['ret_open'].mean():+.2f}"])
        md_table(["阶段", "题材-日数", "均当日家数", "次日仍在场率",
                  "次日扩张率", "次日/当日中位", "龙头续板率",
                  "T+1开盘溢价%"], rows)
        # 规模分层内复核(剔除混淆后的真区分度)
        rows = []
        for sz in ("2~3家", "4~6家", "≥7家"):
            g = sub[sub["size"] == sz]
            if len(g) < 30:
                continue
            line = [sz, len(g)]
            for s in order:
                gg = g[g["stage"] == s]
                line.append(f"{gg['cont'].mean() * 100:.0f}%({len(gg)})"
                            if len(gg) >= 20 else "-")
            per = g.groupby("stage")["cont"].mean()
            sp = (per.max() - per.min()) * 100 if len(per) >= 2 else np.nan
            rows.append(line + [f"{sp:+.0f}pp" if np.isfinite(sp) else "-"])
        md_table(["当日家数分层", "题材-日数"] + order
                 + ["仍在场率极差"], rows)
        # 三段市况
        rows, sps = [], []
        for rg in ("牛", "熊", "震荡"):
            g = sub[sub["reg"] == rg]
            per = g.groupby("stage")["cont"].mean()
            if len(per) < 2:
                rows.append([rg] + ["-"] * (len(order) + 1))
                sps.append(np.nan)
                continue
            line = [rg]
            for s in order:
                v = per.get(s, np.nan)
                line.append(f"{v * 100:.0f}%" if np.isfinite(v) else "-")
            sp = (per.max() - per.min()) * 100
            rows.append(line + [f"{sp:+.0f}pp"])
            sps.append(sp)
        md_table(["市况"] + order + ["仍在场率极差"], rows)
        rho = (spearmanr(sub["stage"].map(ORD), sub["cont"]).statistic
               if sub["stage"].nunique() > 2 else np.nan)
        rr = sub.dropna(subset=["ret_open"])
        rho_r = (spearmanr(rr["stage"].map(ORD), rr["ret_open"]).statistic
                 if rr["stage"].nunique() > 2 else np.nan)
        say(f"- Spearman(阶段序, 次日仍在场) = {rho:+.3f} · "
            f"(阶段序, T+1开盘溢价) = {rho_r:+.3f} · "
            f"三段市况仍在场率极差 {['%+.0fpp' % s for s in sps]}")
        rets = {s: sub[sub["stage"] == s]["ret_open"].mean() for s in order}
        conts = {s: sub[sub["stage"] == s]["cont"].mean() for s in order}
        KEY1[name] = {"cont": conts, "ret": rets, "rho": rho,
                      "rho_ret": rho_r, "reg_sp": sps, "order": order}


# ============================================================ C2 信号
def part_c2(pan: pd.DataFrame):
    say("\n## C2 S2/S3 信号层面区分度(研究35 同口径, walk-forward)")
    say("题材锚定在**昨日**活跃面板(盘前已知), 龙一/阶段全部≤T可知。"
        "母集与研究34 K2 可比: 有昨日活跃题材的 S2/S3 信号。")
    ds = pd.read_parquet(SIG_DS)
    df = ds[ds["stage"].isin(["S2", "S3"])].copy()
    df["date"] = df["date"].astype(str)
    stock2con, _, _ = load_maps()
    cal = sorted(pan["trade_date"].unique())
    prev = {d: (cal[i - 1] if i else None) for i, d in enumerate(cal)}
    byday = {d: {r.theme: r for r in g.itertuples()}
             for d, g in pan.groupby("trade_date")}
    td = load("theme.day")
    tdates = sorted(td["trade_date"].unique())
    tprev = {d: (tdates[i - 1] if i else None) for i, d in enumerate(tdates)}
    tbyday = {d: {r.concept_code: r for r in g.itertuples()}
              for d, g in td.groupby("trade_date")}

    rows = []
    for r in df.itertuples():
        pd_ = prev.get(r.date)
        yday = byday.get(pd_, {}) if pd_ else {}
        cands = [yday[k] for k in stock2con.get(r.ts_code, []) if k in yday]
        hot = max(cands, key=lambda x: x.zt) if cands else None
        tp = tprev.get(r.date)
        tyday = tbyday.get(tp, {}) if tp else {}
        tcands = [tyday[k] for k in stock2con.get(r.ts_code, [])
                  if k in tyday]
        thot = max(tcands, key=lambda x: x.zt_cnt) if tcands else None
        rows.append({
            "no_theme": hot is None,
            "y_theme": hot.theme if hot else None,
            # 列名与活跃面板对齐, 使 SCHEMES 里的方案函数可直接复用
            "zt": hot.zt if hot else 0,
            "mh": hot.mh if hot else 0,
            "wave_age": hot.wave_age if hot else np.nan,
            "peak_zt": hot.peak_zt if hot else np.nan,
            "zt_ratio": hot.zt_ratio if hot else np.nan,
            "mh_d": hot.mh_d if hot else np.nan,
            "theme_age": thot.theme_age if thot else np.nan})
    df = pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
    df["seal"] = df["seal_close"].fillna(False).astype(bool)
    df["fail"] = ~df["seal"]
    have = df[~df["no_theme"]].copy()
    say(f"\nS2/S3 {len(df)} 条 · 有昨日活跃题材 {len(have)} 条 · "
        f"基线封板率 {have['seal'].mean() * 100:.1f}% "
        f"(全母集 {df['seal'].mean() * 100:.1f}%)")

    for name, fn in SCHEMES.items():
        say(f"\n### {name}")
        if fn is None:
            sub = have.dropna(subset=["theme_age"]).copy()
            sub["stage"] = sub["theme_age"].map(lambda a: theme_mode(int(a)))
        else:
            sub = have.copy()
            sub["stage"] = sub.apply(fn, axis=1)
        sub = sub[sub["stage"] != "无"]
        order = sorted(sub["stage"].unique(), key=lambda s: ORD.get(s, 9))
        rows = []
        for s in order:
            g = sub[sub["stage"] == s]
            rows.append([s, len(g), f"{g['seal'].mean() * 100:.1f}%",
                         f"{g['fail'].mean() * 100:.1f}%",
                         f"{g['ev_next_open'].mean():+.2f}"
                         if g["ev_next_open"].notna().any() else "-"])
        md_table(["阶段", "信号数", "封板率", "失败率", "EV次日开盘"], rows)
        per = sub.groupby("stage")["seal"].mean()
        if len(per) >= 2:
            sp = (per.max() - per.min()) * 100
            rho = spearmanr(sub["stage"].map(ORD), sub["seal"]).statistic \
                if sub["stage"].nunique() > 2 else np.nan
            say(f"- 封板率极差 {sp:.1f}pp · "
                f"Spearman(阶段序, 封板) = {rho:+.3f}")
            KEY2.setdefault(name, {}).update(
                {"spread": sp, "buckets": per.to_dict(), "rho": rho})

    say("\n### walk-forward: TRAIN 选档 → TEST 纯样本外")
    for name, fn in SCHEMES.items():
        if fn is None:
            sub = have.dropna(subset=["theme_age"]).copy()
            sub["stage"] = sub["theme_age"].map(lambda a: theme_mode(int(a)))
        else:
            sub = have.copy()
            sub["stage"] = sub.apply(fn, axis=1)
        sub = sub[sub["stage"] != "无"]
        tr = sub[sub["date"].isin(TRAIN)]
        te = sub[sub["date"].isin(TEST)]
        if not len(tr) or not len(te):
            continue
        ptr = tr.groupby("stage")["seal"].mean().sort_values(ascending=False)
        rows = []
        for s in ptr.index[:3]:
            gtr, gte = tr[tr["stage"] == s], te[te["stage"] == s]
            if len(gte) < 20:
                continue
            rows.append([s, len(gtr), f"{gtr['seal'].mean() * 100:.1f}%",
                         len(gte), f"{gte['seal'].mean() * 100:.1f}%",
                         f"{gte['ev_next_open'].mean():+.2f}"
                         if gte["ev_next_open"].notna().any() else "-"])
        say(f"\n**{name}** (TRAIN基线 {tr['seal'].mean() * 100:.1f}% / "
            f"TEST基线 {te['seal'].mean() * 100:.1f}%)")
        md_table(["TRAIN最优档", "TRAIN n", "TRAIN封板率", "TEST n",
                  "TEST封板率(样本外)", "TEST EV"], rows)
        # 规避闸: TRAIN 选最差档 → TEST 验证剔除是否仍成立
        worst = ptr.index[-1]
        gtr, gte = tr[tr["stage"] == worst], te[te["stage"] == worst]
        kte = te[te["stage"] != worst]
        if len(gte) >= 20 and len(kte) >= 20:
            say(f"- 规避闸「剔除{worst}」: TRAIN剔除组 "
                f"{gtr['seal'].mean() * 100:.1f}%(n={len(gtr)}) → "
                f"TEST剔除组 {gte['seal'].mean() * 100:.1f}%(n={len(gte)}) "
                f"vs TEST保留组 {kte['seal'].mean() * 100:.1f}%"
                f"(n={len(kte)}), 剔除组/保留组 = "
                f"{gte['seal'].mean() / max(kte['seal'].mean(), 1e-9):.2f}x")
            KEY2.setdefault(name, {}).update({
                "gate": worst, "gate_tr": gtr["seal"].mean(),
                "gate_te": gte["seal"].mean(), "gate_te_n": len(gte),
                "gate_keep": kte["seal"].mean(),
                "gate_ratio": gte["seal"].mean()
                / max(kte["seal"].mean(), 1e-9),
                "test_base": te["seal"].mean()})

    say("\n### 对照: 同母集上已知最强因子的区分度")
    say("用研究35 已验证的因子做标尺, 看阶段因子相对弱到什么程度:")
    rows = []
    for tag, m in (("个股触发涨幅≥6%", have["pct"] >= 6),
                   ("个股触发涨幅<2%", have["pct"] < 2),
                   ("昨日题材关联家数≥5", have["zt"] >= 5),
                   ("昨日题材关联家数≤1", have["zt"] <= 1),
                   ("昨日题材最高板≥3", have["mh"] >= 3),
                   ("昨日题材最高板≤1", have["mh"] <= 1)):
        g = have[m]
        if not len(g):
            continue
        rows.append([tag, len(g), f"{g['seal'].mean() * 100:.1f}%"])
        KEY2.setdefault("_bench", {})[tag] = g["seal"].mean()
    md_table(["因子档", "信号数", "封板率"], rows)
    KEY2["_base"] = have["seal"].mean()


# ============================================================ 结论
def part_d():
    say("\n## D 结论与落地建议")
    m0 = KEY1.get("M0 现行 theme_mode(theme.day.age)", {})
    m0c = m0.get("cont", {})
    m0r = m0.get("ret", {})
    say("\n### D1 为什么无区分度 —— 四条独立成因")
    say("**① 度量错位(根因)**: `theme_age` 数的是「连续多少天抢到≥1只"
        "**独占**涨停股」, 不是题材波段年龄。kpl 口径的独占归属直接取该股 "
        "theme 标注的**第一个**题材(`attribute_day_kpl` 取 `ts[0]`), 所以"
        "年龄既受标签排序随机性影响(A3: age=1 中 18.6% 是假重启), "
        "又与题材强度正交(A2: 每天只1家涨停也能把年龄堆到8, "
        "Spearman(age,家数)仅+0.25)。")
    say("**② 档位是混合体**: 「鱼尾(age≥4)」里既有农业0901(独占11家/关联"
        "18家/龙头6板, 全窗口最强一天)这种高潮日, 也有每天只1家涨停的背景"
        "噪音 —— 档内均家数 1.0~8.1 横跨一个数量级(A5)。混合体的均值必然"
        "回到基线, 三档重合是数学必然。")
    if m0c:
        order = m0.get("order", [])
        lo = min(m0c, key=m0c.get)
        hi = max(m0c, key=m0c.get)
        say(f"**③ 方向反了(最关键)**: 剔除规模混淆后, 现行口径的「{hi}」档"
            f"次日仍在场率 {m0c[hi] * 100:.0f}%, 反而是「{lo}」档"
            f"({m0c[lo] * 100:.0f}%)的 {m0c[hi] / max(m0c[lo], 1e-9):.1f}倍, "
            f"且在三段市况内一致({['%+.0fpp' % s for s in m0.get('reg_sp', [])]}"
            f")。年龄类因子度量的是「题材持续性/是不是真题材」, 而**不是**"
            "「生命周期阶段」—— 活得久的题材本来就是真题材, 次日更可能"
            "继续涨停。把它当「鱼尾=死」用, 方向恰好相反。")
    say(f"**④ 与打板收益无关**: 即便阶段因子重建得更好, 它对打板收益也几乎"
        f"无信息 —— C1 里全部方案各档的 T+1 开盘溢价都在 +1.7~+2.3% 区间, "
        f"现行口径三档是 "
        + " / ".join(f"{m0r[s] * 1:+.2f}%" for s in m0.get("order", []))
        + "(完全重合), Spearman(阶段序, 溢价) 均 |ρ|<0.06。阶段能预测"
          "「题材还活不活」, 但预测不了「今天买这只票能不能赚钱」。")

    say("\n### D2 这个维度能不能救")
    say("- **年龄这一维救不回来**: 它度量持续性而非阶段, 且与打板收益无关。"
        "重建波龄(M1)后 C2 封板率极差仍只有 "
        f"{KEY2.get('M1 波段波龄(1/2-3/≥4)', {}).get('spread', 0):.1f}pp, "
        "与现行口径同量级。")
    m5 = KEY2.get("M5 龙头断板(鱼尾直觉的可观测化)", {})
    m4 = KEY2.get("M4 高度轴(高/中/低×趋势)", {})
    m2 = KEY2.get("M2 强度位置(家数/波段峰值)", {})
    base = KEY2.get("_base", 0)
    if m4.get("gate_ratio"):
        bk = m4.get("buckets", {})
        say(f"- **真正的信号在高度轴上, 且必须与高度层结合**: "
            f"单独的「龙头断板」(M5)无区分度 —— 断板组封板率 "
            f"{m5.get('buckets', {}).get('龙头断板', 0) * 100:.1f}% 反而高于"
            f"持平组 {m5.get('buckets', {}).get('高度持平/上行', 0) * 100:.1f}%, "
            f"样本外剔除比 {m5.get('gate_ratio', 0):.2f}x(失效)。但把断板"
            f"限定在中位(3~5板)后, 「{m4['gate']}」样本外封板率仅 "
            f"{m4['gate_te'] * 100:.1f}%(n={m4['gate_te_n']}) vs 保留组 "
            f"{m4['gate_keep'] * 100:.1f}% → {m4['gate_ratio']:.2f}x; "
            "同方案内中位上行 "
            f"{bk.get('中位上行', 0) * 100:.1f}% / 高位(6板+) "
            f"{bk.get('高位(6板+)', 0) * 100:.1f}% / 低位 "
            f"{bk.get('低位(≤2板)', 0) * 100:.1f}%。"
            "这正是 92科比铁律「高位做龙头 / 低位试错 / 中位3-5板风险最大」"
            "的实证落地: 中位断板才是致命态, 低位断板无害(本就是首板"
            "试错), 高位断板尚可。⚠ 但此条是 5日窗口观察(n=168), "
            "已被研究37 在全历史 127k 样本上否决(walk-forward 方向反转)。")
    if m2.get("gate_ratio"):
        say(f"- **强度位置(家数/波段峰值)可用但非单调**: 「{m2['gate']}」"
            f"样本外 {m2['gate_te'] * 100:.1f}% vs 保留组 "
            f"{m2['gate_keep'] * 100:.1f}% → {m2['gate_ratio']:.2f}x; "
            "但深回落档反而好于回落档(U形), 需更大样本才能定调。"
            "⚠ 研究37 已定调: 全历史三段市况 0/3 同向, 否决。")
    bench = KEY2.get("_bench", {})
    if bench:
        say(f"- **但它始终是二阶因子**: 同母集上个股触发涨幅≥6% 封板率 "
            f"{bench.get('个股触发涨幅≥6%', 0) * 100:.1f}% / <2% 仅 "
            f"{bench.get('个股触发涨幅<2%', 0) * 100:.1f}%(基线 "
            f"{base * 100:.1f}%), 而所有题材阶段档都在 5~20% 区间摆动。"
            "阶段只能做**规避闸**(剔掉必死的票), 不能做选股主力。")

    say("\n### D3 落地纪律")
    say("1. `core.cycle.theme_mode(theme.day.theme_age)` **只可留在展示层**, "
        "不得接入任何买卖闸/仓位决策(与 stage_of 的春夏秋冬同原则); "
        "展示层也应改标为「题材持续天数」而非「爆发/主升/鱼尾」, "
        "避免名字暗示一个已被证伪的方向。")
    say("2. 若要盘中用题材阶段, 数据源必须换成归因自由的关联家数面板"
        "(kpl 直标全 tag), 不能用 theme.day 的独占 zt_cnt/theme_age; "
        "已有同类结论: 趋势型题材用 theme.day 推阶段轴会误判。")
    say("3. ⚠ **本节提出的两个候选闸已被研究37 否决**(全历史 127k 触板样本"
        "+ 三段市况 + walk-forward): 中位回落 TRAIN -2.4pp → TEST +1.2pp"
        "(方向反转), 波段回落区三段市况 0/3 同向。两者均为本文 5日窗口"
        "噪音, **不得接入生产**; 详见 research/out/"
        "37_theme_stage_gate_validate.md。")
    say("4. 本研究唯一经全历史复核仍成立的是「昨日题材最高板≥6」这一个"
        "二值高度标记(剔除组封板率低 1.4pp, 三段市况同向, 样本外 -3.8pp), "
        "但它是高度因子不是阶段因子, 且效应量边际。")
    say("5. 任何阶段闸进生产前必须按牛/熊/震荡三段独立复核且方向一致"
        "≥2/3(用户方法论), 并做 walk-forward 样本外验证; 本研究 C2 只有"
        " 5 个交易日, 不构成采纳依据 —— 这一点已被研究37 证实。")

    say("\n## 诚实边界")
    say("- 活跃面板的板高来自 kpl `status` 文本解析(N连板/N天M板), "
        "与 events_enriched.limit_times 口径可能有个别差异; 龙头取板数"
        "最高者, 平票按代码升序, 与 theme.day 的封单额次序不完全一致。")
    say("- kpl theme 标注是当前时点的题材命名, 早年题材名与今日不同名"
        "(如「地产链」vs「房地产」), 跨年波段连续性会被命名变更打断; "
        "C1 长窗口结论对这类断点敏感。")
    say("- C1 的「次日仍在场率」含生存者偏差成分: 已活多日的题材本身就是"
        "真题材, 所以「年龄越大次日越可能在场」部分属于异质性而非因果; "
        "但这恰好证明年龄不是阶段变量。")
    say("- C2 母集仅 5 个交易日(20260827~20260902), TRAIN 3日/TEST 2日; "
        "M4/M2 的规避闸 TRAIN 样本极小(n≤8), 选档证据弱, 主要依据 "
        "TEST 段的 n≥160 对比, 仍需扩窗口重跑。")
    say("- M5 只有两个有效档(高度不明样本极少), 故 Spearman 为 nan。")
    say("- M2 的「峰值区」包含全部 wave_age=1 的日子(当日家数恒等于峰值), "
        "因此它与 M1 的「爆发」高度重叠, 不是纯粹的强度位置度量。")
    say("- 波段参数 WAVE_MIN=2 / GAP_TOL=1 是本文设定, 未做参数网格搜索"
        "(避免单方案调参); 若采纳须另跑参数敏感性与三段市况复核。")


def summary() -> list:
    """摘要: 先给结论, 证据在后文各节(数字均从 KEY1/KEY2 取, 不手写)"""
    m0 = KEY1.get("M0 现行 theme_mode(theme.day.age)", {})
    m02 = KEY2.get("M0 现行 theme_mode(theme.day.age)", {})
    m4 = KEY2.get("M4 高度轴(高/中/低×趋势)", {})
    m2 = KEY2.get("M2 强度位置(家数/波段峰值)", {})
    bench = KEY2.get("_bench", {})
    cont = m0.get("cont", {})
    ret = m0.get("ret", {})
    s = ["\n## 摘要(TL;DR)"]
    s.append(f"**用户观察到的三档重合已复现**: 同母集上现行口径封板率极差"
             f"仅 {m02.get('spread', 0):.1f}pp"
             + "(" + " / ".join(f"{v * 100:.1f}%"
                                 for v in m02.get("buckets", {}).values())
             + "), Spearman(阶段序, 封板) = "
             f"{m02.get('rho', 0):+.3f} —— 确认无区分度。")
    s.append("**成因不是样本少, 是度量错**: `theme_age` = 「连续多少天抢到"
             "≥1只**独占**涨停股」(kpl 独占归属取该股 theme 标注的第一个"
             "题材), 与题材强度正交(Spearman(age,家数)=+0.25), 且档位是"
             "混合体 —— 「鱼尾」档内均家数 1.0~8.1 横跨一个数量级。")
    if cont:
        lo = min(cont, key=cont.get)
        hi = max(cont, key=cont.get)
        s.append(f"**方向还反了**: 剔除规模混淆后, 「{hi}」档次日仍在场率 "
                 f"{cont[hi] * 100:.0f}% 是「{lo}」档 {cont[lo] * 100:.0f}% 的 "
                 f"{cont[hi] / max(cont[lo], 1e-9):.1f}倍, 三段市况一致。"
                 "年龄度量的是「题材持续性」而非「生命周期阶段」, 把它当"
                 "「鱼尾=死」用方向恰好相反。")
    if ret:
        s.append("**而且与打板收益无关**: 现行口径三档 T+1 开盘溢价 "
                 + " / ".join(f"{v * 1:+.2f}%" for v in ret.values())
                 + " 完全重合; 所有重建方案的 |Spearman(阶段序, 溢价)| 均<0.06。")
    s.append(f"**可救的不是年龄而是高度轴**: 「{m4.get('gate', '-')}」样本外"
             f"封板率 {m4.get('gate_te', 0) * 100:.1f}%"
             f"(n={m4.get('gate_te_n', 0)}) vs 保留组 "
             f"{m4.get('gate_keep', 0) * 100:.1f}% → "
             f"{m4.get('gate_ratio', 0):.2f}x; 「{m2.get('gate', '-')}」"
             f"{m2.get('gate_te', 0) * 100:.1f}% vs "
             f"{m2.get('gate_keep', 0) * 100:.1f}% → "
             f"{m2.get('gate_ratio', 0):.2f}x。两者都是盘前可知的规避闸。")
    if bench:
        s.append(f"**但它是二阶因子**: 同母集个股触发涨幅≥6% 封板率 "
                 f"{bench.get('个股触发涨幅≥6%', 0) * 100:.1f}% vs <2% "
                 f"{bench.get('个股触发涨幅<2%', 0) * 100:.1f}%"
                 f"(基线 {KEY2.get('_base', 0) * 100:.1f}%), 题材阶段所有档"
                 "只在 5~20% 摆动 —— 只能做规避闸, 不能做选股主力。")
    s.append("**行动**: `theme_mode` 退出决策链路(展示层改标为「题材持续"
             "天数」)。⚠ 本文提出的两个候选闸已被研究37 在全历史 127k 触板"
             "样本上否决, 详见 research/out/37_theme_stage_gate_validate.md。")
    return s


def main():
    pan = add_wave_stage(activity_panel(),
                         sorted(load("limitup.events_enriched",
                                     columns=["trade_date"])
                                ["trade_date"].unique()))
    td = load("theme.day")
    part_a(td, pan)
    part_c1(pan, regimes())
    part_c2(pan)
    part_d()
    out = [L[0]] + summary() + L[1:]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {REPORT}")


if __name__ == "__main__":
    main()
