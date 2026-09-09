# -*- coding: utf-8 -*-
"""研究36: 个股级 S 因子重定义 —— 多方案并行对比

背景(已有研究汇总):
  · 研究12/14c/15/16 定了现行 S1/S2/S3 规则; 暴拉分支已按 EV 证伪删除
  · 研究30 实盘复核: S3「竞价量爆」绝对阈值 vr≥5 失效(96.7%触发率),
    改横截面分位后仍只 13.2%; 5 个正增量因子 amt_speed/theme_zt/tover/
    vwap_dev/ramp; 加全部20个新因子反而降 AUC
  · 研究24b 定稿 V5: 结构闸 g_chip≥CHIP_GATE + 融合分排序, 次日胜率
    69.7%→74.6%, Sharpe 7.75→9.24
  · 研究31: 竞价量比闸是封板率强因子但**次日胜率反向**, 只作密度过滤器
  · 研究35: 决策时刻可知因子里 pct 严格单调但 pct≥6% EV 负;
    T3+S2 = 全体系最优(TEST 42.3%/EV+1.00); S3 有 98% 的量来自失效分支
  · 定稿结论: **打板 EV 由入场价位而非封板率决定**, 封板率与 EV 系统性反向

问题: 现行 S1/S2/S3 **不是质量阶梯** —— S2/S3 是并列分支且 S2 质量高于
S3; S3 的量被已证失效的分支占据。本研究按用户方法论(多方案并行 + 数据
对比选优)评估候选重定义, 判据用 **EV 而非封板率**(定稿结论)。

方案:
  A0 现状          S1=pct≥1 / S2=pct≥2+颠簸 / S3=高开稳封相|高开剧震|量爆
  A1 删失效分支    S3 只留「高开稳封相」, 删高开剧震与量爆分支
  A2 结构闸升级    S3 = S2 且 V5 结构闸通过(g_chip≥CHIP_GATE);
                   高开稳封相并入 S2 标记, 不再占 S3
  A3 pct门槛       S2 加 pct≥4(研究35 lift 4.60x), S3 = S2且pct≥6
  A4 T联动         S3 = S2 且 T3(题材启动确认); S2 要求 T≥T2

产物: research/out/36_s_redefine.md
用法: python research/36_s_redefine.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.structure import CHIP_GATE  # noqa: E402
from core.theme_signal import (LD_NEG_GAP, T2_MIN_ZT, T3_MIN_HT,  # noqa: E402
                               T3_MIN_ZT)
from datastore import load  # noqa: E402

LIVE = DATA / "live"
OUT = ROOT / "research" / "out"
REPORT = OUT / "36_s_redefine.md"
DAYS = ["20260827", "20260828", "20260831", "20260901", "20260902",
        "20260903", "20260904"]
TRAIN = ["20260827", "20260828", "20260831", "20260901"]
TEST = ["20260902", "20260903", "20260904"]
L = []


def say(s=""):
    print(s)
    L.append(s)


def md_table(header, rows):
    say("| " + " | ".join(header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")


def sec(hms) -> int:
    s = str(hms or "").replace(":", "")
    if len(s) < 6 or not s[:6].isdigit():
        return 0
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


# ---------------------------------------------------------------- 卖出模拟
# 定稿卖出规则的模拟已抽到 core/exit_rules.py 作唯一出处,
# research/36 与 research/backfill_tsig.py 共用它避免口径分叉。
from core.exit_rules import simulate_exit  # noqa: E402


# ---------------------------------------------------------------- 数据集
def build() -> pd.DataFrame:
    """7 日全部 S1/S2/S3 信号 + 结果标签 + T级 + V5结构闸

    结果标签分两层(用户指正: 提高封板率的目标是抓**高度连板票**,
    首板封死与5板封死的 alpha 完全不同, 不能当等价):
      seal     当日封板(任意高度)
      seal_lb  当日封板且连板数≥2  ← 真正的 alpha 来源
      next_seal 次日仍在涨停事件表(续板) ← 能不能接到二波
    """
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "first_time", "limit_times"])
    sealed = {(r.trade_date, r.ts_code) for r in ev.itertuples()}
    lb_of = {(r.trade_date, r.ts_code): int(r.limit_times or 1)
             for r in ev.itertuples()}
    seal_sec: dict = {}
    for r in ev.itertuples():
        seal_sec.setdefault(r.trade_date, {})[r.ts_code] = sec(r.first_time)
    # 昨日连板数(盘前已知, 无前视): 0 = 昨日未涨停
    lb_by_day: dict = {}
    for r in ev.itertuples():
        lb_by_day.setdefault(r.trade_date, {})[r.ts_code] = int(
            r.limit_times or 1)

    stock2con, _, _ = load_maps()
    con2stock = load_con2stock()
    td = load("theme.day")
    dates = sorted(td["trade_date"].unique())
    prev = {d: (dates[i - 1] if i > 0 else None) for i, d in enumerate(dates)}
    byday = {d: {r.concept_code: r for r in g.itertuples()}
             for d, g in td.groupby("trade_date")}

    panel = load("market.daily_panel",
                 columns=["trade_date", "ts_code", "open", "pre_close",
                          "close"])
    pxd: dict = {}
    for r in panel.itertuples():
        pxd.setdefault(r.trade_date, {})[r.ts_code] = r
    pdates = sorted(panel["trade_date"].unique())
    nxt = {d: (pdates[i + 1] if i + 1 < len(pdates) else None)
           for i, d in enumerate(pdates)}

    rows = []
    ipx_cache: dict = {}
    for d in DAYS:
        f = LIVE / f"presig_state_{d}.json"
        if not f.exists():
            continue
        sigs = json.loads(f.read_text(encoding="utf-8"))["signals"]
        yday = byday.get(prev.get(d), {}) if prev.get(d) else {}
        ss = seal_sec.get(d, {})
        nd = nxt.get(d)
        # 买入当日分时(算当日最高价作止损参考价) + 次日分时(卖出模拟)
        def _ipx(day):
            if day is None:
                return {}
            if day not in ipx_cache:
                fp = LIVE / f"intraday_px_{day}.json"
                try:
                    ipx_cache[day] = json.loads(
                        fp.read_text(encoding="utf-8")) if fp.exists() else {}
                except Exception:
                    ipx_cache[day] = {}
            return ipx_cache[day]
        ipx_d, ipx_nd = _ipx(d), _ipx(nd)
        for s in sigs:
            c = s.get("ts_code")
            if not c:
                continue
            tsec0 = sec(s.get("pt"))
            # ---- T级(与 core/theme_signal 同判据, 锚定昨日天梯) ----
            cands = [yday[k] for k in stock2con.get(c, []) if k in yday]
            hot = max(cands, key=lambda x: x.zt_cnt) if cands else None
            if hot is None:
                T = "T1"
            else:
                zt_live = sum(1 for m in con2stock.get(hot.concept_code, [])
                              if 0 < ss.get(m, 0) <= tsec0) if tsec0 else 0
                lc = hot.leader_code
                ld_gap = None
                pc = pxd.get(d, {}).get(lc) if lc else None
                if pc and pc.pre_close and pc.pre_close > 0 and pc.open:
                    ld_gap = (pc.open / pc.pre_close - 1) * 100
                if ld_gap is not None and ld_gap <= LD_NEG_GAP:
                    T = "T1"
                elif zt_live >= T3_MIN_ZT and hot.max_height >= T3_MIN_HT:
                    T = "T3"
                elif zt_live >= T2_MIN_ZT:
                    T = "T2"
                else:
                    T = "T1"
            pb = s.get("pb")
            o = pxd.get(nd, {}).get(c) if nd else None
            is_seal = (d, c) in sealed
            lb = lb_of.get((d, c), 0)
            pd_ = prev.get(d)
            y_lb = lb_by_day.get(pd_, {}).get(c, 0) if pd_ else 0
            # 成交判定: 挂限价单 pb 等回踩, 信号后价格曾 ≤ pb 才算成交
            # (研究30 口径)。未成交就没盈亏, 不能计入盈亏比
            ph = s.get("px_hist") or []
            pt0 = str(s.get("pt") or "")
            fill = bool(pb) and any(
                len(e) >= 2 and e[1] and str(e[0]) >= pt0 and e[1] <= pb
                for e in ph)
            # 真实卖出规则模拟(需次日分时 + 次日涨停价 + 买入当日最高)
            ev_exit, exit_why = None, None
            if fill and pb and nd:
                nd_px = pxd.get(nd, {}).get(c)
                pts_nd = ipx_nd.get(c) or []
                pts_d = ipx_d.get(c) or []
                day_hi = max((float(e[1]) for e in pts_d
                              if len(e) >= 2 and e[1] and e[1] > 0),
                             default=0.0)
                lp_nd = 0.0
                if nd_px and nd_px.pre_close and nd_px.pre_close > 0:
                    ratio = 0.20 if c[:2] in ("30", "68") else 0.10
                    lp_nd = nd_px.pre_close * (1 + ratio)
                ev_exit, exit_why = simulate_exit(
                    pts_nd, pb, day_hi, lp_nd,
                    nd_px.close if nd_px else 0.0)
            rows.append({
                "date": d, "ts_code": c, "name": s.get("name", ""),
                "stage": s.get("stage"), "branch": s.get("why", ""),
                "pct": s.get("pct"), "pathvol": s.get("pathvol"),
                "accel": s.get("accel"), "tsec": tsec0, "pb": pb,
                "g_chip": (s.get("struct") or {}).get("g_chip"),
                "gate": (s.get("struct") or {}).get("gate"),
                "v5": (s.get("struct") or {}).get("v5"),
                "auc_gate": (s.get("auc") or {}).get("gate"),
                "T": T, "fill": fill,
                "y_lb": y_lb,
                "seal": is_seal,
                "lb": lb,
                "seal_lb": bool(is_seal and lb >= 2),
                "next_seal": bool(nd and (nd, c) in sealed),
                "ev": ((o.open / pb - 1) * 100
                       if o and pb and o.open and o.open > 0 else None),
                "ev_close": ((o.close / pb - 1) * 100
                             if o and pb and o.close and o.close > 0
                             else None),
                "ev_exit": ev_exit, "exit_why": exit_why})
    df = pd.DataFrame(rows)
    df["ev"] = pd.to_numeric(df["ev"], errors="coerce")
    df["ev_exit"] = pd.to_numeric(df["ev_exit"], errors="coerce")
    return df


# ---------------------------------------------------------------- 方案
def scheme_levels(df: pd.DataFrame, name: str) -> pd.Series:
    """按方案给每条信号重新定 S 级(数字越大越好)"""
    st, br = df["stage"], df["branch"]
    s2_base = (st == "S2")
    gap_ok = br.str.contains("高开稳封相", na=False)
    if name == "A0":                       # 现状
        return st
    if name == "A1":                       # 删失效分支
        return pd.Series(
            ["S1" if x == "S1" else "S2" if x == "S2"
             else ("S3" if g else None)
             for x, g in zip(st, gap_ok)], index=df.index)
    if name == "A2":                       # 结构闸升级
        out = []
        for x, g, gate in zip(st, gap_ok, df["gate"].fillna(False)):
            if x == "S1":
                out.append("S1")
            elif x == "S2" or g:           # 高开稳封相并入 S2
                out.append("S3" if gate else "S2")
            else:
                out.append(None)           # 量爆/剧震分支不再产信号
        return pd.Series(out, index=df.index)
    if name == "A3":                       # pct 门槛
        out = []
        for x, g, p in zip(st, gap_ok, df["pct"].fillna(0)):
            if x == "S1":
                out.append("S1")
            elif (x == "S2" or g) and p >= 6:
                out.append("S3")
            elif (x == "S2" or g) and p >= 4:
                out.append("S2")
            else:
                out.append(None)
        return pd.Series(out, index=df.index)
    if name == "A4":                       # T 联动
        out = []
        for x, g, T in zip(st, gap_ok, df["T"]):
            if x == "S1":
                out.append("S1" if T != "T1" else None)
            elif (x == "S2" or g) and T == "T3":
                out.append("S3")
            elif (x == "S2" or g) and T == "T2":
                out.append("S2")
            else:
                out.append(None)
        return pd.Series(out, index=df.index)
    # ---- 第二批: 先删掉竞价/开盘量爆分支(用户定为无观察意义) ----
    # core = 颠簸确认(S2) 或 高开类 S3, 不含量爆分支
    core = (st == "S2") | gap_ok | br.str.contains("高开剧震", na=False)
    gate = df["gate"].fillna(False)
    pct = df["pct"].fillna(0)
    if name == "B1":                       # 删量爆 + T联动
        return pd.Series(
            ["S3" if c and T == "T3" else "S2" if c and T == "T2" else None
             for c, T in zip(core, df["T"])], index=df.index)
    if name == "B2":                       # 删量爆 + T联动 + 结构闸
        return pd.Series(
            ["S3" if c and T == "T3" and gt else
             "S2" if c and T in ("T2", "T3") else None
             for c, T, gt in zip(core, df["T"], gate)], index=df.index)
    if name == "B3":                       # 删量爆 + T3 + 入场价位上限
        return pd.Series(
            ["S3" if c and T == "T3" and p < 6 else
             "S2" if c and T == "T2" and p < 6 else None
             for c, T, p in zip(core, df["T"], pct)], index=df.index)
    if name == "B4":                       # 三重叠加(最严)
        return pd.Series(
            ["S3" if c and T == "T3" and gt and p < 6 else None
             for c, T, gt, p in zip(core, df["T"], gate, pct)],
            index=df.index)
    # ---- 第三批: 纳入昨日连板高度(盘前已知, 无前视) ----
    # G1 证实盈亏比随高度单调上升(首板2.18→2板2.82→3板4.13), 但现行
    # S 因子完全不看连板高度。注意定稿策略 MAX_LIANBAN=1(只买昨日
    # 首板, n≥2 剔除), 故 C 组分两条路线并行对比: 守定稿 vs 放宽。
    ylb = df["y_lb"].fillna(0)
    if name == "C1":                       # 守定稿: 昨日首板 + T3
        return pd.Series(
            ["S3" if c and T == "T3" and y == 1 else
             "S2" if c and T == "T2" and y == 1 else None
             for c, T, y in zip(core, df["T"], ylb)], index=df.index)
    if name == "C2":                       # 守定稿 + 结构闸
        return pd.Series(
            ["S3" if c and T == "T3" and y == 1 and gt else None
             for c, T, y, gt in zip(core, df["T"], ylb, gate)],
            index=df.index)
    if name == "C3":                       # 放宽: 昨日≥2板 + T3
        return pd.Series(
            ["S3" if c and T == "T3" and y >= 2 else None
             for c, T, y in zip(core, df["T"], ylb)], index=df.index)
    if name == "C4":                       # 放宽: 昨日≥2板 + T3 + 结构闸
        return pd.Series(
            ["S3" if c and T == "T3" and y >= 2 and gt else None
             for c, T, y, gt in zip(core, df["T"], ylb, gate)],
            index=df.index)
    if name == "C5":                       # 昨日涨停即可(≥1板) + T3 + 结构闸
        return pd.Series(
            ["S3" if c and T == "T3" and y >= 1 and gt else None
             for c, T, y, gt in zip(core, df["T"], ylb, gate)],
            index=df.index)
    if name == "C6":                       # 不限高度(对照): T3 + 结构闸 = B2
        return pd.Series(
            ["S3" if c and T == "T3" and gt else None
             for c, T, gt in zip(core, df["T"], gate)], index=df.index)
    raise ValueError(name)


SCHEMES = ["A0", "A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
           "C1", "C2", "C3", "C4", "C5"]
SCHEME_DESC = {
    "A0": "现状(S1/S2/S3 原样)",
    "A1": "删失效分支(S3只留高开稳封相)",
    "A2": "结构闸升级(S3=S2且V5结构闸通过)",
    "A3": "pct门槛(S2需≥4%, S3需≥6%)",
    "A4": "T联动(S3=S2且T3, S2需T2)",
    "B1": "删量爆 + T联动",
    "B2": "删量爆 + T联动 + 结构闸",
    "B3": "删量爆 + T3 + 入场价位<6%",
    "B4": "三重叠加(删量爆+T3+结构闸+价位<6%)",
    "C1": "守定稿: 昨日首板 + T3",
    "C2": "守定稿: 昨日首板 + T3 + 结构闸",
    "C3": "放宽: 昨日≥2板 + T3",
    "C4": "放宽: 昨日≥2板 + T3 + 结构闸",
    "C5": "昨日涨停即可(≥1板) + T3 + 结构闸",
}


def main():
    df = build()
    say("# 研究36: 个股级 S 因子重定义（多方案并行对比）")
    say(f"窗口 {DAYS[0]}~{DAYS[-1]}({len(DAYS)}日) · 信号 {len(df)} 条 "
        f"· S1 {int((df['stage'] == 'S1').sum())} / "
        f"S2 {int((df['stage'] == 'S2').sum())} / "
        f"S3 {int((df['stage'] == 'S3').sum())}")
    say("\n判据用 **EV 而非封板率**——项目定稿结论「打板 EV 由入场价位而非"
        "封板率决定」, 封板率与 EV 系统性反向。")

    say("\n## A 现状体检: S 级不是质量阶梯")
    rows = []
    for st in ("S1", "S2", "S3"):
        sub = df[df["stage"] == st]
        rows.append([st, len(sub), f"{sub['seal'].mean() * 100:.1f}%",
                     f"{sub['ev'].mean():+.2f}"
                     if sub["ev"].notna().any() else "-"])
    md_table(["现行S级", "信号数", "封板率", "EV次日开盘"], rows)
    say("\nS3 分支拆解(量被失效分支占据):")
    rows = []
    for br, sub in df[df["stage"] == "S3"].groupby("branch"):
        rows.append([br, len(sub), f"{len(sub) / max(1, int((df['stage'] == 'S3').sum())) * 100:.0f}%",
                     f"{sub['seal'].mean() * 100:.1f}%",
                     f"{sub['ev'].mean():+.2f}"
                     if sub["ev"].notna().any() else "-"])
    rows.sort(key=lambda x: -x[1])
    md_table(["S3分支", "信号数", "占S3比", "封板率", "EV"], rows)

    say("\n## B 五方案并行（全窗口）")
    for sc in SCHEMES:
        lv = scheme_levels(df, sc)
        say(f"\n### {sc} {SCHEME_DESC[sc]}")
        rows = []
        for L_ in ("S3", "S2", "S1"):
            sub = df[lv == L_]
            if not len(sub):
                rows.append([L_, 0, "-", "-"])
                continue
            rows.append([L_, len(sub), f"{sub['seal'].mean() * 100:.1f}%",
                         f"{sub['ev'].mean():+.2f}"
                         if sub["ev"].notna().any() else "-"])
        dropped = int(lv.isna().sum())
        md_table(["级", "信号数", "封板率", "EV次日开盘"], rows)
        say(f"被剔除(不再产信号): {dropped} 条 "
            f"({dropped / len(df) * 100:.0f}%)")

    say("\n## C 顶层(S3)质量对比 —— 选优判据")
    rows = []
    for sc in SCHEMES:
        lv = scheme_levels(df, sc)
        sub = df[lv == "S3"]
        rows.append([f"{sc} {SCHEME_DESC[sc]}", len(sub),
                     f"{sub['seal'].mean() * 100:.1f}%" if len(sub) else "-",
                     f"{sub['ev'].mean():+.2f}"
                     if len(sub) and sub["ev"].notna().any() else "-"])
    rows.sort(key=lambda x: -(float(x[3]) if x[3] != "-" else -99))
    md_table(["方案", "S3信号数", "S3封板率", "S3 EV"], rows)

    say("\n## D 单调性检验（S1<S2<S3 是否成立, 按EV）")
    rows = []
    for sc in SCHEMES:
        lv = scheme_levels(df, sc)
        evs = {}
        for L_ in ("S1", "S2", "S3"):
            sub = df[lv == L_]
            evs[L_] = sub["ev"].mean() if len(sub) and sub["ev"].notna().any() \
                else None
        ok = (evs["S1"] is not None and evs["S2"] is not None
              and evs["S3"] is not None
              and evs["S1"] <= evs["S2"] <= evs["S3"])
        rows.append([sc,
                     f"{evs['S1']:+.2f}" if evs["S1"] is not None else "-",
                     f"{evs['S2']:+.2f}" if evs["S2"] is not None else "-",
                     f"{evs['S3']:+.2f}" if evs["S3"] is not None else "-",
                     "✓ 单调" if ok else "✗ 不单调"])
    md_table(["方案", "S1 EV", "S2 EV", "S3 EV", "单调性"], rows)

    say("\n## E walk-forward（TRAIN 选 → TEST 样本外验）")
    say("**怎么读这张表**: 只看 TEST 三列。TRAIN 好看可能是过拟合; "
        "判据用 EV 不用封板率(定稿结论: 打板EV由入场价位而非封板率决定, "
        "两者系统性反向)。TEST EV 为正且封板率不低于基线 = 候选可用。")
    tr = df[df["date"].isin(TRAIN)]
    te = df[df["date"].isin(TEST)]
    say(f"TRAIN {TRAIN} {len(tr)}条 基线封板率 {tr['seal'].mean() * 100:.1f}% "
        f"EV {tr['ev'].mean():+.2f} | TEST {TEST} {len(te)}条 "
        f"基线 {te['seal'].mean() * 100:.1f}% EV {te['ev'].mean():+.2f}")
    rows = []
    for sc in SCHEMES:
        ltr, lte = scheme_levels(tr, sc), scheme_levels(te, sc)
        a, b = tr[ltr == "S3"], te[lte == "S3"]
        ok = (len(b) and b["ev"].notna().any() and b["ev"].mean() > 0
              and b["seal"].mean() > te["seal"].mean())
        rows.append([f"{sc} {SCHEME_DESC[sc]}", len(a),
                     f"{a['seal'].mean() * 100:.1f}%" if len(a) else "-",
                     f"{a['ev'].mean():+.2f}"
                     if len(a) and a["ev"].notna().any() else "-",
                     len(b),
                     f"{b['seal'].mean() * 100:.1f}%" if len(b) else "-",
                     f"{b['ev'].mean():+.2f}"
                     if len(b) and b["ev"].notna().any() else "-",
                     "✓ 可用" if ok else "✗"])
    md_table(["方案", "TRAIN S3数", "TRAIN封板率", "TRAIN EV",
              "TEST S3数", "TEST封板率", "TEST EV", "判定"], rows)

    say("\n## F 连板捕获能力（用户指正的正确判据）")
    say("提高封板率的目标不是「封住」而是「抓到高度连板票」——首板封死"
        "与 5 板封死的 alpha 完全不同。故除 seal 外再看三个口径:")
    say("  · **连板捕获率** = 封板且连板数≥2 的占比(真 alpha 来源)")
    say("  · **续板率**     = 次日仍在涨停事件表(能不能接到二波)")
    say("  · **分层 EV**    = 首板 EV vs 连板 EV(验证高度是否真带来溢价)")

    say("\n### F1 封板票的高度分层 EV(全窗口, 验证高度是否带 alpha)")
    rows = []
    sl = df[df["seal"]]
    for tag, lo, hi in (("首板(lb=1)", 1, 1), ("2板", 2, 2), ("3板", 3, 3),
                        ("4板及以上", 4, 99)):
        sub = sl[(sl["lb"] >= lo) & (sl["lb"] <= hi)]
        if not len(sub):
            continue
        rows.append([tag, len(sub),
                     f"{sub['next_seal'].mean() * 100:.1f}%",
                     f"{sub['ev'].mean():+.2f}"
                     if sub["ev"].notna().any() else "-"])
    md_table(["封板高度", "样本数", "次日续板率", "EV次日开盘"], rows)

    say("\n### F2 九方案的连板捕获能力（TEST 样本外）")
    te = df[df["date"].isin(TEST)]
    base_lb = te["seal_lb"].mean()
    base_ns = te[te["seal"]]["next_seal"].mean() if te["seal"].any() else 0
    say(f"TEST 基线: 连板捕获率 {base_lb * 100:.2f}% · "
        f"封板票续板率 {base_ns * 100:.1f}%")
    rows = []
    for sc in SCHEMES:
        lv = scheme_levels(te, sc)
        sub = te[lv == "S3"]
        if not len(sub):
            rows.append([f"{sc} {SCHEME_DESC[sc]}", 0, "-", "-", "-", "-"])
            continue
        slb = sub[sub["seal"]]
        rows.append([
            f"{sc} {SCHEME_DESC[sc]}", len(sub),
            f"{sub['seal'].mean() * 100:.1f}%",
            f"{sub['seal_lb'].mean() * 100:.1f}%",
            f"{(slb['next_seal'].mean() * 100):.1f}%" if len(slb) else "-",
            f"{sub['ev'].mean():+.2f}"
            if sub["ev"].notna().any() else "-"])
    rows.sort(key=lambda x: -(float(x[3].rstrip("%"))
                              if x[3] != "-" else 0))
    md_table(["方案", "S3数", "封板率", "**连板捕获率**", "续板率",
              "EV次日开盘"], rows)

    say("\n### F3 连板捕获数绝对值（同样重要: 率高但量少也没用）")
    rows = []
    for sc in SCHEMES:
        lv = scheme_levels(te, sc)
        sub = te[lv == "S3"]
        rows.append([f"{sc} {SCHEME_DESC[sc]}",
                     int(sub["seal"].sum()), int(sub["seal_lb"].sum()),
                     int(sub["next_seal"].sum())])
    rows.sort(key=lambda x: -x[2])
    md_table(["方案", "TEST封板总数", "**TEST连板捕获数**", "TEST续板数"],
             rows)

    say("\n## G 盈亏比评估（研究24b 定稿口径）")
    say("EV 均值掩盖了分布。打板的真实盈亏结构是「小赢大亏」还是"
        "「大赢小亏」, 均值看不出来。本节只看**成交样本**(挂限价单"
        " pb 等回踩, 信号后价格曾≤pb 才算成交), 未成交无盈亏不计入。")
    say("  · 胜率     = 收益>0 的占比")
    say("  · 盈亏比   = 平均盈利 / |平均亏损|")
    say("  · 期望     = 胜率×平均盈利 + (1-胜率)×平均亏损")

    def pl_stats(sub: pd.DataFrame, col: str) -> tuple:
        """(样本数, 胜率, 平均盈利, 平均亏损, 盈亏比, 期望)"""
        v = sub[col].dropna()
        if not len(v):
            return 0, None, None, None, None, None
        w, l = v[v > 0], v[v <= 0]
        aw = w.mean() if len(w) else 0.0
        al = l.mean() if len(l) else 0.0
        ratio = (aw / abs(al)) if al < 0 else None
        exp = (len(w) / len(v)) * aw + (len(l) / len(v)) * al
        return len(v), len(w) / len(v), aw, al, ratio, exp

    say("\n### G1 封板高度分层的盈亏结构（全窗口, 次日收盘离场）")
    say("验证用户假设: 高度连板票的盈亏比是否真的更好。")
    rows = []
    sl = df[df["seal"] & df["fill"]]
    for tag, lo, hi in (("首板(lb=1)", 1, 1), ("2板", 2, 2), ("3板", 3, 3),
                        ("4板及以上", 4, 99)):
        sub = sl[(sl["lb"] >= lo) & (sl["lb"] <= hi)]
        n, wr, aw, al, ratio, exp = pl_stats(sub, "ev_close")
        if not n:
            continue
        rows.append([tag, n, f"{wr * 100:.1f}%", f"{aw:+.2f}", f"{al:+.2f}",
                     f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}"])
    md_table(["封板高度", "成交样本", "胜率", "平均盈利", "平均亏损",
              "**盈亏比**", "期望"], rows)

    say("\n### G2 封板 vs 未封板的盈亏结构（全窗口成交样本）")
    rows = []
    for tag, sub in (("封板票", df[df["seal"] & df["fill"]]),
                     ("未封板票", df[~df["seal"] & df["fill"]]),
                     ("全部成交", df[df["fill"]])):
        for col, lbl in (("ev", "次日开盘离场"), ("ev_close", "次日收盘离场")):
            n, wr, aw, al, ratio, exp = pl_stats(sub, col)
            if not n:
                continue
            rows.append([f"{tag} · {lbl}", n, f"{wr * 100:.1f}%",
                         f"{aw:+.2f}", f"{al:+.2f}",
                         f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}"])
    md_table(["分组", "成交样本", "胜率", "平均盈利", "平均亏损",
              "**盈亏比**", "期望"], rows)

    say("\n### G3 九方案的盈亏比（TEST 样本外, 次日收盘离场）")
    say("**怎么读**: 期望>0 且盈亏比>1 才是真能赚。封板率高但盈亏比<1"
        " = 赢的次数多但赢得少、亏得多, 长期仍亏。")
    te = df[df["date"].isin(TEST)]
    n, wr, aw, al, ratio, exp = pl_stats(te[te["fill"]], "ev_close")
    say(f"TEST 基线(全部成交): {n}样本 胜率{wr * 100:.1f}% "
        f"盈亏比{ratio:.2f} 期望{exp:+.2f}" if n else "TEST 基线: 无成交样本")
    rows = []
    for sc in SCHEMES:
        lv = scheme_levels(te, sc)
        sub = te[(lv == "S3") & te["fill"]]
        n, wr, aw, al, ratio, exp = pl_stats(sub, "ev_close")
        if not n:
            rows.append([f"{sc} {SCHEME_DESC[sc]}", 0, "-", "-", "-",
                         "-", "-", "✗"])
            continue
        ok = exp > 0 and ratio and ratio > 1
        rows.append([f"{sc} {SCHEME_DESC[sc]}", n, f"{wr * 100:.1f}%",
                     f"{aw:+.2f}", f"{al:+.2f}",
                     f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}",
                     "✓" if ok else "✗"])
    rows.sort(key=lambda x: -(float(x[6]) if x[6] not in ("-",) else -99))
    md_table(["方案", "成交样本", "胜率", "平均盈利", "平均亏损",
              "**盈亏比**", "期望", "判定"], rows)

    say("\n## H 昨日连板高度维度（新增, 对准 alpha 来源）")
    say("G1 证实盈亏比随高度单调上升, 但现行 S 因子完全不看连板高度。"
        "本节先做单因子分档, 再看守定稿(MAX_LIANBAN=1) vs 放宽两条路线。")

    say("\n### H1 昨日连板高度单因子分档（全窗口）")
    rows = []
    for tag, lo, hi in (("昨日未涨停(y_lb=0)", 0, 0), ("昨日首板(y_lb=1)", 1, 1),
                        ("昨日2板", 2, 2), ("昨日3板及以上", 3, 99)):
        sub = df[(df["y_lb"] >= lo) & (df["y_lb"] <= hi) & df["fill"]]
        n, wr, aw, al, ratio, exp = pl_stats(sub, "ev_close")
        if not n:
            continue
        rows.append([tag, len(df[(df["y_lb"] >= lo) & (df["y_lb"] <= hi)]),
                     n,
                     f"{sub['seal'].mean() * 100:.1f}%",
                     f"{sub['seal_lb'].mean() * 100:.1f}%",
                     f"{wr * 100:.1f}%",
                     f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}"])
    md_table(["昨日连板高度", "信号数", "成交样本", "封板率",
              "**连板捕获率**", "胜率", "盈亏比", "期望"], rows)

    say("\n### H2 C组方案的盈亏比（TEST 样本外）")
    te = df[df["date"].isin(TEST)]
    rows = []
    for sc in ["C1", "C2", "C3", "C4", "C5", "B2"]:
        lv = scheme_levels(te, sc)
        sub = te[(lv == "S3") & te["fill"]]
        n, wr, aw, al, ratio, exp = pl_stats(sub, "ev_close")
        allsub = te[lv == "S3"]
        if not n:
            rows.append([f"{sc} {SCHEME_DESC[sc]}", len(allsub), 0, "-",
                         "-", "-", "-", "-"])
            continue
        ok = exp > 0 and ratio and ratio > 1
        rows.append([f"{sc} {SCHEME_DESC[sc]}", len(allsub), n,
                     f"{allsub['seal_lb'].mean() * 100:.1f}%",
                     f"{wr * 100:.1f}%",
                     f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}",
                     "✓" if ok else "✗"])
    rows.sort(key=lambda x: -(float(x[6]) if x[6] != "-" else -99))
    md_table(["方案", "S3数", "成交样本", "连板捕获率", "胜率",
              "**盈亏比**", "期望", "判定"], rows)

    say("\n## I 真实卖出规则重算期望（定稿口径 vs 简单口径）")
    say("前面所有结论都用「次日开盘/收盘离场」算, 但定稿策略的卖出端是"
        "分层的(回落止损5% → P2 10:10封板唯一判据 → 14:57强平 → 封板续持)。"
        "本节按真实规则逐点重算, 看结论是否改变。")
    say("**数据限制必须先看**: intraday_px 来自 radar_log(只录 pct≥1% 或"
        " prob≥0.2 的票), 58% 的票轨迹在 14:00 前就断了。轨迹不足的"
        "样本回退次日收盘价并单独标记, **不冒充完整模拟**。")

    say("\n### I1 离场原因分布与各自盈亏（全窗口成交样本）")
    rows = []
    fl = df[df["fill"] & df["ev_exit"].notna()]
    for why, sub in fl.groupby("exit_why"):
        n, wr, aw, al, ratio, exp = pl_stats(sub, "ev_exit")
        rows.append([why, len(sub),
                     f"{len(sub) / len(fl) * 100:.0f}%",
                     f"{wr * 100:.1f}%", f"{aw:+.2f}", f"{al:+.2f}",
                     f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}"])
    rows.sort(key=lambda x: -x[1])
    md_table(["离场原因", "样本数", "占比", "胜率", "平均盈利",
              "平均亏损", "盈亏比", "期望"], rows)

    say("\n### I2 三种离场口径的期望对比（全窗口成交样本）")
    rows = []
    for col, lbl in (("ev", "次日开盘离场(简单)"),
                     ("ev_close", "次日收盘离场(简单)"),
                     ("ev_exit", "**真实卖出规则(定稿)**")):
        sub = df[df["fill"]]
        n, wr, aw, al, ratio, exp = pl_stats(sub, col)
        if not n:
            continue
        rows.append([lbl, n, f"{wr * 100:.1f}%", f"{aw:+.2f}",
                     f"{al:+.2f}", f"{ratio:.2f}" if ratio else "-",
                     f"{exp:+.2f}"])
    md_table(["离场口径", "成交样本", "胜率", "平均盈利", "平均亏损",
              "盈亏比", "**期望**"], rows)
    say("\n剔除「轨迹不足回退」后的纯模拟子集(只保留真正跑完规则的):")
    pure = fl[~fl["exit_why"].str.contains("回退", na=False)]
    n, wr, aw, al, ratio, exp = pl_stats(pure, "ev_exit")
    if n:
        say(f"  {n}样本 胜率{wr * 100:.1f}% 盈亏比"
            f"{ratio:.2f} 期望{exp:+.2f}"
            if ratio else f"  {n}样本 胜率{wr * 100:.1f}% 期望{exp:+.2f}")

    say("\n### I3 主要方案在真实卖出规则下的期望（TEST 样本外）")
    say("**这才是最终判据** —— 前面 G3/H2 用次日收盘算的期望可能低估了"
        "真实策略表现(定稿卖出端比「次日收盘」聪明得多)。")
    te = df[df["date"].isin(TEST)]
    n, wr, aw, al, ratio, exp = pl_stats(te[te["fill"]], "ev_exit")
    if n:
        say(f"TEST 基线(全部成交): {n}样本 胜率{wr * 100:.1f}% "
            f"盈亏比{ratio:.2f} 期望{exp:+.2f}" if ratio
            else f"TEST 基线: {n}样本 胜率{wr * 100:.1f}% 期望{exp:+.2f}")
    rows = []
    for sc in ["A0", "A2", "A4", "B1", "B2", "C1", "C2", "C5"]:
        lv = scheme_levels(te, sc)
        sub = te[(lv == "S3") & te["fill"]]
        n, wr, aw, al, ratio, exp = pl_stats(sub, "ev_exit")
        if not n:
            rows.append([f"{sc} {SCHEME_DESC[sc]}", 0, "-", "-", "-", "-"])
            continue
        ok = exp > 0
        rows.append([f"{sc} {SCHEME_DESC[sc]}", n, f"{wr * 100:.1f}%",
                     f"{ratio:.2f}" if ratio else "-", f"{exp:+.2f}",
                     "✓" if ok else "✗"])
    rows.sort(key=lambda x: -(float(x[4]) if x[4] != "-" else -99))
    md_table(["方案", "成交样本", "胜率", "盈亏比", "**期望**", "判定"], rows)

    say("\n## 诚实边界")
    say(f"- 窗口仅 {len(DAYS)} 个交易日, TRAIN {len(TRAIN)}日 / "
        f"TEST {len(TEST)}日, **未覆盖牛/熊/震荡三段**; "
        f"任何方案要写进 core/early_signal.py 前必须按三段市况重跑。")
    say("- T 级用 events.first_time ≤ 信号时刻计数(离线权威口径), 生产 "
        "core/theme_signal 用盘中贴死涨停价实时计数, 盘中值略低 → "
        "A4 方案的实盘 S3 数量会比这里少。")
    say("- EV 口径 = 信号买价 pb → 次日开盘; 未扣手续费与滑点, "
        "且 pb 缺失的信号不计入 EV(封板率仍计入)。")
    say("- A2/A3/A4 会改变 S2/S3 的**触发语义**, 属生产口径变更; "
        "本研究只做离线评估, 不改 core/early_signal.py。")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n落盘 {REPORT}")


if __name__ == "__main__":
    main()
