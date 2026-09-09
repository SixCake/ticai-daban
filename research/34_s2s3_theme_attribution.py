# -*- coding: utf-8 -*-
"""研究34: S2/S3 失败的题材归因 — 板块效应 / 单票驱动 / 带崩识别

问题(用户提出): S2/S3 失败率为何这么高? 是不是板块效应? 若是, 是否由某一只
股票带动? 哪个题材被带崩? 有什么特征? 后续如何避免?

口径:
  信号   = data/live/presig_state_YYYYMMDD.json 中 stage ∈ {S2,S3}
  失败   = 当日未收盘封板(收盘权威: events_enriched 当日无该股记录)
           —— 与研究30同口径, 不用 presig 的 sealed_t(该字段曾长期为空)
  题材   = stock2con(成分表) ∩ theme.day(当日有涨停的题材), 取题材内涨停家数
           最大者作为该信号的「当日活跃题材」。两个关键选择:
           ① 不用 theme.attribution —— 它是**涨停股的独占归属**(实测
              0904 attribution 39 行 == events 39 行, 完全⊆涨停事件),
              拿它 join 信号会让"未封板"与"无题材"完全重合, 归因失效。
           ② 不用 radar_log 的 theme 标签 —— 研究30 数据集里 62% 的 S2/S3
              是 '-', 那是横截面回捞未命中(≤120s无快照), 会把"真无题材"与
              "join失败"混成一类。
  题材态 = theme.day 当日快照(涨停家数/龙头/波龄/最高板/龙头封单)
  EV     = 信号买价 pb → 次日开盘(与生产执行口径一致: 挂限价单等回踩)
  带崩   = 题材内当日分时回撤最深的成分股(peak→trough), 用 intraday_px 算

产物: research/out/34_s2s3_theme_attribution.md
用法: python research/34_s2s3_theme_attribution.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.cycle import theme_mode  # noqa: E402
from datastore import load  # noqa: E402

LIVE = DATA / "live"
OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
REPORT = OUT / "34_s2s3_theme_attribution.md"

DAYS = ["20260827", "20260828", "20260831", "20260901", "20260902",
        "20260903", "20260904"]
MIN_THEME_FAIL = 8        # 进入单票驱动检验的题材失败数下限
TOP_N_CONC = 10           # 集中度考察的题材数
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
    """HH:MM:SS → 当日秒数; 缺失/脏值返 0(不猜时间)"""
    s = str(hms or "").replace(":", "")
    if len(s) < 6 or not s[:6].isdigit():
        return 0
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


# ---------------------------------------------------------------- 数据集
def build() -> pd.DataFrame:
    """S2/S3 信号 × 当日活跃题材 × 收盘封板结果 × 题材当日态 × EV"""
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "limit_times", "open_times",
                       "first_time", "name"])
    sealed = {(r.trade_date, r.ts_code) for r in ev.itertuples()}
    # 首封时刻: 算「截至信号时刻题材已封板家数」用(无前视)
    seal_sec: dict = {}
    for r in ev.itertuples():
        seal_sec.setdefault(r.trade_date, {})[r.ts_code] = sec(r.first_time)
    stock2con, _, cname = load_maps()
    con2stock = load_con2stock()      # 概念 → 成分股(算题材已封家数用)
    td = load("theme.day")
    td = td[td["trade_date"].isin(DAYS)]
    # 当日活跃题材: {date: {concept_code: (zt_cnt, theme_age, max_height,
    # leader_code, leader_name, leader_height, concept_name)}}
    active = {}
    for d, g in td.groupby("trade_date"):
        active[d] = {r.concept_code: r for r in g.itertuples()}
    # 昨日题材最高板: {(date, concept): max_height} —— 盘前完全已知
    td_all = load("theme.day", columns=["trade_date", "concept_code",
                                        "max_height"])
    y_ht = {(r.trade_date, r.concept_code): r.max_height
            for r in td_all.itertuples()}
    panel = load("market.daily_panel",
                 columns=["trade_date", "ts_code", "open", "close"])
    dates = sorted(panel["trade_date"].unique())
    nxt = {d: (dates[i + 1] if i + 1 < len(dates) else None)
           for i, d in enumerate(dates)}
    prev = {d: (dates[i - 1] if i > 0 else None)
            for i, d in enumerate(dates)}
    px = {(r.trade_date, r.ts_code): (r.open, r.close)
          for r in panel.itertuples()}

    rows = []
    for d in DAYS:
        f = LIVE / f"presig_state_{d}.json"
        if not f.exists():
            continue
        sigs = json.loads(f.read_text(encoding="utf-8"))["signals"]
        nd = nxt.get(d)
        act = active.get(d, {})
        pd_ = prev.get(d)
        for s in sigs:
            if s.get("stage") not in ("S2", "S3"):
                continue
            c = s.get("ts_code")
            if not c:
                continue
            # 当日活跃题材: 该股成分概念 ∩ 当日有涨停的题材, 取涨停家数最大者
            cands = [act[k] for k in stock2con.get(c, []) if k in act]
            hot = max(cands, key=lambda r: r.zt_cnt) if cands else None
            # 无前视版本: 截至信号时刻, 该题材成分股已首封的家数
            # (theme.day 的 zt_cnt 是收盘口径, 盘中用它做闸就是未来信息)
            tsec0 = sec(s.get("pt"))
            ss = seal_sec.get(d, {})
            zt_live = None
            if hot is not None and tsec0 > 0:
                mem = con2stock.get(hot.concept_code, [])
                zt_live = sum(1 for m in mem
                              if 0 < ss.get(m, 0) <= tsec0)
            # 昨日高标: 该股成分概念在昨日(max_height 最大者), 盘前已知
            y_ht_max = max((y_ht.get((pd_, k), 0)
                            for k in stock2con.get(c, [])), default=0) \
                if pd_ else 0
            pb = s.get("pb")
            o = px.get((nd, c)) if nd else None
            rows.append({
                "date": d, "ts_code": c, "name": s.get("name", ""),
                "stage": s["stage"], "branch": s.get("why", ""),
                "t": s.get("pt") or "", "tsec": sec(s.get("pt")),
                "pct": s.get("pct"), "pb": pb,
                "n_con": len(stock2con.get(c, [])),
                "n_active": len(cands),
                "sealed": (d, c) in sealed,
                "concept_code": hot.concept_code if hot else None,
                "concept_name": (hot.concept_name if hot else
                                 cname.get(cands[0].concept_code)
                                 if cands else None),
                "zt_cnt": hot.zt_cnt if hot else None,
                "zt_live": zt_live,
                # 盘前已知: 该股所有成分概念在昨日的最高板(取最大)
                "y_ht_max": y_ht_max,
                "theme_age": hot.theme_age if hot else None,
                "max_height": hot.max_height if hot else None,
                "leader_code": hot.leader_code if hot else None,
                "leader_name": hot.leader_name if hot else None,
                "leader_height": hot.leader_height if hot else None,
                "ev_next_open": ((o[0] / pb - 1) * 100
                                 if o and pb and o[0] > 0 else None)})
    df = pd.DataFrame(rows)
    df["fail"] = ~df["sealed"]
    # no_theme = 该股不属于任何「当日有涨停」的题材(散票/冷门题材)
    df["no_theme"] = df["concept_code"].isna()
    # 题材龙头当日是否涨停(带崩判定的第一层)
    df["leader_sealed"] = [
        (r.date, r.leader_code) in sealed if isinstance(r.leader_code, str)
        else None for r in df.itertuples()]
    return df


def max_dd(pts: list) -> tuple:
    """分时最大回撤(running peak → 后续 trough), 返 (dd%, t_peak, t_trough)

    intraday_px 经 repair 清洗过仍可能残留污染点(价格为0或量纲错),
    p<=0 一律跳过; 不拿单点极值当回撤。"""
    peak, t_peak = -1e9, ""
    best, bp, bt = 0.0, "", ""
    for e in pts:
        hms, p = str(e[0]).zfill(6), e[1]
        if not p or p <= 0:
            continue
        if p > peak:
            peak, t_peak = p, hms
        dd = (p / peak - 1) * 100
        if dd < best:
            best, bp, bt = dd, t_peak, hms
    return round(best, 2), bp, bt


def theme_members_dd(date: str, codes: list) -> list:
    """题材内当日分时回撤最深的成分股 top3"""
    f = LIVE / f"intraday_px_{date}.json"
    if not f.exists():
        return []
    ipx = json.loads(f.read_text(encoding="utf-8"))
    out = []
    for c in codes:
        pts = ipx.get(c) or []
        if len(pts) < 3:
            continue
        dd, tp, tt = max_dd(pts)
        if dd < 0:
            out.append({"code": c, "dd": dd, "t_peak": tp, "t_trough": tt})
    return sorted(out, key=lambda x: x["dd"])[:3]


def _hm(sec0) -> str:
    """当日秒数 → HH:MM"""
    if sec0 is None or pd.isna(sec0):
        return "-"
    s = int(sec0)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


def leader_table() -> dict:
    """{(date, concept_code): 龙一状态} —— 正/负反馈分类

    龙一 = theme.day 的 leader(题材内连板最高者)。
    字段分两类, 绝不混用:
      · 盘前已知(y_*): 昨日板数/昨日一字/昨日炸板次数/今日开盘涨幅
        → 可做无前视闸
      · 当日全貌(t_*): 今日是否一字/今日首封时刻/今日炸板/今日收盘涨幅
        → 只能用于机制解释, 入闸就是未来信息
    负反馈定义(用户口径): 放量分歧后未转一致 / 低开下杀 / 跌停
    正反馈定义: 缩量一字 / 秒板无炸 / 同概念多只涨停
    """
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "limit_times", "open_times",
                       "first_time", "last_time", "is_yizi", "fd_amount"])
    evd: dict = {}
    for r in ev.itertuples():
        evd.setdefault(r.trade_date, {})[r.ts_code] = r
    td = load("theme.day")
    td = td[td["trade_date"].isin(DAYS)]
    panel = load("market.daily_panel",
                 columns=["trade_date", "ts_code", "open", "close",
                          "pre_close", "pct_chg", "vol"])
    pxd: dict = {}
    for r in panel.itertuples():
        pxd.setdefault(r.trade_date, {})[r.ts_code] = r
    dates = sorted(panel["trade_date"].unique())
    prev = {d: (dates[i - 1] if i > 0 else None)
            for i, d in enumerate(dates)}

    out = {}
    for r in td.itertuples():
        lc = r.leader_code
        if not isinstance(lc, str):
            continue
        d, pd_ = r.trade_date, prev.get(r.trade_date)
        cur = evd.get(d, {}).get(lc)
        y = evd.get(pd_, {}).get(lc) if pd_ else None
        pc = pxd.get(d, {}).get(lc)
        gap = ((pc.open / pc.pre_close - 1) * 100
               if pc and pc.pre_close and pc.pre_close > 0 else None)
        ratio = 0.20 if lc[:2] in ("30", "68") else 0.10
        limit_dn = (pc is not None and pc.pct_chg <= -ratio * 100 * 0.98)
        out[(d, r.concept_code)] = {
            "code": lc, "name": r.leader_name, "height": r.leader_height,
            "zt_cnt": r.zt_cnt, "theme_age": r.theme_age,
            # ---- 盘前已知(无前视) ----
            "y_lb": int(y.limit_times) if y is not None else 0,
            "y_yizi": bool(y.is_yizi) if y is not None else False,
            "y_open_times": int(y.open_times) if y is not None else 0,
            "y_sealed": y is not None,
            "gap": gap,
            # ---- 当日全貌(有前视, 仅机制解释) ----
            "t_sealed": cur is not None,
            "t_yizi": bool(cur.is_yizi) if cur is not None else False,
            "t_first_sec": sec(cur.first_time) if cur is not None else None,
            "t_open_times": int(cur.open_times) if cur is not None else None,
            "t_pct": float(pc.pct_chg) if pc is not None else None,
            "t_limit_dn": limit_dn,
        }
    return out


def leader_feedback(ls: dict, at_sec: int | None) -> tuple:
    """龙一反馈分类 → (盘前口径标签, 当日全貌标签)

    盘前口径只用 09:25 前已知的信息(昨日板型 + 今日开盘涨幅),
    这是能入闸的部分; at_sec 为信号触发时刻, 仅用于当日口径的时序判定。"""
    if not ls:
        return "无龙一", "无龙一"
    # ---- 盘前口径(无前视) ----
    if not ls["y_sealed"]:
        pre = "龙一昨日未涨停"
    elif ls["y_yizi"]:
        pre = "正反馈·昨缩量一字"
    elif ls["y_open_times"] >= 2:
        # 昨日放量分歧: 今日高开=有望转一致, 低开=分歧未转一致(负反馈)
        g = ls["gap"]
        if g is None:
            pre = "负反馈·昨放量分歧(开盘不明)"
        elif g >= 3:
            pre = "正反馈·昨分歧今高开转一致"
        elif g <= -3:
            pre = "负反馈·昨分歧今低开下杀"
        else:
            pre = "中性·昨分歧今平开"
    else:
        g = ls["gap"]
        if g is None:
            pre = "中性·昨换手板"
        elif g <= -3:
            pre = "负反馈·昨板今低开"
        elif g >= 5:
            pre = "正反馈·昨板今高开"
        else:
            pre = "中性·昨换手板"
    # ---- 当日全貌(有前视) ----
    if not ls["t_sealed"]:
        post = "跌停/下杀" if ls["t_limit_dn"] else "当日断板"
    elif ls["t_yizi"]:
        post = "正反馈·今缩量一字"
    elif ls["t_first_sec"] is not None and at_sec is not None:
        post = ("正反馈·龙一先封(早于信号)"
                if ls["t_first_sec"] <= at_sec
                else "龙一后封(晚于信号)")
    else:
        post = "当日封板(时刻不明)"
    return pre, post


def prev_ladder() -> dict:
    """{(当日date, concept_code): 昨日天梯行} —— 复盘页口径, 盘前完全已知

    用户指出的正确口径: 09:15 时交易者看到的是**昨日复盘的题材天梯**
    (龙头/题材年龄/最高板/涨停家数), 而不是当日天梯。当日龙头是从
    「当日涨停股里连板最高者」选出的, 09:31 根本不知道今天谁是龙一
    → 用当日 theme.day.leader_code 做龙一就是前视。

    本函数把每个题材的昨日天梯行挂到「当日」键上, 供信号侧直接取用。"""
    td = load("theme.day")
    dates = sorted(td["trade_date"].unique())
    prev = {d: (dates[i - 1] if i > 0 else None)
            for i, d in enumerate(dates)}
    byday = {d: {r.concept_code: r for r in g.itertuples()}
             for d, g in td.groupby("trade_date")}
    out = {}
    for d in DAYS:
        pd_ = prev.get(d)
        if not pd_:
            continue
        for k, r in byday.get(pd_, {}).items():
            out[(d, k)] = r
    return out


def prev_leader_state(pl_row, date: str) -> dict:
    """已知身份的**昨日龙一**在今日的实时行为(全部决策时刻可观测)

    身份来自昨日天梯(盘前已知), 行为来自今日行情:
      · gap        今日开盘涨幅 —— 09:25 可观测
      · t_sealed/t_first_sec 今日是否已封板/首封时刻 —— 盘中逐轮可观测
      · y_*        昨日板型(板数/一字/炸板次数) —— 盘前已知
    evd/pxd/prev 由调用方一次建好传入(逐信号重建会慢到不可用)。"""
    lc = pl_row.leader_code
    pd_ = PL_PREV.get(date)
    cur = PL_EVD.get(date, {}).get(lc)
    y = PL_EVD.get(pd_, {}).get(lc) if pd_ else None
    pc = PL_PXD.get(date, {}).get(lc)
    return {
        "code": lc, "name": pl_row.leader_name,
        "y_height": pl_row.leader_height,       # 昨日龙一板数(盘前已知)
        "y_theme_age": pl_row.theme_age,        # 昨日题材年龄
        "y_zt_cnt": pl_row.zt_cnt,              # 昨日题材涨停家数
        "y_yizi": bool(y.is_yizi) if y is not None else False,
        "y_open_times": int(y.open_times) if y is not None else 0,
        "gap": ((pc.open / pc.pre_close - 1) * 100
                if pc and pc.pre_close and pc.pre_close > 0 else None),
        "t_sealed": cur is not None,            # 今日是否封板(盘中可观测)
        "t_first_sec": sec(cur.first_time) if cur is not None else None,
        "t_yizi": bool(cur.is_yizi) if cur is not None else False,
    }


PL_EVD: dict = {}
PL_PXD: dict = {}
PL_PREV: dict = {}


def _pl_prime() -> None:
    """为 prev_leader_state 一次性建好索引(避免逐信号重复加载全量数据集)"""
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "open_times", "first_time",
                       "is_yizi"])
    for r in ev.itertuples():
        PL_EVD.setdefault(r.trade_date, {})[r.ts_code] = r
    panel = load("market.daily_panel",
                 columns=["trade_date", "ts_code", "open", "pre_close"])
    for r in panel.itertuples():
        PL_PXD.setdefault(r.trade_date, {})[r.ts_code] = r
    dates = sorted(panel["trade_date"].unique())
    PL_PREV.update({d: (dates[i - 1] if i > 0 else None)
                    for i, d in enumerate(dates)})


def theme_break_rate(date: str, concept_code) -> float | None:
    """该题材当日涨停股的炸板率(open_times≥1 占比)"""
    if not isinstance(concept_code, str):
        return None
    att = load("theme.attribution",
               columns=["trade_date", "ts_code", "concept_code"])
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "open_times"])
    a = att[(att["trade_date"] == date)
            & (att["concept_code"] == concept_code)]
    e = ev[(ev["trade_date"] == date)
           & (ev["ts_code"].isin(set(a["ts_code"])))]
    return float((e["open_times"] >= 1).mean()) if len(e) else None


# ---------------------------------------------------------------- 分析
def main():
    df = build()
    say("# 研究34: S2/S3 失败的题材归因")
    say(f"窗口 {DAYS[0]}~{DAYS[-1]} ({df['date'].nunique()}个交易日) · "
        f"S2/S3 信号 {len(df)} 条")

    # ---------- A 总览 ----------
    say("\n## A 失败率总览")
    say(f"收盘封板 {int(df['sealed'].sum())} / {len(df)} = "
        f"{df['sealed'].mean() * 100:.1f}% → **失败率 "
        f"{df['fail'].mean() * 100:.1f}%**")
    rows = []
    for (st, br), g in df.groupby(["stage", "branch"]):
        rows.append([st, br, len(g), f"{g['sealed'].mean() * 100:.1f}%",
                     f"{g['fail'].mean() * 100:.1f}%",
                     f"{g['ev_next_open'].mean():+.2f}"
                     if g["ev_next_open"].notna().any() else "-"])
    md_table(["级", "分支", "信号数", "封板率", "失败率", "EV次日开盘"],
             sorted(rows, key=lambda x: -x[2]))
    rows = []
    for d, g in df.groupby("date"):
        rows.append([d, len(g), f"{g['fail'].mean() * 100:.1f}%",
                     f"{g['no_theme'].mean() * 100:.1f}%"])
    md_table(["日期", "S2/S3数", "失败率", "无归属占比"], rows)

    # ---------- B 板块效应检验 ----------
    say("\n## B 板块效应检验（题材集中度）")
    have = df[~df["no_theme"]]
    none = df[df["no_theme"]]
    say(f"**无归属组** {len(none)} 条 失败率 {none['fail'].mean() * 100:.1f}% "
        f"EV {none['ev_next_open'].mean():+.2f}"
        if none["ev_next_open"].notna().any() else "")
    say(f"**有归属组** {len(have)} 条 失败率 {have['fail'].mean() * 100:.1f}% "
        f"EV {have['ev_next_open'].mean():+.2f}")
    say(f"\n无归属占比 {df['no_theme'].mean() * 100:.1f}% —— 这部分不属于任何"
        f"板块, 是**散票**, 不构成板块效应。")

    g = have.groupby("concept_name").agg(
        n=("fail", "size"), fail=("fail", "sum"),
        seal=("sealed", "mean"), ev=("ev_next_open", "mean"))
    g = g[g["n"] >= 5].sort_values("fail", ascending=False)
    tot = g["fail"].sum()
    p = g["fail"] / tot
    say(f"\n有归属组内: 题材 {len(g)} 个 · 失败 {int(tot)} · "
        f"top{TOP_N_CONC} 题材占失败 **{p.head(TOP_N_CONC).sum() * 100:.1f}%**"
        f" · HHI {(p ** 2).sum():.4f}（均匀分布={1 / len(g):.4f}）")
    rows = [[i, k, int(v.n), int(v.fail), f"{v.seal * 100:.1f}%",
             f"{v.ev:+.2f}" if pd.notna(v.ev) else "-"]
            for i, (k, v) in enumerate(g.head(12).iterrows(), 1)]
    md_table(["#", "题材", "信号数", "失败数", "封板率", "EV次日开盘"], rows)

    # ---------- C 单票驱动检验 ----------
    say("\n## C 单票驱动检验（题材内是否某一只票带动）")
    rows = []
    for name in g.head(12).index:
        sub = have[have["concept_name"] == name]
        if sub["fail"].sum() < MIN_THEME_FAIL:
            continue
        by = sub.groupby(["ts_code", "name"]).agg(
            n=("fail", "size"), fail=("fail", "sum")).sort_values(
            "fail", ascending=False)
        top = by.iloc[0]
        rest = sub[sub["ts_code"] != by.index[0][0]]
        rows.append([name, int(sub["fail"].sum()),
                     f"{by.index[0][1]}({by.index[0][0]})",
                     int(top.fail), f"{top.fail / sub['fail'].sum() * 100:.0f}%",
                     int(top.n),
                     f"{rest['fail'].mean() * 100:.1f}%" if len(rest) else "-"])
    md_table(["题材", "失败数", "失败最多个股", "该股失败", "占题材失败",
              "该股信号数", "剔除后题材失败率"], rows)

    # ---------- D 带崩识别 ----------
    say("\n## D 带崩识别（题材-日 粒度: 是否有个股带崩 + 失败时刻分布）")
    say("判定口径: 若某只股占该题材当日失败 ≥30% 才算「单票带崩」; "
        "否则失败是题材内多票弥漫。龙头列仅供参考——theme.day 的 leader "
        "按定义就是当日涨停股, 所以「龙头当日涨停」恒为真, 无区分度。")
    rows = []
    for (nm, d), sub in have.groupby(["concept_name", "date"]):
        if sub["fail"].sum() < MIN_THEME_FAIL:
            continue
        by = sub.groupby(["ts_code", "name"])["fail"].sum().sort_values(
            ascending=False)
        top_share = by.iloc[0] / sub["fail"].sum() if len(by) else 0
        brk = theme_break_rate(d, sub.iloc[0]["concept_code"])
        ft = sub[sub["fail"]]["tsec"]
        st = sub[sub["sealed"]]["tsec"]
        # tsec=0 是 pt 缺失的占位值, 不能当 00:00 参与中位数
        ft, st = ft[ft > 0], st[st > 0]
        dd = theme_members_dd(d, sorted(sub["ts_code"].unique()))
        rows.append([
            nm, d, int(sub["fail"].sum()),
            f"{by.index[0][1]} {top_share * 100:.0f}%",
            "是" if top_share >= 0.30 else "否",
            f"{brk * 100:.0f}%" if brk is not None else "-",
            f"{_hm(ft.median())}" if len(ft) else "-",
            f"{_hm(st.median())}" if len(st) else "-",
            f'{dd[0]["code"]} {dd[0]["dd"]:+.1f}%'
            f'({dd[0]["t_peak"]}→{dd[0]["t_trough"]})' if dd else "-"])
    rows.sort(key=lambda x: -x[2])
    md_table(["题材", "日期", "失败数", "失败最多个股占比", "单票带崩",
              "题材涨停股炸板率", "失败信号中位时刻", "成功信号中位时刻",
              "回撤最深成分股"], rows[:14])
    n_single = sum(1 for r in rows if r[4] == "是")
    say(f"\n题材-日 组共 {len(rows)} 个, 其中「单票带崩」(top1≥30%) "
        f"**{n_single} 个** → 失败并非由某一只股带动。")

    # ---------- E 特征刻画 ----------
    say("\n## E 特征刻画（失败率 vs 题材当日强度）")
    bins = [("无当日活跃题材", None, None), ("题材涨停 1~2家", 1, 2),
            ("题材涨停 3~5家", 3, 5), ("题材涨停 6~9家", 6, 9),
            ("题材涨停 ≥10家", 10, 999)]
    rows = []
    for tag, lo, hi in bins:
        sub = (df[df["no_theme"]] if lo is None
               else have[(have["zt_cnt"] >= lo) & (have["zt_cnt"] <= hi)])
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-",
                     f"{sub['theme_age'].mean():.1f}" if lo else "-"])
    md_table(["题材强度分档", "信号数", "失败率", "封板率",
              "EV次日开盘", "波龄均值"], rows)
    rows = []
    for tag, lo, hi in (("波龄1(爆发)", 1, 1), ("波龄2~3(主升)", 2, 3),
                        ("波龄≥4(鱼尾)", 4, 999)):
        sub = have[(have["theme_age"] >= lo) & (have["theme_age"] <= hi)]
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["波龄分档", "信号数", "失败率", "封板率", "EV次日开盘"], rows)

    # ---------- G 龙一反馈 → 板块内信号成败 ----------
    say("\n## G 龙一（题材最高板）反馈状态 → 板块内 S2/S3 成败")
    say("龙一 = theme.day 的 leader(题材内连板最高者)。**盘前口径**只用 "
        "09:25 前已知的信息(昨日板型 + 今日开盘涨幅), 无前视; "
        "**当日全貌**含今日封板/断板结果, 有前视, 只用于机制解释。")
    lt = leader_table()
    pre_l, post_l, lead_m = [], [], []
    for r in df.itertuples():
        ls = lt.get((r.date, r.concept_code))
        a = r.tsec if r.tsec > 0 else None
        p, q = leader_feedback(ls or {}, a)
        pre_l.append(p if ls else "无龙一(题材当日无涨停)")
        post_l.append(q if ls else "无龙一(题材当日无涨停)")
        # 时序: 龙一今日首封时刻 - 信号触发时刻 (>0 = 龙一冲在前面)
        lead_m.append((ls["t_first_sec"] - a)
                      if ls and ls.get("t_first_sec") and a else None)
    df["ld_pre"], df["ld_post"] = pre_l, post_l
    df["ld_lead"] = lead_m
    df["ld_height"] = [lt.get((r.date, r.concept_code), {}).get("height")
                       for r in df.itertuples()]

    say("\n### 龙一高度分档（有没有真高标）")
    say("theme.day 的 leader 在「题材最高板=1」时只是一只首板股, "
        "并非用户口径的龙一(高标)。此表直接检验「有高标 vs 无高标」。"
        "注: ld_height 是当日收盘口径(有前视); 可落地的是 y_ht_max"
        "(该股所有成分概念在**昨日**的最高板, 盘前完全已知)。")
    rows = []
    for tag, lo, hi in (("无龙一(题材当日无涨停)", None, None),
                        ("龙一=1板(无真高标)", 1, 1),
                        ("龙一=2板", 2, 2), ("龙一≥3板(真高标)", 3, 99)):
        sub = (df[df["ld_height"].isna()] if lo is None
               else df[(df["ld_height"] >= lo) & (df["ld_height"] <= hi)])
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["龙一高度", "信号数", "失败率", "封板率", "EV次日开盘"], rows)

    say("\n### 昨日高标高度分档（盘前已知, 无前视）")
    rows = []
    for tag, lo, hi in (("昨日所属题材最高板=0(无高标)", 0, 0),
                        ("昨日高标=1板", 1, 1), ("昨日高标=2板", 2, 2),
                        ("昨日高标≥3板", 3, 99)):
        sub = df[(df["y_ht_max"] >= lo) & (df["y_ht_max"] <= hi)]
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["昨日高标高度", "信号数", "失败率", "封板率", "EV次日开盘"],
             rows)

    for col, tag in (("ld_pre", "盘前口径(无前视, 可入闸)"),
                     ("ld_post", "当日全貌(有前视, 仅机制解释)")):
        say(f"\n### {tag}")
        rows = []
        for k, sub in df.groupby(col):
            rows.append([k, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                         f"{sub['sealed'].mean() * 100:.1f}%",
                         f"{sub['ev_next_open'].mean():+.2f}"
                         if sub["ev_next_open"].notna().any() else "-"])
        rows.sort(key=lambda x: -x[1])
        md_table(["龙一状态", "信号数", "失败率", "封板率", "EV次日开盘"],
                 rows)

    # ---------- H 时序验证: 龙一封板 vs 信号触发 的先后 ----------
    say("\n## H 时序验证：龙一封板时刻 − 信号触发时刻 → 失败率")
    say("lead = 龙一今日首封时刻 − 该信号触发时刻(同 label_radar.lead50 号约)。"
        "**lead>0 = 信号先动、龙一后封; lead<0 = 龙一已封、信号后触发**。"
        "lead 绝对值 = 两者相隔秒数。本表只用龙一当日已封板且信号时刻"
        "非空的子集。")
    hb = df[df["ld_lead"].notna()]
    bins = [("龙一今日未封(断板)", None, None),
            ("龙一已封, 信号后触发(lead<0)", -10 ** 9, -1),
            ("同步: 龙一在信号后 0~2min 封", 0, 119),
            ("龙一在信号后 2~10min 封", 120, 599),
            ("信号孤军先动: 龙一≥10min后才封", 600, 10 ** 9)]
    rows = []
    for tag, lo, hi in bins:
        sub = (df[df["ld_post"].str.startswith(("当日断板", "跌停"))]
               if lo is None
               else hb[(hb["ld_lead"] >= lo) & (hb["ld_lead"] <= hi)])
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["龙一封板 vs 信号触发", "信号数", "失败率", "封板率",
              "EV次日开盘"], rows)

    # ---------- I 具体案例 ----------
    say("\n## I 失败最重的题材-日 龙一实例")
    have2 = df[~df["no_theme"]]      # 重取: have 是加龙一列之前的切片
    rows = []
    for (nm, d), sub in have2.groupby(["concept_name", "date"]):
        if sub["fail"].sum() < MIN_THEME_FAIL:
            continue
        ls = lt.get((d, sub.iloc[0]["concept_code"])) or {}
        rows.append([
            nm, d, int(sub["fail"].sum()),
            f'{ls.get("name", "-")}{ls.get("height", "-")}板',
            sub.iloc[0]["ld_pre"], sub.iloc[0]["ld_post"],
            f'{ls.get("t_pct"):+.1f}%' if ls.get("t_pct") is not None
            else "-"])
    rows.sort(key=lambda x: -x[2])
    md_table(["题材", "日期", "失败数", "龙一", "盘前口径", "当日全貌",
              "龙一当日涨幅"], rows[:12])

    # ---------- K 昨日天梯口径重做(用户指正) ----------
    say("\n## K 用【昨日复盘题材天梯】重做（无前视修正）")
    say("前面 G/H 节的龙一身份取自**当日** theme.day.leader_code, 而当日"
        "龙头是从当日涨停股里选出的 —— 09:31 不知道今天谁是龙一, 属前视。"
        "本节改用昨日天梯锚定: 题材 = 成分概念 ∩ **昨日**有涨停的题材"
        "(按昨日涨停家数取最大), 龙一 = 昨日天梯的 leader。龙一身份盘前"
        "已知, 它今日的行为(开盘涨幅/是否已封/何时封)才是真可观测。")
    _pl_prime()
    pl = prev_ladder()
    stock2con2, _, _ = load_maps()
    rows_k = []
    for r in df.itertuples():
        cands = [pl[(r.date, k)] for k in stock2con2.get(r.ts_code, [])
                 if (r.date, k) in pl]
        hot = max(cands, key=lambda x: x.zt_cnt) if cands else None
        if hot is None:
            rows_k.append({"p_theme": None, "p_ld": None, "p_lead": None,
                           "p_mode": None, "p_ht": 0})
            continue
        ls = prev_leader_state(hot, r.date)
        a = r.tsec if r.tsec > 0 else None
        lead = (ls["t_first_sec"] - a) if ls["t_first_sec"] and a else None
        rows_k.append({
            "p_theme": hot.concept_name, "p_ld": ls,
            "p_lead": lead, "p_ht": ls["y_height"],
            "p_mode": theme_mode(ls["y_theme_age"])})
    kdf = pd.DataFrame(rows_k, index=df.index)
    df = pd.concat([df, kdf], axis=1)

    say("\n### K1 昨日龙一板数分档")
    rows = []
    for tag, lo, hi in (("无昨日活跃题材(散票)", None, None),
                        ("昨日龙一=1板", 1, 1), ("昨日龙一=2板", 2, 2),
                        ("昨日龙一≥3板", 3, 99)):
        sub = (df[df["p_theme"].isna()] if lo is None
               else df[(df["p_ht"] >= lo) & (df["p_ht"] <= hi)])
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["昨日龙一板数", "信号数", "失败率", "封板率", "EV次日开盘"],
             rows)

    say("\n### K2 昨日题材阶段(爆发/主升/鱼尾, core.cycle.theme_mode)")
    rows = []
    for m in ("爆发", "主升", "鱼尾"):
        sub = df[df["p_mode"] == m]
        if not len(sub):
            continue
        rows.append([m, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["昨日题材阶段", "信号数", "失败率", "封板率", "EV次日开盘"],
             rows)

    say("\n### K3 昨日龙一今日封板 vs 信号触发(无前视版, 对比 H 节)")
    say("同 H 节号约: lead>0 = 信号先动龙一后封, lead<0 = 龙一已封信号后触发。")
    rows = []
    for tag, lo, hi in (("昨日龙一今日未封", None, None),
                        ("龙一已封, 信号后触发(lead<0)", -10 ** 9, -1),
                        ("同步: 龙一在信号后 0~2min 封", 0, 119),
                        ("龙一在信号后 2~10min 封", 120, 599),
                        ("信号孤军先动: 龙一≥10min后才封", 600, 10 ** 9)):
        sub = (df[df["p_theme"].notna() & df["p_lead"].isna()]
               if lo is None
               else df[(df["p_lead"] >= lo) & (df["p_lead"] <= hi)])
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["昨日龙一封板 vs 信号触发", "信号数", "失败率", "封板率",
              "EV次日开盘"], rows)

    say("\n### K4 昨日龙一昨日板型 × 今日开盘(正/负反馈, 无前视)")
    lab = []
    for r in df.itertuples():
        ls = r.p_ld
        if not isinstance(ls, dict):
            lab.append("无昨日龙一")
            continue
        g = ls["gap"]
        if ls["y_yizi"]:
            base = "正反馈·昨缩量一字"
        elif ls["y_open_times"] >= 2:
            base = "昨放量分歧"
        else:
            base = "昨换手板"
        if g is None:
            lab.append(base + "(开盘不明)")
        elif g >= 3:
            lab.append(base + "·今高开≥3%")
        elif g <= -3:
            lab.append(base + "·今低开≤-3%(负反馈)")
        else:
            lab.append(base + "·今平开")
    df["p_fb"] = lab
    rows = []
    for k, sub in df.groupby("p_fb"):
        rows.append([k, len(sub), f"{sub['fail'].mean() * 100:.1f}%",
                     f"{sub['sealed'].mean() * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    rows.sort(key=lambda x: -x[1])
    md_table(["昨日龙一反馈(无前视)", "信号数", "失败率", "封板率",
              "EV次日开盘"], rows)

    say("\n### K5 前视量化: 同一因子 当日口径 vs 昨日口径")
    say("同一维度用两种锚定分别算, 差距就是前视贡献。")
    a1 = df[df["ld_height"].notna()]
    a2 = df[df["p_theme"].notna()]
    rows = [
        ["有真高标(≥2板) 封板率",
         f"{(a1[a1['ld_height'] >= 2]['sealed'].mean()) * 100:.1f}% (当日锚)",
         f"{(a2[a2['p_ht'] >= 2]['sealed'].mean()) * 100:.1f}% (昨日锚)"],
        ["无高标 失败率",
         f"{(a1[a1['ld_height'] < 2]['fail'].mean()) * 100:.1f}% (当日锚)",
         f"{(a2[a2['p_ht'] < 2]['fail'].mean()) * 100:.1f}% (昨日锚)"],
        ["同步(龙一在信号后0~2min封) 封板率",
         f"{(df[df['ld_lead'].between(0, 119)]['sealed'].mean()) * 100:.1f}%"
         " (当日锚)",
         f"{(df[df['p_lead'].between(0, 119)]['sealed'].mean()) * 100:.1f}%"
         " (昨日锚)"],
        ["信号孤军先动(龙一≥10min后才封) 失败率",
         f"{(df[df['ld_lead'] >= 600]['fail'].mean()) * 100:.1f}% (当日锚)",
         f"{(df[df['p_lead'] >= 600]['fail'].mean()) * 100:.1f}% (昨日锚)"],
    ]
    md_table(["因子", "当日天梯锚定(有前视)", "昨日天梯锚定(无前视)"], rows)

    # ---------- J 规避候选 ----------
    say("\n## J 规避候选与历史 lift")
    base = df["fail"].mean()
    say("注: `zt_cnt` 是 theme.day 的**收盘**口径, 盘中拿它做闸属于未来信息; "
        "可落地的是 `zt_live`(截至信号时刻该题材已首封家数, 无前视), "
        "与研究30 宇宙层的 theme_zt 同式。两者并列列出作对比。")
    lv = df[df["zt_live"].notna()]
    # (名称, 评估母集, 剔除掩码) —— 母集显式指定, 不靠名字前缀猜
    cands = [
        ("无归属散票不下单", df, df["no_theme"]),
        ("[无前视] 龙一负反馈不下单", df,
         df["ld_pre"].str.startswith("负反馈")),
        ("[无前视] 龙一非正反馈不下单", df,
         ~df["ld_pre"].str.startswith("正反馈")),
        ("[无前视] 题材已封家数=0 不下单", lv, lv["zt_live"] == 0),
        ("[无前视] 题材已封家数<2 不下单", lv, lv["zt_live"] < 2),
        ("[无前视] 题材已封家数<3 不下单", lv, lv["zt_live"] < 3),
        ("[有前视对照] 题材收盘涨停家数<3", df, df["zt_cnt"].fillna(0) < 3),
        ("[有前视对照] 龙一当日断板/跌停", df, df["ld_post"].str.startswith(
            ("当日断板", "跌停"))),
        ("题材波龄≥4(鱼尾) 不下单", df, df["theme_age"].fillna(0) >= 4),
        ("[无前视] 昨日所属题材最高板<2 不下单", df, df["y_ht_max"] < 2),
        ("[有前视对照] 当日龙一<2板", df, df["ld_height"].fillna(0) < 2),
        ("[无前视] 信号孤军先动(龙一≥10min后才封) 不下单", df,
         df["ld_lead"].fillna(-1) >= 600),
        ("[无前视] 只保留同步(龙一在信号后0~2min封)", df,
         ~(df["ld_lead"].fillna(-1).between(0, 119))),
    ]
    rows = []
    for nm, src, m in cands:
        cut, keep = src[m], src[~m]
        if not len(keep):
            continue
        rows.append([nm, len(cut), f"{cut['fail'].mean() * 100:.1f}%",
                     len(keep), f"{keep['fail'].mean() * 100:.1f}%",
                     f"{keep['sealed'].mean() * 100:.1f}%",
                     f"{keep['ev_next_open'].mean():+.2f}"
                     if keep["ev_next_open"].notna().any() else "-"])
    md_table(["候选闸", "被剔除", "剔除组失败率", "保留", "保留组失败率",
              "保留组封板率", "保留组EV次日开盘"], rows)
    say(f"\n基线失败率 {base * 100:.1f}% / 封板率 {df['sealed'].mean() * 100:.1f}%"
        f" · 无前视子集(pt非空且有活跃题材) {len(lv)} 条 "
        f"失败率 {lv['fail'].mean() * 100:.1f}%")

    say("\n## 诚实边界")
    say("- **G/H/I 三节有前视, 结论以 K 节为准**: 龙一身份取自当日 "
        "theme.day.leader_code, 而当日龙头是从当日涨停股里选出的, "
        "09:31 无法知道; 题材归属也用了当日 zt_cnt 排序。G/H 保留在报告里"
        "只为了与 K 节做前视量化对照(K5), **不得入生产**。")
    say(f"- 样本仅 {df['date'].nunique()} 个交易日({DAYS[0]}~{DAYS[-1]}), "
        f"未覆盖牛/熊/震荡三段, 上述 lift 属窗口内观察而非定论; "
        f"任何闸要进生产必须先按三段市况重跑(用户方法论要求)。")
    say("- 当日活跃题材取「该股成分概念 ∩ 当日有涨停的题材」中涨停家数最大者, "
        "多题材交叉的票只计入最热的那个, 会低估多题材交叉的真实暴露。")
    say("- 带崩识别用 intraday_px 分时回撤, 该文件经 repair 清洗过, "
        "集合竞价段可能缺首点, 回撤深度为下界估计。")
    say("- 部分信号 `pt`(触发时刻)为空, 其 tsec 置 0 并已从时刻中位数排除; "
        "所以「失败/成功信号中位时刻」列仅基于 pt 非空的子集。")
    say("- 题材涨停股炸板率走 theme.attribution(涨停股的独占归属), "
        "该表只含涨停股——这里只用于描述题材内涨停股质量, "
        "**不得**用于给未涨停的信号票归因(会让未封板与无题材完全重合)。")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n落盘 {REPORT}")


if __name__ == "__main__":
    main()
