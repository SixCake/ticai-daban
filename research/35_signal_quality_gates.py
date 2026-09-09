# -*- coding: utf-8 -*-
"""研究35: S2/S3 信号质量闸 —— 只用「决策时刻可知」的信息

用户指正(研究34 的方法论错误): 研究34 的「同步共振」档需要知道龙一将在
何时封板, 而信号触发那一刻并不知道 → 那是事后统计标签, 不是决策因子。
本研究把约束收紧为: **每个特征的取值时刻必须 ≤ 信号触发时刻 T**。

特征分组(全部 ≤T 可知):
  A 个股轨迹(T时刻快照)  pct/pathvol/accel/r3/vr/tover/dist/mins_open
                         /gap_open/odip3/amp3  ← 复用研究30 已建数据集
  B 题材实时             theme_zt_live = 截至T该题材已封板家数
  C 昨日龙一实时         ld_sealed(截至T是否已封) / ld_age(已封多久)
                         / ld_pct(截至T涨幅) / ld_gap(今日开盘涨幅)
  D 盘前静态             y_ht_max(昨日所属题材最高板) / y_zt_cnt / 题材阶段
  E 市场实时             mkt_zt_live = 截至T全市场已封板家数

题材与龙一锚定在**昨日天梯**(复盘页口径, 09:15 已知), 不用当日天梯
(当日龙头从当日涨停股里选出, 盘中无法知道 → 前视)。

验证: walk-forward —— TRAIN 段选闸, TEST 段纯样本外复核。只报告
TEST 段仍成立的闸; 任何闸要进生产还须按牛/熊/震荡三段重跑。

产物: research/out/35_signal_quality_gates.md
用法: python research/35_signal_quality_gates.py
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
from core.theme_signal import (LD_NEG_GAP, T2_MIN_ZT, T3_MIN_HT,  # noqa: E402
                               T3_MIN_ZT)
from datastore import load  # noqa: E402

LIVE = DATA / "live"
OUT = ROOT / "research" / "out"
REPORT = OUT / "35_signal_quality_gates.md"
SIG_DS = OUT / "30_sig_dataset.parquet"      # 研究30 已建, 横截面特征≤T无未来

TRAIN = ["20260827", "20260828", "20260831"]
TEST = ["20260901", "20260902"]
MIN_KEEP_TEST = 60          # TEST 段保留样本下限(低于此不做结论)
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


# ---------------------------------------------------------------- 数据集
def build() -> pd.DataFrame:
    """S2/S3 信号 + 全部 ≤T 可知特征 + 收盘封板标签"""
    ds = pd.read_parquet(SIG_DS)
    df = ds[ds["stage"].isin(["S2", "S3"])].copy()
    df["date"] = df["date"].astype(str)

    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "first_time"])
    seal_sec: dict = {}
    for r in ev.itertuples():
        seal_sec.setdefault(r.trade_date, {})[r.ts_code] = sec(r.first_time)

    stock2con, _, _ = load_maps()
    con2stock = load_con2stock()

    td = load("theme.day")
    dates = sorted(td["trade_date"].unique())
    prev = {d: (dates[i - 1] if i > 0 else None) for i, d in enumerate(dates)}
    byday = {d: {r.concept_code: r for r in g.itertuples()}
             for d, g in td.groupby("trade_date")}

    panel = load("market.daily_panel",
                 columns=["trade_date", "ts_code", "open", "pre_close"])
    pxd: dict = {}
    for r in panel.itertuples():
        pxd.setdefault(r.trade_date, {})[r.ts_code] = r

    ipx: dict = {}
    for d in sorted(set(df["date"])):
        f = LIVE / f"intraday_px_{d}.json"
        if f.exists():
            ipx[d] = json.loads(f.read_text(encoding="utf-8"))

    rows = []
    for r in df.itertuples():
        d, c, t = r.date, r.ts_code, int(r.tsec)
        pd_ = prev.get(d)
        # ---- 题材/龙一锚定在昨日天梯(盘前已知) ----
        yday = byday.get(pd_, {}) if pd_ else {}
        cands = [yday[k] for k in stock2con.get(c, []) if k in yday]
        hot = max(cands, key=lambda x: x.zt_cnt) if cands else None
        out = {"no_theme": hot is None,
               # 昨日所属题材最高板: 只算昨日有涨停的那些题材
               "y_ht_max": max((yday[k].max_height
                                for k in stock2con.get(c, []) if k in yday),
                               default=0),
               "y_zt_cnt": hot.zt_cnt if hot else 0,
               "y_mode": theme_mode(hot.theme_age) if hot else None}
        # ---- B 题材实时: 截至T该题材已封板家数 ----
        ss = seal_sec.get(d, {})
        out["theme_zt_live"] = (
            sum(1 for m in con2stock.get(hot.concept_code, [])
                if 0 < ss.get(m, 0) <= t) if hot and t > 0 else 0)
        # ---- E 市场实时: 截至T全市场已封板家数 ----
        out["mkt_zt_live"] = sum(1 for v in ss.values() if 0 < v <= t) \
            if t > 0 else 0
        # ---- C 昨日龙一实时(截至T可观测) ----
        if hot is None:
            out.update({"ld_sealed": False, "ld_age": None,
                        "ld_pct": None, "ld_gap": None})
        else:
            lc = hot.leader_code
            fs = ss.get(lc, 0)
            sealed_now = bool(fs) and fs <= t          # 截至T已封板
            pc = pxd.get(d, {}).get(lc)
            gap = ((pc.open / pc.pre_close - 1) * 100
                   if pc and pc.pre_close and pc.pre_close > 0 else None)
            pts = (ipx.get(d, {}) or {}).get(lc) or []
            pct_now = None
            if pts and pc and pc.pre_close > 0 and t > 0:
                hhmmss = f"{t // 3600:02d}{t % 3600 // 60:02d}{t % 60:02d}"
                prior = [e for e in pts if str(e[0]).zfill(6) <= hhmmss
                         and e[1] and e[1] > 0]
                if prior:
                    pct_now = (prior[-1][1] / pc.pre_close - 1) * 100
            out.update({"ld_sealed": sealed_now,
                        "ld_age": (t - fs) if sealed_now else None,
                        "ld_pct": pct_now, "ld_gap": gap})
        rows.append(out)
    feat = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, feat], axis=1)
    df["seal"] = df["seal_close"].fillna(False).astype(bool)
    return df


# ---------------------------------------------------------------- 分析
BUCKETS = [
    # (标签, 列, 分箱函数)
    ("A 个股触发涨幅 pct", "pct", [2, 4, 6, 9]),
    ("A 距涨停 dist", "dist", [0.5, 1.5, 3.0]),
    ("A 路径波动 pathvol", "pathvol", [0.2, 0.5, 0.9]),
    ("A 开盘分钟 mins_open", "mins_open", [5, 15, 40]),
    ("A 开盘3min振幅 amp3", "amp3", [2, 4.3, 7]),
    ("B 题材截至T已封家数", "theme_zt_live", [1, 2, 4]),
    ("E 全市场截至T已封家数", "mkt_zt_live", [10, 30, 60]),
    ("C 昨日龙一截至T涨幅", "ld_pct", [-3, 0, 5, 9]),
    ("C 昨日龙一开盘涨幅", "ld_gap", [-3, 0, 3, 6]),
    ("D 昨日所属题材最高板", "y_ht_max", [1, 2, 3]),
]


def bucket_table(d: pd.DataFrame) -> None:
    for label, col, edges in BUCKETS:
        say(f"\n### {label}")
        rows = []
        s = d[d[col].notna()] if col not in ("theme_zt_live", "mkt_zt_live",
                                             "y_ht_max") else d
        lo = None
        for e in edges + [None]:
            if lo is None and e is not None:
                sub = s[s[col] < e]
                tag = f"<{e}"
            elif e is not None:
                sub = s[(s[col] >= lo) & (s[col] < e)]
                tag = f"[{lo},{e})"
            else:
                sub = s[s[col] >= lo]
                tag = f">={lo}"
            lo = e
            if not len(sub):
                continue
            rows.append([tag, len(sub), f"{sub['seal'].mean() * 100:.1f}%",
                         f"{(1 - sub['seal'].mean()) * 100:.1f}%"])
        md_table(["分档", "信号数", "封板率", "失败率"], rows)


def candidate_gates(d: pd.DataFrame) -> list:
    """候选闸: 全部只用 ≤T 可知特征。返回 (名称, 保留掩码)"""
    return [
        ("散票不做(无昨日活跃题材)", ~d["no_theme"]),
        ("昨日所属题材最高板≥2", d["y_ht_max"] >= 2),
        ("昨日龙一今低开≤-3% → 不做",
         ~(d["ld_gap"].fillna(0) <= -3)),
        ("要求昨日龙一截至T已封板(未封则不做)", d["ld_sealed"]),
        ("题材截至T已封家数≥1", d["theme_zt_live"] >= 1),
        ("题材截至T已封家数≥2", d["theme_zt_live"] >= 2),
        ("全市场截至T已封≥10", d["mkt_zt_live"] >= 10),
        ("个股触发涨幅≥4%", d["pct"] >= 4),
        ("个股触发涨幅≥6%", d["pct"] >= 6),
        ("距涨停≤1.5%", d["dist"].fillna(99) <= 1.5),
        ("开盘≤15min内触发", d["mins_open"] <= 15),
        ("昨日龙一截至T涨幅≥5%", d["ld_pct"].fillna(-99) >= 5),
    ]


def gate_report(d: pd.DataFrame, tag: str) -> None:
    say(f"\n### {tag}（基线封板率 {d['seal'].mean() * 100:.1f}% / "
        f"样本 {len(d)}）")
    base = d["seal"].mean()
    rows = []
    for nm, m in candidate_gates(d):
        keep = d[m]
        if len(keep) < MIN_KEEP_TEST and tag.startswith("TEST"):
            rows.append([nm, len(keep), "样本不足", "-", "-"])
            continue
        lift = keep["seal"].mean() / base if base > 0 else 0
        rows.append([nm, len(keep),
                     f"{keep['seal'].mean() * 100:.1f}%",
                     f"{lift:.2f}x",
                     f"{keep['ev_next_open'].mean():+.2f}"
                     if keep["ev_next_open"].notna().any() else "-"])
    rows.sort(key=lambda x: -(float(x[3].rstrip("x"))
                              if x[3] not in ("-", "样本不足") else 0))
    md_table(["闸(全部≤T可知)", "保留样本", "保留组封板率", "lift",
              "EV次日开盘"], rows)


def main():
    df = build()
    say("# 研究35: S2/S3 信号质量闸（只用决策时刻可知的信息）")
    say(f"窗口 {df['date'].min()}~{df['date'].max()} · S2/S3 {len(df)} 条 · "
        f"封板 {int(df['seal'].sum())} = {df['seal'].mean() * 100:.1f}%")
    say("\n约束: 每个特征取值时刻 ≤ 信号触发时刻 T。题材/龙一锚定在"
        "**昨日天梯**(09:15已知), 不用当日天梯(当日龙头盘中无法知道)。")
    say("研究34 的「同步共振」档因需要预知龙一何时封板, 不在本研究候选内。")

    say("\n## A 单因子分档（全窗口）")
    bucket_table(df)

    say("\n## B 昨日龙一截至T状态 × 结果（决策时刻可知）")
    rows = []
    for tag, m in (("无昨日活跃题材(散票)", df["no_theme"]),
                   ("龙一今低开≤-3%", df["ld_gap"].fillna(0) <= -3),
                   ("龙一截至T已封板", (~df["no_theme"]) & df["ld_sealed"]),
                   ("龙一截至T未封", (~df["no_theme"]) & ~df["ld_sealed"]),
                   ("龙一截至T涨幅≥9%(贴板)",
                    df["ld_pct"].fillna(-99) >= 9),
                   ("龙一截至T涨幅0~9%",
                    df["ld_pct"].fillna(-99).between(0, 9))):
        sub = df[m]
        if not len(sub):
            continue
        rows.append([tag, len(sub), f"{sub['seal'].mean() * 100:.1f}%",
                     f"{(1 - sub['seal'].mean()) * 100:.1f}%",
                     f"{sub['ev_next_open'].mean():+.2f}"
                     if sub["ev_next_open"].notna().any() else "-"])
    md_table(["昨日龙一截至T状态", "信号数", "封板率", "失败率",
              "EV次日开盘"], rows)

    say("\n## C walk-forward: TRAIN 选闸 → TEST 纯样本外")
    tr = df[df["date"].isin(TRAIN)]
    te = df[df["date"].isin(TEST)]
    say(f"TRAIN {sorted(tr['date'].unique())} {len(tr)}条 "
        f"封板率 {tr['seal'].mean() * 100:.1f}% | "
        f"TEST {sorted(te['date'].unique())} {len(te)}条 "
        f"封板率 {te['seal'].mean() * 100:.1f}%")
    gate_report(tr, "TRAIN 段")
    gate_report(te, "TEST 段(样本外)")

    say("\n## E T级 × S级 交叉表（T3+S3 是否最优）")
    say("用 core/theme_signal 的同一判据给每条信号打 T 标签。注意本研究"
        "的 theme_zt_live 用 events.first_time ≤ 信号时刻计数(离线权威口径), "
        "而生产 core/theme_signal 用盘中贴死涨停价实时计数——两者同为"
        "「截至T已封家数」, 盘中值会略低于收盘权威值。")

    def _t_of(r):
        if r.no_theme:
            return "T1"
        if r.ld_gap is not None and r.ld_gap <= LD_NEG_GAP:
            return "T1"
        if r.theme_zt_live >= T3_MIN_ZT and r.y_ht_max >= T3_MIN_HT:
            return "T3"
        if r.theme_zt_live >= T2_MIN_ZT:
            return "T2"
        return "T1"
    df["T"] = df.apply(_t_of, axis=1)

    say("\n### E1 T级 × S级(全窗口)")
    rows = []
    for T in ("T3", "T2", "T1"):
        for st in ("S2", "S3"):
            sub = df[(df["T"] == T) & (df["stage"] == st)]
            if not len(sub):
                continue
            rows.append([f"{T}+{st}", len(sub),
                         f"{sub['seal'].mean() * 100:.1f}%",
                         f"{(1 - sub['seal'].mean()) * 100:.1f}%",
                         f"{sub['ev_next_open'].mean():+.2f}"
                         if sub["ev_next_open"].notna().any() else "-"])
    rows.sort(key=lambda x: -float(x[2].rstrip("%")))
    md_table(["组合", "信号数", "封板率", "失败率", "EV次日开盘"], rows)

    say("\n### E2 T级 × S3分支（S3 内部差异极大, 必须拆开）")
    rows = []
    for T in ("T3", "T2", "T1"):
        for br, sub in df[(df["T"] == T) & (df["stage"] == "S3")].groupby(
                "branch"):
            if len(sub) < 10:
                continue
            rows.append([f"{T}+{br}", len(sub),
                         f"{sub['seal'].mean() * 100:.1f}%",
                         f"{sub['ev_next_open'].mean():+.2f}"
                         if sub["ev_next_open"].notna().any() else "-"])
    rows.sort(key=lambda x: -float(x[2].rstrip("%")))
    md_table(["组合", "信号数", "封板率", "EV次日开盘"], rows)

    say("\n### E3 T3+S3 的 TRAIN/TEST 分段（样本外复核）")
    # 重切: C 节的 tr/te 是加 T 列之前的切片, 不带 T 列
    tr2 = df[df["date"].isin(TRAIN)]
    te2 = df[df["date"].isin(TEST)]
    rows = []
    for tag, d in (("TRAIN", tr2), ("TEST", te2)):
        for combo, m in (("T3+S3", (d["T"] == "T3") & (d["stage"] == "S3")),
                         ("T3+S2", (d["T"] == "T3") & (d["stage"] == "S2")),
                         ("T3 全部", d["T"] == "T3"),
                         ("全样本基线", d["T"].notna())):
            sub = d[m]
            rows.append([tag, combo, len(sub),
                         f"{sub['seal'].mean() * 100:.1f}%" if len(sub)
                         else "-",
                         f"{sub['ev_next_open'].mean():+.2f}"
                         if len(sub) and sub["ev_next_open"].notna().any()
                         else "-"])
    md_table(["段", "组合", "信号数", "封板率", "EV次日开盘"], rows)

    say("\n## D 组合闸（TRAIN 选, TEST 验）")
    combos = [
        ("散票不做 + 昨日高标≥2",
         lambda d: (~d["no_theme"]) & (d["y_ht_max"] >= 2)),
        ("散票不做 + 龙一今低开≤-3%不做",
         lambda d: (~d["no_theme"]) & ~(d["ld_gap"].fillna(0) <= -3)),
        ("散票不做 + 龙一截至T已封",
         lambda d: (~d["no_theme"]) & d["ld_sealed"]),
        ("散票不做 + 高标≥2 + 龙一今低开不做",
         lambda d: (~d["no_theme"]) & (d["y_ht_max"] >= 2)
         & ~(d["ld_gap"].fillna(0) <= -3)),
        ("上述 + 题材截至T已封≥1",
         lambda d: (~d["no_theme"]) & (d["y_ht_max"] >= 2)
         & ~(d["ld_gap"].fillna(0) <= -3) & (d["theme_zt_live"] >= 1)),
        ("上述 + 个股触发涨幅≥4%",
         lambda d: (~d["no_theme"]) & (d["y_ht_max"] >= 2)
         & ~(d["ld_gap"].fillna(0) <= -3) & (d["theme_zt_live"] >= 1)
         & (d["pct"] >= 4)),
    ]
    rows = []
    for nm, fn in combos:
        ktr, kte = tr[fn(tr)], te[fn(te)]
        rows.append([nm, len(ktr),
                     f"{ktr['seal'].mean() * 100:.1f}%" if len(ktr) else "-",
                     len(kte),
                     f"{kte['seal'].mean() * 100:.1f}%"
                     if len(kte) >= MIN_KEEP_TEST else "样本不足",
                     f"{kte['ev_next_open'].mean():+.2f}"
                     if len(kte) and kte["ev_next_open"].notna().any()
                     else "-"])
    md_table(["组合闸", "TRAIN保留", "TRAIN封板率", "TEST保留",
              "TEST封板率(样本外)", "TEST EV"], rows)

    say("\n## 诚实边界")
    say(f"- 窗口仅 {df['date'].nunique()} 个交易日, TRAIN {len(TRAIN)}日 / "
        f"TEST {len(TEST)}日, 未覆盖牛/熊/震荡三段; 任何闸进生产前必须"
        f"按三段市况重跑(用户方法论要求)。")
    say("- 特征复用研究30 数据集(0827~0902), 故 0903/0904 未纳入; "
        "研究30 的横截面特征按≤120s最近邻回捞且不含未来, 口径安全。")
    say("- `tover/dist/prob/heat/trank` 非空率仅 63%(雷达横截面回捞未命中), "
        "涉及这些列的分档只基于该子集。")
    say("- 龙一身份锚定昨日天梯; 若昨日该题材无涨停则记为散票, "
        "与「当日有涨停但昨日无」的题材不同, 后者不在本口径内。")

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"\n落盘 {REPORT}")


if __name__ == "__main__":
    main()
