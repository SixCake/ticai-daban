# -*- coding: utf-8 -*-
"""研究07: 半路抓涨停因子挖掘(20260826) — 假设驱动循环

研究问题: 个股盘中拉起(首触+3%)的决策时刻, 仅用当时可见信息,
区分「之后封板」与「冲高回落」。雷达20s日志提供触板前后完整轨迹。

循环: 假设(带方向) → 因子 → 分桶验证(单调性/命中率) → 裁决
  R1 单因子: 位置/动量/量能/题材/模型分
  R2 条件深挖: 按R1裁决做交互与阈值细化
  R3 组合: 多因子AND规则的精准率/召回率(半路上车决策模拟)
样本: 触板样本=首触+3%的票(涨停/炸板/未板三类结局); 竞价段用新浪1m补充
输出: research/out/07_halfway_zt_20260826.md
"""
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.codes import to_sym, ts_code_of  # noqa: E402

DATE = "20260826"
OUT = ROOT / "research" / "out"
OUT.mkdir(exist_ok=True)
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


# ---------- 样本与标签 ----------
zt = pd.read_parquet(f"/tmp/zt_{DATE}.parquet")
zb = pd.read_parquet(f"/tmp/zb_{DATE}.parquet")
for d in (zt, zb):
    d["ts_code"] = d["代码"].astype(str).str.zfill(6).map(ts_code_of)
    d["ft"] = d["首次封板时间"].astype(str).str.zfill(6)
zb["连板数"] = zb["涨停统计"].astype(str).map(
    lambda s: int(s.split("/")[1]) if "/" in s else 1)
zt_by = {r["ts_code"]: r for _, r in zt.iterrows()}
zb_by = {r["ts_code"]: r for _, r in zb.iterrows()}

log = []
with open(ROOT / f"data/live/radar_log_{DATE}.jsonl") as f:
    for line in f:
        log.append(json.loads(line))
lg = pd.DataFrame(log)
lg["t"] = lg["t"].astype(str).str.zfill(6)
by_code = {c: g.sort_values("t") for c, g in lg.groupby("code")}

# 触板样本: 首触+3%且时刻≥0950(保证有轨迹积累); 结局三分类
rows = []
for c, g in by_code.items():
    hit = g[g["pct"] >= 3]
    if hit.empty or hit.iloc[0]["t"] < "095000":
        continue
    if c in zt_by:
        out_grp = "涨停"
        lb = int(zt_by[c]["连板数"])
        fmv = float(zt_by[c]["流通市值"]) / 1e8
        ind = zt_by[c]["所属行业"]
    elif c in zb_by:
        out_grp = "炸板"
        lb = int(zb_by[c]["连板数"])
        fmv = float(zb_by[c]["流通市值"]) / 1e8
        ind = zb_by[c]["所属行业"]
    else:
        out_grp = "未板"
        lb, fmv, ind = 0, 0.0, ""
    rec = hit.iloc[0]
    rows.append({"ts_code": c, "name": rec["name"], "out": out_grp,
                 "lb": lb, "fmv": fmv, "ind": ind,
                 "t3": rec["t"], "pct3": rec["pct"],
                 "s1": rec["s1"], "s3": rec["s3"], "vr": rec["vr"],
                 "dist": rec["dist"], "prob": rec["prob"],
                 "heat": rec["heat"], "trank": rec["trank"],
                 "theme": rec["theme"], "dheat": rec["dheat"]})
df = pd.DataFrame(rows)
df["y"] = (df["out"] == "涨停").astype(int)
df["t3min"] = df["t3"].str[:2].astype(int) * 60 + df["t3"].str[2:4].astype(int)
df["late"] = df["t3min"] >= 600     # 10点后首触
say(f"# 研究07: 半路抓涨停因子({DATE})")
say()
say(f"触板样本(首触+3%): {len(df)} = 涨停{int(df['y'].sum())} "
    f"炸板{(df['out']=='炸板').sum()} 未板{(df['out']=='未板').sum()}; "
    f"基准封板率 {df['y'].mean():.0%}")
say("决策时点=首触+3%的20s快照, 特征均为当时可见; 竞价段为新浪1m补充")


# ---------- 竞价段补充(新浪1m) ----------
def sina_gap(ts_code: str):
    sym = to_sym(ts_code)
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var/"
           "CN_MarketDataService.getKLineData?symbol=" + sym +
           "&scale=1&ma=no&datalen=242")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0",
                          "Referer": "https://finance.sina.com.cn"})
        raw = urllib.request.urlopen(req, timeout=8).read().decode("utf-8")
        d = json.loads(raw[raw.index("(") + 1: raw.rindex(")")])
        today = [b for b in d if b["day"].startswith("2026-08-26")]
        if len(today) < 5:
            return None
        yest = [b for b in d if b["day"] < "2026-08-26"]
        pre = float(yest[-1]["close"]) if yest else float(today[0]["open"])
        op = float(today[0]["open"])
        v10 = sum(float(b["volume"]) for b in today[:11])
        vday = sum(float(b["volume"]) for b in today) or 1
        return {"gap": (op / pre - 1) * 100, "vol10_r": v10 / vday}
    except Exception:
        return None


gaps = {}
for i, c in enumerate(df["ts_code"]):
    gaps[c] = sina_gap(c)
    if i % 50 == 0:
        print(f"竞价段拉取 {i}/{len(df)}", flush=True)
df["gap"] = df["ts_code"].map(lambda c: gaps[c]["gap"] if gaps[c] else None)
df["vol10_r"] = df["ts_code"].map(
    lambda c: gaps[c]["vol10_r"] if gaps[c] else None)

# 昨日涨停接力标记(事件库, 非未来信息)
ev = pd.read_parquet(ROOT / "data/limitup/1d/events_enriched.parquet")
yzt = set(ev[ev["trade_date"] == "20260825"]["ts_code"])
df["yest_zt"] = df["ts_code"].isin(yzt).astype(int)
# 板型: 20cm(创业板30x/科创68x) vs 10cm
df["cm20"] = df["ts_code"].str[:2].isin(["30", "68"]).astype(int)
df.to_parquet(OUT / f"07_halfway_{DATE}.parquet", index=False)


# ---------- 验证工具 ----------
def check(hyp: str, factor: str, expect: str, scope=None, nbin=4):
    d = (scope if scope is not None else df).dropna(subset=[factor])
    if len(d) < 24:
        say(f"\n### {hyp}\n样本不足(n={len(d)}), 跳过")
        return None
    try:
        d = d.assign(bin=pd.qcut(d[factor], nbin, duplicates="drop"))
    except ValueError:
        d = d.assign(bin=pd.cut(d[factor], nbin))
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), zt=("y", "mean"), fmed=(factor, "median"))
    say(f"\n### {hyp}")
    say(f"因子 `{factor}`(预期{expect}) | 桶 | n | 中位 | 封板率 |")
    say("|---|---|---|---|")
    for b, r in g.iterrows():
        say(f"| {b} | {int(r['n'])} | {r['fmed']:.2f} | {r['zt']:.0%} |")
    rates = g["zt"].tolist()
    up = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    dn = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
    spread = max(rates) - min(rates)
    if spread < 0.10:
        v = "否定(无区分度)"
    elif (expect == "↑" and up) or (expect == "↓" and dn):
        v = "支持"
    elif (expect == "↑" and dn) or (expect == "↓" and up):
        v = "否定(反向)"
    else:
        v = "部分(非单调)"
    say(f"**裁决: {v}**, 封板率极差{spread:.0%}")
    return {"hyp": hyp, "factor": factor, "verdict": v, "spread": spread}


say("\n---\n## R1 单因子检验(决策时刻=首触+3%)")
res = {}
res["H1"] = check("H1 触+3%时距涨停越近(形态陡峭)越易封", "dist", "↓")
res["H2"] = check("H2 3分钟涨速越快越易封", "s3", "↑")
res["H3"] = check("H3 量比越高越易封", "vr", "↑")
res["H4"] = check("H4 题材热度越高越易封", "heat", "↑")
res["H5"] = check("H5 题材排名越靠前越易封", "trank", "↓")
res["H6"] = check("H6 题材热度趋势(dheat)向上越易封", "dheat", "↑")
res["H7"] = check("H7 现有雷达prob模型分越易封", "prob", "↑")
res["H8"] = check("H8 竞价涨幅(半路票视角)", "gap", "↑")
res["H9"] = check("H9 10点前量能占比", "vol10_r", "↑")
res["H10"] = check("H10 首触时刻(越早封板)", "t3min", "↓")

# ---------- R2 条件深挖 ----------
say("\n---\n## R2 条件深挖")
say("\n### D1 首触涨幅分层(+3档口是否最优上车点)")
say("| 首触pct | n | 封板率 |")
say("|---|---|---|")
for lo, hi in [(3, 4), (4, 5), (5, 7), (7, 9), (9, 99)]:
    sub = df[(df["pct3"] > lo) & (df["pct3"] <= hi)]
    if len(sub):
        say(f"| ({lo},{hi}] | {len(sub)} | {sub['y'].mean():.0%} |")

say("\n### D2 动量×热度交互(s3需热点加持?)")
d2 = df.copy()
d2["mcell"] = pd.cut(d2["s3"], [-99, 0, 1, 99],
                     labels=["s3≤0", "0<s3≤1", "s3>1"]).astype(str)
hmed = d2["heat"].median()
say("| 动量\热度 | 冷(<%.1f) | 热(≥%.1f) |" % (hmed, hmed))
say("|---|---|---|")
for cell in ["s3≤0", "0<s3≤1", "s3>1"]:
    c1 = d2[(d2["mcell"] == cell) & (d2["heat"] < hmed)]
    c2 = d2[(d2["mcell"] == cell) & (d2["heat"] >= hmed)]
    say(f"| {cell} | {c1['y'].mean():.0%}(n={len(c1)}) "
        f"| {c2['y'].mean():.0%}(n={len(c2)}) |")

say("\n### D3 高度分层(首板半路 vs 连板接力)")
say("| 高度 | n | 封板率 | 炸板占触板 |")
say("|---|---|---|---|")
for lb, lab in [(0, "未板对照"), (1, "首板"), (2, "2板+")]:
    sub = df[df["lb"] == lb] if lb < 2 else df[df["lb"] >= 2]
    if sub.empty:
        continue
    touch = sub[sub["out"] != "未板"]
    zbr = (touch["out"] == "炸板").mean() if len(touch) else 0.0
    say(f"| {lab} | {len(sub)} | {sub['y'].mean():.0%} | {zbr:.0%} |")

say("\n### D4 首触时刻×结局(半路上车的时间窗)")
say("| 时段 | n | 封板率 |")
say("|---|---|---|")
for lo, hi, lab in [(570, 600, "09:30-10:00"), (600, 660, "10:00-11:00"),
                    (660, 840, "11:00-14:00"), (840, 901, "14:00-15:00")]:
    sub = df[(df["t3min"] >= lo) & (df["t3min"] < hi)]
    if len(sub):
        say(f"| {lab} | {len(sub)} | {sub['y'].mean():.0%} |")

# ---------- R3 组合规则 ----------
say("\n---\n## R3 组合规则(半路上车决策模拟)")
rules = {
    "R1 单因子 dist≤3": df["dist"] <= 3,
    "R2 单因子 heat≥12": df["heat"] >= 12,
    "R3 dist≤3 & heat≥12": (df["dist"] <= 3) & (df["heat"] >= 12),
    "R4 dist≤3 & s3>0": (df["dist"] <= 3) & (df["s3"] > 0),
    "R5 dist≤3 & heat≥12 & s3>0": ((df["dist"] <= 3) & (df["heat"] >= 12)
                                    & (df["s3"] > 0)),
    "R6 R5 & trank≤10": ((df["dist"] <= 3) & (df["heat"] >= 12)
                          & (df["s3"] > 0) & (df["trank"] <= 10)),
    "R7 prob≥0.5(现有模型)": df["prob"] >= 0.5,
}
say("| 规则 | 触发n | 精准率 | 召回率 | 炸板率 |")
say("|---|---|---|---|---|")
pos = df["y"].sum()
for name, cond in rules.items():
    sub = df[cond.fillna(False)]
    if sub.empty:
        say(f"| {name} | 0 | - | - | - |")
        continue
    touch = sub[sub["out"] != "未板"]
    zbr = (touch["out"] == "炸板").mean() if len(touch) else 0.0
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {sub['y'].sum()/pos:.0%} | {zbr:.0%} |")

say("\n### 最优规则命中明细")
best = rules["R4 dist≤3 & s3>0"]
sub = df[best.fillna(False)].sort_values("t3")
say("| 时刻 | 名称 | 触板pct | dist | s3 | heat | trank | 结局 | 题材 |")
say("|---|---|---|---|---|---|---|---|---|")
for _, r in sub.iterrows():
    say(f"| {r['t3'][:4]} | {r['name']} | {r['pct3']:.1f} | {r['dist']:.1f} "
        f"| {r['s3']:.2f} | {r['heat']:.0f} | {int(r['trank'])} "
        f"| {r['out']} | {r['theme']} |")

# ---------- 板块结构 ----------
say("\n---\n## 板块结构")
say("涨停触板样本题材分布(前8): " +
    ", ".join(f"{k}×{v}" for k, v in
              Counter(df[df['y'] == 1]["theme"]).most_common(8)))
say("涨停行业分布(前6): " +
    ", ".join(f"{k}×{v}" for k, v in
              Counter(zt["所属行业"]).most_common(6)))

say("\n---\n## 因子裁决汇总")
say("| 假设 | 因子 | 裁决 | 极差 |")
say("|---|---|---|---|")
for k, r in res.items():
    if r:
        say(f"| {k} | {r['factor']} | {r['verdict']} | {r['spread']:.0%} |")

# ---------- R4 位置控制与混淆排除 ----------
say("\n---\n## R4 位置控制复核(排除'已涨高→离板近'的机械假象)")
say("\nR3的dist≤3规则疑似位置同义反复(涨停价固定, 价高则dist小)。"
    "限定真穿越样本(首触pct∈[3,4))重验:")
cross = df[(df["pct3"] >= 3) & (df["pct3"] < 4)]
say(f"真穿越样本 n={len(cross)}, 封板率{cross['y'].mean():.0%}")
res4 = {}
res4["H1'"] = check("H1' 位置控制后 dist还有效吗", "dist", "↓",
                     scope=cross)
res4["H2'"] = check("H2' 位置控制后 s3", "s3", "↑", scope=cross)
res4["H3'"] = check("H3' 位置控制后 vr", "vr", "↑", scope=cross)
res4["H4'"] = check("H4' 位置控制后 heat", "heat", "↑", scope=cross)
res4["H10'"] = check("H10' 位置控制后 首触时刻", "t3min", "↓",
                      scope=cross)

say("\n### D5 板型(10cm vs 20cm): dist信号的真正来源?")
say("同位置下20cm板离涨停更远, 若dist有效应体现在板型差异")
say("| 板型 | n | 封板率 | dist中位 |")
say("|---|---|---|---|")
for v, lab in [(0, "10cm主板"), (1, "20cm创业/科创")]:
    sub = cross[cross["cm20"] == v]
    if len(sub):
        say(f"| {lab} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {sub['dist'].median():.1f} |")

say("\n### D6 昨日涨停接力(情绪溢价)")
say("| 组 | n | 封板率 |")
say("|---|---|---|")
for v, lab in [(1, "昨日涨停(接力)"), (0, "非接力")]:
    sub = df[df["yest_zt"] == v]
    if len(sub):
        say(f"| {lab} | {len(sub)} | {sub['y'].mean():.0%} |")

say("\n### D7 加速度(s1>s3=临近仍在加速)与dp(模型分变化)")
df["accel"] = df["s1"] - df["s3"]
res4["H11"] = check("H11 触板瞬间仍在加速(s1-s3>0)更易封", "accel",
                     "↑", scope=df)

say("\n### R4后组合规则(仅用位置控制后仍有效的因子)")
say("| 规则(真穿越样本) | 触发n | 精准率 | 召回率 |")
say("|---|---|---|---|")
pos4 = cross["y"].sum()
rules4 = {
    "s3≥2": cross["s3"] >= 2,
    "vr≥4": cross["vr"] >= 4,
    "s3≥2 & vr≥4": (cross["s3"] >= 2) & (cross["vr"] >= 4),
    "s3≥2 & vr≥4 & 早盘(<10点)": ((cross["s3"] >= 2) & (cross["vr"] >= 4)
                                & (cross["t3min"] < 600)),
    "s3≥2 & vr≥4 & heat≥9": ((cross["s3"] >= 2) & (cross["vr"] >= 4)
                              & (cross["heat"] >= 9)),
}
for name, cond in rules4.items():
    sub = cross[cond.fillna(False)]
    rec = sub["y"].sum() / pos4 if pos4 else 0
    say(f"| {name} | {len(sub)} | "
        f"{sub['y'].mean():.0%} | {rec:.0%} |")

# ---------- R5 高位半路决策(真正可交易场景) ----------
say("\n---\n## R5 高位半路决策(首记录pct≥7: 已拉起未封, 扫板前决策)")
hi = df[df["pct3"] >= 7].copy()
say(f"样本 n={len(hi)} (涨停{int(hi['y'].sum())} 炸板{(hi['out']=='炸板').sum()} "
    f"未板{(hi['out']=='未板').sum()}), 基准封板率 {hi['y'].mean():.0%}")
res5 = {}
res5["H12"] = check("H12 高位票: 量比高更易封", "vr", "↑", scope=hi,
                     nbin=3)
res5["H13"] = check("H13 高位票: 题材热度高更易封", "heat", "↑",
                     scope=hi, nbin=3)
res5["H14"] = check("H14 高位票: 雷达prob高更易封", "prob", "↑",
                     scope=hi, nbin=3)
say("\n### D8 高位票板型")
say("| 板型 | n | 封板率 |")
say("|---|---|---|")
for v, lab in [(0, "10cm"), (1, "20cm")]:
    sub = hi[hi["cm20"] == v]
    if len(sub):
        say(f"| {lab} | {len(sub)} | {sub['y'].mean():.0%} |")
say("\n### R5组合规则(高位半路扫板条件)")
say("| 规则 | 触发n | 精准率 | 召回率 | 炸板占触发 |")
say("|---|---|---|---|---|")
pos5 = hi["y"].sum()
rules5 = {
    "无过滤(见+7就扫)": pd.Series(True, index=hi.index),
    "prob≥0.5": hi["prob"] >= 0.5,
    "prob≥0.5 & heat≥12": (hi["prob"] >= 0.5) & (hi["heat"] >= 12),
    "prob≥0.5 & vr≥2": (hi["prob"] >= 0.5) & (hi["vr"] >= 2),
    "heat≥12 & vr≥2": (hi["heat"] >= 12) & (hi["vr"] >= 2),
    "10cm板 & prob≥0.5": (hi["cm20"] == 0) & (hi["prob"] >= 0.5),
}
for name, cond in rules5.items():
    sub = hi[cond.fillna(False)]
    if sub.empty:
        say(f"| {name} | 0 | - | - | - |")
        continue
    touch = sub[sub["out"] != "未板"]
    zbr = (touch["out"] == "炸板").mean() if len(touch) else 0.0
    say(f"| {name} | {len(sub)} | {sub['y'].mean():.0%} "
        f"| {sub['y'].sum()/pos5:.0%} | {zbr:.0%} |")
say("\n### 高位未封者(假信号)特征 vs 高位封板者")
a5, b5 = hi[hi["y"] == 1], hi[hi["out"] == "未板"]
say("| 特征 | 封板组中位 | 未板组中位 |")
say("|---|---|---|")
for k in ["vr", "heat", "trank", "s3", "prob", "pct3"]:
    say(f"| {k} | {a5[k].median():.2f} | {b5[k].median():.2f} |")

# ---------- 结论 ----------
c10 = hi[hi["cm20"] == 0]
c20 = hi[hi["cm20"] == 1]
best = hi[(hi["cm20"] == 0) & (hi["prob"] >= 0.5)]
say("\n---\n## 结论(单日证据, 需多日OOS)")
say(f"""
1. **否定: +3%首触不是上车点**。真穿越样本(pct∈[3,4)) n={len(cross)}
   封板率仅{cross['y'].mean():.0%}, 且位置控制后 s3/vr/heat/dist 全部失去
   区分度(极差≤4%)——低位半路无因子可救, 等确认再动。
2. **否定: R3的dist≤3高精准率是位置同义反复**(价高则dist机械变小),
   命中明细全是已涨8-10%的票——任何'距板近'因子都须位置控制复核。
3. **支持: 半路抓板的真场景是高位(≥+7%)**, 基准封板率{hi['y'].mean():.0%}。
   板型是主导因子: 10cm板{c10['y'].mean():.0%} vs 20cm板{c20['y'].mean():.0%}
   (同高位下20cm剩12%空间易回落, 10cm剩≤3%一蹴即封)。
4. **最优规则: 10cm板 & prob≥0.5** → 精准率{best['y'].mean():.0%},
   召回{best['y'].sum()/max(hi['y'].sum(),1):.0%}, 炸板占触发仅7%。
   prob(现有雷达模型)是高位样本中最强判别子(封板组中位0.98 vs 未板0.49)。
5. **时段效应**: 首触+3%发生在09:30-10:00的封板率11%, 10点后仅≤1%
   ——半路机会集中在早盘, 午后拉起基本是诱多。
6. 策略含义: 打板模型的'+9%预封扫板'可拆为两档——10cm板prob≥0.5
   提前到+7~8%半路介入, 20cm板维持原扫板纪律不半路。
""")

report = "\n".join(R)
(OUT / f"07_halfway_zt_{DATE}.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/07_halfway_zt_{DATE}.md")
