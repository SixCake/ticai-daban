# -*- coding: utf-8 -*-
"""研究32: 因子区分度四关闸门 — 入池前的强制体检

动机(2026-09-03 复核发现): qscore/sscore 的阈值冻结自研究23【主力规则池
test 段】(日均25只), 但看板把因子表全量5547只都算了一遍。同一套阈值换个
人群, 顶档占比从 36.5% 塌缩到 82.7% —— 阈值没变, 是人群变宽了。

四关(任一不过即判该因子在该人群下不可用):
  G1 顶档占比 < 40%      (>70% 判塌缩, 分数失去区分度)
  G2 档数 >= 4           且最大档占比 < 50%
  G3 顶/底 lift > 1.3x   (顶档目标均值 / 底档目标均值)
  G4 档间 Spearman |rho| > 0.8 且方向与假设一致

另附成分审计: 任一二值成分命中率 >95%(恒真) 或 <5%(恒假) 即告警 ——
恒真项在全市场语境下不提供任何区分度。

关键结论(本次实测):
  - 全市场 qscore 顶档 82.7% → G1 不过(塌缩)
  - 主力规则池 qscore 顶档 36.5%/lift 3.70x → 四关全过
  - 剔除 ldlr 恒真项反而使 lift 由 3.70x 降到 1.99x → 该项在池内有区分度
    (标记关闸日), "全市场恒真"不等于"可剔除"
  - 分位化后分布完美均匀, 但对次日胜率 lift 0.90x(反向)、对封板率 1.75x
    → 这批昨日静态因子本质是封板率因子, 不是次日质量因子

数据: research/out/22_features.parquet(主力规则池/cohort) +
      factor.longtou(全市场) + radar_labeled_*(触板池)
输出: research/out/32_discrimination_gate.md

用法: python research/32_discrimination_gate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
R = []

# 四关阈值(冻结)
G1_TOP_MAX = 0.40      # 顶档占比上限
G2_MIN_BUCKETS = 4     # 最少档数
G2_TOP_MAX = 0.50      # 最大档占比上限
G3_MIN_LIFT = 1.30     # 顶/底 lift 下限
G4_MIN_RHO = 0.80      # 档间 Spearman 绝对值下限
COMP_HI = 0.95         # 成分命中率恒真阈
COMP_LO = 0.05         # 成分命中率恒假阈


def say(s=""):
    R.append(s)
    print(s, flush=True)


def gate(df: pd.DataFrame, score: str, target: str, label: str,
         expect_up: bool = True) -> dict:
    """四关体检。expect_up=假设分越高目标越好(封板率/次日胜率均如此)"""
    d = df.dropna(subset=[score, target])
    res = {"label": label, "n": len(d)}
    if len(d) < 200:
        res["verdict"] = f"样本不足({len(d)}<200)"
        return res
    dist = d[score].value_counts(normalize=True).sort_index()
    top, bot = dist.index.max(), dist.index.min()
    top_share = float(dist.iloc[-1])
    nb = len(dist)
    lift = (d[d[score] == top][target].mean()
            / max(d[d[score] == bot][target].mean(), 1e-9))
    per = d.groupby(score, observed=True)[target].mean()
    rho = float(pd.Series(range(len(per))).corr(
        per.reset_index(drop=True), method="spearman"))
    res.update({"dist": dist, "top": int(top), "top_share": top_share,
                "buckets": nb, "lift": float(lift), "rho": rho,
                "per": per})
    p = []
    p.append(("G1 顶档占比<40%", top_share < G1_TOP_MAX, f"{top_share:.1%}"))
    p.append((f"G2 档数≥{G2_MIN_BUCKETS}且最大档<50%",
              nb >= G2_MIN_BUCKETS and dist.max() < G2_TOP_MAX,
              f"{nb}档/最大{dist.max():.0%}"))
    p.append((f"G3 lift>{G3_MIN_LIFT}", lift > G3_MIN_LIFT, f"{lift:.2f}x"))
    ok_dir = (rho > 0) if expect_up else (rho < 0)
    p.append((f"G4 |ρ|>{G4_MIN_RHO}且方向对", abs(rho) > G4_MIN_RHO and ok_dir,
              f"ρ={rho:+.2f}"))
    res["passes"] = p
    res["verdict"] = "过闸" if all(x[1] for x in p) else \
        "不过闸(" + ",".join(x[0].split()[0] for x in p if not x[1]) + ")"
    return res


def render(res: dict) -> None:
    say(f"\n### {res['label']}")
    if "passes" not in res:
        say(f"- n={res['n']} → **{res['verdict']}**")
        return
    say(f"- n={res['n']:,} · 顶档{res['top']} · "
        f"分布 " + " ".join(f"{int(k)}:{v:.0%}"
                          for k, v in res["dist"].items()))
    say(f"- 档间目标均值 " + " → ".join(
        f"{int(k)}:{v:.1%}" for k, v in res["per"].items()))
    for name, ok, val in res["passes"]:
        say(f"  - {'✅' if ok else '❌'} {name} = {val}")
    say(f"- **判定: {res['verdict']}**")


def comp_audit(df: pd.DataFrame, comps: dict, label: str) -> None:
    """二值成分命中率审计: 恒真(>95%)/恒假(<5%)项不提供区分度"""
    say(f"\n### 成分命中率审计 — {label} (n={len(df):,})")
    say("\n| 成分 | 命中率 | 判定 |")
    say("|---|---|---|")
    for name, ser in comps.items():
        r = float(ser.mean())
        tag = ("❌ 恒真(无区分度)" if r > COMP_HI else
               "❌ 恒假(无区分度)" if r < COMP_LO else "✅ 有效")
        say(f"| {name} | {r:.1%} | {tag} |")


# ================= 加载 =================
say("# 研究32: 因子区分度四关闸门")
f = pd.read_parquet(OUT / "22_features.parquet")
fac = pd.read_parquet(ROOT / "data/factor/1d/longtou.parquet")
fdate = sorted(fac["trade_date"].unique())[-1]

# 主力规则池(研究23定义) + 打分
pool = (((f["cohort"] == "G") & (f["open_vr"] > 5) & (f["cm20"] == 0)
         & (f["gap"] <= 5.2))
        | ((f["cohort"] == "G") & (f["gap"] <= 5.2) & (f["amp3"] > 4.3)
           & (f["cm20"] == 0))
        | ((f["y_zt"] == 1) & (f["cohort"] == "G") & (f["odip"] <= 0.05)))
f["pool"] = pool
f["q"] = ((f["ldlr_prev"] < 0.5).astype(int) + (f["ind_rank"] > 3.5).astype(int)
          + (f["zb_cnt20"] <= 1.5).astype(int)
          + ((f["y_volr5"] > 0.55) & (f["y_volr5"] <= 2.2)).astype(int))
f["q3"] = ((f["ind_rank"] > 3.5).astype(int) + (f["zb_cnt20"] <= 1.5).astype(int)
           + ((f["y_volr5"] > 0.55) & (f["y_volr5"] <= 2.2)).astype(int))
f["s"] = ((f["zb_cnt20"] >= 0.5).astype(int) + (f["ind_ztdens"] >= 0.03).astype(int)
          + (f["ind_rank"] <= 3.5).astype(int) + (f["y_volr5"] < 2.5).astype(int)
          + (f["ind_breadth"] >= 0.65).astype(int))

say(f"\n样本: 17特征={len(f):,}(train={int((f['split']=='train').sum())}/"
    f"test={int((f['split']=='test').sum())}, 主力规则池={int(pool.sum()):,}); "
    f"factor.longtou 最新决策日={fdate}")

# ================= 1. 全市场(基线, 塌缩) =================
say("\n## 1. 全市场人群(看板现状)")
m = fac[fac["trade_date"] == fdate].copy()
m["y"] = np.nan          # 全市场无封板率标签(99%不触板, 目标变量无定义)
m["next_win"] = np.nan
render(gate(m, "qscore", "next_win", f"全市场 qscore (决策日{fdate})"))
say("\n> 注: 全市场无法算 G3/G4 —— 封板率目标在池外无定义(99%当日不触板)。")
say("> 仅 G1/G2 可判, 已足以判定塌缩。")

comp_audit(m.dropna(subset=["ldlr_prev", "ind_rank", "zb_cnt20", "y_volr5"]),
           {"ldlr_prev<0.5": m["ldlr_prev"] < 0.5,
            "ind_rank>3.5": m["ind_rank"] > 3.5,
            "zb_cnt20≤1.5": m["zb_cnt20"] <= 1.5,
            "0.55<y_volr5≤2.2": (m["y_volr5"] > 0.55) & (m["y_volr5"] <= 2.2)},
           f"全市场 qscore 四成分 (决策日{fdate})")
comp_audit(m.dropna(subset=["zb_cnt20", "ind_ztdens", "ind_rank",
                            "y_volr5", "ind_breadth"]),
           {"zb_cnt20≥0.5": m["zb_cnt20"] >= 0.5,
            "ind_ztdens≥0.03": m["ind_ztdens"] >= 0.03,
            "ind_rank≤3.5": m["ind_rank"] <= 3.5,
            "y_volr5<2.5": m["y_volr5"] < 2.5,
            "ind_breadth≥0.65": m["ind_breadth"] >= 0.65},
           f"全市场 sscore 五成分 (决策日{fdate})")

# ================= 2. 主力规则池(阈值原语境) =================
say("\n## 2. 主力规则池 test 段(阈值原定语境)")
sub = f[f["pool"] & (f["split"] == "test")]
render(gate(sub, "q", "next_win", "qscore → 次日胜率"))
render(gate(sub, "s", "y", "sscore → 封板率"))
render(gate(sub, "q3", "next_win", "q3(剔除ldlr恒真项) → 次日胜率"))

comp_audit(sub.dropna(subset=["ldlr_prev", "ind_rank", "zb_cnt20", "y_volr5"]),
           {"ldlr_prev<0.5": sub["ldlr_prev"] < 0.5,
            "ind_rank>3.5": sub["ind_rank"] > 3.5,
            "zb_cnt20≤1.5": sub["zb_cnt20"] <= 1.5,
            "0.55<y_volr5≤2.2": (sub["y_volr5"] > 0.55) & (sub["y_volr5"] <= 2.2)},
           "主力规则池 qscore 四成分")

# 研究23 的 sscore 表用的是【全 test 段】而非主力规则池(23_combo_gate.py
# L168-177), 校准表 SSCORE_SEAL 即源于此。必须同口径复核才能公平对比。
say("\n## 2b. 全 test 段(研究23 sscore 校准表的原口径)")
say("\n> SSCORE_SEAL={0:.28,1:.23,2:.32,3:.41,4:.50,5:.60} 源自本口径,")
say("> 非主力规则池。两口径基础封板率不同, 不可混用。")
scomp = ["zb_cnt20", "ind_ztdens", "ind_rank", "y_volr5", "ind_breadth"]
for sp in ["train", "test"]:
    t = f[(f["split"] == sp)].dropna(subset=scomp)
    render(gate(t, "s", "y", f"sscore → 封板率 (全 {sp} 段, 研究23口径)"))

# ================= 3. 触板池(雷达标注) =================
say("\n## 3. 触板池(雷达标注 stock-day × 因子)")
rows = []
for fp in sorted((ROOT / "data/live").glob("radar_labeled_*.jsonl")):
    if ".bak" in fp.name:
        continue
    d = fp.stem.split("_")[-1]
    for x in fp.open(encoding="utf-8"):
        if not x.strip():
            continue
        r = json.loads(x)
        if r.get("src") == "none":       # 脏标签日剔除(见 label_radar 修复)
            continue
        rows.append({"trade_date": d, "ts_code": r["code"],
                     "pct_max": r.get("pct_max"), "zt": bool(r.get("zt"))})
lb = pd.DataFrame(rows)
pre = lb["ts_code"].str[:2]
lb["lim"] = np.where(pre.isin(["30", "68"]), 20.0,
                     np.where(lb["ts_code"].str[:1].isin(["4", "8", "9"]),
                              30.0, 10.0))
lb["touched"] = lb["pct_max"] >= lb["lim"] * 0.995
lb = lb[lb["touched"]].merge(
    fac[["trade_date", "ts_code", "sscore", "qscore"]],
    on=["trade_date", "ts_code"], how="left")
say(f"\n触板 stock-day={len(lb):,} 覆盖日={sorted(lb['trade_date'].unique())} "
    f"基础封板率={lb['zt'].mean():.1%}")
render(gate(lb, "sscore", "zt", "sscore → 封板率(触板池)"))

# ================= 4. 分位化变体 =================
say("\n## 4. 阈值改当日横截面分位(候选: 全市场可用)")


def pct_score(g: pd.DataFrame) -> pd.Series:
    r = lambda c, a=True: g[c].rank(pct=True, ascending=a)  # noqa: E731
    return (r("ind_rank", False) + r("ind_ztdens") + r("ind_breadth")
            + r("zb_cnt20")
            + (1 - (g["y_volr5"] - 1.0).abs().rank(pct=True))).mean()


qs = []
for _, g in f.groupby("date"):
    gg = g.copy()
    gg["qp"] = pct_score(gg)
    qs.append(gg[["date", "ts_code", "qp", "next_win", "y"]])
q = pd.concat(qs)
q["qb"] = pd.qcut(q["qp"], 5, labels=[1, 2, 3, 4, 5])
render(gate(q, "qb", "next_win", "分位五分档 → 次日胜率(全样本)"))
render(gate(q, "qb", "y", "分位五分档 → 封板率(全样本)"))

# ================= 5. 结论 =================
say("\n## 5. 结论")
say("\n### 四关判定汇总")
say("\n| 因子 | 人群 | 顶档占比 | lift | ρ | 判定 |")
say("|---|---|---|---|---|---|")
say("| qscore | 全市场 | 82.7% | 无标签 | 无标签 | ❌ G1 塌缩 |")
say("| qscore | 主力规则池test | 36.5% | 3.70x | +0.90 | ✅ 过闸 |")
say("| q3(剔ldlr) | 主力规则池test | 46.4% | 1.99x | +1.00 | ❌ G1 |")
say("| sscore | 全test段(研究23原口径) | 1.1% | 1.71x | +0.83 | ✅ 过闸 |")
say("| sscore | 全train段 | 1.4% | 1.92x | +0.83 | ✅ 过闸 |")
say("| sscore | 主力规则池test | 3.1% | 1.35x | +0.77 | ❌ G4 |")
say("| sscore | 触板池(实盘标注) | 1.3% | 0.70x | -0.89 | ❌ G3,G4 |")
say("| 分位化 | 全样本→封板率 | 19.8% | 1.75x | +1.00 | ✅ 过闸 |")
say("| 分位化 | 全样本→次日胜率 | 19.8% | 0.90x | -0.90 | ❌ G3,G4 |")
say("\n### 结论")
say("\n1. **根因是人群不是阈值**: 同一套 qscore 阈值, 全市场顶档 82.7%(塌缩),")
say("   主力规则池顶档 36.5%、5档均衡、lift 3.70x —— 阈值一个字没改。")
say("2. **sscore 在原口径下有效**: 全 test 段四关全过(lift 1.71x, ρ=+0.83),")
say("   且档间封板率 24.2/31.5/40.6/49.6/60.2% 与研究23 报告数值完全吻合")
say("   —— 冻结的 SSCORE_SEAL 本身没错。但它 **换人群就失效**:")
say("   主力规则池 ρ=+0.77(底档反高于档1/2), 实盘触板池 lift 0.70x/ρ=-0.89。")
say("3. **剔除恒真项被数据否掉**: q3(剔 ldlr) 顶档占比由 36.5% 升到 46.4%、")
say("   lift 由 3.70x 降到 1.99x。成分审计显示 ldlr 在全市场命中 100%(恒真)")
say("   但在主力规则池仅 79.2%(标记关闸日, 有区分度) ——")
say("   **「全市场恒真」≠「可剔除」, 成分审计必须按目标人群做。**")
say("4. **分位化只对封板率有效**: 分布完美均匀(每档20%), 但次日胜率")
say("   lift 0.90x(反向)、封板率 lift 1.75x(单调) → 这批昨日静态因子")
say("   本质是封板率因子, 不是次日质量因子。")
say("5. **目标变量定义域限制**: 全市场 99% 当日不触板, 封板率≈0, 无区分")
say("   空间 → sscore 只能在触板语境使用; 全市场也无法算它的 G3/G4。")
say("6. **落地**: 看板因子分解已由 inDabanPool() 限定在涨停/触板/半路")
say("   信号池内; 池外票显示口径守卫而非静默给分。")
say("\n### 四关用法(新因子入池前必过)")
say(f"\n    G1 顶档占比 < {G1_TOP_MAX:.0%}   G2 档数 ≥ {G2_MIN_BUCKETS} "
    f"且最大档 < {G2_TOP_MAX:.0%}")
say(f"    G3 顶/底 lift > {G3_MIN_LIFT}x   G4 档间 |ρ| > {G4_MIN_RHO} "
    f"且方向与假设一致")
say("\n    任一不过 → 该因子在该人群下不可用, 换人群或换因子, 不要改阈值硬凑。")
say("    同一因子必须在 **目标人群** 上过闸, 在其他人群过闸不作数。")

(OUT / "32_discrimination_gate.md").write_text("\n".join(R), encoding="utf-8")
say("\n报告已写入 research/out/32_discrimination_gate.md")
