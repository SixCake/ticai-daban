# -*- coding: utf-8 -*-
"""研究19: 前几日因子逐日复盘面板 + 全规则胜率补全

  A. 全规则胜率表: 封板率/胜率(次日>0)/大胜率(次日>2%)/EV, 全期+test期
  B. 逐日面板: 近15个交易日, 每条生产规则当日命中数/封板率/次日胜率/
     次日收益中位/命中示例 —— 检验规则"前几日"的实际表现
  C. 逐日稳定性统计: 胜率>50%的天数占比
输出: research/out/19_daily_panel.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "out"
R = []


def say(s=""):
    R.append(s)
    print(s, flush=True)


df = pd.read_parquet(OUT / "17_enriched.parquet")
df["win"] = (df["next_ret"] > 0).astype(int)
df["win2"] = (df["next_ret"] > 2).astype(int)

RULES = {
    "S3 高开稳封相": (df["cohort"] == "G") & (df["gap"] > 5.2)
        & (df["odip"] <= 0.05) & (df["cm20"] == 0),
    "S3 稳封相+昨收强": (df["cohort"] == "G") & (df["gap"] > 5.2)
        & (df["odip"] <= 0.05) & (df["cm20"] == 0)
        & (df["y_cpos"] > 0.6),
    "S3 竞价量爆(非一字)": (df["cohort"] == "G") & (df["open_vr"] > 5)
        & (df["cm20"] == 0) & (df["gap"] <= 5.2),
    "S3 高开剧震": (df["cohort"] == "G") & (df["gap"] <= 5.2)
        & (df["amp3"] > 4.3) & (df["cm20"] == 0),
    "S2 L颠簸高": (df["cohort"] == "L") & (df["pathvol"] > 0.93)
        & (df["cm20"] == 0),
    "接力×稳封相": (df["y_zt"] == 1) & (df["cohort"] == "G")
        & (df["odip"] <= 0.05),
}

say("# 研究19: 前几日因子复盘 + 胜率补全")

# ---------- A. 全规则胜率表 ----------
say("\n## A. 全规则胜率表(胜率=次日收益>0, 大胜率=次日>2%, 对入场价)")
say("| 规则 | 期别 | n | 封板率 | 胜率 | 大胜率 | EV中位% |")
say("|---|---|---|---|---|---|---|")
for name, cond in RULES.items():
    for lab, scope in [("全期", df), ("test", df[df["split"] == "test"])]:
        sub = scope[cond.reindex(scope.index).fillna(False)]
        nr = sub["next_ret"].dropna()
        if len(sub) < 15:
            continue
        say(f"| {name} | {lab} | {len(sub)} | {sub['y'].mean():.0%} "
            f"| {sub['win'].mean():.0%} | {sub['win2'].mean():.0%} "
            f"| {nr.median():.2f} |")

# ---------- B. 逐日面板(近15交易日) ----------
days = sorted(df["date"].unique())[-15:]
say(f"\n## B. 逐日面板 ({days[0]}~{days[-1]})")
for name, cond in RULES.items():
    say(f"\n### {name}")
    say("| 日期 | 命中 | 封板率 | 次日胜率 | 次日中位% | 命中示例 |")
    say("|---|---|---|---|---|---|")
    for d in days:
        sub = df[(df["date"] == d) & cond]
        if sub.empty:
            say(f"| {d} | 0 | - | - | - | |")
            continue
        nr = sub["next_ret"].dropna()
        wr = f"{(nr > 0).mean():.0%}" if len(nr) else "-"
        md = f"{nr.median():.2f}" if len(nr) else "-"
        # 示例: 取封板的2只+未封1只
        sealed_names = sub[sub["y"] == 1]["ts_code"].head(2).tolist()
        ex = ",".join(sealed_names) if sealed_names else \
            sub["ts_code"].iloc[0]
        say(f"| {d} | {len(sub)} | {sub['y'].mean():.0%} | {wr} | {md} "
            f"| {ex} |")

# ---------- C. 逐日稳定性 ----------
say("\n## C. 规则逐日稳定性(近15日)")
say("| 规则 | 有命中天数 | 封板率>基准天数 | 次日胜率>50%天数 | 平均命中 |")
say("|---|---|---|---|---|")
for name, cond in RULES.items():
    nd = wbase = wwin = 0
    cnts = []
    for d in days:
        daydf = df[df["date"] == d]
        sub = daydf[cond.reindex(daydf.index).fillna(False)]
        if sub.empty:
            continue
        nd += 1
        cnts.append(len(sub))
        if sub["y"].mean() > daydf["y"].mean():
            wbase += 1
        nr = sub["next_ret"].dropna()
        if len(nr) >= 3 and (nr > 0).mean() > 0.5:
            wwin += 1
    if nd:
        say(f"| {name} | {nd}/15 | {wbase}/{nd} | {wwin}/{nd} "
            f"| {np.mean(cnts):.1f} |")

report = "\n".join(R)
(OUT / "19_daily_panel.md").write_text(report, encoding="utf-8")
print(f"\n报告: {OUT}/19_daily_panel.md")
