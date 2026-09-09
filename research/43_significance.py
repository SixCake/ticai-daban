# -*- coding: utf-8 -*-
"""研究43: 显著性检验与多重比较校正 —— 推翻重来的第一步

**为何做**: 研究40/41/42 报告了一堆"因子有效""组合最优"的结论, 但
**从未做过显著性检验**。自查发现:
  研究42 的"唯一正期望 +0.307%" → t=1.24, 95%CI [-0.179%, +0.792%]
  **包含 0, 与零无统计差别**
  而且是在 4档×4阶段×3角色×4类型 = **192 个组合**里挑出来的最好一个
  → 典型 p-hacking, 找到1个正期望是必然结果不是发现

本研究把 40/41/42 的每个结论都算 t 值与置信区间, 并做 Bonferroni 校正,
只保留统计上站得住的。

**判据**:
  ① 单因子: 5档分位的**首末档期望差**做两样本 t 检验
  ② 组合: 组合期望 vs 0 做单样本 t 检验
  ③ 多重比较: Bonferroni 校正 —— 检验数 m, 阈值 |t| > z_{1-α/(2m)}
     192 个组合在 α=0.05 下阈值约 |t|>3.5(而非 1.96)
  ④ 效应量: 期望差的绝对值(不只看显著性, 也看是否有实用价值)

口径沿用研究40(全宇宙日线近似)。**注意**: 该口径的离场是"未封板持有到
收盘", 与定稿规则(P2 10:10未封板即卖)不同 —— 所以本研究的结论只回答
"哪些因子的相对优劣是真的", 不回答"实盘 EV 是多少"。后者需分钟数据。

产物: research/out/43_significance.md
用法: python research/43_significance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.exit_rules import pl_stats                        # noqa: E402
from core.shape import CONT, BOOL, GROUP                    # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
UNIV = Path(__file__).resolve().parent.parent / "data" / "factor" \
    / "fulluniv_panel.parquet"
DIMS = Path(__file__).resolve().parent.parent / "data" / "factor" \
    / "themedims_panel.parquet"

TARGETS = [3.0, 4.0]        # 只测最优的两档(减少检验数)
ALPHA = 0.05
say = print


def ttest_2samp(a, b) -> tuple:
    """两样本 t 检验 → (均值差, t值, p值, 差值的95%CI)"""
    a = np.asarray([x for x in a if x is not None and not np.isnan(x)])
    b = np.asarray([x for x in b if x is not None and not np.isnan(x)])
    if len(a) < 100 or len(b) < 100:
        return (None, None, None, None)
    diff = a.mean() - b.mean()
    t, p = st.ttest_ind(a, b, equal_var=False)      # Welch
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return (round(diff, 3), round(float(t), 2), round(float(p), 5),
            (round(ci[0], 3), round(ci[1], 3)))


def ttest_1samp(v) -> tuple:
    """单样本 t 检验(vs 0) → (均值, t值, p值, 95%CI)"""
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)])
    if len(v) < 100:
        return (None, None, None, None)
    t, p = st.ttest_1samp(v, 0.0)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return (round(v.mean(), 3), round(float(t), 2), round(float(p), 5),
            (round(v.mean() - 1.96 * se, 3), round(v.mean() + 1.96 * se, 3)))


def main():
    if not UNIV.exists():
        say(f"缺少 {UNIV}, 先跑 research/40_full_universe.py")
        return
    d = pd.read_parquet(UNIV)
    say(f"全宇宙 {len(d)} 条票·日")
    # 题材维度(研究42 产出, 若无则跳过题材部分)
    has_dims = DIMS.exists()
    if has_dims:
        dm = pd.read_parquet(DIMS)
        d = d.merge(dm[["trade_date", "ts_code", "theme_stage", "role",
                        "theme_type"]], on=["trade_date", "ts_code"],
                    how="left")
        say(f"已合并题材维度 {len(dm)} 条")
    else:
        say("无题材维度缓存, 只检验形态/量价因子")

    # ---- 统计检验数(用于 Bonferroni) ----
    n_fac_tests = len(CONT + BOOL) * len(TARGETS)
    n_combo_tests = 0
    if has_dims:
        n_combo_tests = (len(TARGETS) * 3 * 3 * 3)   # 档×阶段×角色×类型
    m = n_fac_tests + n_combo_tests
    # Bonferroni 阈值
    from scipy.stats import norm
    t_crit = norm.ppf(1 - ALPHA / (2 * m)) if m else 1.96
    say(f"检验总数 m={m} (因子{n_fac_tests} + 组合{n_combo_tests})")
    say(f"Bonferroni 阈值 |t| > {t_crit:.2f} (未校正为 1.96)")

    say("检验因子…")
    fac_res = []
    for tgt in TARGETS:
        t = int(tgt)
        ret, ok = f"e{t}_ret", f"e{t}_ok"
        base = d[d[ok] & d[ret].notna()]
        for fac in CONT + BOOL:
            sub = base[base[fac].notna()]
            if len(sub) < 5000:
                continue
            if fac in BOOL:
                g0 = sub[sub[fac] == 0][ret]
                g1 = sub[sub[fac] == 1][ret]
                lab0, lab1 = "0", "1"
            else:
                try:
                    sub = sub.assign(q=pd.qcut(sub[fac], 5, labels=False,
                                               duplicates="drop"))
                except Exception:
                    continue
                qs = sorted(sub["q"].dropna().unique())
                if len(qs) < 5:
                    continue
                g0 = sub[sub["q"] == qs[0]][ret]
                g1 = sub[sub["q"] == qs[-1]][ret]
                lab0, lab1 = f"档{qs[0]+1}", f"档{qs[-1]+1}"
            diff, tv, pv, ci = ttest_2samp(list(g1), list(g0))
            if tv is None:
                continue
            fac_res.append({
                "fac": fac, "grp": GROUP[fac], "tgt": t,
                "cmp": f"{lab1} - {lab0}",
                "n0": len(g0), "n1": len(g1),
                "diff": diff, "t": tv, "p": pv, "ci": ci,
                "sig_bonf": abs(tv) > t_crit,
                "sig_raw": abs(tv) > 1.96,
            })
    say(f"  因子检验 {len(fac_res)} 项")

    say("检验题材组合…")
    combo_res = []
    if has_dims:
        for tgt in TARGETS:
            t = int(tgt)
            ret, ok = f"e{t}_ret", f"e{t}_ok"
            for stg in ("爆发", "主升", "鱼尾"):
                for role in ("昨日龙头", "昨日题材内"):
                    for tt in ("可持续型", "中间型", "消息面型"):
                        sub = d[(d[ok] & d[ret].notna())
                                & (d["theme_stage"] == stg)
                                & (d["role"] == role)
                                & (d["theme_type"] == tt)]
                        mean, tv, pv, ci = ttest_1samp(list(sub[ret]))
                        if tv is None:
                            continue
                        combo_res.append({
                            "tgt": t, "stage": stg, "role": role,
                            "type": tt, "n": len(sub),
                            "mean": mean, "t": tv, "p": pv, "ci": ci,
                            "sig_bonf": abs(tv) > t_crit,
                            "sig_raw": abs(tv) > 1.96,
                            "positive": mean is not None and mean > 0,
                        })
        say(f"  组合检验 {len(combo_res)} 项")

    write_report(d, fac_res, combo_res, m, t_crit, has_dims)
    say(f"\n报告已写入 {OUT/'43_significance.md'}")


def write_report(d, fac_res, combo_res, m, t_crit, has_dims):
    L = []
    A = L.append
    A("# 研究43: 显著性检验与多重比较校正\n")
    A(f"- 全宇宙 {len(d)} 条票·日")
    A(f"- 检验总数 **m={m}**, Bonferroni 阈值 **|t| > {t_crit:.2f}**"
      f"(未校正为 1.96)")
    A("- 判据: Welch 两样本 t 检验(因子首末档) / 单样本 t 检验(组合 vs 0)\n")

    A("## 为何做\n")
    A("研究40/41/42 报告了大量\"因子有效\"\"组合最优\"的结论, 但**从未做"
      "显著性检验**。自查:\n")
    A("- 研究42 的\"唯一正期望 +0.307%\" → **t=1.24, 95%CI [-0.179%, "
      "+0.792%] 包含 0**, 与零无统计差别")
    A("- 而且是在 **192 个组合**里挑出的最好一个 → 典型 p-hacking\n")
    A("本研究把所有结论都算 t 值与 CI, 并做 Bonferroni 校正。\n")

    A("## 因子检验结果\n")
    A(f"阈值: |t| > {t_crit:.2f}(Bonferroni) / |t| > 1.96(未校正)\n")
    A("| 因子 | 组 | 档 | 对比 | n(末) | n(首) | 期望差(pp) | t | p | "
      "95%CI | Bonf | 未校正 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    fac_res.sort(key=lambda x: -abs(x["t"]))
    for r in fac_res:
        A(f"| `{r['fac']}` | {r['grp']} | {r['tgt']}% | {r['cmp']} "
          f"| {r['n1']} | {r['n0']} | **{r['diff']}** | {r['t']} | {r['p']} "
          f"| [{r['ci'][0]}, {r['ci'][1]}] "
          f"| {'**是**' if r['sig_bonf'] else '否'} "
          f"| {'是' if r['sig_raw'] else '否'} |")
    A("")
    nb = sum(1 for r in fac_res if r["sig_bonf"])
    nr = sum(1 for r in fac_res if r["sig_raw"])
    A(f"**{len(fac_res)} 项检验: Bonferroni 通过 {nb} 项, "
      f"未校正通过 {nr} 项。**\n")
    if nb:
        A("### 通过 Bonferroni 的因子\n")
        for r in fac_res:
            if r["sig_bonf"]:
                A(f"- `{r['fac']}`({r['grp']}) {r['tgt']}%档 {r['cmp']}: "
                  f"期望差 **{r['diff']}pp**, t={r['t']}, "
                  f"CI [{r['ci'][0]}, {r['ci'][1]}]")
        A("")
    else:
        A("### **无因子通过 Bonferroni 校正**\n")

    if has_dims and combo_res:
        A("## 题材组合检验结果\n")
        A("| 档 | 阶段 | 角色 | 题材类型 | n | 期望% | t | p | 95%CI | "
          "正期望 | Bonf |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        combo_res.sort(key=lambda x: -abs(x["t"]))
        for r in combo_res:
            A(f"| {r['tgt']}% | {r['stage']} | {r['role']} | {r['type']} "
              f"| {r['n']} | **{r['mean']}** | {r['t']} | {r['p']} "
              f"| [{r['ci'][0]}, {r['ci'][1]}] "
              f"| {'是' if r['positive'] else '否'} "
              f"| {'**是**' if r['sig_bonf'] else '否'} |")
        A("")
        pb = sum(1 for r in combo_res if r["sig_bonf"] and r["positive"])
        pos = sum(1 for r in combo_res if r["positive"])
        A(f"**{len(combo_res)} 个组合: 正期望 {pos} 个, "
          f"其中通过 Bonferroni **{pb}** 个。**\n")
        if not pb:
            A("→ **没有任何组合的正期望在统计上站得住**。研究42 报告的"
              "\"唯一正期望\"是 p-hacking 的产物。\n")

    A("## 结论\n")
    A("1. **多重比较是主要问题** —— 192 个组合里挑最好的 1 个, 未校正阈值"
      " 1.96 会大量误判; Bonferroni 校正后阈值提到 "
      f"{t_crit:.2f}")
    A("2. 只有通过 Bonferroni 的结论才可进入下一步")
    A("3. **本研究口径的局限**: 离场是\"未封板持有到收盘\", 与定稿规则"
      "(P2 10:10未封板即卖)不同; 宇宙含 10:30 后才达标的票(实盘不买)。"
      "所以本研究只回答\"哪些因子的**相对优劣**是真的\", "
      "不回答\"实盘 EV 是多少\" —— 后者需分钟数据")

    (OUT / "43_significance.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
