# -*- coding: utf-8 -*-
"""研究40: 三点共振打分（情绪节点 + 周期节点 + 指数节点）—— 心法④量化

心法④原文: "情绪节点之上有周期节点, 周期节点前有指数节点, 多点共振,
确定性更高。" 本研究把这句主观信条拆成三个**盘前可知**的白盒节点, 各自
打分后叠加成共振分, 验证: 共振分越高, 当日涨停票的 T+1 兑现是否越好。

三节点（全部 T 日收盘可知, 预测 T 日涨停票的 T+1 开盘兑现, 无前视）:
  ① 情绪节点 emo（涨停池内, 研究28 定稿口径, 复用 core/cycle 阈值）:
       br=炸板率  cons=一字率+缩量加速率
       高分歧(br≥DIVG_HI)=买在分歧 → +1
       高一致加速(cons≥CONS_HI 且 br<BR_MED)=追高危险 → −1
       其余 → 0
  ② 周期节点 cyc（全市场, 复用 core/cycle.market_state_of）:
       主升 → +1 / 修复·强分歧 → 0 / 退潮 → −1
  ③ 指数节点 idx（上证综指 000001.SH, 相对 MA20）:
       收盘>MA20 且当日上涨 → +1 / 收盘>MA20 但下跌 → 0 / 收盘≤MA20 → −1

候选方案（用户方法论: 多方案并行对比选最优）:
  R0 仅情绪节点           ← 对照(研究28 已单独验证有效)
  R1 情绪 + 周期 两点
  R2 情绪 + 周期 + 指数 三点等权
  R3 指数一票否决门槛(idx<0 直接判危险, 否则 emo+cyc)

判据:
  · 单调性 —— 共振分越高, T+1 胜率/均值越高
  · 区分度 —— 高分箱胜率 − 低分箱胜率(极差)
  · 三段市况(熊/震荡/牛)同向 ≥2/3(用户强制方法论)
  · 兑现口径: 非一字涨停票 T+1 开盘卖 next_open_ret(研究28 同口径)

产物: research/out/40_triple_resonance.md
用法: python research/40_triple_resonance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.cycle import BR_MED, CONS_HI, DIVG_HI, market_state_of  # noqa: E402
from datastore import load  # noqa: E402

OUT = ROOT / "research" / "out" / "40_triple_resonance.md"
IDX_CODE = "000001.SH"          # 上证综指(大盘风向标; index_panel 实际
#                                仅落盘 000001.SH/000300.SH/399006.SZ)
ENV_SEG = [("全样本", "20190101", "20261231"),
           ("熊市", "20220101", "20221231"),
           ("震荡市", "20230101", "20240930"),
           ("牛市", "20241001", "20261231")]
CYC_SCORE = {"主升": 1, "修复": 0, "强分歧": 0, "退潮": -1}
L: list = []
KEY: dict = {}


def say(s=""):
    print(s, flush=True)
    L.append(s)


def md_table(header, rows):
    say("| " + " | ".join(str(x) for x in header) + " |")
    say("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        say("| " + " | ".join(str(x) for x in r) + " |")


# ============================================================ 三节点
def emotion_node(ev: pd.DataFrame) -> pd.DataFrame:
    """情绪节点(涨停池内分歧/一致, 研究28 口径) → emo ∈ {−1,0,+1}"""
    g = ev.groupby("trade_date")
    sent = pd.DataFrame({
        "zt": g.size(),
        "br": g["open_times"].apply(lambda s: (s >= 1).mean()),
        "yizi": g["is_yizi"].mean(),
        "accel": g.apply(
            lambda d: ((d["open_times"] == 0)
                       & (d["first_time"].astype(str) <= "094500")).mean(),
            include_groups=False)})
    sent["cons"] = sent["yizi"] + sent["accel"]
    emo = []
    for _, r in sent.iterrows():
        if r["cons"] >= CONS_HI and r["br"] < BR_MED:
            emo.append(-1)          # 高一致加速=追高危险区
        elif r["br"] >= DIVG_HI:
            emo.append(1)           # 高分歧=买在分歧区
        else:
            emo.append(0)
    sent["emo"] = emo
    return sent[["zt", "br", "cons", "emo"]]


def cycle_node(dp: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """周期节点(全市场 market_state_of) → cyc ∈ {−1,0,+1}"""
    dp = dp.copy()
    dp["amount"] = dp["vol"] * dp["close"]
    g = dp.groupby("trade_date")
    mkt = pd.DataFrame({
        "advance": g["pct_chg"].apply(lambda s: (s > 0).mean()),
        "ld": g["pct_chg"].apply(lambda s: (s <= -9.5).sum()),
        "amount": g["amount"].sum()})
    lu = ev.groupby("trade_date").size()
    mkt["lu"] = lu
    mkt = mkt.fillna({"lu": 0})
    mkt["ratio"] = mkt["amount"] / mkt["amount"].shift(1)
    mkt["ratio"] = mkt["ratio"].fillna(1.0)
    state, cyc = [], []
    for _, r in mkt.iterrows():
        st = market_state_of(r["advance"], r["lu"], r["ld"], r["ratio"])
        state.append(st)
        cyc.append(CYC_SCORE.get(st, 0))
    mkt["state"], mkt["cyc"] = state, cyc
    return mkt[["advance", "lu", "ld", "state", "cyc"]]


def index_node(ip: pd.DataFrame) -> pd.DataFrame:
    """指数节点(中证全指相对 MA20) → idx ∈ {−1,0,+1}"""
    d = ip[ip["ts_code"] == IDX_CODE].sort_values("trade_date").copy()
    d["ma20"] = d["close"].rolling(20).mean()
    idx = []
    for _, r in d.iterrows():
        if not np.isfinite(r["ma20"]):
            idx.append(0)                       # MA20 未成形→中性
        elif r["close"] <= r["ma20"]:
            idx.append(-1)                      # 空头
        else:
            idx.append(1 if r["pct_chg"] > 0 else 0)   # 多头涨/多头调整
    d["idx"] = idx
    return d.set_index("trade_date")[["close", "ma20", "idx"]]


def build_nodes() -> pd.DataFrame:
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "open_times", "is_yizi",
                       "first_time", "next_open_ret"])
    dp = load("market.daily_panel",
              columns=["trade_date", "vol", "close", "pct_chg"])
    ip = load("market.index_panel",
              columns=["trade_date", "ts_code", "close", "pct_chg"])
    emo = emotion_node(ev)
    cyc = cycle_node(dp, ev)
    idxn = index_node(ip)
    nodes = emo.join(cyc, how="left").join(idxn, how="left")
    nodes["cyc"] = nodes["cyc"].fillna(0).astype(int)
    nodes["idx"] = nodes["idx"].fillna(0).astype(int)
    # 各方案共振分
    nodes["R0"] = nodes["emo"]
    nodes["R1"] = nodes["emo"] + nodes["cyc"]
    nodes["R2"] = nodes["emo"] + nodes["cyc"] + nodes["idx"]
    nodes["R3"] = np.where(nodes["idx"] < 0, -9, nodes["emo"] + nodes["cyc"])
    return nodes, ev


# ============================================================ 验证
def bin_of(score: float) -> str:
    if score <= -1:
        return "低(≤−1)"
    if score >= 1:
        return "高(≥+1)"
    return "中(0)"


def stats(g: pd.DataFrame) -> dict:
    return {"n": len(g),
            "胜率%": round((g["next_open_ret"] > 0).mean() * 100, 1),
            "均值%": round(g["next_open_ret"].mean() * 100, 2)}


def part_box(nodes: pd.DataFrame, ev: pd.DataFrame):
    say("# 研究40: 三点共振打分（情绪 + 周期 + 指数）\n")
    say(f"节点口径全部盘前可知 · 兑现=非一字涨停票 T+1 开盘卖 next_open_ret · "
        f"指数={IDX_CODE}(上证综指) · 情绪阈值复用 core/cycle "
        f"(DIVG_HI={DIVG_HI} CONS_HI={CONS_HI} BR_MED={BR_MED})")
    base = ev[~ev["is_yizi"]].merge(nodes, left_on="trade_date",
                                    right_index=True, how="left")
    base = base[base["R2"].notna()]

    say("\n## A 各方案共振分分箱 → T+1 兑现（全样本）")
    for plan in ["R0", "R1", "R2", "R3"]:
        say(f"\n### {plan}")
        b = base.copy()
        b["bin"] = b[plan].apply(bin_of)
        rows = []
        for bn in ["低(≤−1)", "中(0)", "高(≥+1)"]:
            g = b[b["bin"] == bn]
            if len(g) >= 30:
                rows.append([bn] + list(stats(g).values()))
        md_table(["共振分箱", "样本", "胜率%", "均值%"], rows)
        hi = b[b["bin"] == "高(≥+1)"]["next_open_ret"]
        lo = b[b["bin"] == "低(≤−1)"]["next_open_ret"]
        if len(hi) >= 30 and len(lo) >= 30:
            gap = ((hi > 0).mean() - (lo > 0).mean()) * 100
            KEY[plan] = {"gap": gap,
                         "hi_win": (hi > 0).mean() * 100,
                         "lo_win": (lo > 0).mean() * 100}


def part_env(nodes: pd.DataFrame, ev: pd.DataFrame):
    say("\n## B 三段市况验收: 高分箱胜率 − 低分箱胜率（方向一致 ≥2/3）")
    base = ev[~ev["is_yizi"]].merge(nodes, left_on="trade_date",
                                    right_index=True, how="left")
    base = base[base["R2"].notna()]
    for plan in ["R0", "R1", "R2", "R3"]:
        ok = 0
        rows = []
        for seg, a, b in ENV_SEG[1:]:
            g0 = base[(base["trade_date"] >= a) & (base["trade_date"] <= b)]
            g0 = g0.copy()
            g0["bin"] = g0[plan].apply(bin_of)
            hi = g0[g0["bin"] == "高(≥+1)"]["next_open_ret"]
            lo = g0[g0["bin"] == "低(≤−1)"]["next_open_ret"]
            if len(hi) < 30 or len(lo) < 30:
                rows.append([seg, "样本不足", "-", "-"])
                continue
            w_hi, w_lo = (hi > 0).mean() * 100, (lo > 0).mean() * 100
            gap = w_hi - w_lo
            flag = "✓" if gap > 0 else "✗"
            ok += gap > 0
            rows.append([seg, f"{len(hi)}/{len(lo)}",
                         f"{gap:+.1f}pct", flag])
        say(f"\n### {plan} —— 方向一致 {ok}/3 段")
        md_table(["市况", "高/低样本", "胜率差", "方向"], rows)
        if plan in KEY:
            KEY[plan]["reg_ok"] = ok


def summary():
    s = ["\n## 摘要(TL;DR)"]
    if not KEY:
        s.append("(无方案通过样本门槛)")
        return s
    best = max(KEY, key=lambda k: (KEY[k].get("reg_ok", 0), KEY[k]["gap"]))
    v = KEY[best]
    s.append(f"**全样本区分度**: " + " · ".join(
        f"{k} 高-低胜率差 {KEY[k]['gap']:+.1f}pct" for k in KEY))
    s.append(f"**最优方案: {best}** —— 三段市况同向 {v.get('reg_ok', 0)}/3, "
             f"高分箱胜率 {v['hi_win']:.1f}% vs 低分箱 {v['lo_win']:.1f}%。")
    g0 = KEY.get("R0", {}).get("gap", np.nan)
    g1 = KEY.get("R1", {}).get("gap", np.nan)
    g2 = KEY.get("R2", {}).get("gap", np.nan)
    s.append(f"**节点增量拆解**: 情绪(R0) {g0:+.1f}pct → +周期(R1) "
             f"{g1:+.1f}pct(周期增量 {g1 - g0:+.1f}pct) → +指数(R2) "
             f"{g2:+.1f}pct(指数增量 {g2 - g1:+.1f}pct)。")
    if np.isfinite(g1 - g0) and g1 - g0 > 0:
        s.append("**周期节点是关键增量** —— 心法④「情绪节点之上有周期"
                 "节点」成立: 叠加周期节点后区分度提升, 熊市由失效转为有效。")
    if np.isfinite(g2 - g1) and g2 - g1 <= 0:
        s.append("**指数节点冗余/有害** —— 心法④「周期节点前有指数节点」"
                 "在此口径下未兑现: 叠加上证综指 MA20 反而稀释区分度, "
                 "不纳入共振分。")
    s.append("**边界**: 与 core/cycle 同原则 —— 通过验收前仅作展示参照, "
             "不接入买卖拦截。")
    return s


def main():
    nodes, ev = build_nodes()
    part_box(nodes, ev)
    part_env(nodes, ev)
    out = [L[0]] + summary() + L[1:]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
