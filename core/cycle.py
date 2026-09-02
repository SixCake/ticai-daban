# -*- coding: utf-8 -*-
"""92科比式情绪周期四阶段还原（唯一出处, 启发式v0校准中）

春 = 低位试错期(老周期退潮结束, 小仓位试错新题材)
夏 = 主升阶段(主线明确, 龙头打出高度, 赚钱效应扩散)
秋 = 高位震荡期(龙头滞涨, 内部分化, 只宜轻仓博弈)
冬 = 主跌阶段(亏钱效应扩散, 高位批量杀跌, 空仓休息)

仓位跟随分歧/一致(研究28定稿): 高分歧0.8 / 中性0.5 / 高一致加速0.2,
阶段(春夏秋冬)仅展示参照 —— 仅页面展示, 不接入任何买卖拦截
(与V5结构层影子同原则)。

铁律还原: 高位(6板+)做龙头 / 低位(首板一进二)试错 / 中位(3-5板)
风险最大——高度分布条把中位单独标橙警示。
分歧/一致: 炸板≥1=分歧释放(买在分歧), 无炸板加速=一致(卖在一致)。
题材mode: 波龄1=爆发(切换窗口) / 2-3=主升 / ≥4=鱼尾(补涨性质, 不能格局)。
"""

POS_BY_STAGE = {"春": 0.3, "夏": 0.8, "秋": 0.3, "冬": 0.0}  # 历史展示参照

# 研究28全期三分位定稿: 仓位由分歧/一致维度驱动(阶段仅展示)
#   高分歧(br≥0.50)=买在分歧区 pos0.8; 高一致加速(cons≥0.33且br<中位)=
#   追高危险区 pos0.2; 中性 pos0.5。三环境验收: 震荡+9.9pct 牛+2.7pct
#   熊-0.3pct(持平) → 2/3方向一致过门槛
DIVG_HI = 0.50
CONS_HI = 0.33
BR_MED = 0.47

STAGE_NAME = {"春": "春 · 低位试错", "夏": "夏 · 主升",
              "秋": "秋 · 高位震荡", "冬": "冬 · 主跌"}


def stage_of(hist: list, cur: dict, prev_state=None) -> dict:
    """情绪四阶段判定 v0

    hist: 当日sentiment快照序列(升序), 项含 zt_count/exit_count/
          broken_rate/max_height; cur: 当前sentiment; prev_state:
          昨日市场级状态(主升/修复/强分歧/退潮, 仅入why展示)
    返回 {stage, pos, why}
    """
    why = []
    br = cur.get("broken_rate") or 0
    mh = cur.get("max_height") or 0
    zt_up = zt_dn = br_up = False
    tail = hist[-30:]                     # 近约30分钟趋势
    if len(tail) >= 6:
        half = len(tail) // 2
        z0 = sum(h.get("zt_count", 0) for h in tail[:half]) / half
        z1 = sum(h.get("zt_count", 0) for h in tail[half:]) / (len(tail) - half)
        b0 = sum(h.get("broken_rate", 0) for h in tail[:half]) / half
        b1 = sum(h.get("broken_rate", 0) for h in tail[half:]) / (len(tail) - half)
        zt_up = z1 > z0 + 1
        zt_dn = z1 < z0 - 1
        br_up = b1 > b0 + 0.05
    if br > 0.3 or (zt_dn and mh <= 3):
        stage = "冬"
        why.append(f"炸板率{br:.0%}过高" if br > 0.3
                   else "涨停家数递减且高度≤3")
    elif br < 0.2 and mh >= 5 and zt_up:
        stage = "夏"
        why.append(f"炸板率{br:.0%}低+最高板{mh}+家数扩张")
    elif mh >= 5:
        stage = "秋"
        why.append(f"最高板{mh}高位, " + ("炸板率抬升" if br_up
                   else f"炸板率{br:.0%}但家数无扩张" if br < 0.2
                   else "炸板率不低") + ", 震荡格局")
    else:
        stage = "春"
        why.append(f"高度{mh}不高+炸板率{br:.0%}可控, 新周期试错")
    if prev_state:
        why.append(f"昨日市场:{prev_state}")
    # 仓位v1(研究28): 分歧/一致驱动, 阶段仅展示参照
    cons = (cur.get("yizi_proxy") or 0) + (cur.get("accel") or 0)
    if cons >= CONS_HI and br < BR_MED:
        pos = 0.2
        why.append(f"高一致加速(cons{cons:.2f})=追高危险区, 降仓")
    elif br >= DIVG_HI:
        pos = 0.8
        why.append(f"高分歧(br{br:.0%})=买在分歧区")
    else:
        pos = 0.5
        why.append("分歧/一致中性")
    return {"stage": stage, "pos": pos, "why": why}


def height_dist(rows) -> dict:
    """连板高度分布: 低(1-2板)/中(3-5板风险区)/高(6板+); rows项含height"""
    low = mid = high = 0
    for r in rows:
        h = r["height"] if isinstance(r, dict) else r.height
        if h <= 2:
            low += 1
        elif h <= 5:
            mid += 1
        else:
            high += 1
    n = low + mid + high
    return {"low": low, "mid": mid, "high": high,
            "mid_ratio": round(mid / n, 3) if n else 0.0}


def divg_of(open_times: int) -> str:
    """炸板≥1=分歧释放(买在分歧); 无炸板=一致加速(卖在一致)"""
    return "分歧" if open_times >= 1 else "一致"


def theme_mode(age: int) -> str:
    """波龄1=爆发(切换窗口) / 2-3=主升 / ≥4=鱼尾(补涨性质)"""
    if age <= 1:
        return "爆发"
    if age <= 3:
        return "主升"
    return "鱼尾"


def market_state_of(advance: float, limit_up: int, limit_down: int,
                    amount_ratio: float) -> str:
    """市场级状态(同 research/longtou.build_market_context 口径, 仅展示参照)

    主升/修复/强分歧/退潮 四档
    """
    def clip(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))
    cycle = clip(45 * advance + 25 * clip(limit_up / 80)
                 + 15 * clip(1 - limit_down / 20)
                 + 15 * clip(amount_ratio / 1.10), 0, 100)
    if cycle >= 65 and limit_down <= 5:
        return "主升"
    if cycle >= 50:
        return "修复"
    if cycle >= 38:
        return "强分歧"
    return "退潮"


def load_prev_market_state(today_s: str):
    """上一交易日市场级状态: daily_panel+涨停事件现算(无现成落盘数据集)

    跌停家数以 daily_panel 跌幅≤-9.5% 近似; 失败降级 None
    """
    try:
        from datastore import load
        dp = load("market.daily_panel",
                  columns=["trade_date", "pct_chg", "vol", "close"])
        dp["amount"] = dp["vol"] * dp["close"]
        ev = load("limitup.events", columns=["trade_date"])
    except Exception:
        return None
    dates = sorted(dp["trade_date"].unique())
    prev = max((d for d in dates if d < today_s), default=None)
    if prev is None:
        return None
    pos = dates.index(prev)
    day = dp[dp["trade_date"] == prev]
    amt = float(day["amount"].sum())
    ratio = 1.0
    if pos > 0:
        amt2 = float(dp[dp["trade_date"] == dates[pos - 1]]["amount"].sum())
        if amt2 > 0:
            ratio = amt / amt2
    adv = float((day["pct_chg"] > 0).mean())
    ld = int((day["pct_chg"] <= -9.5).sum())
    lu = int((ev["trade_date"] == prev).sum()) if len(ev) else 0
    return market_state_of(adv, lu, ld, ratio)
