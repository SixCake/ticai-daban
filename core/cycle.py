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
题材阶段(爆发/主升/鱼尾): 研究39 定稿 —— 锚点是**波次**(空档≥5交易日
算新波), 不是日龄。旧口径 theme_mode(theme_age) 已废弃(方向反置且无区分度)。
两者均仅展示参照, 不接任何买卖拦截(与V5结构层影子同原则)。
"""

POS_BY_STAGE = {"春": 0.3, "夏": 0.8, "秋": 0.3, "冬": 0.0}  # 历史展示参照

# 题材阶段(研究39 定稿): 在场门槛与新波冷却期(交易日)
WAVE_MIN_ZT = 2         # 关联涨停家数≥2 才算题材在场
WAVE_COOLDOWN = 5       # 与上一在场日空档≥5交易日 → 新波(研究38 定档)
BURST_MIN_ZT = 3        # 【已不再用于阶段判定】旧版「真爆发」家数门槛,
                        # 研究39 用它提区分度, 但把首波弱起归为「主升」
                        # 语义上说不通(第1波第1天不可能是主升), 已回退。
                        # 仅研究39/40/41 脚本作为历史对照保留引用。
# 波次回看窗口(研究40 定档): 只数窗口内开启的波段, 不做全历史累计。
#   研究40 实测: 全历史累计口径在看板实际面对的近期样本里**方向反置**
#   (主升 32.8% 反而低于鱼尾 40.4%, 鱼尾档占近期样本 88%);
#   近1年与近2年四条硬条件均全过(全历史方向正确/三段市况3-3同向/
#   近期方向正确/三档最小占比≥10%), 近1年 Spearman -0.270 略强于
#   近2年 -0.259, 近期极差 35.3pp vs 36.6pp(差异<2.5pp 属持平);
#   取近1年是因为更贴合「当下这轮行情」(用户定);
#   近半年不可取: 鱼尾档占比降到7.4%, 三档退化成两档。
WAVE_LOOKBACK = 244     # ≈1年交易日

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


def segment_waves(positions: list[int], active: list[bool],
                  cooldown: int = WAVE_COOLDOWN,
                  lookback: int | None = WAVE_LOOKBACK) -> list:
    """波段切分（波次口径单一出处，研究38/39/40 验证口径）

    positions: 升序的交易日下标列表 —— 题材**有任何涨停**(关联家数≥1)
               的日子, 不是只取在场日;
    active:    对应位置是否「在场」(关联家数≥WAVE_MIN_ZT);
    cooldown:  与上一行空档≥cooldown 个交易日 → 新波;
    lookback:  回看交易日数 —— 波次 = 在 [T-lookback, T] 内开启的波段个数;
               None = 全历史累计(研究40 已证该口径在近期方向反置, 不用)。
               波段起点早于窗口 → 记 1(窗口内正在进行的第一波),
               避免窗口起点处的截断偏差。

    口径要点(研究38 实测): 新波只能由**在场日**开启, 但空档是相对
    「上一有任何涨停的日子」算的 —— 即题材彻底消失(连1家都没有)
    ≥cooldown 天后重新起量才算新一波。若改成只在在场日上算空档,
    波次会被推高到上百(实测最大102), 几乎全落入鱼尾档。
    cooldown=2/3 会把同一波的拖动误判成新波, 延续率梯度从
    Spearman -0.151 降到 -0.001/-0.053。

    返回: 与 positions 等长的波次列表。
    纯函数, 不读任何数据 —— core.theme_wave 调用。
    """
    # ① 先做累计切波, 找出所有波段起点
    starts, prev = [], None
    for p, a in zip(positions, active):
        if prev is None or (a and p - prev - 1 >= cooldown):
            starts.append(p)
        prev = p
    # ② 转成滚动窗口计数(数窗口内实际发生了几波)
    out = []
    for p in positions:
        lo = p - lookback if lookback is not None else -1
        n = sum(1 for s in starts if lo < s <= p)
        out.append(n if n >= 1 else 1)
    return out


def theme_stage(wave_no: int | None, zt: int) -> str:
    """题材阶段（定稿：**纯语义**，只看波次，不掺家数）

    爆发 = 第1波(wave_no≤1) —— 题材近1年内首次被炒
    主升 = 第2波(wave_no=2) —— 已验证过一轮, 正在第二轮
    鱼尾 = 第3波+(wave_no≥3) —— 反复炒过, 只剩补涨
    无   = 题材不在场(关联家数<WAVE_MIN_ZT) —— 不构成阶段

    ⚠ 为何不用家数调阶段（重要教训）:
      研究39 曾加「首波且家数≥BURST_MIN_ZT 才算爆发」的门槛, 把爆-鱼
      梯度从 +20.7pp 提到 +34.0pp。但代价是把「第1波第1天但只2家涨停」
      的题材标成「主升」—— 第1波第1天无论如何不可能是主升。
      **阶段是语义概念, 不得为了提区分度而扭曲语义（本末倒置）**。
      强度该单独展示(看板已有「关联」列 = zt_all), 不要混进阶段。
      回退后四条硬条件仍全过: 方向正确(全历史+近期) / 三段市况3-3同向 /
      最小档占比18.8%, 梯度 +20.7pp —— 语义正确与区分度并不冲突。

    为何锚点是波次而不是日龄（研究36/37/38 链路）:
      · 旧锚点 theme_age 度量的是「连续多少天抢到≥1只独占涨停股」,
        是持续性日龄而非阶段, 且方向被命名反置;
      · 波次与日龄正交（Spearman≈0）。
    为何波次用滚动回看窗口（研究40）:
      · 全历史累计在近期样本里方向反置, WAVE_LOOKBACK 定档 244。

    输入口径（全部盘前可知, 无前视）:
      wave_no = 空档≥WAVE_COOLDOWN 交易日算新波, 窗口内累计第几波
      zt      = 当日题材**关联**涨停家数（kpl 直标全 tag）
      两者均须来自归因自由活跃面板, **不得**用 theme.day 的独占
      zt_cnt/theme_age（研究36 已证伪）。

    ⚠ 使用边界: 仅用于题材追踪/复盘展示/延续性预测。
      不得接入买卖闸或仓位决策 —— 在封板率与 EV 两个交易目标上
      无区分度（研究36/37 已证）。
    """
    if wave_no is None or zt < WAVE_MIN_ZT:
        return "无"          # 不在场(关联涨停<WAVE_MIN_ZT家)不构成题材阶段
    if wave_no >= 3:
        return "鱼尾"
    if wave_no <= 1:
        return "爆发"
    return "主升"


def theme_mode(age: int) -> str:
    """⚠ 已废弃（仅旧看板兼容保留）—— 请改用 theme_stage(wave_no, zt)

    旧定义：波龄1=爆发 / 2-3=主升 / ≥4=鱼尾，输入是 theme.day.theme_age。
    研究36/37 已证伪: theme_age 是持续性日龄而非阶段，三档对封板率无
    区分度（极差0.5pp），且方向与命名相反（鱼尾档次日延续率最高）。
    研究39 已将同名三档重定义为波次口径，见 theme_stage()。
    """
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
