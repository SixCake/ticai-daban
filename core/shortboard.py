# -*- coding: utf-8 -*-
"""高位龙头短板 + 二波状态机（精简5态）— 唯一出处（研究29验证产物）

定位（方案 龙头短板二波追踪层 Phase 2）: 现有天梯/角色是"当日涨停快照",
前几日龙头今日未涨停即被完全丢弃。本模块提供一个 additive 的"龙头短板"
持久层口径: 近N日题材龙头∪市场高板即使今日未涨停也保留追踪, 用5态刻画
其高位状态。**仅作展示/风险标注, 不含任何买入语义**(研究29证伪: 整个高位
龙头短板cohort前向收益为负, 高位均值回归主导, 不能作打板买入信号)。

研究29关键结论(6.5年全历史, 牛/熊/震荡三段一致):
  相对风险排序 绕异动 > 断板观察 > 逻辑失效 > 二波候选 ≈ 放量分歧
  T+1开盘收益  绕异动-0.4% / 断板-0.9% / 失效-1.2% / 二波候选-1.5% / 放量分歧-1.7%
  → "放量换手分歧后冲高(二波候选)"历史平均是追高/出货陷阱, 非机会(证伪直觉);
    "缩量窄幅绕异动"才是最抗跌的高位状态。故5态按风险标注, 冲高态标警惕非看多。

Cohort(近N日题材龙头∪市场高板, 今日未涨停, 高位守卫):
  由调用方(review/poller)从 theme.day/events_enriched 构建候选并排除今日池,
  本模块提供 passes_guard 高位守卫 与 peak_features 峰谷派生。

5态判据(自上而下首个命中; 与 research/29.shortboard_state 同口径, 阈值冻结):
  逻辑失效 peak_drawdown≤-30% 或 (neg_streak≥2 & neg_deep) 或 (ldlr_prev≥0.5 & pct≤-5%)
  二波候选 pct≥3% 且 cpos≥0.6 且 pressure_recovery≥0.35 且 ldlr_prev<0.5
  放量分歧 volr5≥1.5 且 cpos<0.5 且 peak_drawdown>-25% 且 neg_streak<2
  绕异动   volr5<0.8 且 |pct|<3% 且 cpos≥0.5 且 peak_drawdown>-15%
  断板观察 默认
字段口径: pct/cpos=当日T实际(离线=T收盘bar / 盘中=报价); volr5/neg_streak/
  neg_deep/ldlr_prev=factor.longtou的T行(=T-1结构值); peak_drawdown/
  pressure_recovery=近20日峰谷(≤T-1)对比现价。
"""

SHORTBOARD_CONTRACT = "shortboard-r1-20260903"

# ---- 冻结阈值(研究29验证; 与 research/29_shortboard_secondwave.py 一致) ----
N_WINDOW = 5            # 近N日龙头/高板回看
PEAK_WIN = 20           # 峰/谷回看窗口(第一波压力峰)
GUARD_DD = -0.30        # 高位守卫: 距峰回撤下限(未深破位)
GUARD_WAVE = 0.35       # 高位守卫: 20日最大涨幅下限
HB_LIMIT = 3            # 市场高板入池连板阈值
# 监管严重异常波动线(沪深交易所): 连续10交易日累计涨幅偏离+100% / 连续30交易日+200%
# →停牌核查。"绕异动"=主力控盘卡在此线附近(达线的YD_NEAR), 非缩量窄幅。
YD_LINE10 = 100.0
YD_LINE30 = 200.0
YD_NEAR = 0.8           # 接近阈值(10日≥80% 或 30日≥160% 视为绕异动)

# 5态(展示顺序 = 风险从低到高; 研究29前向收益: 断板-0.9%<失效-1.2%<二波-1.4%<放量-1.7%<绕异动-2.3%)
STATE_ORDER = ["断板观察", "逻辑失效", "二波候选", "放量分歧", "绕异动"]

# 状态 → 风险标注(研究29前向收益嵌入note; level供前端配色, 冲高态=警惕非看多)
STATE_RISK = {
    "绕异动":   {"risk": "接近异动线", "level": "danger",
                 "note": "10日累计涨幅接近监管严重异常波动线(+100%/30日+200%), 极高位·停牌核查/反转风险; 历史T+1开盘-2.3%(cohort最差)"},
    "断板观察": {"risk": "中性观察", "level": "neutral",
                 "note": "断板默认态; 历史T+1开盘-0.9%"},
    "逻辑失效": {"risk": "退场", "level": "fail",
                 "note": "深破位/连续负反馈; 历史T+1开盘-1.2%"},
    "二波候选": {"risk": "追高警惕", "level": "warn",
                 "note": "放量冲高; 历史多为追高/出货陷阱, T+1开盘-1.5%"},
    "放量分歧": {"risk": "出货警惕", "level": "danger",
                 "note": "放量低收; 历史T+1开盘-1.7%(cohort最差)"},
}

# 状态 → 前端徽章CSS类(配色映射, 与web/dashboard.html的b-sb-*一致)
STATE_BADGE = {st: f"b-sb-{STATE_RISK[st]['level']}" for st in STATE_ORDER}


def peak_features(bars: list, cur_close) -> dict:
    """近PEAK_WIN根(≤T-1)算第一波峰/谷, 对比现价cur_close(离线=T收盘/盘中=最新价)。
    bars=[(date, high, low, close)] 升序。返回 peak_drawdown/pressure_recovery/
    wave_ret20; 数据不足返回None(禁止补缺)。"""
    win = bars[-PEAK_WIN:]
    highs = [b[1] for b in win if b[1] is not None and b[1] == b[1]]
    lows = [b[2] for b in win if b[2] is not None and b[2] == b[2]]
    if (not highs or not lows or cur_close is None
            or cur_close != cur_close or cur_close <= 0):
        return {"peak_drawdown": None, "pressure_recovery": None,
                "wave_ret20": None}
    peak = max(highs)
    trough = min(lows)
    dd = cur_close / peak - 1 if peak > 0 else None
    rng = peak - trough
    rec = (cur_close - trough) / rng if rng > 0 else 0.0
    wave = peak / trough - 1 if trough > 0 else None
    return {
        "peak_drawdown": round(dd, 4) if dd is not None else None,
        "pressure_recovery": round(max(0.0, min(1.0, rec)), 4),
        "wave_ret20": round(wave, 4) if wave is not None else None,
    }


def passes_guard(peak_drawdown, wave_ret20, max_lt) -> bool:
    """高位守卫: 未深破位 且 (20日最大涨幅≥阈值 或 窗口内曾连板≥HB_LIMIT)"""
    if peak_drawdown is None or peak_drawdown <= GUARD_DD:
        return False
    wave_ok = wave_ret20 is not None and wave_ret20 >= GUARD_WAVE
    hb_ok = max_lt is not None and max_lt >= HB_LIMIT
    return bool(wave_ok or hb_ok)


def struct_from_bars(bars: list) -> dict:
    """从≤T-1的日bar算 volr5/neg_streak/neg_deep, 口径对齐 factor_longtou.py
    (y_volr5=昨量/前5日均量; neg_streak=截至昨日连续收跌; neg_deep=近3日有≤-5%)。
    bars=[(date, high, low, close, vol, pct_chg)] 升序, 末根=T-1。
    与 review(factor.longtou) 同源同值(均派生自daily_panel), 供盘中复用。"""
    if len(bars) < 2:
        return {"volr5": None, "neg_streak": None, "neg_deep": None}
    vols = [b[4] for b in bars[-6:-1]
            if b[4] is not None and b[4] == b[4] and b[4] > 0]
    last_vol = bars[-1][4]
    volr5 = None
    if len(vols) >= 3 and last_vol is not None and last_vol == last_vol:
        volr5 = round(last_vol / (sum(vols) / len(vols)), 4)
    neg_streak = 0                       # 截至末根(T-1)连续收跌天数
    for b in reversed(bars):
        pc = b[5]
        if pc is not None and pc == pc and pc < 0:
            neg_streak += 1
        else:
            break
    recent = [b[5] for b in bars[-3:] if b[5] is not None and b[5] == b[5]]
    neg_deep = bool(recent) and min(recent) <= -5.0
    return {"volr5": volr5, "neg_streak": neg_streak, "neg_deep": neg_deep}


def death_test_of(code: str, bars: list) -> bool:
    """死亡测试(移植 research/longtou.detect_death_test): 第一波峰后出现
    回撤≥10% 或 当日跌≥7% 或 触板未封(量比≥1.5+收盘位置<0.55)。
    第一波资格: 峰涨幅≥35% 或 峰前20日涨停≥3。一旦触发不可逆(后续创新高
    不能抹掉)。bars=[(date,high,low,close,vol,pct_chg)]升序(≤T-1)。"""
    n = len(bars)
    if n < 3:
        return False
    rate = 0.20 if str(code)[:2] in ("30", "68") else 0.10
    limit_f, touch_f, cpos_f, vr_f = [], [], [], []
    for i, b in enumerate(bars):
        hi, lo, cl, vol, pct = b[1], b[2], b[3], b[4], b[5]
        if cl is None or cl != cl or pct is None:
            limit_f.append(False); touch_f.append(False)
            cpos_f.append(0.5); vr_f.append(None)
            continue
        pre = cl / (1 + pct / 100) if (1 + pct / 100) != 0 else None
        lp = round(pre * (1 + rate), 2) if pre and pre > 0 else None
        limit_f.append(bool(lp and cl >= lp * 0.999))
        touch_f.append(bool(lp and hi is not None and hi == hi
                            and hi >= lp * 0.999))
        cpos_f.append((cl - lo) / (hi - lo)
                      if (hi and lo and hi == hi and lo == lo and hi > lo)
                      else 0.5)
        pv = [bars[j][4] for j in range(max(0, i - 5), i)
              if bars[j][4] is not None and bars[j][4] == bars[j][4]
              and bars[j][4] > 0]
        vr_f.append(vol / (sum(pv) / len(pv))
                    if len(pv) >= 3 and vol and vol == vol else None)
    peak_close = bars[0][3]
    peak_index = 0
    running_low = bars[0][3]
    for i in range(1, n):
        prior_peak, prior_peak_index = peak_close, peak_index
        peak_gain = (prior_peak / running_low - 1
                     if running_low and running_low > 0 else 0.0)
        prior_limits = sum(1 for f in limit_f[max(0, i - 20):i] if f)
        wave_qualified = peak_gain >= 0.35 or prior_limits >= 3
        cur_ret = (bars[i][5] / 100) if bars[i][5] is not None else 0.0
        drawdown = (bars[i][3] / prior_peak - 1
                    if prior_peak and prior_peak > 0 else 0.0)
        broken_board = bool(touch_f[i] and not limit_f[i]
                            and vr_f[i] is not None and vr_f[i] >= 1.5
                            and cpos_f[i] < 0.55)
        if wave_qualified and i > prior_peak_index and (
                drawdown <= -0.10 or cur_ret <= -0.07 or broken_board):
            return True
        if bars[i][3] is not None and bars[i][3] == bars[i][3]:
            if bars[i][3] > peak_close:
                peak_close = bars[i][3]
                peak_index = i
            running_low = min(running_low, bars[i][3])
    return False


def _cum_gain(bars: list, n: int, today_pct) -> float | None:
    """近n交易日累计涨幅%(含当日today_pct); bars=≤T-1取末n-1根。
    窗口不足(次新股)返回None。监管异动口径用累计涨幅作偏离值近似。"""
    seg = bars[-(n - 1):]
    if len(seg) < n - 1 or today_pct is None:
        return None
    prod = 1.0
    for b in seg:
        pc = b[5]
        if pc is None or pc != pc:
            return None
        prod *= (1 + pc / 100)
    prod *= (1 + today_pct / 100)
    return round((prod - 1) * 100, 1)


def build_snapshot(pct, cpos, volr5, neg_streak, neg_deep, ldlr_prev,
                   peak_drawdown, pressure_recovery, death_test=False,
                   cum10=None, cum30=None) -> dict:
    """组装状态机快照(字段口径见模块docstring)"""
    return {"pct": pct, "cpos": cpos, "volr5": volr5,
            "neg_streak": neg_streak, "neg_deep": neg_deep,
            "ldlr_prev": ldlr_prev, "peak_drawdown": peak_drawdown,
            "pressure_recovery": pressure_recovery,
            "death_test": bool(death_test),
            "cum10": cum10, "cum30": cum30}


def build_cohort(recent_leaders, recent_highboards, exclude,
                 bars_by_code, cur_close_by_code=None) -> list:
    """龙头短板cohort(唯一出处): 近N日题材龙头∪市场高板 - 今日涨停池, 过高位守卫。
    recent_leaders: 可迭代code(近N日theme.day龙头);
    recent_highboards: {code: 窗口内最大连板};
    exclude: 今日涨停池(可迭代code);
    bars_by_code: {code: ≤T-1日bar};
    cur_close_by_code: {code: 现价}(离线=T收盘/盘中=报价), 缺失用bars末根close。
    返回通过高位守卫的 [code](升序)。"""
    hb = recent_highboards if recent_highboards is not None else {}
    leaders = set(recent_leaders) if recent_leaders is not None else set()
    excl = set(exclude) if exclude is not None else set()
    cand = (leaders | set(hb)) - excl
    cc = cur_close_by_code or {}
    out = []
    for c in sorted(cand):
        bars = bars_by_code.get(c, [])
        if not bars:
            continue
        cur = cc.get(c) or bars[-1][3]
        pf = peak_features(bars, cur)
        if passes_guard(pf["peak_drawdown"], pf["wave_ret20"], hb.get(c, 0)):
            out.append(c)
    return out


def shortboard_snapshot(code, bars, factor_row=None, live_quote=None,
                        day_bar=None):
    """单票状态机快照(唯一出处)。返回 (snapshot, peak_dict) 或 (None, None)。
    bars=[(date,high,low,close,vol,pct_chg)]升序(≤T-1);
    factor_row: factor.longtou行(取ldlr_prev, 可None);
    live_quote: 盘中报价{price,pct,high,low}(盘中路径);
    day_bar: 离线当日T bar(date,high,low,close,vol,pct_chg)(离线路径)。
    现价/pct/cpos: 盘中取报价, 离线取day_bar; volr5/neg从bars算(与factor同值)。"""
    ldlr = None
    if factor_row is not None:
        v = getattr(factor_row, "ldlr_prev", None)
        if v is not None and v == v:          # 非None且非NaN
            ldlr = float(v)
    if live_quote is not None:
        px, pct = live_quote.get("price"), live_quote.get("pct")
        hi, lo = live_quote.get("high"), live_quote.get("low")
        if not px or px <= 0 or pct is None:
            return None, None
        cpos = (px - lo) / (hi - lo) if (hi and lo and hi > lo) else 0.5
        cur_close = px
    elif day_bar is not None:
        hi, lo, cl, pct = day_bar[1], day_bar[2], day_bar[3], day_bar[5]
        if cl is None or cl != cl or cl <= 0 or pct is None:
            return None, None
        cpos = (cl - lo) / (hi - lo) if (hi and lo and hi > lo) else 0.5
        cur_close = cl
    else:
        return None, None
    pf = peak_features(bars, cur_close)
    st = struct_from_bars(bars)
    # death_test 仅扫近期窗口(PEAK_WIN+1根), 避免陈旧死亡测试把新连板第一波误判为二波;
    # cum30 才用全窗口(32根)。与 research/29 的21根 death_test 窗口同口径。
    dt = death_test_of(code, bars[-(PEAK_WIN + 1):])
    snap = build_snapshot(pct=float(pct), cpos=cpos, volr5=st["volr5"],
                          neg_streak=st["neg_streak"], neg_deep=st["neg_deep"],
                          ldlr_prev=ldlr, peak_drawdown=pf["peak_drawdown"],
                          pressure_recovery=pf["pressure_recovery"],
                          death_test=dt,
                          cum10=_cum_gain(bars, 10, float(pct)),
                          cum30=_cum_gain(bars, 30, float(pct)))
    return snap, pf


def shortboard_state_of(snap: dict, prior_state=None) -> tuple:
    """精简5态(自上而下首个命中) + 失效粘滞。返回 (state, reason)。
    与 research/29.shortboard_state 同口径; prior_state=逻辑失效时粘滞不回退。"""
    if prior_state == "逻辑失效":
        return "逻辑失效", "失效粘滞(前次已失效)"
    dd = snap.get("peak_drawdown")
    rec = snap.get("pressure_recovery")
    ns = snap.get("neg_streak")
    nd = snap.get("neg_deep")
    ldlr = snap.get("ldlr_prev")
    pct = snap.get("pct")
    cpos = snap.get("cpos")
    volr5 = snap.get("volr5")
    if dd is None or pct is None or cpos is None:
        return "断板观察", "字段不足(缺当日价/回撤)"
    # 1 逻辑失效
    if (dd <= GUARD_DD
            or (ns is not None and ns >= 2 and bool(nd))
            or (ldlr is not None and ldlr >= 0.5 and pct <= -5.0)):
        return "逻辑失效", f"回撤{dd:.0%}/连续负反馈/环境阻断"
    # 2 二波候选(需第一波已死亡测试: 峰后破位/触板未封, 排除第一波延续误标)
    if (snap.get("death_test") and pct >= 3.0 and cpos >= 0.6
            and rec is not None and rec >= 0.35
            and (ldlr is None or ldlr < 0.5)):
        return "二波候选", f"死亡测试后冲高{pct:.1f}%+修复{rec:.0%}"
    # 3 放量分歧(出货警惕)
    if (volr5 is not None and volr5 >= 1.5 and cpos < 0.5
            and dd > -0.25 and (ns is None or ns < 2)):
        return "放量分歧", f"放量{volr5:.1f}倍低收(cpos{cpos:.2f})"
    # 4 绕异动(接近监管严重异常波动线: 10日累计涨幅近+100% / 30日近+200%)
    c10, c30 = snap.get("cum10"), snap.get("cum30")
    if ((c10 is not None and c10 >= YD_LINE10 * YD_NEAR)
            or (c30 is not None and c30 >= YD_LINE30 * YD_NEAR)):
        return "绕异动", f"接近异动线(10日{c10 or 0:.0f}%/30日{c30 or 0:.0f}%)"
    # 5 断板观察
    return "断板观察", "断板默认态(未触发其他判据)"
