# -*- coding: utf-8 -*-
"""tri_lever — 决策链路三杠杆打板策略（独立于 v5_daban）

三杠杆(研究49/51/52验证, 盈亏比优先):
  ① 入场价位闸: 只在 pct<ENTRY_PCT_MAX(4%) 低位买入(研究49: 盈亏比0.99→1.34)
  ② E3续持闸: 封板票E3≥E3_HOLD(0.20)续持次日(研究51: 盈亏比3.74), E3低尾盘卖
  ③ 不止损: 不封板票持到14:55清仓(研究52: 不止损盈亏比1.31>止损5%的1.15)

硬约束(同 _template/SPEC.md):
  · 禁止 import 其它策略目录 / core/ quotes/ apps/
  · 数据只经注入 API 取(ticai_struct/ticai_signals/ticai_seal_strength)
  · 代码口径 rqalpha 格式(000001.XSHE)
"""
from rqalpha.api import *

from rqalpha_mod_ticai.broker import limit_price_of

# ---------- 三杠杆参数(改这里, 不要改框架) ----------
MAX_POS = 3               # 最大持仓数(每仓 1/3)
ENTRY_PCT_MAX = 4.0       # 杠杆①入场价位闸: 只买 pct<4%(研究49)
E3_HOLD = 0.20            # 杠杆②E3续持闸: 封板E3≥0.20续持(研究51盈亏比3.74)
STOP_ENABLE = False       # 杠杆③不止损(研究52: 不止损盈亏比1.31最优)
STOP = 0.05               # 回落止损(仅 STOP_ENABLE=True 时生效)
SCAN_END = "10:30"        # 10:30后不新买(V5定稿: 规避尾盘追高)
CLEAR_MIN = 14 * 60 + 55  # 14:55清仓时刻
SEAL_EPS = 0.9995         # 封死判据(贴死涨停价)


def init(context):
    set_benchmark("DBBNCH.XSHG")
    context.candidates = set()      # 盘前结构闸候选池
    context.entry = {}              # code -> {ep 买入价, hi 当日最高}
    context.traded = set()          # 当日已买(防重复)
    context.failed = set()          # 拒单名单(一字板/涨停)
    context.pending = {}            # code -> order_id(在途挂单占仓)
    scheduler.run_daily(clear_unsealed, time_rule=CLEAR_MIN)


def before_trading(context):
    """盘前: 结构闸选股(g_chip 4项≥3健康), 构建候选池"""
    struct = ticai_struct()
    context.candidates = {c for c, s in struct.items() if s.get("gate")}
    context.traded = set()
    context.failed = set()
    logger.info(f"盘前候选池 {len(context.candidates)} 只")


def handle_bar(context, bar_dict):
    """盘中(每20s): 杠杆③不止损 + 杠杆①入场价位闸买入"""
    now_hm = context.now.strftime("%H:%M") if hasattr(context, "now") else ""

    # ---- 杠杆③: 不止损(STOP_ENABLE=False默认关闭) ----
    # 研究52: 不止损盈亏比1.31>止损5%的1.15且回撤更小; 不封板票持到14:55清仓
    if STOP_ENABLE:
        for code, pos in list(context.portfolio.positions.items()):
            st = context.entry.get(code)
            if st is None or pos.sellable <= 0:
                continue
            bar = bar_dict[code] if code in bar_dict else None
            if bar is None:
                continue
            last = float(bar.last)
            if last != last:
                continue
            st["hi"] = max(st["hi"], last)
            ref = max(st["ep"], st["hi"])
            if last <= ref * (1 - STOP):
                order_target_percent(code, 0)
                context.entry.pop(code, None)
                logger.info(f"止损 {code} @{last:.2f} (参考{ref:.2f})")

    # ---- 买入: 10:30后不新买 ----
    if now_hm and now_hm > SCAN_END:
        return
    open_ids = {o.order_id for o in get_open_orders()}
    for c in list(context.pending.keys()):
        if c in context.portfolio.positions:
            del context.pending[c]
        elif context.pending[c] not in open_ids:
            context.traded.discard(c)
            del context.pending[c]
    slots = MAX_POS - len(context.portfolio.positions) - len(context.pending)
    if slots <= 0:
        return
    struct = ticai_struct()
    # ---- 杠杆①: 入场价位闸 pct<ENTRY_PCT_MAX ----
    sigs = [s for s in ticai_signals()
            if s["stage"] in ("S2", "S3")
            and (s.get("pct") or 0) < ENTRY_PCT_MAX
            and s["code"] in context.candidates
            and s["code"] not in context.traded
            and s["code"] not in context.failed
            and s["code"] not in context.pending
            and s["code"] not in context.portfolio.positions]
    if not sigs:
        return
    sigs.sort(key=lambda s: -(struct.get(s["code"], {}).get("v5") or 0))
    for s in sigs[:slots]:
        code = s["code"]
        px = float(s.get("price0") or 0)
        if px <= 0:
            continue
        limit_up = float(s.get("limit_px") or 0) or None
        pre = None
        if code in bar_dict:
            bar = bar_dict[code]
            if not limit_up:
                lu = bar.limit_up
                limit_up = float(lu) if lu == lu else None
            pc = bar.prev_close
            pre = float(pc) if pc == pc else None
        lmt = limit_price_of(px, pre, limit_up=limit_up)
        if limit_up and lmt >= limit_up:
            context.failed.add(code)
            continue
        cash_per = min(context.portfolio.total_value / MAX_POS,
                       context.portfolio.cash)
        qty = int(cash_per / lmt / 100) * 100
        if qty < 100:
            continue
        od = order_shares(code, qty, LimitOrder(lmt))
        if od is None:
            context.failed.add(code)
            continue
        context.traded.add(code)
        context.pending[code] = od.order_id
        context.entry[code] = {"ep": px, "hi": px}
        logger.info(f"买入 {code} [{s.get('why')}] 触发{px:.2f} 限价{lmt:.2f}")


def clear_unsealed(context, bar_dict):
    """14:55: 未封板清仓; 杠杆②封板E3≥E3_HOLD续持, E3低尾盘卖"""
    seal = ticai_seal_strength()      # {代码: E3封单强度}
    for code, pos in list(context.portfolio.positions.items()):
        if pos.sellable <= 0:
            continue
        bar = bar_dict_last(code)
        limit_up = None
        try:
            limit_up = float(bar.limit_up) if bar is not None else None
        except Exception:
            limit_up = None
        sealed = (bar is not None and limit_up and limit_up == limit_up
                  and float(bar.last) >= limit_up * SEAL_EPS)
        if sealed:
            e3 = seal.get(code, 0.0)
            if e3 >= E3_HOLD:         # 杠杆②: E3高续持(盈亏比3.74)
                logger.info(f"封板续持 {code} @{float(bar.last):.2f} E3={e3:.2f}")
                continue
            order_target_percent(code, 0)   # E3低尾盘卖(盈亏比1.37<3)
            context.entry.pop(code, None)
            logger.info(f"14:55清仓(封板但E3低{e3:.2f}<{E3_HOLD}) {code}")
            continue
        order_target_percent(code, 0)
        context.entry.pop(code, None)
        logger.info(f"14:55清仓(未封板) {code}")


def after_trading(context):
    n_pos = sum(1 for p in context.portfolio.positions.values()
                if float(getattr(p, "quantity", 0) or 0) > 0)
    logger.info(f"收盘: 持仓 {n_pos} 只, 当日买入 {len(context.traded)} 笔")


def bar_dict_last(code):
    """最新bar(收盘价与涨停价); 无数据返回None"""
    try:
        bars = history_bars(code, 1, "1d", ["close", "limit_up"])
        if bars is None or not len(bars):
            return None
        return bars[-1]
    except Exception:
        return None
