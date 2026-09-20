# -*- coding: utf-8 -*-
"""profit_engine — 盈利引擎独立策略（决策链路验证的唯一达标高盈亏比模式）

【为什么独立】决策链路已闭环证明: 打板「买入等封板」模式整体盈亏比数学上
封顶 ~1.3(86%买入票不封板→亏损单锚定平均亏损)。唯一达标(盈亏比3.74)的是
「盈利引擎」子集 —— 昨日封板 E3 高票次日开盘卖(研究51, 38701样本跨
2019-2026, 胜率80.1%, 期望+4.37)。本策略把这个盈利引擎从 tri_lever 的混合
池里拎出来: 独立进程/独立账户/分离度量, 牺牲频率换盈亏比。

【盈利引擎口径】(研究51, 决策链路 2026-09-09 闭环验证)
  T日盘中低位(pct≤4%)打板买入(研究49: 盈亏比0.99→1.34)
    → T日14:55 封板确认 + E3≥0.20 → 标记盈利引擎续持
    → T+1日开盘卖出 —— 捕获「隔夜溢价」(昨日涨停价→次日开盘)
  3.74盈利只能由「昨日打板持有封板票→次日开盘卖」获得; 次日接力买入
  (开盘买入)隔夜溢价已被开盘价吸收, 盈亏比仅1.18(不可行)。

【与 tri_lever 的关键区别】
  · tri_lever 续持后持有到次日尾盘(次日14:55再判断), 混合度量盈亏比1.25
  · profit_engine 续持后【次日开盘卖】(捕获隔夜溢价3.74), 分离度量盈亏比

【T+1 现实修正】A股当日买入不可当日卖。故决策链路研究口径的「不封板票
当天清仓」在真实交易不可行 —— 当日买入票全部持有过夜, T+1开盘卖。本策略
据此分两类过夜持仓: 封板E3高→盈利引擎(T+1开盘卖享3.74); 不封板/E3低→
亏损单(T+1开盘卖隔离)。买入端无法预知封板(封板率与盈亏比负相关, 研究51),
故买入端同 tri_lever 低位打板, 盈利引擎是「买入后」的子集筛选。

【E3 数据链路】ticai_seal_strength() 从 limitup.events_enriched 现算
(fd_amount/amount)。回放/回测口径完整; 盘中实时当日 events_enriched 尚未
落盘时 E3 可能为空(决策链路已知缺口)→ 届时封板票全部按亏损单处理, 待
poller 实时封单额接入后补齐。

硬约束(同 _template/SPEC.md):
  · 禁止 import 其它策略目录 / core/ quotes/ apps/
  · 数据只经注入 API 取(ticai_struct/ticai_signals/ticai_seal_strength)
  · 代码口径 rqalpha 格式(000001.XSHE)
"""
from rqalpha.api import *

from rqalpha_mod_ticai.broker import limit_price_of

# ---------- 盈利引擎参数(改这里, 不要改框架) ----------
MAX_POS = 3               # 最大持仓数(每仓 1/3)
ENTRY_PCT_MAX = 4.0       # 低位入场闸: 只买 pct<4%(研究49盈亏比0.99→1.34)
E3_HOLD = 0.20            # 盈利引擎续持闸: 封板E3≥0.20(研究51盈亏比3.74)
SCAN_END = "10:30"        # 10:30后不新买(时段闸, 规避尾盘追高)
CLEAR_MIN = 14 * 60 + 55  # 14:55封板确认时刻
SEAL_EPS = 0.9995         # 封死判据(贴死涨停价)


def init(context):
    set_benchmark("DBBNCH.XSHG")
    context.candidates = set()      # 盘前结构闸候选池
    context.entry = {}              # code -> {ep 买入价, hi 当日最高}
    context.traded = set()          # 当日已买(防重复)
    context.failed = set()          # 拒单名单(一字板/涨停买不进)
    context.pending = {}            # code -> order_id(在途挂单占仓)
    context.engine = set()          # 盈利引擎: T日封板E3高, T+1开盘卖(3.74)
    context.losscut = set()         # 亏损单: T日不封板/E3低, T+1开盘卖(隔离)
    context._sell_date = None       # T+1开盘卖日期守卫(每日只卖一次)
    scheduler.run_daily(mark_overnight, time_rule=CLEAR_MIN)


def before_trading(context):
    """盘前: 结构闸选股(g_chip 4项≥3健康)。
    不重置 engine/losscut —— 它们是昨日续持票, 要在今日开盘卖。"""
    struct = ticai_struct()
    context.candidates = {c for c, s in struct.items() if s.get("gate")}
    context.traded = set()
    context.failed = set()
    logger.info(f"盘前候选池 {len(context.candidates)} 只")


def handle_bar(context, bar_dict):
    """盘中(每20s): ①盈利引擎T+1开盘卖(隔夜溢价) ②低位打板买入"""
    now_hm = context.now.strftime("%H:%M") if hasattr(context, "now") else ""
    today = context.now.date() if hasattr(context, "now") else None

    # ---- ①盈利引擎核心: T+1开盘卖(捕获隔夜溢价, 研究51盈亏比3.74) ----
    # 昨日标记的过夜持仓今日首根bar无条件开盘卖出(日期守卫保证每日一次)。
    # 盈利引擎与亏损单都开盘卖, 差别只在归属标记(供分离度量): 盈利引擎
    # 享隔夜溢价3.74, 亏损单是被隔离的不封板/E3低票。
    if today is not None and context._sell_date != today:
        context._sell_date = today
        for code in list(context.engine):
            pos = context.portfolio.positions.get(code)
            if pos is not None and pos.sellable > 0:
                order_target_percent(code, 0)
                logger.info(f"盈利引擎次日开盘卖 {code}"
                            f"(隔夜溢价, 研究51盈亏比3.74)")
            context.engine.discard(code)
        for code in list(context.losscut):
            pos = context.portfolio.positions.get(code)
            if pos is not None and pos.sellable > 0:
                order_target_percent(code, 0)
                logger.info(f"亏损单次日开盘卖 {code}(不封板/E3低, 隔离)")
            context.losscut.discard(code)

    # ---- ②买入: 10:30后不新买(时段闸) ----
    if now_hm and now_hm > SCAN_END:
        return
    open_ids = {o.order_id for o in get_open_orders()}
    for c in list(context.pending.keys()):
        if c in context.portfolio.positions:
            del context.pending[c]              # 已成交 → 转过夜标记
        elif context.pending[c] not in open_ids:
            context.traded.discard(c)
            del context.pending[c]              # 已撤 → 释放占仓
    slots = MAX_POS - len(context.portfolio.positions) - len(context.pending)
    if slots <= 0:
        return
    struct = ticai_struct()
    # ---- 低位入场闸: 只买 pct<ENTRY_PCT_MAX(研究49) ----
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
        # 挂价被涨停价封顶 → 触发价已贴近/超过涨停(一字板或已封死), 买不进
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


def mark_overnight(context, bar_dict):
    """14:55: 判断当日买入票封板+E3, 标记过夜归属(供T+1开盘卖)。
    注: 当日买入票T+1不可卖(sellable=0), 故此处只标记不卖 —— 盈利引擎
    「当天清仓」在真实T+1下不可行, 一律持有过夜次日开盘卖。"""
    seal = ticai_seal_strength()      # {代码: E3封单强度=封单额/成交额}
    n_engine = n_loss = 0
    for code, pos in list(context.portfolio.positions.items()):
        if code in context.engine or code in context.losscut:
            continue                  # 昨日已标记的跳过(今日开盘已卖)
        bar = bar_dict_last(code)
        limit_up = None
        try:
            limit_up = float(bar.limit_up) if bar is not None else None
        except Exception:
            limit_up = None
        sealed = (bar is not None and limit_up and limit_up == limit_up
                  and float(bar.last) >= limit_up * SEAL_EPS)
        e3 = seal.get(code, 0.0)
        if sealed and e3 >= E3_HOLD:
            context.engine.add(code)  # 盈利引擎: T+1开盘卖(盈亏比3.74)
            n_engine += 1
            logger.info(f"盈利引擎续持标记 {code} E3={e3:.2f}(T+1开盘卖)")
        else:
            context.losscut.add(code)  # 亏损单: T+1开盘卖(隔离)
            n_loss += 1
            reason = "不封板" if not sealed else f"封板但E3低{e3:.2f}"
            logger.info(f"亏损单标记 {code}({reason}, T+1开盘卖隔离)")
    logger.info(f"14:55过夜标记: 盈利引擎{n_engine}只 亏损单{n_loss}只")


def after_trading(context):
    """收盘: 分离度量记录(盈利引擎子集 vs 亏损单子集)。
    盈亏比的分离统计从 sim.log 的归属标记聚合 —— 盈利引擎子集目标3+,
    亏损单子集追求最小化, 不再看被平均掉的整体盈亏比。"""
    n_pos = sum(1 for p in context.portfolio.positions.values()
                if float(getattr(p, "quantity", 0) or 0) > 0)
    logger.info(f"收盘: 持仓{n_pos}只(盈利引擎续持{len(context.engine)} "
                f"亏损单{len(context.losscut)}) 当日买入{len(context.traded)}笔")


def bar_dict_last(code):
    """最新bar(收盘价与涨停价); 无数据返回None"""
    try:
        bars = history_bars(code, 1, "1d", ["close", "limit_up"])
        if bars is None or not len(bars):
            return None
        return bars[-1]
    except Exception:
        return None
