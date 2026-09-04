# -*- coding: utf-8 -*-
"""V5 两层打板策略 — 移植自 research/jq_v5_strategy.py（研究22/23/24/24b 定稿）

两层架构（与聚宽版同口径，但数据源改为项目雷达）:
  结构层（盘前，T-1）  ticai_struct() 的 gate —— g_chip 4 项中 ≥3 项健康
                       (行业排名>3.5 / 近20日炸板≤1 / 昨日量比∈(0.55,2.2] / 连跌≥3)
                       见 core/structure.py，与聚宽版 prepare() 同源实现
  盘中层（触发）       ticai_signals() 的 S2/S3 —— 研究14-21 前向验证规则
                       S3 高开稳封相/高开剧震/竞价量爆 ≡ 聚宽版 G组
                       S2 颠簸高/颠簸加速          ≡ 聚宽版 L组
  融合排序             v5 融合分降序占仓（同 cycle 多触发时分数优先，非先到先得）

出场（沿用研究口径）:
  回落 5% 止损（参考价 = max(买入价, 当日最高)）
  14:55 未封板清仓；封板续持至次日
  OPT_A 次日开盘保护：昨日持仓低开 ≤ -3% 首根 bar 直接卖

未移植项（数据缺口，见 SPEC.md 的注入 API 表）:
  OPT_C 的高位/换手/流通市值过滤 —— 聚宽版用 get_valuation 取
  turnover_ratio/circulating_market_cap，本框架的注入 API 尚未暴露这三个
  字段；rise20 需 20 日日线。补齐后可在此加回，当前仅靠结构闸过滤。

代码口径: 一律 rqalpha 格式（000001.XSHE），注入 API 已转换好。
"""
from rqalpha.api import *

from rqalpha_mod_ticai.broker import limit_price_of

# ---------- 参数（与 research/jq_v5_strategy.py 对齐）----------
MAXPOS = 3               # 最大持仓数（每仓 1/3）
STOP = 0.05              # 回落止损
CLEAR_MIN = 14 * 60 + 55  # 未封板清仓时刻 14:55 = 距午夜 895 分钟
                          # (rqalpha 的 scheduler.run_daily 的 time_rule 是
                          #  整数分钟数, 不接受 '14:55' 字符串)
SCAN_END = "10:30"       # 该时刻后不再新买（时段效应: ≤10:00 胜率82.4%/盈亏比3.09,
                         # >10:00 降至 75.5%/1.59）
OPEN_CUT = -3.0          # OPT_A 次日开盘保护阈值(%)
SEAL_EPS = 0.995         # 封死判定容差（价 ≥ 涨停价×0.995）
RISK_OFF = 0.70          # 市场风险分 ≥ 此值则当日不新买(风险规避)


def init(context):
    set_benchmark("DBBNCH.XSHG")
    context.candidates = set()
    context.entry = {}            # code -> {ep, hi}
    context.traded = set()
    context.pending = {}          # code -> order_id（在途挂单占仓）
    context.risk_off = False      # AI 风险 feed 判定的当日规避开关
    # after_trading 阶段不能下单(rqalpha 报 "You cannot call
    # order_target_percent when executing [日内交易后]"), 故未封清仓
    # 用 scheduler 定时在 14:55 做 —— 同聚宽版 run_daily(endofday, '14:55')
    scheduler.run_daily(clear_unsealed, time_rule=CLEAR_MIN)


def before_trading(context):
    """盘前: 结构闸选股。
    OPT_A 次日开盘保护不在这里做 —— before_trading 早于当日首根 bar,
    拿不到当日开盘价; 按聚宽版的做法放在 handle_bar 里带日期守卫。"""
    # ---- 结构层选股: 全部 g_chip 闸通过的票进候选集 ----
    # 与聚宽版 prepare() 同口径: `g.cand = {c: f for ... if f['gate']}` ——
    # 候选集是"全部闸通过"(几千只)而不是排名前 N。若限成前 N 会与盘中
    # 信号集交集极小(实测全天只剩 1 次买入尝试) —— 结构层负责"能不能买",
    # 盘中层负责"什么时候买", v5 只在同 cycle 多触发时用于排序。
    struct = ticai_struct()
    passed = {c for c, s in struct.items() if s.get("gate")}
    context.candidates = passed
    context.traded = set()

    # ---- AI feed 订阅(时间戳闸门已由框架施加, 无需自己过滤时间) ----
    # 只能读 config.yaml 里声明过的 feed; 未声明会被拒绝并告警
    risk = ai_feed("market_risk")
    context.risk_off = False
    if risk:
        score = risk[-1].get("score") or 0.0
        context.risk_off = score >= RISK_OFF
        logger.info(f"市场风险 feed: {risk[-1].get('text')} → "
                    f"{'当日风险规避, 不新买' if context.risk_off else '正常交易'}")
    narr = ai_feed("theme_narrative")
    if narr:
        logger.info(f"题材叙事 feed {len(narr)} 条, 前3: "
                    f"{[(e.get('topic'), e.get('score')) for e in narr[:3]]}")

    # logbook 的 logger 用 {} 风格而非 % 风格, 统一用 f-string
    logger.info(f"盘前: 结构分覆盖 {len(struct)} 只, 闸通过(候选集) "
                f"{len(passed)} 只")


def open_auction(context, bar_dict):
    """竞价(09:25): 签同 handle_bar 为两参数(context, bar_dict)。
    S3 竞价量爆分支已由雷达判定, 此处只做日志。"""
    s3 = [s for s in ticai_signals("S3")]
    if s3:
        logger.info(f"竞价 S3 信号 {len(s3)} 只: "
                    f"{[(s['name'], s['why']) for s in s3[:5]]}")


def _freq(context) -> str:
    try:
        return context.config.base.frequency
    except Exception:
        return "1m"


def handle_bar(context, bar_dict):
    """盘中(每 20s): OPT_A 开盘保护 + 回落止损 + 信号触发买入。
    日频(1d)下单日只有一根 15:00 的 bar, 此时 SCAN_END(10:30) 不适用:
    日频是"当日若触发信号则按触发价近似成交"的粗粒度近似, 豁免盘中
    时段限制; 能否成交仍由 FILL_SIM 闸门与涨停拒单约束。"""
    now_hm = context.now.strftime("%H:%M") if hasattr(context, "now") else ""

    # ---- OPT_A: 当日首根 bar 检查昨日持仓的开盘缺口 ----
    # (针对 T+1 导致当日止损失效后的隔夜跳空敞口; 日期守卫保证只跑一次)
    today = context.now.date() if hasattr(context, "now") else None
    if today is not None and getattr(context, "_ocut_date", None) != today:
        context._ocut_date = today
        for code, pos in list(context.portfolio.positions.items()):
            if code in context.entry:
                continue                    # 当日买入的由止损接管
            bar = bar_dict[code] if code in bar_dict else None
            if bar is None:
                continue
            pre = float(bar.prev_close)
            if pre != pre or pre <= 0:      # NaN / 无昨收
                continue
            gap = (float(bar.open) / pre - 1) * 100
            if gap <= OPEN_CUT:
                order_target_percent(code, 0)
                context.entry.pop(code, None)
                logger.info(f"次日开盘保护卖出 {code} 开盘{gap:+.1f}%"
                            f"(阈值{OPEN_CUT:.1f}%)")
    # ---- 持仓管理: 回落止损（参考价 = max(买入价, 当日最高)）----
    # 注: portfolio.positions 里是 StockPositionProxy, T+1 可卖量字段叫
    # sellable(= 持仓 - 今日买入 - 已冻结), 不叫 closable
    for code, pos in list(context.portfolio.positions.items()):
        st = context.entry.get(code)
        if st is None or pos.sellable <= 0:
            continue                        # 当日买入 T+1 不可卖
        bar = bar_dict[code] if code in bar_dict else None
        if bar is None:
            continue
        last = float(bar.last)
        if last != last:                    # NaN
            continue
        st["hi"] = max(st["hi"], last)
        ref = max(st["ep"], st["hi"])
        if last <= ref * (1 - STOP):
            order_target_percent(code, 0)
            context.entry.pop(code, None)
            logger.info(f"止损 {code} @{last:.2f} (参考{ref:.2f})")

    # ---- 买入: 10:30 后不再新买（时段效应）; 日频豁免 ----
    if _freq(context) != "1d" and now_hm and now_hm > SCAN_END:
        return
    if getattr(context, "risk_off", False):
        return                        # AI 风险 feed 判定当日规避
    # 在途挂单也占仓（同聚宽版 g.pending）: 否则挂单未成交时 positions
    # 仍为 0 → slots 永远算成 3 → 会超额下单
    # 注: rqalpha 的 get_open_orders() 返回 List[Order](不是 dict)
    open_ids = {o.order_id for o in get_open_orders()}
    for c in list(context.pending.keys()):
        if c in context.portfolio.positions:
            del context.pending[c]              # 已成交 → 转持仓管理
        elif context.pending[c] not in open_ids:
            context.traded.discard(c)
            del context.pending[c]              # 已撤 → 释放占仓
    slots = MAXPOS - len(context.portfolio.positions) - len(context.pending)
    if slots <= 0:
        return
    struct = ticai_struct()
    sigs = [s for s in ticai_signals()
            if s["stage"] in ("S2", "S3")
            and s["code"] in context.candidates
            and s["code"] not in context.traded
            and s["code"] not in context.pending
            and s["code"] not in context.portfolio.positions]
    if not sigs:
        return
    # 融合分降序占仓（分数优先，而非先到先得）
    sigs.sort(key=lambda s: -(struct.get(s["code"], {}).get("v5") or 0))
    for s in sigs[:slots]:
        code = s["code"]
        px = float(s.get("price0") or 0)
        if px <= 0:
            continue
        cash_per = min(context.portfolio.total_value / MAXPOS,
                       context.portfolio.cash)   # cash = 可用资金
        qty = int(cash_per / px / 100) * 100
        if qty < 100:
            logger.info(f"现金不足一手 跳过 {code} "
                        f"(可用{context.portfolio.cash:.0f} 需{px * 100:.0f})")
            continue
        # 高挂限价: 触发价×1.005, 涨停价封顶（实盘扫板同款挂法）
        pre = None
        if code in bar_dict:
            pc = bar_dict[code].prev_close
            pre = float(pc) if pc == pc else None
        lmt = limit_price_of(px, pre)
        # rqalpha 的限价单类叫 LimitOrder(聚宽叫 LimitOrderStyle)
        od = order_shares(code, qty, LimitOrder(lmt))
        if od is None:
            continue                        # 下单失败(涨停/停牌等)
        context.traded.add(code)
        context.pending[code] = od.order_id
        context.entry[code] = {"ep": px, "hi": px}
        v5 = struct.get(code, {}).get("v5")
        logger.info(f"买入 {code} [{s.get('why')}] 触发{px:.2f} 限价{lmt:.2f} "
                    f"数量{qty} v5={f'{v5:.3f}' if v5 is not None else '-'}")


def clear_unsealed(context, bar_dict):
    """14:55 定时: 未封板清仓; 封板续持至次日。
    用 scheduler.run_daily 而非 after_trading —— 后者不能下单。
    注: scheduler 注册的函数也必须取两参数(context, bar_dict)。"""
    for code, pos in list(context.portfolio.positions.items()):
        if pos.sellable <= 0:
            continue                        # 当日买入 T+1 不可卖
        bar = bar_dict_last(code)
        limit_up = None
        try:
            limit_up = float(bar.limit_up) if bar is not None else None
        except Exception:
            limit_up = None
        sealed = (bar is not None and limit_up and limit_up == limit_up
                  and float(bar.last) >= limit_up * SEAL_EPS)
        if sealed:
            logger.info(f"封板续持 {code} @{float(bar.last):.2f}")
            continue
        order_target_percent(code, 0)
        context.entry.pop(code, None)
        logger.info(f"14:55 清仓(未封板) {code}")


def after_trading(context):
    """收盘(15:30): 只能读不能下单, 故只做当日复盘日志"""
    n_pos = len(context.portfolio.positions)
    logger.info(f"收盘: 持仓 {n_pos} 只, 当日买入尝试 "
                f"{len(context.traded)} 笔")


def bar_dict_last(code):
    """最新 bar（收盘价与涨停价）; 无数据返回 None"""
    try:
        bars = history_bars(code, 1, "1d", ["close", "limit_up"])
        if bars is None or not len(bars):
            return None
        return bars[-1]
    except Exception:
        return None
