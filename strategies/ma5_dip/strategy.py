# -*- coding: utf-8 -*-
"""ma5_dip — 首板回调5日线低吸（跨日分批 + VWAP低开判定 + ATR追踪止盈）

规则（与需求方确认定稿）:
  候选池  盘前读私有 feed dip_pool(build/build_dip_pool.py 每晚产出):
          近8个交易日内首板、之后未再涨停、T-1未破MA5; 含预计算 ma5/atr。
  低吸    价格回踩 ma5×(1+DIP_BAND) 且未破 ma5×(1-BREAK_TOL) → 限价买入。
          每票预算=总值/MAX_POS, 分 N_TRANCHES 笔; 加仓须较上笔成交价再低
          ≥TRANCHE_STEP, 每日每票最多1笔(跨日分批, "第二天继续低吸")。
  次日    开盘≥MA5 → 正常回踩加仓/续持;
          开盘<MA5(低开) → 不立即卖, 10:00后看当日VWAP: 价在VWAP上方续持,
          下方清仓(1d回测无分时 → 降级为只看尾盘收盘判, 打日志)。
  尾盘    14:55 价格 < ma5×(1-BREAK_TOL) → 清仓; 当日买入笔 T+1 不可卖
          → 记 exit_open 次日开盘无条件卖。未破 → 续持。
  止盈    浮盈且 持仓期最高价回撤 ≥ ATR_MULT×ATR → 追踪止盈清仓;
          止盈后封锁该票加仓, 残余(T+1锁定)次日开盘卖。

硬约束(同 _template/SPEC.md): 只用注入 API, 不 import core/apps/其它策略。
"""
from rqalpha.api import *

# ---------- 可调参数(改这里, 不要改框架) ----------
MAX_POS = 5              # 最大同时持票数
N_TRANCHES = 3           # 每票最大买入笔数(跨日)
TRANCHE_STEP = 0.01      # 加仓价须较上笔成交价低 ≥1%
DIP_BAND = 0.005         # 低吸触发带: 价格 ≤ ma5×(1+0.5%) 视为回踩到位
BREAK_TOL = 0.01         # 破线容差: < ma5×(1-1%) 算跌破(不接/尾盘清仓)
ATR_MULT = 1.5           # 追踪止盈: 最高价回撤 ≥1.5×ATR 且浮盈
VWAP_AT = "10:00"        # 低开票的 VWAP 判定时刻
TAIL_MIN = 14 * 60 + 55  # 尾盘判定 14:55


def init(context):
    set_benchmark("DBBNCH.XSHG")
    context.pool = {}          # code -> {ma5, atr, name, board_date}(当日feed)
    context.tr = {}            # code -> {n, last_px, hi, atr, ma5, qty_seen, closed}
    context.pend = {}          # code -> {oid, q_before}(在途限价单)
    context.bought_today = set()
    context.exit_open = set()  # 次日开盘无条件清仓名单(T+1锁定/止盈残余)
    context.day_open = {}      # code -> 当日开盘价(首根bar)
    context.weak_gap = set()   # 当日低开(<ma5)持仓票
    context.vwap_done = set()  # 已完成VWAP判定(或降级)的票
    scheduler.run_daily(tail_check, time_rule=TAIL_MIN)


def before_trading(context):
    entries = ai_feed("private:dip_pool")
    pool = {}
    for e in entries:
        x = e.get("extra") or {}
        if e.get("topic") and x.get("ma5"):
            pool[e["topic"]] = x
    context.pool = pool
    context.bought_today = set()
    context.day_open = {}
    context.weak_gap = set()
    context.vwap_done = set()
    # 注意: context.pend 不在此清空 —— 低吸限价单可能挂多日, 次日成交
    # 仍要靠 pend 跟踪更新 tranche 状态(清空会丢成交检测, 实测踩坑)
    # 持仓票的 ma5 用当日池刷新(仍在池); 掉出池的沿用 tr 快照,
    # 盘中用 history_bars 现算兜底(_ma5_of)
    for code, st in context.tr.items():
        if code in pool:
            st["ma5"] = pool[code]["ma5"]
            st["atr"] = pool[code]["atr"]
        # 止盈清仓后的残余(T+1锁定) → 次日开盘卖
        if st.get("closed"):
            context.exit_open.add(code)
    logger.info(f"低吸池 {len(pool)} 只 | 持仓 {len(context.tr)} | "
                f"开盘待清 {sorted(context.exit_open)}")


def _pos_qty(context, code) -> int:
    pos = context.portfolio.positions.get(code)
    return int(getattr(pos, "quantity", 0) or 0) if pos is not None else 0


def _ma5_of(context, code) -> float | None:
    """持仓票的 MA5: 当日池优先; 掉出池用 T-1 起5根日线现算(排除当日bar,
    保证 live/1d回测同口径 —— 1d回测的 history_bars 含当日完整bar)"""
    p = context.pool.get(code)
    if p:
        return float(p["ma5"])
    st = context.tr.get(code)
    if st and st.get("ma5"):
        return st["ma5"]
    return None


def _ma5_recalc(context, code) -> float | None:
    """掉出候选池的持仓票: 用 T-1 起5根日线收盘现算 MA5。
    datetime 字段是 rqalpha 整数编码(YYYYMMDDHHMMSS×100, 见
    data_source.dt_to_int), //1000000 取 YYYYMMDD; 排除当日bar, 保证
    live(面板只到T-1)与1d回测(1d bar含当日)同口径。"""
    try:
        bars = history_bars(code, 6, "1d", ["datetime", "close"])
        if bars is None or not len(bars):
            return None
        today = int(context.now.strftime("%Y%m%d"))
        closes = [float(b["close"]) for b in bars
                  if int(b["datetime"]) // 1000000 < today]
        if len(closes) < 5:
            return None
        return sum(closes[-5:]) / 5
    except Exception as e:
        logger.info(f"MA5现算失败 {code}: {e!r}")
        return None


def _bar_of(bar_dict, code):
    """安全取 bar: BarMap 的 `in` 只查 universe(常年为空, 实测坑),
    __getitem__ 才会按需向 data_source 取数(无数据返回 NaN BarObject)"""
    try:
        b = bar_dict[code]
    except Exception:
        return None
    v = b.last
    if v is None or v != v:          # NaN = 该票当轮无数据
        return None
    return b


def _last_of(bar_dict, code):
    b = _bar_of(bar_dict, code)
    return float(b.last) if b is not None else None


def _settle_pend(context):
    """在途限价单结算: 成交量增加→更新 tranche 状态; 单消失且未成交→撤除"""
    if not context.pend:
        return
    open_ids = {o.order_id for o in get_open_orders()}
    for code, pd_ in list(context.pend.items()):
        q = _pos_qty(context, code)
        if q > pd_["q_before"]:                      # (部分)成交
            pos = context.portfolio.positions[code]
            px = float(getattr(pos, "avg_price", 0) or pd_["lmt"])
            st = context.tr.get(code)
            if st is None:
                p = context.pool.get(code, {})
                st = {"n": 0, "last_px": px, "hi": px,
                      "atr": float(p.get("atr") or 0),
                      "ma5": float(p.get("ma5") or 0),
                      "qty_seen": 0, "closed": False}
                context.tr[code] = st
            st["n"] += 1
            st["last_px"] = px
            st["hi"] = max(st["hi"], px)
            st["qty_seen"] = q
            logger.info(f"成交 {code} 第{st['n']}笔 @{px:.2f} 数量+{q - pd_['q_before']}")
            del context.pend[code]
        elif pd_["oid"] not in open_ids:
            # 单已消失且未成交 = 被拒或日终失效。1d 回测下属常态:
            # 限价单只按当根日线 close 撮合, 收盘未回到限价内即失效
            # (1m 盘中才是真实挂单等回踩语义)
            del context.pend[code]


def _vwap_of(context, code):
    """当日VWAP(元/股) = Σamt/Σvol, 无分时(1d回测/票不在雷达)返回None。

    单位口径(实测): intraday_px 的 vol 是【手】(qmt口径, SPEC注释"股"不准),
    amt 是元 → amt/vol ≈ 价格×100, 按分时中位价做量纲归一; 归一后仍偏离
    中位价2倍以上的视为脏数据(qmt断连期 vol/amt 是脏字段, 见server.py
    同款处理), 返回 None 交由尾盘判定兜底。"""
    try:
        pts = ticai_intraday(code)
    except Exception:
        return None
    if not pts or len(pts) < 30:      # 数据太少(刚开盘/回测无分时)不判
        return None
    rows = [(float(p[1]), float(p[2]), float(p[3])) for p in pts
            if p[1] and p[2] and float(p[1]) > 0 and float(p[2]) > 0 and p[3]]
    if len(rows) < 30:
        return None
    vol = sum(r[1] for r in rows)
    amt = sum(r[2] for r in rows)
    if vol <= 0:
        return None
    vw = amt / vol
    pxs = sorted(r[0] for r in rows)
    med = pxs[len(pxs) // 2]
    if med > 0 and vw > 50 * med:      # vol为手 → 归一到股
        vw /= 100.0
    if med > 0 and not (0.5 * med <= vw <= 2 * med):
        return None                    # 脏数据不判
    return vw


def _sell_all(context, code, why):
    """清仓: 可卖部分市价卖, T+1锁定部分记 exit_open 次日开盘卖"""
    # 撤掉在途低吸挂单(否则清仓后挂单跨日成交 → 违背退出意图)
    pd_ = context.pend.pop(code, None)
    if pd_ is not None:
        for o in get_open_orders():
            if o.order_id == pd_["oid"]:
                cancel_order(o)
                break
    pos = context.portfolio.positions.get(code)
    locked = 0
    if pos is not None:
        locked = int(getattr(pos, "quantity", 0) or 0) - \
            int(getattr(pos, "sellable", 0) or 0)
        if pos.sellable > 0:
            order_target_percent(code, 0)
    st = context.tr.get(code)
    if st:
        st["closed"] = True
    if locked > 0:
        context.exit_open.add(code)
        logger.info(f"{why} {code} 卖出可卖部分, T+1锁定{locked}股次日开盘卖")
    else:
        context.tr.pop(code, None)
        logger.info(f"{why} {code} 清仓")


def handle_bar(context, bar_dict):
    now_hm = context.now.strftime("%H:%M")
    _settle_pend(context)

    # ---- ① exit_open: 开盘后无条件清仓 ----
    if context.exit_open and now_hm >= "09:30":
        for code in sorted(context.exit_open):
            if _pos_qty(context, code) <= 0:
                context.exit_open.discard(code)
                context.tr.pop(code, None)
                continue
            pos = context.portfolio.positions[code]
            if pos.sellable > 0:
                order_target_percent(code, 0)
                logger.info(f"开盘清仓(昨日破线/止盈残余) {code}")
            context.exit_open.discard(code)
            context.tr.pop(code, None)

    # ---- ② 记录当日开盘价 + 低开标记 ----
    for code in list(context.tr) + [c for c in context.pool
                                    if c not in context.tr]:
        if code in context.day_open:
            continue
        bar = _bar_of(bar_dict, code)
        if bar is None:
            continue
        o = float(bar.open) if bar.open == bar.open else None
        if o is None or o <= 0:
            continue
        context.day_open[code] = o
        ma5 = _ma5_of(context, code)
        if code in context.tr and ma5 and o < ma5 and \
                not context.tr[code].get("closed"):
            context.weak_gap.add(code)
            logger.info(f"低开标记 {code} 开盘{o:.2f} < MA5 {ma5:.2f}, "
                        f"{VWAP_AT}后看VWAP")

    # ---- ③ 低开票 VWAP 判定(10:00后, 每票一次) ----
    if now_hm >= VWAP_AT and context.weak_gap:
        for code in sorted(context.weak_gap & context.tr.keys()):
            if code in context.vwap_done:
                continue
            last = _last_of(bar_dict, code)
            if last is None:
                continue
            vw = _vwap_of(context, code)
            context.vwap_done.add(code)
            if vw is None:
                logger.info(f"VWAP降级(无分时) {code}: 交由尾盘收盘判")
                continue
            if last < vw:
                _sell_all(context, code,
                          f"低开弱于VWAP({last:.2f}<{vw:.2f}) 清仓")
            else:
                logger.info(f"低开但站上VWAP({last:.2f}≥{vw:.2f}) 续持 {code}")

    # ---- ④ ATR 追踪止盈(浮盈 + 最高回撤≥ATR_MULT×ATR) ----
    for code, pos in list(context.portfolio.positions.items()):
        if int(getattr(pos, "quantity", 0) or 0) <= 0:
            continue
        st = context.tr.get(code)
        last = _last_of(bar_dict, code)
        if st is None or last is None or st.get("closed"):
            continue
        st["hi"] = max(st["hi"], last)
        avg = float(getattr(pos, "avg_price", 0) or st["last_px"])
        atr = st.get("atr") or 0
        if atr > 0 and last > avg and st["hi"] - last >= ATR_MULT * atr:
            _sell_all(context, code,
                      f"ATR追踪止盈(高{st['hi']:.2f}回撤≥{ATR_MULT}×ATR"
                      f"={atr:.2f}) @{last:.2f}")

    # ---- ⑤ 低吸买入(新票开仓 / 持仓加仓) ----
    held = sum(1 for c, p in context.portfolio.positions.items()
               if int(getattr(p, "quantity", 0) or 0) > 0)
    new_pend = sum(1 for c in context.pend if c not in context.tr)
    slots = MAX_POS - held - new_pend
    for code, p in context.pool.items():
        if code in context.pend or code in context.bought_today:
            continue
        st = context.tr.get(code)
        if st is not None and (st.get("closed") or st["n"] >= N_TRANCHES):
            continue
        if st is None and slots <= 0:
            continue
        if code in context.exit_open:
            continue
        bar = _bar_of(bar_dict, code)
        if bar is None:
            continue
        last = float(bar.last)
        low = float(bar.low) if bar.low == bar.low else None
        if low is None:
            continue
        ma5 = float(p["ma5"])
        tp = ma5 * (1 + DIP_BAND)           # 触发/限价
        floor = ma5 * (1 - BREAK_TOL)       # 破线不接
        if last < floor:
            continue
        touched = low <= tp                 # 20s bar 或 1d bar 的 low 触碰
        if not touched:
            continue
        if st is not None:
            # 加仓: 须较上笔成交价再低 ≥TRANCHE_STEP
            if last > st["last_px"] * (1 - TRANCHE_STEP):
                continue
            tp = min(tp, st["last_px"] * (1 - TRANCHE_STEP))
            tp = max(tp, floor)
        budget = context.portfolio.total_value / MAX_POS
        cash_t = min(budget / N_TRANCHES, context.portfolio.cash)
        lmt = round(tp, 2)
        qty = int(cash_t / lmt / 100) * 100
        if qty < 100:
            continue
        # q_before 必须在下单【前】记录: 1d 回测下 order_shares 对可成交
        # 限价单同步即时撮合, 下单后再取持仓量会把本次成交算进基线,
        # 导致 settle 永远检测不到成交(实测踩坑)
        q_before = _pos_qty(context, code)
        od = order_shares(code, qty, LimitOrder(lmt))
        if od is None:
            continue
        context.pend[code] = {"oid": od.order_id, "q_before": q_before,
                              "lmt": lmt}
        context.bought_today.add(code)
        if st is None:
            slots -= 1
        logger.info(f"低吸挂单 {code} {p.get('name', '')} "
                    f"第{(st['n'] if st else 0) + 1}笔 限价{lmt:.2f} "
                    f"(MA5={ma5:.2f} 现价{last:.2f})")


def tail_check(context, bar_dict):
    """14:55 尾盘判定: 收盘价跌破 MA5×(1-BREAK_TOL) → 清仓; 未破续持"""
    # 先结算在途单: 1d 频率下 scheduler(14:55) 先于 handle_bar 执行,
    # 不先结算则 _sell_all 会把未被检测的成交挂单直接撤除(实测踩坑)
    _settle_pend(context)
    for code, pos in list(context.portfolio.positions.items()):
        if int(getattr(pos, "quantity", 0) or 0) <= 0:
            continue
        st = context.tr.get(code)
        if st is not None and st.get("closed"):
            continue                       # 止盈残余, 次日开盘卖
        last = _last_of(bar_dict, code)
        if last is None:                   # 20s快照缺席 → 日线兜底
            try:
                b = history_bars(code, 1, "1d", ["close"])
                last = float(b[-1]["close"]) if b is not None and len(b) else None
            except Exception:
                last = None
        ma5 = _ma5_of(context, code) or _ma5_recalc(context, code)
        if last is None or ma5 is None:
            logger.info(f"尾盘判定数据缺失 {code} (last={last} ma5={ma5}), 默认续持")
            continue
        if last < ma5 * (1 - BREAK_TOL):
            _sell_all(context, code,
                      f"尾盘破5日线({last:.2f}<{ma5 * (1 - BREAK_TOL):.2f})")
        else:
            logger.info(f"尾盘续持 {code} {last:.2f} ≥ MA5×{1 - BREAK_TOL:.2f}"
                        f"={ma5 * (1 - BREAK_TOL):.2f}")


def after_trading(context):
    n = sum(1 for p in context.portfolio.positions.values()
            if int(getattr(p, "quantity", 0) or 0) > 0)
    logger.info(f"收盘: 持仓{n}只 当日挂单{len(context.bought_today)}票 "
                f"次日开盘待清{len(context.exit_open)}")
