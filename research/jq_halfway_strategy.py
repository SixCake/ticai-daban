# -*- coding: utf-8 -*-
"""聚宽策略: 半路涨停前向信号体系 v2(ticai-daban 研究14-21 移植)

回测建议: 2025-09-01 ~ 2026-08-25, 初始资金100万, 频率每分钟

信号(决策时刻只用之前信息, 无前视):
  S3-G组(高开≥1%, 09:35第4根bar后判定一次):
    稳封相: gap>5.2 且 开盘4bar回撤odip≤0.05 (昨收强>0.6加标记)
    剧震:   gap≤5.2 且 开盘4bar振幅amp3>4.3
    竞价量爆: 首bar量/近5日均分钟量≥5 (辅助标记)
  S2-L组(非高开, 盘中每分钟检测首触+1%):
    颠簸高: 触板前11根bar轨迹波动pathvol>0.93
  仅主板10cm(60x/00x), 排ST/停牌/次新(<60日K线)

执行(研究17/18验证: 回踩无逆向选择, 挂单等回踩EV更高):
  G组限价低于决策价0.5点 / L组低于触发价0.4点, 10分钟未成交撤单
仓位: 最多3仓, 每仓1/3总资产
出场: 盘中回落5%止损(参考价=max(入场,当日最高));
      14:55未封板清仓; 封板续持至次日14:55(封板再续)
"""
from jqdata import *
import numpy as np
import pandas as pd

# ============ 参数(研究结论) ============
MAXPOS = 3
GAP_BIG = 5.2        # S3 高开大幅阈值
ODIP_MAX = 0.05      # 稳封相开盘回撤上限
AMP3_MIN = 4.3       # 剧震振幅下限
PV_TH = 0.93         # L组颠簸阈值(1min口径)
OVR_TH = 5.0         # 竞价量爆阈值
DEPTH_G = 0.005      # G组挂单深度(0.5点)
DEPTH_L = 0.004      # L组挂单深度(0.4点)
STOP = 0.05          # 盘中回落止损
OPEN_MIN = 60        # 上市天数下限
ORDER_LIFE = 600     # 挂单存活秒数


def initialize(context):
    set_benchmark('000001.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0003,
                             close_commission=0.0003,
                             min_commission=5), type='stock')
    set_slippage(FixedSlippage(0.002))
    g.cand = {}           # code -> {pre, y_cpos, avg_min_vol}
    g.state = {}          # code -> {jd, ep, hi}
    g.g_done = set()      # 高开票(09:35已评估G组)
    g.l_done = set()      # 已评估过L触板的票
    g.trades_today = set()
    g.pending = {}        # code -> order_id
    run_daily(prepare, '09:05')
    run_daily(g_signals, '09:35')
    run_daily(intraday, 'every_bar')
    run_daily(endofday, '14:55')


def prepare(context):
    """盘前: 候选池 + 昨日静态特征(全部截至上一交易日, 无前视)"""
    dt = context.current_dt.date()
    prev = get_trade_days(end_date=dt, count=2)[0]
    g.state = {c: s for c, s in g.state.items()
               if c in context.portfolio.positions}
    g.g_done = set()
    g.l_done = set()
    g.trades_today = set()
    g.pending = {}
    codes = get_index_stocks('000985.XSHG')
    codes = [c for c in codes
             if c[:3] in ('600', '601', '603', '605', '000', '001',
                          '002', '003')]
    st = get_extras('is_st', codes, end_date=prev, count=1).iloc[0]
    codes = [c for c in codes if not st.get(c, True)]
    if not codes:
        return
    d1 = get_price(codes, end_date=prev, count=6, frequency='daily',
                   fields=['close', 'high', 'low', 'volume'],
                   panel=False)
    cnt = get_price(codes, end_date=prev, count=OPEN_MIN,
                    frequency='daily', fields=['close'], panel=False)
    nbar = cnt.groupby('code').size()
    g.cand = {}
    for c in codes:
        sub = d1[d1['code'] == c]
        if len(sub) < 6 or nbar.get(c, 0) < OPEN_MIN:
            continue
        ph = float(sub['high'].iloc[-1])
        pl = float(sub['low'].iloc[-1])
        pc = float(sub['close'].iloc[-1])
        if pc <= 0:
            continue
        g.cand[c] = {
            'pre': pc,
            'y_cpos': (pc - pl) / (ph - pl) if ph > pl else 0.5,
            'avg_min_vol': float(sub['volume'].iloc[-5:].mean()) / 240.0,
        }
    log.info('候选池 %d 只' % len(g.cand))


def limit_price_of(code, pre):
    return round(pre * 1.1, 2)


def place_limit(context, c, dprice, depth, why):
    """限价挂单(低于决策价depth), 下单即建状态; 返回是否下单成功"""
    lmt = round(dprice * (1 - depth), 2)
    cash_per = min(context.portfolio.total_value / MAXPOS,
                   context.portfolio.available_cash)
    amt = int(cash_per / lmt / 100) * 100
    if amt < 100:
        return False
    o = order(c, amt, LimitOrderStyle(lmt))
    if o is None:
        return False
    g.state[c] = {'jd': context.current_dt.date(), 'ep': lmt, 'hi': lmt}
    g.trades_today.add(c)
    g.pending[c] = o.order_id
    log.info('挂单 %s [%s] 限价%.2f 数量%d' % (c, why, lmt, amt))
    return True


def g_signals(context):
    """09:35: G组高开信号(开盘4根bar后判定一次)"""
    if not g.cand:
        return
    codes = list(g.cand.keys())
    px = get_price(codes, end_date=context.current_dt, count=4,
                   frequency='1m',
                   fields=['close', 'high', 'low', 'volume'],
                   panel=False)
    cur = get_current_data()
    slots = MAXPOS - len(context.portfolio.positions)
    n_sig = 0
    for c in codes:
        info = g.cand[c]
        pre = info['pre']
        sub = px[px['code'] == c]
        if c not in cur or cur[c].paused or len(sub) < 4:
            continue
        close = sub['close'].values.astype(float)
        high = sub['high'].values.astype(float)
        low = sub['low'].values.astype(float)
        vol = sub['volume'].values.astype(float)
        pct = (close / pre - 1) * 100
        if pct[0] < 1.0:
            continue                        # 非高开 → 留给盘中L组检测
        g.g_done.add(c)
        gap = pct[3]
        hi3 = float(high.max()) / pre * 100
        lo3 = float(low.min()) / pre * 100
        odip = hi3 - pct[3]
        amp3 = hi3 - lo3
        why = None
        if gap > GAP_BIG and odip <= ODIP_MAX:
            why = '稳封相' + ('+昨收强' if info['y_cpos'] > 0.6 else '')
        elif gap <= GAP_BIG and amp3 > AMP3_MIN:
            why = '剧震'
        if why is None:
            continue
        ovr = vol[0] / info['avg_min_vol'] if info['avg_min_vol'] > 0 else 0
        if ovr >= OVR_TH:
            why += '+量爆'
        n_sig += 1
        if slots > 0 and c not in context.portfolio.positions:
            if place_limit(context, c, float(close[3]), DEPTH_G, why):
                slots -= 1
    log.info('G组信号 %d 个' % n_sig)


def intraday(context):
    """每分钟: 撤超时挂单/清理未成交状态/回落止损/L组触板检测"""
    now = context.current_dt
    positions = context.portfolio.positions
    # 1. 超10分钟未成交挂单撤单
    for oid, o in get_open_orders().items():
        if (now - o.add_time).total_seconds() > ORDER_LIFE:
            cancel_order(o)
    # 2. pending状态同步: 已成交→保留state; 已撤未成交→清除
    open_ids = set(get_open_orders().keys())
    for c, oid in list(g.pending.items()):
        if c in positions:
            del g.pending[c]                # 已成交
        elif oid not in open_ids:
            g.state.pop(c, None)            # 已撤未成交
            g.trades_today.discard(c)
            del g.pending[c]
    # 3. 回落5%止损(参考价=max(入场,当日最高))
    cur = get_current_data()
    for c in list(positions.keys()):
        st = g.state.get(c)
        price = cur[c].last_price if c in cur else 0
        if price <= 0 or st is None:
            continue
        st['hi'] = max(st['hi'], price)
        ref = max(st['ep'], st['hi'])
        if price <= ref * (1 - STOP):
            order_target(c, 0)
            g.state.pop(c, None)
            log.info('止损 %s @%.2f (参考%.2f)' % (c, price, ref))
    # 4. L组: 非高开票首触+1%检测(每分钟, 触板时刻决策)
    slots = MAXPOS - len(positions)
    if slots <= 0 or not g.cand:
        return
    touch = []
    for c, info in g.cand.items():
        if (c in g.g_done or c in g.l_done or c in positions
                or c in g.pending):
            continue
        price = cur[c].last_price if c in cur else 0
        if price >= info['pre'] * 1.01:
            touch.append(c)
    if not touch:
        return
    px = get_price(touch, end_date=now, count=15, frequency='1m',
                   fields=['close'], panel=False)
    for c in touch:
        g.l_done.add(c)                     # 触板时刻只判一次
        info = g.cand[c]
        pre = info['pre']
        sub = px[px['code'] == c]
        if len(sub) < 11:
            continue
        seg = (sub['close'].values.astype(float) / pre - 1) * 100
        seg = seg[-11:]
        if float(np.diff(seg).std()) > PV_TH:
            if place_limit(context, c, float(sub['close'].iloc[-1]),
                           DEPTH_L, 'L颠簸高'):
                slots -= 1
                if slots <= 0:
                    break


def endofday(context):
    """14:55: 非今日买入的持仓, 未封板清仓; 封板续持"""
    cur = get_current_data()
    today = context.current_dt.date()
    for c in list(context.portfolio.positions.keys()):
        if c in g.trades_today:
            continue                        # 今日买入→明日处理
        st = g.state.get(c)
        if st is None:
            order_target(c, 0)
            continue
        pre = g.cand.get(c, {}).get('pre')
        if pre is None:
            prev_d = get_trade_days(end_date=today, count=2)[0]
            y = get_price(c, end_date=prev_d, count=1, frequency='daily',
                          fields=['close'], panel=False)
            pre = float(y['close'].iloc[-1]) if len(y) >= 1 else None
        price = cur[c].last_price if c in cur else 0
        sealed = pre is not None and price >= limit_price_of(c, pre) * 0.995
        if not sealed:
            order_target(c, 0)
            g.state.pop(c, None)
            log.info('尾盘清仓 %s @%.2f' % (c, price))
        else:
            st['jd'] = today
            log.info('封板续持 %s' % c)
