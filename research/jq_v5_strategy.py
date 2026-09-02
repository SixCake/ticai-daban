# -*- coding: utf-8 -*-
"""聚宽策略: V5 两层架构打板(研究22/23/24/24b 定稿移植, 分钟级回测)

架构(与 ticai-daban core/structure.py 同口径):
  结构层(盘前, T-1 日线): g_chip 结构闸 — 4项中≥3项健康:
    ① 行业内涨幅排名>3.5(非板块前3, 研究22证伪龙头稀缺性)
    ② 近20日炸板(触板未封)≤1次
    ③ 昨日量比∈(0.55, 2.2]甜蜜区
    ④ 连跌≥3日(反抽票次日质量优)
  盘中层(触发, 沿用研究14-21验证规则):
    G组 09:35: 高开稳封相/剧震;  L组 每分钟: 首触+1%且颠簸高
  融合排序: 同分钟多触发按 v5 = 0.5*g_intra + 0.25*g_eco + 0.25*g_chip
    降序占仓(最多3仓, 每仓1/3); 早盘信号天然优先(先到先占)

对比开关(多方案并行回测):
  MODE='v1' — 现有方案近似: 无结构闸, 触发即挂单(先到先得)
  MODE='v5' — 最终形态: 结构闸过滤 + 融合分排序
  FILL_SIM=0.30 — 涨停买入成交概率约束(用户指定), 1.0=不模拟

回测建议: 2025-09-01~2026-08-25(研究同窗), 初始资金100万, 频率每分钟;
三段行情分段复核: 偏多/震荡/偏空各跑一遍对比曲线。
出场沿用研究口径: 回落5%止损; 14:55未封板清仓; 封板续持至次日14:55。
"""
from jqdata import *
import numpy as np
import pandas as pd
import random

# ============ 参数 ============
MODE = 'v5'            # 'v5'=结构闸+融合排序 / 'v1'=现有方案对照
FILL_SIM = 0.30        # 买入成交概率模拟(用户约束); 1.0=关闭
MAXPOS = 3
GAP_BIG = 5.2
ODIP_MAX = 0.05
AMP3_MIN = 4.3
PV_TH = 0.93           # L组颠簸阈值(1min口径)
R3_TH = 4.8            # 暴拉阈值(近3分钟涨幅)
OVR_TH = 5.0
DEPTH_G = 0.005
DEPTH_L = 0.004
STOP = 0.05
OPEN_MIN = 60
ORDER_LIFE = 120       # 挂单存活秒数: 高挂限价本应即时成交, 超时未成交=一字板等买不进, 撤单释放现金占仓
PX_BATCH = 600         # 1m横截面单批票数(过小慢, 过大静默返空)
L_SCAN_GAP = 1         # 触板全池扫描间隔(分钟), 1=每根bar扫描
L_SCAN_END = '1030'    # 10:30后只监控持仓(止损/清仓/封板续持), 不再新买;
                       # 依据: 时段效应 ≤10:00信号次日胜率82.4%/盈亏比3.09,
                       # >10:00 降至 75.5%/1.59
ZB_WIN = 20            # 炸板疤痕窗口
GATE_MIN = 3           # 结构闸: 4项中≥3项健康
# ============ 优化开关(并行回测用: 逐个打开对比, 同窗同种子) ============
OPT_A = True           # 次日开盘保护: 持仓票次日首根bar低开≤OPEN_CUT直接卖(不等14:55)
OPEN_CUT = -3.0        # 开盘保护阈值(%)
OPT_B = True           # 触发线上移: +1% → +3%(过滤弱启动)
TOUCH_PCT = 3.0 if OPT_B else 1.0
OPT_C = True           # 高位/换手/市值过滤(实盘同款防接飞刀)
RISE20_MAX = 20.0      # 近20日累计涨幅上限(%)
TURNOVER_MAX = 20.0    # 前日换手率上限(%)
MCAP_MIN, MCAP_MAX = 5.0, 300.0   # 流通市值窗口(亿)
ORDER_SLIP = 0.005     # 高挂限价单容差: 限价=触发价×(1+容差), 涨停价封顶;
                       # 限价≥现价时聚宽以现价即时成交(保成交+价格上界保护)
TOUCH_K = 1 + TOUCH_PCT / 100.0
RNG = random.Random(42)
STAT_ZERO = {'cand': 0, 'gate': 0, 'sig': 0, 'sim_skip': 0, 'order': 0,
             'g_hasbar': 0, 'g_open1': 0, 'g_ampmax': 0.0,
             'touch': 0, 'touch_late': 0, 'pvmax': 0.0,
             'up1_max': 0, 'px_med': 0.0}


def stat_of(key):
    """g.stat 安全访问: 键缺失自动补零(防 prepare 未执行/旧状态覆盖)"""
    if not hasattr(g, 'stat'):
        g.stat = dict(STAT_ZERO)
    return g.stat.setdefault(key, 0)


def initialize(context):
    set_benchmark('000001.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0003,
                             close_commission=0.0003,
                             min_commission=5), type='stock')
    set_slippage(FixedSlippage(0.002))
    g.cand = {}           # code -> {pre, g_chip, gate, v5_base, items}
    g.state = {}          # code -> {jd, ep, hi}
    g.g_done = set()
    g.l_done = set()
    g.trades_today = set()
    g.pending = {}
    g.stat = dict(STAT_ZERO)
    run_daily(prepare, '09:05')
    run_daily(g_signals, '09:35')
    run_daily(intraday, 'every_bar')
    run_daily(endofday, '14:55')
    run_daily(summary, '14:58')
    log.info('=== V5策略启动 === MODE=%s FILL_SIM=%.2f MAXPOS=%d | '
             '优化: A次日开盘保护(%+.1f%%)=%s B触发线%.1f%%=%s C高位换手市值过滤=%s | '
             '盘中: PV_TH=%.2f R3_TH=%.1f GAP_BIG=%.1f AMP3_MIN=%.1f '
             'ODIP_MAX=%.2f | 结构: ZB_WIN=%d GATE_MIN=%d/4 '
             '甜蜜区(0.55,2.2] 连跌≥3 | 执行: 触发即市价买入 止损%.0f%% | '
             '扫描: 每%dm一次 %s后停扫 | 出场: 14:55未封清仓 封板续持' % (
                 MODE, FILL_SIM, MAXPOS,
                 OPEN_CUT, OPT_A, TOUCH_PCT, OPT_B, OPT_C,
                 PV_TH, R3_TH, GAP_BIG, AMP3_MIN,
                 ODIP_MAX, ZB_WIN, GATE_MIN, STOP * 100,
                 L_SCAN_GAP, L_SCAN_END))


# ---------- 结构层(盘前, 只用T-1数据) ----------

def prepare(context):
    dt = context.current_dt.date()
    prev = get_trade_days(end_date=dt, count=2)[0]
    g.state = {c: s for c, s in g.state.items()
               if c in context.portfolio.positions}
    g.g_done, g.l_done, g.trades_today, g.pending = set(), set(), set(), {}
    g.stat = dict(STAT_ZERO)
    codes = get_index_stocks('000985.XSHG')      # 中证全指
    codes = [c for c in codes
             if c[:3] in ('600', '601', '603', '605', '000', '001',
                          '002', '003')]           # 仅主板10cm
    st = get_extras('is_st', codes, end_date=prev, count=1).iloc[0]
    codes = [c for c in codes if not st.get(c, True)]
    if not codes:
        return
    d22 = get_price(codes, end_date=prev, count=ZB_WIN + 2,
                    frequency='daily',
                    fields=['close', 'high', 'low', 'volume'],
                    panel=False)
    cnt = get_price(codes, end_date=prev, count=OPEN_MIN,
                    frequency='daily', fields=['close'], panel=False)
    nbar = cnt.groupby('code').size()
    # 行业映射(聚宽申万一级, 点时口径; 批量调用, 返回{code:{...}})
    try:
        ind_raw = get_industry(codes, date=prev) or {}
    except Exception as e:
        log.info('行业映射拉取失败(降级无行业): %s' % str(e)[:80])
        ind_raw = {}
    ind_of = {}
    for c in codes:
        info = ind_raw.get(c, {}) if isinstance(ind_raw, dict) else {}
        sw = info.get('sw_l1') or {}
        ind_of[c] = sw.get('industry_code')
    # ---- 逐票 T-1 结构因子 ----
    feats = {}
    for c in codes:
        sub = d22[d22['code'] == c]
        if len(sub) < ZB_WIN + 1 or nbar.get(c, 0) < OPEN_MIN:
            continue
        cl = sub['close'].values.astype(float)
        hi = sub['high'].values.astype(float)
        vo = sub['volume'].values.astype(float)
        pre = cl[-1]
        if pre <= 0:
            continue
        # ② 炸板疤痕: 近20日触板未封(涨停价=前收×1.1近似, 主板)
        zb = 0
        for i in range(len(cl) - ZB_WIN, len(cl)):
            lp = round(cl[i - 1] * 1.1, 2)
            if hi[i] >= lp * 0.999 and cl[i] < lp * 0.999:
                zb += 1
        # ③ 昨日量比甜蜜区
        vma5 = float(vo[-6:-1].mean()) if len(vo) >= 6 else 0.0
        volr5 = vo[-1] / vma5 if vma5 > 0 else None
        # ④ 连跌天数
        streak = 0
        for i in range(len(cl) - 1, 0, -1):
            if cl[i] < cl[i - 1]:
                streak += 1
            else:
                break
        feats[c] = {
            'pre': pre,
            'y_cpos': float((cl[-1] - sub['low'].iloc[-1]) /
                            (sub['high'].iloc[-1] - sub['low'].iloc[-1]))
            if sub['high'].iloc[-1] > sub['low'].iloc[-1] else 0.5,
            'avg_min_vol': float(vo[-5:].mean()) / 240.0,
            'zb': zb, 'volr5': volr5, 'neg': streak,
            'y_pct': (cl[-1] / cl[-2] - 1) * 100 if cl[-2] > 0 else 0.0,
            'rise20': ((cl[-1] / cl[-21] - 1) * 100
                       if len(cl) >= 21 and cl[-21] > 0 else None),
            'ind': ind_of.get(c),
        }
    # ---- OPT_C: 高位/换手/市值过滤(实盘同款防接飞刀) ----
    if OPT_C and feats:
        val = {}
        try:
            vdf = get_valuation(list(feats.keys()), end_date=prev, count=1,
                                fields=['code', 'turnover_ratio',
                                        'circulating_market_cap'])
            if vdf is not None and len(vdf):
                for r in vdf.itertuples():
                    val[r.code] = (r.turnover_ratio,
                                    r.circulating_market_cap)
        except Exception as e:
            log.info('估值拉取失败(降级仅高位过滤): %s' % str(e)[:60])
        d_hi = d_to = d_mc = 0
        for c in list(feats.keys()):
            f = feats[c]
            if f['rise20'] is not None and f['rise20'] > RISE20_MAX:
                d_hi += 1
                del feats[c]
                continue
            tv, mc = val.get(c, (None, None))
            if tv is not None and tv > TURNOVER_MAX:
                d_to += 1
                del feats[c]
                continue
            if mc is not None and not (MCAP_MIN <= mc <= MCAP_MAX):
                d_mc += 1
                del feats[c]
        log.info('OPT_C过滤: 高位(>%.0f%%)=%d 换手(>%.0f%%)=%d '
                 '市值(%.0f~%.0f亿外)=%d 剩余%d' % (
                     RISE20_MAX, d_hi, TURNOVER_MAX, d_to,
                     MCAP_MIN, MCAP_MAX, d_mc, len(feats)))
    g.stat['cand'] = len(feats)
    # ---- 行业昨日统计(候选池内近似口径) ----
    ind_ctx = {}
    by_ind = {}
    for c, f in feats.items():
        if f['ind']:
            by_ind.setdefault(f['ind'], []).append((c, f['y_pct']))
    for ind, pairs in by_ind.items():
        if len(pairs) < 5:
            continue
        ranked = sorted(pairs, key=lambda x: -x[1])
        ind_ctx[ind] = {
            'breadth': sum(1 for _, p in pairs if p > 0) / len(pairs),
            'rank': {c: i + 1 for i, (c, _) in enumerate(ranked)},
        }
    # ---- g_chip 结构闸 + v5_base ----
    zb_all = sorted(f['zb'] for f in feats.values())
    br_all = sorted(ctx['breadth'] for ctx in ind_ctx.values()) or [0.5]

    def pct_of(grid, v):
        return float(np.searchsorted(grid, v, side='right')) / len(grid)

    for c, f in feats.items():
        ctx = ind_ctx.get(f['ind'])
        sweet = f['volr5'] is not None and 0.55 < f['volr5'] <= 2.2
        items = []
        if ctx:
            items.append(ctx['rank'].get(c, 999) > 3.5)
        items.append(f['zb'] <= 1.5)
        items.append(sweet)
        items.append(f['neg'] >= 2.5)
        g_chip = sum(items) / 4.0
        eco = [pct_of(zb_all, f['zb']), 1.0 if sweet else 0.0]
        if ctx:
            eco += [pct_of(br_all, ctx['breadth']),
                    1.0 - pct_of(zb_all, ctx['rank'].get(c, 999))]
        f['g_chip'] = round(g_chip, 3)
        f['gate'] = sum(items) >= GATE_MIN
        f['items'] = items
        f['v5_base'] = round(0.25 * sum(eco) / len(eco)
                             + 0.25 * g_chip, 4)
        g.stat['gate'] += 1 if f['gate'] else 0
        g.cand[c] = f
    if MODE == 'v5':
        g.cand = {c: f for c, f in g.cand.items() if f['gate']}
    # ---- 详细日志: 闸门逐项统计 + 入池明细 ----
    n_all = len(feats)
    if n_all:
        c_ok = {'rank': 0, 'zb': 0, 'sweet': 0, 'neg': 0, 'noind': 0}
        for c, f in feats.items():
            ctx = ind_ctx.get(f['ind']) if f['ind'] else None
            if ctx is None:
                c_ok['noind'] += 1
            elif ctx['rank'].get(c, 999) > 3.5:
                c_ok['rank'] += 1
            if f['zb'] <= 1.5:
                c_ok['zb'] += 1
            if f['volr5'] is not None and 0.55 < f['volr5'] <= 2.2:
                c_ok['sweet'] += 1
            if f['neg'] >= 2.5:
                c_ok['neg'] += 1
        log.info('闸门逐项通过数(共%d): 排名非前3=%d 无行业ctx=%d '
                 '炸板≤1=%d 量比甜蜜=%d 连跌≥3=%d' % (
                     n_all, c_ok['rank'], c_ok['noind'], c_ok['zb'],
                     c_ok['sweet'], c_ok['neg']))
    log.info('[%s] 候选%d 闸通过%d 入池%d' % (
        MODE, g.stat['cand'], g.stat['gate'], len(g.cand)))
    if g.cand:
        lines = []
        for c, f in sorted(g.cand.items(),
                           key=lambda kv: -kv[1]['v5_base'])[:40]:
            ctx = ind_ctx.get(f['ind']) if f['ind'] else None
            lines.append(
                '  %s gchip=%.2f v5b=%.3f zb=%d vr5=%s neg=%d '
                'rank=%s breadth=%s' % (
                    c, f['g_chip'], f['v5_base'], f['zb'],
                    '%.2f' % f['volr5'] if f['volr5'] is not None else '-',
                    f['neg'],
                    ctx['rank'].get(c, '-') if ctx else '-',
                    '%.2f' % ctx['breadth'] if ctx else '-'))
        log.info('入池明细(前40, 按v5_base降序):\n' + '\n'.join(lines))


def limit_price_of(code, pre):
    return round(pre * 1.1, 2)


def get_px_1m(codes, end_dt, count, fields=('close', 'high', 'low', 'volume')):
    """分钟bar分批拉取(聚宽单次全市场横截面会静默返空, 分批口径)"""
    frames = []
    for i in range(0, len(codes), PX_BATCH):
        part = codes[i:i + PX_BATCH]
        try:
            r = get_price(part, end_date=end_dt, count=count,
                          frequency='1m', fields=list(fields),
                          panel=False)
            if r is not None and len(r):
                frames.append(r)
        except Exception as e:
            log.info('1m拉取批次失败 %d-%d: %s' % (
                i, i + PX_BATCH, str(e)[:60]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_px(codes, now):
    """现价快照=最近1根1m收盘(回测环境 get_current_data 可能为空,
    全部现价以此为准); vol=0 视为停牌/无成交。返回 (px, vol)"""
    f = get_px_1m(codes, now, 1, fields=('close', 'volume'))
    px, vol = {}, {}
    if len(f):
        last = f.groupby('code').tail(1)
        for r in last.itertuples():
            px[r.code] = float(r.close)
            vol[r.code] = float(r.volume)
    return px, vol


def v5_of(c, g_intra):
    """完整V5 = v5_base + 0.5*g_intra(盘中因子归一)"""
    return g.cand[c]['v5_base'] + 0.5 * min(max(g_intra, 0.0), 1.0)


def try_buy(context, picks):
    """picks=[(code, dprice, depth, why, v5)] 按v5降序占仓。
    执行口径: 触发即买入——高挂限价单: 限价 = min(触发价×(1+ORDER_SLIP),
    涨停价)。限价≥现价时聚宽以现价即时成交(效果同市价), 但成交价
    有上界: 防暴拉瞬间异常高价, 涨停价封顶(实盘扫板同款挂法)。
    FILL_SIM 成交概率约束在下单前抽样(模拟涨停买入难度)。
    在途挂单占仓: 未成交的挂单(一字板等)同样占仓, 超时由 intraday 撤单释放。"""
    slots = MAXPOS - len(context.portfolio.positions) - len(g.pending)
    picks = [p for p in picks
             if p[0] not in context.portfolio.positions
             and p[0] not in g.pending]
    if MODE == 'v5':
        picks.sort(key=lambda p: -p[4])         # 融合分降序(先到先得变成分数优先)
    for c, dprice, depth, why, v5 in picks:
        if slots <= 0:
            break
        g.stat['sig'] += 1
        if RNG.random() >= FILL_SIM:             # 成交概率模拟
            g.stat['sim_skip'] += 1
            log.info('成交模拟放弃 %s [%s] v5=%.3f' % (c, why, v5))
            continue
        cash_per = min(context.portfolio.total_value / MAXPOS,
                       context.portfolio.available_cash)
        amt = int(cash_per / dprice / 100) * 100
        if amt < 100:
            log.info('现金不足一手 跳过 %s (可用%.0f 需%.0f)' % (
                c, context.portfolio.available_cash, dprice * 100))
            continue
        # 高挂限价: 触发价×(1+容差), 涨停价封顶; 限价≥现价→即时成交
        pre_b = g.cand.get(c, {}).get('pre')
        lmt = dprice * (1 + ORDER_SLIP)
        if pre_b:
            lmt = min(lmt, round(pre_b * 1.1, 2))
        lmt = round(lmt, 2)
        o = order(c, amt, LimitOrderStyle(lmt))
        if o is None:
            continue
        g.pending[c] = o.order_id       # 在途占仓; 成交/撤单后由intraday清理
        if o.filled > 0:
            del g.pending[c]            # 即时成交: 转入持仓, 不重复占仓
            g.state[c] = {'jd': context.current_dt.date(), 'ep': dprice,
                          'hi': dprice}
            g.trades_today.add(c)
            g.stat['order'] += 1
            slots -= 1
            log.info('买入成交 %s [%s] 触发%.2f 限价%.2f 数量%d v5=%.3f' % (
                c, why, dprice, lmt, o.filled, v5))
        else:
            log.info('下单未成交 %s [%s] (涨停/停牌等)' % (c, why))


# ---------- 盘中层(触发规则沿用研究14-21) ----------

def g_signals(context):
    """09:35: G组高开信号"""
    if not g.cand:
        return
    codes = list(g.cand.keys())
    px = get_px_1m(codes, context.current_dt, 4)
    if len(px):
        g.stat['px_med'] = len(px)
    px0, vol0 = fetch_px(codes, context.current_dt)
    picks = []
    n_hasbar = n_open1 = 0
    amp_max = 0.0
    for c in codes:
        info = g.cand[c]
        pre = info['pre']
        sub = px[px['code'] == c]
        if len(sub) < 4 or vol0.get(c, 0.0) <= 0:
            continue
        n_hasbar += 1
        close = sub['close'].values.astype(float)
        pct = (close / pre - 1) * 100
        if pct[0] < 1.0:
            continue
        n_open1 += 1
        g.g_done.add(c)
        hi3 = float(sub['high'].max()) / pre * 100
        lo3 = float(sub['low'].min()) / pre * 100
        gap = pct[3]
        odip = hi3 - gap
        amp3 = hi3 - lo3
        amp_max = max(amp_max, amp3)
        why = None
        if gap > GAP_BIG and odip <= ODIP_MAX:
            why = '稳封相'
        elif gap <= GAP_BIG and amp3 > AMP3_MIN:
            why = '剧震'
        if why is None:
            continue
        g_intra = min(amp3 / (2 * AMP3_MIN), 1.0)    # 振幅归一近似
        picks.append((c, float(close[3]), DEPTH_G, why,
                      v5_of(c, g_intra)))
    g.stat['g_hasbar'] = n_hasbar
    g.stat['g_open1'] = n_open1
    g.stat['g_ampmax'] = round(amp_max, 2)
    log.info('G组漏斗: 有分钟bar %d/%d(拉取行数%.0f) | 高开≥1%% %d | 最大振幅%.2f'
             '(阈值%.1f) | 信号 %d' % (
                 n_hasbar, len(codes), stat_of('px_med'), n_open1,
                 amp_max, AMP3_MIN, len(picks)))
    try_buy(context, picks)


def intraday(context):
    """每分钟: 回落止损 / L组触板检测(触发即买, 无挂单管理)"""
    now = context.current_dt
    positions = context.portfolio.positions
    # ---- 首bar调试心跳: 定位 现价/pre 口径问题 ----
    if getattr(g, '_dbg_date', None) != now.date():
        g._dbg_date = now.date()
        px_dbg0, _ = fetch_px(list(g.cand.keys())[:5], now)
        sample = ' | '.join(
            '%s last=%.2f pre=%.2f' % (c, px_dbg0.get(c, -1.0), info['pre'])
            for c, info in list(g.cand.items())[:5])
        px_rows = get_px_1m(list(g.cand.keys())[:20], now, 3,
                            fields=('close',))
        log.info('DEBUG首bar %s: 入池=%d | 样例[%s] | '
                 '1m拉取行数=%d' % (
                     now.strftime('%H:%M'), len(g.cand), sample,
                     len(px_rows)))
    # ---- OPT_A: 次日开盘保护(昨日持仓低开≤OPEN_CUT 首根bar直接卖;
    #      针对T+1导致当日止损失效后的隔夜跳空敞口) ----
    if getattr(g, '_ocut_date', None) != now.date():
        g._ocut_date = now.date()
        if OPT_A and positions:
            px_oc, _ = fetch_px(list(positions.keys()), now)
            for c in list(positions.keys()):
                if c in g.trades_today:
                    continue
                pre_oc = g.cand.get(c, {}).get('pre')
                if pre_oc is None:
                    prev_d = get_trade_days(end_date=now.date(),
                                            count=2)[0]
                    y = get_price(c, end_date=prev_d, count=1,
                                  frequency='daily', fields=['close'],
                                  panel=False)
                    pre_oc = float(y['close'].iloc[-1]) if len(y) else None
                p_oc = px_oc.get(c, 0.0)
                if pre_oc and p_oc > 0 and \
                        (p_oc / pre_oc - 1) * 100 <= OPEN_CUT:
                    order_target(c, 0)
                    g.state.pop(c, None)
                    log.info('次日开盘保护卖出 %s 开盘%+.1f%%(阈值%.1f%%)'
                             ' @%.2f' % (
                                 c, (p_oc / pre_oc - 1) * 100,
                                 OPEN_CUT, p_oc))
    # ---- 挂单管理: 超时未成交=一字板等买不进, 撤单释放现金/占仓;
    #      已成交转持仓, 已撤未成交释放(不再当日重试同一票) ----
    for oid, o in get_open_orders().items():
        if (now - o.add_time).total_seconds() > ORDER_LIFE:
            cancel_order(o)
    open_ids = set(get_open_orders().keys())
    for c, oid in list(g.pending.items()):
        if c in positions:
            del g.pending[c]                # 已成交 → 持仓管理
        elif oid not in open_ids:
            g.trades_today.discard(c)
            del g.pending[c]                # 已撤 → 释放现金与占仓
    px_hold, _ = fetch_px(list(positions.keys()), now) if positions else ({}, {})
    for c in list(positions.keys()):
        st = g.state.get(c)
        price = px_hold.get(c, 0.0)
        if price <= 0 or st is None:
            continue
        if c in g.trades_today:
            continue                    # T+1: 当日买入不可卖, 止损顺延至次日(OPEN_CUT保护)
        st['hi'] = max(st['hi'], price)
        ref = max(st['ep'], st['hi'])
        if price <= ref * (1 - STOP):
            order_target(c, 0)
            g.state.pop(c, None)
            log.info('止损 %s @%.2f' % (c, price))
    # L组: 首触+1%检测 → 颠簸高+暴拉双因子排序
    if not g.cand:
        return
    hm = now.strftime('%H%M')
    if hm > L_SCAN_END:
        return
    if (now.hour * 60 + now.minute) % L_SCAN_GAP:
        return
    touch = []
    n_up1 = 0
    px_now, vol_now = fetch_px(list(g.cand.keys()), now)
    for c, info in g.cand.items():
        if c in positions:
            continue
        price = px_now.get(c, 0.0)
        if price <= 0 or vol_now.get(c, 0.0) <= 0:
            continue                    # 无成交/停牌跳过
        if price >= info['pre'] * TOUCH_K:
            n_up1 += 1
            if c not in g.g_done and c not in g.l_done:
                touch.append(c)
        g.stat['up1_max'] = max(stat_of('up1_max'), n_up1)
    if not touch:
        return
    g.stat['touch'] += len(touch)
    px = get_px_1m(touch, now, 15, fields=('close',))
    picks = []
    for c in touch:
        info = g.cand[c]
        pre = info['pre']
        sub = px[px['code'] == c]
        if len(sub) < 11:
            # 早盘bar不足: 不标记, 下一分钟重判(修复: 旧版早触票被永久跳过)
            continue
        g.l_done.add(c)                     # 触板时刻只判一次(≥11bar后)
        g.stat['touch_late'] += 1
        seg = (sub['close'].values.astype(float) / pre - 1) * 100
        seg = seg[-11:]
        pv = float(np.diff(seg).std())
        g.stat['pvmax'] = max(g.stat['pvmax'], pv)
        if pv <= PV_TH:
            continue
        r3 = float(seg[-1] - seg[-4]) if len(seg) >= 4 else 0.0
        g_intra = (min(pv / (2 * PV_TH), 1.0)
                   + min(max(r3, 0) / (2 * R3_TH), 1.0)) / 2
        why = 'L颠簸' + ('+暴拉' if r3 >= R3_TH else '')
        picks.append((c, float(sub['close'].iloc[-1]), DEPTH_L, why,
                      v5_of(c, g_intra)))
    try_buy(context, picks)


def endofday(context):
    """14:55: 非今日买入持仓, 未封板清仓; 封板续持"""
    today = context.current_dt.date()
    px_eod, _ = fetch_px(list(context.portfolio.positions.keys()),
                         context.current_dt)
    for c in list(context.portfolio.positions.keys()):
        if c in g.trades_today:
            continue
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
        price = px_eod.get(c, 0.0)
        sealed = pre is not None and price >= limit_price_of(c, pre) * 0.995
        if not sealed:
            order_target(c, 0)
            g.state.pop(c, None)
        else:
            st['jd'] = today


def summary(context):
    s = g.stat
    log.info('日终[%s] 候选%d 闸通过%d 信号%d 模拟放弃%d 实挂%d' % (
        MODE, s['cand'], s['gate'], s['sig'], s['sim_skip'],
        s['order']))
    log.info('漏斗诊断: G组[有bar%d 高开≥1%%共%d 最大振幅%.2f] '
             'L组[触板人次%d 已判定%d 最大pv%.2f(阈值%.2f)] '
             '盘中同时≥%.0f%%峰值%d只' % (
                 s['g_hasbar'], s['g_open1'], s['g_ampmax'],
                 s['touch'], s['touch_late'], s['pvmax'], PV_TH,
                 TOUCH_PCT, s['up1_max']))
    if context.portfolio.positions:
        px_sum, _ = fetch_px(list(context.portfolio.positions.keys()),
                             context.current_dt)
        lines = []
        for c, pos in context.portfolio.positions.items():
            price = px_sum.get(c, pos.avg_cost)
            ret = (price / pos.avg_cost - 1) * 100 if pos.avg_cost else 0
            sealed = ''
            pre = g.cand.get(c, {}).get('pre')
            if pre and price >= limit_price_of(c, pre) * 0.995:
                sealed = ' [已封板→续持]'
            lines.append('  %s 成本%.2f 现价%.2f 盈亏%+.2f%%%s' % (
                c, pos.avg_cost, price, ret, sealed))
        log.info('持仓明细(%d):\n' % len(lines) + '\n'.join(lines))
