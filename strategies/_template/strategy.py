# -*- coding: utf-8 -*-
"""策略模板 — 四段钩子骨架（复制本目录开始新策略）

规范见同目录 SPEC.md。硬约束提醒:
  · 禁止 import 其它策略目录 / 项目的 core/ quotes/ apps/
  · 禁止硬编码 data/ 路径, 数据只经注入 API 取
  · AI 只经 ai_feed(name), 且 name 须在 config.yaml 的 feeds 里声明
  · 代码口径一律 rqalpha 格式(000001.XSHE), API 已转换好
"""
from rqalpha.api import *

# ---------- 可调参数(改这里, 不要改框架) ----------
MAX_POS = 3               # 最大持仓数
STOP_LOSS = 0.05          # 回落止损(参考价=max(买入价,当日最高))
CLEAR_UNSEALED = "14:55"  # 未封板清仓时刻


def init(context):
    """框架初始化(只跑一次)。set_benchmark 只能在这里调。"""
    # 基准: DBBNCH.XSHG=自建打板基准(全A等权); 宽基需先跑
    # collect/fetch_index_panel.py 采集后改用 000300.XSHG 等
    set_benchmark("DBBNCH.XSHG")
    context.candidates = []       # 盘前选出的候选池(rqalpha 口径代码)
    context.entry = {}            # code -> {ep 买入价, hi 当日最高}
    context.traded = set()        # 当日已买(防重复下单)


def before_trading(context):
    """盘前(09:00): 用 T-1 结构因子选股, 构建候选池"""
    struct = ticai_struct()
    # V5 结构闸: gate=True 表示 4 项中 ≥3 项健康(见 core/structure.py)
    passed = {c: s for c, s in struct.items() if s.get("gate")}
    # 按 v5 融合分降序取前 N
    ranked = sorted(passed.items(), key=lambda kv: -(kv[1].get("v5") or 0))
    context.candidates = [c for c, _ in ranked[:40]]
    context.entry = {}
    context.traded = set()

    # 订阅的 AI feed(时间戳闸门已由框架施加, 无需自己过滤)
    # narrative = ai_feed("theme_narrative")
    # if narrative:
    #     logger.info(f"题材叙事强度: {[(e['topic'], e['score']) for e in narrative]}")
    # logbook 的 logger 用 {} 风格而非 % 风格, 统一用 f-string
    logger.info(f"盘前候选池 {len(context.candidates)} 只")


def open_auction(context, bar_dict):
    """竞价(09:25): 签同 handle_bar 为两参数(context, bar_dict)。
    bar_dict 由 rqalpha executor 自动挂上(竞价价走 get_open_auction_bar)。
    本模板留空, 按需实现(竞价量爆筛选/竞价挂单)。"""
    pass


def handle_bar(context, bar_dict):
    """盘中(每 20s 一轮): 信号触发买入 + 持仓管理"""
    # ---- 持仓管理: 回落止损 ----
    for code, pos in list(context.portfolio.positions.items()):
        st = context.entry.get(code)
        if st is None:
            continue
        last = bar_dict[code].last if code in bar_dict else None
        if last is None or last != last:      # NaN 判定
            continue
        st["hi"] = max(st["hi"], float(last))
        ref = max(st["ep"], st["hi"])
        if float(last) <= ref * (1 - STOP_LOSS):
            order_target_percent(code, 0)
            context.entry.pop(code, None)
            logger.info(f"止损 {code} @{last:.2f} (参考{ref:.2f})")

    # ---- 买入: S2/S3 信号触发 ----
    slots = MAX_POS - len(context.portfolio.positions)
    if slots <= 0:
        return
    signals = [s for s in ticai_signals()
               if s["stage"] in ("S2", "S3")
               and s["code"] in context.candidates
               and s["code"] not in context.traded
               and s["code"] not in context.portfolio.positions]
    if not signals:
        return
    # 按 V5 融合分降序占仓(同分钟多触发时分数优先, 而非先到先得)
    struct = ticai_struct()
    signals.sort(key=lambda s: -(struct.get(s["code"], {}).get("v5") or 0))
    for s in signals[:slots]:
        code = s["code"]
        cash_per = min(context.portfolio.total_value / MAX_POS,
                       context.portfolio.cash)   # cash = 可用资金
        px = float(s.get("price0") or 0)
        if px <= 0:
            continue
        qty = int(cash_per / px / 100) * 100
        if qty < 100:
            continue
        # 高挂限价: 触发价×(1+0.005), 涨停价封顶(实盘扫板同款挂法)
        from rqalpha_mod_ticai.broker import limit_price_of
        pre = bar_dict[code].prev_close if code in bar_dict else None
        lmt = limit_price_of(px, float(pre) if pre == pre else None)
        # rqalpha 的限价单类叫 LimitOrder(聚宽叫 LimitOrderStyle)
        order_shares(code, qty, LimitOrder(lmt))
        context.traded.add(code)
        context.entry[code] = {"ep": px, "hi": px}
        logger.info(f"买入 {code} [{s.get('why')}] 触发{px:.2f} "
                    f"限价{lmt:.2f} 数量{qty}")


def after_trading(context):
    """收盘(15:30): 未封板清仓(T+1 下当日买入不可卖, 由框架保证)"""
    # 注: portfolio.positions 里是 StockPositionProxy, T+1 可卖量字段叫
    # sellable(= 持仓 - 今日买入 - 已冻结), 不叫 closable
    for code, pos in list(context.portfolio.positions.items()):
        if pos.sellable <= 0:
            continue                     # 当日买入, T+1 不可卖
        if code in context.traded:
            continue                     # 当日买入的不动
        order_target_percent(code, 0)
        context.entry.pop(code, None)
        logger.info(f"收盘清仓 {code}")
