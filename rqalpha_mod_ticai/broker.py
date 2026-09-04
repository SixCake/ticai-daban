# -*- coding: utf-8 -*-
"""执行口径叠加 — 打板买入难度模拟(rqalpha 无对应概念的部分)

rqalpha sys_simulation 已原生提供的约束(本模块不重复实现, 只确认生效):
  price_limit=True(默认)  涨跌停拒单 — reaches_limit_up 判定
                          price >= limit_up - tick_size + tolerance
                          故一字板(价==涨停价)天然买不进;
                          限价单抛 OrderNotMatchable(挂单保留可重试),
                          市价单抛 OrderRejected
  inactive_limit=True     bar 无量撤单
  volume_limit=True       单笔不超过该 bar 成交量的 25%
  T+1                     股票账户原生保证(当日买入不可卖)

本模块只补 rqalpha 没有的东西:
  FILL_SIM  成交概率抽样 — 模拟涨停排队买不进的概率(研究定稿 0.30)。
            rqalpha 撮合是确定性的, 没有"排队成交概率"概念; 打板策略
            的真实成交率远低于撮合器假设, 不模拟会严重高估胜率。
            实现为 ORDER_PENDING_NEW 监听器: 抽样不通过则
            order.mark_rejected() + 发 ORDER_UNSOLICITED_UPDATE。
            只对买入抽样(卖出不受限)。

为何必须额外发 ORDER_UNSOLICITED_UPDATE(实测踩坑):
  Account 在 ORDER_PENDING_NEW 就冻结现金(_on_order_pending_new), 而
  解冻只监听 ORDER_UNSOLICITED_UPDATE / ORDER_CANCELLATION_PASS ——
  它不监听 ORDER_CREATION_REJECT。只调 mark_rejected() 而不发事件 →
  冻结现金永不释放(实测差额 334,635 元卡在 frozen_cash 里,
  cash 与 total_value 对不上)。做法与 simulation_broker.after_trading 一致。

为何必须延迟到 POST_SYSTEM_INIT 才注册(实测踩坑):
  rqalpha/main.py 里 mod_handler.start_up() 在前(147行),
  env.set_portfolio(Portfolio(...)) 在后(170行), POST_SYSTEM_INIT 更后
  (174行)。Account 的事件监听器在 Portfolio 构造时才注册, 若在
  Mod.start_up 里直接注册本闸门 → 本监听器排在 Account 之前执行,
  拒单时现金还没冻结 → 解冻事件变成空操作, 随后 Account 又把现金冻上
  → 永久冻结。故先听 POST_SYSTEM_INIT, 在其中再注册 ORDER_PENDING_NEW,
  保证排在 Account 之后。

高挂限价容差(ORDER_SLIP)是策略侧的挂价约定而非撮合约束, 故以工具函数
形式暴露给策略(limit_price_of), 不在撮合层拦截。
"""
import random

from rqalpha.const import SIDE
from rqalpha.core.events import EVENT, Event

DEFAULT_FILL_SIM = 0.30
DEFAULT_SEED = 42
DEFAULT_ORDER_SLIP = 0.005


class FillSimGate:
    """成交概率抽样闸门。种子固定 → 回测可复现(同序列同结果)。"""

    def __init__(self, fill_sim: float = DEFAULT_FILL_SIM,
                 seed: int = DEFAULT_SEED):
        self.fill_sim = float(fill_sim)
        self._rng = random.Random(seed)
        self._env = None
        self.n_pass = 0
        self.n_skip = 0
        self.skipped: list = []        # 被抽掉的订单摘要(供看板展示)

    def install(self, env):
        """fill_sim>=1.0 时不安装(关闭模拟)。
        先听 POST_SYSTEM_INIT, 在其中再注册 ORDER_PENDING_NEW ——
        因为 Account 的监听器在 Portfolio 构造时才注册, 而 Portfolio
        在所有 Mod.start_up 之后才创建(见 rqalpha/main.py)。"""
        if self.fill_sim >= 1.0:
            return
        self._env = env
        env.event_bus.add_listener(EVENT.POST_SYSTEM_INIT, self._late_install)

    def _late_install(self, event):
        self._env.event_bus.add_listener(EVENT.ORDER_PENDING_NEW, self._gate)

    def _gate(self, event):
        order = event.order
        if order.side != SIDE.BUY:
            return                     # 卖出不受成交概率约束
        if self._rng.random() < self.fill_sim:
            self.n_pass += 1
            return
        self.n_skip += 1
        self.skipped.append({
            "order_book_id": order.order_book_id,
            "quantity": int(order.quantity),
            "price": float(order.price) if order.price == order.price else None,
        })
        if len(self.skipped) > 500:
            self.skipped = self.skipped[-500:]
        reason = f"成交概率模拟放弃(FILL_SIM={self.fill_sim})"
        order.mark_rejected(reason)
        # 必须额外发事件释放冻结现金 — Account 不监听 ORDER_CREATION_REJECT,
        # 只监听 ORDER_UNSOLICITED_UPDATE/CANCELLATION_PASS 来解冻
        if self._env is not None:
            self._env.event_bus.publish_event(
                Event(EVENT.ORDER_UNSOLICITED_UPDATE,
                      account=event.account, order=order))

    def stats(self) -> dict:
        return {"fill_sim": self.fill_sim, "pass": self.n_pass,
                "skip": self.n_skip,
                "skip_rate": round(self.n_skip / max(1, self.n_pass
                                                     + self.n_skip), 3)}


def limit_price_of(trigger_px: float, pre_close: float | None,
                   slip: float = DEFAULT_ORDER_SLIP) -> float:
    """高挂限价单价格 = min(触发价×(1+slip), 涨停价)。

    研究定稿口径(与 research/jq_v5_strategy.py 一致): 限价≥现价时
    rqalpha 以现价即时成交(效果同市价), 但成交价有上界 —— 防暴拉瞬间
    异常高价, 涨停价封顶(实盘扫板同款挂法)。
    pre_close 缺失时不封顶(调用方应保证有昨收)。
    """
    px = round(trigger_px * (1 + slip), 2)
    if pre_close and pre_close > 0:
        # 涨停价档位由调用方传入的昨收推算; 此处按主板 10% 兜底,
        # 精确档位见 data_source._load_panel 的向量化口径
        px = min(px, round(pre_close * 1.1, 2))
    return px
