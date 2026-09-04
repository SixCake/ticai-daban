# -*- coding: utf-8 -*-
"""rqalpha_mod_ticai — ticai-daban 策略模拟框架的 rqalpha 扩展 Mod

职责边界(与项目现有分层一致):
  数据/领域逻辑  core/ (heat/prob/early_signal/structure/seesaw/times...)
  行情采集      quotes/ collect/
  本 Mod        只做"把项目数据与信号接入 rqalpha"的适配层, 不复制领域逻辑

对外契约(rqalpha 6.x Mod 规范):
  load_mod()   返回 AbstractMod 实例
  __config__   默认配置 dict, rqalpha 会与策略 __config__ 的 mod.ticai 合并

  注意: rqalpha 官方文档里的 `__mod_config__`(yaml 字符串) 是 2.x/3.x 的
  旧写法; 6.3.0 实际读的是模块级 `__config__` dict
  (见 rqalpha/mod/__init__.py ModHandler.set_env)。

priority=200: ModHandler 按 priority 升序 start_up, 系统 Mod 默认 100。
本 Mod 必须在 sys_simulation 之后运行 —— sys_simulation 的 start_up 会
set_data_source/set_event_source/set_broker/set_price_board, 我们要覆盖
前两者(数据源与事件源)但保留后两者(撮合器与价格板)。

启用方式(两种):
  ① pip install -e . 后 `rqalpha mod enable ticai`
  ② 策略 __config__ 里显式指定 lib(本项目用这种, 免打包):
     "mod": {"ticai": {"enabled": True, "lib": "rqalpha_mod_ticai", ...}}
"""
from .codes import from_rq, to_rq  # noqa: F401  (对外暴露代码映射)
from .feeds import cutoff_of, cutoff_now, read_feed  # noqa: F401


def load_mod():
    from .mod import TicaiMod
    return TicaiMod()


__config__ = {
    # 优先级: 必须 > 100 才能在 sys_simulation 之后覆盖数据源/事件源
    "priority": 200,

    # ---- 运行模式 ----
    # replay = 历史回放(读 intraday_px 全量快照时刻, 用于回测/复核)
    # live   = 盘中实时(轮询 radar.json, 用于当日模拟)
    "mode": "replay",
    "poll_interval": 20,      # live 模式轮询秒数(与雷达 INTERVAL 对齐)
    "catchup": True,          # live 模式盘中启动时补跑已过去的快照时刻

    # ---- 执行口径(阈值来自研究定稿, 改动须同步 research/jq_v5_strategy.py) ----
    "order_slip": 0.005,      # 高挂限价容差: 限价=触发价×(1+此值), 涨停价封顶
    "fill_sim": 0.30,         # 成交概率抽样(模拟涨停排队买不进); 1.0=关闭
    "fill_seed": 42,          # 抽样种子(保证回测可复现)
    # 涨跌停拒单与一字板拒买由 rqalpha sys_simulation 的 price_limit 原生
    # 提供(reaches_limit_up: price >= limit_up - tick_size + tolerance),
    # 本 Mod 不重复实现; 此两项仅作显式声明, 关闭需改 sys_simulation 配置
    "block_limit_up_buy": True,
    "block_yizi_buy": True,

    # ---- 策略身份与 AI feed 订阅 ----
    "strategy": "",           # 策略名(由 sim.py 传入; 用于落盘目录与私有feed)
    "feeds": [],              # 该策略声明可订阅的 feed 名白名单(隔离规范)

    # ---- 绩效口径 ----
    "risk_free_rate": 0.015,  # 常数年化无风险利率(供夏普/alpha/索提诺)

    # ---- 落盘 ----
    "state_dir": "sim",       # data/{state_dir}/ 为落盘根目录
}
