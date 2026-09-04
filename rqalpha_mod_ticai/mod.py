# -*- coding: utf-8 -*-
"""TicaiMod — 把项目数据源/信号/执行口径接入 rqalpha

start_up 做五件事(顺序有依赖, 不可调换):
  1. set_data_source(TicaiDataSource)    覆盖 sys_simulation 的 bundle 数据源
  2. set_event_source(Live|Replay)       覆盖 sys_simulation 的历史回测事件源
     (broker 与 price_board 保留 sys_simulation 的 — 涨跌停拒单/无量撤单/
      成交量限制/T+1 都由它原生提供, 我们不重复造)
  3. FillSimGate.install                 叠加成交概率抽样(rqalpha 无此概念)
  4. api.install                         注入策略取数 API(须在策略加载前)
  5. StateRecorder.install               盘中状态落盘 + 收盘净值结算
"""
from rqalpha.interface import AbstractMod

from . import api
from .broker import FillSimGate
from .data_source import TicaiDataSource
from .event_source import TicaiLiveEventSource, TicaiReplayEventSource
from .state import StateRecorder


class TicaiMod(AbstractMod):

    def __init__(self):
        self._ds = None
        self._gate = None
        self._recorder = None
        self._api = None

    def start_up(self, env, mod_config):
        mode = str(getattr(mod_config, "mode", "replay")).lower()
        strategy = str(getattr(mod_config, "strategy", "") or "")
        feeds_allowed = list(getattr(mod_config, "feeds", []) or [])
        # run 记录目录: 回测与模拟同构(data/sim/runs/{run_id}/), 未指定时
        # 落默认主模拟目录, 兼容直接 CLI 跑。
        from pathlib import Path
        from config import DATA
        run_dir = Path(getattr(mod_config, "run_dir", "") or
                       (DATA / "sim" / "runs" /
                        f"{strategy or 'default'}__main"))

        # 1. 数据源(本地 parquet + 雷达盘中快照, 不用米筐 bundle)
        self._ds = TicaiDataSource(
            risk_free_rate=float(getattr(mod_config, "risk_free_rate", 0.015)))
        env.set_data_source(self._ds)

        # 2. 事件源: live(轮询 radar.json) 或 replay(读 intraday_px 全量时刻)
        if mode == "live":
            es = TicaiLiveEventSource(
                env, self._ds,
                poll_interval=int(getattr(mod_config, "poll_interval", 20)),
                catchup=bool(getattr(mod_config, "catchup", True)))
        else:
            es = TicaiReplayEventSource(env, self._ds)
        env.set_event_source(es)

        # 3. 成交概率抽样闸门(rqalpha 撮合是确定性的, 无排队成交概率概念)
        self._gate = FillSimGate(
            fill_sim=float(getattr(mod_config, "fill_sim", 0.30)),
            seed=int(getattr(mod_config, "fill_seed", 42)))
        self._gate.install(env)

        # 4. 注入 API(必须在策略文件被加载前完成 — export_as_api 写入
        #    rqalpha.api.__all__, 策略的 `from rqalpha.api import *` 才可见)
        self._api = api.install(env, strategy, feeds_allowed)

        # 5. 状态落盘与结算
        self._recorder = StateRecorder(env, strategy or "default", run_dir,
                                       fill_gate=self._gate, api=self._api)
        self._recorder.install(env)

        print(f"[ticai] Mod 就绪 mode={mode} strategy={strategy or '-'} "
              f"feeds={feeds_allowed or '-'} "
              f"fill_sim={self._gate.fill_sim} run_dir={run_dir.name}")

    def tear_down(self, code, exception=None):
        # 收盘结算已由 POST_AFTER_TRADING 触发; 此处只兜底(异常退出时
        # 未走到 after_trading, 补写一次 state 便于看板排查)
        if self._recorder is not None:
            try:
                self._recorder.on_bar(None)
            except Exception:
                pass
        if self._gate is not None:
            print(f"[ticai] 成交概率闸门统计: {self._gate.stats()}")
