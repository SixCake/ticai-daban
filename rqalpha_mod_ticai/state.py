# -*- coding: utf-8 -*-
"""策略模拟状态落盘 — 看板「策略模拟/策略回测」详情的数据源

落盘(全部在 run_dir = data/sim/runs/{run_id}/ 下, 回测与模拟同构):
  {run_dir}/state/{date}.json   每 cycle 覆盖写(盘中实时快照)
  {run_dir}/equity.parquet      每日结算追加(跨日净值序列)

为什么每 cycle 都写: 看板需要盘中实时观察(持仓/挂单/净值随时间变化),
且进程被杀后能从当日 state 恢复展示(账户本身由 rqalpha 的 persist 机制
恢复, state 文件只是展示层快照, 不参与账户恢复)。

净值曲线(equity_curve)记录当日每个 cycle 的净值, 供看板画当日曲线;
跨日曲线从 equity parquet 取。
"""
import json
import time
from pathlib import Path

from rqalpha.core.events import EVENT

from . import metrics
from .codes import from_rq

CURVE_STEP_SEC = 60            # 净值曲线采样间隔(按模拟时间, 非墙钟)
WRITE_MIN_INTERVAL = 5.0       # 落盘限频(墙钟秒, 防小文件频繁写)


def state_path(run_dir, date: str) -> Path:
    return Path(run_dir) / "state" / f"{date}.json"


def load_state(run_dir, date: str) -> dict:
    """读当日状态; 缺失返回空 dict(看板容错)"""
    p = state_path(run_dir, date)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


class StateRecorder:
    """盘中状态落盘 + 收盘净值结算"""

    def __init__(self, env, strategy: str, run_dir, fill_gate=None, api=None):
        self._env = env
        self.strategy = strategy
        self._run_dir = Path(run_dir)
        self._gate = fill_gate
        self._api = api
        self._curve: list = []           # [[ts, equity], ...] 当日净值曲线
        self._curve_day: str = ""        # 当前曲线归属日(跳日时重置)
        self._last_curve_sec = -10 ** 9  # 上次采样点的模拟时刻(秒)
        self._siglog: list = []          # 策略信号日志(由策略通过 log_signal 追加)
        self._last_write = 0.0
        self._last_write_sim_sec = -10 ** 9  # 上次落盘的模拟时刻(秒)

    def install(self, env=None):
        env = env or self._env
        env.event_bus.add_listener(EVENT.POST_BAR, self.on_bar)
        env.event_bus.add_listener(EVENT.POST_AFTER_TRADING, self.on_after)
        env.event_bus.add_listener(EVENT.POST_OPEN_AUCTION, self.on_bar)

    def log_signal(self, item: dict):
        """策略侧记录自定义信号日志(供看板展示触发原因链)"""
        self._siglog.append(item)
        if len(self._siglog) > 500:
            self._siglog = self._siglog[-500:]

    # ---------- 采集 ----------

    def _positions(self) -> list:
        out = []
        try:
            acc = self._env.portfolio.stock_account
        except Exception:
            return out
        for pos in acc.get_positions():
            qty = int(pos.quantity or 0)
            if qty <= 0:
                continue
            avg = float(pos.avg_price or 0.0)
            last = float(pos.last_price or avg)
            out.append({
                "code": pos.order_book_id,
                "ts_code": from_rq(pos.order_book_id),
                "name": getattr(pos.instrument, "symbol", "") or "",
                "qty": qty,
                "avg_price": round(avg, 3),
                "last_price": round(last, 3),
                "market_value": round(float(pos.market_value or 0.0), 2),
                "pnl": round(float(pos.pnl or 0.0), 2),
                "pnl_pct": round((last / avg - 1) * 100, 3) if avg > 0 else None,
                # stock_account.get_positions() 返回真实 Position(有 closable);
                # 而 portfolio.positions 返回 StockPositionProxy(叫 sellable)。
                # 两者都兼容, 取不到则返 None(不伪造 0)
                "closable": int(getattr(pos, "closable",
                                        getattr(pos, "sellable", 0)) or 0),
            })
        return out

    def _orders(self) -> list:
        out = []
        broker = getattr(self._env, "broker", None)
        if broker is None:
            return out
        try:
            open_orders = broker.get_open_orders()
        except Exception:
            return out
        items = open_orders.values() if hasattr(open_orders, "values") \
            else open_orders
        for o in items:
            out.append({
                "code": o.order_book_id,
                "ts_code": from_rq(o.order_book_id),
                "side": str(getattr(o.side, "name", o.side)),
                "qty": int(o.quantity or 0),
                "filled": int(o.filled_quantity or 0),
                "price": (round(float(o.price), 3)
                          if o.price == o.price else None),
                "status": str(getattr(o.status, "name", o.status)),
            })
        return out

    def _update_curve(self) -> None:
        """每 cycle 更新当日净值曲线(内存操作, 不限频)。
        限频按【模拟时间】: 回放模式全天在几十秒墙钟内跑完,
        若按墙钟限频会把绝大多数 cycle 跳掉 → 曲线只剩 2 个点。

        跳日必须重置: 多日回放时若不重置, 曲线会跨日累积并撞上
        MAX_CURVE_POINTS 上限 → "当日净值曲线"里混进了前几日的点。"""
        pf = getattr(self._env, "portfolio", None)
        if pf is None:
            return
        equity = float(pf.total_value or 0.0)
        dt = self._env.calendar_dt
        day = dt.strftime("%Y%m%d")
        ts = dt.strftime("%H:%M:%S")
        if day != self._curve_day:          # 新交易日 → 曲线归零
            self._curve_day = day
            self._curve = []
            self._last_curve_sec = -10 ** 9
        cur_sec = dt.hour * 3600 + dt.minute * 60 + dt.second
        # 节流: 距上次采样 ≥CURVE_STEP_SEC 才加点。注意不能写成
        # "ts 不同 OR 60s 已过" —— 雷达 cycle 间隔 20s, 每轮 ts 都不同,
        # OR 会让限频完全失效(实测 0902 那天累到 1517 点)。
        if (not self._curve
                or cur_sec - self._last_curve_sec >= CURVE_STEP_SEC):
            self._curve.append([ts, round(equity, 2)])
            self._last_curve_sec = cur_sec

    def _snapshot(self) -> dict:
        pf = getattr(self._env, "portfolio", None)
        if pf is None:
            # 系统启动失败/未走到初始化 → 无法快照(不伪造 0 净值)
            raise RuntimeError("Environment.portfolio 尚未初始化")
        equity = float(pf.total_value or 0.0)   # 总权益(portfolio_value 已废弃)
        cash = float(pf.cash or 0.0)
        frozen = float(getattr(pf, "frozen_cash", 0.0) or 0.0)
        mv = float(pf.market_value or 0.0)
        start_cash = float(pf.starting_cash or equity or 0.0)
        day = self._env.calendar_dt.strftime("%Y%m%d")
        ts = self._env.calendar_dt.strftime("%H:%M:%S")
        # 基准净值(供看板对齐展示) — 基准 ID 从 api 实例取, 不从
        # config.base.benchmark 取(故意不设它, 避免 sys_analyser 因
        # 基准数据不覆盖回测区间而抛异常终止 run)
        bench = None
        bm_id = getattr(self._api, "benchmark_id", None) if self._api else None
        try:
            if bm_id:
                ds = getattr(self._env.data_proxy, "_data_source", None)
                if ds is not None and hasattr(ds, "benchmark_close"):
                    bench = ds.benchmark_close(bm_id, self._env.calendar_dt)
                    if bench is not None:
                        bench = round(bench, 2)
        except Exception:
            bench = None
        eq_df = metrics.load_equity(self._run_dir / "equity.parquet")
        return {
            "date": day, "strategy": self.strategy, "ts": ts,
            "equity": round(equity, 2), "cash": round(cash, 2),
            "frozen_cash": round(frozen, 2),
            "market_value": round(mv, 2),
            "start_cash": round(start_cash, 2),
            "day_pnl": round(float(pf.daily_pnl or 0.0), 2),
            "total_pnl": round(equity - start_cash, 2),
            "total_pnl_pct": (round((equity / start_cash - 1) * 100, 3)
                              if start_cash > 0 else None),
            "benchmark": bench,
            "benchmark_id": bm_id,
            "positions": self._positions(),
            "orders": self._orders(),
            "equity_curve": self._curve,
            "signals": self._siglog,
            "fill_sim": self._gate.stats() if self._gate else None,
            # 被成交概率闸门抽掉的订单(模拟涨停排队买不进)。
            # 供看板"交易详情"展示为已拒单行, 而不是当成一个莫名其妙的
            # 卡片指标("16过/28抽掉"那种呈现是错的, 用户已指出)。
            "fill_sim_skipped": (list(self._gate.skipped)
                                 if self._gate else []),
            "metrics": metrics.compute_metrics(eq_df),
        }

    # ---------- 事件回调 ----------

    def on_bar(self, event):
        """每 cycle: 曲线总是更新(内存, 便宜); 文件落盘限频按墙钟
        (≥WRITE_MIN_INTERVAL 秒)避免小文件频繁写。

        但 catchup 快进会把全天 bar 在几秒墙钟内跑完, 纯墙钟限频会
        把整段补跑压掉一次落盘都没有(午休无新 bar 时看板就看到
        竞价旧快照)。故叠加模拟时间闸: 距上次落盘的模拟时刻
        ≥30min 强制写一次 → 补跑尾态(约每半小时一个快照)可见。"""
        self._update_curve()
        dt = getattr(self._env, "calendar_dt", None)
        sim_sec = (dt.hour * 3600 + dt.minute * 60 + dt.second) if dt else 0
        now = time.time()
        force = (sim_sec - self._last_write_sim_sec) >= 1800
        if not force and now - self._last_write < WRITE_MIN_INTERVAL:
            return
        self._last_write = now
        self._last_write_sim_sec = sim_sec
        try:
            snap = self._snapshot()
            p = state_path(self._run_dir, snap["date"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(snap, ensure_ascii=False),
                         encoding="utf-8")
        except Exception as e:
            print(f"[ticai] 状态落盘失败: {e}")

    def on_after(self, event):
        """收盘: 写最终 state; 仅非 1d 频率追加当日净值到跨日序列。
        1d 回测是独立实验, 其净值不混入 live/1m 的跨日记录(否则不同
        频率的结果揉在同一条收益曲线里, 污染整体口径)。"""
        try:
            snap = self._snapshot()
            p = state_path(self._run_dir, snap["date"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(snap, ensure_ascii=False),
                         encoding="utf-8")
            freq = getattr(self._env.config.base, "frequency", "1m")
            if freq != "1d":
                metrics.append_equity(self._run_dir / "equity.parquet", {
                    "trade_date": snap["date"],
                    "equity": snap["equity"],
                    "cash": snap["cash"],
                    "position_value": snap["market_value"],
                    "benchmark": snap["benchmark"],
                })
            print(f"[ticai] {self.strategy} 结算 {snap['date']} "
                  f"净值 {snap['equity']:.2f} "
                  f"当日盈亏 {snap['day_pnl']:+.2f} "
                  f"持仓 {len(snap['positions'])}"
                  + ("" if freq != "1d"
                     else " (1d实验: 不追加跨日净值, 收run时由 pkl 写本run记录)"))
        except Exception as e:
            print(f"[ticai] 结算失败: {e}")
