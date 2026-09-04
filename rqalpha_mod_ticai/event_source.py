# -*- coding: utf-8 -*-
"""事件源 — 盘中实时(Live) 与 历史回放(Replay) 两个实现

设计依据(ADR-0004): 两者读同一批落盘文件, 这是"同一份数据源"成立的唯一方式。
  回放  data/live/intraday_px_{date}.json 的快照时刻序列(全量, 一次性)
  盘中  data/live/radar.json 的 ts 字段(474KB/0.02s, 每 20s 轮询)
        + 同日的 intraday_px(mtime 守护增量重读)

为什么盘中不直接用 snapshot_times(): 该方法要扫 intraday_px 全部点
(最大分区实测 210 万点), 每 20s 轮询承受不了; 而 radar.json 是雷达每
cycle 覆盖写的小文件, 取当前 ts 只需 0.02s。

四段钩子的事件映射(rqalpha 原生支持全部四段, 无需自己造阶段):
  BEFORE_TRADING  → 策略 before_trading(context)     盘前
  OPEN_AUCTION    → 策略 open_auction(context)       竞价(09:25)
  BAR             → 策略 handle_bar(context, bar_dict) 盘中(每 20s 一轮)
  AFTER_TRADING   → 策略 after_trading(context)      收盘
"""
import json
import time
from datetime import date, datetime, time as dtime

from rqalpha.core.events import EVENT, Event
from rqalpha.interface import AbstractEventSource

from config import DATA

LIVE = DATA / "live"
# 事件时刻约定(与 sys_simulation 的相对偏移一致)
BEFORE_TRADING_DT = dtime(9, 0)
OPEN_AUCTION_DT = dtime(9, 25)
DAY_BAR_DT = dtime(15, 0)
AFTER_TRADING_DT = dtime(15, 30)
# 盘中轮询与收盘判定
POLL_INTERVAL = 20
MARKET_CLOSE_HHMM = "150100"       # 15:01 后停止发 BAR, 转 AFTER_TRADING
MARKET_OPEN_HHMM = "091500"


def _hhmmss_to_dt(d: date, hhmmss: str) -> datetime:
    h = str(hhmmss).zfill(6)
    return datetime(d.year, d.month, d.day,
                    int(h[:2]), int(h[2:4]), int(h[4:6]))


class _BaseEventSource(AbstractEventSource):
    """共用: 交易日序列 + 四段事件的构造"""

    def __init__(self, env, data_source):
        self._env = env
        self._ds = data_source

    def _trading_dates(self, start_date, end_date) -> list:
        from rqalpha.const import TRADING_CALENDAR_TYPE
        cal = self._ds.get_trading_calendars()[TRADING_CALENDAR_TYPE.CN_STOCK]
        s = start_date if isinstance(start_date, date) else start_date.date()
        e = end_date if isinstance(end_date, date) else end_date.date()
        return [d.date() for d in cal if s <= d.date() <= e]

    def _day_events(self, d: date, snapshot_times: list, frequency: str):
        """单日四段事件(不含盘中 BAR 循环, 由子类决定 BAR 来源)"""
        yield Event(EVENT.BEFORE_TRADING,
                    calendar_dt=datetime.combine(d, BEFORE_TRADING_DT),
                    trading_dt=datetime.combine(d, BEFORE_TRADING_DT))
        yield Event(EVENT.OPEN_AUCTION,
                    calendar_dt=datetime.combine(d, OPEN_AUCTION_DT),
                    trading_dt=datetime.combine(d, OPEN_AUCTION_DT))
        if frequency == "1d":
            # 日频回测: 单日一根 bar(同 sys_simulation 口径)
            dt = datetime.combine(d, DAY_BAR_DT)
            yield Event(EVENT.BAR, calendar_dt=dt, trading_dt=dt)
        else:
            for t in snapshot_times:
                dt = _hhmmss_to_dt(d, t)
                yield Event(EVENT.BAR, calendar_dt=dt, trading_dt=dt)
        dt_at = datetime.combine(d, AFTER_TRADING_DT)
        yield Event(EVENT.AFTER_TRADING, calendar_dt=dt_at, trading_dt=dt_at)


class TicaiReplayEventSource(_BaseEventSource):
    """历史回放: 读 intraday_px 的全量快照时刻驱动 BAR。

    与盘中模拟读同一批文件 → 回测结果与盘中模拟严格同口径。
    frequency='1d' 时退化为单日一根 bar(纯日线回测, 不需要盘中数据)。
    """

    def events(self, start_date, end_date, frequency):
        for d in self._trading_dates(start_date, end_date):
            if frequency == "1d":
                times: list = []
            else:
                day = d.strftime("%Y%m%d")
                times = self._ds.snapshot_times(day)
                if not times:
                    # 无盘中数据的日子跳过盘中事件, 只走盘前/竞价/收盘
                    # (回放窗口受限于 intraday_px 已有分区, 见 ADR-0004)
                    times = []
            yield from self._day_events(d, times, frequency)


class TicaiLiveEventSource(_BaseEventSource):
    """盘中实时: 轮询 radar.json 取当前快照时刻, 有新时刻则发 BAR。

    rqalpha 的事件循环是单线程消费生成器, 故在生成器内 sleep 等待新数据
    是安全的(不会阻塞其它线程 —— 本来就没有其它线程)。

    盘中启动补跑(catchup): 进程若在盘中启动(如 10:30), 先把已过去的快照
    时刻逐个补发 BAR, 再转实时轮询 —— 保证 handle_bar 看到完整当日轨迹,
    与回放行为一致。策略侧的信号感知不依赖补跑(ticai_signals() 读的是
    presig_state 当日累积文件, 天然含全天信号)。
    """

    def __init__(self, env, data_source, poll_interval: int = POLL_INTERVAL,
                 catchup: bool = True):
        super().__init__(env, data_source)
        self._poll = int(poll_interval)
        self._catchup = bool(catchup)

    def _radar_ts(self) -> str | None:
        """radar.json 的 ts(HH:MM:SS) → HHMMSS; 读失败返回 None"""
        f = LIVE / "radar.json"
        if not f.exists():
            return None
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
        ts = str(d.get("ts") or "")
        return ts.replace(":", "") if len(ts) == 8 else None

    def _wait_until(self, d: date, hhmm: str, poll: int = 10) -> bool:
        """阻塞等到当日 hhmm 时刻(或已过); 返回是否已到"""
        target = datetime.combine(
            d, dtime(int(hhmm[:2]), int(hhmm[2:4])))
        while datetime.now() < target:
            time.sleep(poll)
        return True

    def events(self, start_date, end_date, frequency):
        for d in self._trading_dates(start_date, end_date):
            day = d.strftime("%Y%m%d")
            now_day = date.today()
            if d > now_day:
                continue                     # 未来交易日不跑
            is_today = (d == now_day)

            # ---- 盘前 ----
            if is_today:
                self._wait_until(d, MARKET_OPEN_HHMM)
            yield Event(EVENT.BEFORE_TRADING,
                        calendar_dt=datetime.combine(d, BEFORE_TRADING_DT),
                        trading_dt=datetime.combine(d, BEFORE_TRADING_DT))

            # ---- 竞价(等到 09:25 且雷达已有竞价快照) ----
            if is_today:
                self._wait_until(d, "092500")
            yield Event(EVENT.OPEN_AUCTION,
                        calendar_dt=datetime.combine(d, OPEN_AUCTION_DT),
                        trading_dt=datetime.combine(d, OPEN_AUCTION_DT))

            # ---- 盘中 ----
            if frequency != "1d":
                emitted: set = set()
                # 补跑: 盘中启动时把已过去的快照时刻逐个补发
                if self._catchup:
                    cur = datetime.now().strftime("%H%M%S") \
                        if is_today else MARKET_CLOSE_HHMM
                    for t in self._ds.snapshot_times(day):
                        if t > cur:
                            break
                        emitted.add(t)
                        dt = _hhmmss_to_dt(d, t)
                        yield Event(EVENT.BAR, calendar_dt=dt, trading_dt=dt)
                # 实时轮询
                last_ts = None
                while True:
                    if is_today:
                        now_hms = datetime.now().strftime("%H%M%S")
                        if now_hms >= MARKET_CLOSE_HHMM:
                            break
                        time.sleep(self._poll)
                        self._ds.refresh_intraday(day)   # mtime 守护增量重读
                        ts = self._radar_ts()
                        if ts is None or ts == last_ts or ts in emitted:
                            continue
                        # 雷达 ts 可能落在两个快照点之间 → 取 ≤ts 的最新快照点
                        cand = [x for x in self._ds.snapshot_times(day)
                                if x <= ts and x not in emitted]
                        if not cand:
                            last_ts = ts
                            continue
                        for t in cand:
                            emitted.add(t)
                            dt = _hhmmss_to_dt(d, t)
                            yield Event(EVENT.BAR, calendar_dt=dt,
                                        trading_dt=dt)
                        last_ts = ts
                    else:
                        break                # 历史日: 补跑已完成, 转收盘

            # ---- 收盘 ----
            dt_at = datetime.combine(d, AFTER_TRADING_DT)
            yield Event(EVENT.AFTER_TRADING, calendar_dt=dt_at,
                        trading_dt=dt_at)
            if is_today:
                # 当日跑完即结束(盘中模拟是单日进程, 次日由 sim.py 重新拉起)
                return
