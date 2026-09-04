# -*- coding: utf-8 -*-
"""TicaiDataSource — rqalpha 数据源(本地 parquet + 雷达盘中快照)

设计依据(ADR-0001/0004):
  直接继承 AbstractDataSource 而非 BaseDataSource, 不依赖米筐 bundle。
  官方 rqalpha-mod-tushare 是 2017 demo(用已下线的 ts.get_k_data、pandas 2.0
  已移除的 as_matrix), 且仍甩回 BaseDataSource → 等于还是要 bundle。
  回测与盘中读同一批落盘文件, 保证"同一份数据源"严格成立。

数据映射:
  交易日历        data/meta/trade_cal.parquet
  Instruments     data/meta/qmt_names.json(名称) + daily_panel(上市/退市日推断)
  日线 bar        data/market/1d/daily_panel.parquet (tushare pro.daily 采集)
  盘中快照        data/live/intraday_px_{date}.json (雷达 20s 落盘)
  集合竞价        data/live/open_traj_{date}.json + intraday_px 首点
  指数基准        data/market/1d/index_panel.parquet (可选, 缺失则不提供)
  打板基准        DBBNCH.XSHG — 全A等权涨幅累积指数(由 daily_panel 自算)

涨跌停价不在 daily_panel 中, 由 pre_close × 档位现算(复用 quotes/qmt.py
的 _limit_ratio 口径), 因为 rqalpha 撮合器依赖 bar 里的 limit_up/limit_down
做价格夹取(原生并不拒绝涨停买入, 拒绝逻辑在 broker.py)。

代码口径: rqalpha 内部一律 .XSHE/.XSHG; 转换只在 codes.py 发生。
"""
import json
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from config import DATA
from datastore import path_of
from quotes.qmt import _limit_ratio

from .codes import TS2RQ, exchange_of, from_rq, to_rq

LIVE = DATA / "live"

# rqalpha DayBarStore 原生 dtype(7字段) — history_bars 必须能产出这些列
DAY_BAR_NAMES = ["datetime", "open", "close", "high", "low", "volume",
                 "total_turnover"]
# 本项目扩展字段: 撮合器要 limit_up/limit_down, BarObject 要 last/prev_close
FULL_BAR_DTYPE = np.dtype([
    ("datetime", np.int64),
    ("open", np.float64), ("close", np.float64),
    ("high", np.float64), ("low", np.float64),
    ("volume", np.float64), ("total_turnover", np.float64),
    ("limit_up", np.float64), ("limit_down", np.float64),
    ("prev_close", np.float64), ("last", np.float64),
])

# 打板基准(自建虚拟指数) — 全A等权涨幅累积, 起点1000
DBBNCH_ID = "DBBNCH.XSHG"
DBBNCH_BASE = 1000.0

# 交易时段(用于 get_trading_minutes_for 与竞价判定)
AM_START, AM_END = time(9, 30), time(11, 30)
PM_START, PM_END = time(13, 0), time(15, 0)
AUCTION_DT = time(9, 25)


def dt_to_int(dt) -> int:
    """rqalpha datetime 字段编码(同 utils.datetime_func.convert_dt_to_int)"""
    return (dt.year * 10000000000 + dt.month * 100000000 + dt.day * 1000000
            + dt.hour * 10000 + dt.minute * 100 + dt.second)


def _board_type(ts_code: str) -> str:
    """rqalpha board_type: MainBoard/GEM/KSH(BJS 已剔除)"""
    if ts_code.startswith("68"):
        return "KSH"
    if ts_code.startswith("30"):
        return "GEM"
    return "MainBoard"


class TicaiDataSource:
    """rqalpha AbstractDataSource 实现。

    刻意不继承 AbstractDataSource: 该抽象类的部分方法签名带 futures-only
    语义(get_settle_price/get_yield_curve 等), 继承后必须逐一实现或抛错,
    而 rqalpha 只在 data_proxy 调用到对应方法时才触发。改为鸭子类型:
    只实现股票链路真正会被调用的方法, 其余按需抛 NotImplementedError
    (rqalpha 的 data_proxy 对该异常有 forward-compatible 分支)。
    """

    def __init__(self, risk_free_rate: float = 0.015):
        # 常数年化无风险利率(供夏普/alpha/索提诺); 中国短端量级 1.5%
        self.risk_free_rate = float(risk_free_rate)
        self._cal: pd.DatetimeIndex | None = None
        self._names: dict = {}
        self._by_code: dict | None = None     # ts_code -> np.ndarray(FULL_BAR_DTYPE)
        self._panel_dates: pd.DatetimeIndex | None = None
        self._instruments: dict = {}          # order_book_id -> Instrument
        self._index_panel: dict | None = None  # index_id -> np.ndarray
        self._dbbnch: np.ndarray | None = None
        self._intraday: dict = {}             # YYYYMMDD -> 原始 {ts_code: [[t,px,vol,amt]]}
        self._intraday_norm: dict = {}        # YYYYMMDD -> {ts_code: [(t,px,vol股,amt)]}
        self._intraday_mt: dict = {}          # YYYYMMDD -> mtime(盘中增量重读守护)
        self._auction: dict = {}              # YYYYMMDD -> {ts_code: (px,vol,amt)}
        self._snap_times: dict = {}           # YYYYMMDD -> [HHMMSS...] 回放时钟
        self._struct: dict = {}               # YYYYMMDD -> {ts_code: 结构分}

    # ---------- 交易日历 ----------

    def get_trading_calendars(self) -> dict:
        """rqalpha 要求 {TRADING_CALENDAR_TYPE: DatetimeIndex}(非交易所字符串);
        A 股沪深共用 CN_STOCK 一份日历。"""
        from rqalpha.const import TRADING_CALENDAR_TYPE
        if self._cal is None:
            cal = pd.read_parquet(path_of("meta.trade_cal"))
            open_days = cal[cal["is_open"] == 1]["cal_date"]
            idx = pd.to_datetime(open_days, format="%Y%m%d").sort_values()
            self._cal = pd.DatetimeIndex(idx.unique())
        return {TRADING_CALENDAR_TYPE.CN_STOCK: self._cal}

    # ---------- 名称与日线 ----------

    def _load_names(self) -> dict:
        """{ts_code: name}; qmt_names.json 结构为 {date, data}(见项目记忆)"""
        if not self._names:
            f = DATA / "meta" / "qmt_names.json"
            if f.exists():
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    self._names = d.get("data") or {}
                except Exception:
                    self._names = {}
        return self._names

    def _load_panel(self) -> dict:
        """日线面板 → {ts_code: np.ndarray(FULL_BAR_DTYPE)} 一次性建索引。

        全向量化计算 + 边界切片(非逐票循环): daily_panel 约 760 万行,
        逐票 Python 循环要几十秒, 向量化后降到几秒。
        datetime 字段直接用 YYYYMMDD × 1e6 —— 恰好等于 rqalpha 的
        year*1e10+month*1e8+day*1e6 午夜编码。
        total_turnover 用 close×vol×100 近似(与 apps/server.py 同口径 —
        面板只有 vol(手) 无成交额, 精确额需 VWAP, 项目一贯用此近似)。"""
        if self._by_code is not None:
            return self._by_code
        df = pd.read_parquet(path_of("market.daily_panel"))
        names = self._load_names()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        self._panel_dates = pd.DatetimeIndex(pd.to_datetime(
            pd.unique(df["trade_date"]), format="%Y%m%d").sort_values())

        code_s = df["ts_code"].astype(str)
        nm_s = code_s.map(lambda c: names.get(c, ""))
        # 涨跌停档位向量化(同 quotes/qmt.py _limit_ratio 口径)
        pre2 = code_s.str[:2]
        is_st = nm_s.str.upper().str.contains("ST", na=False) \
            | nm_s.str.contains("退", na=False)
        ratio = np.where(pre2.isin(["30", "68"]).values, 0.20,
                         np.where(is_st.values, 0.05, 0.10))

        td = df["trade_date"].astype(np.int64).values
        pre = df["pre_close"].astype(float).values
        op = df["open"].astype(float).values
        cl = df["close"].astype(float).values
        # 首日无 pre_close: 用当日 open 兜底(涨跌停价仅影响撮合夹价)
        pre = np.where(np.isnan(pre) | (pre <= 0), op, pre)
        vol = df["vol"].astype(float).values          # 手

        rec = np.empty(len(df), dtype=FULL_BAR_DTYPE)
        rec["datetime"] = td * 1000000
        rec["open"] = op
        rec["close"] = cl
        rec["high"] = df["high"].astype(float).values
        rec["low"] = df["low"].astype(float).values
        rec["volume"] = vol * 100.0                   # 手 → 股(rqalpha口径)
        rec["total_turnover"] = cl * vol * 100.0      # 元(近似)
        rec["limit_up"] = np.round(pre * (1 + ratio), 2)
        rec["limit_down"] = np.round(pre * (1 - ratio), 2)
        rec["prev_close"] = pre
        rec["last"] = cl

        # 按 ts_code 边界切片(已排序, 相邻不等处即分组边界)
        codes = code_s.values
        bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
        out = {}
        for i in range(len(bounds) - 1):
            out[codes[bounds[i]]] = rec[bounds[i]:bounds[i + 1]]
        self._by_code = out
        return out

    # ---------- Instruments ----------

    def get_instruments(self, id_or_syms=None, types=None):
        """Instrument 列表。

        listed_date 由日线面板首个交易日推断(近似口径: 面板起点
        20191128, 早于该日上市的票会被截到 20191128, 对打板策略无影响)。

        de_listed_date 不能用面板末日! 面板末日只代表"之后还没采到数据",
        不代表退市 —— 实测把末日当退市日会使 rqalpha 报
        "002969.XSHE is not listing!" 而拒单。口径: 在 qmt_names.json
        (当日在市名单)里的票视为正常上市 → 20991231; 不在名单但在面板里
        的才用面板末日(大概率已退市)。"""
        from rqalpha.const import INSTRUMENT_TYPE
        from rqalpha.model.instrument import Instrument
        if not self._instruments:
            by_code = self._load_panel()
            names = self._load_names()
            for code, rec in by_code.items():
                rq_id = to_rq(code)
                if rq_id is None:          # 北交所等不支持后缀
                    continue
                nm = names.get(code, code)
                first = int(rec["datetime"][0])
                last = int(rec["datetime"][-1])
                # 在市名单里 → 未退市; 否则用面板末日
                de_listed = "20991231" if code in names \
                    else _int_to_date_str(last)
                self._instruments[rq_id] = Instrument({
                    "order_book_id": rq_id,
                    "symbol": nm,
                    "type": INSTRUMENT_TYPE.CS,
                    "exchange": exchange_of(rq_id),
                    "board_type": _board_type(code),
                    "round_lot": 100,
                    "status": "Active" if code in names else "Delisted",
                    "listed_date": _int_to_date_str(first),
                    "de_listed_date": de_listed,
                })
            # 虚拟基准 + 指数(见 _load_benchmarks)
            for ins in self._load_benchmarks():
                self._instruments[ins.order_book_id] = ins
        ids = id_or_syms
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            return [self._instruments[i] for i in ids if i in self._instruments]
        return list(self._instruments.values())

    def _load_benchmarks(self) -> list:
        """基准 Instrument: 指数(需 index_panel) + 自建打板基准 DBBNCH"""
        from rqalpha.const import INSTRUMENT_TYPE
        from rqalpha.model.instrument import Instrument
        out = []
        ip = path_of("market.index_panel") if _has_index_panel() else None
        if ip is not None:
            df = pd.read_parquet(ip)
            df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
            self._index_panel = {}
            has_amt = "amount" in df.columns
            cl = df["close"].astype(float).values
            rec_all = np.empty(len(df), dtype=FULL_BAR_DTYPE)
            rec_all["datetime"] = df["trade_date"].astype(np.int64).values * 1000000
            for k in ("open", "close", "high", "low"):
                rec_all[k] = df[k].astype(float).values
            rec_all["volume"] = df["vol"].astype(float).values * 100.0
            rec_all["total_turnover"] = (df["amount"].astype(float).values
                                         if has_amt else 0.0)
            rec_all["limit_up"] = np.nan
            rec_all["limit_down"] = np.nan
            rec_all["prev_close"] = df["pre_close"].astype(float).values
            rec_all["last"] = cl
            codes = df["ts_code"].astype(str).values
            bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
            for i in range(len(bounds) - 1):
                idx_code = codes[bounds[i]]
                rq_id = to_rq(idx_code)
                if rq_id is None:
                    continue
                rec = rec_all[bounds[i]:bounds[i + 1]]
                self._index_panel[rq_id] = rec
                out.append(Instrument({
                    "order_book_id": rq_id,
                    "symbol": _INDEX_NAMES.get(idx_code, idx_code),
                    "type": INSTRUMENT_TYPE.INDX,
                    "exchange": exchange_of(rq_id),
                    "round_lot": 100, "status": "Active",
                    "listed_date": _int_to_date_str(int(rec["datetime"][0])),
                    "de_listed_date": "20991231",
                }))
        # 打板基准: 全A等权涨幅累积指数(数据现成, 无需额外采集)
        out.append(Instrument({
            "order_book_id": DBBNCH_ID,
            "symbol": "打板基准(全A等权)",
            "type": INSTRUMENT_TYPE.INDX,
            "exchange": "XSHG",
            "round_lot": 100, "status": "Active",
            "listed_date": "20191128", "de_listed_date": "20991231",
        }))
        return out

    def _trailing_fill(self, rec: np.ndarray, last_close: float) -> np.ndarray:
        """对面板未覆盖的尾部交易日做前向填充(收益=0)。

        为何需要: sys_analyser 算基准收益时要求基准 bar 覆盖
        [回测首日的前一交易日 .. 回测末日]。盘中实时模式下 daily_panel
        还没补尾(靠 daily_update.sh 收盘后跑), 回测末日=今天 → 基准缺
        最后一天 → 长度对不上 → 整个 run 崩掉。
        前向填充让缺失日收益=0(基准当日未知, 持平), 不崩且不伪造涨跌。"""
        from rqalpha.const import TRADING_CALENDAR_TYPE
        cal = self.get_trading_calendars()[TRADING_CALENDAR_TYPE.CN_STOCK]
        panel_last = int(rec["datetime"][-1]) // 1000000
        today = int(date.today().strftime("%Y%m%d"))
        extra = []
        for d in cal:
            di = int(d.strftime("%Y%m%d"))
            if panel_last < di <= today:
                extra.append(di)
        if not extra:
            return rec
        add = np.empty(len(extra), dtype=FULL_BAR_DTYPE)
        for i, di in enumerate(extra):
            add["datetime"][i] = di * 1000000
            for k in ("open", "close", "high", "low", "last", "prev_close"):
                add[k][i] = last_close
            add["volume"][i] = 0.0
            add["total_turnover"][i] = 0.0
            add["limit_up"][i] = np.nan
            add["limit_down"][i] = np.nan
        return np.concatenate([rec, add])

    def _dbbnch_bars(self) -> np.ndarray:
        """全A等权涨幅累积指数 bar 序列(懒算缓存)"""
        if self._dbbnch is None:
            df = pd.read_parquet(path_of("market.daily_panel"),
                                 columns=["trade_date", "pct_chg"])
            # 等权日收益(停牌/异常值由 mean 自动跳过 NaN)
            g = df.groupby("trade_date")["pct_chg"].mean() / 100.0
            g = g.sort_index().fillna(0.0)
            vals = (DBBNCH_BASE * (1.0 + g).cumprod()).values.astype(float)
            rec = np.empty(len(g), dtype=FULL_BAR_DTYPE)
            rec["datetime"] = g.index.astype(np.int64).values * 1000000
            for k in ("open", "close", "high", "low", "last"):
                rec[k] = vals
            rec["volume"] = 0.0
            rec["total_turnover"] = 0.0
            rec["limit_up"] = np.nan
            rec["limit_down"] = np.nan
            rec["prev_close"] = np.concatenate([[DBBNCH_BASE], vals[:-1]])
            rec = self._trailing_fill(rec, float(vals[-1]))
            self._dbbnch = rec
        return self._dbbnch

    def _bars_of(self, order_book_id: str) -> np.ndarray | None:
        """按 rqalpha 口径取 bar 序列(个股/指数/打板基准统一入口)"""
        if order_book_id == DBBNCH_ID:
            return self._dbbnch_bars()
        if self._index_panel and order_book_id in self._index_panel:
            return self._index_panel[order_book_id]
        ts = from_rq(order_book_id)
        if ts is None:
            return None
        return self._load_panel().get(ts)

    # ---------- bar 查询 ----------

    def history_bars(self, instrument, bar_count, frequency, fields, dt,
                     skip_suspended=True, include_now=False,
                     adjust_type="pre", adjust_orig=None):
        """语义与 rqalpha BaseDataSource.history_bars 严格一致(实测对齐):

          i = searchsorted(dt_int, side='right')   # 含 dt 本身
          bar_count=None → left=0(返回截至 dt 的全部 bar)
          否则 left = max(0, i - bar_count)
          return bars[left:i]

        两个易错点(均为实测踩坑):
        ① bar_count=None 不是"1根"而是"全部"。sys_analyser 算基准收益时
           传 bar_count=None, 若当成 1 只会返回 1 根 bar → 长度对不上
           → 报 "benchmark available data ... <= backtest end date" 而
           整个 run 崩掉, 且 alpha/beta/夏普/信息比率全为 nan。
        ② 1d 分支忽略 include_now(它只对分钟/tick 有意义), 总是含 dt。

        不复权口径(与聚宽 use_real_price=True 一致), adjust_type 忽略。"""
        if frequency not in ("1d", "1m"):
            raise NotImplementedError(f"frequency {frequency} 不支持")
        bars = self._bars_of(instrument.order_book_id)
        if bars is None or len(bars) == 0:
            return None
        if frequency == "1d":
            dt_int = dt_to_int(_as_dt(dt))
            i = int(bars["datetime"].searchsorted(dt_int, side="right"))
            if bar_count is None:
                left = 0
            else:
                left = max(0, i - bar_count)
            seg = bars[left:i]
            if len(seg) == 0:
                return None
            return _select_fields(seg, fields)
        # 1m: 由盘中快照合成(见 get_bar)
        day = _date_str_of(dt)
        code = from_rq(instrument.order_book_id)
        pts = self.points_of(day, code) if code else []
        if not pts:
            return None
        upto = [p for p in pts if p[0] <= _hhmmss_of(dt)]
        if not upto:
            return None
        seg = upto[-(bar_count or 1):]
        rec = np.empty(len(seg), dtype=FULL_BAR_DTYPE)
        base_dt = _midnight_int(dt)
        for i, (hms, px, vol, amt) in enumerate(seg):
            rec["datetime"][i] = base_dt + _hhmmss_to_int(hms)
            for k in ("open", "close", "high", "low", "last"):
                rec[k][i] = px
            rec["volume"][i] = vol          # _intraday_of 已归一为股
            rec["total_turnover"][i] = amt
            nm = self._load_names().get(code, "")
            pre = self._prev_close(code, dt)
            ratio = _limit_ratio(code, nm)
            rec["limit_up"][i] = round(pre * (1 + ratio), 2) if pre else np.nan
            rec["limit_down"][i] = round(pre * (1 - ratio), 2) if pre else np.nan
            rec["prev_close"][i] = pre or np.nan
        return _select_fields(rec, fields)

    def get_bar(self, instrument, dt, frequency):
        """单根 bar(numpy 结构化标量); 无数据返回 None。
        '1m' 用盘中快照最近邻(≤dt 的最后一点)合成, 使撮合器能拿到
        limit_up/limit_down/volume。"""
        bars = self._bars_of(instrument.order_book_id)
        if frequency == "1d":
            if bars is None:
                return None
            pos = _search_dt(bars, dt, include_now=True)
            return None if pos < 0 else bars[pos]
        if frequency != "1m":
            return None
        code = from_rq(instrument.order_book_id)
        if code is None:
            return None
        pts = self.points_of(_date_str_of(dt), code)
        if not pts:
            # 盘中快照缺失 → 退回日线当日 bar(撮合仍可夹价, 不至于全拒)
            if bars is None:
                return None
            pos = _search_dt(bars, dt, include_now=True)
            return None if pos < 0 else bars[pos]
        hms = _hhmmss_of(dt)
        best = None
        for p in pts:
            if p[0] <= hms:
                best = p
            else:
                break
        if best is None:
            best = pts[0]
        _, px, vol, amt = best
        nm = self._load_names().get(code, "")
        pre = self._prev_close(code, dt)
        ratio = _limit_ratio(code, nm)
        rec = np.empty(1, dtype=FULL_BAR_DTYPE)
        rec["datetime"][0] = _midnight_int(dt) + _hhmmss_to_int(best[0])
        for k in ("open", "close", "high", "low", "last"):
            rec[k][0] = px
        rec["volume"][0] = vol              # _intraday_of 已归一为股
        rec["total_turnover"][0] = amt
        rec["limit_up"][0] = round(pre * (1 + ratio), 2) if pre else np.nan
        rec["limit_down"][0] = round(pre * (1 - ratio), 2) if pre else np.nan
        rec["prev_close"][0] = pre or np.nan
        return rec[0]

    def current_snapshot(self, instrument, frequency, dt):
        """盘中快照(TickObject 口径); 仅 1m/盘中粒度提供"""
        code = from_rq(instrument.order_book_id)
        if code is None:
            return None
        pts = self.points_of(_date_str_of(dt), code)
        if not pts:
            return None
        hms = _hhmmss_of(dt)
        best = None
        for p in pts:
            if p[0] <= hms:
                best = p
            else:
                break
        if best is None:
            return None
        from rqalpha.model.tick import TickObject
        h, px, vol, amt = best
        pre = self._prev_close(code, dt) or px
        nm = self._load_names().get(code, "")
        ratio = _limit_ratio(code, nm)
        y, m, d = dt.year, dt.month, dt.day
        return TickObject(instrument, {
            "order_book_id": instrument.order_book_id,
            "datetime": datetime(y, m, d, int(h[:2]), int(h[2:4]), int(h[4:])),
            "last": px, "open": px, "high": px, "low": px,
            "prev_close": pre,
            "limit_up": round(pre * (1 + ratio), 2),
            "limit_down": round(pre * (1 - ratio), 2),
            "volume": vol, "total_turnover": amt,
        })

    # ---------- 集合竞价 ----------

    OPEN_AUCTION_BAR_FIELDS = ["datetime", "open", "limit_up", "limit_down",
                               "volume", "total_turnover"]

    def get_open_auction_bar(self, instrument, dt):
        """竞价 bar(dict): 取 09:25 首个盘中快照点; 缺失退回日线开盘价。
        rqalpha 撮合器在 OPEN_AUCTION 阶段用 .open 定成交价。"""
        code = from_rq(instrument.order_book_id)
        pre = self._prev_close(code, dt) if code else None
        nm = self._load_names().get(code, "") if code else ""
        ratio = _limit_ratio(code, nm) if code else 0.10
        day = _date_str_of(dt)
        px, vol, amt = None, 0.0, 0.0
        if code:
            a = self.auction_point(day, code)
            if a:
                px, vol, amt = a
        if px is None and code:
            bars = self._bars_of(instrument.order_book_id)
            if bars is not None:
                pos = _search_dt(bars, dt, include_now=True)
                if pos >= 0:
                    px = float(bars["open"][pos])
        bar = dict.fromkeys(self.OPEN_AUCTION_BAR_FIELDS, np.nan)
        if px is not None:
            bar["datetime"] = dt_to_int(datetime(dt.year, dt.month, dt.day, 9, 25))
            bar["open"] = px
            bar["volume"] = vol             # _intraday_of/_auction_of 已归一为股
            bar["total_turnover"] = amt
            if pre:
                bar["limit_up"] = round(pre * (1 + ratio), 2)
                bar["limit_down"] = round(pre * (1 - ratio), 2)
        bar["last"] = bar["open"]
        return bar

    def get_open_auction_volume(self, instrument, dt):
        code = from_rq(instrument.order_book_id)
        if not code:
            return np.nan
        a = self.auction_point(_date_str_of(dt), code)
        return a[1] if a else np.nan        # 已是股

    # ---------- 其它必需接口 ----------

    def available_data_range(self, frequency):
        """日频: 面板首末日; 分钟频: intraday_px 已有分区的首末日"""
        if frequency == "1d":
            self._load_panel()
            d = self._panel_dates
            return d[0].date(), d[-1].date()
        dates = _intraday_dates()
        if not dates:
            today = date.today()
            return today, today
        f = datetime.strptime(dates[0], "%Y%m%d").date()
        t = datetime.strptime(dates[-1], "%Y%m%d").date()
        return f, t

    def get_trading_minutes_for(self, instrument, trading_dt):
        """股票交易分钟(09:31~11:30 + 13:01~15:00), 同 sys_simulation 口径"""
        from rqalpha.utils.datetime_func import convert_dt_to_int
        d = trading_dt.date() if hasattr(trading_dt, "date") else trading_dt
        step = timedelta(minutes=1)
        out = []
        cur = datetime.combine(d, AM_START) + step     # 09:31 起
        end_am = datetime.combine(d, AM_END)
        while cur <= end_am:
            out.append(convert_dt_to_int(cur))
            cur += step
        cur = datetime.combine(d, PM_START) + step     # 13:01 起
        end_pm = datetime.combine(d, PM_END)
        while cur <= end_pm:
            out.append(convert_dt_to_int(cur))
            cur += step
        return sorted(out)

    def is_st_stock(self, order_book_id, dates):
        """ST 判定用当前名称快照(历史名称变更不可考 — 与项目一贯口径一致)"""
        code = from_rq(order_book_id) or ""
        nm = self._load_names().get(code, "")
        flag = ("ST" in nm.upper()) or ("退" in nm)
        return [flag] * len(dates)

    def is_suspended(self, order_book_id, dates):
        """停牌 = 该日在面板中无记录。

        两个必须处理的边界(均为实测踩坑):

        ① 传入的可能是带时分秒的 datetime(盘中下单时 rqalpha 传
           2026-08-31 09:34:47), 而面板 bar 的 datetime 是午夜编码
           (20260831000000)。不归一就会查不到 → 把正常交易的票误判为
           停牌而全部拒单。故比较前必须先截到当日零点。

        ② 面板末日之后的日期不能当停牌! daily_panel 靠 daily_update.sh
           收盘后补尾, 盘中当日还没补时全部票都会被误判停牌。
           面板未覆盖的日期视为"无法判定" → 返回 False。"""
        bars = self._bars_of(order_book_id)
        if bars is None:
            return [True] * len(dates)
        have = set(int(x) for x in bars["datetime"])
        panel_last = max(have)
        out = []
        for d in dates:
            # 截到当日零点: 盘中 datetime 带时分秒, 与午夜编码对不上
            day_int = dt_to_int(_as_dt(d)) // 1000000 * 1000000
            if day_int > panel_last:
                out.append(False)      # 面板未覆盖 → 无法判定, 不拦
            else:
                out.append(day_int not in have)
        return out

    # 不复权口径: 无分红/拆股数据(与聚宽 use_real_price=True 一致)
    def get_dividend(self, instrument):
        return None

    def get_split(self, instrument):
        return None

    def get_share_transformation(self, order_book_id):
        return None

    def get_algo_bar(self, id_or_ins, start_min, end_min, dt):
        return None

    # ---------- 期货/期权专用(本项目不支持) ----------

    def get_settle_price(self, instrument, date_):
        raise NotImplementedError("本项目仅支持 A 股")

    def get_futures_trading_parameters(self, instrument, dt):
        raise NotImplementedError("本项目仅支持 A 股")

    def get_yield_curve(self, start_date, end_date, tenor=None):
        """国债收益率曲线 → 无风险利率。
    
        不能返回空表! sys_analyser 的 data_proxy.get_risk_free_rate 靠它
        取无风险利率; 返回空 → risk_free=nan → 夏普/alpha/索提诺全为 nan
        (实测踩坑)。本项目不采国债曲线, 用常数年化无风险利率代替
        (默认 1.5%, 中国短端利率量级), 全部 tenor 同值。
    
        返回 DataFrame: index=日期, columns=各 tenor('0S','1M',...,'50Y')。
        get_risk_free_rate 对短回测只会取 '0S', 但补齐全列更稳。"""
        from rqalpha.utils.risk_free_helper import YIELD_CURVE_TENORS
        s = start_date if isinstance(start_date, date) else start_date.date()
        e = end_date if isinstance(end_date, date) else end_date.date()
        if e < s:
            e = s
        days = pd.date_range(s, e, freq="D")
        cols = list(YIELD_CURVE_TENORS.values())
        df = pd.DataFrame(self.risk_free_rate, index=days, columns=cols)
        if tenor is not None:
            ts = [tenor] if isinstance(tenor, str) else list(tenor)
            ts = [t for t in ts if t in df.columns]
            if ts:
                df = df[ts]
        return df

    def get_exchange_rate(self, trading_date, local, settlement=None):
        raise NotImplementedError("本项目仅支持 A 股")

    # ---------- tick(本项目用 20s 快照走 BAR 事件, 不用 tick 链路) ----------

    def get_merge_ticks(self, order_book_id_list, trading_date, last_dt=None):
        raise NotImplementedError("盘中走 BAR 事件(20s 快照), 不用 tick 链路")

    def history_ticks(self, instrument, count, dt):
        raise NotImplementedError("盘中走 BAR 事件(20s 快照), 不用 tick 链路")

    # ---------- 盘中快照缓存 ----------

    def refresh_intraday(self, day: str) -> bool:
        """盘中模式用: mtime 变动则丢弃缓存重读。
        返回是否发生了重读(供事件源判断是否有新快照)。

        必须连 _snap_times 一起失效: 它是回放时钟(快照时刻并集),
        live 轮询靠它找"还没发过的新快照点"。只失效 raw 不失效它 →
        catchup 之后候选永远为空, 午后 bar 全丢(实测 20260904:
        state 冻在补跑尾态 11:26, radar 已到 14:00)。"""
        f = LIVE / f"intraday_px_{day}.json"
        if not f.exists():
            return False
        mt = f.stat().st_mtime
        if self._intraday_mt.get(day) == mt:
            return False
        self._intraday_mt[day] = mt
        self._intraday.pop(day, None)
        self._intraday_norm.pop(day, None)
        self._auction.pop(day, None)
        self._snap_times.pop(day, None)
        self.intraday_points.cache_clear()
        return True

    def _raw_of(self, day: str) -> dict | None:
        """intraday_px_{day}.json 原始内容 {ts_code: [[HHMMSS,px,vol,amt]]}。
        只存原始 list, 不做逐点归一 —— 实测最大分区 210 万点,
        全量归一要十几秒, 盘中每 20s 轮询无法承受; 故改为按 code 惰性归一。"""
        if day not in self._intraday:
            f = LIVE / f"intraday_px_{day}.json"
            d = {}
            if f.exists():
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    d = {}
            self._intraday[day] = d
        return self._intraday[day] or None

    def points_of(self, day: str, ts_code: str) -> list:
        """单票归一后的分时点 [(HHMMSS, px, vol股, amt元)], 时间升序。

        volume 修正(必需): 雷达 intraday_px 的 volume 字段跨日不一致
        —— 实测 20260827/20260903 非零率 0%, 20260828/31 为 67~84%,
        20260901/02 仅 2~5%(QMT 推送路径与日bar兜底路径的字段差异)。
        rqalpha 撮合器在 inactive_limit 下遇 bar_volume==0 会直接撤单,
        故 vol==0 且 amount>0 时用 amount/price 反推股数(元÷元/股=股,
        精确而非近似), 保证撮合不因数据缺口而全部拒单。"""
        cache = self._intraday_norm.setdefault(day, {})
        if ts_code not in cache:
            raw = (self._raw_of(day) or {}).get(ts_code) or []
            norm = []
            for p in raw:
                if len(p) < 2:
                    continue
                px = float(p[1])
                vol = float(p[2]) if len(p) >= 3 else 0.0
                amt = float(p[3]) if len(p) >= 4 else 0.0
                if vol <= 0 and amt > 0 and px > 0:
                    vol = amt / px        # 元 ÷ (元/股) = 股
                norm.append((str(p[0]), px, vol, amt))
            norm.sort(key=lambda x: x[0])
            cache[ts_code] = norm
        return cache[ts_code]

    def _auction_of(self, day: str) -> dict:
        """竞价快照 {ts_code: (px, vol股, amt元)}: 取 intraday_px 首个 ≤092500 的点。
        只对已被访问过的票惰性计算(同 points_of 的延迟口径), 避免全量扫描。"""
        cache = self._auction.setdefault(day, {})
        return cache

    def auction_point(self, day: str, ts_code: str):
        """单票竞价点 (px, vol, amt); 无则 None"""
        cache = self._auction_of(day)
        if ts_code not in cache:
            pts = self.points_of(day, ts_code)
            hit = None
            for p in pts:
                if p[0] <= "092500":
                    hit = (p[1], p[2], p[3])
                    break
            cache[ts_code] = hit
        return cache[ts_code]

    def _prev_close(self, ts_code: str, dt) -> float | None:
        """dt 前一交易日收盘价(用于涨跌停价)"""
        if ts_code is None:
            return None
        bars = self._load_panel().get(ts_code)
        if bars is None:
            return None
        pos = _search_dt(bars, dt, include_now=True)
        if pos < 0:
            return None
        v = float(bars["prev_close"][pos])
        return v if v > 0 else None

    def benchmark_close(self, order_book_id: str, dt) -> float | None:
        """基准在 dt 时刻的收盘价(供看板将策略净值与基准对齐展示)。
        无数据返回 None(不伪造 0)。"""
        bars = self._bars_of(order_book_id)
        if bars is None or len(bars) == 0:
            return None
        pos = _search_dt(bars, dt, include_now=True)
        return None if pos < 0 else float(bars["close"][pos])

    # ---------- 结构层快照 ----------

    def struct_snapshot(self, day: str, codes: list | None = None) -> dict:
        """{ts_code: {g_chip, gate, v5_base, zb_cnt20, y_volr5, neg_streak,
        ind_rank, ind_breadth}} —— 用 core/structure.py 从日线现算。

        为何不从雷达取: 雷达确实每日对全宇宙算 build_struct_scores,
        但只把它作为影子字段挂在 presignal 上 —— 而结构分是信号触发
        时才挂的, 策略在 before_trading(09:00) 时拿不到(那时还没有任何
        信号), 候选池会为空 → 一笔不买(实测踩坑)。故改为从 daily_panel
        现算全市场结构分, 盘中与回放都能用且确定性可复现。

        ldlr_prev 传 None: 它是全市场常量无截面区分度, 不进 g_chip
        (见 core/structure.py 注释), 仅透传; 回放时若去拉当日值反而
        会引入未来信息。

        按日缓存(每日只算一次)。"""
        if day in self._struct:
            return self._struct[day]
        from core.structure import build_struct_scores
        by_code = self._load_panel()
        target = codes if codes is not None else list(by_code.keys())
        bars_by = {}
        for c in target:
            rec = by_code.get(c)
            if rec is None:
                continue
            # core.structure 要的格式: [(date, high, low, close, vol)] 升序
            rows = []
            for i in range(len(rec)):
                d = _int_to_date_str(int(rec["datetime"][i]))
                if d > day:
                    break
                rows.append((d, float(rec["high"][i]), float(rec["low"][i]),
                             float(rec["close"][i]),
                             float(rec["volume"][i]) / 100.0))   # 股→手
            if rows:
                bars_by[c] = rows
        try:
            out = build_struct_scores(list(bars_by.keys()), bars_by, None)
        except Exception as e:
            print(f"[ticai] 结构分计算失败({day}): {e}")
            out = {}
        self._struct[day] = out
        return out

    # ---------- 供注入 API 使用的只读视图 ----------

    @lru_cache(maxsize=64)
    def intraday_points(self, day: str, ts_code: str):
        """策略侧读分时轨迹(只读元组, 防策略误改缓存)"""
        pts = self.points_of(day, ts_code)
        return tuple(pts) if pts else ()

    def snapshot_times(self, day: str) -> list:
        """当日雷达快照时刻序列(HHMMSS 升序去重) —— 回放事件源的驱动时钟。

        雷达每 cycle(20s) 写一批快照, 故 intraday_px 内全部时间戳的并集
        就是 cycle 序列。盘中模式不用本方法(改用轻量的 radar.json 取
        当前 ts, 474KB/0.02s), 因为本方法要扫全部点(最大分区 210 万点)。
        结果按日缓存, 回放时每日只算一次。"""
        if day in self._snap_times:
            return self._snap_times[day]
        raw = self._raw_of(day) or {}
        ts = set()
        for pts in raw.values():
            for p in pts:
                if p:
                    ts.add(str(p[0]))
        out = sorted(ts)
        self._snap_times[day] = out
        return out


# ---------- 工具函数 ----------

_INDEX_NAMES = {"000300.SH": "沪深300", "000985.SH": "中证全指",
                "000001.SH": "上证指数", "399006.SZ": "创业板指"}


def _has_index_panel() -> bool:
    try:
        return path_of("market.index_panel").exists()
    except Exception:
        return False


def _int_to_date_str(dt_int: int) -> str:
    return str(dt_int // 1000000)


def _as_dt(d):
    """date/datetime/Timestamp → datetime"""
    if isinstance(d, datetime):
        return d
    if hasattr(d, "to_pydatetime"):
        return d.to_pydatetime()
    return datetime.combine(d, time(0, 0))


def _date_str_of(dt) -> str:
    return _as_dt(dt).strftime("%Y%m%d")


def _hhmmss_of(dt) -> str:
    return _as_dt(dt).strftime("%H%M%S")


def _midnight_int(dt) -> int:
    d = _as_dt(dt)
    return d.year * 10000000000 + d.month * 100000000 + d.day * 1000000


def _hhmmss_to_int(hms: str) -> int:
    hms = str(hms).zfill(6)
    return int(hms[:2]) * 10000 + int(hms[2:4]) * 100 + int(hms[4:6])


def _search_dt(bars: np.ndarray, dt, include_now: bool = True) -> int:
    """返回 bars 中 ≤dt 的最后一根下标; 无则 -1"""
    target = dt_to_int(_as_dt(dt))
    if not include_now:
        target -= 1
    pos = int(bars["datetime"].searchsorted(target, side="right")) - 1
    return pos if pos >= 0 else -1


def _select_fields(seg: np.ndarray, fields) -> np.ndarray:
    """按 fields 裁列; fields=None 返回全部原生字段"""
    if fields is None:
        names = DAY_BAR_NAMES + ["limit_up", "limit_down", "prev_close", "last"]
    elif isinstance(fields, str):
        names = [fields]
    else:
        names = list(fields)
    names = [n for n in names if n in seg.dtype.names]
    if not names:
        return seg
    return seg[names]


def _intraday_dates() -> list:
    """已有 intraday_px 分区的日期(升序)"""
    if not LIVE.exists():
        return []
    out = []
    for f in LIVE.glob("intraday_px_*.json"):
        d = f.stem.replace("intraday_px_", "")
        if d.isdigit() and len(d) == 8 and ".pre_repair" not in f.name:
            out.append(d)
    return sorted(out)
