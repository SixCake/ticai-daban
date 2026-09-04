# -*- coding: utf-8 -*-
"""注入 API — 策略取数据的唯一通道(策略隔离规范的载体)

规范硬约束(见 strategies/_template/SPEC.md):
  策略禁止直接读 data/ 路径, 只能通过本模块注入的 API 取数据。
  理由: ① 代码口径统一(项目内部 .SZ, rqalpha 内部 .XSHE, 转换只在
  codes.py) ② 时间戳闸门统一施加, 防未来信息 ③ 数据源可替换而不改策略。

两条闸门(防回测未来信息污染):
  信号闸门  presig_state 的 signal.t <= 当前模拟时刻 才可见
  feed闸门  AI feed 条目的产出时间 ts <= 当前模拟时刻 才可见
盘中模式下"当前模拟时刻"就是墙钟, 回放模式下是回放的 calendar_dt。

代码口径: 所有返回给策略的 ts_code 一律已转成 rqalpha 口径(.XSHE/.XSHG),
策略不需要也不应该调 codes.py。
"""
import json

from rqalpha.api import export_as_api
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.const import EXECUTION_PHASE
from rqalpha.environment import Environment

from config import DATA

from . import feeds
from .codes import to_rq

LIVE = DATA / "live"

# 全阶段可用的 API 允许的执行阶段
_ALL_PHASES = (EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING,
               EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR,
               EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.AFTER_TRADING)
_INTRADAY_PHASES = (EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR,
                    EXECUTION_PHASE.SCHEDULED)


class TicaiApi:
    """注入 API 的实现。由 TicaiMod.start_up 构造并注册。

    strategy      当前策略名(用于私有 feed 定位与鉴权日志)
    feeds_allowed 该策略 config.yaml 声明可订阅的 feed 名白名单
    """

    def __init__(self, env, strategy: str, feeds_allowed: list | None = None):
        self._env = env
        self.strategy = strategy or ""
        self.feeds_allowed = set(feeds_allowed or [])
        self.benchmark_id: str | None = None    # 由 set_benchmark() 设定
        self._radar_cache: tuple = (None, None)     # (mtime, data)
        self._presig_cache: tuple = (None, None, None)   # (day, mtime, signals)

    # ---------- 时间基准 ----------

    def now(self):
        """当前模拟时刻: 回放=calendar_dt, 盘中=calendar_dt(即墙钟)"""
        return Environment.get_instance().calendar_dt

    def day(self) -> str:
        return self.now().strftime("%Y%m%d")

    def hhmmss(self) -> str:
        return self.now().strftime("%H:%M:%S")

    def cutoff(self) -> float:
        """feed 闸门的 epoch 值"""
        return self.now().timestamp()

    # ---------- 数据读取(带 mtime 缓存) ----------

    def _radar(self) -> dict:
        f = LIVE / "radar.json"
        if not f.exists():
            return {}
        mt = f.stat().st_mtime
        if self._radar_cache[0] == mt:
            return self._radar_cache[1]
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        self._radar_cache = (mt, d)
        return d

    def _presignals(self) -> list:
        """当日累积前向信号(presig_state), 按 mtime+日期缓存"""
        day = self.day()
        f = LIVE / f"presig_state_{day}.json"
        if not f.exists():
            return []
        mt = f.stat().st_mtime
        if self._presig_cache[0] == day and self._presig_cache[1] == mt:
            return self._presig_cache[2]
        try:
            sigs = json.loads(f.read_text(encoding="utf-8")).get("signals", [])
        except Exception:
            sigs = []
        self._presig_cache = (day, mt, sigs)
        return sigs

    # ---------- 对外方法 ----------

    def signals(self, stage: str | None = None) -> list:
        """前向预警信号(S1/S2/S3), 已施加时间戳闸门 + 转 rqalpha 代码口径。

        返回项字段: code(rqalpha口径) name stage pct why exec r3 accel
        pathvol vr limit_px t pb pt price0 sealed_t touch_t zb_cnt zt_ev
        struct(g_chip/gate/v5/zb20/ir)
        stage 传 'S2'/'S3' 可只取该级; 传 None 取全部(含 S1 观察名单)。
        """
        now_hms = self.hhmmss()
        out = []
        for s in self._presignals():
            if str(s.get("t") or "") > now_hms:
                continue                      # 未来信号, 闸门拦截
            if stage and s.get("stage") != stage:
                continue
            code = to_rq(s.get("ts_code"))
            if code is None:
                continue                      # 北交所等不支持
            d = dict(s)
            d["code"] = code
            d.pop("ts_code", None)
            d.pop("px_hist", None)            # 分时轨迹走 intraday_points, 不随信号返回
            out.append(d)
        return out

    def _data_source(self):
        return getattr(self._env.data_proxy, "_data_source", None)

    def struct(self) -> dict:
        """V5 结构层分 {rqalpha代码: {g_chip, gate, v5, zb20, ir}}。

        数据源: DataSource.struct_snapshot() —— 用 core/structure.py 从
        daily_panel 现算全市场结构分。不从雷达的 presignal 影子字段取,
        因为那个字段是信号触发时才挂的, before_trading 时拿不到。

        v5 此处等于 v5_base(不含盘中组) —— 盘中因子 r3/pathvol 要在
        信号触发后才有, 盘前选股阶段本来就只能用 T-1 结构分(与聚宽版
        prepare() 同口径)。盘中若要完整融合分, 用 signals() 里的
        struct.v5(雷达已算好)。
        """
        ds = self._data_source()
        if ds is None or not hasattr(ds, "struct_snapshot"):
            return {}
        snap = ds.struct_snapshot(self.day())
        out = {}
        for c, s in snap.items():
            code = to_rq(c)
            if code is None:
                continue
            d = dict(s)
            d["v5"] = d.get("v5_base")          # 盘前口径: 不含盘中组
            d["zb20"] = d.pop("zb_cnt20", None)
            d["ir"] = d.pop("ind_rank", None)
            out[code] = d
        return out

    def theme_heat(self) -> list:
        """题材热度自算排名(core/heat.py 口径), 按 heat 降序"""
        return [dict(t) for t in (self._radar().get("themes") or [])]

    def sw_flow(self) -> list:
        """申万一级/二级资金流向聚合(core/heat.py sw_aggregate 口径)"""
        return [dict(r) for r in (self._radar().get("sw") or [])]

    def seesaw(self) -> list:
        """龙头拐头·跷跷板事件(core/seesaw.py 口径)"""
        return list((self._radar().get("seesaw") or {}).get("events") or [])

    def intraday(self, code: str) -> list:
        """个股分时轨迹 [(HHMMSS, px, vol股, amt元)], 已按当前时刻截断"""
        from .codes import from_rq
        ts = code if code.endswith(("XSHE", "XSHG")) else to_rq(code)
        if ts is None:
            return []
        tsc = from_rq(ts)
        ds = self._data_source()
        if ds is None or not hasattr(ds, "points_of"):
            return []
        now_h = self.now().strftime("%H%M%S")
        return [p for p in ds.points_of(self.day(), tsc) if p[0] <= now_h]

    def ai(self, name: str, topic: str | None = None) -> list:
        """订阅的 AI feed 条目(已施加时间戳闸门)。

        鉴权: 只允许 config.yaml 声明过的 feed 名; 未声明的返回空列表并
        记录告警 —— 策略不得越权读其它策略的私有 feed(隔离规范)。
        私有 feed 命名约定: 'private:{feed_name}' → 读本策略专属目录。
        """
        if name not in self.feeds_allowed:
            ExecutionContext.logger.warning(
                f"[ticai] 策略 {self.strategy} 未声明订阅 feed '{name}', "
                f"已拒绝(需在 config.yaml 的 feeds 里声明)")
            return []
        if name.startswith("private:"):
            real = name[len("private:"):]
            return feeds.read_feed(real, self.day(), self.cutoff(),
                                   strategy=self.strategy)
        es = feeds.read_feed(name, self.day(), self.cutoff())
        if topic:
            es = [e for e in es if e.get("topic") == topic]
        return es

    def set_benchmark(self, order_book_id: str) -> None:
        """聚宽同款基准设定: 在 init() 里调用。

        刻意不写 rqalpha 的 config.base.benchmark —— sys_analyser 会在结算时
        校验基准数据必须覆盖回测区间+1天, 不满足直接抛异常终止整个 run。
        而基准数据的新鲜度取决于 daily_panel 是否已补尾(实测 20260903
        未补时 DBBNCH 只到 0902 → 回测 0903 就崩)。
        改为存本模块, 由 metrics.py 自己取基准价序列算 IR/alpha/beta:
        缺数据时优雅返回 None, 不影响主链路。
        可用值: 指数(需 market.index_panel 已采集) 或 DBBNCH.XSHG(自建打板基准)。
        """
        self.benchmark_id = order_book_id


# ---------- 注入(模块级单例, 由 Mod.start_up 赋值) ----------

_INSTANCE: TicaiApi | None = None


def install(env, strategy: str, feeds_allowed: list | None = None) -> TicaiApi:
    """构造单例并把 API 注入 rqalpha.api 命名空间(须在策略加载前调用)"""
    global _INSTANCE
    _INSTANCE = TicaiApi(env, strategy, feeds_allowed)

    def _inst() -> TicaiApi:
        if _INSTANCE is None:
            raise RuntimeError("ticai API 未初始化(Mod 未启用?)")
        return _INSTANCE

    @export_as_api
    @ExecutionContext.enforce_phase(*_ALL_PHASES)
    def ticai_signals(stage=None):
        """前向预警信号(S1/S2/S3), 已施加时间戳闸门, 代码为 rqalpha 口径"""
        return _inst().signals(stage)

    @export_as_api
    @ExecutionContext.enforce_phase(*_ALL_PHASES)
    def ticai_struct():
        """V5 结构层影子分 {代码: {g_chip, gate, v5, zb20, ir}}"""
        return _inst().struct()

    @export_as_api
    @ExecutionContext.enforce_phase(*_ALL_PHASES)
    def ticai_theme_heat():
        """题材热度自算排名(按 heat 降序)"""
        return _inst().theme_heat()

    @export_as_api
    @ExecutionContext.enforce_phase(*_INTRADAY_PHASES)
    def ticai_sw_flow():
        """申万资金流向聚合"""
        return _inst().sw_flow()

    @export_as_api
    @ExecutionContext.enforce_phase(*_INTRADAY_PHASES)
    def ticai_seesaw():
        """龙头拐头·跷跷板事件"""
        return _inst().seesaw()

    @export_as_api
    @ExecutionContext.enforce_phase(*_INTRADAY_PHASES)
    def ticai_intraday(code):
        """个股分时轨迹 [(HHMMSS, px, vol股, amt元)], 已按当前时刻截断"""
        return _inst().intraday(code)

    @export_as_api
    @ExecutionContext.enforce_phase(*_ALL_PHASES)
    def ai_feed(name, topic=None):
        """订阅的 AI feed(时间戳闸门 + config.yaml 声明鉴权)"""
        return _inst().ai(name, topic)

    @export_as_api
    @ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT)
    def set_benchmark(order_book_id):
        """设定绩效基准(聚宽同款); 只能在 init() 里调用"""
        return _inst().set_benchmark(order_book_id)

    return _INSTANCE


def instance() -> TicaiApi | None:
    return _INSTANCE
