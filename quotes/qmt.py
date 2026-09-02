# -*- coding: utf-8 -*-
"""大QMT实时行情适配层（与 quotes/tx.py 同契约, QUOTE_SOURCE=qmt 时启用）

链路: 盘中实时 = subscribe_whole_quote 推送(Redis pubsub, 增量累积);
      盘前/收盘/推送陈旧 = FormulaServer快路径日bar横截面降级。
连接配置全部来自环境变量 BIGQMT_*（~/.zshrc 或 .env, qmt-net lan/frp 切换
只改 BIGQMT_REDIS_HOST/BIGQMT_FORMULA_HOST, 本文件零改动跨网络）。

字段口径对齐腾讯源:
  price/pct  ← tick lastPrice/lastClose（降级时: 日bar最新/次新close）
  amount(元) ← tick当日累计成交额
  vr(量比)   ← 今日每分钟均量 / 近5日每分钟均量（近5日量每日缓存一次）
  limit_px   ← 昨收×涨幅上限取2位（主板10% / 创业板·科创20% / ST 5%）
  float_mv/tover ← 实时链路不消费(热度/概率/中军均不用), 恒置0;
                   tushare float_mv/float_share 需更高积分, 不引入依赖
  name       ← tushare stock_basic 每日缓存（一次全市场; ST过滤与涨停档位依赖）

缓存: data/meta/qmt_{names,avg5vol}.json, 可用
`python -m quotes.qmt build-cache` 盘前预建（建议加入 daily_update.sh）。
"""
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from config import DATA

META = DATA / "meta"
# xtquant_big_convert 源码路径（未pip install时加入sys.path）
BIGQMT_SRC = Path(os.environ.get(
    "BIGQMT_SRC_PATH", "~/aiproject/xtquant_big_convert/src")).expanduser()

_xtdata = None
_client_lock = threading.Lock()


def _client():
    """懒加载 bigqmt 兼容层（configure() 读 BIGQMT_* 环境变量）;
    加锁保证多线程下只初始化一次, 避免FormulaServer多连接"""
    global _xtdata
    if _xtdata is None:
        with _client_lock:
            if _xtdata is None:
                if str(BIGQMT_SRC) not in sys.path:
                    sys.path.insert(0, str(BIGQMT_SRC))
                # 实时横截面不落本地盘缓存(默认开则每轮写~5000个parquet)
                os.environ.setdefault("BIGQMT_LOCAL_CACHE_ENABLED", "0")
                from bigqmt_signal_trader.xtquant_compat import configure, xtdata
                # FormulaServer断连后RPC回退窗口30s→5s: 全市场横截面走RPC
                # 会退化到分钟级, 宁可快速失败由radar退避机制接管
                configure(redis_config={"formula_server": {
                    "failure_cooldown_seconds": 5}})
                _xtdata = xtdata
    return _xtdata


# ---------- 每日静态缓存 ----------

def _load_cache(fname: str) -> dict:
    p = META / fname
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(fname: str, obj: dict):
    META.mkdir(exist_ok=True)
    (META / fname).write_text(json.dumps(obj, ensure_ascii=False),
                              encoding="utf-8")


def _names() -> dict:
    """合约名称缓存 {code: name}; 缺失/过期时同步刷新(tushare一次全市场,秒级)"""
    cache = _load_cache("qmt_names.json")
    if cache.get("date") == datetime.now().strftime("%Y%m%d") \
            and cache.get("data"):
        return cache["data"]
    try:
        return build_names()
    except Exception as e:
        print(f"[qmt] 名称缓存刷新失败, 用旧值: {e}")
        return cache.get("data", {})


def build_names(codes: list | None = None) -> dict:
    """tushare stock_basic 一次拉全市场名称, 写缓存并返回 {code: name}"""
    from config import get_pro
    pro = get_pro()
    df = pro.stock_basic(fields="ts_code,name")
    names = dict(zip(df["ts_code"], df["name"]))
    _save_cache("qmt_names.json",
                {"date": datetime.now().strftime("%Y%m%d"), "data": names})
    print(f"[qmt] 名称缓存刷新完成 {len(names)}只")
    return names


def _avg5vol(codes: list) -> dict:
    """近5个已完成交易日日均成交量(股)缓存; 日内不变, 每日首调构建"""
    today = datetime.now().strftime("%Y%m%d")
    cache = _load_cache("qmt_avg5vol.json")
    data = cache.get("data", {}) if cache.get("date") == today else {}
    miss = [c for c in codes if c not in data]
    if miss:
        try:
            xt = _client()
            # 首次构建拉全宇宙一次(单请求), 避免radar分批触发多次count=6
            res = xt.get_market_data_ex(
                field_list=["volume"],
                stock_list=sorted(set(_universe()) | set(miss)), period="1d",
                count=6, dividend_type="none", chunk_size=0)
            for c, df in (res or {}).items():
                try:
                    # 排除最新bar所在日(盘中=今日, 日历滞后=上一交易日)
                    last_d = str(df.index[-1])[:8]
                    vols = [float(v) for d, v in zip(df.index, df["volume"])
                            if str(d)[:8] != last_d and float(v) > 0]
                    data[c] = sum(vols[-5:]) / len(vols[-5:]) if vols else 0.0
                except Exception:
                    data[c] = 0.0
            for c in miss:
                data.setdefault(c, 0.0)
            _save_cache("qmt_avg5vol.json", {"date": today, "data": data})
        except Exception as e:
            print(f"[qmt] 近5日均量构建失败: {e}")
    return data


def _floatmv() -> dict:
    """流通市值(元): 实时链路不消费, 恒返回空(tover=0)。
    若日后 tushare 积分开通 float_mv 字段, 可在此恢复每日缓存。"""
    return {}


# ---------- 派生字段 ----------

def _limit_ratio(code: str, name: str) -> float:
    """涨停档位: 创业板30x/科创68x=20%; ST(主板)=5%; 其余主板=10%"""
    if code.startswith(("30", "68")):
        return 0.20
    if "ST" in name.upper() or "退" in name:
        return 0.05
    return 0.10


def _elapsed_min(now: datetime) -> float:
    """已进行的交易分钟数（9:30-11:30 + 13:00-15:00, 全天240）"""
    hm = now.hour * 60 + now.minute
    am = max(0.0, min(hm, 11 * 60 + 30) - (9 * 60 + 30))
    pm = max(0.0, min(hm, 15 * 60) - 13 * 60)
    return am + pm


# ---------- 盘中推送（唯一实时源） ----------
# QMT服务端K线不落当日盘中数据(实测1d/1m bar停在上一交易日收盘),
# 盘中实时价只能来自 subscribe_whole_quote 推送。注意: 必须用会话层
# 裸订阅——兼容层包装会附一次全市场get_full_tick快照对齐(实测>90s
# 必超时); 基线对齐改由日bar横截面承担(见fetch_quotes降级分支)。
_push = {"ticks": {}, "ts": 0.0}
_push_lock = threading.Lock()
_push_started = False
PUSH_FRESH = 60.0    # 推送新鲜度阈值(秒), 超过降级日bar横截面


def _on_quote(data):
    if not data:
        return
    _push["ticks"].update(data)
    _push["ts"] = time.time()


def _tick_time(t: dict) -> float:
    """tick自带时间戳(毫秒); 缺失返回0"""
    try:
        return float(t.get("time") or 0) / 1000
    except Exception:
        return 0.0


def _ensure_push():
    """后台启动全市场tick推送订阅(失败指数退避重试, 上限60s)"""
    global _push_started
    with _push_lock:
        if _push_started:
            return
        _push_started = True

    def run():
        backoff = 10
        while True:
            try:
                xt = _client()
                session = xt._whole_quote_session()
                session.start()
                session.subscribe_whole_quote(["SH", "SZ"],
                                              callback=_on_quote)
                print("[qmt] 实时推送订阅成功(全市场增量)")
                return
            except Exception as e:
                print(f"[qmt] 推送订阅失败, {backoff}s后重试: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

    threading.Thread(target=run, daemon=True).start()


# ---------- 主入口（与 quotes/tx.py 同契约） ----------

# 横截面快照缓存: FormulaServer单连接串行, radar的60只/批×87批逐个排队
# frp下~20s/轮; 而全宇宙一次横截面仅~1.7s。故miss时直接拉全宇宙,
# 3s窗口内其余分批调用全部命中(L1快照本就3秒粒度)。
_snap = {"ts": 0.0, "data": {}}
_snap_lock = threading.Lock()
SNAP_TTL = 3.0
_uni_cache = {"date": "", "codes": []}


def _valid(code: str) -> bool:
    """沪深A股白名单: 概念成分含脏代码(.NQ等), QMT遇无效代码会整单拒绝"""
    return (code.endswith((".SH", ".SZ"))
            and code[:6].isdigit()
            and code[:2] in ("60", "68", "00", "30"))


def _universe() -> list:
    """雷达宇宙(题材成分去重去北交所), 每日缓存; 与radar同源同口径"""
    today = datetime.now().strftime("%Y%m%d")
    if _uni_cache["date"] != today:
        from core.attribute import load_con2stock
        _uni_cache["codes"] = sorted(
            {c for cs in load_con2stock().values() for c in cs
             if _valid(c)})
        _uni_cache["date"] = today
    return _uni_cache["codes"]


def _snapshot(codes: list) -> dict:
    now = time.time()
    wanted = set(codes)
    if now - _snap["ts"] < SNAP_TTL and wanted <= _snap["data"].keys():
        return _snap["data"]
    with _snap_lock:
        if time.time() - _snap["ts"] < SNAP_TTL \
                and wanted <= _snap["data"].keys():
            return _snap["data"]
        try:
            xt = _client()
            fetch_list = sorted(set(_universe()) | wanted)
            data = xt.get_market_data_ex(
                field_list=["close", "volume", "amount"],
                stock_list=fetch_list, period="1d", count=2,
                dividend_type="none", chunk_size=0,
                timeout_seconds=15) or {}
        except Exception as e:
            print(f"[qmt] 横截面拉取失败: {e}")
            data = {}
        _snap.update(ts=time.time(), data=data)
        return data


def _row(c: str, price: float, pre: float, vol: float, amt: float,
         names: dict, avg5: dict, fmv: dict, emin: float,
         open_px: float = 0.0) -> dict | None:
    """由 现价/昨收/累计量额 组装契约字段; 无效价格返回None"""
    if price <= 0 or pre <= 0:
        return None
    name = names.get(c, "")
    a5 = avg5.get(c, 0.0)
    vr = (vol / emin) / (a5 / 240) if emin > 0 and a5 > 0 else 0.0
    mv = fmv.get(c, 0.0)
    return {
        "name": name, "price": price,
        "open": open_px,
        "volume": vol,                      # 当日累计量(股), 供分时量/VWAP
        "pct": (price - pre) / pre * 100, "amount": amt,
        "float_mv": mv, "vr": round(vr, 3),
        "limit_px": round(pre * (1 + _limit_ratio(c, name)), 2),
        "tover": round(amt / mv * 100, 4) if mv > 0 else 0.0}


def fetch_quotes(codes: list[str]) -> dict:
    """{ts_code: {name, price, pct, amount(元), float_mv(元),
    vr(量比), limit_px(涨停价), tover(换手率%)}}; 失败返回{}由调用方退避。
    盘中优先用推送tick; 推送陈旧/缺失的票降级日bar横截面。"""
    if not codes:
        return {}
    codes = [c for c in codes if _valid(c)]
    if not codes:
        return {}
    _ensure_push()
    names = _names()
    avg5 = _avg5vol(list(codes))
    fmv = _floatmv()
    emin = _elapsed_min(datetime.now())
    out = {}
    miss = []
    now_ts = time.time()
    if now_ts - _push["ts"] < PUSH_FRESH:
        for c in codes:                       # 盘中: 推送tick优先
            t = _push["ticks"].get(c)
            row = None
            # tick自身也须新鲜(会话静默时累积表会整体变旧)
            if t and now_ts - _tick_time(t) < PUSH_FRESH:
                try:
                    row = _row(c, float(t.get("lastPrice") or 0),
                               float(t.get("lastClose") or 0),
                               float(t.get("volume") or 0),
                               float(t.get("amount") or 0),
                               names, avg5, fmv, emin,
                               float(t.get("open") or 0))
                except Exception:
                    row = None
            if row:
                out[c] = row
            else:
                miss.append(c)
    else:
        miss = list(codes)
    if miss and emin <= 0:                    # 盘前/收盘: 日bar横截面口径有效
        res = _snapshot(miss)
        wanted = set(miss)
        today = datetime.now().strftime("%Y%m%d")
        for c, df in res.items():
            if c not in wanted:
                continue
            try:
                if df is None or len(df) < 2:  # 无昨收(新股首日/停牌)跳过
                    continue
                # 最新bar非今日(K线未落当日)时, 累计量是昨日全天量,
                # 量比无意义置0, 避免脏值进热度/概率公式
                vol = (float(df["volume"].iloc[-1])
                       if str(df.index[-1])[:8] == today else 0.0)
                row = _row(c, float(df["close"].iloc[-1]),
                           float(df["close"].iloc[-2]),
                           vol, float(df["amount"].iloc[-1]),
                           names, avg5, fmv, emin)
                if row:
                    out[c] = row
            except Exception:
                continue
    return out


# ---------- 昨日收盘位置(研究16: S3叠加因子, 盘前每日一次) ----------

def fetch_prev_cpos(codes: list) -> dict:
    """{code: 昨日收盘位置(昨收在昨日高低价区间的位置0-1)}。
    研究16验证: S3a叠加昨收强(>0.6)封板率 89%→92%, EV +2.67→+4.47。
    横截面单请求(FormulaServer快路径), 失败返回空dict(信号降级不阻断)"""
    try:
        xt = _client()
        res = xt.get_market_data_ex(
            field_list=["high", "low", "close"],
            stock_list=[c for c in codes if _valid(c)], period="1d",
            count=2, dividend_type="none", chunk_size=0,
            timeout_seconds=15) or {}
    except Exception as e:
        print(f"[qmt] 昨日收盘位置拉取失败: {e}")
        return {}
    out = {}
    today = datetime.now().strftime("%Y%m%d")
    for c, dfd in res.items():
        try:
            if dfd is None or len(dfd) < 2:
                continue
            # 昨日=最后一根非今日bar(盘前最后bar是昨日, 盘中可能是今日)
            rows = [(str(ix)[:8], hi, lo, cl) for ix, hi, lo, cl
                    in zip(dfd.index, dfd["high"], dfd["low"], dfd["close"])]
            prev = [r for r in rows if r[0] != today]
            if not prev:
                continue
            _, hi, lo, cl = prev[-1]
            hi, lo, cl = float(hi), float(lo), float(cl)
            if hi > lo > 0:
                out[c] = (cl - lo) / (hi - lo)
        except Exception:
            continue
    return out


# ---------- 日bar结构快照(研究24 V5结构层, 盘前每日一次) ----------

def fetch_daily_bars(codes: list, count: int = 22) -> dict:
    """{code: [(date, high, low, close, vol)] 升序}，供 core/structure
    计算 T-1 结构因子(zb_cnt20/y_volr5/neg_streak/行业排名)。
    横截面单请求(FormulaServer快路径), 失败返回空dict(影子降级不阻断)"""
    try:
        xt = _client()
        res = xt.get_market_data_ex(
            field_list=["high", "low", "close", "volume"],
            stock_list=[c for c in codes if _valid(c)], period="1d",
            count=count, dividend_type="none", chunk_size=0,
            timeout_seconds=30) or {}
    except Exception as e:
        print(f"[qmt] 日bar结构快照拉取失败: {e}")
        return {}
    out = {}
    for c, dfd in res.items():
        try:
            if dfd is None or len(dfd) < 21:
                continue
            out[c] = [(str(ix)[:8], float(hi), float(lo), float(cl),
                       float(v))
                      for ix, hi, lo, cl, v in zip(
                          dfd.index, dfd["high"], dfd["low"],
                          dfd["close"], dfd["volume"])]
        except Exception:
            continue
    return out


if __name__ == "__main__":
    # 盘前预建缓存: python -m quotes.qmt build-cache
    if len(sys.argv) > 1 and sys.argv[1] == "build-cache":
        t0 = time.time()
        build_names()
        print(f"[qmt] 缓存预建完成, 耗时{time.time() - t0:.0f}s")
    else:
        sample = ["600519.SH", "000001.SZ", "300750.SZ"]
        print(json.dumps(fetch_quotes(sample), ensure_ascii=False, indent=1))
