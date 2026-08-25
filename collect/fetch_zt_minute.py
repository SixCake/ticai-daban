# -*- coding: utf-8 -*-
"""涨停/触板标的 1分钟K线采集（增量）

每日集合 = 封板组(tushare events_enriched, 当日则用 live/latest.json 池)
         + 炸板组(东财 stock_zt_pool_zbgc_em: 触板未封)
分钟线源: 东财K线API直连(多host切换):
  push2his   —— 多日深度(约5+交易日), 可能IP冷却
  push2delay —— 仅当日深度, 兜底
涨停价:   tushare stk_limit 按日全市场一次

输出: data/minutes/zt_minute_YYYYMMDD.parquet
  date ts_code name grp(sealed/zb) height open_times first_time zb_times
  is_yizi is_st float_mv turnover_ratio fd_amount
  t(HHMM) open high low close vol(手) amount(元) limit_px

用法:
  python collect/fetch_zt_minute.py            # 增量: 最近--max-days个交易日缺失日
  python collect/fetch_zt_minute.py --date 20260825
  python collect/fetch_zt_minute.py --force    # 覆盖已有
"""
import argparse
import json
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

# akshare 内部 requests 无默认超时 → 全局兜底
_orig_request = requests.api.request


def _request_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", 12)
    return _orig_request(*args, **kwargs)


requests.api.request = _request_with_timeout

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from core.codes import ts_code_of  # noqa: E402

MIN_DIR = DATA / "minutes"
SLEEP = 1.2          # push2his 限流敏感, 低速稳定优先
BACKOFF_MAX = 30.0
EM_HOSTS = ["push2his.eastmoney.com", "push2delay.eastmoney.com",
            "92.push2his.eastmoney.com"]
_host_idx = 0


def secid_of(ts_code: str) -> str:
    code, exch = ts_code.split(".")
    return ("1." if exch == "SH" else "0.") + code


def fetch_minute_kline(ts_code: str, retries: int = 2) -> list[str] | None:
    """东财1分钟K线原始字符串列表 ['YYYY-MM-DD HH:MM,o,c,h,l,vol,amount',...]"""
    global _host_idx
    params = {
        "secid": secid_of(ts_code), "klt": "1", "fqt": "0",
        "lmt": "2500", "end": "20500101",
        "fields1": "f1,f2,f3", "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    for attempt in range(retries + 1):
        for k in range(len(EM_HOSTS)):
            host = EM_HOSTS[(_host_idx + k) % len(EM_HOSTS)]
            try:
                r = requests.get(f"https://{host}/api/qt/stock/kline/get",
                                 params=params, timeout=12,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    continue
                d = (r.json().get("data") or {}).get("klines")
                if d:
                    _host_idx = (_host_idx + k) % len(EM_HOSTS)
                    return d
                # 200但空(停牌/无数据)不视为host故障
                return None
            except Exception:
                continue
        time.sleep(1.0 + attempt)
    return None


def probe_depth() -> tuple[str, str]:
    """探测当前可用host的1分钟深度起点, 返回 (起点YYYYMMDD, host)"""
    global _host_idx
    for k in range(len(EM_HOSTS)):
        _host_idx = k
        bars = fetch_minute_kline("600519.SH", retries=0)
        if bars:
            return bars[0][:10].replace("-", ""), EM_HOSTS[k]
    return "", ""


def trade_dates(start: str) -> list[str]:
    import tushare as ts
    import config
    pro = ts.pro_api(config.get_token())
    cal = pro.trade_cal(exchange="SSE", start_date=start,
                        end_date=pd.Timestamp.today().strftime("%Y%m%d"),
                        is_open="1")
    return sorted(cal["cal_date"].tolist())


def sealed_set(pro, date: str) -> pd.DataFrame:
    """封板组: 历史取events_enriched, 当日取live池"""
    ev = pd.read_parquet(DATA / "events_enriched.parquet")
    sub = ev[ev["trade_date"] == date]
    if len(sub):
        return pd.DataFrame({
            "ts_code": sub["ts_code"], "name": sub["name"],
            "height": sub["limit_times"].astype(int),
            "open_times": sub["open_times"].astype(int),
            "first_time": sub["first_time"].astype(str).str.zfill(6),
            "zb_times": 0,
            "is_yizi": sub["is_yizi"].astype(bool),
            "is_st": sub["is_st"].astype(bool),
            "float_mv": sub["float_mv"].astype(float),
            "turnover_ratio": sub["turnover_ratio"].astype(float),
            "fd_amount": sub["fd_amount"].astype(float)})
    lf = DATA / "live" / "latest.json"
    if lf.exists():
        d = json.loads(lf.read_text(encoding="utf-8"))
        if d.get("date") == date:
            rows = [{
                "ts_code": p["ts_code"], "name": p["name"],
                "height": int(p.get("height") or 1),
                "open_times": int(p.get("open_times") or 0),
                "first_time": str(p.get("first_time") or "").zfill(6),
                "zb_times": 0, "is_yizi": False, "is_st": False,
                "float_mv": float("nan"), "turnover_ratio": float("nan"),
                "fd_amount": float(p.get("fd_amount") or float("nan"))}
                for p in d.get("pool", [])]
            return pd.DataFrame(rows)
    return pd.DataFrame()


def zb_set(date: str) -> pd.DataFrame:
    """炸板组: 东财炸板股池"""
    try:
        zb = ak.stock_zt_pool_zbgc_em(date=date)
    except Exception as e:
        print(f"  炸板池 {date} 失败: {e}", flush=True)
        return pd.DataFrame()
    if zb is None or not len(zb):
        return pd.DataFrame()
    return pd.DataFrame({
        "ts_code": [ts_code_of(c) for c in zb["代码"]],
        "name": zb["名称"].values,
        "height": zb["涨停统计"].str.split("/").str[0].astype(int).values,
        "open_times": 0,
        "first_time": zb["首次封板时间"].astype(str).str.zfill(6).values,
        "zb_times": zb["炸板次数"].astype(int).values,
        "is_yizi": False, "is_st": zb["名称"].str.upper().str.contains("ST").values,
        "float_mv": zb["流通市值"].astype(float).values,  # 元, 与tushare一致
        "turnover_ratio": zb["换手率"].astype(float).values,
        "fd_amount": float("nan")})


def limit_prices(pro, date: str) -> dict:
    """当日全市场涨停价 {ts_code: up_limit}"""
    df = pro.stk_limit(trade_date=date)
    if df is None or not len(df):
        return {}
    return dict(zip(df["ts_code"], df["up_limit"].astype(float)))


def fetch_day(pro, date: str, limit: int = 0) -> pd.DataFrame:
    sealed = sealed_set(pro, date)
    sealed["grp"] = "sealed"
    zb = zb_set(date)
    zb["grp"] = "zb"
    uni = pd.concat([sealed, zb], ignore_index=True)
    uni = uni.drop_duplicates(subset="ts_code", keep="first")
    if limit > 0:
        uni = uni.head(limit)
    if not len(uni):
        print(f"  {date}: 空集合", flush=True)
        return pd.DataFrame()
    lp = limit_prices(pro, date)
    miss_lp = uni[~uni["ts_code"].isin(lp)]
    if len(miss_lp):
        print(f"  警告: {len(miss_lp)}只缺涨停价 "
              f"{miss_lp['ts_code'].head(3).tolist()}", flush=True)
    key = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    frames, bad, streak = [], 0, 0
    for n, r in enumerate(uni.itertuples(), 1):
        bars = fetch_minute_kline(r.ts_code)
        time.sleep(SLEEP)
        rows = []
        if bars:
            for b in bars:
                if not b.startswith(key):
                    continue
                tm, o, c, h, l, v, a = b.split(",")
                rows.append((tm[11:16].replace(":", ""), float(o), float(h),
                             float(l), float(c), float(v), float(a)))
        if not rows:
            bad += 1
            streak += 1
            if streak >= 3:  # 连续失败 → 疑似限流, 退避
                wait = min(BACKOFF_MAX, 5.0 * streak)
                print(f"  {date}: 连续失败{streak}, 退避{wait:.0f}s",
                      flush=True)
                time.sleep(wait)
            continue
        streak = 0
        m = pd.DataFrame(rows, columns=["t", "open", "high", "low",
                                        "close", "vol", "amount"])
        frames.append(pd.DataFrame({
            "date": date, "ts_code": r.ts_code, "name": r.name,
            "grp": r.grp, "height": r.height, "open_times": r.open_times,
            "first_time": r.first_time, "zb_times": r.zb_times,
            "is_yizi": r.is_yizi, "is_st": r.is_st,
            "float_mv": r.float_mv, "turnover_ratio": r.turnover_ratio,
            "fd_amount": r.fd_amount,
            "t": m["t"].values, "open": m["open"].values,
            "high": m["high"].values, "low": m["low"].values,
            "close": m["close"].values, "vol": m["vol"].values,
            "amount": m["amount"].values,
            "limit_px": lp.get(r.ts_code, float("nan"))}))
        if n % 20 == 0:
            print(f"  {date}: {n}/{len(uni)} ok={len(frames)} bad={bad}",
                  flush=True)
    if not frames:
        print(f"  {date}: 全部拉取失败", flush=True)
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    print(f"  {date}: {uni['ts_code'].nunique()}只 失败{bad} bars={len(out)} "
          f"封板{len(sealed)}/炸板{len(zb)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="只跑指定日期 YYYYMMDD")
    ap.add_argument("--max-days", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0, help="每日只取前N只(冒烟)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    MIN_DIR.mkdir(parents=True, exist_ok=True)
    start, host = probe_depth()
    if not start:
        print("所有东财host均不可用, 稍后重试", flush=True)
        return
    print(f"分钟线深度起点: {start} (host={host})", flush=True)
    if args.date:
        dates = [args.date]
    else:
        today = pd.Timestamp.today().strftime("%Y%m%d")
        dates = trade_dates(start)
        dates = [d for d in dates if d <= today][-args.max_days:]
    import tushare as ts
    import config
    pro = ts.pro_api(config.get_token())
    for d in dates:
        f = MIN_DIR / f"zt_minute_{d}.parquet"
        if f.exists() and not args.force:
            print(f"  {d}: 已存在, 跳过", flush=True)
            continue
        out = fetch_day(pro, d, limit=args.limit)
        if len(out):
            out.to_parquet(f, index=False)
            print(f"  → {f}", flush=True)
        time.sleep(0.5)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
