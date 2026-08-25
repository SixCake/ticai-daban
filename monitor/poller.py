# -*- coding: utf-8 -*-
"""盘中实时轮询引擎

每60秒拉取akshare当日涨停池 → 实时独占归属算题材天梯 → 现实格候选识别
 → 写 data/live/latest.json + data/live/intraday_YYYYMMDD.jsonl

现实格条件(研究02无前视格): 题材独占涨停≥8家 + 炸板≥1次后回封 +
  炸板≤3次 + 最后封板≤11:00(午前回封)
炸板池接口被网络封锁, 用涨停池快照diff做断板状态机: 出池=炸板, 再入池=回封
"""
import bisect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, get_pro  # noqa: E402
from build.attribute import (attribute_day, load_con2stock, load_maps,  # noqa: E402
                             touch_map)
from tx_quote import fetch_quotes  # noqa: E402

LIVE = DATA / "live"
LIVE.mkdir(exist_ok=True)
INTERVAL = 60
SLOW_INTERVAL = 300  # 非交易时段

pro = get_pro()


def ts_code_of(code6: str) -> str:
    code6 = str(code6).zfill(6)
    if code6.startswith(("60", "68")):
        return f"{code6}.SH"
    if code6.startswith(("00", "30")):
        return f"{code6}.SZ"
    return f"{code6}.BJ"


def trade_days_upto(end: str) -> list[str]:
    cache = DATA / "trade_cal_cache.parquet"
    if cache.exists():
        days = pd.read_parquet(cache)["cal_date"].tolist()
        if days and days[-1] >= end:
            return [d for d in days if d <= end]
    cal = pro.trade_cal(exchange="SSE", start_date="20190101", end_date=end,
                        is_open="1")
    days = sorted(cal["cal_date"].tolist())
    cal.to_parquet(cache, index=False)
    return days


def is_trading_now(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return 915 <= hm <= 1505


def fetch_pool(date: str) -> pd.DataFrame | None:
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.stock_zt_pool_em(date=date)
            return df if df is not None and len(df) else None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def norm_pool(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    p["ts_code"] = p["代码"].astype(str).str.zfill(6).map(ts_code_of)
    p["first_time"] = p["首次封板时间"].astype(str).str.zfill(6)
    p["last_time"] = p["最后封板时间"].astype(str).str.zfill(6)
    return p


class DayState:
    def __init__(self, date: str, stock2con: dict, msize: dict, cname: dict,
                 age_base: dict, con2stock: dict, att_set: set,
                 att_dates: list):
        self.date = date
        self.stock2con = stock2con
        self.msize = msize
        self.cname = cname
        self.age_base = age_base      # concept -> 截至上一交易日的连续活跃天数
        self.con2stock = con2stock
        self.att_set = att_set        # (trade_date, ts_code, concept_code) 历史归属
        self.att_dates = att_dates
        self.pool_codes: set = set()
        self.exits: dict = {}         # ts_code -> 首次出池时刻(今日炸板未回封候选)
        self.exit_count = 0
        self.history = []

    def update(self, pool: pd.DataFrame, ts_str: str):
        codes = pool["ts_code"].tolist()
        new_set = set(codes)
        # 出池 = 炸板(或开板回落)
        for c in self.pool_codes - new_set:
            if c not in self.exits:
                self.exits[c] = ts_str
                self.exit_count += 1
        self.pool_codes = new_set

        attr, rounds = attribute_day(codes, self.stock2con, self.msize)
        raw_cnt, touches = touch_map(codes, self.stock2con, self.msize)
        # 题材天梯
        themes = {}
        info = pool.set_index("ts_code")
        for c, k in attr.items():
            if k == "UNASSIGNED":
                continue
            t = themes.setdefault(k, {"codes": []})
            t["codes"].append(c)
        ladder = []
        for k, t in themes.items():
            rows = info.loc[[c for c in t["codes"] if c in info.index]]
            if rows.empty:
                continue
            rows = rows.sort_values(["连板数", "封板资金", "first_time", "炸板次数"],
                                    ascending=[False, False, True, True])
            leader = rows.iloc[0]
            age = self.age_base.get(k, 0) + 1
            ladder.append({
                "concept_code": k, "name": self.cname.get(k, k),
                "zt_cnt": len(t["codes"]),
                "zt_cnt_raw": raw_cnt.get(k, 0),
                "max_height": int(rows["连板数"].max()),
                "theme_age": age,
                "leader_code": leader.name, "leader_name": leader["名称"],
                "leader_height": int(leader["连板数"]),
            })
        ladder.sort(key=lambda x: (-x["zt_cnt"], -x["max_height"]))

        zt_by_concept = {t["concept_code"]: t["zt_cnt"] for t in ladder}

        # ---- 角色徽章: 龙头/连板/共振/补涨 ----
        age_by = {t["concept_code"]: t["theme_age"] for t in ladder}
        leader_by = {t["concept_code"]: t["leader_code"] for t in ladder}
        dpos = bisect.bisect_left(self.att_dates, self.date) - 1

        def roles_of(code, k, h):
            roles = []
            age = age_by.get(k, 1)
            if leader_by.get(k) == code:
                roles.append("龙头")
            elif h >= 2:
                roles.append("连板")
            if h == 1 and age == 1:
                roles.append("共振")
            if h <= 2 and age >= 2 and dpos >= 0:
                appeared = any((self.att_dates[dpos - i], code, k) in self.att_set
                               for i in range(0, min(age - 1, dpos + 1)))
                if not appeared:
                    roles.append("补涨")
            return roles

        # ---- 活中军B: 腾讯实时报价, 热门题材成分内涨幅>=5%成交额最大者 ----
        hot = [t for t in ladder if t["zt_cnt"] >= 4][:6]
        want = []
        for t in hot:
            want.extend(self.con2stock.get(t["concept_code"], []))
        want = [c for c in dict.fromkeys(want) if c not in self.pool_codes][:1200]
        quotes = fetch_quotes(want) if want else {}
        for t in hot:
            cands = [quotes[c] for c in self.con2stock.get(t["concept_code"], [])
                     if c in quotes and quotes[c]["pct"] >= 5]
            if cands:
                zj = max(cands, key=lambda x: x["amount"])
                t["zhongjun"] = {"name": zj["name"], "pct": zj["pct"],
                                 "amount": zj["amount"]}
        # 现实格候选
        candidates = []
        for _, r in pool.iterrows():
            k = attr.get(r["ts_code"])
            if k == "UNASSIGNED" or k is None:
                continue
            zt_cnt = zt_by_concept.get(k, 0)
            if (zt_cnt >= 8 and r["炸板次数"] >= 1 and r["炸板次数"] <= 3
                    and r["last_time"] <= "110000"):
                candidates.append({
                    "ts_code": r["ts_code"], "name": r["名称"],
                    "height": int(r["连板数"]), "theme": self.cname.get(k, k),
                    "theme_cnt": int(zt_cnt), "open_times": int(r["炸板次数"]),
                    "first_time": r["first_time"], "last_time": r["last_time"],
                    "fd_amount": float(r["封板资金"]),
                    "float_mv": float(r["流通市值"]),
                    "industry": r["所属行业"],
                })
        candidates.sort(key=lambda x: (-x["theme_cnt"], -x["height"],
                                       x["last_time"]))

        pool_list = []
        for _, r in pool.sort_values(["连板数", "封板资金"],
                                     ascending=[False, False]).iterrows():
            k = attr.get(r["ts_code"])
            prim = self.cname.get(k, "-") if k not in (None, "UNASSIGNED") else "-"
            tnames = [self.cname.get(k2, k2)
                      for k2 in touches.get(r["ts_code"], [])]
            themes_list = ([prim] + [n for n in tnames if n != prim]
                           if prim != "-" else tnames)
            pool_list.append({
                "ts_code": r["ts_code"], "name": r["名称"],
                "height": int(r["连板数"]), "theme": prim,
                "themes": themes_list[:8],
                "roles": roles_of(r["ts_code"], k, int(r["连板数"])),
                "theme_cnt": int(zt_by_concept.get(k, 0)) if k in zt_by_concept else 0,
                "open_times": int(r["炸板次数"]),
                "first_time": r["first_time"], "last_time": r["last_time"],
                "fd_amount": float(r["封板资金"]),
                "pct": float(r["涨跌幅"]), "industry": r["所属行业"],
            })

        heights = pool["连板数"].max() if len(pool) else 0
        sentiment = {
            "zt_count": len(pool), "exit_count": self.exit_count,
            "broken_rate": round(self.exit_count / max(1, len(pool) + self.exit_count), 3),
            "max_height": int(heights),
            "ladder_2plus": int((pool["连板数"] >= 2).sum()) if len(pool) else 0,
            "candidates": len(candidates),
        }

        snap = {
            "ts": ts_str, "date": self.date,
            "status": "live" if is_trading_now(datetime.now()) else "snapshot",
            "sentiment": sentiment, "themes": ladder,
            "candidates": candidates, "pool": pool_list,
            "exits": [{"ts_code": c, "time": t} for c, t in self.exits.items()],
        }
        (LIVE / "latest.json").write_text(
            json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        slim = {"ts": ts_str, "sentiment": sentiment,
                "top_themes": [(t["name"], t["zt_cnt"]) for t in ladder[:5]],
                "candidates": [c["ts_code"] for c in candidates]}
        with open(LIVE / f"intraday_{self.date}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
        return snap


def main():
    stock2con, msize, cname = load_maps()
    con2stock = load_con2stock()
    att = pd.read_parquet(DATA / "attribution.parquet")
    att = att[att["concept_code"] != "UNASSIGNED"]
    att_set = set(zip(att["trade_date"], att["ts_code"], att["concept_code"]))
    att_dates = sorted(att["trade_date"].unique())
    now = datetime.now()
    today = now.strftime("%Y%m%d")
    days = trade_days_upto(today)
    last_td = days[-1]
    # 连续活跃天数基线(截至parquet最后日期)
    age_base = {}
    tdf = DATA / "theme_day.parquet"
    if tdf.exists():
        td = pd.read_parquet(tdf, columns=["trade_date", "concept_code"])
        last_pd_date = td["trade_date"].max()
        dates_sorted = sorted(td["trade_date"].unique())
        pos = {d: i for i, d in enumerate(dates_sorted)}
        for k, grp in td.groupby("concept_code"):
            p = grp["trade_date"].map(pos).values
            streak = 1
            for i in range(len(p) - 1, 0, -1):
                if p[i] == p[i - 1] + 1:
                    streak += 1
                else:
                    break
            if p[-1] == len(dates_sorted) - 1:
                age_base[k] = streak
        print(f"题材年龄基线: 截至{last_pd_date}, {len(age_base)}个题材")

    state = None
    cur_date = None
    print(f"轮询引擎启动, 最近交易日 {last_td}, 间隔 {INTERVAL}s")
    while True:
        now = datetime.now()
        trading = is_trading_now(now)
        today = now.strftime("%Y%m%d")
        hm = now.strftime("%H%M")
        # akshare对未开盘日期会返回上一交易日数据, 须自行定标
        if today in days and hm >= "0915":
            target = today          # 盘中实时 / 收盘后当日终值
        elif today in days:
            target = days[-2] if len(days) >= 2 else days[-1]  # 开盘前看前一交易日
        else:
            target = last_td        # 周末/节假日看最近交易日
        if state is None or cur_date != target:
            pool = fetch_pool(target)
            if pool is not None:
                state = DayState(target, stock2con, msize, cname, age_base,
                                 con2stock, att_set, att_dates)
                cur_date = target
                print(f"[{now:%H:%M:%S}] 初始化 {target}, 池内 {len(pool)} 只")
        if state is not None:
            pool = fetch_pool(cur_date)
            if pool is not None:
                snap = state.update(norm_pool(pool), now.strftime("%H:%M:%S"))
                s = snap["sentiment"]
                print(f"[{now:%H:%M:%S}] {cur_date} 涨停{s['zt_count']} "
                      f"炸板{s['exit_count']} 候选{s['candidates']} "
                      f"TOP题材:{[t['name'] for t in snap['themes'][:3]]}")
            else:
                print(f"[{now:%H:%M:%S}] 拉取失败, 保留上一快照")
        time.sleep(INTERVAL if trading else SLOW_INTERVAL)


if __name__ == "__main__":
    main()
