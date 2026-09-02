# -*- coding: utf-8 -*-
"""盘中实时轮询引擎

每60秒拉取akshare当日涨停池 → 实时独占归属算题材天梯 → 现实格候选识别
 → 写 data/live/latest.json + data/live/intraday_YYYYMMDD.jsonl

现实格候选(core/reality.py研究口径的盘中实时近似, 涨停池无is_yizi/is_st
字段, 由池自身过滤承担): 题材独占涨停≥8家 + 炸板≥1次后回封 + 炸板≤3次
+ 最后封板≤11:00(午前回封)
炸板池接口被网络封锁, 用涨停池快照diff做断板状态机: 出池=炸板, 再入池=回封
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, get_pro  # noqa: E402
from core.attribute import (attribute_of, conf_level, load_con2stock,  # noqa: E402
                            load_maps, touches_of)
from core.calendar import is_polling_hours  # noqa: E402
from core.cycle import (divg_of, height_dist,  # noqa: E402
                        load_prev_market_state as calc_prev_state,
                        stage_of, theme_mode)
from core.roles import RoleContext, roles_of  # noqa: E402
from core.shortboard import (HB_LIMIT, N_WINDOW, PEAK_WIN, STATE_ORDER,  # noqa: E402
                            STATE_RISK, build_cohort, shortboard_snapshot,
                            shortboard_state_of)
from datastore import load, path_of, save  # noqa: E402
from quotes import fetch_quotes  # noqa: E402  # QUOTE_SOURCE分发: tx|qmt
from quotes.tx import fetch_quotes as fetch_quotes_tx  # noqa: E402  # qmt失败兜底
from quotes.zt_pool import fetch_pool, norm_pool  # noqa: E402

LIVE = DATA / "live"
LIVE.mkdir(exist_ok=True)
INTERVAL = 60
SLOW_INTERVAL = 300  # 非交易时段

pro = get_pro()


def load_industry_map() -> dict:
    """ts_code→申万行业, 供归属置信的行业错位检查(置信口径见core.attribute)"""
    p = DATA / "meta" / "industry_map.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


IND_MAP = load_industry_map()


def trade_days_upto(end: str) -> list[str]:
    cache = path_of("meta.trade_cal")
    if cache.exists():
        days = load("meta.trade_cal")["cal_date"].tolist()
        if days and days[-1] >= end:
            return [d for d in days if d <= end]
    cal = pro.trade_cal(exchange="SSE", start_date="20190101", end_date=end,
                        is_open="1")
    days = sorted(cal["cal_date"].tolist())
    save("meta.trade_cal", cal)
    return days


_prev_state_cache: dict = {}


def get_prev_state(date: str):
    """昨日市场级状态(带缓存, daily_panel现算, 92科比框架展示参照)"""
    if date not in _prev_state_cache:
        _prev_state_cache[date] = calc_prev_state(date)
        print(f"昨日市场状态({date}): {_prev_state_cache[date]}")
    return _prev_state_cache[date]


def _shortboard_baseline(date: str, days: list):
    """龙头短板cohort基线: 近N日题材龙头(∪引领题材名) + 市场高板(连板≥HB_LIMIT)
    + 名称。days=交易日历(升序)。返回 (led, hb_lt, names)。"""
    di = days.index(date) if date in days else len(days)
    win = days[max(0, di - N_WINDOW):di]        # 近N日(严格早于当日)
    led, hb_lt, names = {}, {}, {}
    if not win:
        return led, hb_lt, names
    try:
        td = load("theme.day",
                  columns=["trade_date", "concept_name", "leader_code"])
        for r in td[td["trade_date"].isin(win)].itertuples():
            if pd.notna(r.leader_code):
                led.setdefault(r.leader_code, [])
                if r.concept_name not in led[r.leader_code]:
                    led[r.leader_code].append(r.concept_name)
        ev = load("limitup.events_enriched",
                  columns=["trade_date", "ts_code", "name", "limit_times"])
        hw = ev[(ev["trade_date"].isin(win)) & (ev["limit_times"] >= HB_LIMIT)]
        hb_lt = hw.groupby("ts_code")["limit_times"].max().to_dict()
        names = ev.sort_values("trade_date").groupby("ts_code")["name"].last().to_dict()
    except Exception as e:
        print(f"[shortboard] 基线构建失败: {e}")
    return led, hb_lt, names


def _shortboard_bars(codes, date: str) -> dict:
    """{code: [(date,high,low,close,vol,pct_chg)] 升序, 严格<T的末PEAK_WIN+2根},
    供盘中龙头短板峰谷/结构字段。过滤trade_date<date保证恒≤T-1(与复盘
    prior_bars同口径), 不受面板是否已补到当日影响。"""
    codes = list(codes)
    if not codes:
        return {}
    try:
        pn = load("market.daily_panel",
                  columns=["trade_date", "ts_code", "high", "low",
                           "close", "vol", "pct_chg"])
    except Exception as e:
        print(f"[shortboard] 面板加载失败: {e}")
        return {}
    sub = pn[(pn["ts_code"].isin(codes))
             & (pn["trade_date"] < date)].sort_values(["ts_code", "trade_date"])
    return {c: [(r.trade_date, r.high, r.low, r.close, r.vol, r.pct_chg)
                for r in g.tail(32).itertuples()]   # 32根覆盖30日累计涨幅窗口
            for c, g in sub.groupby("ts_code")}


class DayState:
    def __init__(self, date: str, stock2con: dict, msize: dict, cname: dict,
                 age_base: dict, con2stock: dict, att_set: set,
                 att_dates: list, prev_state=None,
                 sb_led=None, sb_hb_lt=None, sb_names=None, sb_bars=None):
        self.date = date
        self.stock2con = stock2con
        self.msize = msize
        self.cname = cname
        self.age_base = age_base      # concept -> 截至上一交易日的连续活跃天数
        self.con2stock = con2stock
        self.att_set = att_set        # (trade_date, ts_code, concept_code) 历史归属
        self.att_dates = att_dates
        self.prev_state = prev_state  # 昨日market_state(情绪周期参照)
        # 龙头短板层(研究29, 展示/风险标注): 基线cohort + ≤T-1日bar
        self.sb_led = sb_led or {}        # code -> 引领题材名
        self.sb_hb_lt = sb_hb_lt or {}    # code -> 窗口内最大连板
        self.sb_names = sb_names or {}
        self.sb_bars = sb_bars or {}
        self.sb_prior = {}                # code -> 上次状态(失效粘滞)
        self.pool_codes: set = set()
        self.exits: dict = {}         # ts_code -> 首次出池时刻(今日炸板未回封候选)
        self.exit_count = 0
        self.history = []             # 当日sentiment序列(情绪周期趋势输入)

    def update(self, pool: pd.DataFrame, ts_str: str):
        codes = pool["ts_code"].tolist()
        new_set = set(codes)
        # 出池 = 炸板(或开板回落)
        for c in self.pool_codes - new_set:
            if c not in self.exits:
                self.exits[c] = ts_str
                self.exit_count += 1
        self.pool_codes = new_set

        # 题材归属统一入口(core.attribute, kpl源直标/延续, ths源投票)
        attr, _src = attribute_of(self.date, codes, self.stock2con,
                                  self.msize)
        raw_cnt, touches = touches_of(self.date, codes, self.stock2con,
                                       self.msize)
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
            # 行业纯度: 独占成员主导行业占比, <60%且≥3只=虹吸嫌疑(展示层标⚠离散)
            inds = rows["所属行业"].dropna().tolist()
            top_ind, ind_share = "-", None
            if inds:
                top_ind = max(set(inds), key=inds.count)
                ind_share = round(sum(1 for i in inds if i == top_ind) / len(inds), 2)
            ladder.append({
                "concept_code": k, "name": self.cname.get(k, k),
                "zt_cnt": len(t["codes"]),
                "zt_cnt_raw": raw_cnt.get(k, 0),
                "max_height": int(rows["连板数"].max()),
                "theme_age": age,
                "mode": theme_mode(age),
                "leader_code": leader.name, "leader_name": leader["名称"],
                "leader_height": int(leader["连板数"]),
                "ind_top": top_ind, "ind_share": ind_share,
            })
        ladder.sort(key=lambda x: (-x["zt_cnt"], -x["max_height"]))

        zt_by_concept = {t["concept_code"]: t["zt_cnt"] for t in ladder}

        # ---- 角色徽章: 龙头/连板/共振/补涨 (core.roles单一口径) ----
        rctx = RoleContext(
            leader_by={t["concept_code"]: t["leader_code"] for t in ladder},
            age_by={t["concept_code"]: t["theme_age"] for t in ladder},
            att_set=self.att_set, dates=self.att_dates, date=self.date)

        # ---- 活中军B: 腾讯实时报价, 热门题材成分内涨幅>=5%成交额最大者 ----
        hot = [t for t in ladder if t["zt_cnt"] >= 4][:6]
        want = []
        for t in hot:
            want.extend(self.con2stock.get(t["concept_code"], []))
        want = [c for c in dict.fromkeys(want) if c not in self.pool_codes][:1200]
        quotes = fetch_quotes(want) if want else {}
        if want and not quotes:   # qmt源失败 → 腾讯兜底, 中军识别不断供
            quotes = fetch_quotes_tx(want)
        for t in hot:
            cands = [quotes[c] for c in self.con2stock.get(t["concept_code"], [])
                     if c in quotes and quotes[c]["pct"] >= 5]
            if cands:
                zj = max(cands, key=lambda x: x["amount"])
                t["zhongjun"] = {"name": zj["name"], "pct": zj["pct"],
                                 "amount": zj["amount"]}
        # ---- 高位龙头短板(研究29, 展示/风险标注): cohort-今日池, 报价+≤T-1结构算5态 ----
        sb_cand = [c for c in dict.fromkeys(list(self.sb_led) + list(self.sb_hb_lt))
                   if c not in new_set]
        sb_list = []
        if sb_cand:
            sbq = fetch_quotes(sb_cand)
            if not sbq:                       # qmt失败 → 腾讯兜底
                sbq = fetch_quotes_tx(sb_cand)
            cur_close = {c: sbq[c].get("price") for c in sb_cand if c in sbq}
            # cohort(唯一出处 core.build_cohort, 过高位守卫)
            cohort = build_cohort(self.sb_led, self.sb_hb_lt, new_set,
                                  self.sb_bars, cur_close)
            for c in cohort:
                snap_sb, pf = shortboard_snapshot(c, self.sb_bars.get(c, []),
                                                  None, live_quote=sbq.get(c))
                if snap_sb is None:
                    continue
                state, reason = shortboard_state_of(snap_sb, self.sb_prior.get(c))
                self.sb_prior[c] = state
                risk = STATE_RISK[state]
                sb_list.append({
                    "ts_code": c,
                    "name": (sbq.get(c, {}).get("name")
                             or self.sb_names.get(c, c)),
                    "state": state, "risk": risk["risk"], "level": risk["level"],
                    "note": risk["note"],
                    "reason": reason, "led_themes": self.sb_led.get(c, [])[:3],
                    "height": int(self.sb_hb_lt.get(c, 0)),
                    "drawdown": (round(pf["peak_drawdown"] * 100, 1)
                                 if pf["peak_drawdown"] is not None else None),
                    "recovery": (round(pf["pressure_recovery"] * 100)
                                 if pf["pressure_recovery"] is not None else None),
                    "volr5": (round(snap_sb["volr5"], 2)
                              if snap_sb["volr5"] is not None else None),
                    "cpos": round(snap_sb["cpos"], 2),
                    "pct": round(snap_sb["pct"], 2)})
        sb_list.sort(key=lambda x: (STATE_ORDER.index(x["state"]),
                                    -(x["drawdown"] or -999)))
        for t in ladder:                       # 前几日龙头挂回其引领题材
            sbl = [s for s in sb_list if t["name"] in s["led_themes"]]
            if sbl:
                t["shortboard_leaders"] = sbl
        # 现实格候选(core/reality.py口径的盘中近似)
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
            prim = self.cname.get(k, k) if k not in (None, "UNASSIGNED") else "-"
            tnames = [self.cname.get(k2, k2)
                      for k2 in touches.get(r["ts_code"], [])]
            themes_list = ([prim] + [n for n in tnames if n != prim]
                           if prim != "-" else tnames)
            # 归属置信(单一口径在core.attribute): 候选稀疏或行业错位→low;
            # 仅标记不改归属, 现实格/角色等下游口径不变
            cand_n = len(self.stock2con.get(r["ts_code"], []))
            conf = conf_level(r["ts_code"], k, cand_n,
                              self.con2stock, IND_MAP)
            pool_list.append({
                "ts_code": r["ts_code"], "name": r["名称"],
                "height": int(r["连板数"]), "theme": prim,
                "themes": themes_list[:8],
                "attr_conf": conf,
                "roles": roles_of(rctx, r["ts_code"], k, int(r["连板数"])),
                "theme_cnt": int(zt_by_concept.get(k, 0)) if k in zt_by_concept else 0,
                "open_times": int(r["炸板次数"]),
                "divg": divg_of(int(r["炸板次数"])),
                "first_time": r["first_time"], "last_time": r["last_time"],
                "fd_amount": float(r["封板资金"]),
                "pct": float(r["涨跌幅"]), "industry": r["所属行业"],
            })

        heights = pool["连板数"].max() if len(pool) else 0
        # 分歧/一致维度(研究28仓位驱动): 炸板率=分歧; 一字代理+缩量加速=一致
        n_pool = len(pool)
        accel = float(((pool["炸板次数"] == 0)
                       & (pool["first_time"].astype(str) <= "094500")).mean()) \
            if n_pool else 0.0
        yizi_proxy = float(
            (pool["first_time"].astype(str) <= "093000").mean()) \
            if n_pool else 0.0
        sentiment = {
            "zt_count": len(pool), "exit_count": self.exit_count,
            "broken_rate": round(self.exit_count / max(1, len(pool) + self.exit_count), 3),
            "max_height": int(heights),
            "ladder_2plus": int((pool["连板数"] >= 2).sum()) if len(pool) else 0,
            "candidates": len(candidates),
            "accel": round(accel, 3), "yizi_proxy": round(yizi_proxy, 3),
        }
        # 92科比框架: 情绪四阶段+高/中/低分布(展示口径, core/cycle.py唯一出处)
        self.history.append(sentiment)
        cycle = stage_of(self.history, sentiment, self.prev_state)
        cycle["dist"] = height_dist(pool_list)

        snap = {
            "ts": ts_str, "date": self.date,
            "status": "live" if is_polling_hours(datetime.now()) else "snapshot",
            "sentiment": sentiment, "cycle": cycle, "themes": ladder,
            "candidates": candidates, "pool": pool_list, "shortboard": sb_list,
            "exits": [{"ts_code": c, "time": t} for c, t in self.exits.items()],
        }
        (LIVE / "latest.json").write_text(
            json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        slim = {"ts": ts_str, "sentiment": sentiment,
                "stage": cycle["stage"],
                "top_themes": [(t["name"], t["zt_cnt"]) for t in ladder[:5]],
                "candidates": [c["ts_code"] for c in candidates]}
        with open(LIVE / f"intraday_{self.date}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
        return snap


def main():
    stock2con, msize, cname = load_maps()
    con2stock = load_con2stock()
    att = load("theme.attribution")
    att = att[att["concept_code"] != "UNASSIGNED"]
    att_set = set(zip(att["trade_date"], att["ts_code"], att["concept_code"]))
    att_dates = sorted(att["trade_date"].unique())
    now = datetime.now()
    today = now.strftime("%Y%m%d")
    days = trade_days_upto(today)
    last_td = days[-1]
    # 连续活跃天数基线(截至parquet最后日期)
    age_base = {}
    tdf = path_of("theme.day")
    if tdf.exists():
        td = load("theme.day", columns=["trade_date", "concept_code"])
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
        trading = is_polling_hours(now)
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
                sb_led, sb_hb_lt, sb_names = _shortboard_baseline(target, days)
                sb_bars = _shortboard_bars(set(sb_led) | set(sb_hb_lt), target)
                state = DayState(target, stock2con, msize, cname, age_base,
                                 con2stock, att_set, att_dates,
                                 prev_state=get_prev_state(target),
                                 sb_led=sb_led, sb_hb_lt=sb_hb_lt,
                                 sb_names=sb_names, sb_bars=sb_bars)
                cur_date = target
                print(f"[{now:%H:%M:%S}] 初始化 {target}, 池内 {len(pool)} 只, "
                      f"龙头短板cohort {len(set(sb_led) | set(sb_hb_lt))} 只")
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
