# -*- coding: utf-8 -*-
"""复盘快照生成器(离线全历史可用)

build_review(date) → dict:
  题材天梯 / 情绪 / 连板天梯 / 现实格命中事件及其T+1兑现 / 前一交易日现实格的兑现追踪
口径: 现实格 core/reality.py, 角色 core/roles.py（与盘中poller同一出处）
CLI: python review.py [date]  → 写 data/review/review_DATE.json
     python review.py --last N → 批量重生成近N个交易日快照
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from core.attribute import (conf_level, load_con2stock, load_maps,  # noqa: E402
                            touches_of)
from core.reality import reality_mask  # noqa: E402
from core.roles import RoleContext, roles_of  # noqa: E402
from core.shortboard import (HB_LIMIT, N_WINDOW, PEAK_WIN, STATE_ORDER,  # noqa: E402
                            STATE_RISK, build_cohort, shortboard_snapshot,
                            shortboard_state_of)
from datastore import load, path_of  # noqa: E402

_MAPS = None
_CON2STOCK = None
_PANEL = None
_MEMNAMES = None
_DS_CACHE: dict = {}


def _load_cached(name: str, columns: list) -> pd.DataFrame:
    """数据集读取 + 按mtime失效缓存

    server长驻进程下 daily_update 新增行后无需重启即可生效;
    数据集缺失返回空表(展示层自行降级)。
    """
    p = path_of(name)
    if not p.exists():
        return pd.DataFrame()
    mt = p.stat().st_mtime
    ck = _DS_CACHE.get(name)
    if ck is None or ck[0] != mt:
        ck = (mt, load(name, columns=columns))
        _DS_CACHE[name] = ck
    return ck[1]


def _ind_map():
    """ts_code→行业, 供归属置信的行业错位检查(口径见core.attribute)"""
    p = DATA / "meta" / "industry_map.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


IND_MAP = _ind_map()


def _maps():
    global _MAPS
    if _MAPS is None:
        _MAPS = load_maps()
    return _MAPS


def _con2stock():
    global _CON2STOCK
    if _CON2STOCK is None:
        _CON2STOCK = load_con2stock()
    return _CON2STOCK


def _ensure_panel() -> pd.DataFrame:
    """全A日度面板(含high/low, 供中军B与龙头短板峰谷共用), 全局缓存一次"""
    global _PANEL
    if _PANEL is None:
        _PANEL = load("market.daily_panel",
                      columns=["trade_date", "ts_code", "pct_chg",
                               "close", "vol", "high", "low"])
    return _PANEL


def _panel_index(date: str) -> pd.DataFrame:
    """当日行情切片(ts_code索引), 供离线中军B计算"""
    pn = _ensure_panel()
    return pn[pn["trade_date"] == date].set_index("ts_code")


def _panel_bars(codes: list, start: str, end: str) -> dict:
    """{code: [(date, high, low, close, pct_chg)] 升序}, 供龙头短板峰谷/当日bar"""
    pn = _ensure_panel()
    sub = pn[(pn["trade_date"] >= start) & (pn["trade_date"] <= end)
             & (pn["ts_code"].isin(codes))].sort_values(
        ["ts_code", "trade_date"])
    return {c: list(zip(g["trade_date"], g["high"], g["low"], g["close"],
                        g["vol"], g["pct_chg"]))
            for c, g in sub.groupby("ts_code")}


def _factor_rows(date: str, codes: list) -> dict:
    """{code: factor.longtou行(ldlr_prev, =T-1市场值)}; 仅取ldlr_prev环境闸,
    volr5/neg_streak/neg_deep改由struct_from_bars从面板算(与盘中同源)。
    按日期pyarrow过滤只读当日行(~5000), 不全量加载7.6M行。"""
    p = path_of("factor.longtou")
    if not p.exists():
        return {}
    try:
        sub = pd.read_parquet(p, columns=["ts_code", "ldlr_prev"],
                              filters=[("trade_date", "=", date)])
    except Exception:
        return {}
    sub = sub[sub["ts_code"].isin(codes)]
    return {r.ts_code: r for r in sub.itertuples()}


def _memnames():
    global _MEMNAMES
    if _MEMNAMES is None:
        mem = load("theme.members", columns=["con_code", "con_name"])
        out = dict(zip(mem["con_code"], mem["con_name"]))
        try:  # kpl源下中军候选可能仅在开盘啦成分中, 名字映射补齐
            kpl = load("theme.kpl_members", columns=["con_code", "con_name"])
            for c, n in zip(kpl["con_code"], kpl["con_name"]):
                out.setdefault(c, n)
        except FileNotFoundError:
            pass
        _MEMNAMES = out
    return _MEMNAMES


def _themes_of(code: str, prim, touches: dict, cname: dict) -> list[str]:
    """独占主概念在前, 其余当日触及概念按触及家数降序, 截断8个"""
    prim = prim if isinstance(prim, str) and pd.notna(prim) else "-"
    tnames = [cname.get(k, k) for k in touches.get(code, [])]
    if prim == "-":
        return tnames[:8]
    return ([prim] + [n for n in tnames if n != prim])[:8]


def _conf_of(code: str, concept_code, stock2con: dict) -> str:
    """归属置信(与poller同口径, 单一出处在core.attribute):
    候选稀疏或行业错位→low; 仅标记供展示层提示, 不改归属与下游口径"""
    if concept_code is None or (isinstance(concept_code, float)
                                and pd.isna(concept_code)):
        return "none"
    return conf_level(code, concept_code, len(stock2con.get(code, [])),
                      _con2stock(), IND_MAP)


def _reasons(date: str) -> dict:
    """涨停原因字典 {ts_code: {text,tag,status,rate,src}}

    口径: 同花顺涨停池榜单(limitup.ths_limit.lu_desc)为权威源(当日16点后可得,
    lu_desc/tag/status齐备); 榜单外标的(北交所等)用开盘啦事件库lu_desc兜底,
    kpl的status是连板信息(首板/N连板), 归入tag位。rate为近一年封板率%(原接口为0~1比例)。
    """
    out = {}
    kpl = _load_cached("limitup.kpl_events",
                       ["trade_date", "ts_code", "tag", "lu_desc", "status"])
    if len(kpl):
        kp = kpl[(kpl["trade_date"] == date) & (kpl["tag"] == "涨停")]
        for r in kp.itertuples():
            if isinstance(r.lu_desc, str) and r.lu_desc.strip():
                out[r.ts_code] = {"text": r.lu_desc.strip(),
                                  "tag": str(r.status) if pd.notna(r.status)
                                  else "", "status": "", "rate": None,
                                  "src": "kpl"}
    ths = _load_cached("limitup.ths_limit",
                       ["trade_date", "ts_code", "lu_desc", "tag", "status",
                        "limit_up_suc_rate"])
    if len(ths):
        for r in ths[ths["trade_date"] == date].itertuples():
            if isinstance(r.lu_desc, str) and r.lu_desc.strip():
                out[r.ts_code] = {
                    "text": r.lu_desc.strip(),
                    "tag": str(r.tag) if pd.notna(r.tag) else "",
                    "status": str(r.status) if pd.notna(r.status) else "",
                    "rate": (round(float(r.limit_up_suc_rate) * 100, 1)
                             if pd.notna(r.limit_up_suc_rate) else None),
                    "src": "ths"}
    return out


def _lu_fields(code: str, reasons: dict) -> dict:
    """涨停原因附加字段(展示层点击弹层用; 无数据则整组省略, 前端不渲染入口)"""
    r = reasons.get(code)
    if not r:
        return {}
    return {"lu_reason": r["text"], "lu_tag": r["tag"],
            "lu_status": r["status"], "lu_rate": r["rate"], "lu_src": r["src"]}


OUT = DATA / "review"
OUT.mkdir(exist_ok=True)


def _events(date: str) -> pd.DataFrame:
    ev = load("limitup.events_enriched")
    att = load("theme.attribution")
    td = load("theme.day")
    df = ev.merge(att[["trade_date", "ts_code", "concept_code"]],
                  on=["trade_date", "ts_code"], how="left")
    df = df.merge(td[["trade_date", "concept_code", "concept_name", "zt_cnt",
                      "theme_age"]], on=["trade_date", "concept_code"],
                  how="left")
    return df


def _shortboard(date: str, dates: list, day: pd.DataFrame,
                td_all: pd.DataFrame) -> list:
    """高位龙头短板层(研究29验证, 展示/风险标注): 近N日题材龙头∪市场高板,
    今日未涨停, 高位守卫, 5态状态机。前几日龙头今日未涨停也保留可见。"""
    if date not in dates:
        return []
    di = dates.index(date)
    win = dates[max(0, di - N_WINDOW):di]        # 近N日(严格早于当日)
    if not win:
        return []
    # 近N日题材龙头 + 其引领题材名(供天梯归位)
    tw = td_all[td_all["trade_date"].isin(win)]
    led: dict = {}
    for r in tw.itertuples():
        if pd.notna(r.leader_code):
            led.setdefault(r.leader_code, [])
            if r.concept_name not in led[r.leader_code]:
                led[r.leader_code].append(r.concept_name)
    # 近N日市场高板(连板≥HB_LIMIT)
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "name", "limit_times"])
    hw = ev[(ev["trade_date"].isin(win)) & (ev["limit_times"] >= HB_LIMIT)]
    hb_lt = hw.groupby("ts_code")["limit_times"].max().to_dict()
    names = ev.sort_values("trade_date").groupby("ts_code")["name"].last().to_dict()
    cand = (set(led) | set(hb_lt)) - set(day["ts_code"])
    if not cand:
        return []
    start = dates[max(0, di - 32)]        # 覆盖30日累计涨幅窗口(监管异动)
    bars_all = _panel_bars(sorted(cand), start, date)   # 含当日T bar
    prior_bars, day_bar, cur_close = {}, {}, {}
    for c, bl in bars_all.items():
        prior_bars[c] = [b for b in bl if b[0] != date]   # ≤T-1 峰谷/结构参照
        tb = next((b for b in bl if b[0] == date), None)
        if tb is not None:
            day_bar[c] = tb
            cur_close[c] = tb[3]
    # cohort(唯一出处 core.build_cohort, 过高位守卫) + factor行(取ldlr_prev)
    cohort = build_cohort(led, hb_lt, day["ts_code"], prior_bars, cur_close)
    fac = _factor_rows(date, cohort)
    out = []
    for c in cohort:
        snap, pf = shortboard_snapshot(c, prior_bars.get(c, []), fac.get(c),
                                       day_bar=day_bar.get(c))
        if snap is None:                         # 当日无bar(停牌/无成交)
            continue
        state, reason = shortboard_state_of(snap)
        risk = STATE_RISK[state]
        out.append({
            "ts_code": c, "name": names.get(c, c),
            "state": state, "risk": risk["risk"], "level": risk["level"],
            "note": risk["note"],
            "reason": reason, "led_themes": led.get(c, [])[:3],
            "height": int(hb_lt.get(c, 0)),
            "drawdown": (round(pf["peak_drawdown"] * 100, 1)
                         if pf["peak_drawdown"] is not None else None),
            "recovery": (round(pf["pressure_recovery"] * 100)
                         if pf["pressure_recovery"] is not None else None),
            "volr5": (round(snap["volr5"], 2)
                      if snap["volr5"] is not None else None),
            "cpos": round(snap["cpos"], 2), "pct": round(snap["pct"], 2)})
    out.sort(key=lambda x: (STATE_ORDER.index(x["state"]),
                            -(x["drawdown"] or -999)))
    return out


def build_review(date: str) -> dict:
    df = _events(date)
    dates = sorted(df["trade_date"].unique())
    day = df[df["trade_date"] == date].copy()
    if day.empty:
        return {"date": date, "error": "该日期无涨停事件数据"}
    prev = dates[dates.index(date) - 1] if dates.index(date) > 0 else None
    stock2con, msize, cname = _maps()
    _, touches = touches_of(date, day["ts_code"].tolist(), stock2con, msize)
    rsn = _reasons(date)          # 涨停原因(同花顺榜单/kpl兜底)

    td_all = load("theme.day")
    ladder = td_all[td_all["trade_date"] == date].sort_values(
        ["zt_cnt", "max_height"], ascending=False)

    # ---- 角色判定: core.roles单一口径（与研究04一致, 与盘中poller一致） ----
    att_ok = df[df["concept_code"].notna() &
                (df["concept_code"] != "UNASSIGNED")]
    att_set = set(zip(att_ok["trade_date"], att_ok["ts_code"],
                      att_ok["concept_code"]))
    daytd = td_all[td_all["trade_date"] == date]
    rctx = RoleContext(
        leader_by=dict(zip(daytd["concept_code"], daytd["leader_code"])),
        age_by=dict(zip(daytd["concept_code"], daytd["theme_age"])),
        att_set=att_set, dates=dates, date=date)

    # 情绪
    yizi_n = int(day["is_yizi"].sum())
    sentiment = {
        "zt_count": int(len(day)), "yizi_count": yizi_n,
        "broken_board": int((day["open_times"] >= 1).sum()),
        "max_height": int(day["limit_times"].max()) if len(day) else 0,
        "ladder_2plus": int((day["limit_times"] >= 2).sum()),
    }

    # 连板天梯(2板+)
    lb = day[day["limit_times"] >= 2].sort_values(
        ["limit_times", "fd_amount"], ascending=False)
    ladder_stocks = [
        {"ts_code": r.ts_code, "name": r.name, "height": int(r.limit_times),
         "theme": r.concept_name if pd.notna(r.concept_name) else "-",
         "themes": _themes_of(r.ts_code, r.concept_name, touches, cname),
         "attr_conf": _conf_of(r.ts_code, r.concept_code, stock2con),
         "roles": roles_of(rctx, r.ts_code, r.concept_code, int(r.limit_times)),
         "open_times": int(r.open_times), "first_time": str(r.first_time),
         "fd_amount": float(r.fd_amount) if pd.notna(r.fd_amount) else 0,
         "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                           if pd.notna(r.next_open_ret) else None),
         **_lu_fields(r.ts_code, rsn)}
        for r in lb.itertuples()]

    # 当日现实格命中 + 兑现
    rc = day[reality_mask(day)].sort_values("fd_amount", ascending=False)
    rc_list = [
        {"ts_code": r.ts_code, "name": r.name, "height": int(r.limit_times),
         "theme": r.concept_name if pd.notna(r.concept_name) else "-",
         "attr_conf": _conf_of(r.ts_code, r.concept_code, stock2con),
         "theme_cnt": int(r.zt_cnt), "open_times": int(r.open_times),
         "first_time": str(r.first_time), "last_time": str(r.last_time),
         "fd_amount": float(r.fd_amount) if pd.notna(r.fd_amount) else 0,
         "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                           if pd.notna(r.next_open_ret) else None),
         "next_close_ret": (round(float(r.next_close_ret) * 100, 2)
                            if pd.notna(r.next_close_ret) else None),
         **_lu_fields(r.ts_code, rsn)}
        for r in rc.itertuples()]

    # 前一交易日现实格的T+1兑现(=当日验证)
    prev_rc_list = []
    if prev:
        pday = df[df["trade_date"] == prev]
        prc = pday[reality_mask(pday)]
        for r in prc.itertuples():
            prev_rc_list.append({
                "ts_code": r.ts_code, "name": r.name,
                "theme": r.concept_name if pd.notna(r.concept_name) else "-",
                "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                                  if pd.notna(r.next_open_ret) else None),
                "date": prev})

    # 高位龙头短板层(研究29, 展示/风险标注; 前几日龙头今日未涨停也保留可见)
    sb_list = _shortboard(date, dates, day, td_all)

    # 题材天梯 (+离线中军B: 成分内涨幅≥5%且未涨停的成交额最大者)
    con2stock = _con2stock()
    memnames = _memnames()
    zt_set = set(day["ts_code"])
    pn = None
    themes = []
    for r in ladder.head(30).itertuples():
        entry = {"concept_code": r.concept_code, "name": r.concept_name,
                 "zt_cnt": int(r.zt_cnt), "zt_cnt_raw": int(r.zt_cnt_raw),
                 "max_height": int(r.max_height),
                 "theme_age": int(r.theme_age),
                 "leader_name": r.leader_name,
                 "leader_height": int(r.leader_height)}
        # 行业纯度: 当日独占成员主导行业占比, 与盘中poller同口径(展示层标⚠离散)
        sub_inds = day.loc[day["concept_code"] == r.concept_code,
                           "industry"].dropna().tolist()
        if sub_inds:
            top_ind = max(set(sub_inds), key=sub_inds.count)
            entry["ind_top"] = top_ind
            entry["ind_share"] = round(
                sum(1 for i in sub_inds if i == top_ind) / len(sub_inds), 2)
        members = [c for c in con2stock.get(r.concept_code, [])
                   if c not in zt_set]
        if members:
            if pn is None:
                pn = _panel_index(date)
            rows = pn.loc[[c for c in members if c in pn.index]]
            rows = rows[rows["pct_chg"] >= 5]
            if len(rows):
                rows = rows.assign(amount=rows["vol"] * rows["close"])
                rows = rows.dropna(subset=["amount"])
            if len(rows):
                zc = rows["amount"].idxmax()
                entry["zhongjun"] = {
                    "name": memnames.get(zc, zc), "code": zc,
                    "pct": round(float(rows.loc[zc, "pct_chg"]), 2),
                    "amount": float(rows.loc[zc, "amount"])}
        # 前几日龙头今日未涨停 → 挂到该题材下(解决天梯丢弃高位龙头短板)
        sbl = [s for s in sb_list if entry["name"] in s["led_themes"]]
        if sbl:
            entry["shortboard_leaders"] = sbl
        themes.append(entry)

    # 全池明细(供表格)
    pool_list = [
        {"ts_code": r.ts_code, "name": r.name, "height": int(r.limit_times),
         "theme": r.concept_name if pd.notna(r.concept_name) else "-",
         "themes": _themes_of(r.ts_code, r.concept_name, touches, cname),
         "attr_conf": _conf_of(r.ts_code, r.concept_code, stock2con),
         "roles": roles_of(rctx, r.ts_code, r.concept_code, int(r.limit_times)),
         "theme_cnt": int(r.zt_cnt) if pd.notna(r.zt_cnt) else 0,
         "open_times": int(r.open_times), "first_time": str(r.first_time),
         "last_time": str(r.last_time),
         "fd_amount": float(r.fd_amount) if pd.notna(r.fd_amount) else 0,
         "industry": r.industry, "is_yizi": bool(r.is_yizi),
         "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                           if pd.notna(r.next_open_ret) else None),
         **_lu_fields(r.ts_code, rsn)}
        for r in day.sort_values(["limit_times", "fd_amount"],
                                 ascending=False).itertuples()]

    # 现实格历史统计(近30日命中表现)
    stats = {}
    rc_hist = df[reality_mask(df) & df["next_open_ret"].notna()]
    if len(rc_hist):
        recent = rc_hist.tail(200)
        stats = {"hist_n": int(len(rc_hist)),
                 "hist_mean": round(float(rc_hist["next_open_ret"].mean()) * 100, 2),
                 "recent_n": int(len(recent)),
                 "recent_mean": round(float(recent["next_open_ret"].mean()) * 100, 2)}

    return {"date": date, "prev_date": prev, "sentiment": sentiment,
            "themes": themes, "ladder_stocks": ladder_stocks,
            "reality_cells": rc_list, "prev_reality_cells": prev_rc_list,
            "pool": pool_list, "stats": stats, "shortboard": sb_list}


def _write(date: str):
    snap = build_review(date)
    if "error" in snap:
        print(f"复盘快照 {date}: {snap['error']}")
        return
    out = OUT / f"review_{date}.json"
    out.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    n_rsn = sum(1 for p in snap.get("pool", []) if p.get("lu_reason"))
    print(f"复盘快照 {date}: 涨停{snap['sentiment']['zt_count']} "
          f"现实格命中{len(snap.get('reality_cells', []))} "
          f"涨停原因{n_rsn} → {out}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--last":        # 批量重生成近N个交易日快照(口径变更后刷新)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        ev = load("limitup.events_enriched", columns=["trade_date"])
        for d in sorted(ev["trade_date"].unique())[-n:]:
            _write(d)
        return
    date = arg
    if not date:
        ev = load("limitup.events_enriched", columns=["trade_date"])
        date = ev["trade_date"].max()
    _write(date)


if __name__ == "__main__":
    main()
