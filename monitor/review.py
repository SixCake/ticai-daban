# -*- coding: utf-8 -*-
"""复盘快照生成器(离线全历史可用)

build_review(date) → dict:
  题材天梯 / 情绪 / 连板天梯 / 现实格命中事件及其T+1兑现 / 前一交易日现实格的兑现追踪
CLI: python review.py [date]  → 写 data/review/review_DATE.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from build.attribute import load_con2stock, load_maps, touch_map  # noqa: E402

_MAPS = None
_CON2STOCK = None
_PANEL = None
_MEMNAMES = None


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


def _panel_index(date: str) -> pd.DataFrame:
    """当日行情切片(ts_code索引), 供离线中军B计算"""
    global _PANEL
    if _PANEL is None:
        _PANEL = pd.read_parquet(DATA / "daily_panel.parquet",
                                 columns=["trade_date", "ts_code", "pct_chg",
                                          "close", "vol"])
    pn = _PANEL[_PANEL["trade_date"] == date]
    return pn.set_index("ts_code")


def _memnames():
    global _MEMNAMES
    if _MEMNAMES is None:
        mem = pd.read_parquet(DATA / "concept_members.parquet",
                              columns=["con_code", "con_name"])
        _MEMNAMES = dict(zip(mem["con_code"], mem["con_name"]))
    return _MEMNAMES


def _themes_of(code: str, prim, touches: dict, cname: dict) -> list[str]:
    """独占主概念在前, 其余当日触及概念按触及家数降序, 截断8个"""
    prim = prim if isinstance(prim, str) and pd.notna(prim) else "-"
    tnames = [cname.get(k, k) for k in touches.get(code, [])]
    if prim == "-":
        return tnames[:8]
    return ([prim] + [n for n in tnames if n != prim])[:8]


OUT = DATA / "review"
OUT.mkdir(exist_ok=True)


def _events(date: str) -> pd.DataFrame:
    ev = pd.read_parquet(DATA / "events_enriched.parquet")
    att = pd.read_parquet(DATA / "attribution.parquet")
    td = pd.read_parquet(DATA / "theme_day.parquet")
    df = ev.merge(att[["trade_date", "ts_code", "concept_code"]],
                  on=["trade_date", "ts_code"], how="left")
    df = df.merge(td[["trade_date", "concept_code", "concept_name", "zt_cnt",
                      "theme_age"]], on=["trade_date", "concept_code"],
                  how="left")
    return df


def reality_mask(df: pd.DataFrame) -> pd.Series:
    """无前视现实格: 大热点+炸板早回封+午前+炸板≤3"""
    lastm = df["last_time"].astype(str).str.zfill(6)
    return ((df["zt_cnt"] >= 8) & (df["open_times"] >= 1) &
            (df["open_times"] <= 3) & (lastm < "140000") &
            (lastm <= "110000") & (~df["is_yizi"]) & (~df["is_st"]))


def build_review(date: str) -> dict:
    df = _events(date)
    dates = sorted(df["trade_date"].unique())
    day = df[df["trade_date"] == date].copy()
    if day.empty:
        return {"date": date, "error": "该日期无涨停事件数据"}
    prev = dates[dates.index(date) - 1] if dates.index(date) > 0 else None
    stock2con, msize, cname = _maps()
    _, touches = touch_map(day["ts_code"].tolist(), stock2con, msize)

    td_all = pd.read_parquet(DATA / "theme_day.parquet")
    ladder = td_all[td_all["trade_date"] == date].sort_values(
        ["zt_cnt", "max_height"], ascending=False)

    # ---- 角色判定: 龙头/连板/共振/补涨 (与研究04口径一致) ----
    att_ok = df[df["concept_code"].notna() &
                (df["concept_code"] != "UNASSIGNED")]
    att_set = set(zip(att_ok["trade_date"], att_ok["ts_code"],
                      att_ok["concept_code"]))
    dpos = dates.index(date)
    daytd = td_all[td_all["trade_date"] == date]
    leader_by = dict(zip(daytd["concept_code"], daytd["leader_code"]))
    age_by = dict(zip(daytd["concept_code"], daytd["theme_age"]))

    def roles_of(code, k, h):
        if not isinstance(k, str) or k == "UNASSIGNED":
            return []
        roles = []
        age = age_by.get(k, 1)
        if leader_by.get(k) == code:
            roles.append("龙头")
        elif h >= 2:
            roles.append("连板")
        if h == 1 and age == 1:
            roles.append("共振")
        if h <= 2 and age >= 2 and dpos >= 1:
            appeared = any((dates[dpos - i], code, k) in att_set
                           for i in range(1, min(age, dpos + 1)))
            if not appeared:
                roles.append("补涨")
        return roles

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
         "roles": roles_of(r.ts_code, r.concept_code, int(r.limit_times)),
         "open_times": int(r.open_times), "first_time": str(r.first_time),
         "fd_amount": float(r.fd_amount) if pd.notna(r.fd_amount) else 0,
         "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                           if pd.notna(r.next_open_ret) else None)}
        for r in lb.itertuples()]

    # 当日现实格命中 + 兑现
    rc = day[reality_mask(day)].sort_values("fd_amount", ascending=False)
    rc_list = [
        {"ts_code": r.ts_code, "name": r.name, "height": int(r.limit_times),
         "theme": r.concept_name if pd.notna(r.concept_name) else "-",
         "theme_cnt": int(r.zt_cnt), "open_times": int(r.open_times),
         "first_time": str(r.first_time), "last_time": str(r.last_time),
         "fd_amount": float(r.fd_amount) if pd.notna(r.fd_amount) else 0,
         "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                           if pd.notna(r.next_open_ret) else None),
         "next_close_ret": (round(float(r.next_close_ret) * 100, 2)
                            if pd.notna(r.next_close_ret) else None)}
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
        themes.append(entry)

    # 全池明细(供表格)
    pool_list = [
        {"ts_code": r.ts_code, "name": r.name, "height": int(r.limit_times),
         "theme": r.concept_name if pd.notna(r.concept_name) else "-",
         "themes": _themes_of(r.ts_code, r.concept_name, touches, cname),
         "roles": roles_of(r.ts_code, r.concept_code, int(r.limit_times)),
         "theme_cnt": int(r.zt_cnt) if pd.notna(r.zt_cnt) else 0,
         "open_times": int(r.open_times), "first_time": str(r.first_time),
         "last_time": str(r.last_time),
         "fd_amount": float(r.fd_amount) if pd.notna(r.fd_amount) else 0,
         "industry": r.industry, "is_yizi": bool(r.is_yizi),
         "next_open_ret": (round(float(r.next_open_ret) * 100, 2)
                           if pd.notna(r.next_open_ret) else None)}
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
            "pool": pool_list, "stats": stats}


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        ev = pd.read_parquet(DATA / "events_enriched.parquet",
                             columns=["trade_date"])
        date = ev["trade_date"].max()
    snap = build_review(date)
    if "error" in snap:
        print(f"复盘快照 {date}: {snap['error']}")
        return
    out = OUT / f"review_{date}.json"
    out.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    print(f"复盘快照 {date}: 涨停{snap['sentiment']['zt_count']} "
          f"现实格命中{len(snap.get('reality_cells', []))} → {out}")


if __name__ == "__main__":
    main()
