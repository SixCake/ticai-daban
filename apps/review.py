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
from core.cycle import theme_stage  # noqa: E402
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
    """全A日度面板(含high/low, 供中军B与龙头断板峰谷共用), 全局缓存一次"""
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
    """{code: [(date, high, low, close, pct_chg)] 升序}, 供龙头断板峰谷/当日bar"""
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
    """独占主概念在前, 其余当日关联概念按关联家数降序, 截断8个"""
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

# ---------- 龙虎榜·游资席位(hmlist.picks, 每日复盘的昨日兑现总结+今日信号) ----------

HM_RELIABILITY_WIN = 40   # 席位可靠性榜窗口(有picks的交易日数)


def _hm_picks_rows(df: pd.DataFrame) -> list:
    def _f(v, nd=2):
        return round(float(v), nd) if pd.notna(v) else None
    out = []
    for r in df.sort_values("score", ascending=False,
                            na_position="last").itertuples():
        out.append({
            "hm_name": r.hm_name, "ts_code": r.ts_code, "ts_name": r.ts_name,
            "grade": r.grade if isinstance(r.grade, str) else "NA",
            "score": _f(r.score, 1),
            "hist_signals": int(r.hist_signals) if pd.notna(r.hist_signals) else None,
            "hist_win_rate": _f(r.hist_win_rate, 1),
            "pred_ret1": _f(r.pred_ret1),
            "buy_amount": float(r.buy_amount) if pd.notna(r.buy_amount) else None,
            "net_amount": float(r.net_amount) if pd.notna(r.net_amount) else None,
            "base_pct": _f(r.base_pct),
            "act_open_ret": _f(r.act_open_ret),
            "act_close_ret": _f(r.act_close_ret),
            "verdict": r.verdict if isinstance(r.verdict, str) else "pending",
            "hit_top": bool(r.hit_top) if pd.notna(r.hit_top) else False})
    return out


def _hm_block(date: str) -> dict:
    """龙虎榜席位块: 今日TOP1信号 + 昨日兑现总结 + 席位可靠性榜。
    picks 缺失或当日无数据时返回 {}(前端整节隐藏)。"""
    if not path_of("hmlist.picks").exists():
        return {}
    picks = _load_cached("hmlist.picks", None)
    if picks.empty:
        return {}
    picks = picks.assign(trade_date=picks["trade_date"].astype(str))
    pdates = sorted(picks["trade_date"].unique())

    today = picks[picks["trade_date"] == date]
    prev_d = max((d for d in pdates if d < date), default=None)
    prev = picks[picks["trade_date"] == prev_d] if prev_d else picks.iloc[0:0]

    # 昨日兑现总结(已验证行)
    summary = {}
    done = prev[prev["verdict"].isin(["win", "lose"])]
    if len(done):
        top = done[done["hit_top"]]
        summary = {
            "date": prev_d, "n": int(len(prev)), "n_done": int(len(done)),
            "wins": int((done["verdict"] == "win").sum()),
            "win_rate": round(float((done["verdict"] == "win").mean()) * 100, 1),
            "mean_ret": round(float(done["act_close_ret"].mean()), 2),
            "mean_open": (round(float(done["act_open_ret"].mean()), 2)
                          if done["act_open_ret"].notna().any() else None),
            "top_n": int(len(top)),
            "top_win_rate": (round(float((top["verdict"] == "win").mean()) * 100, 1)
                             if len(top) else None),
            "top_mean_ret": (round(float(top["act_close_ret"].mean()), 2)
                             if len(top) else None)}
        # 预测vs实际相关性(全窗口有预测值的已验证行, 模型校准参考)
        cal = picks[picks["pred_ret1"].notna()
                    & picks["act_close_ret"].notna()
                    & picks["verdict"].isin(["win", "lose"])]
        if len(cal) >= 20:
            corr = float(np.corrcoef(cal["pred_ret1"], cal["act_close_ret"])[0, 1])
            summary["pred_corr"] = round(corr, 3) if np.isfinite(corr) else None

    # 席位可靠性榜(近N个有picks的交易日, 按出手数排序)
    win_dates = [d for d in pdates if d <= date][-HM_RELIABILITY_WIN:]
    wp = picks[picks["trade_date"].isin(win_dates)
               & picks["verdict"].isin(["win", "lose"])]
    reliability = []
    for name, g in wp.groupby("hm_name"):
        g = g.sort_values("trade_date")
        n = len(g)
        if n < 3:                       # 样本过少不入榜
            continue
        wins = int((g["verdict"] == "win").sum())
        streak, sv = 0, None
        for v in reversed(g["verdict"].tolist()):
            if sv is None:
                sv, streak = v, 1
            elif v == sv:
                streak += 1
            else:
                break
        reliability.append({
            "hm_name": name, "n": n, "wins": wins,
            "win_rate": round(wins / n * 100, 1),
            "mean_ret": round(float(g["act_close_ret"].mean()), 2),
            "mean_2d": (round(float(g["act_2d_ret"].mean()), 2)
                        if g["act_2d_ret"].notna().any() else None),
            "streak": streak if sv == "win" else -streak})
    reliability.sort(key=lambda r: (-r["n"], -r["win_rate"]))

    if today.empty and prev.empty and not reliability:
        return {}
    return {"picks_today": _hm_picks_rows(today),
            "picks_prev": _hm_picks_rows(prev),
            "prev_summary": summary,
            "reliability": reliability[:20],
            "reliability_win": len(win_dates)}


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
    """高位龙头断板层(研究29验证, 展示/风险标注): 近N日题材龙头∪市场高板,
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


LADDER_HEAT_WIN = 3     # 天梯排序键: 近N日累计家数(研究42 定档)
LADDER_EBB_MIN = 3      # 昨日家数≥此值且今日消失 → 补进天梯(ghost)
EBB_PEAK_MIN = 3        # 退潮峰值门槛: 近3日峰值≥此值(题材曾经足够热)
EBB_RATIO = 0.4         # 退潮降温比例: 今日关联家数≤峰值×此值(衰减60%+)


def _theme_ladder(td_all: pd.DataFrame, date: str, dates: list) -> pd.DataFrame:
    """题材天梯（具备连续性，研究42 定稿）

    两个修正（旧版是纯当日快照, 不具备连续性）:
      ① **排序键改为近3日累计家数** —— 研究42 实测: 近3日累计的
         Top10次日在场命中率 64.3% vs 当日家数 59.6%（+4.8pp）,
         Spearman +0.506 vs +0.380。当日家数仍是次级排序键。
      ② **昨日热门但今日消失的题材补进天梯** —— 当日无独占归属时
         题材会从 theme.day 整行消失(如光模块), 旧版完全看不到。
         研究42: 昨≥3且今=0 的题材次日在场率仅 22.9%, 是最极端的
         退潮信号, 必须可见。与个股层「龙头断板层」对称。

    输出额外列: heat3(近3日累计) / zt_traj(轨迹列表) / ebbing(退潮标记)
    / ghost(今日无涨停, 行来自昨日)
    """
    if date not in dates:
        return td_all[td_all["trade_date"] == date]
    di = dates.index(date)
    win = dates[max(0, di - LADDER_HEAT_WIN + 1):di + 1]
    tw = td_all[td_all["trade_date"].isin(win)]

    # 近3日累计家数(用 zt_all 归因自由口径, 无则降级 zt_cnt)
    col = "zt_all" if "zt_all" in tw.columns else "zt_cnt"
    heat3 = tw.groupby("concept_code")[col].sum()
    # 轨迹: 每个题材在窗口内逐日家数(缺失日补0)
    traj: dict = {}
    for code in heat3.index:
        sub = tw[tw["concept_code"] == code].set_index("trade_date")
        traj[code] = [int(sub[col].get(d, 0) or 0) for d in win]

    cur = td_all[td_all["trade_date"] == date].copy()
    have = set(cur["concept_code"])

    # 昨日热门但今日消失 → 补昨日行
    prev_d = dates[di - 1] if di > 0 else None
    ghosts = []
    if prev_d:
        pv = td_all[td_all["trade_date"] == prev_d]
        for r in pv.itertuples():
            if r.concept_code in have:
                continue
            if int(getattr(r, col, 0) or 0) < LADDER_EBB_MIN:
                continue
            g = r._asdict()
            g["trade_date"] = date          # 挂到当日, 供下游统一处理
            g["zt_cnt"] = 0                 # 今日无涨停
            g["zt_cnt_raw"] = 0
            if "zt_all" in g:
                g["zt_all"] = 0
            g["ghost"] = True
            ghosts.append(g)
    if ghosts:
        cur = pd.concat([cur, pd.DataFrame(ghosts)], ignore_index=True)
        # concat 后原有行的 ghost 列是 NaN, 必须显式置 False
        # (bool(NaN)=True 会把正常行误标成 ghost)
        cur["ghost"] = cur["ghost"].astype(object).where(
            cur["ghost"].notna(), False).astype(bool)
    else:
        cur["ghost"] = False

    cur["heat3"] = cur["concept_code"].map(heat3).fillna(0)
    cur["zt_traj"] = cur["concept_code"].map(traj)
    # 退潮 = 近3日峰值足够热(peak≥EBB_PEAK_MIN) 且 今日大幅降温
    # (今日关联家数≤峰值×EBB_RATIO, 即衰减60%+)。
    # 用**相对降温比例**而非绝对阈值(zt_all<=1): 绝对阈值会①漏判渐进
    # 退潮(如机器人1→11→2降温82%但今日2>1) ②误标小基数(峰值2今日1)。
    # 用 zt_all(关联)而非 zt_cnt(独占): 独占受kpl归属随机性污染。
    peak = cur["zt_traj"].apply(
        lambda x: max(x) if isinstance(x, list) and x else 0)
    cur["ebbing"] = ((peak >= EBB_PEAK_MIN)
                     & (cur["zt_all"] <= peak * EBB_RATIO))
    return cur.sort_values(["heat3", "zt_cnt", "max_height"],
                           ascending=False)


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
    ladder = _theme_ladder(td_all, date, dates)

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

    # 高位龙头断板层(研究29, 展示/风险标注; 前几日龙头今日未涨停也保留可见)
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
                 "zt_all": int(getattr(r, "zt_all", 0) or 0),
                 "max_height": int(r.max_height),
                 "theme_age": int(r.theme_age),
                 "wave_no": int(getattr(r, "wave_no", 0) or 0),
                 # 题材阶段(定稿): 纯波次语义, 不用家数调阶段。
                 # theme_age 保留作持续性展示("N天"), 不再驱动阶段。
                 "mode": theme_stage(int(getattr(r, "wave_no", 0) or 0) or None,
                                     int(getattr(r, "zt_all", 0) or 0)),
                 # 天梯连续性(研究42): 近3日累计热度 / 轨迹 / 退潮标记
                 "heat3": int(getattr(r, "heat3", 0) or 0),
                 "zt_traj": list(getattr(r, "zt_traj", None) or []),
                 "ebbing": bool(getattr(r, "ebbing", False)),
                 "ghost": bool(getattr(r, "ghost", False)),
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
        # 前几日龙头今日未涨停 → 挂到该题材下(解决天梯丢弃高位龙头断板)
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
            "pool": pool_list, "stats": stats, "shortboard": sb_list,
            "hm": _hm_block(date)}


def build_theme_replay(concept_code: str, start: str, end: str,
                       member_sources: list | None = None) -> dict:
    """题材纵向生命周期复盘工作簿(给定题材+窗口, 逐日还原, 数据驱动)。

    与单日 build_review 正交。修复后的口径(消除之前的三类缺陷):
      股票池: 统一走 kpl_theme_pool(member_sources curated 概念成分), 不用
        被污染的事件级直标 con2stock[原始名]; 同一池贯穿 KB 与复盘。
      阶段轴: 用【篮子中位净值曲线】拐点(数据驱动, 适用趋势型),
        不用 theme.day 涨停数(对趋势型会误判)。
      角色: 按【峰值时序 offset + 强度分位】判(去除纯全程涨幅分位的前视),
        并透明化 peak_mult/peak_offset_days 供审计。
    事件/叙事列从 theme_catalyst feed 预填(缺则 null, 待人工补)。
    """
    from rqalpha_mod_ticai import feeds
    from collect.build_theme_kb import (kpl_theme_pool, basket_curve,
                                        derive_stage_axis_basket)
    srcs = member_sources or [concept_code]
    pool_src = "kpl"
    members = kpl_theme_pool(srcs)
    if not members:                     # kpl 概念名未命中 → 降级归属直标(标注来源)
        members = list(_con2stock().get(concept_code) or [])
        pool_src = "con2stock(降级, 可能污染)"
    members = [c for c in members if not str(c).endswith(".BJ")]
    if not members:
        return {"error": f"题材 '{concept_code}' 无成分(kpl/con2stock 均未命中)"}
    memnames = _memnames()
    # 剔除 ST/退(复牌类涨幅扭曲角色初筛, 且非打板可交易标的; 同 heat/radar 口径)
    members = [c for c in members
               if not any(w in str(memnames.get(c, "")) for w in ("ST", "退"))]
    bars = _panel_bars(members, start, end)
    if not bars:
        return {"error": f"窗口 {start}~{end} 无行情(daily_panel 未覆盖?)"}
    bmap = {c: {x[0]: x for x in bl} for c, bl in bars.items()}  # O(1) 日查
    dates = sorted({d for bl in bars.values() for d, *_ in bl})
    didx = {d: i for i, d in enumerate(dates)}

    # 篮子净值曲线 + 数据驱动阶段轴(中位净值抗单只异常; 复用已加载面板)
    panel = _ensure_panel()
    bcurve = basket_curve(members, start, end, panel=panel)
    axis = (derive_stage_axis_basket(members, [start, end], panel=panel)
            if len(bcurve) else [])
    basket_peak_date = None
    if len(bcurve):
        basket_peak_date = str(bcurve.loc[bcurve["med_nv"].idxmax(), "date"])
    bpeak_i = didx.get(basket_peak_date)

    def _stage_of_day(d: str) -> str | None:
        """按阶段轴边界给每日贴阶段(最后一个 date<=d 的阶段)。"""
        if not axis:
            return None
        st = "潜伏"
        for a in axis:
            if str(a["date"]) <= d:
                st = a["stage"]
            else:
                break
        return st

    # theme.day 涨停口径(仅作连板参考, 不驱动阶段; 趋势型题材可能空)
    try:
        td = load("theme.day")
        td = td[(td["concept_code"] == concept_code)
                & (td["trade_date"] >= start) & (td["trade_date"] <= end)]
        td_by = {str(r["trade_date"]): r for _, r in td.iterrows()}
    except Exception:
        td_by = {}

    # 逐日还原表
    daily = []
    for d in dates:
        day_pcts = []
        best_c, best_pct = None, -999.0
        for c, dm in bmap.items():
            row = dm.get(d)
            if row is not None and row[5] is not None:
                p = float(row[5])
                day_pcts.append(p)
                if p > best_pct:
                    best_pct, best_c = p, c
        basket = round(sum(day_pcts) / len(day_pcts), 3) if day_pcts else None
        tdr = td_by.get(d)
        lead = ({"name": memnames.get(best_c, best_c), "ts_code": best_c,
                 "pct": round(best_pct, 2)} if best_c is not None else None)
        narr = None
        try:
            for e in feeds.read_feed("theme_catalyst", d):
                if e.get("topic") and (e["topic"] == concept_code
                                        or concept_code in e["topic"]):
                    narr = (e.get("extra") or {}).get("first_catalyst") or e.get("text")
                    break
        except Exception:
            pass
        daily.append({
            "date": d, "basket_pct": basket, "stage": _stage_of_day(d),
            "zt_cnt": (int(tdr["zt_cnt"]) if tdr is not None
                       and pd.notna(tdr["zt_cnt"]) else None),
            "lead": lead, "event": narr, "narrative": None,
        })

    # 全成分股池: 全程涨幅 + 峰值倍数/日 + 相对主题高潮的峰值时序
    raw = []
    for c, bl in bars.items():
        if len(bl) < 2:
            continue
        closes = [x[3] for x in bl if x[3]]
        if not closes or not closes[0]:
            continue
        total = round((closes[-1] / closes[0] - 1) * 100, 2)
        peak, peak_d = 1.0, bl[0][0]
        for x in bl:
            if x[3] and closes[0]:
                cum = x[3] / closes[0]
                if cum > peak:
                    peak, peak_d = cum, x[0]
        off = (didx.get(peak_d) - bpeak_i) if (bpeak_i is not None
                                               and peak_d in didx) else None
        raw.append({"ts_code": c, "name": memnames.get(c, c),
                    "total_pct": total, "peak_pct": round((peak - 1) * 100, 2),
                    "peak_mult": round(peak, 2), "peak_date": peak_d,
                    "peak_offset_days": off})
    # 强度分位(按峰值倍数) + 时序 → 规范四角色(龙头/中军/补涨/影子)
    raw.sort(key=lambda r: -r["peak_mult"])
    n = len(raw)
    for i, r in enumerate(raw):
        mrank = 1 - (i / n) if n else 0        # 峰值倍数分位(越大越强)
        off = r["peak_offset_days"]
        if r["total_pct"] < 0 or mrank < 0.30:
            r["role"] = "影子"                 # 弱势/边缘/独立逻辑
        elif off is not None and off > 20:
            r["role"] = "补涨"                 # 峰值晚于主题高潮
        elif mrank >= 0.70:
            r["role"] = "龙头"                 # 强度顶部(含早/晚波, 看offset区分)
        else:
            r["role"] = "中军"
        r["mult_rank"] = round(mrank, 3)
    pool = sorted(raw, key=lambda r: -r["total_pct"])
    return {"concept_code": concept_code, "window": [start, end],
            "member_sources": srcs, "pool_source": pool_src,
            "n_members": len(members), "n_days": len(dates),
            "basket_peak_date": basket_peak_date,
            "stage_axis": axis,
            "basket_curve": [[str(r["date"]), round(float(r["med_nv"]), 4),
                              round(float(r["mean_nv"]), 4)]
                             for _, r in bcurve.iterrows()] if len(bcurve) else [],
            "daily": daily, "pool": pool,
            "leaders": [r for r in pool if r["role"] in ("龙头", "中军")][:12]}


# ---------- 情绪型复盘(kpl涨停梯队, 与趋势型正交) ----------

# kpl theme 噪音词(非产业涨停原因, 经kpl×THS对比验证占33.6%, 须过滤)
KPL_NOISE = ["ST板块", "ST摘帽", "摘帽", "超跌", "中报增长", "一季报增长",
             "三季报增长", "年报增长", "并购重组", "股权转让", "国有企业",
             "次新股", "高送转", "预盈预增"]


def _limit_stage_axis(zt_series: list) -> list:
    """情绪型阶段轴: 按题材涨停家数曲线切周期(拐点=涨停家数突变)。

    与趋势型(篮子净值拐点)不同, 情绪型用涨停家数: 启动=首次成集群(≥3),
    高潮=滚动5日累计涨停最密集窗口中心(避免脉冲式题材单日并列峰值导致
    启动=高潮同日退化), 退潮=高潮后回落到低位(<3)。透明可审计。
    """
    if not zt_series:
        return []
    dates = [d for d, _ in zt_series]
    ns = [n for _, n in zt_series]
    axis = []
    start_d = next((d for d, n in zt_series if n >= 3), dates[0])
    axis.append({"stage": "启动", "date": start_d,
                 "zt_cnt": next((n for d, n in zt_series if d == start_d), 0)})
    W = 5                                  # 滚动窗口(脉冲式题材用密集期非单日)
    best_i, best_sum = 0, -1
    for i in range(len(ns)):
        s = sum(ns[max(0, i - W + 1):i + 1])
        if s > best_sum:
            best_sum, best_i = s, i
    axis.append({"stage": "高潮", "date": dates[best_i], "zt_cnt": ns[best_i],
                 "roll5_cnt": best_sum})
    peak_d = dates[best_i]
    after = [(d, n) for d, n in zt_series if d > peak_d]
    end_d = next((d for d, n in after if n < 3), None)
    if end_d:
        axis.append({"stage": "退潮", "date": end_d,
                     "zt_cnt": next(n for d, n in after if d == end_d)})
    return axis


def build_theme_limit_replay(theme_kw: str, start: str, end: str) -> dict:
    """情绪型题材复盘: kpl_events 涨停标注重建题材逐日涨停梯队。

    与 build_theme_replay(趋势型/篮子净值/依赖daily_panel)正交。适用:
      ① 早年题材(daily_panel未覆盖主升段, 如2019猪肉主升段无行情);
      ② 游资主导的情绪型题材(涨停梯队/龙头/连板)。
    数据源 limitup.kpl_events.theme(顿号分隔多题材, 2018起)。

    口径(经kpl×THS对比验证): kpl稳定性0.89适合追踪发酵集群, 但须过滤噪音
    (ST板块/超跌/业绩/重组等非产业涨停原因, 占33.6%); 连板高度从
    events_enriched取(2019-11起有, 早年主升段缺失则null诚实标注)。
    """
    try:
        ev = load("limitup.kpl_events")
    except Exception as e:
        return {"error": f"kpl_events 加载失败: {e}"}
    ev = ev[(ev["tag"] == "涨停") & (ev["trade_date"] >= start)
            & (ev["trade_date"] <= end)].copy()
    if ev.empty:
        return {"error": f"窗口 {start}~{end} kpl_events 无涨停数据"}
    memnames = _memnames()
    rows = []
    for _, r in ev.iterrows():
        nm = str(r.get("name") or "")
        if any(w in nm for w in ("ST", "退")):   # 剔除ST/退(同heat/radar/趋势型口径)
            continue
        for t in str(r.get("theme") or "").split("、"):
            t = t.strip()
            if not t or theme_kw not in t or any(w in t for w in KPL_NOISE):
                continue
            rows.append({"trade_date": r["trade_date"], "ts_code": r["ts_code"],
                         "name": r.get("name"), "theme": t,
                         "lu_time": r.get("lu_time"),
                         "lu_desc": r.get("lu_desc"), "status": r.get("status")})
    if not rows:
        return {"error": f"题材 '{theme_kw}' 窗口内无匹配涨停(过滤噪音后)"}
    ex = pd.DataFrame(rows).drop_duplicates(["trade_date", "ts_code"])
    dates = sorted(ex["trade_date"].unique())
    # 名称: kpl_events自带name优先(覆盖比_memnames全, 含早年退市股), 兜底_memnames
    name_map = {}
    for _, r in ex.iterrows():
        name_map[r["ts_code"]] = (r.get("name")
                                  or memnames.get(r["ts_code"]) or r["ts_code"])
    # 连板高度(events_enriched.limit_times 连板数, 2019-11起; 早年主升段缺失→null)
    try:
        ee = load("limitup.events_enriched",
                  columns=["trade_date", "ts_code", "limit_times"])
        ee = ee[(ee["trade_date"] >= start) & (ee["trade_date"] <= end)]
        hmap = {(r["trade_date"], r["ts_code"]): r["limit_times"]
                for _, r in ee.iterrows()}
    except Exception:
        hmap = {}
    daily = []
    for d in dates:
        day = ex[ex["trade_date"] == d].copy()
        day["_lu"] = pd.to_numeric(day["lu_time"], errors="coerce")
        lead_row = day.sort_values("_lu").iloc[0] if len(day) else None
        lead = ({"name": name_map.get(lead_row["ts_code"], lead_row["ts_code"]),
                 "ts_code": lead_row["ts_code"],
                 "height": hmap.get((d, lead_row["ts_code"]))}
                if lead_row is not None else None)
        heights = [hmap.get((d, c)) for c in day["ts_code"]]
        max_h = max([h for h in heights if h], default=None)
        daily.append({"date": d, "zt_cnt": len(day), "max_height": max_h,
                      "lead": lead,
                      "stocks": [{"name": name_map.get(c, c), "ts_code": c,
                                  "height": hmap.get((d, c))}
                                 for c in day["ts_code"]]})
    freq = ex["ts_code"].value_counts()
    pool = [{"ts_code": c, "name": name_map.get(c, c), "zt_days": int(n)}
            for c, n in freq.items()]
    zt_series = [(d, int((ex["trade_date"] == d).sum())) for d in dates]
    axis = _limit_stage_axis(zt_series)
    peak_d = next((a["date"] for a in axis if a["stage"] == "高潮"), None)
    return {"theme_kw": theme_kw, "window": [start, end],
            "caliber": "kpl情绪型(涨停梯队)", "n_days": len(dates),
            "peak_date": peak_d, "stage_axis": axis, "daily": daily,
            "pool": sorted(pool, key=lambda r: -r["zt_days"]),
            "leaders": sorted(pool, key=lambda r: -r["zt_days"])[:10]}


def replay_to_config(wb: dict, top_n: int = 12) -> dict:
    """把复盘工作簿转为 build_theme_kb 的 config 骨架(半自动写回)。

    量价字段(阶段轴/股池/角色初筛/峰值时序)均数据驱动自动填; 定性字段
    (驱动类型/第一催化/四维画像/证伪条款)留占位符, 待人工校验后跑
    build_theme_kb --config 落库。阶段轴直接透传(不重算)。
    """
    code = wb["concept_code"]
    start, end = wb["window"]
    stocks = []
    top = sorted(wb["pool"], key=lambda r: -r["peak_mult"])[:top_n]
    for r in top:
        stocks.append({"ts_code": r["ts_code"], "name": r["name"],
                       "chain_node": "待核实", "role": r["role"],
                       "moat": None, "realize_cycle": None,
                       "competition": None, "ceiling": None,
                       "benefit_purity": None, "four_dim_score": None})
    return {
        "concept_code": code, "name": f"{code}(待命名)",
        "driver_type": "产业",          # 待人工改: 政策/产业/事件/技术/周期/海外映射
        "parent_theme": None, "first_catalyst": "(待填: 第一催化)",
        "catalyst_conf": "低", "window": [start, end],
        "env_precondition": "(待填)", "falsification": "(待填: 证伪条款)",
        "playbook_ref": f"待命名_{code}",
        "member_sources": wb.get("member_sources") or [code],
        "stage_axis": wb.get("stage_axis") or [],
        "stocks": stocks, "signals": [], "similar": [],
    }


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
    args = sys.argv[1:]
    if args and args[0] == "--limit":      # 情绪型题材复盘(kpl涨停梯队)
        pos = [a for a in args[1:] if not a.startswith("--")]
        kw = pos[0] if len(pos) > 0 else None
        start = pos[1] if len(pos) > 1 else None
        end = pos[2] if len(pos) > 2 else None
        if not (kw and start and end):
            print("用法: python apps/review.py --limit <题材关键词> "
                  "<start YYYYMMDD> <end YYYYMMDD>")
            return
        wb = build_theme_limit_replay(kw, start, end)
        if "error" in wb:
            print(f"情绪型复盘 {kw}: {wb['error']}")
            return
        out = OUT / f"theme_limit_{kw}_{start}_{end}.json"
        out.write_text(json.dumps(wb, ensure_ascii=False), encoding="utf-8")
        print(f"情绪型复盘 {kw} {start}~{end}: 交易日{wb['n_days']} "
              f"高潮{wb['peak_date']} "
              f"龙头{wb['leaders'][0]['name'] if wb['leaders'] else '-'} → {out}")
        return
    if args and args[0] == "--theme":      # 题材纵向生命周期复盘工作簿
        pos = [a for a in args[1:] if not a.startswith("--")]
        code = pos[0] if len(pos) > 0 else None
        start = pos[1] if len(pos) > 1 else None
        end = pos[2] if len(pos) > 2 else None
        if not (code and start and end):
            print("用法: python apps/review.py --theme <concept_code> "
                  "<start YYYYMMDD> <end YYYYMMDD> "
                  "[--members kpl概念名1,名2] [--emit-config]")
            return
        msources = None
        for a in args:
            if a.startswith("--members="):
                msources = [x for x in a.split("=", 1)[1].split(",") if x]
        wb = build_theme_replay(code, start, end, member_sources=msources)
        if "error" in wb:
            print(f"题材复盘 {code}: {wb['error']}")
            return
        out = OUT / f"theme_replay_{code}_{start}_{end}.json"
        out.write_text(json.dumps(wb, ensure_ascii=False), encoding="utf-8")
        print(f"题材复盘 {code} {start}~{end}: 成分{wb['n_members']} "
              f"交易日{wb['n_days']} 龙头{wb['leaders'][0]['name'] if wb['leaders'] else '-'}"
              f" → {out}")
        if "--emit-config" in args:
            cfg = replay_to_config(wb)
            cfgp = DATA / "theme" / "kb_configs" / f"{code}_{start}_{end}.json"
            cfgp.parent.mkdir(parents=True, exist_ok=True)
            cfgp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            print(f"config 骨架(待人工填定性字段后跑 build_theme_kb): {cfgp}")
        return
    arg = args[0] if args else None
    if arg == "--last":        # 批量重生成近N个交易日快照(口径变更后刷新)
        n = int(args[1]) if len(args) > 1 else 30
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
