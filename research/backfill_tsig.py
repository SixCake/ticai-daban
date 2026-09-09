# -*- coding: utf-8 -*-
"""历史 presig_state 回填 T 级与昨日连板高度（口径与生产同源）

背景: T1/T2/T3 题材级信号(core/theme_signal.py)是在 20260904 之后才加入
生产的, 故 0904 及更早的 presig_state 只有 S 级(stage), 没有 t_sig。
本脚本按最新逻辑回填, 让历史数据也能看 S/T 双轨表达。

口径保证: **直接调用 core/theme_signal 的 prev_ladder_of / t_level_of**,
不在本脚本重写判定逻辑 —— 避免研究与生产口径分叉。

与研究脚本的口径差异(必须知道):
  生产 core/theme_signal 用**盘中贴死涨停价**实时计数题材封板家数;
  本脚本(离线回填)用 events_enriched.first_time ≤ 信号时刻计数, 这是
  收盘权威口径, 会略高于盘中实时值。故回填的 T 级偏乐观, 只供回看,
  不可当作当时的真实决策状态。

回填字段:
  t_sig = {level, theme, zt_live, y_ht, ld_name, ld_gap, neg_fb, why}
          与 apps/radar._t_brief 同结构
  y_lb  = 昨日连板数(0=昨日未涨停), 研究36 H1 证实的关键维度
  t_src = "offline_backfill"  与盘中 "live" 区分(Forward Return 方法论)
  ev_exit / exit_why = 按定稿卖出规则模拟的次日离场收益与离场原因
          (调 core/exit_rules.simulate_exit, 与研究36 同口径)

用法:
  python research/backfill_tsig.py 20260904
  python research/backfill_tsig.py --all      # 所有有 presig_state 的日期
  python research/backfill_tsig.py 20260904 --dry   # 只看统计不落盘
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.exit_rules import simulate_exit  # noqa: E402
from core.theme_signal import prev_ladder_of, t_level_of  # noqa: E402
from datastore import load  # noqa: E402

LIVE = DATA / "live"


def sec(hms) -> int:
    s = str(hms or "").replace(":", "")
    if len(s) < 6 or not s[:6].isdigit():
        return 0
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])


def backfill(date: str, dry: bool = False) -> dict:
    """回填单日 → 统计字典"""
    f = LIVE / f"presig_state_{date}.json"
    if not f.exists():
        print(f"{date}: 无 presig_state, 跳过")
        return {}
    st = json.loads(f.read_text(encoding="utf-8"))
    sigs = st.get("signals", [])
    if not sigs:
        print(f"{date}: 信号为空, 跳过")
        return {}

    # ---- 昨日天梯(T 级判定的静态锚, 09:15 已知) ----
    td = load("theme.day", columns=["trade_date", "concept_code",
                                    "concept_name", "leader_code",
                                    "leader_name", "leader_height",
                                    "max_height", "zt_cnt", "theme_age"])
    prev = max((d for d in td["trade_date"].unique() if d < date),
               default=None)
    if not prev:
        print(f"{date}: 无昨日天梯, 全部记散票 T1")
        ladder = {}
    else:
        ladder = prev_ladder_of(td[td["trade_date"] == prev])
    print(f"{date}: 昨日天梯 {prev} · {len(ladder)} 个题材")

    # ---- 当日封板首封时刻(离线权威口径) ----
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "first_time", "limit_times"])
    seal_sec = {r.ts_code: sec(r.first_time)
                for r in ev[ev["trade_date"] == date].itertuples()}
    lb_day = {r.ts_code: int(r.limit_times or 1)
              for r in ev[ev["trade_date"] == date].itertuples()}
    lb_prev = {r.ts_code: int(r.limit_times or 1)
               for r in ev[ev["trade_date"] == prev].itertuples()} if prev \
        else {}

    # ---- 龙一今日开盘涨幅(从日线面板反推昨收) ----
    pn = load("market.daily_panel",
              columns=["trade_date", "ts_code", "open", "pre_close",
                       "close"])
    px = {r.ts_code: r for r in pn[pn["trade_date"] == date].itertuples()}

    stock2con, _, _ = load_maps()
    con2stock = load_con2stock()

    # ---- 卖出模拟所需: 次日分时 + 次日日线(涨停价/收盘) + 买入当日最高 ----
    nxt = max((d for d in pn["trade_date"].unique() if d > date),
              default=None)
    px_next = ({r.ts_code: r
                for r in pn[pn["trade_date"] == nxt].itertuples()}
               if nxt else {})
    ipx_cache: dict = {}

    def _ipx(day):
        if day is None:
            return {}
        if day not in ipx_cache:
            fp = LIVE / f"intraday_px_{day}.json"
            try:
                ipx_cache[day] = json.loads(
                    fp.read_text(encoding="utf-8")) if fp.exists() else {}
            except Exception:
                ipx_cache[day] = {}
        return ipx_cache[day]
    ipx_d, ipx_nd = _ipx(date), _ipx(nxt)
    # 申万一级/二级映射(供看板已封板/未封板表的「板块列」)
    fsw = DATA / "meta" / "sw_map.json"
    sw_map = json.loads(fsw.read_text(encoding="utf-8")) if fsw.exists() else {}

    lv_cnt = Counter()
    n_notheme = 0
    for s in sigs:
        c = s.get("ts_code")
        if not c:
            continue
        tsec0 = sec(s.get("pt"))
        # 题材: 成分概念 ∩ 昨日活跃题材, 按昨日涨停家数取最大
        # 必须保留 concept_code 键(prev_ladder_of 返回的 dict 内不含该字段)
        cands = [(k, ladder[k]) for k in stock2con.get(c, []) if k in ladder]
        hot = max(cands, key=lambda x: x[1]["zt_cnt"])[1] if cands else None
        hot_code = max(cands, key=lambda x: x[1]["zt_cnt"])[0] if cands \
            else None
        if hot is None:
            n_notheme += 1
            lvl, evd = t_level_of(None, 0, None)
            theme_name = None
        else:
            # 截至信号时刻该题材已封板家数(离线权威口径)
            zt_live = sum(1 for m in con2stock.get(hot_code, [])
                          if 0 < seal_sec.get(m, 0) <= tsec0) if tsec0 \
                else 0
            lc = hot["leader_code"]
            ld_gap = None
            r = px.get(lc)
            if lc and r and r.pre_close and r.pre_close > 0 and r.open:
                ld_gap = (r.open / r.pre_close - 1) * 100
            lvl, evd = t_level_of(hot, zt_live, ld_gap)
            theme_name = hot["name"]
        lv_cnt[lvl] += 1
        s["t_sig"] = {"level": lvl, "theme": theme_name,
                      "zt_live": evd.get("zt_now", 0),
                      "y_ht": evd.get("y_ht", 0),
                      "ld_name": evd.get("ld_name"),
                      "ld_gap": (round(evd["ld_gap"], 2)
                                 if evd.get("ld_gap") is not None else None),
                      "neg_fb": evd.get("neg_fb", False),
                      "why": evd.get("why", "")}
        s["t_src"] = "offline_backfill"
        s["y_lb"] = lb_prev.get(c, 0)
        s["lb"] = lb_day.get(c, 0)
        # 定稿卖出规则模拟: 需买价 + 次日分时 + 次日涨停价 + 买入当日最高
        pb = s.get("pb")
        ev_exit, exit_why = None, None
        if pb and nxt:
            nd_px = px_next.get(c)
            pts_nd = ipx_nd.get(c) or []
            # 买入当日最高价: 优先 px_hist(信号自带), 其次当日分时文件
            his = [float(e[1]) for e in (s.get("px_hist") or [])
                   if len(e) >= 2 and e[1] and e[1] > 0]
            day_hi = max(his) if his else max(
                (float(e[1]) for e in (ipx_d.get(c) or [])
                 if len(e) >= 2 and e[1] and e[1] > 0), default=0.0)
            lp_nd = 0.0
            if nd_px and nd_px.pre_close and nd_px.pre_close > 0:
                ratio = 0.20 if c[:2] in ("30", "68") else 0.10
                lp_nd = nd_px.pre_close * (1 + ratio)
            ev_exit, exit_why = simulate_exit(
                pts_nd, pb, day_hi, lp_nd,
                nd_px.close if nd_px else 0.0)
        s["ev_exit"] = (round(ev_exit, 2) if ev_exit is not None else None)
        s["exit_why"] = exit_why
        msw = sw_map.get(c)
        if msw:
            s["sw_l1"] = msw.get("l1")
            s["sw_l2"] = msw.get("l2")

    print(f"{date}: 信号 {len(sigs)} 条 → T级分布 {dict(lv_cnt)} "
          f"散票 {n_notheme} 条({n_notheme / len(sigs) * 100:.0f}%)")
    yl = Counter(s.get("y_lb", 0) for s in sigs)
    print(f"{date}: 昨日连板分布 {dict(sorted(yl.items()))}")
    ex = [s for s in sigs if s.get("ev_exit") is not None]
    if ex:
        vals = [s["ev_exit"] for s in ex]
        w = [x for x in vals if x > 0]
        l = [x for x in vals if x <= 0]
        aw = sum(w) / len(w) if w else 0.0
        al = sum(l) / len(l) if l else 0.0
        exp = (len(w) / len(vals)) * aw + (len(l) / len(vals)) * al
        print(f"{date}: 卖出模拟 {len(ex)} 样本 胜率{len(w) / len(vals) * 100:.1f}% "
              f"盈亏比{(aw / abs(al)) if al < 0 else 0:.2f} 期望{exp:+.2f}%")
        print(f"{date}: 离场原因 {dict(Counter(s.get('exit_why') for s in ex))}")
    else:
        print(f"{date}: 无卖出模拟样本(缺次日分时或无买价)")

    if dry:
        print(f"{date}: --dry 未落盘")
        return {"levels": dict(lv_cnt), "n": len(sigs)}

    bak = f.with_name(f.stem + ".pre_tsig" + f.suffix)
    if not bak.exists():
        shutil.copy2(f, bak)
        print(f"{date}: 备份 → {bak.name}")
    f.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    print(f"{date}: 已落盘 → {f.name}")
    return {"levels": dict(lv_cnt), "n": len(sigs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", help="YYYYMMDD")
    ap.add_argument("--all", action="store_true",
                    help="所有有 presig_state 的日期")
    ap.add_argument("--dry", action="store_true", help="只看统计不落盘")
    a = ap.parse_args()

    if a.all:
        dates = sorted(p.stem.replace("presig_state_", "")
                       for p in LIVE.glob("presig_state_*.json")
                       if p.stem.replace("presig_state_", "").isdigit()
                       and len(p.stem.replace("presig_state_", "")) == 8)
    elif a.date:
        dates = [a.date]
    else:
        # 默认: 最新一日
        ev = load("limitup.events_enriched", columns=["trade_date"])
        dates = [str(ev["trade_date"].max())]
    total = Counter()
    for d in dates:
        r = backfill(d, a.dry)
        for k, v in (r.get("levels") or {}).items():
            total[k] += v
    if len(dates) > 1:
        print(f"\n合计 {len(dates)} 日 T级分布: {dict(total)}")


if __name__ == "__main__":
    main()
