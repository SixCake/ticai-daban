# -*- coding: utf-8 -*-
"""龙头拐头下跌 → 板块跟跌 + 跷跷板对手板块（唯一出处）

盘中监测用户观察到的规律: 概念龙头开始下跌时, 中军/后排跟跌(消息面),
或存量资金切向同步上涨的另一个板块(跷跷板/资金流)。

龙头双口径:
  A = 上一交易日 theme_day.leader_code(连板高度→封单额→首封早, 跨日延续)
  B = 当日概念成分内涨幅最高者(剔ST, 盘中谁在带队)

下跌候选定义(监测期并行打点, 由研究25回测选优):
  D1 涨速转负: 3分钟涨速 ≤ -0.5
  D2 高点回落: 较日内最高涨幅回落 ≥ 2.0
  D3 破开盘线: 当前涨幅 < 开盘涨幅 - 0.5
  D4 放量下跌: D1 且当笔成交额增速 > 近5分钟均速

事件后结局回填 +5/+10/+20分钟: 概念均跌/下跌家数占比/中军,
跷跷板对手概念的热度变化与领涨股涨幅(验证资金切换是否成立)。

事件流落盘 data/live/seesaw_YYYYMMDD.jsonl:
  kind=trigger 触发快照 | kind=outcome 结局回填(含观察点分钟数 m)
"""
import json
from collections import deque
from pathlib import Path

from core.momentum import window_diff
from datastore import load, path_of

# 候选定义阈值(监测期固定, 选优交给研究25)
D1_SPEED = -0.5        # 3分钟涨速
D2_PULLBACK = 2.0      # 高点回落(涨幅点)
D3_OPEN_BRK = 0.5      # 破开盘线容忍
COOLDOWN = 1800        # 同概念同龙头重复触发冷却(秒)
OBS_MIN = (5, 10, 20)  # 结局观察点(分钟)
MIN_WATCH = 180        # 龙头至少观察时长(秒), 防启动期噪音
OPP_TOPN = 5           # 跷跷板对手概念数
MONITOR_TOPN = 10      # 热度前N概念纳入监测


def _concept_avg(k: str, con2stock: dict, quotes: dict) -> float | None:
    """概念成分均涨幅(板块级跷跷板口径, 与个股领涨无关)"""
    qs = [quotes[c]["pct"] for c in con2stock.get(k, [])
          if c in quotes and "ST" not in quotes[c]["name"]
          and quotes[c]["limit_px"] > 0]
    return round(sum(qs) / len(qs), 2) if qs else None


def _concept_amount(k: str, con2stock: dict, quotes: dict) -> float:
    """概念成分总成交额(供龙头B相关性权重)"""
    return sum(quotes[c]["amount"] for c in con2stock.get(k, [])
               if c in quotes and "ST" not in quotes[c]["name"]
               and quotes[c]["limit_px"] > 0)


def _concept_stats(k: str, con2stock: dict, quotes: dict) -> dict | None:
    """概念成分截面: 均涨幅/下跌家数占比/中军(涨幅≥5%非涨停成交额最大者)"""
    qs = [quotes[c] for c in con2stock.get(k, [])
          if c in quotes and "ST" not in quotes[c]["name"]
          and quotes[c]["limit_px"] > 0]
    if not qs:
        return None
    n = len(qs)
    avg = sum(q["pct"] for q in qs) / n
    fall = sum(1 for q in qs if q["pct"] < 0) / n
    zj = None
    cands = [(c, quotes[c]) for c in con2stock.get(k, [])
             if c in quotes and "ST" not in quotes[c]["name"]
             and quotes[c]["limit_px"] > 0 and quotes[c]["pct"] >= 5
             and quotes[c]["price"] < quotes[c]["limit_px"] * 0.995]
    if cands:
        c, z = max(cands, key=lambda x: x[1]["amount"])
        zj = {"code": c, "name": z["name"], "pct": round(z["pct"], 2)}
    return {"n": n, "avg_pct": round(avg, 2),
            "fall_ratio": round(fall, 3), "zhongjun": zj}


def load_prev_leaders(today_s: str) -> dict:
    """上一交易日各概念龙头 {concept_code: (leader_code, leader_name)}"""
    p = path_of("theme.day")
    if not p.exists():
        return {}
    td = load("theme.day",
              columns=["trade_date", "concept_code",
                       "leader_code", "leader_name"])
    dates = sorted(td["trade_date"].unique())
    prev = max((d for d in dates if d < today_s), default=None)
    if prev is None:          # 库里只有今天(收盘后重启) → 用最新一天
        prev = dates[-1]
    day = td[td["trade_date"] == prev]
    return {r.concept_code: (r.leader_code, r.leader_name)
            for r in day.itertuples()
            if isinstance(r.leader_code, str) and r.leader_code}


class SeesawTracker:
    """龙头拐头事件跟踪器: radar 每 cycle 调 update()"""

    def __init__(self, con2stock: dict, cname: dict, day_str: str,
                 live_dir: Path):
        self.con2stock = con2stock
        self.cname = cname
        self.day = day_str
        self.log_path = Path(live_dir) / f"seesaw_{day_str}.jsonl"
        self.leader_prev = load_prev_leaders(day_str)   # 口径A
        self.lead_state: dict = {}      # code -> {max_pct, open_pct, t0}
        self.amt_hist: dict = {}        # code -> deque[(t, amount)]
        self.heat_hist: dict = {}       # concept -> deque[(t, heat)]
        self.cooldown: dict = {}        # (concept, code) -> 上次触发epoch
        self.events: list = []          # 当日全部事件(含已回填结局)
        self._heat_rows_cache: list = []  # 当cycle热度行, 供跷跷板候选筛选
        self._zj_codes: set = set()     # 监测概念中军票(分时扩围用)
        self.con_day: dict = {}         # concept -> [[HHMMSS, 板块均涨], ...] 板块级分时
        self._con_reload(live_dir, day_str)
        self._reload()

    # ---------- 重启回载: 板块级分时(重启不丢) ----------
    def _con_reload(self, live_dir, day_str):
        f = Path(live_dir) / f"concept_px_{day_str}.json"
        if f.exists():
            try:
                self.con_day = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                self.con_day = {}

    # ---------- 重启回载: 从当日jsonl恢复事件/结局/冷却 ----------
    def _reload(self):
        if not self.log_path.exists():
            return
        outs = {}
        try:
            for line in self.log_path.read_text(encoding="utf-8") \
                    .strip().splitlines():
                d = json.loads(line)
                if d.get("kind") == "trigger":
                    self.events.append(d)
                    self.cooldown[(d["concept_code"],
                                   d["leader_code"])] = d["te"]
                elif d.get("kind") == "outcome":
                    outs.setdefault((d["concept_code"],
                                     d["leader_code"], d["te"]), {})[
                        str(d["m"])] = d
        except Exception:
            return
        for ev in self.events:
            ev.setdefault("outcomes", {}).update(
                outs.get((ev["concept_code"], ev["leader_code"],
                          ev["te"]), {}))
        if self.events:
            print(f"跷跷板事件回载 {len(self.events)}条")

    def _append(self, rec: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- 龙头候选 ----------
    def _leaders_of(self, k: str, quotes: dict) -> dict:
        """{code: [口径标签]}, A/B可叠加"""
        out: dict = {}
        la = self.leader_prev.get(k)
        if la and la[0] in quotes:
            out.setdefault(la[0], []).append("A")
        mem = sorted((c for c in self.con2stock.get(k, [])
                      if c in quotes and "ST" not in quotes[c]["name"]
                      and quotes[c]["limit_px"] > 0),
                     key=lambda c: -quotes[c]["pct"])
        if mem and quotes[mem[0]]["pct"] >= 3:      # 太弱的头不算带队
            code = mem[0]
            q = quotes[code]
            total = _concept_amount(k, self.con2stock, quotes)
            # 相关性闸: 龙头须在概念内有存在感(成交占比≥2%或成分≤20的小簇),
            # 防蹭概念边缘票因当日领涨被误判为龙头(如青山纸业之于CPO)
            if total <= 0 or len(mem) <= 20 \
                    or q["amount"] / total >= 0.02:
                out.setdefault(code, []).append("B")
        return out

    def _open_pct(self, q: dict) -> float | None:
        """由 open 与 pct 反推开盘涨幅; 无open字段回退None"""
        o, px, pct = q.get("open", 0.0), q["price"], q["pct"]
        if o <= 0 or px <= 0:
            return None
        pre = px / (1 + pct / 100)
        return round((o / pre - 1) * 100, 2) if pre > 0 else None

    def _amount_speed(self, c: str, q: dict, t: float):
        """(当笔成交额增速, 近5分钟均速) 元/秒; 样本不足返回(None, None)"""
        h = self.amt_hist.setdefault(c, deque(maxlen=64))
        amt = q.get("amount", 0)
        v_now = None
        if h and amt > 0:
            dt = t - h[-1][0]
            if dt >= 5:
                v_now = (amt - h[-1][1]) / dt
        v5 = None
        target = t - 300
        for ts, a in h:
            if ts >= target:
                if amt > 0 and t > ts:
                    v5 = (amt - a) / (t - ts)
                break
        h.append((t, amt))
        return v_now, v5

    def update(self, quotes: dict, hist: dict, heat_rows: list,
               t: float, ts_str: str) -> dict:
        """每cycle主入口: 热度历史→龙头检测→触发→结局回填, 返回看板快照"""
        self._heat_rows_cache = heat_rows
        for r in heat_rows:
            hh = self.heat_hist.setdefault(r["concept_code"],
                                           deque(maxlen=48))
            hh.append((t, r["heat"]))
        # 监测概念: 热度TOP10 ∪ 涨停家数≥4
        mon = [r for r in heat_rows[:MONITOR_TOPN]]
        seen = {r["concept_code"] for r in mon}
        for r in heat_rows:
            if r["concept_code"] not in seen and r.get("zt", 0) >= 4:
                mon.append(r)
        # 中军票集合(供分时轨迹扩围落盘): 非涨停成交额最大者
        self._zj_codes = set()
        for r in mon:
            k = r["concept_code"]
            cands = [quotes[c] for c in self.con2stock.get(k, [])
                     if c in quotes and "ST" not in quotes[c]["name"]
                     and quotes[c]["limit_px"] > 0
                     and quotes[c]["price"] < quotes[c]["limit_px"] * 0.995]
            if cands:
                z = max(cands, key=lambda q: q["amount"])
                self._zj_codes |= {c for c in self.con2stock.get(k, [])
                                   if c in quotes and quotes[c] is z}
        # 板块级分时序列(跷跷板板块对比口径): 每cycle记监测概念均涨幅
        hms = ts_str.replace(":", "")
        for r in mon:
            avg = _concept_avg(r["concept_code"], self.con2stock, quotes)
            if avg is not None:
                row = self.con_day.setdefault(r["concept_code"], [])
                if not row or row[-1][0] != hms:
                    row.append([hms, avg])
        for r in mon:
            k = r["concept_code"]
            for code, calibers in self._leaders_of(k, quotes).items():
                q = quotes[code]
                st = self.lead_state.get(code)
                if st is None:
                    st = {"max_pct": q["pct"], "t0": t,
                          "open_pct": self._open_pct(q)}
                    self.lead_state[code] = st
                else:
                    st["max_pct"] = max(st["max_pct"], q["pct"])
                    if st["open_pct"] is None:
                        st["open_pct"] = self._open_pct(q)
                if t - st["t0"] < MIN_WATCH:
                    continue
                s3 = window_diff(hist.get(code), 180, t)
                defs = []
                if s3 <= D1_SPEED:
                    defs.append("D1")
                if st["max_pct"] - q["pct"] >= D2_PULLBACK:
                    defs.append("D2")
                if (st["open_pct"] is not None
                        and q["pct"] < st["open_pct"] - D3_OPEN_BRK):
                    defs.append("D3")
                v_now, v5 = self._amount_speed(code, q, t)
                if "D1" in defs and v_now is not None and v_now > 0 \
                        and (v5 is None or v_now > v5):
                    defs.append("D4")
                if not defs:
                    continue
                ck = (k, code)
                if t - self.cooldown.get(ck, 0) < COOLDOWN:
                    continue
                self.cooldown[ck] = t
                self._trigger(k, r, code, q, calibers, defs, st,
                              s3, quotes, t, ts_str)
        self._fill_outcomes(quotes, heat_rows, t, ts_str)
        return {"events": self._snap_events(t)}

    def _trigger(self, k, heat_row, code, q, calibers, defs, st, s3,
                 quotes, t, ts_str) -> dict:
        # 跷跷板候选: 其他概念中 热度3分钟增量为正 且 头部涨速为正
        dheat_by = {}
        for k2, hh in self.heat_hist.items():
            if k2 == k:
                continue
            d = window_diff(hh, 180, t)
            if d > 0:
                dheat_by[k2] = d
        opp = []
        for r2 in sorted(
                (r2 for r2 in self._heat_rows_cache
                 if r2["concept_code"] in dheat_by
                 and r2.get("s3", 0) > 0),
                key=lambda x: -dheat_by[x["concept_code"]])[:OPP_TOPN]:
            k2 = r2["concept_code"]
            top = self._top_gainer(k2, quotes)
            opp.append({"concept_code": k2, "name": r2["name"],
                        "heat": r2["heat"],
                        "dheat": round(dheat_by[k2], 2),
                        "s3": r2.get("s3", 0),
                        "avg_pct": _concept_avg(k2, self.con2stock,
                                                quotes),
                        "top": top})
        ev = {
            "kind": "trigger", "date": self.day, "t": ts_str, "te": t,
            "concept_code": k, "concept_name": heat_row["name"],
            "leader_code": code, "leader_name": q["name"],
            "calibers": calibers, "defs": defs,
            "pct": round(q["pct"], 2), "max_pct": round(st["max_pct"], 2),
            "open_pct": st["open_pct"], "s3": s3,
            "con_avg": _concept_avg(k, self.con2stock, quotes),
            "members": _concept_stats(k, self.con2stock, quotes),
            "opp": opp, "outcomes": {},
        }
        self.events.append(ev)
        self._append(ev)
        print(f"[{ts_str}] 龙头拐头 {heat_row['name']} "
              f"{q['name']}({'+'.join(calibers)}) {'+'.join(defs)} "
              f"{q['pct']:.2f}% 高{st['max_pct']:.2f}% "
              f"对手{[o['name'] for o in opp[:3]]}")
        return ev

    def _top_gainer(self, k: str, quotes: dict) -> dict | None:
        mem = sorted((c for c in self.con2stock.get(k, [])
                      if c in quotes and "ST" not in quotes[c]["name"]
                      and quotes[c]["limit_px"] > 0),
                     key=lambda c: -quotes[c]["pct"])
        if not mem:
            return None
        q = quotes[mem[0]]
        return {"code": mem[0], "name": q["name"],
                "pct": round(q["pct"], 2)}

    # ---------- 结局回填 ----------
    def _fill_outcomes(self, quotes: dict, heat_rows: list, t: float,
                       ts_str: str):
        heat_by = {r["concept_code"]: r["heat"] for r in heat_rows}
        for ev in self.events:
            outs = ev.setdefault("outcomes", {})
            for m in OBS_MIN:
                if str(m) in outs or t < ev["te"] + m * 60:
                    continue
                opp_now = []
                for o in ev.get("opp", []):
                    k2 = o["concept_code"]
                    h0 = o.get("heat", 0)
                    top = self._top_gainer(k2, quotes)
                    opp_now.append({
                        "concept_code": k2, "name": o["name"],
                        "heat": heat_by.get(k2),
                        "dheat": (round(heat_by[k2] - h0, 2)
                                  if k2 in heat_by else None),
                        "avg_pct": _concept_avg(k2, self.con2stock,
                                                quotes),
                        "top_pct": top["pct"] if top else None,
                        "top_code": top["code"] if top else None})
                outs[str(m)] = {
                    "t": ts_str,
                    "members": _concept_stats(ev["concept_code"],
                                              self.con2stock, quotes),
                    "opp": opp_now}
                rec = {"kind": "outcome", "date": self.day,
                       "te": ev["te"], "concept_code": ev["concept_code"],
                       "leader_code": ev["leader_code"], "m": m,
                       **outs[str(m)]}
                self._append(rec)

    # ---------- 快照(近30分钟事件供看板) ----------
    def _snap_events(self, t: float) -> list:
        out = []
        for ev in reversed(self.events):
            if t - ev["te"] > 1800:
                break
            d = dict(ev)
            d.pop("kind", None)
            out.append(d)
        return out

    @property
    def track_codes(self) -> set:
        """需扩围记录分时轨迹的票: 口径A龙头 + 当日龙头 + 监测概念中军"""
        codes = {c for c, _ in self.leader_prev.values() if c}
        codes |= set(self.lead_state.keys())
        codes |= self._zj_codes
        return codes
