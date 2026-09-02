# -*- coding: utf-8 -*-
"""盘前/盘中预警雷达: 全量成分股扫描(20s) → 题材热度自算排名 + 东财板块对照
 → 个股涨停概率v0排名

题材热度与概率公式见 core/heat.py 与 core/prob.py（领域逻辑唯一出处）；
本文件只负责: 扫描调度、时序状态维护、限流退避、快照与校准日志落盘。

输出 data/live/radar.json; 校准日志 data/live/radar_log_YYYYMMDD.jsonl
(每cycle 20s, 涨幅≥3%或概率≥0.2全量含负例)
"""
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA, QUOTE_SOURCE  # noqa: E402
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.calendar import is_trading_hours  # noqa: E402
from core.early_signal import build_signals, zt_shape_of  # noqa: E402
from core.heat import HOT_THRESHOLD, sw_aggregate, theme_heat  # noqa: E402
from core.momentum import window_diff  # noqa: E402
from core.prob import stock_prob  # noqa: E402
from core.seesaw import SeesawTracker  # noqa: E402
from core.structure import build_struct_scores, fetch_ldlr_prev, v5_full  # noqa: E402
from quotes.eastmoney import fetch_em_boards  # noqa: E402
from quotes import fetch_quotes  # noqa: E402  # QUOTE_SOURCE分发: tx|qmt
from quotes.tx import fetch_quotes as fetch_quotes_tx  # noqa: E402  # qmt失败兜底

LIVE = DATA / "live"
LIVE.mkdir(exist_ok=True)
INTERVAL = 20
# 触板/封死口径分离(研究30复核: 价格≥涨停价×0.995 仅为触板, 不等于封死)
TOUCH_EPS = 0.995            # 触板判据
LOCK_EPS = 0.9995            # 封死判据(价格贴死涨停价)
LOCK_HOLD = 60               # 封死需连续保持的秒数(否则视为未封死)


class Radar:
    def __init__(self):
        self.con2stock = load_con2stock()
        _, _, self.cname = load_maps()
        self.stock2con = defaultdict(list)
        for k, cs in self.con2stock.items():
            for c in cs:
                self.stock2con[c].append(k)
        self.codes = sorted({c for cs in self.con2stock.values() for c in cs
                             if not c.endswith(".BJ")})
        self.hist: dict = {}          # code -> deque[(t, pct)]
        self.cycle = 0
        self.em_cache: list = []
        self.em_ts = 0.0
        self.em_skip = 0              # 东财失败退避(剩余跳过cycle数)
        self.bad_sweep = 0            # 腾讯扫描完整性退避计数
        self._log_prob: dict = {}     # 上轮日志概率, 供dp(概率变化)计算
        self._heat_hist: dict = {}    # 题材热度历史, 供dheat(热度趋势)计算
        self._yest_date: str = ""     # 昨日收盘位置缓存日期(S3叠加因子)
        self._yest_cpos: dict = {}
        self._struct_date: str = ""   # V5结构层快照日期(研究24, 影子)
        self._struct: dict = {}       # code -> {g_chip, gate, v5_base...}
        self._presig_date: str = ""   # 当日累积前向信号(跨cycle保留)
        self._presig_day: dict = {}
        self._presig_px: dict = {}    # (code,stage) -> [[HH:MM:SS, price], ...]
        self._last_px: dict = {}      # code -> 最近已知价(推送静默票的状态推进兑底)
        self._lock_since: dict = {}   # (code,stage) -> 首次贴死涨停价epoch
        self._open_traj: dict = {}    # code -> [[epoch, pct], ...] 开盘窗口轨迹
        self._ot_saved: bool = False
        self._ot_load_date: str = ""
        self._ipx_day: dict = {}      # code -> [[HHMMSS, price, vol, amt], ...] 全天分时
        self._ipx_date: str = ""
        self._sw_traj: dict = {}      # l1 -> [[HHMMSS, pct, net, amt], ...] 申万轨迹(回放/hover)
        self._sw_traj_date: str = ""
        self._sw_map: dict = {}       # code -> {l1,l2,...} 申万映射(mtime守护)
        self._sw_mt = None
        self._focus: dict = {}        # focus.json 专注板块集合(mtime守护)
        self._focus_mt = None
        self.seesaw: SeesawTracker | None = None  # 龙头拐头·跷跷板监测(跨日重建)
        self._seesaw_date: str = ""
        print(f"雷达初始化: {len(self.con2stock)}概念 {len(self.codes)}成分股")

    def sweep(self) -> dict:
        t0 = time.time()
        batches = [self.codes[i:i + 60] for i in range(0, len(self.codes), 60)]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(fetch_quotes, batches))
        quotes = {}
        for r in res:
            quotes.update(r)
        if not quotes:      # qmt源整体失败(推送断+盘中无横截面) → 腾讯兜底
            print("行情源无返回, 兜底腾讯源")
            with ThreadPoolExecutor(8) as ex:
                res = list(ex.map(fetch_quotes_tx, batches))
            for r in res:
                quotes.update(r)
        t = time.time()
        in_ot_win = "0925" <= datetime.fromtimestamp(t).strftime("%H%M") \
            <= "0940"
        for c, q in quotes.items():
            if "ST" in q["name"]:
                continue
            h = self.hist.setdefault(c, deque(maxlen=64))
            h.append((t, q["pct"]))
            if in_ot_win:      # 开盘窗口轨迹采集(落盘后重启不丢)
                ot = self._open_traj.setdefault(c, [])
                if not ot or ot[-1][0] < t:
                    ot.append([t, q["pct"]])
        return quotes

    def once(self) -> float:
        """执行一轮扫描并写出radar.json, 返回耗时(秒)"""
        now = datetime.now()
        t = time.time()
        quotes = self.sweep()
        if not quotes:
            # qmt源推送/横截面全失败时不写陈旧快照, 保留上一轮并退避
            self.bad_sweep += 1
            print(f"[{now:%H:%M:%S}] 行情源无返回, 保留上一快照, 退避")
            return 0.0
        today_s0 = now.strftime("%Y%m%d")
        # 开盘轨迹持久化: 窗口内每 cycle增量落盘(防窗口内重启丢失);
        # 重启后回载(形态分类不丢)
        if self._open_traj and now.strftime("%H%M") <= "0941":
            (LIVE / f"open_traj_{today_s0}.json").write_text(
                json.dumps(self._open_traj), encoding="utf-8")
            self._ot_saved = True
        if now.strftime("%H%M") >= "0941" and not self._ot_saved \
                and self._open_traj:
            (LIVE / f"open_traj_{today_s0}.json").write_text(
                json.dumps(self._open_traj), encoding="utf-8")
            self._ot_saved = True
            print(f"[{now:%H:%M:%S}] 开盘轨迹落盘 {len(self._open_traj)}只")
        if self._ot_load_date != today_s0:
            self._ot_load_date = today_s0
            f_ot = LIVE / f"open_traj_{today_s0}.json"
            if f_ot.exists() and not self._open_traj:
                try:
                    self._open_traj = json.loads(
                        f_ot.read_text(encoding="utf-8"))
                    self._ot_saved = True
                    print(f"[{now:%H:%M:%S}] 开盘轨迹回载 "
                          f"{len(self._open_traj)}只")
                except Exception:
                    pass
        # 限流保护: 腾讯扫描完整性<90% → 指数退避(20s→40s→80s)
        if len(quotes) < 0.9 * len(self.codes):
            self.bad_sweep += 1
            print(f"[{now:%H:%M:%S}] 扫描不完整 {len(quotes)}/{len(self.codes)}, 退避")
        else:
            self.bad_sweep = 0
        # 申万映射(mtime守护, 每日收盘刷新; 缺失则申万聚合为空)
        f_sw = DATA / "meta" / "sw_map.json"
        if f_sw.exists():
            mt = f_sw.stat().st_mtime
            if self._sw_mt != mt:
                self._sw_mt = mt
                try:
                    self._sw_map = json.loads(
                        f_sw.read_text(encoding="utf-8"))
                    print(f"申万映射加载 {len(self._sw_map)}只")
                except Exception as e:
                    print(f"申万映射加载失败: {e}")
        # focus专注板块集合(mtime守护, 服务端 /api/focus 写入)
        f_focus = LIVE / "focus.json"
        if f_focus.exists():
            mt = f_focus.stat().st_mtime
            if self._focus_mt != mt:
                self._focus_mt = mt
                try:
                    self._focus = json.loads(
                        f_focus.read_text(encoding="utf-8"))
                except Exception:
                    self._focus = {}
        else:
            self._focus, self._focus_mt = {}, None
        themes = theme_heat(self.con2stock, self.cname, quotes, self.hist, t)
        heat_by = {r["concept_code"]: r["heat"] for r in themes}
        rank_by = {r["concept_code"]: i + 1 for i, r in enumerate(themes)}
        for r in themes:            # 题材热度历史, 供dheat趋势因子
            hh = self._heat_hist.setdefault(r["concept_code"],
                                            deque(maxlen=48))
            hh.append((t, r["heat"]))
        # 龙头拐头·跷跷板监测(core/seesaw.py): 龙头下跌→跟跌+对手板块,
        # 多定义并行打点供研究25选优, 事件流落盘seesaw_YYYYMMDD.jsonl
        if self._seesaw_date != today_s0:
            self._seesaw_date = today_s0
            self.seesaw = SeesawTracker(self.con2stock, self.cname,
                                        today_s0, LIVE)
        seesaw_snap = self.seesaw.update(quotes, self.hist, themes, t,
                                         now.strftime("%H:%M:%S"))
        # 限流保护: 东财每分钟一次, 失败退避5分钟, 期间用缓存
        if self.em_skip > 0:
            self.em_skip -= 1
            em = self.em_cache
        elif self.cycle % 3 == 0:
            em = fetch_em_boards()
            if em:
                self.em_cache, self.em_ts = em, time.time()
            elif time.time() - self.em_ts < 300:
                em = self.em_cache
                self.em_skip = 3
            else:
                self.em_skip = 15
        else:
            em = self.em_cache
        em_by = {b["name"]: b for b in em}
        our_names = {r["name"] for r in themes}
        external = [b for b in em if b["name"] not in our_names
                    and not b["name"].startswith("昨日")][:15]
        for r in themes:
            b = em_by.get(r["name"]) or next(
                (b for b in em
                 if len(b["name"]) >= 3 and
                 (b["name"] in r["name"] or r["name"] in b["name"])), None)
            r["em"] = {"pct": b["pct"], "speed": b["speed"],
                       "leader": b["leader"]} if b else None
        stocks_all = stock_prob(quotes, heat_by, self.stock2con, self.cname,
                                self.hist, t)
        prob_by = {s["ts_code"]: s for s in stocks_all}
        # 申万一级/二级分级聚合(等权涨幅排序, 复用theme_heat口径)
        sw = sw_aggregate(self._sw_map, quotes, self.hist, prob_by, t)
        # 申万轨迹累积(供看板时间轴回放+hover历史曲线): 每3cycle(~60s)记L1 pct/net/amt
        if self._sw_traj_date != today_s0:
            self._sw_traj_date = today_s0
            f_st = LIVE / f"sw_traj_{today_s0}.json"
            try:
                self._sw_traj = json.loads(f_st.read_text(encoding="utf-8")) \
                    if f_st.exists() else {}
            except Exception:
                self._sw_traj = {}
        if self.cycle % 3 == 0:
            hm = now.strftime("%H%M%S")
            for r in sw:
                self._sw_traj.setdefault(r["l1"], []).append(
                    [hm, r["pct"], r["net"], r["amt"]])
        # 快照题材附带领涨成分股top5(供看板点击展开)
        for r in themes[:40]:
            mem = sorted((c for c in self.con2stock[r["concept_code"]]
                          if c in quotes and "ST" not in quotes[c]["name"]
                          and quotes[c]["limit_px"] > 0),
                         key=lambda c: -quotes[c]["pct"])
            tops = []
            for c in mem[:5]:
                q = quotes[c]
                s = prob_by.get(c)
                tops.append({"name": q["name"],
                             "pct": round(q["pct"], 2),
                             "prob": s["prob"] if s else None,
                             "near": bool(s and s["near"])})
            r["top"] = tops
        near_cnt = sum(1 for s in stocks_all if s["near"])
        n_hot = sum(1 for r in themes if r["heat"] >= HOT_THRESHOLD)
        # 半路前向预警(研究12/14c/16前向验证规则, 纯历史轨迹无未来信息)
        today_s = now.strftime("%Y%m%d")
        if self._yest_date != today_s and QUOTE_SOURCE == "qmt":
            # 每日一次拉昨日收盘位置(S3叠加因子, 失败降级为空)
            from quotes.qmt import fetch_prev_cpos
            self._yest_cpos = fetch_prev_cpos(self.codes)
            self._yest_date = today_s
            print(f"昨日收盘位置加载 {len(self._yest_cpos)}只")
        if self._struct_date != today_s and QUOTE_SOURCE == "qmt":
            # 每日一次 V5 结构层快照(研究24定稿, 影子输出不改触发)
            from quotes.qmt import fetch_daily_bars
            bars = fetch_daily_bars(self.codes)
            yest = max((d for cs in bars.values() for d, *_ in cs
                        if d != today_s), default=None)
            ldlr = fetch_ldlr_prev(yest) if yest else None
            self._struct = build_struct_scores(self.codes, bars, ldlr)
            self._struct_date = today_s
            n_gate = sum(1 for v in self._struct.values() if v["gate"])
            print(f"V5结构层快照 {len(self._struct)}只, 结构闸通过"
                  f" {n_gate}只, ldlr_prev={ldlr}")
        presig = build_signals(self.hist, quotes, t, today_s,
                               self._yest_cpos)
        # 当日累积(每票每级只留首次, 跨cycle不丢)
        if self._presig_date != today_s:
            self._presig_day = {}
            self._presig_date = today_s
            # 重启回载: 从当日状态文件恢复信号与价格时间线, 回溯不丢
            f_ps = LIVE / f"presig_state_{today_s}.json"
            if f_ps.exists():
                try:
                    saved = json.loads(f_ps.read_text(encoding="utf-8"))
                    for d in saved.get("signals", []):
                        key = (d["ts_code"], d["stage"])
                        self._presig_px[key] = d.pop("px_hist", [])
                        d.setdefault("sealed_t", None)   # 兼容旧版状态文件
                        d.setdefault("pb", None)
                        d.setdefault("pt", None)
                        d.setdefault("touch_t", None)
                        d.setdefault("zb_cnt", 0)
                        self._presig_day[key] = d
                    print(f"[{now:%H:%M:%S}] 预警信号回载 "
                          f"{len(self._presig_day)}条")
                except Exception as e:
                    print(f"[{now:%H:%M:%S}] 预警信号回载失败: {e}")
        if self._ipx_date != today_s:      # 全天分时回载(真实价+累计量额)
            self._ipx_date = today_s
            f_ipx = LIVE / f"intraday_px_{today_s}.json"
            try:
                self._ipx_day = json.loads(f_ipx.read_text(encoding="utf-8")) \
                    if f_ipx.exists() else {}
            except Exception:
                self._ipx_day = {}
        for s in presig:
            s["t"] = now.strftime("%H:%M:%S")
            key = (s["ts_code"], s["stage"])
            if key not in self._presig_day:
                # 推荐买入: 比例深度(S3=触发价×0.5% / S2=0.4%, 与研究口径一致)
                q0 = quotes.get(s["ts_code"])
                if q0 and s["stage"] == "S3":
                    s["pb"] = round(q0["price"] * 0.995, 2)
                    s["pt"] = s["t"]
                elif q0 and s["stage"] == "S2":
                    s["pb"] = round(q0["price"] * 0.996, 2)
                    s["pt"] = s["t"]
                else:
                    s["pb"], s["pt"] = None, None
                s["sealed_t"] = None
                if q0:
                    s["price0"] = q0["price"]   # 触发价(信号时刻价)
                # V5结构层影子(研究24): 结构闸+融合分, 只展示不拦截
                st = self._struct.get(s["ts_code"])
                if st:
                    s["struct"] = {
                        "g_chip": st["g_chip"], "gate": st["gate"],
                        "v5": v5_full(st["v5_base"], s.get("r3"),
                                      s.get("pathvol")),
                        "zb20": st["zb_cnt20"], "ir": st["ind_rank"]}
            self._presig_day[key] = s
        # 已封板信号票: 补涨停形态与模型归属(封板后不变, 只算一次;
        # “未知/未细分”类允许在轨迹回载后重算升级)
        for s in self._presig_day.values():
            old = s.get("zt_shape", "")
            if old and "未知" not in old and "未细分" not in old:
                continue
            q = quotes.get(s["ts_code"])
            if q:
                r = zt_shape_of(s["ts_code"], self.hist.get(s["ts_code"]),
                                q, q.get("open", 0.0),
                                self._open_traj.get(s["ts_code"]))
                if r:
                    s["zt_shape"], s["mode"] = r
        # 价格时间线: S2/S3全记 + S1封板后记; 顺带更新封板时刻
        ts_str = now.strftime("%H:%M:%S")
        for key, s in self._presig_day.items():
            q = quotes.get(s["ts_code"])
            px_last = self._last_px.get(s["ts_code"])
            if q:
                self._last_px[s["ts_code"]] = q["price"]
                px_last = q["price"]
            if q:
                self._presig_px.setdefault(key, []).append(
                    [ts_str, q["price"], q.get("volume", 0),
                     q.get("amount", 0)])
            # 封板事件流: 触板≠封死(研究30) — 封死=价格贴死涨停价且
            # 连续保持≥LOCK_HOLD秒; 炸板=曾封死后回落。事件流 1=封死 0=炸板
            if s.get("limit_px") and px_last is not None:
                lp = s["limit_px"]
                if px_last >= lp * TOUCH_EPS and not s.get("touch_t"):
                    s["touch_t"] = ts_str      # 首次触板时刻
                if px_last >= lp * LOCK_EPS:
                    self._lock_since.setdefault(key, time.time())
                else:
                    self._lock_since.pop(key, None)
                since = self._lock_since.get(key)
                cur_seal = bool(since and time.time() - since >= LOCK_HOLD)
                ev = s.setdefault("zt_ev", [])
                # 旧信号已有sealed_t视为初始封板状态
                last = ev[-1][1] if ev else \
                    (1 if s.get("sealed_t") else None)
                if last is None and cur_seal:
                    ev.append([ts_str, 1])
                    # sealed_t 创建时已置None(键已存在), 必须显式判空回填;
                    # setdefault 不覆盖 None 值 → 旧写法使该字段永远为空
                    if not s.get("sealed_t"):
                        s["sealed_t"] = ts_str
                elif last == 1 and not cur_seal:
                    ev.append([ts_str, 0])
                    s["zb_cnt"] = int(s.get("zb_cnt", 0)) + 1
                elif last == 0 and cur_seal:
                    ev.append([ts_str, 1])
        # 回放状态落盘(含价格时间线, 供看板时间轴回溯)
        state = {"date": today_s, "signals": []}
        for key, s in self._presig_day.items():
            d = dict(s)
            d["px_hist"] = self._presig_px.get(key, [])
            state["signals"].append(d)
        state["signals"].sort(
            key=lambda s: (s["stage"] != "S2", s["stage"] != "S3",
                           -s["pct"]))
        (LIVE / f"presig_state_{today_s}.json").write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8")
        presig_all = sorted(
            self._presig_day.values(),
            key=lambda s: (s["stage"] != "S2", s["stage"] != "S3",
                           -s["pct"]))
        for s in presig:
            if s["stage"] == "S2":
                print(f"[{now:%H:%M:%S}] 前向预警 "
                      f"{s['name']} {s['pct']}% [{s['why']}] "
                      f"r3={s['r3']} pv={s['pathvol']}")
        self.cycle += 1
        focus_snap = self._build_focus(sw, themes, presig_all)
        snap = {"ts": now.strftime("%H:%M:%S"), "trading": True,
                "interval": INTERVAL, "themes": themes[:40],
                "external": external, "near_cnt": near_cnt,
                "n_hot": n_hot,
                "seesaw": seesaw_snap,
                "sw": sw,
                "sw_traj": self._sw_traj,
                "focus": focus_snap,
                "presignals": presig_all[:80],
                "stocks": [s for s in stocks_all
                           if not s["near"] and s["prob"] >= 0.05][:80]}
        (LIVE / "radar.json").write_text(json.dumps(snap, ensure_ascii=False),
                                         encoding="utf-8")
        # 校准日志: 每cycle(20s)一次, 涨幅≥1%或概率≥0.2全量(含负例),
        # 研究05发现1分钟粒度丢失ramp轨迹(赢家首条≥4%日志已在+9%),
        # 研究10发现pct≥3门槛造成启动前盲区(无法提前抓板),
        # 降为≥1%积累启动初期样本; dp=较上轮(20s前)概率变化
        with open(LIVE / f"radar_log_{now:%Y%m%d}.jsonl", "a",
                  encoding="utf-8") as f:
            for c, q in quotes.items():
                if "ST" in q["name"] or c.endswith(".BJ"):
                    continue
                s = prob_by.get(c)
                prob = s["prob"] if s else 0.0
                if q["limit_px"] <= 0 or (q["pct"] < 1 and prob < 0.2):
                    continue
                self._ipx_day.setdefault(c, []).append(
                    [now.strftime("%H%M%S"), q["price"],
                     q.get("volume", 0), q.get("amount", 0)])
                f.write(json.dumps({
                    "t": now.strftime("%H%M%S"), "code": c,
                    "name": q["name"], "pct": round(q["pct"], 2),
                    "vol": q.get("volume", 0), "amt": q.get("amount", 0),
                    "s1": s["s1"] if s else 0.0,
                    "s3": s["s3"] if s else 0.0,
                    "s5": s["s5"] if s else 0.0,
                    "vr": q["vr"], "tover": q["tover"],
                    "dist": s["dist"] if s else 0.0,
                    "prob": prob,
                    "dp": round(prob - self._log_prob.get(c, prob), 3),
                    "heat": s["heat"] if s else 0.0,
                    "trank": rank_by.get(s["hk"], 99) if s else 99,
                    "dheat": window_diff(self._heat_hist.get(s["hk"]), 300, t) if s else 0.0,
                    "theme": s["theme"] if s else "-",
                    "near": bool(s and s["near"])},
                    ensure_ascii=False) + "\n")
        # 分时扩围: 口径A龙头/当日龙头/监测概念中军无条件记录轨迹,
        # 供龙头拐头事件离线回测(现有门槛仅覆盖涨幅≥1%票)
        hm_s = now.strftime("%H%M%S")
        for c in self.seesaw.track_codes:
            q = quotes.get(c)
            if not q or q.get("limit_px", 0) <= 0:
                continue
            tr = self._ipx_day.setdefault(c, [])
            if not tr or tr[-1][0] != hm_s:
                tr.append([hm_s, q["price"], q.get("volume", 0),
                           q.get("amount", 0)])
        # ~5min 周期落盘; 15:00后每轮强制落盘(收盘尾段不因进程
        # 非优雅退出丢失, 实测某日只落到14:56, 尾段全丢)
        if self.cycle % 15 == 0 or now.strftime("%H%M") >= "1500":
            if self._ipx_day:
                try:
                    (LIVE / f"intraday_px_{now:%Y%m%d}.json").write_text(
                        json.dumps(self._ipx_day), encoding="utf-8")
                except Exception as e:
                    print(f"[{now:%H:%M:%S}] intraday_px 落盘失败: {e}")
            if self.seesaw and self.seesaw.con_day:   # 板块级分时同步落盘
                try:
                    (LIVE / f"concept_px_{now:%Y%m%d}.json").write_text(
                        json.dumps(self.seesaw.con_day), encoding="utf-8")
                except Exception as e:
                    print(f"[{now:%H:%M:%S}] concept_px 落盘失败: {e}")
            if self._sw_traj:      # 申万轨迹同步落盘(重启回载不丢)
                try:
                    (LIVE / f"sw_traj_{now:%Y%m%d}.json").write_text(
                        json.dumps(self._sw_traj), encoding="utf-8")
                except Exception as e:
                    print(f"[{now:%H:%M:%S}] sw_traj 落盘失败: {e}")
        self._log_prob = {c: s["prob"] for c, s in prob_by.items()}
        top = snap["stocks"]
        print(f"[{now:%H:%M:%S}] 雷达 扫描{len(quotes)} "
              f"热题:{[r['name'] for r in themes[:3]]} "
              f"概率TOP:{[(s['name'], s['prob']) for s in top[:3]]} "
              f"耗时{time.time() - t:.1f}s")
        return time.time() - t

    def _build_focus(self, sw: list, themes: list, presig_all: list):
        """专注面板数据: focused申万聚合 + focused概念热度 + 属于focus的
        S2/S3信号。focus为空返回None(前端不渲染面板)。
        信号归属: 申万靠sw_map反查L1/L2, 概念靠stock2con求交集。"""
        items = (self._focus or {}).get("items", [])
        if not items:
            return None
        f_l1 = {it["name"] for it in items if it.get("type") == "sw_l1"}
        f_l2 = {it["name"] for it in items if it.get("type") == "sw_l2"}
        f_con = {it["name"] for it in items if it.get("type") == "concept"}
        f_con_code = {it["code"] for it in items
                      if it.get("type") == "concept" and it.get("code")}
        fsw = []
        for r in sw:
            if r["l1"] in f_l1:
                fsw.append(r)
            else:
                l2s = [x for x in r["l2"] if x["l2"] in f_l2]
                if l2s:
                    rr = dict(r)
                    rr["l2"] = l2s
                    fsw.append(rr)
        fcon = [t for t in themes
                if t["name"] in f_con or t["concept_code"] in f_con_code]
        fsig = []
        for s in presig_all:
            if s.get("stage") == "S1":
                continue
            m = self._sw_map.get(s["ts_code"])
            in_sw = bool(m and (m["l1"] in f_l1 or m["l2"] in f_l2))
            in_con = any(c in f_con_code
                         for c in self.stock2con.get(s["ts_code"], []))
            if in_sw or in_con:
                fsig.append(s)
        return {"items": items, "sw": fsw, "concepts": fcon,
                "signals": fsig[:40]}

    def flush_state(self):
        """内存状态强制落盘(优雅退出/崩溃前调用)"""
        today_s = datetime.now().strftime("%Y%m%d")
        try:
            if self._open_traj:
                (LIVE / f"open_traj_{today_s}.json").write_text(
                    json.dumps(self._open_traj), encoding="utf-8")
            state = {"date": today_s, "signals": []}
            for key, s in self._presig_day.items():
                d = dict(s)
                d["px_hist"] = self._presig_px.get(key, [])
                state["signals"].append(d)
            if state["signals"]:
                (LIVE / f"presig_state_{today_s}.json").write_text(
                    json.dumps(state, ensure_ascii=False), encoding="utf-8")
            if self._ipx_day:
                (LIVE / f"intraday_px_{today_s}.json").write_text(
                    json.dumps(self._ipx_day), encoding="utf-8")
            if self.seesaw and self.seesaw.con_day:
                (LIVE / f"concept_px_{today_s}.json").write_text(
                    json.dumps(self.seesaw.con_day), encoding="utf-8")
            if self._sw_traj:
                (LIVE / f"sw_traj_{today_s}.json").write_text(
                    json.dumps(self._sw_traj), encoding="utf-8")
            print(f"内存状态落盘完成: 信号{len(state['signals'])}条 "
                  f"开盘轨迹{len(self._open_traj)}只 "
                  f"分时{len(self._ipx_day)}只")
        except Exception as e:
            print(f"内存状态落盘失败: {e}")

    def run(self):
        while True:
            if not is_trading_hours(datetime.now()):
                time.sleep(120)
                continue
            elapsed = self.once()
            time.sleep(max(1.0, INTERVAL * min(4, 2 ** self.bad_sweep)
                           - elapsed))


if __name__ == "__main__":
    import signal as _signal
    _radar = Radar()

    def _shutdown(signum, frame):
        print("收到终止信号, 落盘内存轨迹后退出...")
        _radar.flush_state()
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _shutdown)
    _signal.signal(_signal.SIGINT, _shutdown)
    _radar.run()
