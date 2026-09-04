# -*- coding: utf-8 -*-
"""半路涨停前向预警信号 — 生产实现(唯一出处)

规则来源: 研究12(238日全窗口) + 研究14c(walk-forward前向验证通过)。
只用决策时刻之前的20s轨迹(雷达hist), 无未来信息:

  S1 提前感知: 首触 ≥+1% (观察名单, 宁滥勿缺)
  S2 确认信号: 触 ≥+2% 且满足任一:
     - 颠簸高: 10cm板 且 pathvol > PV_HIGH (前向1.95x, EV+3.65%)
     - 颠簸中+急加速: pathvol > PV_MID 且 accel > 4 (训练期2.58x叶子)
     注: 暴拉分支已在研究15按EV证伪(封板53%但入场→次日仅+0.2%, 追高吃溢价), 移除
  S3 高开确认(开盘后第4分钟判定, 研究14/15 G组walk-forward):
     - 10cm 且 gap≥5.2 且 开盘3min回撤odip≤0.05 (前向89%封板)
       叠加昨日收强(y_cpos>0.6)时标记+: 92%封板, EV+4.47%(研究16)
     - 10cm 且 gap≤5.2 且 开盘3min振幅amp3>4.3 (前向64%封板)
     - 10cm 且 量比截面分位vr_pct≥0.95(开盘量爆)
       研究30实盘复核: 原绝对阈值 vr≥5 口径失效(生产vr=今日每分钟
       均量/近5日每分钟均量, 开盘20min内中位数7.4、vr≥5叠73%),
       导致该分支占全部S3的98%且封板率3.7%低于宇宙基线6.9%(反向);
       改用同快照横截面分位后, 样本外最优档(0.68~1.00)封板率13.2%
       vs 档1 4.9%。
       注: 本分支原名"竞价量爆"已改名为"开盘量爆"——它在09:34取样,
       用的是盘中量比而非竞价量比; 真正的竞价量比另走影子字段。

竞价口径分离(2026-09-03事故后定稿):
  竞价涨幅(gap)优先从竞价快照取(当日open口径, 全天可取、出处确定);
  无快照时才回退到 hist[0](雷达首轮轨迹点), 因为盘中重启/行情源
  故障时 hist[0] 可能不是竞价样本 → gap 与一字板判定全错。

口径换算: hist为20s粒度, pathvol(1min口径阈值0.93/0.5)按
sqrt(20/60)换算到20s粒度; r3/accel与bar粒度无关, 阈值原样。
执行口径(研究17/18验证: 回踩无逆向选择, 挂单等回踩EV单调升):
  S2 → exec=挂限价单低于触发价0.3~0.6点等回踩
  S3 → exec=挂限价单低于开盘价0.5~1.0点等回踩
每票每日每级只报一次(状态由日期隔离)。
"""
from bisect import bisect_left
from collections import deque

SCALE = (20 / 60) ** 0.5
PV_HIGH = 0.93 * SCALE        # ≈0.54, 对应1min口径0.93
PV_MID = 0.50 * SCALE         # ≈0.29
R3_BURST = 4.8                # 仅存档: 暴拉已按EV证伪, 不再触发信号
ACCEL_HOT = 4.0
VR_OPEN_HOT = 5.0             # 仅存档: 绝对量比阈值已证失效(研究30)
VR_PCT_HOT = 0.95             # S3 开盘量爆用同快照横截面分位
VR_PCT_MIN_N = 20             # 截面样本不足时不触发(宁缺不滥)
VR_PCT_MIN_PCT = 2.0          # S3 分位排名域下限(研究30口径, 勿改)


def cm20(code: str) -> bool:
    """20cm板(创业板30x/科创68x)"""
    return code[:2] in ("30", "68")


WIN_SEC = 600                 # pathvol计算窗口10min
GAP_BIG = 5.2                 # S3 高开大幅阈值
ODIP_TIGHT = 0.05             # S3 开盘回撤上限
AMP3_HOT = 4.3                # S3 开盘3min振幅下限
GAP_WIN = (9 * 3600 + 34 * 60, 9 * 3600 + 50 * 60)  # S3判定时间窗
_alerted: dict = {}           # (date, code) -> set(stages)


def prime_alerted(date: str, keys) -> None:
    """重启回载当日信号后预置已报集合。
    _alerted 只在内存, 重启后为空 → build_signals 会对已落盘的信号重报,
    而 radar 无条件覆盖 _presig_day[key] → 原触发时刻/触发价/买入价/
    封板事件流全部丢失。回载后必须调用本函数封口。
    keys: [(code, stage), ...]"""
    for c, stage in keys:
        _alerted.setdefault((date, c), set()).add(stage)


def _series(hist: deque, t: float, secs: int) -> list:
    """取截至t最近secs秒内的pct序列(时间升序)"""
    cut = t - secs
    return [p for (ts, p) in hist if ts >= cut]


def pathvol_of(series: list) -> float:
    """轨迹颠簸度: 相邻样本涨幅差的标准差(样本数不足返回0)"""
    if len(series) < 8:
        return 0.0
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    m = sum(diffs) / len(diffs)
    return (sum((d - m) ** 2 for d in diffs) / len(diffs)) ** 0.5


def pct_at(hist: deque, t: float, back_sec: int) -> float | None:
    """t-back_sec 时刻(最近邻)的pct"""
    cut = t - back_sec
    best = None
    for (ts, p) in hist:
        if ts <= cut:
            best = p
        else:
            break
    return best


def _sec_of_day(ts: float) -> int:
    import datetime as _dt
    d = _dt.datetime.fromtimestamp(ts)
    return d.hour * 3600 + d.minute * 60 + d.second


def vr_pct(quotes: dict, min_pct: float = VR_PCT_MIN_PCT) -> dict:
    """同快照横截面量比分位 {code: 0~1}。
    口径与研究30一致: 仅在涨幅≥min_pct且vr>0的非ST/北交所票内排名。
    截面样本不足时返回空字典(分位恒0 → 该分支不触发)。
    min_pct 可配: S3 开盘量爆传2.0(研究30原口径勿改), 竞价闸传1.0
    (S1候选域——若用2.0域则gap∈[1,2)的票分位恒0自动不过闸,
    等于悄悄把竞价闸变成2%门槛)。"""
    vals = sorted(q["vr"] for c, q in quotes.items()
                  if q.get("vr", 0) > 0 and q["pct"] >= min_pct
                  and "ST" not in q["name"] and not c.endswith(".BJ"))
    if len(vals) < VR_PCT_MIN_N:
        return {}
    return {c: bisect_left(vals, q["vr"]) / len(vals)
            for c, q in quotes.items() if q.get("vr", 0) > 0}


def build_signals(hist_by: dict, quotes: dict, t: float,
                  date: str, yest_cpos: dict | None = None,
                  auction: dict | None = None) -> list:
    """扫描全部行情, 输出前向预警列表(按强度降序)。
    hist_by: {code: deque[(epoch, pct)]} 雷达20s轨迹
    quotes: {code: 行情dict} 当前快照
    yest_cpos: {code: 昨日收盘位置0-1} 可选, S3叠加标记用
    auction: {code: {gap,...}} 竞价快照 可选, S3高开幅度优先取此"""
    out = []
    vrp = vr_pct(quotes)           # 量比截面分位(S3开盘量爆用)
    for c, q in quotes.items():
        if "ST" in q["name"] or c.endswith(".BJ"):
            continue
        pct = q["pct"]
        key = (date, c)
        done = _alerted.setdefault(key, set())
        hist = hist_by.get(c)
        if hist is None or len(hist) < 3:
            continue
        if pct >= 1 and "S1" not in done:
            done.add("S1")
            out.append(_mk("S1", c, q, hist, t, ""))
        # S3 高开确认: 开盘后第4分钟起判定一次(需开盘起累积轨迹)
        if ("S3" not in done and GAP_WIN[0] <= _sec_of_day(t) <= GAP_WIN[1]
                and len(hist) >= 5 and not cm20(c)):
            first_ts, first_p = hist[0]
            if _sec_of_day(first_ts) <= 9 * 3600 + 32 * 60:
                head = list(hist)[:4]
                # 竞价涨幅优先取竞价快照(出处确定); 无快照才用首轮轨迹点
                ag = (auction or {}).get(c)
                gap = (ag["gap"] if ag and ag.get("gap") is not None
                       else first_p)
                hi3 = max(p for _, p in head)
                odip = hi3 - pct
                amp3 = hi3 - min(p for _, p in head)
                why3 = []
                if gap >= GAP_BIG and odip <= ODIP_TIGHT:
                    why3.append("高开稳封相")
                elif gap <= GAP_BIG and amp3 > AMP3_HOT:
                    why3.append("高开剧震")
                elif vrp.get(c, 0.0) >= VR_PCT_HOT:
                    why3.append("开盘量爆")
                if why3:
                    done.add("S3")
                    s = _mk("S3", c, q, hist, t, why3[0])
                    s["gap"] = round(gap, 2)
                    s["vr_pct"] = round(vrp.get(c, 0.0), 3)
                    # 研究16叠加: 高开稳封相×昨日收强 → 92%封板/EV+4.47%
                    yc = (yest_cpos or {}).get(c)
                    if (why3[0] == "高开稳封相" and yc is not None
                            and yc > 0.6):
                        s["why"] = "高开稳封相+昨收强"
                    out.append(s)
                else:
                    done.add("S3")      # 不满足也只判一次
        if pct >= 2 and "S2" not in done:
            series = _series(hist, t, WIN_SEC)
            pv = pathvol_of(series)
            p3 = pct_at(hist, t, 180)
            p1a = pct_at(hist, t, 60)
            p1b = pct_at(hist, t, 120)
            r3 = pct - p3 if p3 is not None else 0.0
            accel = ((pct - p1a) - (p1a - p1b)
                     if p1a is not None and p1b is not None else 0.0)
            is20 = cm20(c)
            why = []
            # 暴拉分支已移除(研究15 EV证伪: 封板53%但总收益+0.2%)
            if not is20 and pv > PV_HIGH:
                why.append("颠簸高")
            if pv > PV_MID and accel > ACCEL_HOT:
                why.append("颠簸加速")
            if why:
                done.add("S2")
                out.append(_mk("S2", c, q, hist, t, "+".join(why),
                               pv=pv, r3=r3, accel=accel))
    out.sort(key=lambda s: (s["stage"] != "S2", -s["pct"]))
    return out


def zt_shape_of(code: str, hist, q: dict, open_px: float = 0.0,
                open_traj=None, auc_gap: float | None = None):
    """已封板票的涨停形态与模型归属分类。
    返回 (形态, 模式) 或 None(未封板/无数据)。
    形态: 一字板/高开稳封/高开剧震封板/高开拉升封板/颠簸拉升封板/平拉封板
    模式: 对应 S3稳封相/S3剧震/S2颠簸高(半路)/组外未标记/不可捕捉
    open_px: 当日开盘价(轨迹缺失时的补救源)
    open_traj: 开盘窗口轨迹[[epoch,pct],...] (雷达落盘回载, 重启不丢)
    auc_gap: 竞价涨幅(竞价快照口径), 高开/一字判定的首选出处"""
    lp = q.get("limit_px", 0)
    if lp <= 0 or q["price"] < lp * 0.995:
        return None
    ratio = 0.20 if cm20(code) else 0.10
    limit_pct = ratio * 100
    import datetime as _dt

    def _ts_ok(ts):
        d = _dt.datetime.fromtimestamp(float(ts))
        return d.hour * 60 + d.minute <= 9 * 60 + 32

    # 高开幅度出处优先级: 竞价快照 > 当日open价推算 > 轨迹首点。
    # 盘中重启/行情源故障时 samples[0] 可能不是竞价样本 → 会把非一字板
    # 误判成一字板(2026-09-03事故: 陈旧日bar被当成竞价行情)
    if auc_gap is None and open_px > 0 and lp > 0:
        auc_gap = (open_px / (lp / (1 + ratio)) - 1) * 100

    samples, src = [], ""
    live = list(hist) if hist else []
    if len(live) >= 3 and _ts_ok(live[0][0]):
        samples, src = live, "live"
    elif open_traj and _ts_ok(open_traj[0][0]):
        samples, src = [(float(ts), float(p)) for ts, p in open_traj], "open"
    if not samples:
        # 开盘轨迹缺失且无回载: 只能靠竞价涨幅粗判高开/一字
        if auc_gap is not None:
            if auc_gap >= limit_pct * 0.97:
                return ("一字板", "不可捕捉(一字)")
            if auc_gap >= 1.0:
                return ("高开拉升封板", "G组(开盘轨迹缺失,未细分)")
        return ("已封板", "形态未知(轨迹缺失)")
    first_p = auc_gap if auc_gap is not None else samples[0][1]
    head = [p for _, p in samples[:12]]       # 开盘前4分钟
    if first_p >= limit_pct * 0.97:
        return ("一字板", "不可捕捉(一字)")
    if first_p >= 1.0:                        # 高开 → G组口径
        hi3, lo3 = max(head), min(head)
        gap, amp3 = first_p, hi3 - lo3
        odip = hi3 - head[-1]
        if gap > GAP_BIG and odip <= ODIP_TIGHT:
            return ("高开稳封", "S3稳封相")
        if gap <= GAP_BIG and amp3 > AMP3_HOT:
            return ("高开剧震封板", "S3剧震")
        return ("高开拉升封板", "G组未标记")
    # 平开/低开 → L组口径(封板前轨迹颠簸度)
    if src == "open":                         # 仅开盘窗口, 封板前轨迹缺失
        return ("平开拉升封板", "L组(仅开盘轨迹)")
    tail = [p for _, p in samples][-11:]
    if len(tail) >= 8:
        diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
        m = sum(diffs) / len(diffs)
        pv = (sum((x - m) ** 2 for x in diffs) / len(diffs)) ** 0.5
        if pv > PV_HIGH:
            return ("颠簸拉升封板", "S2颠簸高(半路模式)")
    return ("平拉封板", "L组未标记(模型漏抓)")


def _mk(stage: str, c: str, q: dict, hist: deque, t: float,
        why: str, pv: float = 0.0, r3: float = 0.0,
        accel: float = 0.0) -> dict:
    exec_hint = ("挂限价单低于开盘价0.5~1.0点等回踩" if stage == "S3"
                 else "挂限价单低于触发价0.3~0.6点等回踩"
                 if stage == "S2" else "观察")
    return {
        "stage": stage, "ts_code": c, "name": q["name"],
        "pct": round(q["pct"], 2), "why": why, "exec": exec_hint,
        "r3": round(r3, 2), "accel": round(accel, 2),
        "pathvol": round(pv, 3), "vr": round(q["vr"], 2),
        "limit_px": q["limit_px"],
    }
