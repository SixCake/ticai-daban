# -*- coding: utf-8 -*-
"""概念独占归属（题材归属唯一权威出处）

数据源(config.CONCEPT_SOURCE):
  kpl(默认) = 开盘啦题材。归属层用事件级theme直标(当日数据T+1才可拉,
    盘中/当日盘后用最近标注延续法近似, 与官方直标一致率~68%且无未来
    信息); 成分层(雷达宇宙/热度/中军)用板块成分∪近窗口theme标注对。
    2026-09 A/B验证: 未归属率0.15%/最大簇占比23%, 优于同花顺投票的
    1.25%/48%。
  ths = 同花顺概念(旧源, 保留供回滚对照)。迭代投票归属:
  1. 只用 is_theme=True 的概念（过滤指数样本/属性类/超大杂烩）
  2. 每股候选 = 其成分概念 ∩ 当日有涨停股出现的概念
  3. 迭代: 每概念统计当前归属家数 → 每股归到候选中家数最大者
     平票取当日涨停密度(raw/成分数)更高者(真热点优先, v2),
     再平取成分数更小者(更聚焦), 再平取概念代码小者
  4. 收敛到不动点（上限20轮）；无候选者归 UNASSIGNED

v2密度tie-break背景(研究03): v1"成分数小"会把深中华A锁进两轮车(77成分,raw1)
而丢掉黄金概念(82成分,raw3)、金健米业锁进乳业(35,raw1)丢掉粮食概念(47,raw4)。
密度tie-break使大热点漏标 48.8%→21.7%, 现实格 n 984→4237、日聚类t 22.6→36.0。

离线全量产物由 build/attribute.py 写出(theme.attribution);
盘中实时归属(poller)与雷达(radar)直接调用本模块函数。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONCEPT_SOURCE, KPL_THEME_WINDOW  # noqa: E402
from datastore import load  # noqa: E402


def load_maps():
    """只读加载: 返回 (stock2con, msize, cname)"""
    if CONCEPT_SOURCE == "kpl":
        return _kpl_maps()
    concepts = load("theme.concepts")
    members = load("theme.members")
    theme = concepts[concepts["is_theme"]]
    theme_codes = set(theme["ts_code"])
    msize = theme.set_index("ts_code")["member_count"].to_dict()
    cname = theme.set_index("ts_code")["name"].to_dict()

    # 股票 → [题材概念]
    mem = members[members["concept_code"].isin(theme_codes)]
    stock2con = (mem.groupby("con_code")["concept_code"]
                 .apply(lambda s: sorted(set(s))).to_dict())
    return stock2con, msize, cname


def load_con2stock() -> dict:
    """概念 → 成分股代码列表"""
    if CONCEPT_SOURCE == "kpl":
        return _kpl_con2stock()
    concepts = load("theme.concepts")
    members = load("theme.members")
    theme_codes = set(concepts[concepts["is_theme"]]["ts_code"])
    mem = members[members["concept_code"].isin(theme_codes)]
    return (mem.groupby("concept_code")["con_code"]
            .apply(lambda s: sorted(set(s))).to_dict())


def attribute_day(codes: list[str], stock2con: dict, msize: dict):
    """单日独占归属, 返回 ({ts_code: concept_code}, 迭代轮数)"""
    raw_cnt = defaultdict(int)          # 归属前每概念触及家数
    cand = {}
    for c in codes:
        cons = stock2con.get(c, [])
        cand[c] = cons
        for k in cons:
            raw_cnt[k] += 1
    # 候选只保留当日有涨停出现的概念
    active = set(raw_cnt)
    for c in codes:
        cand[c] = [k for k in cand[c] if k in active]

    attr = {}
    dens = {k: raw_cnt[k] / msize.get(k, 10**9) for k in active}
    for rnd in range(1, 21):
        cnt = defaultdict(int)
        for c, k in attr.items():
            cnt[k] += 1
        new_attr = {}
        for c in codes:
            ks = cand[c]
            if not ks:
                new_attr[c] = "UNASSIGNED"
                continue
            # cnt最大; 平票取当日涨停密度高(真热点优先); 再平取成分数小; 再平取代码升序
            new_attr[c] = sorted(
                ks, key=lambda k: (-cnt.get(k, 0), -dens.get(k, 0),
                                   msize.get(k, 10**9), k))[0]
        if new_attr == attr:
            return new_attr, rnd
        attr = new_attr
    return attr, 20


def touch_map(codes: list[str], stock2con: dict, msize: dict):
    """多概念触及层(展示用, 不参与独占统计): 每股 → 当日有涨停出现的全部
    成分题材概念。返回 (raw_cnt: {概念: 触及家数}, touches: {股票: [概念,...]}),
    touches 按触及家数降序 → 成分数小 → 代码升序。"""
    raw_cnt = defaultdict(int)
    cons_of = {}
    for c in codes:
        cons = stock2con.get(c, [])
        cons_of[c] = cons
        for k in cons:
            raw_cnt[k] += 1
    touches = {c: sorted(cons_of[c],
                         key=lambda k: (-raw_cnt[k], msize.get(k, 10**9), k))
               for c in codes}
    return dict(raw_cnt), touches


# ---- 归属置信(单一口径, poller盘中/复盘共用) ----
# low 两种成因: ①候选概念<3个=被迫二选一(锦龙股份只剩期货/算力租赁);
# ②行业零存在=该股主行业在归属概念成分中一只都没有(冀衡医药·化学制药→水泥概念);
# 不用占比阈值——题材天然跨行业(化肥含种植/农化), 占比会冤杀真成员;
# 仅标记供展示层提示, 不改归属算法与下游(现实格/角色)口径
def _major_industry(x):
    return re.sub(r"[ⅠⅡⅢ]+$", "", x) if x else None


def industry_match(code: str, concept_code: str, con2stock: dict,
                   industry_map: dict) -> bool:
    """该股主行业(去Ⅰ/Ⅱ/Ⅲ后缀)在归属概念成分中至少存在1只视为匹配;
    行业/成分缺失时视为匹配(宁可放过不冤标)"""
    my = _major_industry(industry_map.get(code))
    if not my:
        return True
    mem = con2stock.get(concept_code, [])
    if not mem:
        return True
    return any(_major_industry(industry_map.get(m)) == my for m in mem)


def conf_level(code: str, concept_code, cand_n: int, con2stock: dict,
               industry_map: dict) -> str:
    """归属置信: none=无归属, low=稀疏或行业零存在, ok=正常"""
    if concept_code in (None, "UNASSIGNED"):
        return "none"
    if cand_n < 3:
        return "low"
    if not industry_match(code, concept_code, con2stock, industry_map):
        return "low"
    return "ok"


# ======================== 开盘啦(kpl)题材源 ========================
# kpl口径: 题材名即concept_code(cname恒等, cname.get(k,k)下游兼容);
# 板块(.KP代码)的名称与事件theme名是两个粒度的名称空间(对齐率仅~12%),
# 成分映射中板块以「板块名」为键与theme名共存, 天梯/热度按名称自然聚合。

_kpl_cache: dict = {}   # 进程级缓存(kpl_events为T+1, 盘中不变)


def split_themes(theme_str) -> list[str]:
    """kpl theme标注拆分为题材列表(顿号/中英逗号分隔, 保序); 空安全"""
    if not isinstance(theme_str, str) or not theme_str or theme_str == "nan":
        return []
    return [p.strip() for p in re.split(r"[,，、]", theme_str) if p.strip()]


def _kpl_events():
    if "ev" not in _kpl_cache:
        _kpl_cache["ev"] = load("limitup.kpl_events")
    return _kpl_cache["ev"]


def _kpl_pairs() -> dict:
    """近KPL_THEME_WINDOW交易日 (ts_code, theme) 事件对, 进程缓存
    返回 {ts_code: set(theme名)}"""
    if "pairs" not in _kpl_cache:
        ev = _kpl_events()
        dates = sorted(ev["trade_date"].unique())[-KPL_THEME_WINDOW:]
        recent = ev[ev["trade_date"].isin(dates)]
        s2t: dict = defaultdict(set)
        for c, th in zip(recent["ts_code"], recent["theme"]):
            for t in split_themes(th):
                s2t[c].add(t)
        _kpl_cache["pairs"] = dict(s2t)
    return _kpl_cache["pairs"]


def _kpl_boards() -> tuple[dict, dict]:
    """(con2stock: 板块名→[成分], msize: 板块名→成分数), 仅is_theme板块"""
    if "boards" not in _kpl_cache:
        concepts = load("theme.kpl_concepts")
        members = load("theme.kpl_members")
        theme_codes = set(concepts[concepts["is_theme"]]["ts_code"])
        mem = members[members["concept_code"].isin(theme_codes)]
        c2s = (mem.groupby("concept_name")["con_code"]
               .apply(sorted).to_dict())
        msize = {k: len(v) for k, v in c2s.items()}
        _kpl_cache["boards"] = (c2s, msize)
    return _kpl_cache["boards"]


def _kpl_maps():
    """kpl口径 load_maps: (stock2con, msize, cname)
    stock2con = 股票→[近窗口theme标注 ∪ 所在板块名];
    msize = 题材名→近窗口标注股票数 / 板块名→板块成分数; cname恒等"""
    pairs = _kpl_pairs()
    board_c2s, board_size = _kpl_boards()
    s2c: dict = defaultdict(set)
    for c, ts in pairs.items():
        s2c[c] |= set(ts)
    for b, cs in board_c2s.items():
        for c in cs:
            s2c[c].add(b)
    msize = {t: sum(1 for ts in pairs.values() if t in ts)
             for t in {t for ts in pairs.values() for t in ts}}
    msize.update(board_size)
    return ({c: sorted(ts) for c, ts in s2c.items()}, msize, {})


def _kpl_con2stock() -> dict:
    """kpl口径 load_con2stock: 题材名/板块名 → 股票列表
    题材名→近窗口被标注股票(动态活跃池, 中军B/跷跷板用),
    板块名→板块成分(静态全量)"""
    pairs = _kpl_pairs()
    board_c2s, _ = _kpl_boards()
    c2s: dict = defaultdict(set)
    for c, ts in pairs.items():
        for t in ts:
            c2s[t].add(c)
    for b, cs in board_c2s.items():
        c2s[b] |= set(cs)
    return {k: sorted(cs) for k, cs in c2s.items()}


def _kpl_last_theme(date: str) -> dict:
    """截至date前一交易日, 每股最近一次kpl标注的main theme(第一个题材)
    按日期缓存(date不变结果不变, poller逐轮复用)"""
    key = ("last", date)
    if key not in _kpl_cache:
        ev = _kpl_events()
        hist = ev[ev["trade_date"] < date]
        hist = hist.sort_values(["trade_date", "ts_code"]).groupby(
            "ts_code").tail(1)
        out = {}
        for c, th in zip(hist["ts_code"], hist["theme"]):
            ts = split_themes(th)
            if ts:
                out[c] = ts[0]
        _kpl_cache[key] = out
    return _kpl_cache[key]


def attribute_day_kpl(date: str, codes: list[str]):
    """kpl口径单日独占归属。返回 ({ts_code: 题材名}, 来源标记)
    来源: direct=当日kpl直标(T+1后重建/复盘) / carry=延续近似(盘中,
    直标缺失票亦由延续兜底)"""
    day = _kpl_events()
    day = day[day["trade_date"] == date]
    direct = {}
    for c, th in zip(day["ts_code"], day["theme"]):
        ts = split_themes(th)
        if ts and c not in direct:
            direct[c] = ts[0]
    last_map = _kpl_last_theme(date)
    attr = {}
    for c in codes:
        if c in direct:
            attr[c] = direct[c]
        elif c in last_map:
            attr[c] = last_map[c]
        else:
            attr[c] = "UNASSIGNED"
    return attr, ("direct" if len(direct) else "carry")


def touch_map_kpl(date: str, codes: list[str]):
    """kpl口径触及层: 当日全部theme标注(直标) / 近窗口标注并集(盘中)
    返回 (raw_cnt, touches) 与 touch_map 同契约。raw_cnt仅统计涨停
    tag事件(保持「涨停触及」语义, 与poller/theme_daily口径一致);
    touches含涨停+炸板全部标注, 按窗口内标注家数降序→题材名升序"""
    day = _kpl_events()
    day = day[day["trade_date"] == date]
    if len(day):
        raw_cnt: dict = defaultdict(int)
        for c, th in zip(day.loc[day["tag"] == "涨停", "ts_code"],
                         day.loc[day["tag"] == "涨停", "theme"]):
            for t in split_themes(th):
                raw_cnt[t] += 1
        touches = {}
        for c, th in zip(day["ts_code"], day["theme"]):
            touches[c] = split_themes(th)
        # 事件宇宙外的票(两源涨停榜差集)用近窗口标注兜底, 与归属层延续法一致
        pairs = _kpl_pairs()
        for c in codes:
            if c not in touches:
                touches[c] = sorted(pairs.get(c, ()))
        return dict(raw_cnt), touches
    pairs = _kpl_pairs()
    wcnt: dict = defaultdict(int)
    for ts in pairs.values():
        for t in ts:
            wcnt[t] += 1
    touches = {c: sorted(pairs.get(c, ()),
                          key=lambda k: (-wcnt.get(k, 0), k))
               for c in codes}
    return dict(wcnt), touches


# ---- 统一入口(盘中poller/复盘review共用, 屏蔽数据源差异) ----

def attribute_of(date: str, codes: list[str], stock2con: dict | None = None,
                 msize: dict | None = None):
    """单日归属统一入口: 返回 ({ts_code: concept_code}, 来源标记)
    ths源可传入已加载映射(poller长驻进程免每轮重读盘)"""
    if CONCEPT_SOURCE == "kpl":
        return attribute_day_kpl(date, codes)
    if stock2con is None or msize is None:
        stock2con, msize, _ = load_maps()
    attr, rnd = attribute_day(codes, stock2con, msize)
    return attr, f"vote{rnd}"


def touches_of(date: str, codes: list[str], stock2con: dict | None = None,
               msize: dict | None = None):
    """触及层统一入口: 返回 (raw_cnt, touches)"""
    if CONCEPT_SOURCE == "kpl":
        return touch_map_kpl(date, codes)
    if stock2con is None or msize is None:
        stock2con, msize, _ = load_maps()
    return touch_map(codes, stock2con, msize)
