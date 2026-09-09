# -*- coding: utf-8 -*-
"""题材级信号 T1/T2/T3 —— 唯一出处（影子口径，不接买卖拦截）

命名体系: **S = 个股级**(S1/S2/S3, core/early_signal), **T = 题材级**(本模块)。
等级规则: **数字越大越好** —— T3 最优, T1 最弱(风险降级)。
与 S 级同向(S1 最早感知最弱 → S3 确认最强), 两套编号语义一致。

判据锚定两层, 都不含未来信息:
  ① **昨日复盘题材天梯**(theme.day 的 D-1 行) —— 09:15 就完全已知。
     刻意不用当日 theme.day: 当日龙头是从当日涨停股里选出的, 盘中无法
     知道今天谁是龙一(研究34 的前视教训)。
  ② **截至当前时刻的盘中实时计数** —— 题材内贴死涨停价的家数。

  T3 题材启动确认  封板家数 ≥ T3_MIN_ZT 且 昨日该题材最高板 ≥ T3_MIN_HT
                   (研究35 TEST 封板率 25.3%, lift 3.48x, EV +0.41)
  T2 题材初步确认  封板家数 ≥ T2_MIN_ZT 且 昨日有活跃题材
                   (研究35 TEST 19.6%, lift 2.69x, EV +0.23)
  T1 题材无确认 / 龙一负反馈
                   · 无昨日活跃题材(散票, 封板率 3.4%, 占 54% 信号量)
                   · 昨日龙一今日开盘 ≤ LD_NEG_GAP(封板率 7.5%)
                   最弱档, 作风险降级标记

封板口径: 现价贴死涨停价(price ≥ limit_px × SEAL_EPS)。用 SEAL_EPS 而非
触板口径 0.995 —— 研究30 实测触板会显著高估封板率(触板≠封死)。

样本边界: 研究35 仅 5 个交易日 / TEST 117 条, 未过牛熊震荡三段验证,
故 T 级**只作观察与看板排序, 不得作为买入闸**(与 V5 结构层、竞价闸同原则)。
"""
SEAL_EPS = 0.9995          # 贴死涨停价判据(同 apps/radar.LOCK_EPS)
T3_MIN_ZT = 2              # T3(最优): 题材内封板家数下限
T3_MIN_HT = 2              # T3(最优): 昨日该题材最高板下限
T2_MIN_ZT = 1              # T2: 题材内封板家数下限
LD_NEG_GAP = -3.0          # T1(最弱): 昨日龙一今日开盘涨幅 ≤ 此值 = 负反馈

LEVELS = ("T1", "T2", "T3")     # 下标越大越好(排序取最优时用 -index)
LEVEL_DESC = {
    "T3": "题材启动确认(封板≥2且昨日有高标) — 最优",
    "T2": "题材初步确认(封板≥1)",
    "T1": "题材无确认/龙一负反馈 — 最弱",
}


def prev_ladder_of(td_prev) -> dict:
    """昨日题材天梯 {concept_code: {name,leader_code,leader_name,
    leader_height,max_height,zt_cnt,theme_age}}

    入参为 theme.day 中 trade_date == D-1 的切片(DataFrame)。
    盘前完全已知, 是 T 级判定唯一的静态锚。"""
    out = {}
    for r in td_prev.itertuples():
        if not isinstance(r.concept_code, str):
            continue
        out[r.concept_code] = {
            "name": r.concept_name,
            "leader_code": r.leader_code if isinstance(r.leader_code, str)
            else None,
            "leader_name": r.leader_name,
            "leader_height": int(r.leader_height or 0),
            "max_height": int(r.max_height or 0),
            "zt_cnt": int(r.zt_cnt or 0),
            "theme_age": int(r.theme_age or 0),
        }
    return out


def theme_seal_codes(members: list, quotes: dict) -> list:
    """题材内当前贴死涨停价的成分股(SEAL_EPS口径, 触板不计)"""
    return [c for c in members
            if c in quotes and quotes[c].get("limit_px", 0) > 0
            and "ST" not in quotes[c].get("name", "")
            and quotes[c]["price"] >= quotes[c]["limit_px"] * SEAL_EPS]


def t_level_of(lad: dict | None, zt_now: int, ld_gap: float | None) -> tuple:
    """单个题材的 T 级判定 → (level, evidence)

    lad:     昨日天梯行(prev_ladder_of 的值); None = 昨日该题材无涨停(散票)
    zt_now:  题材内当前封板家数
    ld_gap:  昨日龙一今日开盘涨幅%; None = 无法取得(不判负反馈)
    """
    if lad is None:
        # 昨日无活跃题材 = 散票。研究34: 占54%信号量且封板率3.4%
        return "T1", {"level": "T1", "why": "昨日无活跃题材(散票)",
                      "zt_now": zt_now, "n_sealed": zt_now,
                      "y_ht": 0, "y_zt": 0, "y_age": 0,
                      "ld_code": None, "ld_name": None, "ld_ht": 0,
                      "ld_gap": None, "neg_fb": False}
    ld_gap_bad = (ld_gap is not None and ld_gap <= LD_NEG_GAP)
    ev = {"level": None, "zt_now": zt_now,
          "y_ht": lad["max_height"], "y_zt": lad["zt_cnt"],
          "y_age": lad["theme_age"],
          "ld_code": lad["leader_code"], "ld_name": lad["leader_name"],
          "ld_ht": lad["leader_height"], "ld_gap": ld_gap,
          "neg_fb": ld_gap_bad}
    if ld_gap_bad:
        # 龙一负反馈优先级最高: 龙一低开下杀时即便题材有封板也降到最弱档
        ev["level"] = "T1"
        ev["why"] = f'龙一{lad["leader_name"]}今低开{ld_gap:+.1f}%(负反馈)'
        return "T1", ev
    if zt_now >= T3_MIN_ZT and lad["max_height"] >= T3_MIN_HT:
        ev["level"] = "T3"
        ev["why"] = (f'封板{zt_now}家且昨日最高{lad["max_height"]}板')
        return "T3", ev
    if zt_now >= T2_MIN_ZT:
        ev["level"] = "T2"
        ev["why"] = f'封板{zt_now}家(昨日最高{lad["max_height"]}板)'
        return "T2", ev
    ev["level"] = "T1"
    ev["why"] = "题材暂无封板"
    return "T1", ev


def build_theme_signals(con2stock: dict, quotes: dict, ladder: dict) -> dict:
    """全题材 T 级快照 {concept_code: evidence}

    只算昨日天梯里出现过的题材(昨日无涨停的题材无从判 T2/T3, 其成分股
    在信号侧记为散票 T1)。每 cycle 调用一次, 纯计算无网络。"""
    out = {}
    for k, lad in ladder.items():
        members = con2stock.get(k, [])
        if not members:
            continue
        sealed = theme_seal_codes(members, quotes)
        lc = lad.get("leader_code")
        ld_gap = None
        if lc and lc in quotes:
            q = quotes[lc]
            op = q.get("open") or 0
            pct = q.get("pct")
            # 昨收从现价与涨幅反推(两个行情源都不直接给 pre_close)
            pre = (q["price"] / (1 + pct / 100)
                   if pct is not None and pct > -100 and q["price"] > 0
                   else 0)
            if op > 0 and pre > 0:
                ld_gap = (op / pre - 1) * 100
            # open 缺失时不拿现价涨幅冒充开盘涨幅——两者语义不同,
            # 冒充就是一次"错值看起来像正常值"
        lvl, ev = t_level_of(lad, len(sealed), ld_gap)
        ev["name"] = lad["name"]
        ev["concept_code"] = k
        ev["sealed_codes"] = sealed[:12]
        ev["n_sealed"] = len(sealed)
        out[k] = ev
    return out


def signal_theme_of(stock2con: dict, tsig: dict, code: str) -> dict | None:
    """个股 → 其所属题材里 T 级最高的那条(供盘面标注 S/T)

    数字越大越好 → 取 level 下标最大者; 同级取封板家数多者。
    都不在昨日天梯内则返回 None(调用方据此记散票 T1)。"""
    best = None
    for k in stock2con.get(code, []):
        ev = tsig.get(k)
        if ev is None:
            continue
        rank = (-LEVELS.index(ev["level"]), -ev["n_sealed"])
        if best is None or rank < best[0]:
            best = (rank, ev)
    return best[1] if best else None


def scatter_evidence() -> dict:
    """散票(不属于任何昨日活跃题材)的 T1 证据"""
    return {"level": "T1", "why": "昨日无活跃题材(散票)", "name": None,
            "concept_code": None, "zt_now": 0, "n_sealed": 0,
            "y_ht": 0, "y_zt": 0, "ld_name": None, "ld_gap": None,
            "neg_fb": False}
