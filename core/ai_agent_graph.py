# -*- coding: utf-8 -*-
"""AI 题材智能体 — LangGraph 多角色编排(消息→题材→选股 的认知层)

拓扑移植自 TradingAgents(graph/setup.py + conditional_logic.py), 但两点关键
差异, 以守住本项目底线:
  ① 节点调 core.llm.chat_json(OpenAI 兼容), 不用 langchain-openai —— 保持
     ADR-0003「LLM 只在生产者」; 本模块只编排, 不含任何买卖/信号逻辑。
  ② 领域角色(非通用): 催化/题材阶段/龙头资金/情绪 4 分析师 → 看多·看空
     辩论(轮次封顶=确定性) → 裁判合成题材级因子。

角色数据输入全部用项目现有原语(无未来信息):
  催化   core.newsfeed 订阅新闻
  阶段   theme.day 的 wave_no/zt_all → core.cycle.theme_stage + theme_kb 剧本
  龙头   radar.json 题材热度/top + theme.kb_stock 四维画像(龙头/中军/补涨)
  情绪   latest.json sentiment(炸板率/涨停家数/最高板)

产物: run_graph() 返回 {conclusion, factor, debate, *_report}。factor 是
题材级因子(theme_stage/strength/sustainability/verdict/basket), 由生产者
(apps/ai_feed.py)落成带 ts 的可回放 feed。basket 只给「看谁」(角色/受益
纯度/理由), 不含买卖点 —— 买卖点仍由 S1/S2/S3 + 结构闸决定。

无 LLM 密钥/调用失败: 各角色降级为「事实数据简报」(rule_baseline, 不伪造
LLM 结论), factor 用确定性规则(_rule_factor)给出, 保证全链路可跑通。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from config import DATA
from core import llm

LIVE = DATA / "live"

# 看多/看空各发言一轮(共 2 turns)后交裁判; 固定轮次 → 确定性可回放
MAX_DEBATE_ROUNDS = 1

# 角色中文名(前端展示 + 事件 role 字段)
ROLE_NAMES = {
    "catalyst": "催化分析师", "chain": "产业链分析师",
    "stage": "题材阶段分析师", "leader": "龙头资金分析师",
    "sentiment": "情绪分析师",
    "bull": "看多研究员", "bear": "看空研究员", "judge": "裁判",
}
# 顺序: 催化→产业链(链路拆解+关联概念/细分赛道)→阶段→龙头→情绪
ANALYSTS = ["catalyst", "chain", "stage", "leader", "sentiment"]


class AgentState(TypedDict, total=False):
    theme: str
    day: str
    cutoff: float
    model: str | None
    news_items: list
    ctx: dict
    catalyst_report: str
    chain_report: str
    stage_report: str
    leader_report: str
    sentiment_report: str
    structs: dict
    debate: dict
    conclusion: str
    factor: dict
    src: str


# ---------- 数据装载(纯读, 无未来信息) ----------

def _load_json(path) -> dict | None:
    try:
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _basket_entry(c: dict, e: dict, rank: int) -> dict:
    """四维穿透候选(WHO to watch): 壁垒/兑现/竞争/天花板来自 KB 个股画像。
    tier: rank<=4 核心, 否则卫星(4+4+2 风险框架分层, 非买卖建议)。"""
    return {"ts_code": c.get("ts_code"), "name": c.get("name"),
            "chain_node": e.get("chain_node"), "role": e.get("role"),
            "moat": e.get("moat"), "realize_cycle": e.get("realize_cycle"),
            "competition": e.get("competition"), "ceiling": e.get("ceiling"),
            "benefit_purity": e.get("benefit_purity"),
            "four_dim_score": e.get("four_dim_score"),
            "tier": "core" if rank <= 4 else "satellite",
            "why": "KB四维画像" if e else "盘中题材龙头/热门",
            "rank": rank}


def _build_basket(theme: str, ctx: dict) -> list:
    """选股候选池(WHO to watch, 不含买卖点): 盘中题材龙头/热门 ∪ KB四维画像。
    选股不止选当日强势股 —— 四维穿透(壁垒/兑现/竞争/天花板)来自 KB,
    并按 4+4+2 分核心/卫星层。历史日(无 radar)退化为 KB 四维分排序。"""
    kbmap: dict = {}
    try:
        from core import theme_kb
        st = theme_kb.load_kb().get("stock")
        if st is not None and not st.empty:
            sub = st[st["concept_code"] == theme]
            for r in sub.itertuples():
                kbmap[r.ts_code] = {
                    "role": getattr(r, "role", None),
                    "chain_node": getattr(r, "chain_node", None),
                    "moat": getattr(r, "moat", None),
                    "realize_cycle": getattr(r, "realize_cycle", None),
                    "competition": getattr(r, "competition", None),
                    "ceiling": getattr(r, "ceiling", None),
                    "benefit_purity": getattr(r, "benefit_purity", None),
                    "four_dim_score": getattr(r, "four_dim_score", None)}
    except Exception:
        pass
    cands = []
    rt = ctx.get("radar_theme") or {}
    for t in rt.get("top") or []:
        cands.append({"ts_code": t.get("code"), "name": t.get("name")})
    if not cands:
        ld = (ctx.get("theme_day") or {}).get("leader_name")
        if ld:
            cands.append({"ts_code": None, "name": ld})
    out = [_basket_entry(c, kbmap.get(c["ts_code"], {}), i + 1)
           for i, c in enumerate(cands[:8])]
    if not out and kbmap:                     # 历史日: 直接用 KB 四维分排序
        rows = sorted(kbmap.items(),
                      key=lambda kv: -(kv[1].get("four_dim_score") or 0))[:8]
        out = [_basket_entry({"ts_code": code, "name": None}, e, i + 1)
               for i, (code, e) in enumerate(rows)]
    return out


_WORTH_JUDGE = ("你是题材深挖价值判断官。判断该题材/消息是否值得深入挖掘。"
                "判据(白盒可解释): 【值得深挖】=有明确催化(政策/产业/事件/技术)+"
                "有板块效应(多只个股联动)+有持续性潜力(非一次性消息); "
                "【不值得】=泛宏观消息(无具体板块)/一次性消息(无持续性)/"
                "无板块效应(仅单只个股)/纯利空/已充分发酵(高位滞涨)。"
                "输出JSON {worth(值得|不值得),reason(判断理由,引用消息依据),"
                "key_points(关键依据列表)}。")


def judge_worth(theme: str, news_items: list, model: str | None = None) -> dict:
    """深挖价值判断闸门: 判断题材是否值得深入挖掘(通过才继续深挖)。

    无LLM时默认放行(不阻断深挖); 有LLM时按白盒判据判断, 返回
    {worth: bool, reason, key_points, src}。"""
    if not llm.available():
        return {"worth": True, "reason": "无LLM密钥, 默认放行", "src": "rule"}
    news_txt = "\n".join(f"[{it.get('t','')}] {it.get('title','')}"
                         for it in (news_items or [])[:8]) or "(无相关消息)"
    user = f"题材: {theme}\n相关消息:\n{news_txt}"
    res = llm.chat_json(_WORTH_JUDGE, user, model=model)
    if not res:
        return {"worth": True, "reason": "判断调用失败, 默认放行", "src": "rule"}
    worth = str(res.get("worth")).strip() in (
        "值得", "值得深挖", "true", "True", "yes", "是")
    return {"worth": bool(worth), "reason": res.get("reason"),
            "key_points": res.get("key_points"), "src": "llm"}


def gather_ctx(theme: str, day: str, cutoff: float) -> dict:
    """预取该题材当日的所有事实数据(供各角色复用, 避免逐节点读盘)。

    防未来信息: theme.day 用 `day` 当日行(历史安全); radar/latest 是**当前**
    快照, 仅当 day==今日才用 —— 补产历史日时不引入当日盘口(否则回放污染)。"""
    ctx: dict = {"theme": theme, "day": day}
    from core.calendar import is_polling_hours
    live = is_polling_hours(datetime.now())   # False=已过实盘交易窗口(盘后/盘前/周末)
    try:
        from datastore import load
        td = load("theme.day", columns=[
            "trade_date", "concept_code", "concept_name", "zt_cnt", "zt_all",
            "wave_no", "theme_age", "max_height", "leader_name",
            "leader_height"])
        row = td[(td["concept_code"] == theme) & (td["trade_date"] == day)]
        if row.empty:
            row = td[(td["concept_name"] == theme) & (td["trade_date"] == day)]
        # 过了实盘交易窗口且今日无行(kpl T+1未入库) → 用上一交易日 + 标注
        if row.empty and not live:
            hist = td[(td["concept_code"] == theme)
                      | (td["concept_name"] == theme)].sort_values("trade_date")
            if len(hist):
                row = hist.tail(1)
                ctx["data_note"] = ("已过实盘交易窗口, 题材数据为上一交易日"
                                    f"({row.iloc[-1]['trade_date']})")
        if not row.empty:
            r = row.iloc[-1]
            ctx["theme_day"] = {
                "zt_cnt": int(r["zt_cnt"] or 0),
                "zt_all": int(r["zt_all"] or 0),
                "wave_no": int(r["wave_no"] or 0),
                "theme_age": int(r["theme_age"] or 0),
                "max_height": int(r["max_height"] or 0),
                "leader_name": r["leader_name"],
                "leader_height": int(r["leader_height"] or 0)}
            from core.cycle import theme_stage
            ctx["stage"] = theme_stage(int(r["wave_no"] or 0) or None,
                                       int(r["zt_all"] or 0))
    except Exception:
        pass
    today = datetime.now().strftime("%Y%m%d")
    if day == today:                          # 盘口快照仅当日可用(防回放污染)
        latest = _load_json(LIVE / "latest.json") or {}
        ctx["sentiment"] = latest.get("sentiment")
        radar = _load_json(LIVE / "radar.json") or {}
        for t in radar.get("themes") or []:
            if t.get("name") == theme or t.get("concept_code") == theme:
                ctx["radar_theme"] = {"heat": t.get("heat"), "zt": t.get("zt"),
                                      "headx": t.get("headx"),
                                      "nmem": t.get("nmem"),
                                      "top": t.get("top") or []}
                em = t.get("em") or {}
                ctx["realtime"] = {
                    "theme_pct": em.get("pct"), "speed": em.get("speed"),
                    "leader": em.get("leader"),
                    "movers": [{"name": x.get("name"), "pct": x.get("pct")}
                               for x in (t.get("top") or [])[:5]]}
                break
        # 盘后: 实时涨跌幅已是收盘快照(非盘中实时), 标注提醒
        if not live and ctx.get("realtime"):
            ctx["data_note"] = ((ctx.get("data_note") or "")
                                + " · 实时涨跌幅为收盘快照(非盘中实时)").strip(" ·")
        # 实时优先兜底(修口径矛盾): theme.day今日行缺失(kpl T+1未入库)时,
        # radar 实时涨停家数才是真相; 否则 brief 说"不在场/0家"而实时 N家涨停,
        # 裁判看到矛盾数据只能"中性观望"。用实时兜底 theme_day + 阶段。
        rt = ctx.get("radar_theme") or {}
        rtm = ctx.get("realtime") or {}
        zt_live = int(rt.get("zt")
                      or len([m for m in (rtm.get("movers") or [])
                              if (m.get("pct") or 0) >= 9.8]))
        ctx["zt_live"] = zt_live
        if not ctx.get("theme_day") and zt_live:
            wno, age = 1, 1
            try:
                from datastore import load
                td2 = load("theme.day", columns=["trade_date", "concept_code",
                                                 "concept_name", "wave_no",
                                                 "theme_age"])
                hist = td2[(td2["concept_code"] == theme)
                           | (td2["concept_name"] == theme)
                           ].sort_values("trade_date")
                if len(hist):
                    wno = int(hist.iloc[-1]["wave_no"] or 0)
                    age = int(hist.iloc[-1]["theme_age"] or 0)
            except Exception:
                pass
            ctx["theme_day"] = {"zt_all": zt_live, "zt_cnt": zt_live,
                                "zt_cnt_raw": zt_live,
                                "max_height": int(rtm.get("max_height") or 0),
                                "leader_name": rtm.get("leader"),
                                "leader_height": 0, "theme_age": age,
                                "wave_no": wno,
                                "_src": "radar实时兜底(theme.day今日未入库)"}
            from core.cycle import theme_stage
            ctx["stage"] = theme_stage(wno or None, zt_live)
    ctx["basket"] = _build_basket(theme, ctx)
    try:
        from core import theme_kb
        kb = theme_kb.load_kb()
        th = kb.get("theme")
        if th is not None and not th.empty:
            hit = th[(th["name"] == theme) | (th["concept_code"] == theme)]
            if len(hit):
                r = hit.iloc[0]
                ctx["playbook_ref"] = r.get("playbook_ref")
                ctx["driver_type"] = r.get("driver_type")
                ctx["parent_theme"] = r.get("parent_theme")
                ctx["kb_falsification"] = r.get("falsification")
        # 关联概念/细分赛道(KB 相似边): 供产业链分析师挖掘潜在机会
        sim = kb.get("similarity")
        rel = []
        if sim is not None and not sim.empty:
            for r in sim.itertuples():
                if getattr(r, "a_name", None) == theme:
                    rel.append({"name": getattr(r, "b_name", None),
                                "sim": getattr(r, "sim_score", None)})
                elif getattr(r, "b_name", None) == theme:
                    rel.append({"name": getattr(r, "a_name", None),
                                "sim": getattr(r, "sim_score", None)})
        if rel:
            ctx["related_concepts_kb"] = rel[:8]
    except Exception:
        pass
    # 智易(zhiyi)真实数据接地: 仅当日(防回放未来信息); 后端偏慢/flaky
    # 故 best-effort, 失败留空由分析师降级为 LLM+KB。给分析师真实行业
    # 估值/成分/财务, 避免产业链/候选全靠 LLM 编造(用户反馈"分析空洞")。
    if day == today:
        try:
            from core import zhiyi
            ctx["zhiyi"] = zhiyi.ground_theme(theme)
        except Exception:
            pass
    return ctx


# ---------- 事实数据简报(无 LLM 时的降级展示, 不伪造) ----------

def _brief_catalyst(state: AgentState) -> str:
    items = state.get("news_items") or []
    zy = (state.get("ctx") or {}).get("zhiyi") or {}
    mats = zy.get("materials") or []
    news = ("关联消息:\n" + "\n".join(
        f"[{it.get('t','')}] {it.get('title','')} ({it.get('source','')})"
        for it in items[:8])) if items else "无订阅新闻"
    mat = ("\n【题材专属素材(智易)】\n" + "\n".join(mats[:3])) if mats else ""
    return news + mat


def _brief_stage(state: AgentState) -> str:
    ctx = state.get("ctx") or {}
    td = ctx.get("theme_day") or {}
    if not td:
        return f"题材「{state.get('theme')}」当日无 theme.day 记录(不在场)"
    return (f"阶段={ctx.get('stage','无')} 波次={td.get('wave_no')} "
            f"关联家数={td.get('zt_all')} 持续={td.get('theme_age')}天 "
            f"最高板={td.get('max_height')} 龙头={td.get('leader_name')}"
            f"({td.get('leader_height')}板)"
            + (f" 剧本={ctx.get('playbook_ref')}" if ctx.get("playbook_ref")
               else ""))


def _brief_leader(state: AgentState) -> str:
    ctx = state.get("ctx") or {}
    rt = ctx.get("radar_theme") or {}
    basket = ctx.get("basket") or []
    head = (f"热度={rt.get('heat')} 涨停={rt.get('zt')} "
            f"头部超额={rt.get('headx')} 成分={rt.get('nmem')}"
            if rt else "无盘中热度快照(历史日)")
    rows = "\n".join(
        f"  {b.get('rank')}. {b.get('name') or b.get('ts_code')} "
        f"角色={b.get('role') or '-'} 环节={b.get('chain_node') or '-'} "
        f"受益纯度={b.get('benefit_purity')} 四维分={b.get('four_dim_score')}"
        for b in basket[:6])
    zy = ctx.get("zhiyi") or {}
    val = (zy.get("valuation") or "")[:1100]
    fin = (zy.get("financials") or "")[:1100]
    rtm = ctx.get("realtime") or {}
    rt_str = ""
    if rtm:
        mv = "、".join(f"{m.get('name')}({m.get('pct')}%)"
                       for m in (rtm.get("movers") or [])[:5])
        rt_str = (f"\n【实时涨跌幅】题材整体={rtm.get('theme_pct')}% "
                  f"龙头={rtm.get('leader')} 个股: {mv}")
    return head + rt_str + ("\n候选池:\n" + rows if rows else "") \
        + (f"\n智易真实估值(PE/PB/百分位):\n{val}" if val else "") \
        + (f"\n智易真实财务:\n{fin}" if fin else "")


def _brief_sentiment(state: AgentState) -> str:
    ctx = state.get("ctx") or {}
    s = ctx.get("sentiment")
    td = ctx.get("theme_day") or {}
    mkt = (f"【全市场】涨停={s.get('zt_count')} 炸板率={s.get('broken_rate')} "
           f"最高板={s.get('max_height')} 加速={s.get('accel')} "
           f"一字占比={s.get('yizi_proxy')}" if s
           else "【全市场】无快照(历史日/盘前)")
    th = (f"【本题材】涨停关联={td.get('zt_all')}家 独占涨停={td.get('zt_cnt')}家 "
          f"题材最高板={td.get('max_height')}" if td
          else "【本题材】当日不在场(无涨停)")
    note = ctx.get("data_note")
    return (f"⚠数据说明: {note}\n" if note else "") + mkt + "\n" + th


def _brief_chain(state: AgentState) -> str:
    ctx = state.get("ctx") or {}
    basket = ctx.get("basket") or []
    nodes: dict = {}
    for b in basket:
        n = b.get("chain_node") or "未分类"
        nodes[n] = nodes.get(n, 0) + 1
    chain_str = " ".join(f"{k}×{v}" for k, v in nodes.items()) or "无KB环节标注"
    rel = ctx.get("related_concepts_kb") or []
    rel_str = "、".join(str(r.get("name")) for r in rel[:6]) or "无KB相似题材"
    zy = ctx.get("zhiyi") or {}
    secs = "、".join(f"{s.get('name')}({s.get('symbol')})"
                   for s in (zy.get("sectors") or [])[:4])
    comp = (zy.get("component") or "")[:1400]
    return (f"驱动类型={ctx.get('driver_type') or '未知'} "
            f"父题材={ctx.get('parent_theme') or '-'}\n"
            f"KB产业链环节分布: {chain_str}\n"
            f"KB关联题材(细分赛道候选): {rel_str}"
            + (f"\n智易识别板块: {secs}" if secs else "")
            + (f"\n智易真实成分数据(行业占比/收益):\n{comp}" if comp else ""))


_BRIEFS = {"catalyst": _brief_catalyst, "chain": _brief_chain,
           "stage": _brief_stage, "leader": _brief_leader,
           "sentiment": _brief_sentiment}


# ---------- 角色 system prompt(领域化, 简洁) ----------

_PROMPTS = {
    "catalyst": (
        "你是A股题材催化分析师。依据给定新闻/素材, 抽该题材的第一催化/驱动类型"
        "(政策|产业|事件|技术|周期|海外映射)/受益产业链环节/催化强度heat(0-10)。"
        "⚠若给定消息与本题材【无直接关联】(如泛宏观/其它板块消息), 不得强行"
        "附会, 必须 first_catalyst=\"催化不足\" 并在 summary 说明; 有专属催化才正常抽。"
        "每条关键判断做信息分层: 标注【确认:公告/财报/监管】或【推测:券商/调研/"
        "媒体/论坛】并给置信度(低/中/高)。只依据给定信息不编造; 输出JSON {summary,"
        "first_catalyst,driver_type,benefit_nodes,heat,evidence:[{claim,"
        "source_type,confirmed(确认|推测),confidence(低|中|高)}]}。"),
    "chain": (
        "你是A股产业链分析师。把该题材按产业链拆解，参考半导体存储模块(上游设备→上游材料→中游设计→"
        "中游制造/IDM→中游封测→下游模组/主控→下游品牌/分销/终端), 每环节给国产"
        "替代逻辑/海外对标/A股映射候选/业务纯度; 并挖掘该题材能引入的**关联概念/"
        "细分赛道**(潜在机会, 不止当日强势股)。❗若提供了「智易识别板块/真实成分"
        "数据」, summary 必须引用其中的板块代码与成分行业占比/收益等具体数字, "
        "不得泛泛而谈; 无真实数据的标的代码不编造。❗每个A股候选须区分来源: 来自"
        "智易真实成分数据的为真实, 你基于产业链知识推断的标注为「推断」; 若智易未"
        "提供成分数据, summary 首句须声明「候选为产业链推断、未经真实成分验证」, "
        "不得把推断候选伪装成真实成分。输出JSON {summary,chain:"
        "[{node,logic,overseas,candidates,purity}],related_concepts:[{name,"
        "relation,opportunity}]}。"),
    "stage": (
        "你是A股题材阶段分析师。依据波次/关联家数/持续天数/龙头高度, 判断该题材"
        "当前处于爆发/主升/鱼尾的语义位置及延续性。阶段锚点是波次(第几波)不是"
        "日龄。输出JSON {summary,sustainability(0-1),turning_factor,"
        "persistence_factor}。summary为2-3句中文解读。"),
    "leader": (
        "你是A股龙头资金分析师。选股候选须综合【产业链分析师候选】+【智易真实"
        "成分】+【盘中涨停龙头】, 不能只看盘中涨停池。判断龙头地位与梯队(龙头/"
        "中军/补涨)与资金主攻方向。⚠若候选池稀疏或四维数据缺失, 明确说\"候选池"
        "不足\", 严禁凭印象指定龙头。❗若提供「智易真实估值」(PE/PB/百分位), "
        "summary 须引用具体数字判断板块估值位置(注明板块级)。输出JSON {summary,"
        "leader,ladder,main_direction,valuation_view}。"),
    "sentiment": (
        "你是A股情绪分析师。⚠注入的涨停家数/炸板率/最高板/加速是【全市场口径】, "
        "不是本题材数据 —— 严禁表述为\"本板块涨停N家\"; 本题材涨停家数以【题材"
        "专属情绪】(若提供)为准。你的职责是判断当前【大盘情绪环境】对本题材打板"
        "的顺逆(全市场分歧/一致)。输出JSON {summary,mood,risk(0-1),market_zt}。"
        "summary为2-3句中文解读。"),
}
_BULL = ("你是看多研究员。基于四份分析报告, 为该题材能延续/候选池能封板建立"
         "最强论证, 反驳看空担忧。输出JSON {argument}(中文, 3-5句)。")
_BEAR = ("你是看空研究员。基于四份分析报告, 论证该题材的风险(高位分歧/补涨"
         "陷阱/退潮), 反驳看多。输出JSON {argument}(中文, 3-5句)。")
_JUDGE = ("你是裁判(投资组合经理)。批判性评估看多/看空辩论, 合成该题材的最终结论"
          "与因子。⚠verdict 判据: 看多=有真实证据表明启动/延续(实时涨停梯队+"
          "涨停扩容+专属催化); 看空=有真实证据表明退潮/风险(龙头断板+涨停萎缩+"
          "炸板恶化); 中性=实时也无任何涨停/催化证据。【关键: 实时关联涨停家数"
          "(zt_live)与涨停梯队(个股涨停)是核心证据 —— 若实时涨停≥3家且有涨停梯队, "
          "这是启动/活跃的真实证据, 必须据此给方向性结论, 严禁因'离线theme.day缺失'"
          "就中性观望; 若数据口径冲突(如离线0家 vs 实时N家), 一律以实时为准并说明】。"
          "⚠结论必须明确、引用具体数字证据(实时涨停N家/涨停梯队/催化/估值百分位), "
          "不得模糊两可或罗列矛盾数据。strength(0-10)须反映实时涨停家数与梯队"
          "(≥5家有梯队→≥6; 1-2家→≤4)。全市场情绪只作环境参考。每个结论给可证伪"
          "信号。portfolio 用4+4+2风险框架非收益承诺, 不给买卖点。输出JSON {verdict"
          "(看多|看空|中性),strength(0-10),sustainability(0-1),confidence(0-1),"
          "conclusion(3-5句),falsification:[失效信号],calendar:[关键日历],"
          "counter_arguments:[3条最易忽略的反对意见],portfolio:{core,satellite,"
          "cash}}。")


# ---------- 节点 ----------

def _mk_analyst(key: str):
    brief = _BRIEFS[key]
    system = _PROMPTS[key]

    def node(state: AgentState) -> dict:
        data = brief(state)
        text, src, res = data, "rule_baseline", None
        if llm.available():
            user = (f"题材: {state.get('theme')}\n当日: {state.get('day')}\n"
                    f"事实数据:\n{data}")
            res = llm.chat_json(system, user, model=state.get("model"))
            if res and (res.get("summary") or res.get("argument")):
                text = res.get("summary") or res.get("argument")
                src = "llm"
        structs = dict(state.get("structs") or {})
        if res:
            structs[key] = res            # 结构化JSON供裁判合成因子
        return {f"{key}_report": text, "structs": structs}

    return node


def _mk_debater(side: str):
    system = _BULL if side == "bull" else _BEAR
    label = ROLE_NAMES[side]

    def node(state: AgentState) -> dict:
        d = state.get("debate") or {"history": "", "count": 0,
                                    "bull": "", "bear": ""}
        reports = "\n\n".join(
            f"【{ROLE_NAMES[k]}】{state.get(f'{k}_report','')}"
            for k in ANALYSTS)
        arg, src = None, "rule_baseline"
        if llm.available():
            user = (f"题材: {state.get('theme')}\n四份分析报告:\n{reports}\n"
                    f"辩论历史:\n{d.get('history','')}")
            res = llm.chat_json(system, user, model=state.get("model"))
            if res and res.get("argument"):
                arg, src = res["argument"], "llm"
        if not arg:                           # 降级: 立场化事实陈述(不伪造)
            stage = (state.get("ctx") or {}).get("stage", "无")
            arg = (f"({label}·规则基线) 题材阶段={stage}; "
                   + ("倾向延续" if side == "bull" else "警惕退潮"))
        newd = {"history": d.get("history", "") + f"\n{label}: {arg}",
                "count": d.get("count", 0) + 1,
                "bull": d.get("bull", "") + (f"\n{arg}" if side == "bull" else ""),
                "bear": d.get("bear", "") + (f"\n{arg}" if side == "bear" else ""),
                "current_response": arg, "last_side": side, "src": src}
        return {"debate": newd}

    return node


def _rule_factor(ctx: dict) -> dict:
    """无 LLM 时的确定性因子(不伪造 LLM 结论)。四维篮子/关联概念/可证伪
    来自 KB(确定性); 强度/延续/表态用规则代理。"""
    stage = ctx.get("stage") or "无"
    td = ctx.get("theme_day") or {}
    rt = ctx.get("radar_theme") or {}
    zt_all = td.get("zt_all") or rt.get("zt") or 0
    heat = rt.get("heat") or 0
    strength = round(min(10.0, zt_all / 2.0 + heat / 20.0), 1)
    sust = {"爆发": 0.65, "主升": 0.45, "鱼尾": 0.30, "无": 0.2}.get(stage, 0.4)
    verdict = "看多" if (stage == "爆发" and zt_all >= 3) else (
        "看空" if stage == "鱼尾" else "中性")
    rel = [{"name": r.get("name"), "relation": "KB相似", "opportunity": None}
           for r in (ctx.get("related_concepts_kb") or [])]
    return {"theme_stage": stage, "strength": strength,
            "sustainability": sust, "verdict": verdict, "confidence": 0.5,
            "driver_type": ctx.get("driver_type"), "first_catalyst": None,
            "industry_chain": [], "related_concepts": rel,
            "basket": ctx.get("basket") or [], "evidence": [],
            "falsification": ([ctx.get("kb_falsification")]
                              if ctx.get("kb_falsification") else []),
            "calendar": [], "counter_arguments": [],
            "portfolio": {"core": "50-60%", "satellite": "20-30%",
                          "cash": "10-20%"}}


def _as_list(v):
    """LLM 有时把列表字段返回成字符串/对象(实测 falsification 会写成串),
    统一归一为 list, 保证因子 schema 稳定且前端 .map 不报错。"""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _split_names(cands) -> list:
    """产业链候选(字符串/列表)拆成标的名列表(取中文名, 去括号代码/说明)。"""
    if isinstance(cands, list):
        out = []
        for c in cands:
            out.extend(_split_names(c))
        return out
    if not isinstance(cands, str):
        return []
    import re
    names = []
    for p in re.split(r"[、,，/;；]", cands):
        m = re.match(r"^([\u4e00-\u9fa5A-Za-z*]{2,})", p.strip())
        if m:
            names.append(m.group(1))
    return names


def _norm_candidates(cands) -> list:
    """候选归一为 [{name,source,purity}]: 兼容 字符串/字符串列表/字典列表
    (P3 提示词让 LLM 输出 candidates 为带 source 标注的字典列表)。"""
    out = []
    if isinstance(cands, str):
        for nm in _split_names(cands):
            out.append({"name": nm, "source": "推断", "purity": None})
    elif isinstance(cands, list):
        for c in cands:
            if isinstance(c, dict):
                nm = c.get("name")
                if nm:
                    out.append({"name": str(nm).strip(),
                                "source": c.get("source"),
                                "purity": c.get("purity")})
            elif isinstance(c, str):
                for nm in _split_names(c):
                    out.append({"name": nm, "source": "推断", "purity": None})
    return out


def _merge_chain_into_basket(basket: list, chain: list, cap: int = 18) -> list:
    """修 P2: 把产业链各环节候选并入选股池(否则产业链候选与选股池脱节)。
    兼容 candidates 为字符串或带source标注的字典列表; 按名去重; 设上限防膨胀。"""
    out = list(basket or [])
    seen = {(b.get("name") or b.get("ts_code")) for b in out}
    rank = len(out)
    for node in (chain or []):
        if not isinstance(node, dict):
            continue
        for cand in _norm_candidates(node.get("candidates")):
            nm = cand.get("name")
            if not nm or nm in seen or rank >= cap:
                continue
            seen.add(nm)
            rank += 1
            out.append({"name": nm, "ts_code": None,
                        "chain_node": node.get("node"), "role": None,
                        "moat": None, "realize_cycle": None,
                        "competition": None, "ceiling": None,
                        "benefit_purity": cand.get("purity"),
                        "four_dim_score": None,
                        "why": f"产业链分析师候选({cand.get('source') or '推断'})",
                        "tier": "satellite", "rank": rank})
    return out


def _judge_node(state: AgentState) -> dict:
    ctx = state.get("ctx") or {}
    d = state.get("debate") or {}
    structs = state.get("structs") or {}
    reports = "\n\n".join(f"【{ROLE_NAMES[k]}】{state.get(f'{k}_report','')}"
                          for k in ANALYSTS)
    base = _rule_factor(ctx)      # theme_stage/basket/关联概念/可证伪 恒确定性
    # 产业链/关联概念(产业链分析师) + 证据/第一催化(催化分析师)
    ch = structs.get("chain") or {}
    cat = structs.get("catalyst") or {}
    if ch.get("chain"):
        base["industry_chain"] = _as_list(ch["chain"])
    if ch.get("related_concepts"):
        base["related_concepts"] = _as_list(ch["related_concepts"])
    if cat.get("evidence"):
        base["evidence"] = _as_list(cat["evidence"])
    base["basket"] = _merge_chain_into_basket(base["basket"],
                                              base["industry_chain"])
    base["first_catalyst"] = cat.get("first_catalyst") or base["first_catalyst"]
    base["driver_type"] = cat.get("driver_type") or base["driver_type"]
    base["realtime"] = ctx.get("realtime")   # 实时涨跌幅(题材整体+个股)
    base["data_note"] = ctx.get("data_note")   # 数据说明(盘后/上一交易日标注)
    conclusion, src, model = "", "rule_baseline", None
    if llm.available():
        note = (state.get("ctx") or {}).get("data_note")
        user = (f"题材: {state.get('theme')}\n"
                + (f"⚠数据说明(须在结论中体现): {note}\n" if note else "")
                + f"各角色分析报告:\n{reports}\n辩论历史:\n{d.get('history','')}")
        res = llm.chat_json(_JUDGE, user, model=state.get("model"))
        if res and res.get("conclusion"):
            conclusion = res["conclusion"]
            base["strength"] = max(0.0, min(10.0, float(res.get("strength")
                                                        or base["strength"])))
            base["sustainability"] = max(0.0, min(1.0, float(
                res.get("sustainability") or base["sustainability"])))
            base["verdict"] = res.get("verdict") if res.get("verdict") in (
                "看多", "看空", "中性") else base["verdict"]
            base["confidence"] = max(0.0, min(1.0, float(
                res.get("confidence") or base["confidence"])))
            base["falsification"] = (_as_list(res.get("falsification"))
                                     or base["falsification"])
            base["calendar"] = _as_list(res.get("calendar"))
            base["counter_arguments"] = _as_list(res.get("counter_arguments"))
            if isinstance(res.get("portfolio"), dict):
                base["portfolio"] = res["portfolio"]
            src, model = "llm", llm.model_name(state.get("model"))
    if not conclusion:
        conclusion = (f"(规则基线) 题材「{state.get('theme')}」阶段="
                      f"{base['theme_stage']} 强度={base['strength']} "
                      f"延续={base['sustainability']} → {base['verdict']}")
    base["model"] = model
    return {"conclusion": conclusion, "factor": base, "src": src}


def _should_continue_debate(state: AgentState) -> str:
    d = state.get("debate") or {}
    if d.get("count", 0) >= 2 * MAX_DEBATE_ROUNDS:
        return "judge"
    return "bear" if d.get("last_side") == "bull" else "bull"


# ---------- 图构建(缓存一份编译结果) ----------

_APP = None


def build_graph():
    global _APP
    if _APP is not None:
        return _APP
    g = StateGraph(AgentState)
    for k in ANALYSTS:
        g.add_node(k, _mk_analyst(k))
    g.add_node("bull", _mk_debater("bull"))
    g.add_node("bear", _mk_debater("bear"))
    g.add_node("judge", _judge_node)
    g.add_edge(START, "catalyst")
    g.add_edge("catalyst", "chain")
    g.add_edge("chain", "stage")
    g.add_edge("stage", "leader")
    g.add_edge("leader", "sentiment")
    g.add_edge("sentiment", "bull")
    g.add_conditional_edges("bull", _should_continue_debate,
                            {"bear": "bear", "judge": "judge"})
    g.add_conditional_edges("bear", _should_continue_debate,
                            {"bull": "bull", "judge": "judge"})
    g.add_edge("judge", END)
    _APP = g.compile()
    return _APP


# ---------- 驱动 + 事件流 ----------

def run_graph(theme: str, day: str | None = None,
              cutoff: float | None = None, news_items: list | None = None,
              emit_cb: Callable[[dict], None] | None = None,
              model: str | None = None) -> dict:
    """跑一次题材智能体分析; 逐节点经 emit_cb 推事件(供 SSE 可视化)。

    emit_cb 事件: {seq,type,role,theme,content,ts,extra}
      type = analyzing_news | role_speak | debate | conclusion
    返回最终 state(含 conclusion/factor/debate/各 report)。生产者据 factor
    落成可回放 feed; 手动点击路径只展示不落盘。"""
    day = day or datetime.now().strftime("%Y%m%d")
    cutoff = cutoff if cutoff is not None else time.time()
    if news_items is None:
        try:
            from core import newsfeed
            news_items = newsfeed.read_subscribed(
                [s["id"] for s in newsfeed.enabled_sources()], day, cutoff,
                limit=20)
        except Exception:
            news_items = []
    ctx = gather_ctx(theme, day, cutoff)
    seq = 0

    def emit(type_: str, role: str | None, content: str, extra: dict | None = None):
        nonlocal seq
        seq += 1
        if emit_cb:
            emit_cb({"seq": seq, "type": type_, "role": role, "theme": theme,
                     "content": content, "ts": time.time(),
                     "extra": extra or {}})

    emit("analyzing_news", None,
         f"分析 {len(news_items)} 条消息 → 题材「{theme}」",
         {"news": [{"t": it.get("t"), "title": it.get("title"),
                    "source": it.get("source")} for it in (news_items or [])[:10]],
          "llm": llm.available()})

    init: AgentState = {"theme": theme, "day": day, "cutoff": cutoff,
                        "model": model, "news_items": news_items or [],
                        "ctx": ctx, "structs": {},
                        "debate": {"history": "", "count": 0,
                                   "bull": "", "bear": ""}}
    app = build_graph()
    final = dict(init)
    for chunk in app.stream(init, stream_mode="updates"):
        for node, upd in chunk.items():
            final.update(upd)
            if node in ANALYSTS:
                emit("role_speak", ROLE_NAMES[node], final.get(f"{node}_report", ""))
            elif node in ("bull", "bear"):
                dd = upd.get("debate") or {}
                emit("debate", ROLE_NAMES[node], dd.get("current_response", ""))
            elif node == "judge":
                emit("conclusion", ROLE_NAMES["judge"],
                     final.get("conclusion", ""), {"factor": final.get("factor")})
    final.setdefault("factor", _rule_factor(ctx))
    final.setdefault("src", "rule_baseline")
    return final


# ---------- 运行态文件(跨进程: 生产者写, server.py SSE 读) ----------

RUN_STATE_PATH = LIVE / "ai_agent_run.json"
MAX_TRACE_EVENTS = 80          # 运行态只留最近N条事件(防单文件膨胀)


class RunTrace:
    """运行态写入器 —— 作为 run_graph 的 emit_cb 用。

    每个题材一次 run: 构造即重置文件(new run_id, events=[]), 每次 emit
    追加事件并重写。server.py 的 SSE 端点轮询本文件 mtime, 变化即把
    快照推给前端(角色卡/辩论/结论逐步填充)。跨进程安全(靠文件不靠内存)。"""

    def __init__(self, theme: str, day: str):
        self.run_id = f"{day}-{theme}-{int(time.time())}"
        self.theme, self.day = theme, day
        self.events: list = []
        self.seq = 0
        self.roles: dict = {}
        self.debate: list = []
        self.conclusion = ""
        self.factor: dict | None = None
        self.src = ""
        self._write()

    def __call__(self, ev: dict) -> None:      # emit_cb
        self.seq = ev.get("seq", self.seq + 1)
        self.events.append(ev)
        if len(self.events) > MAX_TRACE_EVENTS:
            self.events = self.events[-MAX_TRACE_EVENTS:]
        t = ev.get("type")
        if t == "role_speak":
            self.roles[ev.get("role")] = ev.get("content")
        elif t == "debate":
            self.debate.append({"role": ev.get("role"),
                                "content": ev.get("content")})
        elif t == "conclusion":
            self.conclusion = ev.get("content")
            self.factor = (ev.get("extra") or {}).get("factor")
        self._write()

    def finish(self, res: dict) -> None:
        self.conclusion = res.get("conclusion", "")
        self.factor = res.get("factor")
        self.src = res.get("src", "")
        self._write()

    def _write(self) -> None:
        try:
            RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            RUN_STATE_PATH.write_text(json.dumps({
                "run_id": self.run_id, "theme": self.theme, "day": self.day,
                "seq": self.seq, "events": self.events, "roles": self.roles,
                "debate": self.debate, "conclusion": self.conclusion,
                "factor": self.factor, "src": self.src,
                "updated": time.time()}, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass


def read_run_state() -> dict | None:
    """读当前运行态快照(server.py SSE / 页面加载用); 缺失返回 None。"""
    return _load_json(RUN_STATE_PATH)
