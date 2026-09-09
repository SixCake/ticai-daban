# -*- coding: utf-8 -*-
"""智易(zhiyi)金融数据 MCP 技能客户端 — 为题材智能体提供真实数据接地

背景: 题材智能体的分析师若只靠 LLM 通用知识, 产业链/候选/估值都是"编"的,
分析空洞(用户反馈)。本模块封装 ~/Downloads/zhiyi 下的 MCP 技能(JSON-RPC
over HTTP, SSE 响应), 让分析师吃**真实数据**: 行业估值/财务/成分/表现/情绪
+ 投研线索。技能清单见各 SKILL.md, endpoint 均为 bigoal.alipay.com。

调用契约: POST endpoint, body=jsonrpc tools/call; 响应 SSE, 结果在最后一个
`data:` 事件的 result.content[].text(内层通常还是 JSON 串, NATURAL 格式下
真实数据在 data.text)。

健壮性: 无网络/超时/500/解析失败 → 返回空(不伪造), 由调用方降级为 LLM+KB。
部分板块代码对某些视图会 500(实测), 故每个视图独立 best-effort。
"""
from __future__ import annotations

import json

import requests

BASE = "https://bigoal.alipay.com/wukongprod/mcp/public"
ENTITY = f"{BASE}/entity/relation/mcp/"
PLATE = f"{BASE}/plate/mcp/"
MATERIAL = f"{BASE}/material/index/mcp/"
STOCK = f"{BASE}/stock/mcp/"

HEADERS = {"Content-Type": "application/json",
           "Accept": "application/json, text/event-stream"}
TIMEOUT = 15          # 单次调用上限(秒); 后端偏慢, 超时就降级不阻塞

# 视图 → plate 工具名(用专用工具比 industry_view_query 合并调用更稳, 实测)
_PLATE_TOOL = {
    "valuation": "industry_valuation_query_tool",
    "financials": "industry_financial_data_query_tool",
    "performance": "industry_market_performance_query_tool",
    "component": "industry_component_query_tool",
    "sentiment": "industry_market_sentiment_query_tool",
    "risk": "industry_market_risk_query_tool",
}

_SEC_CACHE: dict = {}          # (theme,type) → 识别结果, 题材→代码稳定可缓存


def _mcp(endpoint: str, tool: str, args: dict, timeout: int = TIMEOUT) -> list:
    """调一个 MCP 工具, 解析 SSE, 返回 content[].text 解析后的对象列表。
    任何异常/无结果 → [](不抛, 不伪造)。"""
    body = json.dumps({"jsonrpc": "2.0", "method": "tools/call",
                       "params": {"name": tool, "arguments": args}, "id": 1})
    try:
        r = requests.post(endpoint, headers=HEADERS, data=body, timeout=timeout)
        r.raise_for_status()
        r.encoding = "utf-8"       # SSE 未声明 charset, 不强制会按 Latin-1 解→中文乱码
    except Exception as e:
        print(f"[zhiyi] {tool} 调用失败: {type(e).__name__}: {e}")
        return []
    out = []
    for line in r.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except Exception:
            continue
        res = obj.get("result")
        if not res:
            continue
        for c in res.get("content") or []:
            t = c.get("text")
            if not t:
                continue
            try:
                out.append(json.loads(t))
            except Exception:
                out.append({"text": t})     # 非JSON(如500错误串), 原样留
    return out


def _nat(objs: list) -> str:
    """从 NATURAL 响应抽 data.text(真实数据文本); 跳过 500 错误串。"""
    parts = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        d = o.get("data")
        if isinstance(d, dict) and isinstance(d.get("text"), str):
            parts.append(d["text"])
    return "\n".join(parts)


def _name_related(theme: str, name) -> bool:
    """识别出的板块名是否与题材相关(防智易模糊匹配错配)。
    实测: 无对应板块时智易会返回无关热门概念(代糖概念→DeepSeek概念),
    不过滤会把错配板块的估值/成分当成本题材数据。"""
    if not name:
        return False
    t = str(theme).replace("概念", "").strip()
    n = str(name).replace("概念", "").strip()
    if not t or not n:
        return False
    if t in n or n in t:
        return True
    sa, sb = set(t), set(n)
    return len(sa & sb) / max(1, len(sa | sb)) >= 0.34


def recognize(theme: str, type: str = "行业板块") -> list:
    """题材/实体中文名 → 代码列表 [{symbol,name,abbr}]。带缓存。
    只保留名称与题材相关的识别结果(防错配); 全不相关则返回空。"""
    key = (theme, type)
    if key in _SEC_CACHE:
        return _SEC_CACHE[key]
    objs = _mcp(ENTITY, "financial_entity_recognition_tool",
                {"query": theme, "type": type})
    out = [{"symbol": o.get("symbol"), "name": o.get("name"),
            "abbr": o.get("abbr_name")}
           for o in objs if isinstance(o, dict) and o.get("symbol")]
    out = [e for e in out
           if _name_related(theme, e.get("name") or e.get("abbr"))]
    _SEC_CACHE[key] = out
    return out


def plate_view(codes: list, view: str, limit: int = 2500) -> str:
    """取板块某视图的真实数据文本(估值/财务/成分/表现/情绪/风险)。"""
    tool = _PLATE_TOOL.get(view)
    if not tool or not codes:
        return ""
    return _nat(_mcp(PLATE, tool, {"fetch_data_type": "NATURAL",
                                   "entity_codes": codes}))[:limit]


def research_clue(theme: str, limit: int = 4) -> list:
    """投研线索(专业投资线索报告); agent类接口偏慢(实测>100s), 故短超时
    best-effort, 无则空。"""
    objs = _mcp(MATERIAL, "investment_research_clue_query_tool",
                {"query": f"{theme}板块近三天投研线索",
                 "clue_type": "financial_clue"}, timeout=20)
    out = []
    for o in objs[:limit]:
        if isinstance(o, dict):
            t = (o.get("data") or {}).get("text") if isinstance(
                o.get("data"), dict) else o.get("text")
            if t:
                out.append(str(t)[:800])
    return out


def topic_material(theme: str, limit: int = 3) -> list:
    """题材专属素材(专题原料检索); 比泛快讯更贴题材, 供催化分析师。
    不依赖板块代码(查询式检索), best-effort。"""
    objs = _mcp(MATERIAL, "topic_material_search_tool",
                {"query": f"{theme}板块近期专题"}, timeout=20)
    out = []
    for o in objs[:limit]:
        if isinstance(o, dict):
            t = (o.get("data") or {}).get("text") if isinstance(
                o.get("data"), dict) else o.get("text")
            if t:
                out.append(str(t)[:600])
    return out


def ground_theme(theme: str,
                 views=("valuation", "component", "financials"),
                 with_clue: bool = False, with_material: bool = True) -> dict:
    """为一个题材抓取真实数据接地(供分析师注入)。

    流程: 识别板块代码 → 取最匹配代码的各视图真实数据 → 投研线索。
    每步 best-effort, 失败留空。为控延迟默认只取 3 个视图 + 用最匹配1个代码。
    返回 {theme, sectors, valuation, financials, performance, component,
          sentiment, risk, clues}。"""
    res = {"theme": theme, "sectors": [], "valuation": "", "financials": "",
           "performance": "", "component": "", "sentiment": "", "risk": "",
           "clues": [], "materials": []}
    secs = recognize(theme)
    res["sectors"] = secs[:6]
    if with_material:
        res["materials"] = topic_material(theme)   # 题材素材不依赖板块代码
    codes = [s["symbol"] for s in secs[:1]]      # 最匹配的第一个板块代码
    if not codes:
        return res
    for v in views:
        res[v] = plate_view(codes, v)
    if with_clue:
        res["clues"] = research_clue(theme)
    return res
