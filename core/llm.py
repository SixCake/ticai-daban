# -*- coding: utf-8 -*-
"""LLM 客户端 — OpenAI 兼容接口(env 配置, 无密钥则优雅降级)

设计约束(ADR-0003): LLM 只在生产者(apps/ai_feed.py)里调用, 产出落盘 feed,
策略不直接调 LLM。本模块只做"调用 + 结构化解析", 不含任何策略/信号逻辑。

env 配置(.env 或环境变量, 见 .env.example):
  LLM_API_KEY   密钥(缺失则 available()=False, 生产者跳过, 不伪造)
  LLM_BASE_URL  默认 https://api.openai.com/v1 (可换国产/本地兼容端点)
  LLM_MODEL     默认 gpt-4o-mini

绝不硬编码密钥。绝不在无密钥/调用失败时伪造分析结果(返回 None 由调用方跳过)。
"""
import json
import os

import requests

DEFAULT_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
TIMEOUT = 120


def _cfg() -> dict:
    return {
        "key": os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "base": (os.environ.get("LLM_BASE_URL") or DEFAULT_BASE).rstrip("/"),
        "model": os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
    }


def available() -> bool:
    """有密钥才可用; 无密钥时生产者应跳过(不伪造分析)"""
    c = _cfg()
    return bool(c["key"] and c["key"] not in ("your_key_here", ""))


def model_name(override: str | None = None) -> str:
    """实际使用的模型名(配置级 override 优先于 env)"""
    return override or _cfg()["model"]


def chat_json(system: str, user: str, model: str | None = None,
              temperature: float = 0.2) -> dict | None:
    """调用 LLM 并要求 JSON 输出; 无密钥/失败/解析失败返回 None。

    用 response_format=json_object 约束输出; 解析时容忍 ```json 包裹与
    前后杂讯(回退抽取首个 {...})。调用方据此跳过本轮产出(不伪造)。
    """
    if not available():
        return None
    c = _cfg()
    url = f"{c['base']}/chat/completions"
    payload = {
        "model": model or c["model"],
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    try:
        r = requests.post(url,
                          headers={"Authorization": f"Bearer {c['key']}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[llm] 调用失败: {type(e).__name__}: {e}")
        return None
    return _parse_json(content)


def _parse_json(content: str) -> dict | None:
    """解析 LLM 文本为 dict; 容忍 ```json 包裹与前后杂讯"""
    if not content:
        return None
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None
