# -*- coding: utf-8 -*-
"""ticai-daban 全局配置"""
import os
import re
from pathlib import Path

import tushare as ts

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# 研究窗口起点（limit_list_d 实测起点 2019-11-28）
START_DATE = "20191128"

# 概念过滤: 成分数上限（超过视为杂烩概念, 如融资融券3858只）
MAX_MEMBER_COUNT = 1500

# 概念过滤: 名称关键词黑名单（指数样本类 + 属性类，非题材）
CONCEPT_NOISE_KEYWORDS = [
    # 指数样本/风格池
    "样本股", "成份股", "成分股", "MSCI", "富时", "标普", "沪股通", "深股通",
    "AH股", "B股", "百元股", "高价股", "低价股", "破净", "破发",
    # 属性/事件类（非题材炒作）
    "融资融券", "转融券", "注册制", "股权激励", "增持", "回购", "减持",
    "股权转让", "重组", "预增", "预亏", "预减", "扭亏", "业绩预告",
    "ST板块", "退市", "次新股", "新股", "解禁", "质押", "商誉",
]


def _load_dotenv():
    """轻量.env加载：项目根目录存在.env时，将其中未设置的环境变量注入"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def get_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN")
    if token and token != "your_token_here":
        return token
    m = None
    zshrc = Path(os.path.expanduser("~/.zshrc"))
    if zshrc.exists():
        m = re.search(r"TUSHARE_TOKEN=(\S+)", zshrc.read_text())
    token = m.group(1) if m else None
    if not token:
        raise RuntimeError(
            "TUSHARE_TOKEN not found: 请设置环境变量或在项目根目录创建.env"
            "（参考.env.example）")
    return token


def get_pro():
    ts.set_token(get_token())
    return ts.pro_api()
