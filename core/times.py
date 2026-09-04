# -*- coding: utf-8 -*-
"""封板时间口径 — HHMMSS 规范化唯一出处

背景(pandas 降级踩坑, 见 docs/adr/0005):
  `s.astype(str).str.zfill(6)` 对缺失值的行为在 pandas 版本间不一致:
    pandas 3.0 专用 str dtype: astype(str) 保留 NA → zfill 后仍是 NA
    pandas 2.x object dtype : None → '00None', NaN → '000nan'
  后果有两类:
    ① 写盘污染 — '000nan' 被当成合法时间存入 parquet, 后续无法识别为缺失
    ② 比较失真 — apps/poller.py 用 `first_time <= '094500'` 统计早封板占比,
       '00None' <= '094500' 按字符串比较为 True → 缺失记录被误算成早封板
  tushare limit_list_d 实测缺失率: first_time 10.7%(8/75),
  last_time 20%(15/75), 故这不是理论风险。

本模块用显式掩码把缺失值统一还原为 NA, 使两个 pandas 版本行为一致。
所有涉及封板时间(first_time/last_time)的规范化与比较都必须走本模块。
"""
import pandas as pd

EMPTY_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}


def hhmmss6(s) -> pd.Series:
    """封板时间 → 6位 HHMMSS 字符串; 缺失/空值保持 NA。

    版本无关实现: astype(str) 在各版本对 NA 的处理不同('nan'/'None'/NA),
    故先算掩码再 where 回填 NA, 输出与 pandas 版本无关。
    输入可为 int(93006) / str('93006') / None / NaN。
    """
    s = s if isinstance(s, pd.Series) else pd.Series(s)
    txt = s.astype(str).str.strip()
    # 掩码: 原值非缺失 且 转文本后不是各版本 NA 的字面表述
    mask = s.notna() & ~txt.str.lower().isin(EMPTY_TOKENS)
    return txt.str.zfill(6).where(mask)


def is_before(s, cutoff: str) -> pd.Series:
    """封板时间是否早于 cutoff(HHMMSS 字符串比较)。
    缺失一律判 False(而非 NA) — 供占比统计用, 避免 .mean() 口径随版本漂移。
    语义: 时间缺失说明该股未封板, 不应计入"早封板"。"""
    v = hhmmss6(s)
    return v.notna() & (v.fillna("999999") <= cutoff)


def is_after(s, cutoff: str) -> pd.Series:
    """封板时间是否晚于 cutoff; 缺失判 False"""
    v = hhmmss6(s)
    return v.notna() & (v.fillna("000000") >= cutoff)
