# -*- coding: utf-8 -*-
"""涨停股角色体系（唯一出处）— 研究04/04b口径

角色: 龙头 / 连板(非龙头连板≥2) / 共振(wave首日首板) / 补涨(wave age≥2首次出现)
  龙头  T+1 +1.95%（大热点内 +3.87% vs 跟风 +2.52%）
  连板  全角色最强 +2.74%/笔
  共振  +1.91%
  补涨  +2.07%, 不随wave年龄衰减
中军为描述标签(负alpha), 不在本模块判定。

补涨回看窗口: 当日起向前 age-1 个交易日（严格不含当日）内，该股是否已以
概念k出现过归属记录。dates 是否已含当日均可——dpos 用 bisect 定位到
"最后一个早于当日的日期"，盘中(poller)与离线(review)口径一致。
"""
import bisect
from dataclasses import dataclass, field


@dataclass
class RoleContext:
    """单日角色判定上下文（构造一次，全池复用）

    leader_by: {concept_code: 当日龙头ts_code}
    age_by:    {concept_code: 题材连续活跃天数(wave age)}
    att_set:   {(trade_date, ts_code, concept_code)} 历史归属集合
    dates:     升序交易日列表（att_set覆盖的日期）
    date:      当前判定日
    """
    leader_by: dict
    age_by: dict
    att_set: set
    dates: list
    date: str
    dpos: int = field(init=False)

    def __post_init__(self):
        # 最后一个严格早于当日的日期下标（当日在dates中与否均正确）
        self.dpos = bisect.bisect_left(self.dates, self.date) - 1


def roles_of(ctx: RoleContext, code: str, k, h: int) -> list[str]:
    """返回该涨停股在概念k下的角色列表；k为None/UNASSIGNED时返回空"""
    if not isinstance(k, str) or k == "UNASSIGNED":
        return []
    roles = []
    age = ctx.age_by.get(k, 1)
    if ctx.leader_by.get(k) == code:
        roles.append("龙头")
    elif h >= 2:
        roles.append("连板")
    if h == 1 and age == 1:
        roles.append("共振")
    if h <= 2 and age >= 2 and ctx.dpos >= 0:
        appeared = any((ctx.dates[ctx.dpos - i], code, k) in ctx.att_set
                       for i in range(min(age - 1, ctx.dpos + 1)))
        if not appeared:
            roles.append("补涨")
    return roles
