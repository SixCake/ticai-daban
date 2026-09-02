#!/usr/bin/env python
# coding: utf-8

"""
JoinQuant sector-signal-v3 high-recognition leader second-wave research.

signal3 candidate iteration S3-4.0 (2026-08-25):
- 与 signal2 平行，不修改 signal2 的模型、报告和每日样本；
- 研究第一波辨识度、死亡测试、承接、主动修复、二波候选/确认和逻辑失效；
- 六道门采用通过/条件/阻断/数据不足，阻断不能被影子总分抵消；
- 支持单日或显式 START_DATE/END_DATE，状态只使用各自 T 日已知字段；
- 拆分历史行情与信号日有效覆盖率，显式记录第一波峰值/死亡测试日期；
- 区间运行按研究交易日审计候选连续性、合法状态迁移和重新入池间隔；
- 信号日停牌、无成交和数据不完整分开审计，候选必须显式通过可交易数据门；
- 点时枚举申万行业与聚宽概念，区分稳定题材锚与当日最强战术共振标签；
- 稳定锚初始优先申万行业，概念需连续两日显著胜出才允许切换；
- 定价权连续性只跟稳定锚，战术标签与其他点时标签均不改写六道门与状态机；
- future_label_daily 在全部 T 日状态终检后独立采集 T+1、3/5/10 日影子结果；
- 未来标签区分成熟、部分成熟、待成熟和数据不完整，不回写历史状态、六道门或题材锚；
- 当前产物只允许 candidate，未在聚宽 full 路径验证前不得冻结为 ready。

运行环境：聚宽研究 Notebook，不是策略回测编辑器，不自动下单。
"""

from jqdata import *  # 聚宽研究环境提供。
import builtins as _py_builtins
import datetime as dt
import hashlib
import html
import math
import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd


# jqdata 星号导入后恢复常用 Python 内置函数，避免平台私有对象同名覆盖。
_PY_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "next", "print", "range",
    "round", "set", "sorted", "str", "sum", "tuple", "zip",
)
for _builtin_name in _PY_BUILTIN_NAMES:
    _py_builtins.globals()[_builtin_name] = _py_builtins.getattr(
        _py_builtins, _builtin_name,
    )


SIGNAL3_MODEL_VERSION = "sector-signal-v3.0.6-future-label-shadow-candidate-20260825"
SIGNAL3_STATE_CONTRACT = "signal3-v3.0.3-second-wave-state-r3"
SIGNAL3_GATE_CONTRACT = "signal3-v3.0.0-second-wave-six-gates-r1"
SIGNAL3_THEME_CONTEXT_CONTRACT = "signal3-v3.0.5-stable-theme-anchor-shadow-r1"
SIGNAL3_FUTURE_LABEL_CONTRACT = "signal3-v3.0.6-future-label-shadow-r1"
SIGNAL3_SOURCE_HYPOTHESIS = "docs/report/A股高辨识度大牛股第二波行情形成机制研究报告.md"
SIGNAL3_IMPLEMENTATION_STATUS = (
    "S3-4.0候选：未来标签物理隔离影子采集，保留稳定题材锚与战术上下文；"
    "标签只用于后验校准，不改写v3.0.3六道门、状态或当日候选；分层统计尚未验收"
)

# =========================
# 配置区
# =========================

# 单日优先：TRADE_DATE 有值时忽略 START_DATE/END_DATE。
# 三者均为 None 时自动选择最近已完整收盘交易日。
TRADE_DATE = None
START_DATE = "2026-08-11"
END_DATE = None
LATEST_COMPLETED_TIME = dt.time(15, 30)

AUTO_RUN = True
SAVE_HTML = True
RESOURCE_PROFILE = "2c4g"
PRICE_BATCH_SIZE = 300
VALUATION_BATCH_SIZE = 300
LOOKBACK_TRADE_DAYS = 90
MAX_ANALYSIS_DAYS = 60
TOP_CANDIDATES_PER_DAY = 20
MIN_PRICE_COVERAGE = 0.90
MIN_DAILY_AMOUNT = 50000000.0
MIN_IPO_DAYS = 60
ENABLE_CONCEPT_MEMBERSHIP_CACHE = True
CONCEPT_MEMBERSHIP_CACHE_VERSION = "signal3-theme-context-membership-r1"
STABLE_ANCHOR_SWITCH_MARGIN = 5.0
STABLE_ANCHOR_SWITCH_DAYS = 2
MAX_STABLE_ANCHOR_SWITCH_RATE = 0.35

# 标签只在全部 T 日状态和守恒终检后生成，不得回写 feature_daily 或状态机。
ENABLE_FUTURE_LABELS = True
FUTURE_LABEL_MAX_HORIZON = 10


SECOND_WAVE_STATES = (
    "未入池", "第一波确立", "死亡测试", "承接观察", "主动修复",
    "二波候选", "二波确认", "再分歧", "逻辑失效",
)
LEGAL_STATE_TRANSITIONS = {
    "未入池": ("未入池", "第一波确立", "逻辑失效"),
    "第一波确立": ("第一波确立", "死亡测试", "逻辑失效"),
    "死亡测试": ("死亡测试", "承接观察", "主动修复", "逻辑失效"),
    "承接观察": ("死亡测试", "承接观察", "主动修复", "逻辑失效"),
    "主动修复": ("死亡测试", "承接观察", "主动修复", "二波候选", "再分歧", "逻辑失效"),
    "二波候选": ("主动修复", "二波候选", "二波确认", "再分歧", "逻辑失效"),
    "二波确认": ("二波确认", "再分歧", "逻辑失效"),
    "再分歧": ("承接观察", "主动修复", "二波候选", "二波确认", "再分歧", "逻辑失效"),
    # 逻辑失效在同一次连续区间内保持粘滞；离开候选池后重新入池才允许重新初始化。
    "逻辑失效": ("逻辑失效",),
}
GATE_STATUSES = ("通过", "条件", "阻断", "数据不足")
SECOND_WAVE_COLUMNS = (
    "trade_date", "code", "name", "second_wave_state", "unconstrained_state",
    "applicability_status",
    "first_wave_gate", "theme_vitality_gate", "death_test_gate",
    "chip_restructure_gate", "market_environment_gate", "leader_scarcity_gate",
    "shadow_score", "candidate_rank", "first_wave_score", "wave_return_20_max",
    "limit_up_count_20_max", "max_limit_streak_20", "days_since_peak",
    "first_wave_peak_date", "death_test_date", "prior_observation_date", "prior_state",
    "candidate_streak_days", "state_streak_days", "observation_gap_days", "state_transition",
    "transition_guard_reason",
    "peak_drawdown", "pressure_recovery", "current_return", "relative_market_return",
    "turnover_rate", "amount_ratio5", "close_position", "industry_name",
    "board_state_score", "board_breadth", "board_drive", "market_state",
    "shadow_context_type", "shadow_context_code", "shadow_context_name",
    "shadow_context_score", "shadow_context_member_count",
    "shadow_context_breadth", "shadow_context_drive",
    "shadow_context_candidate_rank", "shadow_context_leader_code",
    "shadow_context_leader_name", "shadow_context_leader_gap",
    "shadow_pricing_power_status", "shadow_context_selection_reason",
    "shadow_anchor_streak_days", "shadow_anchor_previous_name",
    "shadow_anchor_challenger_name", "shadow_anchor_challenger_gap",
    "shadow_anchor_switch_reason",
    "shadow_tactical_context_type", "shadow_tactical_context_code",
    "shadow_tactical_context_name", "shadow_tactical_context_score",
    "shadow_anchor_tactical_relation",
    "shadow_other_contexts",
    "signal_day_trade_status", "recent_nontrading_days_20", "recent_nontrading_days_60",
    "recent_trade_coverage_60",
    "support_evidence", "risk_evidence", "time_boundary", "state_contract",
    "gate_contract",
)
FUTURE_LABEL_COLUMNS = (
    "signal_date", "code", "name", "second_wave_state", "applicability_status",
    "shadow_score", "stable_anchor_type", "stable_anchor_name",
    "entry_date", "entry_status", "available_future_sessions",
    "return_1d", "t1_close_position", "t1_acceptance_status",
    "return_3d", "return_5d", "return_10d", "max_return_10d",
    "max_drawdown_10d", "breakout_after_signal", "label_status",
    "time_boundary", "label_contract",
)
FUTURE_LABEL_STATUSES = ("matured", "partial", "pending", "data_incomplete")
FUTURE_ENTRY_STATUSES = (
    "ready", "paused", "no_trade", "limit_up_locked", "limit_down_locked",
    "data_incomplete", "pending",
)
FUTURE_T1_ACCEPTANCE_STATUSES = (
    "accepted", "neutral", "negative_feedback", "not_tradable",
    "data_incomplete", "pending",
)

# 数据合同继续使用稳定英文列名；只在 HTML 展示层投影成中文，避免显示需求改写计算链。
LATEST_DISPLAY_COLUMNS = (
    "trade_date", "candidate_rank", "target_display", "second_wave_state",
    "state_transition", "applicability_status", "gate_summary", "shadow_score",
    "first_wave_score", "event_timeline", "peak_drawdown", "pressure_recovery",
    "current_return", "shadow_context_display",
    "shadow_tactical_context_display", "shadow_anchor_tactical_relation",
    "shadow_pricing_power_status", "market_state",
)
MARKET_CONTEXT_COLUMNS = (
    "trade_date", "market_state", "market_cycle_score", "market_advance_ratio",
    "market_limit_up_count", "market_limit_down_count", "market_amount",
    "market_amount_ratio", "sample_size",
)
SIGNAL_DAY_COVERAGE_COLUMNS = (
    "trade_date", "universe_count", "eligible_count", "effective_count",
    "paused_count", "no_trade_count", "data_incomplete_count", "excluded_count",
    "paused_codes", "no_trade_codes", "data_incomplete_samples",
    "effective_coverage", "total_coverage",
)
THEME_CONTEXT_AUDIT_COLUMNS = (
    "trade_date", "concept_universe_count", "concept_membership_ready",
    "concept_membership_failed", "candidate_count", "candidates_with_concept",
    "primary_concept_count", "primary_industry_count", "industry_fallback_count",
    "missing_primary_count", "primary_context_conservation",
    "duplicate_candidate_contexts", "stable_anchor_initial_count",
    "stable_anchor_hold_count", "stable_anchor_switch_count",
    "stable_anchor_pending_count", "stable_anchor_action_conservation",
    "meta_context_excluded_count", "status",
    "theme_context_contract",
)
QUALITY_COLUMNS = ("item", "status", "detail")
AUDIT_COLUMNS = (
    "model", "start_date", "end_date", "analysis_days", "universe_codes",
    "candidate_codes", "observation_rows", "future_label_rows", "elapsed_seconds",
    "future_label_price_end", "future_label_matured", "future_label_partial",
    "future_label_pending", "future_label_incomplete",
    "raw_history_coverage", "adjusted_history_coverage", "dual_history_coverage",
    "min_signal_day_coverage", "continuity_status", "continuity_breaks",
    "illegal_transition_count", "candidate_trade_status_violations",
    "signal_day_paused_excluded", "signal_day_no_trade_excluded",
    "signal_day_data_incomplete_excluded",
    "theme_context_status", "theme_context_failures", "theme_context_missing_candidates",
    "anchor_stability_status", "anchor_transition_count", "anchor_switch_count",
    "anchor_switch_rate", "anchor_tactical_divergence_count",
    "state_contract", "gate_contract", "theme_context_contract",
    "future_label_contract", "report_status",
)
REPORT_COLUMN_LABELS = {
    "trade_date": "交易日", "code": "代码", "name": "名称",
    "target_display": "标的", "second_wave_state": "第二波状态",
    "unconstrained_state": "未约束状态",
    "applicability_status": "适用性", "first_wave_gate": "第一波辨识度门",
    "theme_vitality_gate": "题材生命力门", "death_test_gate": "死亡测试门",
    "chip_restructure_gate": "筹码重组门", "market_environment_gate": "市场环境门",
    "leader_scarcity_gate": "龙头稀缺性门", "gate_summary": "六道门",
    "shadow_score": "影子总分", "candidate_rank": "候选排名",
    "first_wave_score": "第一波辨识度分", "wave_return_20_max": "20日最大阶段涨幅",
    "limit_up_count_20_max": "20日最多涨停数", "max_limit_streak_20": "最大连续涨停数",
    "days_since_peak": "距压力峰交易日", "first_wave_peak_date": "第一波压力峰日期",
    "death_test_date": "死亡测试日期", "prior_observation_date": "区间前次观察日期",
    "prior_state": "区间前一连续状态", "candidate_streak_days": "区间候选连续日数",
    "state_streak_days": "区间状态连续日数", "observation_gap_days": "区间观察中断日数",
    "state_transition": "区间状态迁移", "transition_guard_reason": "迁移门说明",
    "event_timeline": "事件轴",
    "peak_drawdown": "压力峰回撤",
    "pressure_recovery": "压力区修复度", "current_return": "当日涨跌幅",
    "relative_market_return": "相对市场收益", "turnover_rate": "换手率",
    "amount_ratio5": "成交额/5日均额", "close_position": "收盘位置",
    "industry_name": "申万行业", "board_state_score": "板块结构分",
    "board_breadth": "板块上涨广度", "board_drive": "板块带动度",
    "market_state": "市场环境", "signal_day_trade_status": "信号日交易状态",
    "shadow_context_type": "稳定题材锚类型",
    "shadow_context_code": "稳定题材锚代码",
    "shadow_context_name": "稳定题材锚名称",
    "shadow_context_display": "稳定题材锚",
    "shadow_context_score": "稳定题材锚选择分",
    "shadow_context_member_count": "稳定题材锚有效成员数",
    "shadow_context_breadth": "稳定题材锚上涨广度",
    "shadow_context_drive": "稳定题材锚带动度",
    "shadow_context_candidate_rank": "候选在稳定锚内排名",
    "shadow_context_leader_code": "稳定锚领先核心代码",
    "shadow_context_leader_name": "稳定锚领先核心名称",
    "shadow_context_leader_gap": "候选落后稳定锚核心分差",
    "shadow_pricing_power_status": "定价权影子状态",
    "shadow_context_selection_reason": "稳定锚选择依据",
    "shadow_anchor_streak_days": "稳定锚连续日数",
    "shadow_anchor_previous_name": "前一稳定题材锚",
    "shadow_anchor_challenger_name": "稳定锚挑战者",
    "shadow_anchor_challenger_gap": "挑战者领先分差",
    "shadow_anchor_switch_reason": "稳定锚切换说明",
    "shadow_tactical_context_type": "战术共振类型",
    "shadow_tactical_context_code": "战术共振代码",
    "shadow_tactical_context_name": "战术共振名称",
    "shadow_tactical_context_score": "战术共振选择分",
    "shadow_tactical_context_display": "当日最强战术共振",
    "shadow_anchor_tactical_relation": "稳定锚/战术关系",
    "shadow_other_contexts": "其他点时标签",
    "recent_nontrading_days_20": "近20日停牌/无成交日数",
    "recent_nontrading_days_60": "近60日停牌/无成交日数",
    "recent_trade_coverage_60": "近60日有成交覆盖率",
    "support_evidence": "支持证据",
    "risk_evidence": "风险证据", "time_boundary": "时间边界",
    "state_contract": "状态合同", "gate_contract": "六门合同",
    "market_cycle_score": "市场周期分", "market_advance_ratio": "上涨家数占比",
    "market_limit_up_count": "涨停数", "market_limit_down_count": "跌停数",
    "market_amount": "市场成交额", "market_amount_ratio": "市场成交额环比",
    "sample_size": "市场样本数", "signal_date": "信号日",
    "universe_count": "点时股票数", "eligible_count": "名称初筛分母",
    "effective_count": "有效特征股票数", "paused_count": "信号日停牌剔除数",
    "no_trade_count": "信号日无成交剔除数",
    "data_incomplete_count": "数据不完整剔除数", "excluded_count": "合计剔除数",
    "paused_codes": "停牌标的", "no_trade_codes": "无成交标的",
    "data_incomplete_samples": "数据不完整样本（最多10只）",
    "effective_coverage": "有效覆盖率",
    "total_coverage": "全股票口径覆盖率",
    "stable_anchor_type": "信号日稳定锚类型",
    "stable_anchor_name": "信号日稳定锚名称",
    "entry_date": "下一交易日", "entry_status": "下一交易日可交易状态",
    "available_future_sessions": "已到期未来交易日数",
    "return_1d": "T+1收盘收益", "t1_close_position": "T+1收盘位置",
    "t1_acceptance_status": "T+1承接标签",
    "return_3d": "未来3日收益", "return_5d": "未来5日收益",
    "return_10d": "未来10日收益", "max_return_10d": "未来10日最大涨幅",
    "max_drawdown_10d": "未来10日最大回撤",
    "breakout_after_signal": "信号后突破", "label_status": "未来标签状态",
    "label_contract": "未来标签合同",
    "item": "质量项目", "status": "状态", "detail": "说明",
    "model": "模型版本", "start_date": "起始交易日", "end_date": "结束交易日",
    "analysis_days": "分析交易日数", "universe_codes": "股票宇宙数",
    "candidate_codes": "候选股票数", "observation_rows": "观察行数",
    "future_label_rows": "未来标签行数", "elapsed_seconds": "耗时（秒）",
    "future_label_price_end": "未来标签行情截止日",
    "future_label_matured": "已成熟标签数",
    "future_label_partial": "部分成熟标签数",
    "future_label_pending": "待成熟标签数",
    "future_label_incomplete": "数据不完整标签数",
    "raw_history_coverage": "未复权历史代码覆盖率",
    "adjusted_history_coverage": "前复权历史代码覆盖率",
    "dual_history_coverage": "双口径历史代码覆盖率",
    "min_signal_day_coverage": "最低信号日有效覆盖率",
    "continuity_status": "连续性审计状态", "continuity_breaks": "连续性错误数",
    "illegal_transition_count": "非法状态迁移数",
    "candidate_trade_status_violations": "候选交易状态违规数",
    "signal_day_paused_excluded": "区间停牌剔除合计",
    "signal_day_no_trade_excluded": "区间无成交剔除合计",
    "signal_day_data_incomplete_excluded": "区间数据不完整剔除合计",
    "concept_universe_count": "点时概念总数",
    "concept_membership_ready": "概念成分读取成功数",
    "concept_membership_failed": "概念成分读取失败数",
    "candidate_count": "候选数", "candidates_with_concept": "有概念候选数",
    "primary_concept_count": "概念稳定锚数",
    "primary_industry_count": "行业稳定锚数",
    "industry_fallback_count": "行业回退数",
    "missing_primary_count": "稳定锚缺失数",
    "primary_context_conservation": "稳定锚守恒",
    "duplicate_candidate_contexts": "候选上下文重复数",
    "stable_anchor_initial_count": "稳定锚新建数",
    "stable_anchor_hold_count": "稳定锚保持数",
    "stable_anchor_switch_count": "稳定锚切换数",
    "stable_anchor_pending_count": "稳定锚待确认挑战数",
    "stable_anchor_action_conservation": "稳定锚动作守恒",
    "meta_context_excluded_count": "锚排除元标签数",
    "theme_context_status": "题材上下文审计状态",
    "theme_context_failures": "区间概念读取失败数",
    "theme_context_missing_candidates": "区间稳定锚缺失候选数",
    "anchor_stability_status": "稳定锚审计状态",
    "anchor_transition_count": "稳定锚可比较迁移数",
    "anchor_switch_count": "稳定锚切换数",
    "anchor_switch_rate": "稳定锚切换率",
    "anchor_tactical_divergence_count": "稳定锚/战术分离行数",
    "theme_context_contract": "题材上下文合同",
    "future_label_contract": "未来标签合同",
    "report_status": "报告状态",
}


def safe_float(value, default=0.0):
    """把有限数值转成 float；None/NaN/无穷不伪装为真实零。"""
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clip(value, lower=0.0, upper=1.0):
    return min(max(safe_float(value), lower), upper)


def gate_result(status, score, support, risk):
    if status not in GATE_STATUSES:
        raise ValueError("非法门状态：%s" % status)
    return {
        "status": status,
        "score": None if score is None else round(clip(score, 0.0, 100.0), 3),
        "support": str(support or "-") if status != "数据不足" else str(support or "数据不足"),
        "risk": str(risk or "-"),
    }


def evaluate_first_wave_gate(snapshot):
    wave_return = finite_or_none(snapshot.get("wave_return_20_max"))
    limit_count = finite_or_none(snapshot.get("limit_up_count_20_max"))
    streak = finite_or_none(snapshot.get("max_limit_streak_20"))
    amount_pct = finite_or_none(snapshot.get("amount_percentile"))
    if None in (wave_return, limit_count, streak, amount_pct):
        return gate_result("数据不足", None, "第一波历史字段不完整", "禁止用0补缺")
    score = (
        35.0 * clip(wave_return / 1.20)
        + 30.0 * clip(limit_count / 7.0)
        + 20.0 * clip(streak / 5.0)
        + 15.0 * clip(amount_pct / 100.0)
    )
    if (wave_return >= 0.60 and limit_count >= 4) or score >= 72.0:
        status = "通过"
    elif wave_return >= 0.35 or limit_count >= 3 or streak >= 3:
        status = "条件"
    else:
        status = "阻断"
    support = "20日最大涨幅%.1f%%；20日最多涨停%.0f；最大连板%.0f；成交分位%.1f" % (
        wave_return * 100.0, limit_count, streak, amount_pct,
    )
    risk = "第一波辨识度不足" if status == "阻断" else "-"
    return gate_result(status, score, support, risk)


def evaluate_theme_vitality_gate(snapshot):
    board_score = finite_or_none(snapshot.get("board_state_score"))
    breadth = finite_or_none(snapshot.get("board_breadth"))
    drive = finite_or_none(snapshot.get("board_drive"))
    lifecycle = snapshot.get("board_lifecycle")
    if None in (board_score, breadth, drive) or not lifecycle:
        return gate_result("数据不足", None, "题材/行业点时结构不完整", "不把缺失题材视为通过")
    score = 45.0 * clip(board_score / 100.0) + 30.0 * clip(breadth) + 25.0 * clip(drive)
    if lifecycle == "退潮" or breadth < 0.25 or board_score < 35.0:
        status = "阻断"
    elif board_score >= 72.0 and breadth >= 0.55 and drive >= 0.30:
        status = "通过"
    else:
        status = "条件"
    support = "%s；结构分%.1f；上涨广度%.1f%%；带动%.2f" % (
        snapshot.get("industry_name") or "未知板块", board_score, breadth * 100.0, drive,
    )
    risk = "所属结构退潮或共振不足" if status == "阻断" else "-"
    return gate_result(status, score, support, risk)


def evaluate_death_test_gate(snapshot):
    triggered = snapshot.get("death_test_triggered")
    drawdown = finite_or_none(snapshot.get("peak_drawdown"))
    recovery = finite_or_none(snapshot.get("pressure_recovery"))
    negative_days = finite_or_none(snapshot.get("consecutive_negative_days"))
    if triggered is None or None in (drawdown, recovery, negative_days):
        return gate_result("数据不足", None, "死亡测试字段不完整", "禁止事后猜测")
    if not bool(triggered):
        return gate_result("条件", 45.0, "第一波已识别但尚无有效死亡测试", "不能把第一波延伸冒充二波")
    score = 40.0 * clip((drawdown + 0.35) / 0.35) + 60.0 * clip(recovery)
    if negative_days >= 2 or drawdown <= -0.35:
        status = "阻断"
    elif recovery >= 0.20 or safe_float(snapshot.get("close_position"), 0.0) >= 0.50:
        status = "通过"
    else:
        status = "条件"
    support = "峰值回撤%.1f%%；压力区修复%.1f%%；连续负反馈%.0f日" % (
        drawdown * 100.0, recovery * 100.0, negative_days,
    )
    risk = "连续负反馈或深度A杀" if status == "阻断" else "-"
    return gate_result(status, score, support, risk)


def evaluate_chip_restructure_gate(snapshot):
    turnover = finite_or_none(snapshot.get("turnover_rate"))
    amount_ratio = finite_or_none(snapshot.get("amount_ratio5"))
    close_position = finite_or_none(snapshot.get("close_position"))
    recovery = finite_or_none(snapshot.get("pressure_recovery"))
    if None in (amount_ratio, close_position, recovery):
        return gate_result("数据不足", None, "成交或收盘位置字段不完整", "禁止把缺失换手视为充分换手")
    turnover_component = 0.50 if turnover is None else clip(turnover / 35.0)
    score = (
        30.0 * turnover_component
        + 25.0 * clip(1.0 - abs(amount_ratio - 1.20) / 1.80)
        + 25.0 * clip(close_position)
        + 20.0 * clip(recovery)
    )
    if amount_ratio >= 2.50 and close_position < 0.30 and recovery < 0.20:
        status = "阻断"
    elif close_position >= 0.55 and 0.55 <= amount_ratio <= 2.20 and recovery >= 0.20:
        status = "通过" if turnover is not None else "条件"
    else:
        status = "条件"
    support = "换手%s；成交/5日均额%.2f；收盘位置%.2f；修复%.1f%%" % (
        "%.1f%%" % turnover if turnover is not None else "缺失",
        amount_ratio, close_position, recovery * 100.0,
    )
    risk = "爆量低收且价格重心未恢复" if status == "阻断" else (
        "历史换手缺失" if turnover is None else "-"
    )
    return gate_result(status, score, support, risk)


def evaluate_market_environment_gate(snapshot):
    state = snapshot.get("market_state")
    cycle = finite_or_none(snapshot.get("market_cycle_score"))
    advance = finite_or_none(snapshot.get("market_advance_ratio"))
    limit_up = finite_or_none(snapshot.get("market_limit_up_count"))
    limit_down = finite_or_none(snapshot.get("market_limit_down_count"))
    amount_ratio = finite_or_none(snapshot.get("market_amount_ratio"))
    if not state or None in (cycle, advance, limit_up, limit_down, amount_ratio):
        return gate_result("数据不足", None, "市场环境字段不完整", "市场缺失时禁止判断二波适用")
    score = (
        35.0 * clip(cycle / 100.0)
        + 25.0 * clip(advance)
        + 20.0 * clip(limit_up / 80.0)
        + 10.0 * clip(1.0 - limit_down / 20.0)
        + 10.0 * clip(amount_ratio / 1.10)
    )
    if state == "退潮" or cycle < 35.0 or limit_down >= max(10.0, limit_up * 0.50) or amount_ratio < 0.60:
        status = "阻断"
    elif state == "主升" and cycle >= 60.0 and limit_down <= 5.0 and amount_ratio >= 0.85:
        status = "通过"
    else:
        status = "条件"
    support = "%s；周期%.1f；上涨%.1f%%；涨/跌停%.0f/%.0f；成交环比%.2f" % (
        state, cycle, advance * 100.0, limit_up, limit_down, amount_ratio,
    )
    risk = "市场退潮或流动性/高位反馈阻断" if status == "阻断" else "-"
    return gate_result(status, score, support, risk)


def evaluate_leader_scarcity_gate(snapshot):
    rank = finite_or_none(snapshot.get("candidate_rank"))
    score_gap = finite_or_none(snapshot.get("top_score_gap"))
    replaced = snapshot.get("replaced_by_new_leader")
    drive = finite_or_none(snapshot.get("board_drive"))
    if None in (rank, score_gap, drive) or replaced is None:
        return gate_result("数据不足", None, "龙头竞争字段不完整", "无法判断是否被新王替代")
    score = 50.0 * clip((11.0 - rank) / 10.0) + 25.0 * clip((10.0 - max(score_gap, 0.0)) / 10.0) + 25.0 * clip(drive)
    if bool(replaced) and score_gap >= 8.0:
        status = "阻断"
    elif rank <= 3 and not bool(replaced):
        status = "通过"
    else:
        status = "条件"
    support = "辨识度排名%.0f；距第一名%.1f分；板块带动%.2f" % (rank, score_gap, drive)
    risk = "新核心显著领先" if status == "阻断" else "-"
    return gate_result(status, score, support, risk)


def combine_gate_results(gates):
    statuses = [item.get("status") for item in gates]
    if any(status == "阻断" for status in statuses):
        return "不适用"
    if any(status == "数据不足" for status in statuses):
        return "数据不足"
    if any(status == "条件" for status in statuses):
        return "条件观察"
    return "适用"


def build_continuity_metadata(snapshot, state):
    """生成只依赖当前研究序列位置的连续性字段；间断后不得沿用旧状态。"""
    prior_state = snapshot.get("prior_state")
    previous_observed_state = snapshot.get("previous_observed_state")
    gap_days = max(int(snapshot.get("observation_gap_days") or 0), 0)
    candidate_streak = max(int(snapshot.get("candidate_streak_days") or 1), 1)
    previous_state_streak = max(int(snapshot.get("prior_state_streak_days") or 0), 0)
    state_streak = previous_state_streak + 1 if prior_state == state else 1
    if prior_state:
        if prior_state == state:
            transition = "区间保持%s（连续%s日）" % (state, state_streak)
        else:
            transition = "%s→%s（区间连续交易日迁移）" % (prior_state, state)
    elif previous_observed_state and gap_days > 0:
        transition = "重新入池（中断%s个研究交易日；前次%s）" % (
            gap_days, previous_observed_state,
        )
    else:
        transition = "区间首次观察"
    return {
        "prior_observation_date": snapshot.get("prior_observation_date"),
        "prior_state": prior_state,
        "candidate_streak_days": candidate_streak,
        "state_streak_days": state_streak,
        "observation_gap_days": gap_days,
        "state_transition": transition,
    }


def infer_second_wave_state(snapshot, gates):
    """只读取当前T日快照和T日前态；不读取任何 future_* 字段。"""
    forbidden = [key for key in snapshot if str(key).startswith("future_")]
    if forbidden:
        raise ValueError("T日状态禁止读取或携带未来字段：%s" % forbidden)
    first_wave = gates["first_wave"]
    theme = gates["theme"]
    death = gates["death"]
    market = gates["market"]
    scarcity = gates["scarcity"]
    prior_state = snapshot.get("prior_state")
    current_return = safe_float(snapshot.get("current_return"))
    relative = safe_float(snapshot.get("relative_market_return"))
    close_position = safe_float(snapshot.get("close_position"), 0.5)
    recovery = safe_float(snapshot.get("pressure_recovery"))
    active_repair = current_return >= 0.03 and relative >= 0.02 and close_position >= 0.60
    critical_block = any(item["status"] == "阻断" for item in (theme, death, market, scarcity))

    if first_wave["status"] == "阻断":
        return "未入池"
    if death["status"] == "阻断" or (
        critical_block and safe_float(snapshot.get("consecutive_negative_days")) >= 2
    ):
        return "逻辑失效"
    if prior_state in ("二波候选", "二波确认") and current_return <= -0.05:
        return "再分歧"
    if not bool(snapshot.get("death_test_triggered")):
        return "第一波确立"
    if bool(snapshot.get("breakout_confirmed")) and active_repair and not critical_block:
        return "二波确认"
    if active_repair and recovery >= 0.35 and not critical_block:
        if all(gates[name]["status"] != "数据不足" for name in gates):
            return "二波候选"
        return "主动修复"
    if active_repair:
        return "主动修复"
    if death["status"] == "通过" and recovery >= 0.15:
        return "承接观察"
    return "死亡测试"


def is_legal_state_transition(prior_state, next_state):
    """区间首日/重新入池可初始化；连续观察必须遵守显式迁移表。"""
    if prior_state is None:
        return next_state in SECOND_WAVE_STATES
    return next_state in LEGAL_STATE_TRANSITIONS.get(prior_state, ())


def apply_state_transition_guard(prior_state, desired_state):
    """拦截跨阶段跳跃；总分和单日强势证据均不能越过状态机。"""
    if prior_state is None:
        return desired_state, "区间首日或重新入池，按T日证据初始化"
    if is_legal_state_transition(prior_state, desired_state):
        return desired_state, "合法迁移：%s→%s" % (prior_state, desired_state)
    if prior_state == "逻辑失效":
        guarded = "逻辑失效"
    elif desired_state in ("未入池", "第一波确立"):
        guarded = "逻辑失效"
    elif prior_state == "第一波确立":
        guarded = "死亡测试"
    elif prior_state in ("死亡测试", "承接观察"):
        guarded = "主动修复" if desired_state in ("二波候选", "二波确认") else prior_state
    elif prior_state == "主动修复" and desired_state == "二波确认":
        guarded = "二波候选"
    elif prior_state == "二波确认":
        guarded = "再分歧"
    elif prior_state == "二波候选":
        guarded = "再分歧"
    else:
        guarded = prior_state
    if not is_legal_state_transition(prior_state, guarded):
        guarded = prior_state
    return guarded, "拦截非法迁移%s→%s，保守落到%s" % (
        prior_state, desired_state, guarded,
    )


def build_second_wave_observation(snapshot):
    """从单个T日快照生成固定列观察行，输入保持不变。"""
    gate_map = {
        "first_wave": evaluate_first_wave_gate(snapshot),
        "theme": evaluate_theme_vitality_gate(snapshot),
        "death": evaluate_death_test_gate(snapshot),
        "chip": evaluate_chip_restructure_gate(snapshot),
        "market": evaluate_market_environment_gate(snapshot),
        "scarcity": evaluate_leader_scarcity_gate(snapshot),
    }
    unconstrained_state = infer_second_wave_state(snapshot, gate_map)
    state, transition_guard_reason = apply_state_transition_guard(
        snapshot.get("prior_state"), unconstrained_state,
    )
    ordered = [gate_map[name] for name in ("first_wave", "theme", "death", "chip", "market", "scarcity")]
    status = combine_gate_results(ordered)
    weights = (0.20, 0.20, 0.20, 0.15, 0.15, 0.10)
    scores = [item.get("score") for item in ordered]
    shadow_score = None if any(score is None for score in scores) else sum(
        score * weight for score, weight in zip(scores, weights)
    )
    continuity = build_continuity_metadata(snapshot, state)
    support = "｜".join(item["support"] for item in ordered if item["support"] != "-")
    risk = "｜".join(item["risk"] for item in ordered if item["risk"] != "-") or "未触发主要阻断"
    nontrading_20 = int(snapshot.get("recent_nontrading_days_20") or 0)
    if nontrading_20 > 0:
        risk += "｜近20日存在%s个停牌/无成交日，连续量价证据需谨慎" % nontrading_20
    row = {
        "trade_date": snapshot.get("trade_date"),
        "code": snapshot.get("code"),
        "name": snapshot.get("name"),
        "second_wave_state": state,
        "unconstrained_state": unconstrained_state,
        "applicability_status": status,
        "first_wave_gate": gate_map["first_wave"]["status"],
        "theme_vitality_gate": gate_map["theme"]["status"],
        "death_test_gate": gate_map["death"]["status"],
        "chip_restructure_gate": gate_map["chip"]["status"],
        "market_environment_gate": gate_map["market"]["status"],
        "leader_scarcity_gate": gate_map["scarcity"]["status"],
        "shadow_score": None if shadow_score is None else round(shadow_score, 3),
        "candidate_rank": snapshot.get("candidate_rank"),
        "first_wave_score": snapshot.get("first_wave_score"),
        "wave_return_20_max": snapshot.get("wave_return_20_max"),
        "limit_up_count_20_max": snapshot.get("limit_up_count_20_max"),
        "max_limit_streak_20": snapshot.get("max_limit_streak_20"),
        "days_since_peak": snapshot.get("days_since_peak"),
        "first_wave_peak_date": snapshot.get("first_wave_peak_date"),
        "death_test_date": snapshot.get("death_test_date"),
        "prior_observation_date": continuity["prior_observation_date"],
        "prior_state": continuity["prior_state"],
        "candidate_streak_days": continuity["candidate_streak_days"],
        "state_streak_days": continuity["state_streak_days"],
        "observation_gap_days": continuity["observation_gap_days"],
        "state_transition": continuity["state_transition"],
        "transition_guard_reason": transition_guard_reason,
        "peak_drawdown": snapshot.get("peak_drawdown"),
        "pressure_recovery": snapshot.get("pressure_recovery"),
        "current_return": snapshot.get("current_return"),
        "relative_market_return": snapshot.get("relative_market_return"),
        "turnover_rate": snapshot.get("turnover_rate"),
        "amount_ratio5": snapshot.get("amount_ratio5"),
        "close_position": snapshot.get("close_position"),
        "industry_name": snapshot.get("industry_name"),
        "board_state_score": snapshot.get("board_state_score"),
        "board_breadth": snapshot.get("board_breadth"),
        "board_drive": snapshot.get("board_drive"),
        "market_state": snapshot.get("market_state"),
        "shadow_context_type": snapshot.get("shadow_context_type"),
        "shadow_context_code": snapshot.get("shadow_context_code"),
        "shadow_context_name": snapshot.get("shadow_context_name"),
        "shadow_context_score": snapshot.get("shadow_context_score"),
        "shadow_context_member_count": snapshot.get("shadow_context_member_count"),
        "shadow_context_breadth": snapshot.get("shadow_context_breadth"),
        "shadow_context_drive": snapshot.get("shadow_context_drive"),
        "shadow_context_candidate_rank": snapshot.get("shadow_context_candidate_rank"),
        "shadow_context_leader_code": snapshot.get("shadow_context_leader_code"),
        "shadow_context_leader_name": snapshot.get("shadow_context_leader_name"),
        "shadow_context_leader_gap": snapshot.get("shadow_context_leader_gap"),
        "shadow_pricing_power_status": snapshot.get("shadow_pricing_power_status"),
        "shadow_context_selection_reason": snapshot.get("shadow_context_selection_reason"),
        "shadow_anchor_streak_days": snapshot.get("shadow_anchor_streak_days"),
        "shadow_anchor_previous_name": snapshot.get("shadow_anchor_previous_name"),
        "shadow_anchor_challenger_name": snapshot.get("shadow_anchor_challenger_name"),
        "shadow_anchor_challenger_gap": snapshot.get("shadow_anchor_challenger_gap"),
        "shadow_anchor_switch_reason": snapshot.get("shadow_anchor_switch_reason"),
        "shadow_tactical_context_type": snapshot.get("shadow_tactical_context_type"),
        "shadow_tactical_context_code": snapshot.get("shadow_tactical_context_code"),
        "shadow_tactical_context_name": snapshot.get("shadow_tactical_context_name"),
        "shadow_tactical_context_score": snapshot.get("shadow_tactical_context_score"),
        "shadow_anchor_tactical_relation": snapshot.get("shadow_anchor_tactical_relation"),
        "shadow_other_contexts": snapshot.get("shadow_other_contexts"),
        "signal_day_trade_status": snapshot.get("signal_day_trade_status") or "ready",
        "recent_nontrading_days_20": nontrading_20,
        "recent_nontrading_days_60": int(snapshot.get("recent_nontrading_days_60") or 0),
        "recent_trade_coverage_60": snapshot.get("recent_trade_coverage_60", 1.0),
        "support_evidence": support or "-",
        "risk_evidence": risk,
        "time_boundary": "只使用T日收盘已知字段；连续日数仅限指定研究区间；T+1及以后只能进入future_label_daily",
        "state_contract": SIGNAL3_STATE_CONTRACT,
        "gate_contract": SIGNAL3_GATE_CONTRACT,
    }
    validate_second_wave_observation(row)
    return row


def validate_second_wave_observation(row):
    if tuple(row.keys()) != SECOND_WAVE_COLUMNS:
        raise ValueError("signal3观察列合同漂移")
    if row.get("second_wave_state") not in SECOND_WAVE_STATES:
        raise ValueError("signal3非法状态：%s" % row.get("second_wave_state"))
    if row.get("unconstrained_state") not in SECOND_WAVE_STATES:
        raise ValueError("signal3非法未约束状态：%s" % row.get("unconstrained_state"))
    if not is_legal_state_transition(row.get("prior_state"), row.get("second_wave_state")):
        raise ValueError("signal3非法连续状态迁移：%s→%s" % (
            row.get("prior_state"), row.get("second_wave_state"),
        ))
    if row.get("signal_day_trade_status") != "ready":
        raise ValueError("signal3候选信号日不可交易：%s" % row.get("signal_day_trade_status"))
    for field in (
        "first_wave_gate", "theme_vitality_gate", "death_test_gate",
        "chip_restructure_gate", "market_environment_gate", "leader_scarcity_gate",
    ):
        if row.get(field) not in GATE_STATUSES:
            raise ValueError("signal3非法门状态：%s=%s" % (field, row.get(field)))
    if row.get("state_contract") != SIGNAL3_STATE_CONTRACT:
        raise ValueError("signal3状态合同不匹配")
    if row.get("gate_contract") != SIGNAL3_GATE_CONTRACT:
        raise ValueError("signal3六门合同不匹配")
    # ISO 日期字符串、date 与 Timestamp 的前10位均可按自然顺序比较，保持纯函数可本地测试。
    trade_date = str(row.get("trade_date"))[:10] if row.get("trade_date") is not None else None
    peak_date = str(row.get("first_wave_peak_date"))[:10] if row.get("first_wave_peak_date") is not None else None
    death_date = str(row.get("death_test_date"))[:10] if row.get("death_test_date") is not None else None
    if peak_date is not None and trade_date is not None and peak_date > trade_date:
        raise ValueError("第一波压力峰日期晚于信号日")
    if death_date is not None and trade_date is not None and death_date > trade_date:
        raise ValueError("死亡测试日期晚于信号日")
    if death_date is not None and peak_date is not None and death_date < peak_date:
        raise ValueError("死亡测试日期早于第一波压力峰")
    if int(row.get("candidate_streak_days") or 0) < 1:
        raise ValueError("候选连续日数非法")
    if int(row.get("state_streak_days") or 0) < 1:
        raise ValueError("状态连续日数非法")
    return True


def chunks(values, size):
    values = list(values)
    size = max(int(size), 1)
    for start in range(0, len(values), size):
        yield values[start:start + size]


def normalize_date(value):
    if value is None:
        return None
    return pd.Timestamp(value).date()


def resolve_research_dates(trade_date=None, start_date=None, end_date=None, now=None):
    now = now or dt.datetime.now()
    if trade_date is not None:
        requested = normalize_date(trade_date)
        dates = [normalize_date(x) for x in get_trade_days(start_date=requested, end_date=requested)]
        if dates != [requested]:
            raise ValueError("TRADE_DATE不是交易日：%s" % requested)
        if requested > now.date() or (requested == now.date() and now.time() < LATEST_COMPLETED_TIME):
            raise ValueError("TRADE_DATE尚未完整收盘：%s" % requested)
        return dates
    cutoff = now.date() if now.time() >= LATEST_COMPLETED_TIME else now.date() - dt.timedelta(days=1)
    end_value = normalize_date(end_date) if end_date is not None else normalize_date(
        get_trade_days(end_date=cutoff, count=1)[-1]
    )
    start_value = normalize_date(start_date) if start_date is not None else end_value
    if start_value > end_value:
        raise ValueError("START_DATE不能晚于END_DATE")
    if end_value > cutoff:
        raise ValueError("END_DATE尚未完整收盘：%s" % end_value)
    dates = [normalize_date(x) for x in get_trade_days(start_date=start_value, end_date=end_value)]
    if not dates:
        raise ValueError("指定区间没有交易日")
    if len(dates) > MAX_ANALYSIS_DAYS:
        raise ValueError("研究区间%s日超过上限%s；请分段运行" % (len(dates), MAX_ANALYSIS_DAYS))
    return dates


def resolve_latest_completed_trade_date(now=None):
    """返回最近完整收盘交易日；未来标签最多只能读到这里。"""
    now = now or dt.datetime.now()
    cutoff = now.date() if now.time() >= LATEST_COMPLETED_TIME else now.date() - dt.timedelta(days=1)
    dates = [normalize_date(value) for value in get_trade_days(end_date=cutoff, count=1)]
    if not dates:
        raise ValueError("无法解析最近完整收盘交易日")
    return dates[-1]


def resolve_future_label_price_end(end_day, now=None):
    """把标签行情窗口限制在研究末日后的最多十个已收盘交易日。"""
    end_day = normalize_date(end_day)
    latest_completed = resolve_latest_completed_trade_date(now)
    if latest_completed <= end_day:
        return end_day
    future_days = [
        normalize_date(value) for value in get_trade_days(
            start_date=end_day, end_date=latest_completed,
        )
        if normalize_date(value) > end_day
    ]
    return future_days[min(len(future_days), FUTURE_LABEL_MAX_HORIZON) - 1] if future_days else end_day


def normalize_price_batch(raw, requested_codes):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(code): frame.copy() for code, frame in raw.items() if isinstance(frame, pd.DataFrame)}
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return {}
    data = raw.copy()
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    elif "time" not in data.columns and "date" not in data.columns:
        index_name = data.index.name or "index"
        data = data.reset_index().rename(columns={index_name: "time"})
    code_column = next((name for name in ("code", "security", "order_book_id") if name in data.columns), None)
    if code_column is None:
        if len(requested_codes) != 1:
            return {}
        data["code"] = requested_codes[0]
        code_column = "code"
    time_column = next((name for name in ("time", "date", "datetime", "index") if name in data.columns), None)
    result = {}
    for code, part in data.groupby(code_column):
        frame = part.copy()
        if time_column:
            frame.index = pd.to_datetime(frame[time_column])
        drop_columns = [name for name in (code_column, time_column) if name and name in frame.columns]
        frame = frame.drop(columns=drop_columns, errors="ignore")
        result[str(code)] = frame.sort_index()
    return result


def load_price_history(codes, history_start, end_date):
    raw_result = {}
    adjusted_result = {}
    batches = list(chunks(codes, PRICE_BATCH_SIZE))
    fields = ["open", "close", "high", "low", "volume", "money", "high_limit", "low_limit", "paused"]
    started = time.time()
    for index, batch in enumerate(batches, 1):
        raw = get_price(
            batch, start_date=history_start, end_date=end_date, frequency="daily",
            fields=fields, fq=None, panel=False,
        )
        adjusted = get_price(
            batch, start_date=history_start, end_date=end_date, frequency="daily",
            fields=["open", "close", "high", "low"], fq="pre", panel=False,
        )
        raw_result.update(normalize_price_batch(raw, batch))
        adjusted_result.update(normalize_price_batch(adjusted, batch))
        if index == 1 or index == len(batches):
            print("[signal3行情] %s/%s批；耗时%.1f秒。" % (index, len(batches), time.time() - started))
    denominator = float(max(len(codes), 1))
    raw_coverage = len(set(raw_result)) / denominator
    adjusted_coverage = len(set(adjusted_result)) / denominator
    dual_coverage = len(set(raw_result).intersection(adjusted_result)) / denominator
    coverage = {
        "raw_history_coverage": raw_coverage,
        "adjusted_history_coverage": adjusted_coverage,
        "dual_history_coverage": dual_coverage,
    }
    if dual_coverage < MIN_PRICE_COVERAGE:
        raise RuntimeError("signal3双口径行情覆盖率%.1f%%低于%.1f%%" % (
            dual_coverage * 100.0, MIN_PRICE_COVERAGE * 100.0,
        ))
    return raw_result, adjusted_result, coverage


def frame_until(frame, day, tail=None):
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    result = frame[pd.to_datetime(frame.index).date <= day].copy()
    return result.tail(tail) if tail else result


def price_row_on_day(frame, day):
    """严格按交易日取一行；不使用最近值填补缺失交易日。"""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    day = normalize_date(day)
    mask = pd.to_datetime(frame.index).date == day
    matched = frame.loc[mask]
    return matched.iloc[-1] if not matched.empty else None


def classify_future_entry_status(raw_row):
    """仅用于 T+1 后验标签，不参与 T 日候选与状态。"""
    if raw_row is None:
        return "data_incomplete"
    if safe_float(raw_row.get("paused"), 0.0) > 0:
        return "paused"
    volume = finite_or_none(raw_row.get("volume"))
    amount = finite_or_none(raw_row.get("money"))
    if volume is None or amount is None:
        return "data_incomplete"
    if volume <= 0 or amount <= 0:
        return "no_trade"
    high = finite_or_none(raw_row.get("high"))
    low = finite_or_none(raw_row.get("low"))
    high_limit = finite_or_none(raw_row.get("high_limit"))
    low_limit = finite_or_none(raw_row.get("low_limit"))
    if None in (high, low, high_limit, low_limit):
        return "data_incomplete"
    if low >= high_limit * 0.999:
        return "limit_up_locked"
    if high <= low_limit * 1.001:
        return "limit_down_locked"
    return "ready"


def classify_t1_acceptance(entry_status, return_1d, close_position):
    """透明的 T+1 后验承接标签；阈值只用于分层，不参与 T 日状态。"""
    if entry_status == "pending":
        return "pending"
    if entry_status == "data_incomplete":
        return "data_incomplete"
    if entry_status != "ready":
        return "not_tradable"
    return_1d = finite_or_none(return_1d)
    close_position = finite_or_none(close_position)
    if return_1d is None or close_position is None:
        return "data_incomplete"
    if return_1d >= 0.0 or (return_1d >= -0.02 and close_position >= 0.65):
        return "accepted"
    if return_1d <= -0.05 or (return_1d < 0.0 and close_position <= 0.25):
        return "negative_feedback"
    return "neutral"


def build_future_label_rows(observation_rows, raw_map, adjusted_map, label_calendar):
    """
    在 T 日研究链完成后生成独立后验标签。

    标签保留每一条观察，部分成熟和待成熟不按失败处理；所有收益均以 T 日前复权
    收盘为基准，T+1 可交易状态使用未复权行情。该函数不得修改观察行或行情表。
    """
    calendar = sorted(set(normalize_date(value) for value in label_calendar))
    result = []
    for observation in observation_rows:
        signal_day = normalize_date(observation.get("trade_date"))
        code = str(observation.get("code") or "")
        future_days = [day for day in calendar if day > signal_day][:FUTURE_LABEL_MAX_HORIZON]
        available_sessions = len(future_days)
        entry_day = future_days[0] if future_days else None
        raw_frame = raw_map.get(code)
        adjusted_frame = adjusted_map.get(code)
        signal_row = price_row_on_day(adjusted_frame, signal_day)
        signal_close = finite_or_none(signal_row.get("close")) if signal_row is not None else None
        entry_row = price_row_on_day(raw_frame, entry_day) if entry_day is not None else None
        entry_status = classify_future_entry_status(entry_row) if entry_day is not None else "pending"

        adjusted_future_rows = [price_row_on_day(adjusted_frame, day) for day in future_days]
        missing_future_price = any(
            row is None
            or finite_or_none(row.get("close")) is None
            or finite_or_none(row.get("high")) is None
            for row in adjusted_future_rows
        )
        if available_sessions == 0:
            label_status = "pending"
        elif (signal_close is None or signal_close <= 0 or missing_future_price
              or entry_status == "data_incomplete"):
            label_status = "data_incomplete"
        elif available_sessions >= FUTURE_LABEL_MAX_HORIZON:
            label_status = "matured"
        else:
            label_status = "partial"

        def horizon_return(horizon):
            if signal_close is None or signal_close <= 0 or available_sessions < horizon:
                return None
            row = adjusted_future_rows[horizon - 1]
            close_value = finite_or_none(row.get("close")) if row is not None else None
            return None if close_value is None else round(close_value / signal_close - 1.0, 6)

        return_1d = horizon_return(1)
        entry_high = finite_or_none(entry_row.get("high")) if entry_row is not None else None
        entry_low = finite_or_none(entry_row.get("low")) if entry_row is not None else None
        entry_close = finite_or_none(entry_row.get("close")) if entry_row is not None else None
        t1_close_position = None
        if None not in (entry_high, entry_low, entry_close) and entry_high > entry_low:
            t1_close_position = round(clip(
                (entry_close - entry_low) / (entry_high - entry_low), 0.0, 1.0,
            ), 6)
        t1_acceptance_status = classify_t1_acceptance(
            entry_status, return_1d, t1_close_position,
        )

        future_closes = []
        future_highs = []
        if signal_close is not None and signal_close > 0:
            for row in adjusted_future_rows:
                if row is None:
                    continue
                close_value = finite_or_none(row.get("close"))
                high_value = finite_or_none(row.get("high"))
                if close_value is not None:
                    future_closes.append(close_value)
                if high_value is not None:
                    future_highs.append(high_value)
        max_return = (
            round(max(future_highs) / signal_close - 1.0, 6)
            if signal_close is not None and signal_close > 0 and future_highs else None
        )
        running_peak = signal_close
        drawdowns = []
        if running_peak is not None and running_peak > 0:
            for close_value in future_closes:
                running_peak = max(running_peak, close_value)
                drawdowns.append(close_value / running_peak - 1.0)
        max_drawdown = round(min(drawdowns), 6) if drawdowns else None

        peak_day = normalize_date(observation.get("first_wave_peak_date"))
        peak_row = price_row_on_day(adjusted_frame, peak_day) if peak_day is not None else None
        peak_close = finite_or_none(peak_row.get("close")) if peak_row is not None else None
        breakout_after_signal = None
        if peak_close is not None and peak_close > 0 and future_highs:
            breakout_after_signal = bool(max(future_highs) >= peak_close * 1.005)

        row = {
            "signal_date": signal_day,
            "code": code,
            "name": observation.get("name"),
            "second_wave_state": observation.get("second_wave_state"),
            "applicability_status": observation.get("applicability_status"),
            "shadow_score": observation.get("shadow_score"),
            "stable_anchor_type": observation.get("shadow_context_type"),
            "stable_anchor_name": observation.get("shadow_context_name"),
            "entry_date": entry_day,
            "entry_status": entry_status,
            "available_future_sessions": available_sessions,
            "return_1d": return_1d,
            "t1_close_position": t1_close_position,
            "t1_acceptance_status": t1_acceptance_status,
            "return_3d": horizon_return(3),
            "return_5d": horizon_return(5),
            "return_10d": horizon_return(10),
            "max_return_10d": max_return,
            "max_drawdown_10d": max_drawdown,
            "breakout_after_signal": breakout_after_signal,
            "label_status": label_status,
            "time_boundary": "仅在T日状态终检后读取T+1及以后；不回写T日状态",
            "label_contract": SIGNAL3_FUTURE_LABEL_CONTRACT,
        }
        if tuple(row) != FUTURE_LABEL_COLUMNS:
            raise RuntimeError("未来标签列合同漂移")
        if row["label_status"] not in FUTURE_LABEL_STATUSES:
            raise RuntimeError("非法未来标签状态：%s" % row["label_status"])
        if row["entry_status"] not in FUTURE_ENTRY_STATUSES:
            raise RuntimeError("非法未来可交易状态：%s" % row["entry_status"])
        if row["t1_acceptance_status"] not in FUTURE_T1_ACCEPTANCE_STATUSES:
            raise RuntimeError("非法T+1承接标签：%s" % row["t1_acceptance_status"])
        result.append(row)
    return result


def pct_changes(values):
    values = [safe_float(value) for value in values]
    result = []
    for index in range(1, len(values)):
        prior = values[index - 1]
        result.append(values[index] / prior - 1.0 if prior > 0 else 0.0)
    return result


def longest_true_streak(flags):
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def max_rolling_gain(closes, window):
    closes = [safe_float(value) for value in closes]
    best = 0.0
    for end in range(1, len(closes)):
        start = max(0, end - int(window))
        base = closes[start]
        if base > 0:
            best = max(best, closes[end] / base - 1.0)
    return best


def max_rolling_sum(flags, window):
    flags = [1 if flag else 0 for flag in flags]
    return max([sum(flags[max(0, end - window + 1):end + 1]) for end in range(len(flags))] or [0])


def detect_death_test(closes, returns, limit_flags, touched_flags, amount_ratios, close_positions):
    """保留第一波压力峰值；后续创新高不能抹掉已经发生的死亡测试。"""
    closes = [safe_float(value) for value in closes]
    if len(closes) < 2:
        return {
            "triggered": False, "event_index": None, "peak_index": 0,
            "peak_close": closes[0] if closes else 0.0,
        }
    running_low = closes[0]
    peak_close = closes[0]
    peak_index = 0
    event_index = None
    for index in range(1, len(closes)):
        prior_peak = peak_close
        prior_peak_index = peak_index
        peak_gain = prior_peak / running_low - 1.0 if running_low > 0 else 0.0
        prior_limits = sum(1 for flag in limit_flags[max(0, index - 20):index] if flag)
        wave_qualified = peak_gain >= 0.35 or prior_limits >= 3
        current_return = returns[index - 1] if index - 1 < len(returns) else 0.0
        drawdown = closes[index] / prior_peak - 1.0 if prior_peak > 0 else 0.0
        broken_board = bool(
            touched_flags[index] and not limit_flags[index]
            and amount_ratios[index] >= 1.50 and close_positions[index] < 0.55
        )
        if wave_qualified and index > prior_peak_index and (
            drawdown <= -0.10 or current_return <= -0.07 or broken_board
        ):
            event_index = index
            peak_close = prior_peak
            peak_index = prior_peak_index
            break
        if closes[index] > peak_close:
            peak_close = closes[index]
            peak_index = index
        running_low = min(running_low, closes[index])
    return {
        "triggered": event_index is not None,
        "event_index": event_index,
        "peak_index": peak_index,
        "peak_close": peak_close,
    }


def inspect_signal_day_trade_status(raw_frame, adjusted_frame, day):
    """区分停牌、无成交与数据不完整，并记录历史无交易日；不把三者混成缺失。"""
    raw = frame_until(raw_frame, day, LOOKBACK_TRADE_DAYS)
    adjusted = frame_until(adjusted_frame, day, LOOKBACK_TRADE_DAYS)
    recent = raw.tail(60) if raw is not None else pd.DataFrame()
    nontrading_flags = []
    if recent is not None and not recent.empty:
        for _, item in recent.iterrows():
            nontrading_flags.append(bool(
                safe_float(item.get("paused"), 0.0) > 0
                or safe_float(item.get("volume"), 0.0) <= 0
                or safe_float(item.get("money"), 0.0) <= 0
            ))
    recent_count = len(nontrading_flags)
    common = raw.index.intersection(adjusted.index) if not raw.empty and not adjusted.empty else []
    result = {
        "status": "data_incomplete",
        "reason": "行情历史不足",
        "recent_nontrading_days_20": sum(nontrading_flags[-20:]),
        "recent_nontrading_days_60": sum(nontrading_flags),
        "recent_trade_coverage_60": (
            (recent_count - sum(nontrading_flags)) / float(recent_count)
            if recent_count > 0 else 0.0
        ),
    }
    if len(raw) < 25 or len(adjusted) < 25 or len(common) < 25:
        return result
    if normalize_date(raw.index[-1]) != day or normalize_date(adjusted.index[-1]) != day:
        result["reason"] = "信号日日线缺失或日期不一致"
        return result
    last = raw.iloc[-1]
    adjusted_last = adjusted.iloc[-1]
    if safe_float(last.get("paused"), 0.0) > 0:
        result.update({"status": "paused", "reason": "信号日paused=True"})
        return result
    if safe_float(last.get("volume"), 0.0) <= 0 or safe_float(last.get("money"), 0.0) <= 0:
        result.update({"status": "no_trade", "reason": "信号日成交量或成交额为0"})
        return result
    if safe_float(adjusted_last.get("close"), 0.0) <= 0:
        result["reason"] = "信号日前复权收盘价无效"
        return result
    result.update({"status": "ready", "reason": "信号日有成交且双口径行情完整"})
    return result


def build_stock_snapshot(
    code, name, start_date, day, raw_frame, adjusted_frame, trade_status=None,
):
    trade_status = trade_status or inspect_signal_day_trade_status(
        raw_frame, adjusted_frame, day,
    )
    if trade_status.get("status") != "ready":
        return None
    raw = frame_until(raw_frame, day, LOOKBACK_TRADE_DAYS)
    adjusted = frame_until(adjusted_frame, day, LOOKBACK_TRADE_DAYS)
    if len(raw) < 25 or len(adjusted) < 25:
        return None
    if normalize_date(raw.index[-1]) != day or normalize_date(adjusted.index[-1]) != day:
        return None
    common_dates = raw.index.intersection(adjusted.index)
    raw = raw.loc[common_dates].sort_index()
    adjusted = adjusted.loc[common_dates].sort_index()
    if len(raw) < 25 or len(adjusted) < 25:
        return None
    last = raw.iloc[-1]
    adjusted_last = adjusted.iloc[-1]
    closes = [safe_float(value) for value in adjusted["close"]]
    returns = pct_changes(closes)
    if not closes or closes[-1] <= 0:
        return None
    limit_flags = []
    touched_flags = []
    close_positions = []
    amount_ratios = []
    amounts = [safe_float(value) for value in raw["money"]]
    for index, (_, row) in enumerate(raw.iterrows()):
        high_limit = safe_float(row.get("high_limit"))
        close_value = safe_float(row.get("close"))
        high_value = safe_float(row.get("high"))
        low_value = safe_float(row.get("low"))
        limit_flags.append(high_limit > 0 and close_value >= high_limit * 0.999)
        touched_flags.append(high_limit > 0 and high_value >= high_limit * 0.999)
        close_positions.append(0.5 if high_value <= low_value else clip((close_value - low_value) / (high_value - low_value)))
        prior_amounts = amounts[max(0, index - 5):index]
        prior_mean = sum(prior_amounts) / float(len(prior_amounts)) if prior_amounts else 0.0
        amount_ratios.append(amounts[index] / prior_mean if prior_mean > 0 else 1.0)
    wave_return = max_rolling_gain(closes[-61:], 20)
    limit_count = max_rolling_sum(limit_flags[-60:], 20)
    streak = longest_true_streak(limit_flags[-60:])
    window_start = max(0, len(closes) - 60)
    window_closes = closes[window_start:]
    detection = detect_death_test(
        window_closes,
        returns[max(0, window_start - 1):],
        limit_flags[window_start:],
        touched_flags[window_start:],
        amount_ratios[window_start:],
        close_positions[window_start:],
    )
    peak_local = int(detection["peak_index"])
    peak_close = safe_float(detection["peak_close"])
    global_peak_index = window_start + peak_local
    event_local = detection.get("event_index")
    global_event_index = window_start + int(event_local) if event_local is not None else None
    first_wave_peak_date = normalize_date(adjusted.index[global_peak_index])
    death_test_date = (
        normalize_date(adjusted.index[global_event_index])
        if global_event_index is not None else None
    )
    after_peak = closes[global_peak_index:]
    trough_close = min(after_peak) if after_peak else closes[-1]
    current_close = closes[-1]
    # 峰值回撤只表达尚未修复的下行距离；突破前高后归零，避免出现“正回撤”。
    drawdown = min(current_close / peak_close - 1.0, 0.0) if peak_close > 0 else 0.0
    recovery = 1.0 if peak_close <= trough_close else clip((current_close - trough_close) / (peak_close - trough_close))
    return_start = max((global_event_index or global_peak_index) - 1, 0)
    after_peak_returns = returns[return_start:]
    negative_days = 0
    for value in reversed(after_peak_returns):
        if value <= -0.05:
            negative_days += 1
        else:
            break
    death_triggered = bool(detection["triggered"])
    current_return = returns[-1] if returns else 0.0
    last_high = safe_float(adjusted_last.get("high"), current_close)
    last_low = safe_float(adjusted_last.get("low"), current_close)
    close_position = 0.5 if last_high <= last_low else clip((current_close - last_low) / (last_high - last_low))
    return {
        "trade_date": day,
        "code": code,
        "name": name,
        "start_date": start_date,
        "ipo_days": max((day - start_date).days, 0),
        "wave_return_20_max": wave_return,
        "limit_up_count_20_max": limit_count,
        "max_limit_streak_20": streak,
        "days_since_peak": len(window_closes) - peak_local - 1,
        "first_wave_peak_date": first_wave_peak_date,
        "death_test_date": death_test_date,
        "peak_drawdown": drawdown,
        "pressure_recovery": recovery,
        "death_test_triggered": death_triggered,
        "consecutive_negative_days": negative_days,
        "breakout_confirmed": current_close >= peak_close * 1.005 and death_triggered,
        "current_return": current_return,
        "close_position": close_position,
        "amount": amounts[-1],
        "previous_amount": amounts[-2],
        "amount_mean5": sum(amounts[-6:-1]) / float(max(len(amounts[-6:-1]), 1)),
        "amount_ratio5": amount_ratios[-1],
        "turnover_rate": None,
        "paused": False,
        "signal_day_trade_status": trade_status.get("status"),
        "recent_nontrading_days_20": int(trade_status.get("recent_nontrading_days_20") or 0),
        "recent_nontrading_days_60": int(trade_status.get("recent_nontrading_days_60") or 0),
        "recent_trade_coverage_60": round(
            safe_float(trade_status.get("recent_trade_coverage_60")), 6,
        ),
        "sealed_limit": bool(limit_flags[-1]),
        "sealed_down": bool(
            safe_float(last.get("low_limit")) > 0
            and safe_float(last.get("close")) <= safe_float(last.get("low_limit")) * 1.001
        ),
        "touched_limit": bool(touched_flags[-1]),
    }


def percentile_map(rows, field):
    ordered = sorted((safe_float(row.get(field)), row.get("code")) for row in rows)
    total = float(max(len(ordered), 1))
    return {code: (index + 0.5) / total * 100.0 for index, (_, code) in enumerate(ordered)}


def first_wave_discovery_score(row):
    return round(
        40.0 * clip(safe_float(row.get("wave_return_20_max")) / 1.20)
        + 30.0 * clip(safe_float(row.get("limit_up_count_20_max")) / 7.0)
        + 20.0 * clip(safe_float(row.get("max_limit_streak_20")) / 5.0)
        + 10.0 * clip(safe_float(row.get("amount_percentile")) / 100.0),
        3,
    )


def discover_candidates(rows):
    amount_percentiles = percentile_map(rows, "amount")
    candidates = []
    for row in rows:
        row = dict(row)
        row["amount_percentile"] = amount_percentiles.get(row.get("code"), 0.0)
        row["first_wave_score"] = first_wave_discovery_score(row)
        broad_entry = (
            safe_float(row.get("wave_return_20_max")) >= 0.35
            or safe_float(row.get("limit_up_count_20_max")) >= 3
            or safe_float(row.get("max_limit_streak_20")) >= 3
        )
        liquid = safe_float(row.get("amount_mean5")) >= MIN_DAILY_AMOUNT
        listed = int(row.get("ipo_days") or 0) >= MIN_IPO_DAYS
        if broad_entry and liquid and listed and not row.get("paused"):
            candidates.append(row)
    candidates.sort(key=lambda item: (-safe_float(item.get("first_wave_score")), str(item.get("code"))))
    for index, row in enumerate(candidates, 1):
        row["candidate_rank"] = index
    return candidates[:TOP_CANDIDATES_PER_DAY]


def load_candidate_valuations(codes, analysis_dates):
    result = {}
    get_valuation_func = globals().get("get_valuation")
    if not codes or get_valuation_func is None:
        return result, "degraded"
    try:
        for batch in chunks(codes, VALUATION_BATCH_SIZE):
            frame = get_valuation_func(
                batch, start_date=analysis_dates[0], end_date=analysis_dates[-1],
                fields=["code", "day", "turnover_ratio"],
            )
            if frame is None or not isinstance(frame, pd.DataFrame):
                continue
            data = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
            for row in data.to_dict("records"):
                code = str(row.get("code") or "")
                day_value = row.get("day") or row.get("date")
                if code and day_value is not None:
                    result[(code, normalize_date(day_value))] = finite_or_none(row.get("turnover_ratio"))
        return result, "ready" if result else "degraded"
    except Exception as exc:
        print("[signal3降级] 候选历史换手读取失败：%s" % str(exc)[:160])
        return {}, "degraded"


def industry_for_codes(codes, day):
    result = {}
    if not codes:
        return result
    try:
        raw = get_industry(codes, date=day)
    except Exception:
        raw = {}
        for code in codes:
            try:
                raw[code] = get_industry(code, date=day).get(code, {})
            except Exception:
                raw[code] = {}
    for code in codes:
        info = (raw or {}).get(code, {}) if isinstance(raw, dict) else {}
        selected = info.get("sw_l2") or info.get("sw_l1") or {}
        result[code] = {
            "industry_code": selected.get("industry_code"),
            "industry_name": selected.get("industry_name"),
        }
    return result


def build_industry_context(day, candidates, feature_rows_by_code):
    contexts = {}
    industry_map = industry_for_codes([row["code"] for row in candidates], day)
    member_cache = {}
    for row in candidates:
        code = row["code"]
        industry = industry_map.get(code, {})
        industry_code = industry.get("industry_code")
        industry_name = industry.get("industry_name")
        if not industry_code:
            contexts[code] = {}
            continue
        if industry_code not in member_cache:
            try:
                member_cache[industry_code] = list(get_industry_stocks(industry_code, date=day))
            except Exception:
                member_cache[industry_code] = []
        members = [feature_rows_by_code[item] for item in member_cache[industry_code] if item in feature_rows_by_code]
        returns = [safe_float(item.get("current_return")) for item in members]
        breadth = sum(1 for value in returns if value > 0.0) / float(max(len(returns), 1))
        median = float(np.nanmedian(returns)) if returns else 0.0
        limit_ratio = sum(1 for item in members if item.get("sealed_limit")) / float(max(len(members), 1))
        drive = clip(0.55 * breadth + 0.45 * clip(limit_ratio / 0.08))
        board_score = 45.0 * breadth + 25.0 * clip((median + 0.02) / 0.04) + 30.0 * clip(limit_ratio / 0.08)
        if breadth < 0.25:
            lifecycle = "退潮"
        elif breadth >= 0.65 and limit_ratio >= 0.04:
            lifecycle = "主升"
        elif breadth >= 0.50:
            lifecycle = "确认"
        else:
            lifecycle = "强分歧"
        contexts[code] = {
            "industry_code": industry_code,
            "industry_name": industry_name,
            "board_state_score": round(board_score, 3),
            "board_breadth": round(breadth, 4),
            "board_median_return": round(median, 4),
            "board_limit_ratio": round(limit_ratio, 4),
            "board_drive": round(drive, 4),
            "board_lifecycle": lifecycle,
            "board_member_count": len(members),
            # 仅供 v3.0.5 稳定锚/战术上下文复用；不会进入 v3.0.3 六道门输出列。
            "_board_member_codes": [item.get("code") for item in members if item.get("code")],
        }
    return contexts


def normalize_concept_catalog(concepts, day):
    """把 get_concepts() 归一成点时概念目录；创建日晚于T日的概念不得进入。"""
    if concepts is None or not isinstance(concepts, pd.DataFrame) or concepts.empty:
        return pd.DataFrame(columns=["concept_code", "concept_name"])
    frame = concepts.copy().reset_index()
    code_column = "code" if "code" in frame.columns else frame.columns[0]
    name_column = "name" if "name" in frame.columns else (
        "display_name" if "display_name" in frame.columns else code_column
    )
    result = pd.DataFrame({
        "concept_code": frame[code_column].astype(str),
        "concept_name": frame[name_column].astype(str),
    })
    if "start_date" in frame.columns:
        starts = pd.to_datetime(frame["start_date"], errors="coerce").dt.date
        result = result[starts.isna() | (starts <= day)]
    result = result.dropna(subset=["concept_code"]).drop_duplicates("concept_code")
    return result.sort_values("concept_code").reset_index(drop=True)


def load_concept_catalog(day):
    """概念目录只决定要读取哪些点时成分，不携带未来行情。"""
    try:
        return normalize_concept_catalog(get_concepts(), day), "ready"
    except Exception as exc:
        print("[signal3题材降级] 概念目录读取失败：%s" % str(exc)[:160])
        return pd.DataFrame(columns=["concept_code", "concept_name"]), "degraded"


def context_member_power_score(row):
    """用T日及以前字段衡量同一上下文中的相对定价权，不读取未来表现。"""
    return round(
        35.0 * clip(safe_float(row.get("wave_return_20_max")) / 1.20)
        + 25.0 * clip(safe_float(row.get("limit_up_count_20_max")) / 7.0)
        + 15.0 * clip(safe_float(row.get("max_limit_streak_20")) / 5.0)
        + 15.0 * clip((safe_float(row.get("current_return")) + 0.03) / 0.13)
        + 10.0 * clip(safe_float(row.get("amount_ratio5")) / 2.0),
        3,
    )


def build_context_evidence(context_type, context_code, context_name, member_codes, feature_rows_by_code):
    """构造一个行业/概念的统一影子证据，成员必须来自T日有效特征池。"""
    unique_codes = []
    seen = set()
    for raw_code in member_codes or []:
        code = str(raw_code)
        if code in feature_rows_by_code and code not in seen:
            unique_codes.append(code)
            seen.add(code)
    members = [feature_rows_by_code[code] for code in unique_codes]
    returns = [safe_float(item.get("current_return")) for item in members]
    breadth = sum(1 for value in returns if value > 0.0) / float(max(len(returns), 1))
    median = float(np.nanmedian(returns)) if returns else 0.0
    limit_ratio = sum(1 for item in members if item.get("sealed_limit")) / float(max(len(members), 1))
    drive = clip(0.55 * breadth + 0.45 * clip(limit_ratio / 0.08))
    board_score = (
        45.0 * breadth
        + 25.0 * clip((median + 0.02) / 0.04)
        + 30.0 * clip(limit_ratio / 0.08)
    )
    power_members = [
        item for item in members
        if safe_float(item.get("amount_mean5")) >= MIN_DAILY_AMOUNT
        and int(item.get("ipo_days") or 0) >= MIN_IPO_DAYS
    ]
    ranked = sorted(
        [
            (context_member_power_score(item), str(item.get("code")), str(item.get("name") or item.get("code")))
            for item in power_members
        ],
        key=lambda item: (-item[0], item[1]),
    )
    rank_by_code = {item[1]: index for index, item in enumerate(ranked, 1)}
    score_by_code = {item[1]: item[0] for item in ranked}
    leader = ranked[0] if ranked else (0.0, None, None)
    return {
        "context_type": context_type,
        "context_code": str(context_code or ""),
        "context_name": str(context_name or context_code or "未知上下文"),
        "member_count": len(members),
        "board_state_score": round(board_score, 3),
        "board_breadth": round(breadth, 4),
        "board_median_return": round(median, 4),
        "board_limit_ratio": round(limit_ratio, 4),
        "board_drive": round(drive, 4),
        "leader_code": leader[1],
        "leader_name": leader[2],
        "leader_score": leader[0],
        "_rank_by_code": rank_by_code,
        "_score_by_code": score_by_code,
    }


def project_candidate_context(candidate_code, evidence):
    """把统一板块证据投影到候选；至少5个有效成员才允许参与主上下文竞争。"""
    code = str(candidate_code)
    rank = evidence.get("_rank_by_code", {}).get(code)
    candidate_score = evidence.get("_score_by_code", {}).get(code)
    member_count = int(evidence.get("member_count") or 0)
    if rank is None or candidate_score is None:
        return None
    rank_component = 100.0 * clip((6.0 - rank) / 5.0)
    reliability = 100.0 * clip((member_count - 4.0) / 26.0)
    selection_score = (
        0.50 * safe_float(evidence.get("board_state_score"))
        + 0.25 * safe_float(candidate_score)
        + 0.15 * rank_component
        + 0.10 * reliability
    )
    return {
        "context_type": evidence.get("context_type"),
        "context_code": evidence.get("context_code"),
        "context_name": evidence.get("context_name"),
        "selection_score": round(selection_score, 3),
        "eligible": member_count >= 5,
        "member_count": member_count,
        "board_state_score": evidence.get("board_state_score"),
        "board_breadth": evidence.get("board_breadth"),
        "board_drive": evidence.get("board_drive"),
        "candidate_rank": rank,
        "candidate_power_score": round(candidate_score, 3),
        "leader_code": evidence.get("leader_code"),
        "leader_name": evidence.get("leader_name"),
        "leader_score": evidence.get("leader_score"),
        "leader_gap": round(max(safe_float(evidence.get("leader_score")) - candidate_score, 0.0), 3),
    }


def select_shadow_primary_context(candidate_code, contexts):
    """选择T日最强战术共振标签；它不再直接等于稳定题材锚。"""
    deduplicated = {}
    duplicate_count = 0
    for item in contexts or []:
        if item is None:
            continue
        key = (str(item.get("context_type")), str(item.get("context_code")))
        if key in deduplicated:
            duplicate_count += 1
            if safe_float(item.get("selection_score")) > safe_float(deduplicated[key].get("selection_score")):
                deduplicated[key] = item
        else:
            deduplicated[key] = item
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (
            0 if item.get("eligible") else 1,
            -safe_float(item.get("selection_score")),
            0 if item.get("context_type") == "concept" else 1,
            str(item.get("context_code")),
        ),
    )
    eligible = [item for item in ordered if item.get("eligible")]
    primary = dict(eligible[0]) if eligible else None
    other = [item for item in ordered if primary is None or (
        item.get("context_type"), item.get("context_code")
    ) != (primary.get("context_type"), primary.get("context_code"))]
    other_labels = "；".join(
        "%s:%s(%.1f)" % (
            "概念" if item.get("context_type") == "concept" else "行业",
            item.get("context_name") or item.get("context_code"),
            safe_float(item.get("selection_score")),
        )
        for item in other[:8]
    ) or "-"
    return primary, other_labels, duplicate_count


def shadow_context_key(item):
    if not item:
        return (None, None)
    return (str(item.get("context_type") or ""), str(item.get("context_code") or ""))


def anchor_context_penalty(item):
    """元数据/宽泛标签仍保留展示，但不应轻易成为第二波稳定题材锚。"""
    if not item or item.get("context_type") != "concept":
        return 0.0
    name = str(item.get("context_name") or "")
    hard_meta = (
        "沪股通", "深股通", "融资融券", "转融券标的", "融资标的",
        "沪港通", "深港通",
    )
    event_meta = (
        "中报预增", "年报预增", "一季报预增", "三季报预增", "业绩预增",
    )
    if any(token in name for token in hard_meta):
        return 100.0
    if any(token in name for token in event_meta):
        return 100.0
    if "国企改革" in name or "央企改革" in name:
        return 8.0
    if "摘帽" in name:
        return 5.0
    return 0.0


def select_stable_shadow_anchor(
    candidate_code, contexts, previous_state=None, analysis_position=0,
):
    """以T日以前的连续胜出确认稳定锚，同时保留当日最强战术标签。"""
    tactical, _, duplicate_count = select_shadow_primary_context(candidate_code, contexts)
    deduplicated = {}
    for item in contexts or []:
        if item is None:
            continue
        key = shadow_context_key(item)
        if key not in deduplicated or safe_float(item.get("selection_score")) > safe_float(
            deduplicated[key].get("selection_score")
        ):
            deduplicated[key] = item
    eligible = [dict(item) for item in deduplicated.values() if item.get("eligible")]
    for item in eligible:
        item["anchor_penalty"] = anchor_context_penalty(item)
        item["anchor_adjusted_score"] = round(
            safe_float(item.get("selection_score")) - safe_float(item.get("anchor_penalty")), 3,
        )
    anchor_options = [item for item in eligible if safe_float(item.get("anchor_penalty")) < 100.0]
    anchor_options = sorted(
        anchor_options,
        key=lambda item: (
            -safe_float(item.get("anchor_adjusted_score")),
            0 if item.get("context_type") == "industry" else 1,
            str(item.get("context_code")),
        ),
    )
    meta_excluded_count = len(eligible) - len(anchor_options)
    previous_state = previous_state or {}
    contiguous = bool(
        previous_state
        and int(previous_state.get("analysis_position", -2)) == int(analysis_position) - 1
    )
    previous_key = (
        str(previous_state.get("anchor_type") or ""),
        str(previous_state.get("anchor_code") or ""),
    )
    previous_anchor = next(
        (item for item in anchor_options if shadow_context_key(item) == previous_key), None,
    ) if contiguous else None
    previous_name = previous_state.get("anchor_name") if contiguous else None
    action = "initial"
    reason = "首次建立稳定锚"
    challenger = None
    challenger_gap = None
    pending_key = (None, None)
    pending_days = 0

    if not anchor_options:
        anchor = None
        action = "missing"
        reason = "无至少5个有效成员的非元数据行业/概念上下文"
    elif previous_anchor is None:
        industries = [item for item in anchor_options if item.get("context_type") == "industry"]
        anchor = dict(industries[0] if industries else anchor_options[0])
        if contiguous and previous_key != ("", ""):
            action = "switch"
            reason = "前一稳定锚退出T日有效上下文，回退到可信上下文"
        else:
            reason = "首次优先采用稳定申万行业；概念需连续显著胜出"
    else:
        anchor = dict(previous_anchor)
        action = "hold"
        challengers = [item for item in anchor_options if shadow_context_key(item) != previous_key]
        if challengers:
            challenger = challengers[0]
            challenger_gap = round(
                safe_float(challenger.get("anchor_adjusted_score"))
                - safe_float(previous_anchor.get("anchor_adjusted_score")), 3,
            )
            qualifies = bool(
                challenger_gap >= STABLE_ANCHOR_SWITCH_MARGIN
                and int(challenger.get("candidate_rank") or 999) <= 2
            )
            if qualifies:
                pending_key = shadow_context_key(challenger)
                previous_pending_key = (
                    str(previous_state.get("pending_type") or ""),
                    str(previous_state.get("pending_code") or ""),
                )
                pending_days = (
                    int(previous_state.get("pending_days") or 0) + 1
                    if pending_key == previous_pending_key else 1
                )
                if pending_days >= STABLE_ANCHOR_SWITCH_DAYS:
                    anchor = dict(challenger)
                    action = "switch"
                    reason = "挑战者连续%s日领先稳定锚至少%.1f分，确认切换" % (
                        pending_days, STABLE_ANCHOR_SWITCH_MARGIN,
                    )
                    pending_key = (None, None)
                    pending_days = 0
                else:
                    action = "pending"
                    reason = "挑战者领先%.1f分，但仅连续%s日，稳定锚暂不切换" % (
                        challenger_gap, pending_days,
                    )
            else:
                reason = "稳定锚保持；挑战者未同时满足领先幅度和板块内前二"
        else:
            reason = "稳定锚保持；无其他合格挑战者"

    anchor_key = shadow_context_key(anchor)
    previous_same = bool(contiguous and anchor_key == previous_key and anchor is not None)
    anchor_streak_days = (
        int(previous_state.get("anchor_streak_days") or 0) + 1 if previous_same else (1 if anchor else 0)
    )
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (
            0 if item.get("eligible") else 1,
            -safe_float(item.get("selection_score")),
            str(item.get("context_code")),
        ),
    )
    other_items = [item for item in ordered if shadow_context_key(item) != anchor_key]
    other_labels = "；".join(
        "%s:%s(%.1f%s)" % (
            "概念" if item.get("context_type") == "concept" else "行业",
            item.get("context_name") or item.get("context_code"),
            safe_float(item.get("selection_score")),
            "/元标签" if anchor_context_penalty(item) >= 100.0 else "",
        )
        for item in other_items[:10]
    ) or "-"
    if len(other_items) > 10:
        other_labels += "；其余%s个标签省略" % (len(other_items) - 10)
    tactical_key = shadow_context_key(tactical)
    relation = "同向" if anchor and tactical_key == anchor_key else "分离"
    tracker_state = {
        "analysis_position": int(analysis_position),
        "anchor_type": anchor.get("context_type") if anchor else None,
        "anchor_code": anchor.get("context_code") if anchor else None,
        "anchor_name": anchor.get("context_name") if anchor else None,
        "anchor_streak_days": anchor_streak_days,
        "pending_type": pending_key[0], "pending_code": pending_key[1],
        "pending_days": pending_days,
    }
    diagnostics = {
        "action": action,
        "previous_name": previous_name,
        "challenger_name": challenger.get("context_name") if challenger else None,
        "challenger_gap": challenger_gap,
        "switch_reason": reason,
        "anchor_streak_days": anchor_streak_days,
        "meta_excluded_count": meta_excluded_count,
        "anchor_tactical_relation": relation,
    }
    return anchor, tactical, other_labels, duplicate_count, tracker_state, diagnostics


def classify_pricing_power(candidate_code, leader_code, previous_leader_code, leader_gap):
    """用相邻研究日的同一上下文领先核心描述定价权变化。"""
    candidate_code = str(candidate_code or "")
    leader_code = str(leader_code or "")
    previous_leader_code = str(previous_leader_code or "")
    gap = safe_float(leader_gap)
    if not leader_code:
        return "上下文数据不足"
    if leader_code == candidate_code:
        if previous_leader_code == candidate_code:
            return "旧龙保持定价权"
        if previous_leader_code:
            return "重新夺回/新核心上位"
        return "候选当前居首"
    if previous_leader_code == candidate_code and gap >= 8.0:
        return "新核心替代"
    if gap >= 8.0:
        return "新核心显著领先"
    if gap >= 3.0:
        return "新王竞争"
    return "定价权接近"


def load_candidate_concept_memberships(day, candidates, concept_catalog):
    """逐概念读取T日成分，只保留至少命中一个候选的成分；禁止退回当前成分。"""
    candidate_codes = set(str(row.get("code")) for row in candidates)
    memberships = {}
    ready_count = 0
    failed_count = 0
    total = len(concept_catalog)
    if not candidate_codes:
        return memberships, ready_count, failed_count
    catalog_codes = [str(value) for value in concept_catalog["concept_code"].tolist()]
    cache_directory = os.path.join(os.path.expanduser("~"), "_cache")
    cache_path = os.path.join(
        cache_directory, "signal3_concept_membership_%s.pkl" % day,
    )
    cached_memberships = None
    if ENABLE_CONCEPT_MEMBERSHIP_CACHE and total > 0:
        try:
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)
            if (
                isinstance(cached, dict)
                and cached.get("version") == CONCEPT_MEMBERSHIP_CACHE_VERSION
                and cached.get("trade_date") == str(day)
                and cached.get("concept_codes") == catalog_codes
                and isinstance(cached.get("memberships"), dict)
            ):
                cached_memberships = cached["memberships"]
                print("[signal3题材缓存] 复用%s个概念点时成分：%s" % (
                    len(cached_memberships), cache_path,
                ))
        except Exception:
            cached_memberships = None
    all_memberships = {}
    if cached_memberships is not None:
        all_memberships = cached_memberships
        ready_count = total
    for index, item in enumerate(concept_catalog.to_dict("records"), 1):
        concept_code = str(item.get("concept_code"))
        if cached_memberships is not None:
            stocks = [str(code) for code in all_memberships.get(concept_code, [])]
        else:
            try:
                stocks = [str(code) for code in get_concept_stocks(concept_code, date=day)]
                ready_count += 1
                all_memberships[concept_code] = list(dict.fromkeys(stocks))
            except Exception as exc:
                failed_count += 1
                if failed_count <= 5:
                    print("[signal3题材降级] %s/%s：%s" % (
                        concept_code, item.get("concept_name"), str(exc)[:120],
                    ))
                continue
        if candidate_codes.intersection(stocks):
            memberships[concept_code] = {
                "concept_name": item.get("concept_name") or concept_code,
                "members": list(dict.fromkeys(stocks)),
            }
        if index % 100 == 0 or index == total:
            print("[signal3题材进度] %s；概念成分=%s/%s；失败=%s。" % (
                day, index, total, failed_count,
            ))
    if (
        ENABLE_CONCEPT_MEMBERSHIP_CACHE and cached_memberships is None
        and total > 0 and failed_count == 0 and ready_count == total
    ):
        try:
            if not os.path.isdir(cache_directory):
                os.makedirs(cache_directory)
            handle = tempfile.NamedTemporaryFile(
                mode="wb", dir=cache_directory, delete=False, suffix=".tmp",
            )
            temp_name = handle.name
            try:
                pickle.dump({
                    "version": CONCEPT_MEMBERSHIP_CACHE_VERSION,
                    "trade_date": str(day),
                    "concept_codes": catalog_codes,
                    "memberships": all_memberships,
                }, handle, protocol=2)
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(temp_name, cache_path)
            finally:
                if not handle.closed:
                    handle.close()
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            print("[signal3题材缓存] 已保存：%s" % cache_path)
        except Exception as exc:
            print("[signal3题材降级] 概念成分缓存写入失败，不影响本次影子计算：%s" % str(exc)[:160])
    return memberships, ready_count, failed_count


def build_shadow_theme_contexts(
    day, candidates, feature_rows_by_code, industry_contexts, concept_catalog,
    anchor_tracker, analysis_position,
):
    """生成稳定题材锚、战术共振标签及逐日守恒审计。"""
    candidate_codes = [str(row.get("code")) for row in candidates]
    concept_memberships, ready_count, failed_count = load_candidate_concept_memberships(
        day, candidates, concept_catalog,
    )
    contexts_by_candidate = {code: [] for code in candidate_codes}
    for code in candidate_codes:
        industry = industry_contexts.get(code, {})
        if industry.get("industry_code"):
            evidence = build_context_evidence(
                "industry", industry.get("industry_code"), industry.get("industry_name"),
                industry.get("_board_member_codes", []), feature_rows_by_code,
            )
            projected = project_candidate_context(code, evidence)
            if projected:
                contexts_by_candidate[code].append(projected)
    candidates_with_concept = set()
    candidates_with_eligible_concept = set()
    for concept_code, membership in concept_memberships.items():
        evidence = build_context_evidence(
            "concept", concept_code, membership.get("concept_name"),
            membership.get("members", []), feature_rows_by_code,
        )
        for code in candidate_codes:
            if code not in evidence.get("_rank_by_code", {}):
                continue
            projected = project_candidate_context(code, evidence)
            if projected:
                contexts_by_candidate[code].append(projected)
                candidates_with_concept.add(code)
                if projected.get("eligible"):
                    candidates_with_eligible_concept.add(code)
    result = {}
    duplicate_total = 0
    primary_concept_count = 0
    primary_industry_count = 0
    industry_fallback_count = 0
    missing_primary_count = 0
    stable_anchor_initial_count = 0
    stable_anchor_hold_count = 0
    stable_anchor_switch_count = 0
    stable_anchor_pending_count = 0
    meta_context_excluded_count = 0
    for code in candidate_codes:
        primary, tactical, other_labels, duplicate_count, tracker_state, diagnostics = (
            select_stable_shadow_anchor(
                code, contexts_by_candidate.get(code, []), anchor_tracker.get(code),
                analysis_position,
            )
        )
        anchor_tracker[code] = tracker_state
        duplicate_total += duplicate_count
        meta_context_excluded_count += diagnostics["meta_excluded_count"]
        action = diagnostics.get("action")
        if action == "initial":
            stable_anchor_initial_count += 1
        elif action == "switch":
            stable_anchor_switch_count += 1
        elif action == "pending":
            stable_anchor_pending_count += 1
        elif action == "hold":
            stable_anchor_hold_count += 1
        if primary is None:
            missing_primary_count += 1
            result[code] = {
                "shadow_context_type": None, "shadow_context_code": None,
                "shadow_context_name": None, "shadow_context_score": None,
                "shadow_context_member_count": 0, "shadow_context_breadth": None,
                "shadow_context_drive": None, "shadow_context_candidate_rank": None,
                "shadow_context_leader_code": None, "shadow_context_leader_name": None,
                "shadow_context_leader_gap": None,
                "shadow_context_selection_reason": diagnostics.get("switch_reason"),
                "shadow_anchor_streak_days": 0,
                "shadow_anchor_previous_name": diagnostics.get("previous_name"),
                "shadow_anchor_challenger_name": diagnostics.get("challenger_name"),
                "shadow_anchor_challenger_gap": diagnostics.get("challenger_gap"),
                "shadow_anchor_switch_reason": diagnostics.get("switch_reason"),
                "shadow_tactical_context_type": tactical.get("context_type") if tactical else None,
                "shadow_tactical_context_code": tactical.get("context_code") if tactical else None,
                "shadow_tactical_context_name": tactical.get("context_name") if tactical else None,
                "shadow_tactical_context_score": tactical.get("selection_score") if tactical else None,
                "shadow_anchor_tactical_relation": diagnostics.get("anchor_tactical_relation"),
                "shadow_other_contexts": other_labels,
            }
            continue
        if primary.get("context_type") == "concept":
            primary_concept_count += 1
        else:
            primary_industry_count += 1
            if code not in candidates_with_eligible_concept:
                industry_fallback_count += 1
        result[code] = {
            "shadow_context_type": primary.get("context_type"),
            "shadow_context_code": primary.get("context_code"),
            "shadow_context_name": primary.get("context_name"),
            "shadow_context_score": primary.get("selection_score"),
            "shadow_context_member_count": primary.get("member_count"),
            "shadow_context_breadth": primary.get("board_breadth"),
            "shadow_context_drive": primary.get("board_drive"),
            "shadow_context_candidate_rank": primary.get("candidate_rank"),
            "shadow_context_leader_code": primary.get("leader_code"),
            "shadow_context_leader_name": primary.get("leader_name"),
            "shadow_context_leader_gap": primary.get("leader_gap"),
            "shadow_context_selection_reason": (
                "%s；从%s个点时上下文选择；锚分%.1f；结构分%.1f；候选锚内第%s；有效成员%s"
                % (
                    diagnostics.get("switch_reason"),
                    len(contexts_by_candidate.get(code, [])),
                    safe_float(primary.get("selection_score")),
                    safe_float(primary.get("board_state_score")),
                    primary.get("candidate_rank"), primary.get("member_count"),
                )
            ),
            "shadow_anchor_streak_days": diagnostics.get("anchor_streak_days"),
            "shadow_anchor_previous_name": diagnostics.get("previous_name"),
            "shadow_anchor_challenger_name": diagnostics.get("challenger_name"),
            "shadow_anchor_challenger_gap": diagnostics.get("challenger_gap"),
            "shadow_anchor_switch_reason": diagnostics.get("switch_reason"),
            "shadow_tactical_context_type": tactical.get("context_type") if tactical else None,
            "shadow_tactical_context_code": tactical.get("context_code") if tactical else None,
            "shadow_tactical_context_name": tactical.get("context_name") if tactical else None,
            "shadow_tactical_context_score": tactical.get("selection_score") if tactical else None,
            "shadow_anchor_tactical_relation": diagnostics.get("anchor_tactical_relation"),
            "shadow_other_contexts": other_labels,
        }
    conserved_count = primary_concept_count + primary_industry_count + missing_primary_count
    action_conserved_count = (
        stable_anchor_initial_count + stable_anchor_hold_count
        + stable_anchor_switch_count + stable_anchor_pending_count + missing_primary_count
    )
    status = (
        "passed" if len(concept_catalog) > 0 and failed_count == 0
        and missing_primary_count == 0 and duplicate_total == 0
        and conserved_count == len(candidate_codes)
        and action_conserved_count == len(candidate_codes) else "degraded"
    )
    audit = {
        "trade_date": day,
        "concept_universe_count": len(concept_catalog),
        "concept_membership_ready": ready_count,
        "concept_membership_failed": failed_count,
        "candidate_count": len(candidate_codes),
        "candidates_with_concept": len(candidates_with_concept),
        "primary_concept_count": primary_concept_count,
        "primary_industry_count": primary_industry_count,
        "industry_fallback_count": industry_fallback_count,
        "missing_primary_count": missing_primary_count,
        "primary_context_conservation": "%s/%s" % (conserved_count, len(candidate_codes)),
        "duplicate_candidate_contexts": duplicate_total,
        "stable_anchor_initial_count": stable_anchor_initial_count,
        "stable_anchor_hold_count": stable_anchor_hold_count,
        "stable_anchor_switch_count": stable_anchor_switch_count,
        "stable_anchor_pending_count": stable_anchor_pending_count,
        "stable_anchor_action_conservation": "%s/%s" % (
            action_conserved_count, len(candidate_codes),
        ),
        "meta_context_excluded_count": meta_context_excluded_count,
        "status": status,
        "theme_context_contract": SIGNAL3_THEME_CONTEXT_CONTRACT,
    }
    return result, audit


def build_market_context(day, rows, previous_amount=None):
    returns = [safe_float(row.get("current_return")) for row in rows]
    advance = sum(1 for value in returns if value > 0.0) / float(max(len(returns), 1))
    limit_up = sum(1 for row in rows if row.get("sealed_limit"))
    limit_down = sum(1 for row in rows if row.get("sealed_down"))
    market_amount = sum(safe_float(row.get("amount")) for row in rows)
    if previous_amount is None:
        previous_amount = sum(safe_float(row.get("previous_amount")) for row in rows)
    amount_ratio = market_amount / previous_amount if previous_amount and previous_amount > 0 else 1.0
    cycle = clip(
        45.0 * advance
        + 25.0 * clip(limit_up / 80.0)
        + 15.0 * clip(1.0 - limit_down / 20.0)
        + 15.0 * clip(amount_ratio / 1.10),
        0.0, 100.0,
    )
    if cycle >= 65.0 and limit_down <= 5:
        state = "主升"
    elif cycle >= 50.0:
        state = "修复"
    elif cycle >= 38.0:
        state = "强分歧"
    else:
        state = "退潮"
    return {
        "trade_date": day,
        "market_state": state,
        "market_cycle_score": round(cycle, 3),
        "market_advance_ratio": round(advance, 4),
        "market_limit_up_count": limit_up,
        "market_limit_down_count": limit_down,
        "market_amount": market_amount,
        "market_amount_ratio": round(amount_ratio, 4),
        "sample_size": len(rows),
    }


def build_feature_rows(day, securities, raw_map, adjusted_map):
    rows = []
    names = {}
    starts = {}
    availability = {
        "paused": [], "no_trade": [], "data_incomplete": [],
    }
    for code, security in securities.iterrows():
        name = str(security.get("display_name", code))
        if "ST" in name or "*" in name or "退" in name:
            continue
        names[str(code)] = name
        starts[str(code)] = normalize_date(security.get("start_date") or day)
    for code in sorted(names):
        trade_status = inspect_signal_day_trade_status(
            raw_map.get(code), adjusted_map.get(code), day,
        )
        status = trade_status.get("status")
        if status != "ready":
            availability.setdefault(status, []).append("%s/%s" % (code, names[code]))
            continue
        row = build_stock_snapshot(
            code, names[code], starts[code], day, raw_map.get(code), adjusted_map.get(code),
            trade_status=trade_status,
        )
        if row:
            rows.append(row)
        else:
            availability["data_incomplete"].append("%s/%s" % (code, names[code]))
    audit = {
        "eligible_count": len(names),
        "effective_count": len(rows),
        "paused_count": len(availability["paused"]),
        "no_trade_count": len(availability["no_trade"]),
        "data_incomplete_count": len(availability["data_incomplete"]),
        "paused_codes": ",".join(availability["paused"]) or "-",
        "no_trade_codes": ",".join(availability["no_trade"]) or "-",
        "data_incomplete_samples": ",".join(availability["data_incomplete"][:10]) or "-",
    }
    audit["excluded_count"] = (
        audit["paused_count"] + audit["no_trade_count"] + audit["data_incomplete_count"]
    )
    if audit["effective_count"] + audit["excluded_count"] != audit["eligible_count"]:
        raise RuntimeError("signal3信号日可交易性守恒失败：%s" % day)
    return rows, audit


def eligible_security_count(securities):
    """返回与 build_feature_rows 同口径的点时名称初筛分母。"""
    count = 0
    for code, security in securities.iterrows():
        name = str(security.get("display_name", code))
        if "ST" in name or "*" in name or "退" in name:
            continue
        count += 1
    return count


def enrich_and_observe(
    day, candidates, feature_rows, market, turnover_map,
    continuity_tracker, context_leader_tracker, anchor_tracker,
    analysis_position, concept_catalog,
):
    by_code = {row["code"]: row for row in feature_rows}
    industry = build_industry_context(day, candidates, by_code)
    shadow_contexts, context_audit = build_shadow_theme_contexts(
        day, candidates, by_code, industry, concept_catalog,
        anchor_tracker, analysis_position,
    )
    top_score = safe_float(candidates[0].get("first_wave_score")) if candidates else 0.0
    observations = []
    selected_context_leaders = {}
    for candidate in candidates:
        snapshot = dict(candidate)
        snapshot.update(industry.get(candidate["code"], {}))
        snapshot.update(shadow_contexts.get(candidate["code"], {}))
        context_key = (
            snapshot.get("shadow_context_type"), snapshot.get("shadow_context_code"),
        )
        prior_context = context_leader_tracker.get(context_key, {}) if all(context_key) else {}
        contiguous_context = bool(
            prior_context
            and int(prior_context.get("analysis_position", -2)) == int(analysis_position) - 1
        )
        previous_leader = prior_context.get("leader_code") if contiguous_context else None
        snapshot["shadow_pricing_power_status"] = classify_pricing_power(
            candidate.get("code"), snapshot.get("shadow_context_leader_code"),
            previous_leader, snapshot.get("shadow_context_leader_gap"),
        )
        if all(context_key):
            selected_context_leaders[context_key] = snapshot.get("shadow_context_leader_code")
        snapshot.update(market)
        snapshot["turnover_rate"] = turnover_map.get((candidate["code"], day))
        snapshot["relative_market_return"] = safe_float(snapshot.get("current_return")) - float(
            np.nanmedian([safe_float(row.get("current_return")) for row in feature_rows])
        )
        snapshot["top_score_gap"] = top_score - safe_float(snapshot.get("first_wave_score"))
        snapshot["replaced_by_new_leader"] = bool(
            snapshot.get("candidate_rank", 99) > 3 and snapshot["top_score_gap"] >= 8.0
        )
        previous = continuity_tracker.get(candidate["code"])
        contiguous = bool(
            previous is not None
            and int(previous.get("analysis_position", -2)) == int(analysis_position) - 1
        )
        snapshot["prior_observation_date"] = previous.get("trade_date") if previous else None
        snapshot["previous_observed_state"] = previous.get("state") if previous else None
        snapshot["prior_state"] = previous.get("state") if contiguous else None
        snapshot["candidate_streak_days"] = (
            int(previous.get("candidate_streak_days", 0)) + 1 if contiguous else 1
        )
        snapshot["prior_state_streak_days"] = (
            int(previous.get("state_streak_days", 0)) if contiguous else 0
        )
        snapshot["observation_gap_days"] = (
            max(int(analysis_position) - int(previous.get("analysis_position", analysis_position)) - 1, 0)
            if previous else 0
        )
        row = build_second_wave_observation(snapshot)
        continuity_tracker[candidate["code"]] = {
            "trade_date": day,
            "analysis_position": int(analysis_position),
            "state": row["second_wave_state"],
            "candidate_streak_days": row["candidate_streak_days"],
            "state_streak_days": row["state_streak_days"],
        }
        observations.append(row)
    for context_key, leader_code in selected_context_leaders.items():
        context_leader_tracker[context_key] = {
            "analysis_position": int(analysis_position),
            "leader_code": leader_code,
        }
    return observations, context_audit


def audit_observation_continuity(observation_rows, analysis_dates):
    """独立复核区间日数、合法迁移与候选交易状态，不用状态机自身结论兜底。"""
    positions = {normalize_date(day): index for index, day in enumerate(analysis_dates)}
    errors = 0
    duplicate_rows = 0
    gap_reentries = 0
    illegal_transitions = 0
    trade_status_violations = 0
    seen = set()
    by_code = {}
    for raw_row in observation_rows:
        row = dict(raw_row)
        day = normalize_date(row.get("trade_date"))
        code = str(row.get("code") or "")
        key = (day, code)
        if key in seen:
            duplicate_rows += 1
            errors += 1
        seen.add(key)
        if day not in positions or not code:
            errors += 1
            continue
        if row.get("signal_day_trade_status") != "ready":
            trade_status_violations += 1
            errors += 1
        by_code.setdefault(code, []).append(row)
    for rows in by_code.values():
        rows.sort(key=lambda item: positions[normalize_date(item.get("trade_date"))])
        previous = None
        for row in rows:
            day = normalize_date(row.get("trade_date"))
            position = positions[day]
            if previous is None:
                expected_gap = 0
                expected_candidate_streak = 1
                expected_prior_state = None
                expected_state_streak = 1
            else:
                previous_day = normalize_date(previous.get("trade_date"))
                prior_position = positions[previous_day]
                expected_gap = max(position - prior_position - 1, 0)
                contiguous = expected_gap == 0
                if not contiguous:
                    gap_reentries += 1
                expected_candidate_streak = (
                    int(previous.get("candidate_streak_days") or 0) + 1 if contiguous else 1
                )
                expected_prior_state = previous.get("second_wave_state") if contiguous else None
                expected_state_streak = (
                    int(previous.get("state_streak_days") or 0) + 1
                    if contiguous and previous.get("second_wave_state") == row.get("second_wave_state")
                    else 1
                )
            checks = (
                int(row.get("observation_gap_days") or 0) == expected_gap,
                int(row.get("candidate_streak_days") or 0) == expected_candidate_streak,
                row.get("prior_state") == expected_prior_state,
                int(row.get("state_streak_days") or 0) == expected_state_streak,
            )
            errors += sum(1 for passed in checks if not passed)
            if not is_legal_state_transition(
                expected_prior_state, row.get("second_wave_state"),
            ):
                illegal_transitions += 1
                errors += 1
            previous = row
    return {
        "status": "passed" if errors == 0 else "failed",
        "breaks": errors,
        "duplicate_rows": duplicate_rows,
        "gap_reentries": gap_reentries,
        "illegal_transitions": illegal_transitions,
        "trade_status_violations": trade_status_violations,
    }


def audit_shadow_anchor_stability(observation_rows, analysis_dates):
    """独立统计连续候选日的稳定锚切换率；战术标签变化不算锚切换。"""
    positions = {normalize_date(day): index for index, day in enumerate(analysis_dates)}
    by_code = {}
    divergence_count = 0
    for raw_row in observation_rows:
        row = dict(raw_row)
        code = str(row.get("code") or "")
        day = normalize_date(row.get("trade_date"))
        if not code or day not in positions:
            continue
        by_code.setdefault(code, []).append(row)
        anchor_key = (row.get("shadow_context_type"), row.get("shadow_context_code"))
        tactical_key = (
            row.get("shadow_tactical_context_type"), row.get("shadow_tactical_context_code"),
        )
        if all(anchor_key) and all(tactical_key) and anchor_key != tactical_key:
            divergence_count += 1
    transitions = 0
    switches = 0
    for rows in by_code.values():
        rows.sort(key=lambda item: positions[normalize_date(item.get("trade_date"))])
        previous = None
        for row in rows:
            if previous is not None:
                current_position = positions[normalize_date(row.get("trade_date"))]
                previous_position = positions[normalize_date(previous.get("trade_date"))]
                previous_key = (
                    previous.get("shadow_context_type"), previous.get("shadow_context_code"),
                )
                current_key = (
                    row.get("shadow_context_type"), row.get("shadow_context_code"),
                )
                if current_position == previous_position + 1 and all(previous_key) and all(current_key):
                    transitions += 1
                    if previous_key != current_key:
                        switches += 1
            previous = row
    switch_rate = switches / float(max(transitions, 1))
    if transitions == 0:
        status = "insufficient"
    elif switch_rate <= MAX_STABLE_ANCHOR_SWITCH_RATE:
        status = "passed"
    else:
        status = "unstable"
    return {
        "status": status,
        "transition_count": transitions,
        "switch_count": switches,
        "switch_rate": round(switch_rate, 6),
        "tactical_divergence_count": divergence_count,
    }


def localize_report_frame(frame, columns=None):
    """生成中文展示副本；不修改内部 DataFrame、列合同或字段顺序。"""
    if frame is None:
        return None
    view = frame.copy()
    if columns is not None:
        view = view.reindex(columns=list(columns))
    return view.rename(columns=REPORT_COLUMN_LABELS)


def build_latest_report_frame(frame):
    """首表只投影核心阅读字段，完整研究列仍保留在可横向滚动的区间明细表。"""
    if frame is None:
        return None
    view = frame.copy()
    if view.empty:
        return localize_report_frame(view, LATEST_DISPLAY_COLUMNS)
    view["target_display"] = view.apply(
        lambda row: "%s（%s）" % (row.get("name") or "-", row.get("code") or "-"),
        axis=1,
    )
    view["shadow_context_display"] = view.apply(
        lambda row: "%s:%s（%s）｜锚%s日" % (
            "概念" if row.get("shadow_context_type") == "concept" else "行业",
            row.get("shadow_context_name") or "-", row.get("shadow_context_code") or "-",
            max(int(safe_float(row.get("shadow_anchor_streak_days"), 0.0)), 0),
        ),
        axis=1,
    )
    view["shadow_tactical_context_display"] = view.apply(
        lambda row: "%s:%s（%s）" % (
            "概念" if row.get("shadow_tactical_context_type") == "concept" else "行业",
            row.get("shadow_tactical_context_name") or "-",
            row.get("shadow_tactical_context_code") or "-",
        ),
        axis=1,
    )
    gate_fields = (
        ("首", "first_wave_gate"), ("题", "theme_vitality_gate"),
        ("死", "death_test_gate"), ("筹", "chip_restructure_gate"),
        ("市", "market_environment_gate"), ("稀", "leader_scarcity_gate"),
    )
    view["gate_summary"] = view.apply(
        lambda row: "｜".join("%s:%s" % (label, row.get(field) or "-") for label, field in gate_fields),
        axis=1,
    )
    view["event_timeline"] = view.apply(
        lambda row: "峰:%s｜死:%s｜距峰:%s日｜候选:%s日｜状态:%s日｜近20无交易:%s日" % (
            row.get("first_wave_peak_date") or "-",
            row.get("death_test_date") or "-",
            int(safe_float(row.get("days_since_peak"), 0.0)),
            max(int(safe_float(row.get("candidate_streak_days"), 1.0)), 1),
            max(int(safe_float(row.get("state_streak_days"), 1.0)), 1),
            max(int(safe_float(row.get("recent_nontrading_days_20"), 0.0)), 0),
        ),
        axis=1,
    )
    return localize_report_frame(view, LATEST_DISPLAY_COLUMNS)


def dataframe_html(frame, title, section_class="", wrap_class="table-wrap", table_class="data-table"):
    if frame is None or frame.empty:
        table = '<p class="empty">无数据</p>'
    else:
        # 聚宽 pandas 0.23 默认会截断长合同/模型文本；HTML审计必须保留完整值。
        with pd.option_context("display.max_colwidth", 10000):
            table = frame.to_html(index=False, border=0, escape=True, classes=table_class)
    return '<section class="%s"><h2>%s</h2><div class="%s">%s</div></section>' % (
        html.escape(section_class), html.escape(title), html.escape(wrap_class), table,
    )


def build_html_report(panel, analysis_dates):
    sections = [
        dataframe_html(
            build_latest_report_frame(panel.get("second_wave_latest")),
            "最新交易日第二波观察", section_class="latest-section",
            wrap_class="table-wrap latest-wrap", table_class="data-table latest-table",
        ),
        dataframe_html(
            localize_report_frame(panel.get("second_wave_history"), SECOND_WAVE_COLUMNS),
            "区间第二波状态明细", section_class="history-section",
            wrap_class="table-wrap history-wrap", table_class="data-table history-table",
        ),
        dataframe_html(
            localize_report_frame(panel.get("market_context"), MARKET_CONTEXT_COLUMNS),
            "市场环境",
        ),
        dataframe_html(
            localize_report_frame(panel.get("theme_context_audit"), THEME_CONTEXT_AUDIT_COLUMNS),
            "稳定题材锚与战术上下文审计",
        ),
        dataframe_html(
            localize_report_frame(panel.get("future_label_daily"), FUTURE_LABEL_COLUMNS),
            "未来标签（影子采集；不回写T日）",
        ),
        dataframe_html(
            localize_report_frame(panel.get("signal_day_coverage"), SIGNAL_DAY_COVERAGE_COLUMNS),
            "信号日可交易性与覆盖率明细",
        ),
        dataframe_html(
            localize_report_frame(panel.get("data_quality"), QUALITY_COLUMNS),
            "数据质量与降级",
        ),
        dataframe_html(
            localize_report_frame(panel.get("run_audit"), AUDIT_COLUMNS),
            "运行审计",
        ),
    ]
    start_day = analysis_dates[0]
    end_day = analysis_dates[-1]
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="signal3-start-date" content="{start}"><meta name="signal3-end-date" content="{end}">
<meta name="signal3-model-version" content="{model}"><meta name="signal3-state-contract" content="{state_contract}">
<meta name="signal3-gate-contract" content="{gate_contract}">
<meta name="signal3-theme-context-contract" content="{theme_context_contract}">
<meta name="signal3-future-label-contract" content="{future_label_contract}"><title>signal3 第二波研究候选报告</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f4f1e8;color:#243126;overflow-x:hidden}}
main{{max-width:2100px;margin:auto;padding:24px}}header,section{{background:#fffdf7;border:1px solid #d9d2c2;border-radius:12px;padding:20px;margin-bottom:18px}}
h1,h2{{margin-top:0}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #ded8ca;padding:8px;text-align:left;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:break-word}}th{{background:#e8eadf;position:sticky;top:0}}.empty{{color:#766f61}}
.latest-section{{width:auto}}.latest-wrap{{overflow:visible}}.latest-table{{width:100%;table-layout:fixed;font-size:12px}}.latest-table th,.latest-table td{{min-width:0;padding:7px 5px}}
.latest-table th:nth-child(1){{width:5%}}.latest-table th:nth-child(2){{width:3%}}.latest-table th:nth-child(3){{width:8%}}.latest-table th:nth-child(5){{width:8%}}.latest-table th:nth-child(7){{width:13%}}.latest-table th:nth-child(10){{width:15%}}.latest-table th:nth-child(15){{width:10%}}
.history-wrap{{overflow-x:auto;overflow-y:visible;-webkit-overflow-scrolling:touch;border:1px solid #ded8ca;border-radius:8px}}
.history-table{{width:max-content;min-width:9800px;table-layout:auto;font-size:12px}}
.history-table th,.history-table td{{min-width:112px;max-width:360px;white-space:normal;overflow-wrap:break-word;word-break:normal;padding:7px 8px}}
.history-table th:nth-child(1),.history-table td:nth-child(1){{position:sticky;left:0;z-index:3;min-width:96px;background:#fffdf7}}
.history-table th:nth-child(2),.history-table td:nth-child(2){{position:sticky;left:112px;z-index:3;min-width:122px;background:#fffdf7}}
.history-table th:nth-child(3),.history-table td:nth-child(3){{position:sticky;left:250px;z-index:3;min-width:90px;background:#fffdf7}}
.history-table th:nth-child(-n+3){{z-index:5;background:#e8eadf}}
.history-section h2::after{{content:"（横向滚动查看完整字段；交易日、代码、名称固定）";font-size:13px;font-weight:400;color:#766f61;margin-left:8px}}
@media(max-width:1100px){{main{{padding:12px}}header,section{{padding:12px}}.latest-table{{font-size:10px}}.latest-table th,.latest-table td{{padding:5px 3px}}}}</style></head>
<body><main><header><h1>signal3 高辨识度个股第二波研究</h1><p>{start} 至 {end} ｜ {model} ｜ candidate</p>
<p>{status}</p><p><b>题材影子合同：</b>{theme_context_contract}；稳定题材锚用于跨日连续解释，
当日战术共振单独展示；两者均不改写六道门和状态。</p>
<p><b>未来标签合同：</b>{future_label_contract}；只在 T 日状态终检后读取已到期结果，
成熟、部分成熟、待成熟和数据不完整分别保留，不回写历史信号。</p>
<p><b>边界：</b>研究信号，不是交易指令；未来标签只作后验校准；100分只作影子解释。</p></header>{sections}</main></body></html>'''.format(
        start=start_day, end=end_day, model=SIGNAL3_MODEL_VERSION,
        state_contract=SIGNAL3_STATE_CONTRACT, gate_contract=SIGNAL3_GATE_CONTRACT,
        theme_context_contract=SIGNAL3_THEME_CONTEXT_CONTRACT,
        future_label_contract=SIGNAL3_FUTURE_LABEL_CONTRACT,
        status=html.escape(SIGNAL3_IMPLEMENTATION_STATUS), sections="".join(sections),
    )


def save_candidate_html(html_text, analysis_dates):
    if len(analysis_dates) == 1:
        filename = "signal3_today_review_%s_candidate.html" % analysis_dates[0]
    else:
        filename = "signal3_range_review_%s_%s_candidate.html" % (analysis_dates[0], analysis_dates[-1])
    target = os.path.abspath(filename)
    directory = os.path.dirname(target)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp")
    temp_name = handle.name
    try:
        handle.write(html_text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temp_name, target)
    finally:
        if not handle.closed:
            handle.close()
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    digest = hashlib.sha256(html_text.encode("utf-8")).hexdigest()
    print("[signal3候选报告] %s；SHA-256=%s" % (target, digest))
    return target, digest


def run_signal3_research(trade_date=None, start_date=None, end_date=None):
    started = time.time()
    analysis_dates = resolve_research_dates(trade_date, start_date, end_date)
    end_day = analysis_dates[-1]
    label_price_end = resolve_future_label_price_end(end_day) if ENABLE_FUTURE_LABELS else end_day
    label_calendar = [normalize_date(value) for value in get_trade_days(
        start_date=analysis_dates[0], end_date=label_price_end,
    )]
    history_days = [normalize_date(value) for value in get_trade_days(
        end_date=end_day, count=LOOKBACK_TRADE_DAYS + len(analysis_dates) + 5,
    )]
    history_start = history_days[0]
    securities_by_day = {day: get_all_securities(["stock"], date=day) for day in analysis_dates}
    all_codes = sorted(set(
        str(code) for frame in securities_by_day.values() for code in frame.index
    ))
    print("[signal3启动] model=%s；区间=%s至%s；股票=%s；profile=%s；题材影子合同=%s；"
          "未来标签合同=%s；标签行情截止=%s。" % (
        SIGNAL3_MODEL_VERSION, analysis_dates[0], end_day, len(all_codes), RESOURCE_PROFILE,
        SIGNAL3_THEME_CONTEXT_CONTRACT, SIGNAL3_FUTURE_LABEL_CONTRACT, label_price_end,
    ))
    raw_map, adjusted_map, price_coverage = load_price_history(
        all_codes, history_start, label_price_end,
    )
    features_by_day = {}
    candidates_by_day = {}
    concept_catalogs_by_day = {}
    market_rows = []
    signal_day_coverage_rows = []
    previous_amount = None
    candidate_union = set()
    for day in analysis_dates:
        rows, availability = build_feature_rows(
            day, securities_by_day[day], raw_map, adjusted_map,
        )
        features_by_day[day] = rows
        eligible_count = availability["eligible_count"]
        effective_coverage = len(rows) / float(max(eligible_count, 1))
        total_coverage = len(rows) / float(max(len(securities_by_day[day]), 1))
        signal_day_coverage_rows.append({
            "trade_date": day,
            "universe_count": len(securities_by_day[day]),
            "eligible_count": eligible_count,
            "effective_count": len(rows),
            "paused_count": availability["paused_count"],
            "no_trade_count": availability["no_trade_count"],
            "data_incomplete_count": availability["data_incomplete_count"],
            "excluded_count": availability["excluded_count"],
            "paused_codes": availability["paused_codes"],
            "no_trade_codes": availability["no_trade_codes"],
            "data_incomplete_samples": availability["data_incomplete_samples"],
            "effective_coverage": effective_coverage,
            "total_coverage": total_coverage,
        })
        if effective_coverage < MIN_PRICE_COVERAGE:
            raise RuntimeError("signal3信号日%s有效覆盖率%.1f%%低于%.1f%%" % (
                day, effective_coverage * 100.0, MIN_PRICE_COVERAGE * 100.0,
            ))
        market = build_market_context(day, rows, previous_amount)
        market_rows.append(market)
        previous_amount = market["market_amount"]
        candidates = discover_candidates(rows)
        candidates_by_day[day] = candidates
        concept_catalog, concept_catalog_status = load_concept_catalog(day)
        concept_catalogs_by_day[day] = concept_catalog
        candidate_union.update(row["code"] for row in candidates)
        print("[signal3候选] %s；有效股票=%s/%s(%.1f%%)；宽松候选=%s；市场=%s；概念目录=%s/%s。" % (
            day, len(rows), eligible_count, effective_coverage * 100.0,
            len(candidates), market["market_state"], len(concept_catalog), concept_catalog_status,
        ))
        print("[signal3可交易性] %s；停牌=%s[%s]；无成交=%s[%s]；数据不完整=%s；守恒=%s/%s。" % (
            day, availability["paused_count"], availability["paused_codes"],
            availability["no_trade_count"], availability["no_trade_codes"],
            availability["data_incomplete_count"],
            availability["effective_count"] + availability["excluded_count"],
            availability["eligible_count"],
        ))
    turnover_map, turnover_status = load_candidate_valuations(sorted(candidate_union), analysis_dates)
    continuity_tracker = {}
    context_leader_tracker = {}
    anchor_tracker = {}
    observation_rows = []
    theme_context_audit_rows = []
    market_by_day = {row["trade_date"]: row for row in market_rows}
    for analysis_position, day in enumerate(analysis_dates):
        day_observations, context_audit = enrich_and_observe(
            day, candidates_by_day[day], features_by_day[day], market_by_day[day],
            turnover_map, continuity_tracker, context_leader_tracker, anchor_tracker,
            analysis_position, concept_catalogs_by_day[day],
        )
        observation_rows.extend(day_observations)
        theme_context_audit_rows.append(context_audit)
        print("[signal3题材审计] %s；状态=%s；概念稳定锚=%s；行业稳定锚=%s；"
              "新建/保持/待确认/切换=%s/%s/%s/%s；动作守恒=%s；锚排除元标签=%s；缺失=%s；"
              "守恒=%s；重复=%s；概念失败=%s。" % (
            day, context_audit["status"], context_audit["primary_concept_count"],
            context_audit["primary_industry_count"],
            context_audit["stable_anchor_initial_count"], context_audit["stable_anchor_hold_count"],
            context_audit["stable_anchor_pending_count"], context_audit["stable_anchor_switch_count"],
            context_audit["stable_anchor_action_conservation"],
            context_audit["meta_context_excluded_count"], context_audit["missing_primary_count"],
            context_audit["primary_context_conservation"],
            context_audit["duplicate_candidate_contexts"], context_audit["concept_membership_failed"],
        ))
    continuity_audit = audit_observation_continuity(observation_rows, analysis_dates)
    if continuity_audit["status"] != "passed":
        raise RuntimeError("signal3区间状态连续性审计失败：%s处" % continuity_audit["breaks"])
    print("[signal3连续性] 状态=%s；错误=%s；非法迁移=%s；交易状态违规=%s；重新入池=%s；重复=%s。" % (
        continuity_audit["status"], continuity_audit["breaks"],
        continuity_audit["illegal_transitions"], continuity_audit["trade_status_violations"],
        continuity_audit["gap_reentries"], continuity_audit["duplicate_rows"],
    ))
    anchor_stability = audit_shadow_anchor_stability(observation_rows, analysis_dates)
    print("[signal3稳定锚终检] 状态=%s；可比较迁移=%s；切换=%s；切换率=%.1f%%；"
          "稳定锚/战术分离行=%s；阈值<=%.1f%%。" % (
              anchor_stability["status"], anchor_stability["transition_count"],
              anchor_stability["switch_count"], anchor_stability["switch_rate"] * 100.0,
              anchor_stability["tactical_divergence_count"],
              MAX_STABLE_ANCHOR_SWITCH_RATE * 100.0,
          ))
    history_frame = pd.DataFrame(observation_rows, columns=SECOND_WAVE_COLUMNS)
    latest_frame = history_frame[history_frame["trade_date"] == end_day].copy() if not history_frame.empty else pd.DataFrame(columns=SECOND_WAVE_COLUMNS)
    future_rows = build_future_label_rows(
        observation_rows, raw_map, adjusted_map, label_calendar,
    ) if ENABLE_FUTURE_LABELS else []
    future_frame = pd.DataFrame(future_rows, columns=FUTURE_LABEL_COLUMNS)
    future_status_counts = {
        status: sum(1 for row in future_rows if row.get("label_status") == status)
        for status in FUTURE_LABEL_STATUSES
    }
    if ENABLE_FUTURE_LABELS and len(future_rows) != len(observation_rows):
        raise RuntimeError("未来标签行守恒失败：%s/%s" % (
            len(future_rows), len(observation_rows),
        ))
    print("[signal3未来标签] 合同=%s；行情截止=%s；行=%s/%s；"
          "成熟/部分/待成熟/不完整=%s/%s/%s/%s；不回写T日。" % (
              SIGNAL3_FUTURE_LABEL_CONTRACT, label_price_end,
              len(future_rows), len(observation_rows),
              future_status_counts["matured"], future_status_counts["partial"],
              future_status_counts["pending"], future_status_counts["data_incomplete"],
          ))
    minimum_signal_coverage = min(
        [item["effective_coverage"] for item in signal_day_coverage_rows] or [0.0]
    )
    latest_signal_coverage = signal_day_coverage_rows[-1]
    paused_excluded = sum(item["paused_count"] for item in signal_day_coverage_rows)
    no_trade_excluded = sum(item["no_trade_count"] for item in signal_day_coverage_rows)
    incomplete_excluded = sum(item["data_incomplete_count"] for item in signal_day_coverage_rows)
    theme_context_failures = sum(item["concept_membership_failed"] for item in theme_context_audit_rows)
    theme_context_missing = sum(item["missing_primary_count"] for item in theme_context_audit_rows)
    theme_context_duplicates = sum(item["duplicate_candidate_contexts"] for item in theme_context_audit_rows)
    theme_context_status = (
        "passed" if all(item.get("status") == "passed" for item in theme_context_audit_rows)
        and theme_context_failures == 0 and theme_context_missing == 0
        and theme_context_duplicates == 0 else "degraded"
    )
    print("[signal3题材终检] 结构状态=%s；稳定锚状态=%s；概念失败=%s；"
          "稳定锚缺失=%s；重复=%s；active_state_contract=%s；"
          "active_gate_contract=%s；影子不改写。" % (
              theme_context_status, anchor_stability["status"],
              theme_context_failures, theme_context_missing,
              theme_context_duplicates, SIGNAL3_STATE_CONTRACT, SIGNAL3_GATE_CONTRACT,
          ))
    quality = pd.DataFrame([
        {"item": "未复权历史代码覆盖", "status": "ready", "detail": "%.1f%%；交易状态口径" % (price_coverage["raw_history_coverage"] * 100.0)},
        {"item": "前复权历史代码覆盖", "status": "ready", "detail": "%.1f%%；趋势信号口径" % (price_coverage["adjusted_history_coverage"] * 100.0)},
        {"item": "双口径历史代码覆盖", "status": "ready", "detail": "%.1f%%；阈值%.1f%%" % (price_coverage["dual_history_coverage"] * 100.0, MIN_PRICE_COVERAGE * 100.0)},
        {"item": "信号日有效覆盖", "status": "ready", "detail": "区间最低%.1f%%；末日%s/%s；总股票口径%.1f%%" % (
            minimum_signal_coverage * 100.0,
            latest_signal_coverage["effective_count"], latest_signal_coverage["eligible_count"],
            latest_signal_coverage["total_coverage"] * 100.0,
        )},
        {"item": "区间状态连续性", "status": continuity_audit["status"], "detail": "错误%s；重新入池%s；重复%s" % (
            continuity_audit["breaks"], continuity_audit["gap_reentries"], continuity_audit["duplicate_rows"],
        )},
        {"item": "合法状态迁移", "status": "ready" if continuity_audit["illegal_transitions"] == 0 else "failed", "detail": "非法迁移%s；未约束状态只作诊断，最终状态必须经过迁移门" % continuity_audit["illegal_transitions"]},
        {"item": "信号日停牌/无成交", "status": "ready", "detail": "区间停牌剔除%s；无成交剔除%s；数据不完整剔除%s；候选交易状态违规%s" % (
            paused_excluded, no_trade_excluded, incomplete_excluded,
            continuity_audit["trade_status_violations"],
        )},
        {"item": "历史无交易日", "status": "candidate", "detail": "候选明细记录近20/60日停牌或无成交日数；当前只作风险证据，不读取未来复牌表现"},
        {"item": "候选历史换手", "status": turnover_status, "detail": "%s条" % len(turnover_map)},
        {"item": "市场环境", "status": "candidate", "detail": "首轮为 signal3 独立代理口径，不宣称等同 signal2 市场周期"},
        {"item": "稳定题材锚结构", "status": theme_context_status, "detail": (
            "点时行业与概念构造稳定锚；概念读取失败%s；稳定锚缺失%s；重复%s；"
            "当前不改写v3.0.3六道门和状态" % (
                theme_context_failures, theme_context_missing, theme_context_duplicates,
            )
        )},
        {"item": "稳定题材锚连续性", "status": anchor_stability["status"], "detail": (
            "连续候选日可比较迁移%s；切换%s；切换率%.1f%%（阈值<=%.1f%%）；"
            "稳定锚与当日战术标签分离%s行" % (
                anchor_stability["transition_count"], anchor_stability["switch_count"],
                anchor_stability["switch_rate"] * 100.0,
                MAX_STABLE_ANCHOR_SWITCH_RATE * 100.0,
                anchor_stability["tactical_divergence_count"],
            )
        )},
        {"item": "历史ST", "status": "degraded", "detail": "当前按点时证券名称过滤；get_extras历史状态仍待平台兼容验证"},
        {"item": "未来标签", "status": "candidate", "detail": (
            "feature_daily与future_label_daily物理隔离；行%s/%s；"
            "成熟/部分/待成熟/不完整=%s/%s/%s/%s；标签行情截止%s；不回写T日" % (
                len(future_rows), len(observation_rows),
                future_status_counts["matured"], future_status_counts["partial"],
                future_status_counts["pending"], future_status_counts["data_incomplete"],
                label_price_end,
            )
        )},
    ])
    audit = pd.DataFrame([{
        "model": SIGNAL3_MODEL_VERSION,
        "start_date": analysis_dates[0],
        "end_date": end_day,
        "analysis_days": len(analysis_dates),
        "universe_codes": len(all_codes),
        "candidate_codes": len(candidate_union),
        "observation_rows": len(observation_rows),
        "future_label_rows": len(future_rows),
        "elapsed_seconds": round(time.time() - started, 2),
        "future_label_price_end": label_price_end,
        "future_label_matured": future_status_counts["matured"],
        "future_label_partial": future_status_counts["partial"],
        "future_label_pending": future_status_counts["pending"],
        "future_label_incomplete": future_status_counts["data_incomplete"],
        "raw_history_coverage": round(price_coverage["raw_history_coverage"], 6),
        "adjusted_history_coverage": round(price_coverage["adjusted_history_coverage"], 6),
        "dual_history_coverage": round(price_coverage["dual_history_coverage"], 6),
        "min_signal_day_coverage": round(minimum_signal_coverage, 6),
        "continuity_status": continuity_audit["status"],
        "continuity_breaks": continuity_audit["breaks"],
        "illegal_transition_count": continuity_audit["illegal_transitions"],
        "candidate_trade_status_violations": continuity_audit["trade_status_violations"],
        "signal_day_paused_excluded": paused_excluded,
        "signal_day_no_trade_excluded": no_trade_excluded,
        "signal_day_data_incomplete_excluded": incomplete_excluded,
        "theme_context_status": theme_context_status,
        "theme_context_failures": theme_context_failures,
        "theme_context_missing_candidates": theme_context_missing,
        "anchor_stability_status": anchor_stability["status"],
        "anchor_transition_count": anchor_stability["transition_count"],
        "anchor_switch_count": anchor_stability["switch_count"],
        "anchor_switch_rate": anchor_stability["switch_rate"],
        "anchor_tactical_divergence_count": anchor_stability["tactical_divergence_count"],
        "state_contract": SIGNAL3_STATE_CONTRACT,
        "gate_contract": SIGNAL3_GATE_CONTRACT,
        "theme_context_contract": SIGNAL3_THEME_CONTEXT_CONTRACT,
        "future_label_contract": SIGNAL3_FUTURE_LABEL_CONTRACT,
        "report_status": "candidate",
    }])
    panel = {
        "second_wave_latest": latest_frame,
        "second_wave_history": history_frame,
        "market_context": pd.DataFrame(market_rows),
        "theme_context_audit": pd.DataFrame(
            theme_context_audit_rows, columns=THEME_CONTEXT_AUDIT_COLUMNS,
        ),
        "future_label_daily": future_frame,
        "signal_day_coverage": pd.DataFrame(
            signal_day_coverage_rows, columns=SIGNAL_DAY_COVERAGE_COLUMNS,
        ),
        "data_quality": quality,
        "run_audit": audit,
    }
    if SAVE_HTML:
        html_text = build_html_report(panel, analysis_dates)
        path, digest = save_candidate_html(html_text, analysis_dates)
        panel["report_path"] = path
        panel["report_sha256"] = digest
    for title in (
        "second_wave_latest", "market_context", "theme_context_audit",
        "future_label_daily", "data_quality", "run_audit",
    ):
        print("\n[signal3输出] %s" % title)
        try:
            display(panel[title])
        except Exception:
            print(panel[title])
    print("[signal3完成] candidate；耗时%.1f秒；未冻结。" % (time.time() - started))
    return panel


if AUTO_RUN:
    panel = run_signal3_research(TRADE_DATE, START_DATE, END_DATE)