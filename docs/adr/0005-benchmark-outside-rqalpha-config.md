# 基准与无风险利率由数据源补齐，而非留空

`config.base.benchmark` 必须设置（默认自建打板基准 `DBBNCH.XSHG`），且数据源的
`get_yield_curve()` 必须返回非空的收益率曲线。两者都是 sys_analyser 算出
alpha/beta/夏普/索提诺/信息比率/超额收益的前提。

## 曾经的反向决策（已废弃）

早期版本刻意**不**设 benchmark、`get_yield_curve` 返回空表，理由是 sys_analyser
会校验基准数据覆盖回测区间、不满足就抛异常终止 run。

事后查明那两次失败的真正根因都不是"数据新鲜度"，而是数据源自身的两个 bug：

1. `history_bars` 把 `bar_count=None` 当成 1 根 bar（正确语义是"截至 dt 的全部"），
   导致 sys_analyser 取基准时长度对不上而报错；
2. `get_yield_curve` 返回空表 → `get_risk_free_rate` 得 nan → 夏普/alpha/索提诺
   全为 nan。

修好这两处后，设置 benchmark 不再崩，且能拿到完整指标格。故废弃"留空"决策。

## 仍保留的两个防护

- **尾部前向填充**：盘中实时模式下 daily_panel 尚未补尾（靠 daily_update.sh
  收盘后跑），回测末日=今天会缺基准最后一天。数据源对面板未覆盖的尾部
  交易日做前向填充（当日收益=0，持平），既不崩也不伪造涨跌。
- **常数无风险利率**：不采国债曲线，用常数年化 1.5%（`risk_free_rate` 可配）
  填全部 tenor。中国短端利率量级，仅供夏普/alpha/索提诺的分母。

## Consequences

指标格里的「日胜率」是 sys_analyser 按**日**算的（正收益天数占比），与按
**笔**算的「笔胜率/笔盈亏比」（FIFO 回合配对）口径不同，页面分开标注，
不要混用。
