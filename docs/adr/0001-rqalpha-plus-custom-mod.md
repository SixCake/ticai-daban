# 用 rqalpha 作引擎 + 自写 Mod，而非自建引擎或官方 tushare mod

策略模拟框架选 rqalpha 6.3.0 作事件驱动引擎，通过自写的 `rqalpha_mod_ticai`
接入本项目数据源；不自建引擎，也不用官方的 `rqalpha-mod-tushare`。

理由：rqalpha 原生提供了我们需要的四段生命周期钩子（含 `OPEN_AUCTION`
竞价阶段）、A 股撮合约束（涨跌停拒单、无量撤单、成交量限制、T+1）与
绩效分析，自建这些成本远高于写适配层。

## Considered Options

- **自建薄引擎**：可完全掌控，但要自己实现撮合器、账户、T+1、绩效分析，
  且失去与 rqalpha 生态的兼容性。
- **官方 `rqalpha-mod-tushare`**：实测不可用 —— 它是 2017 年的 demo，
  README 自述「暂不发 PyPI、不能 pip 安装」，代码用已下线的
  `ts.get_k_data()` 和 pandas 2.0 已移除的 `DataFrame.as_matrix()`，
  且仍甩回 `BaseDataSource` 从而照样需要米筐 bundle。

## Consequences

自写 `TicaiDataSource` 继承的是 `AbstractDataSource` 而非 `BaseDataSource`，
因此**不依赖米筐 bundle**（`rqalpha update-bundle`）。代价是要自己实现
Instrument、交易日历、涨跌停价推算等基础数据接口。

rqalpha 官方文档里的 Mod 配置写法（`__mod_config__` yaml 字符串）是 2.x/3.x
的旧写法，6.3.0 实际读的是模块级 `__config__` dict。
