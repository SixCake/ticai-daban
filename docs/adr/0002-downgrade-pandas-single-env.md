# 项目 pandas 降到 2.x 走单一环境，而非为模拟层建独立 venv

rqalpha 6.3.0 声明 `pandas<3.0.0`，而项目原本跑在 Homebrew 系统 Python 的
pandas 3.0.3 上。决定把项目 pandas 降到 2.x 并全部服务统一跑在项目 `.venv`
里，而不是让模拟层单独用一个环境。

理由：代码扫描证明项目 375 处 pandas 用法零命中 pandas 3.0 独有 API，且
`requirements.txt` 自身声明本来就是 `pandas>=2.0`；降级后逐项基线比对
（数据集元信息 + parquet shape/dtypes/校验值 + akshare/tushare 接口）
0 项不一致。双环境会让依赖变更需要两边同步、容易混淆。

顺带解决了另一个问题：Homebrew 系统 Python 是 PEP 668 externally-managed，
pip 装不进（`--break-system-packages` 有搞坏 Homebrew 的风险），所以本来
就需要一个项目 venv。

## Consequences

`start.sh` 优先用 `.venv/bin/python`；找不到时回退系统 Python 并告警
（此时策略模拟不可用）。

降级暴露出一个真实的版本行为差异：`astype(str)` 对缺失值的处理两版不一致
（3.0 的专用 `str` dtype 保留 NA，2.x 的 object dtype 转成字面 `'nan'`/
`'None'`）。这会让封板时间字段写出 `'000nan'` 并在字符串比较里误判为
早封板。已用 `core/times.py` 统一封装规避，所有封板时间的规范化与比较
都必须走它。

## 回滚预案

若日后必须回到 pandas 3.0，则改为双环境：模拟层单独建 venv 装 pandas 2.x +
rqalpha，跨进程只走文件通信（模拟层与看板本来就只通过 `data/sim/` 交换数据）。
