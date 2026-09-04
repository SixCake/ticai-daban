# 策略开发规范（隔离规范）

本框架的每个策略是一个**独立目录**，跑在**独立 rqalpha 进程**里，有**独立账户与落盘**。
策略之间不得互相依赖。本文是硬约束，不是建议。

## 目录结构

```
strategies/
├── strategies.yaml          # 启用清单 + 各策略初始资金 + 订阅的 feed
├── _template/               # 本规范 + 骨架模板（复制它开始新策略）
│   ├── SPEC.md
│   ├── strategy.py
│   └── config.yaml
└── <你的策略名>/
    ├── strategy.py          # 策略逻辑（唯一代码文件）
    └── config.yaml          # 该策略的配置（资金/基准/订阅 feed/mod 参数）
```

## 硬约束

1. **一策略一目录**，只含 `strategy.py` + `config.yaml`。
2. **禁止 import 其他策略目录**。策略间共享只能通过在 `_template` 里沉淀，
   复制代码而非互相引用。
3. **禁止 import 项目的 `core/` / `quotes/` / `apps/`**。领域逻辑已经通过
   注入 API 暴露；直接 import 会让策略耦合到内部实现，且绕过时间戳闸门。
4. **禁止硬编码 `data/` 路径**。所有数据经注入 API 取。
5. **AI 只能经 `ai_feed(name)` 读**，且 `name` 必须在 `config.yaml` 的
   `feeds` 里声明过。未声明的调用会被拒绝并打告警。
6. **每策略独立初始资金**（默认 100 万）、独立账户、独立落盘目录。

## 四段生命周期钩子

rqalpha 原生支持全部四段，策略按需定义（不定义的钩子不会被调用）：

| 阶段 | 函数签名 | 触发时刻 | 典型用途 |
|---|---|---|---|
| 盘前 | `before_trading(context)` | 09:00 | 用 T-1 结构因子选股、构建候选池 |
| 竞价 | `open_auction(context, bar_dict)` | 09:25 | 竞价量爆/高开筛选、竞价挂单 |
| 盘中 | `handle_bar(context, bar_dict)` | 每 20s 一轮 | 信号触发买入、止损、封板判定 |
| 收盘 | `after_trading(context)` | 15:30 | 未封清仓、当日复盘记录 |

盘中用 `handle_bar` 而非 `handle_tick`：事件源发的是 BAR 事件（20s 快照合成），
不是 TICK；发 TICK 需要构造带买卖盘口的 TickObject 并走 tick_matcher，对本项目
（只有快照价/量/额）是过度设计。20s 粒度的 BAR 已等价于"每 20s 一轮"。

### 频率语义（1m vs 1d）

- `1m`（盘中模拟/回放）：`SCAN_END`（10:30 后不新买）等盘中时段限制生效。
- `1d`（日频回测实验）：单日只有一根 15:00 的 bar，盘中时段限制不适用，
  故豁免 `SCAN_END`；语义变为"当日若触发信号则按触发价近似成交"的粗粒度
  近似，能否成交仍受 FILL_SIM 闸门与涨停拒单约束。
- **每个 run 的记录写入自己的 run 目录**（`data/sim/runs/{run_id}/` 的
  equity/trades/positions）；回测 run 与 live 模拟 run 天然隔离，
  不同频率/起点的结果不会揉进同一条收益曲线。

两个注意点：

1. **盘中钩子是 `handle_bar` 而非 `handle_tick`** —— 本框架的事件源按雷达
   20s 快照发 `BAR` 事件（见 `rqalpha_mod_ticai/event_source.py`）。
2. **`open_auction` 是两参数**（同 `handle_bar`）—— rqalpha 的 executor 会给
   `OPEN_AUCTION` 事件自动挂 `bar_dict`，竞价价走 DataSource 的
   `get_open_auction_bar()`。写成单参数会报
   `open_auction() takes 1 positional argument but 2 were given`。

## 注入 API（策略取数据的唯一通道）

全部来自 `from rqalpha.api import *`，由 `rqalpha_mod_ticai/api.py` 注入。

| API | 返回 | 可用阶段 |
|---|---|---|
| `ticai_signals(stage=None)` | 前向预警信号列表（S1/S2/S3） | 全阶段 |
| `ticai_struct()` | `{代码: {g_chip, gate, v5, zb20, ir}}` V5 结构层影子分 | 全阶段 |
| `ticai_theme_heat()` | 题材热度排名（按 heat 降序） | 全阶段 |
| `ticai_sw_flow()` | 申万资金流向聚合 | 竞价/盘中 |
| `ticai_seesaw()` | 龙头拐头·跷跷板事件 | 竞价/盘中 |
| `ticai_intraday(code)` | 个股分时 `[(HHMMSS, px, vol股, amt元)]` | 竞价/盘中 |
| `ai_feed(name, topic=None)` | 订阅的 AI feed 条目 | 全阶段 |
| `set_benchmark(id)` | 设定绩效基准（聚宽同款） | 仅 `init` |

**代码口径**：所有 API 返回的股票代码一律是 **rqalpha 口径**（`000001.XSHE` /
`600000.XSHG`），可直接用于 `order_shares()` 等下单 API。项目内部其他模块用
`000001.SZ`，转换只在 `rqalpha_mod_ticai/codes.py` 发生 —— 策略不需要关心。

## 两条时间戳闸门（防回测未来信息）

框架自动施加，策略不需要自己过滤时间：

- **信号闸门**：`ticai_signals()` 只返回 `t <= 当前模拟时刻` 的信号
- **feed 闸门**：`ai_feed()` 只返回产出时间 `ts <= 当前模拟时刻` 的条目

盘中模式下"当前模拟时刻"就是墙钟；回放模式下是回放到的 `calendar_dt`。
因此同一份策略代码在盘中与回测下看到的数据边界一致。

## 执行口径（撮合真实性）

由框架统一施加，策略不需要自己实现：

| 约束 | 来源 | 说明 |
|---|---|---|
| 涨停价不能买 / 跌停价不能卖 | rqalpha `sys_simulation.price_limit` | `reaches_limit_up`: `price >= limit_up - tick_size + tolerance` |
| 一字板买不进 | 同上 | 一字板价 == 涨停价，天然被上一条拦住 |
| 无量撤单 | `sys_simulation.inactive_limit` | bar 成交量为 0 则撤单 |
| 单笔 ≤ bar 成交量 25% | `sys_simulation.volume_limit` | 流动性约束 |
| T+1 | rqalpha 股票账户 | 当日买入不可卖 |
| **成交概率抽样** | `rqalpha_mod_ticai/broker.py` | `FILL_SIM=0.30`，模拟涨停排队买不进；rqalpha 原生无此概念 |

策略侧的挂价约定（不是撮合约束）：

```python
from rqalpha_mod_ticai.broker import limit_price_of
lmt = limit_price_of(trigger_px, pre_close, slip=0.005)
order_shares(code, qty, LimitOrder(lmt))     # 注: rqalpha 叫 LimitOrder，聚宽叫 LimitOrderStyle
```

即「高挂限价单」：限价 = 触发价 × 1.005，涨停价封顶。限价 ≥ 现价时 rqalpha
以现价即时成交（效果同市价），但成交价有上界 —— 防暴拉瞬间异常高价。

## AI feed 订阅

AI **不进策略代码**。AI 生产者（`apps/ai_feed.py`）随时产出 feed 文件，
策略只读。

```yaml
# config.yaml
feeds:
  - theme_narrative          # 共享 feed
  - private:my_custom_feed   # 本策略专属私有 feed
```

- 共享 feed：`data/sim/ai_feeds/{feed_name}/{date}.json`
- 私有 feed：`data/sim/ai_feeds/private/{strategy}/{feed_name}/{date}.json`

私有 feed 不与其它策略共享 —— 每个策略可以为自己定制 AI 信息源。
读未声明的 feed 会被拒绝（隔离规范）。

条目结构：

```json
{"ts": 1756872000.0, "t": "09:20:00", "topic": "液冷",
 "score": 0.82, "text": "...", "src": "llm", "extra": {...}}
```

## 绩效指标

- **当日**：`data/sim/runs/{run_id}/state/{date}.json`（净值/持仓/挂单/当日盈亏）
- **跨日**：`data/sim/runs/{run_id}/equity.parquet`（每日结算追加）

Sharpe / 信息比率需要多日序列，故从跨日序列算（`rqalpha_mod_ticai/metrics.py`）。
基准由策略在 `init()` 里用 `set_benchmark()` 指定，可选：

- `DBBNCH.XSHG` —— 自建打板基准（全A等权涨幅累积指数，无需额外采集）
- `000300.XSHG` / `000985.XSHG` / `000001.XSHG` / `399006.XSHE` —— 宽基指数
  （需先跑 `python collect/fetch_index_panel.py` 采集）

基准缺失时 IR/alpha/beta 返回 `None`，不伪造 0。

## 重启回载

不读 state 文件恢复账户，而是靠 live 启动时的 **catchup 确定性回放**：从当日
开盘把已过去的快照时刻重放一遍重建账户。因为 FILL_SIM 种子固定，同一次 run
重复跑结果逐笔一致（已实测验证），所以重放重建出的账户与崩溃前完全相同。

前提：信号源 `presig_state_{date}.json` 在盘上（雷达持续写）、FILL_SIM 种子
固定。两者都满足，故无需额外的账户持久化/恢复机制。

## 策略开发流程

1. 复制 `_template/` 为 `strategies/<策略名>/`
2. 改 `strategy.py` 的四段钩子逻辑
3. 改 `config.yaml`：初始资金、基准、订阅的 feed
4. 在 `strategies/strategies.yaml` 里加一条启用记录
5. 回放验证：`python apps/sim.py --strategy <策略名> --replay --date <日期>`
6. 盘中运行：`bash start.sh`（会按 `strategies.yaml` 拉起所有启用的策略）

## 回测窗口约束（重要）

盘中粒度（20s 快照）的历史数据只有 `data/live/intraday_px_{date}.json`
已有的分区（约 8 天），**且票池有幸存者偏差** —— 雷达只记录涨幅 ≥1% 或
概率 ≥0.2 的票，没涨起来的票没记录。

因此：

- **盘中回放**定位为**实盘验证**（验证策略在真实盘中数据上的行为），
  不是历史验证
- **日频回测**（`market.daily_panel` 2019-11 起，窗口充足）可做历史验证
- 严格的历史因子验证继续用 `research/*.py` 的既有脚本
