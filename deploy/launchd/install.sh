#!/bin/bash
# macOS launchd 盘前自启安装器（模板渲染 + 加载, 零硬编码路径）
#
# 用法:
#   bash deploy/launchd/install.sh                 # 安装并加载（默认 install）
#   bash deploy/launchd/install.sh install --at 09:05
#   bash deploy/launchd/install.sh status          # 查看安装状态
#   bash deploy/launchd/install.sh run-now         # 立即触发一次（验证幂等）
#   bash deploy/launchd/install.sh uninstall       # 卸载并移除 plist
#
# 配置优先级（与 config.py._load_dotenv 的 setdefault 语义一致）:
#   命令行 --at/--label > 已导出的 shell 环境变量 > 项目 .env > 内置默认值
#
# 可配置项:
#   AUTOSTART_AT     盘前启动时刻 HH:MM, 默认 09:10
#   AUTOSTART_LABEL  launchd 任务名, 默认 com.ticai-daban.morning
#                    （改名可并存多份安装, 例如区分不同项目副本）
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TEMPLATE="$HERE/com.ticai-daban.morning.plist.template"
AGENTS_DIR="$HOME/Library/LaunchAgents"

DEFAULT_AT="09:10"
DEFAULT_LABEL="com.ticai-daban.morning"

# ---------- 配置读取 ----------
# .env 取值: 忽略注释与空行, 容忍行内注释与引号（同 config.py._load_dotenv）
dotenv_get() {
  local key="$1" line val
  [ -f "$ROOT/.env" ] || return 0
  line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?$key=" "$ROOT/.env" | tail -1) || true
  [ -n "$line" ] || return 0
  val="${line#*=}"
  val="${val%% #*}"
  val="${val%\"}"; val="${val#\"}"
  val="${val%\'}"; val="${val#\'}"
  # 去除首尾空白
  echo "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# 优先级: shell 环境变量 > .env > 默认值
cfg_get() {
  local key="$1" default="$2" v
  v="${!key}"
  [ -n "$v" ] || v="$(dotenv_get "$key")"
  [ -n "$v" ] || v="$default"
  echo "$v"
}

CMD="install"
AT=""
LABEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    install|uninstall|status|run-now) CMD="$1" ;;
    --at)    AT="$2"; shift ;;
    --label) LABEL="$2"; shift ;;
    -h|--help)
      # 只打印文件头连续注释块(遇第一行非注释即停), 不随注释长度漂移
      awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
      exit 0 ;;
    *) echo "未知参数: $1（用 --help 查看用法）" >&2; exit 1 ;;
  esac
  shift
done

AT="${AT:-$(cfg_get AUTOSTART_AT "$DEFAULT_AT")}"
LABEL="${LABEL:-$(cfg_get AUTOSTART_LABEL "$DEFAULT_LABEL")}"

if ! echo "$AT" | grep -qE '^([01][0-9]|2[0-3]):[0-5][0-9]$'; then
  echo "启动时刻格式非法: '$AT'（应为 HH:MM, 例如 09:10）" >&2
  exit 1
fi
HOUR="${AT%%:*}"
MINUTE="${AT##*:}"
# launchd 的 Hour/Minute 是 integer, 前导零虽可解析但统一去掉更稳妥
HOUR=$((10#$HOUR))
MINUTE=$((10#$MINUTE))

PLIST="$AGENTS_DIR/$LABEL.plist"

# ---------- 子命令 ----------
case "$CMD" in
  status)
    echo "项目目录 : $ROOT"
    echo "任务名   : $LABEL"
    echo "启动时刻 : 工作日 $AT"
    echo "plist    : $PLIST $([ -f "$PLIST" ] && echo '(已安装)' || echo '(未安装)')"
    if launchctl list 2>/dev/null | grep -q "\b$LABEL\b"; then
      launchctl list | grep "\b$LABEL\b" | awk '{print "launchd  : 已加载 (PID="$1" 上次退出码="$2")"}'
    else
      echo "launchd  : 未加载"
    fi
    exit 0 ;;

  run-now)
    if ! launchctl list 2>/dev/null | grep -q "\b$LABEL\b"; then
      echo "任务未加载, 请先执行: bash deploy/launchd/install.sh install" >&2
      exit 1
    fi
    launchctl start "$LABEL"
    echo "已触发一次（start.sh 幂等, 存活进程会被跳过）; 日志: $ROOT/logs/launchd.log"
    exit 0 ;;

  uninstall)
    if [ -f "$PLIST" ]; then
      launchctl unload "$PLIST" 2>/dev/null || true
      rm -f "$PLIST"
      echo "已卸载并移除 $PLIST"
    else
      echo "未安装, 无需卸载"
    fi
    exit 0 ;;

  install)
    if [ ! -f "$TEMPLATE" ]; then
      echo "模板缺失: $TEMPLATE" >&2
      exit 1
    fi
    if [ ! -f "$ROOT/start.sh" ]; then
      echo "未找到 $ROOT/start.sh — 本脚本必须放在项目内 deploy/launchd/ 下" >&2
      exit 1
    fi
    mkdir -p "$AGENTS_DIR" "$ROOT/logs"
    sed -e "s|{{LABEL}}|$LABEL|g" \
        -e "s|{{PROJECT_DIR}}|$ROOT|g" \
        -e "s|{{HOUR}}|$HOUR|g" \
        -e "s|{{MINUTE}}|$MINUTE|g" \
        "$TEMPLATE" > "$PLIST"
    # 渲染结果必须是合法 plist, 否则 launchctl load 会静默失败
    if command -v plutil >/dev/null 2>&1; then
      plutil -lint "$PLIST" >/dev/null
    fi
    # 重复安装: 先卸载旧任务再加载, 否则 launchctl 报 already loaded
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "已安装: $PLIST"
    echo "  项目目录 $ROOT"
    echo "  工作日(周一~周五) $AT 触发 start.sh"
    echo "  运行日志 $ROOT/logs/launchd.log"
    echo "验证: bash deploy/launchd/install.sh run-now   查看: ... status"
    exit 0 ;;
esac
