# -*- coding: utf-8 -*-
"""回补 intraday_px 缺失分时点: radar_log(20s pct轨迹) 就近锚定重建价格。

背景: intraday_px 由雷达进程内存 _ipx_day 周期落盘(cycle%15)或退出时
flush, 进程非优雅退出会丢掉末段积累(实测某日只落到 14:56, 尾段 4 分钟
缺失); 且多数票仅开盘段满足记录条件, 之后 px_hist 只覆盖信号票。
radar_log(append 模式)从 09:25 起完整保留所有(涨幅≥1%或概率≥0.2)票的
20s 快照(t/pct/vol/amt), 可将缺失时刻用 pct×就近昨收锚重建。

锚定方式: 行情源切换(qmt↔腾讯)会造成昨收口径漂移(实测同一票开盘段
base=17.07 / 盘中段 base=17.48, 差2.4%), 全天单一昨收重建会在切换处
产生假跳变; 改用"最近真实点的局部昨收"逐点锚定, 价格连续性最好。

产物: 原位合并回 data/live/intraday_px_DATE.json(已有真实价的时刻不覆盖)。
"""
import json
import shutil
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

LIVE = DATA / "live"
DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
T_LO, T_HI = "091500", "150100"   # 连续竞价+集合竞价窗口(含15:00收盘轮)

# ---------- 1. radar_log 轨迹 ----------
log_rows: dict[str, list] = {}
with open(LIVE / f"radar_log_{DATE}.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r["t"]
        if not (T_LO <= t <= T_HI):    # 盘后手动补跑的行不入库
            continue
        log_rows.setdefault(r["code"], []).append(
            (t, r["pct"], r.get("vol", 0), r.get("amt", 0)))
print(f"radar_log 载入 {len(log_rows)} 只票")

# ---------- 2. 就近锚定回填 ----------
ipx_f = LIVE / f"intraday_px_{DATE}.json"
ipx: dict = json.loads(ipx_f.read_text(encoding="utf-8")) if ipx_f.exists() else {}
n_new_pt, n_new_code = 0, 0

for code, rows in log_rows.items():
    pts = ipx.get(code, [])
    have = {p[0] for p in pts}
    missing = [r for r in rows if r[0] not in have]
    if not missing:
        continue
    # 锚点: ipx真实价点 与 log同时刻pct → 局部昨收(就近源切换自适应)
    pct_by_t = {t: pct for t, pct, _, _ in rows}
    anchors = []                       # [(t, base)]
    for p in pts:
        pct = pct_by_t.get(p[0])
        if pct is not None and pct > -99:      # 防除零
            anchors.append((p[0], p[1] / (1 + pct / 100)))
    if not anchors:
        continue                        # 无锚(纯log票), 放弃重建
    a_t = [a[0] for a in anchors]

    def _base_near(t: str) -> float:
        i = bisect_left(a_t, t)
        cands = []
        if i > 0:
            cands.append(anchors[i - 1])
        if i < len(anchors):
            cands.append(anchors[i])
        return min(cands, key=lambda a: abs(int(a[0]) - int(t)))[1]

    added = [[t, round(_base_near(t) * (1 + pct / 100), 3), vol, amt]
             for t, pct, vol, amt in missing]
    ipx[code] = sorted(pts + added, key=lambda p: p[0])
    n_new_pt += len(added)
    n_new_code += 1

print(f"回填 {n_new_code} 只票 / {n_new_pt} 个缺失点")

# ---------- 3. 备份并写回 ----------
if n_new_pt:
    if ipx_f.exists():
        shutil.copy(ipx_f, LIVE / f"intraday_px_{DATE}.bak.json")
    ipx_f.write_text(json.dumps(ipx), encoding="utf-8")
    bak = f"(备份 intraday_px_{DATE}.bak.json)" if ipx_f.exists() else ""
    print(f"已写回 {ipx_f.name} {bak}")
else:
    print("无可回填点, 文件未动")
