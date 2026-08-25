# -*- coding: utf-8 -*-
"""股票代码转换（唯一出处）"""


def ts_code_of(code6: str) -> str:
    """6位数字代码 → ts_code（60x/68x→SH, 00x/30x→SZ, 其余→BJ）"""
    code6 = str(code6).zfill(6)
    if code6.startswith(("60", "68")):
        return f"{code6}.SH"
    if code6.startswith(("00", "30")):
        return f"{code6}.SZ"
    return f"{code6}.BJ"


def to_sym(ts_code: str) -> str:
    """ts_code → 腾讯行情符号（如 600000.SH → sh600000）"""
    code, exch = ts_code.split(".")
    return ("sh" if exch == "SH" else "sz") + code


def to_ts_code(sym: str) -> str:
    """腾讯行情符号 → ts_code（如 sh600000 → 600000.SH）"""
    exch = "SH" if sym.startswith("sh") else "SZ"
    return f"{sym[2:]}.{exch}"
