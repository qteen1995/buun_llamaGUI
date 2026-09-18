# -*- coding: utf-8 -*-
"""探测可执行文件支持哪些参数：跑一次 `--help`，把输出的所有 flag 收下来。

buun-llama-cpp 迭代很快，不同 build 的参数集不一样。有了这个探测，
界面就能把当前 build 不认识的参数标出来并自动取消勾选，
而不是等用户点了启动才报 invalid argument。
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import schema as S

_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][A-Za-z0-9][A-Za-z0-9_-]*)")
# 行首的「参数定义」： -fa, --flash-attn   /   --embedding(s)   /   -ncmoe, --n-cpu-moe N
_LEAD_RE = re.compile(
    r"^\s*(?P<flags>(?:-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"(?:\s*,\s*-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)*)")
_TOK_RE = re.compile(r"-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*")
_VER_RE = re.compile(r"(?:version|build)[:\s]+([^\n]+)", re.I)

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000


def run_help(exe: str, timeout: float = 25.0) -> Tuple[str, int]:
    """执行 --help 并返回 (合并输出, 退出码)。失败返回 ("", -1)。"""
    flags = CREATE_NO_WINDOW if IS_WINDOWS else 0
    try:
        proc = subprocess.run(
            [exe, "--help"], capture_output=True, timeout=timeout,
            creationflags=flags, cwd=os.path.dirname(exe) or None,
            env={**os.environ, "NO_COLOR": "1"})
    except (OSError, subprocess.TimeoutExpired):
        return "", -1
    out = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    return out + "\n" + err, proc.returncode


def parse_flags(text: str) -> Set[str]:
    """只收「行首的定义行」里的 flag。

    帮助文本里也会在说明、示例、弃用提示里提到 flag，例如
    ``legacy --spec-stage, --draft-*, -mtp flags are removed`` ——
    全量扫描会把它们误判成「支持」。所以优先按行首解析：

        ``-fa, --flash-attn (auto|on|off)`` → {-fa, --flash-attn}
        ``--embedding(s)  restrict to ...`` → {--embedding}
    """
    strict: Set[str] = set()
    for line in (text or "").splitlines():
        m = _LEAD_RE.match(line)
        if not m:
            continue
        rest = line[m.end():]
        if rest and not rest[:1].isspace() and rest[0] != "(":
            continue
        strict.update(_TOK_RE.findall(m.group("flags")))
    if len(strict) >= 20:
        return strict
    # 兜底：某些工具的帮助格式差异很大，退回宽松扫描
    return {m.group(1) for m in _FLAG_RE.finditer(text or "")}


def parse_version(text: str) -> str:
    m = _VER_RE.search(text or "")
    if m:
        return m.group(1).strip()[:60]
    for line in (text or "").splitlines():
        line = line.strip()
        if line and ("llama" in line.lower() or "version" in line.lower()
                     or "build" in line.lower()):
            return line[:80]
    return next((ln.strip()[:80] for ln in (text or "").splitlines()
                 if ln.strip()), "")


def candidates(field: S.F) -> List[str]:
    """某个参数项所有可能出现的 flag（含三态项的反向 flag）。"""
    out: List[str] = []
    if field.flag:
        out.append(field.flag)
    for argv in (field.argmap or {}).values():
        for token in argv:
            if token.startswith("-"):
                out.append(token)
    return out


def supports(flags: Set[str], field: S.F) -> Optional[bool]:
    """True/False；无法判断（没有探测过）返回 None。"""
    if not flags:
        return None
    cands = candidates(field)
    if not cands:
        return True
    return any(c in flags for c in cands)


def probe(exe: str) -> Dict[str, Any]:
    text, code = run_help(exe)
    if code == -1:
        return {"ok": False, "error": "无法执行 --help（超时或文件不可执行）",
                "flags": set(), "version": "", "count": 0, "text": ""}
    flags = parse_flags(text)
    return {
        "ok": bool(flags),
        "error": "" if flags else "--help 输出里没有解析到任何参数",
        "flags": flags,
        "version": parse_version(text),
        "count": len(flags),
        "text": text,
    }


def probe_all(exe_by_mode: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for mode, exe in exe_by_mode.items():
        if exe and os.path.isfile(exe):
            out[mode] = probe(exe)
    return out


def stamp(path: str) -> str:
    try:
        st = os.stat(path)
        return "%d:%d" % (int(st.st_mtime), st.st_size)
    except OSError:
        return ""





def spec_types(flags: Set[str]) -> bool:
    return "--spec-type" in flags
