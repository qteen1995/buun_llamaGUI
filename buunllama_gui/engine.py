# -*- coding: utf-8 -*-
"""buun-llama-cpp 引擎层：码本环境变量、CUDA 运行时依赖诊断。

这个模块集中放「只有 buun 这套引擎才需要」的东西，跟核心逻辑分开：

1. **TCQ 码本** —— ``turbo3_tcq / turbo2_tcq / turbo1_tcq`` 这三个 KV 档位要靠
   ``codebooks/`` 里的码本文件才能工作，路径通过环境变量传给子进程：

       TURBO_TCQ_CB  = <engine>/codebooks/3bit/cb_50iter_finetuned.bin
       TURBO_TCQ_CB2 = <engine>/codebooks/2bit/tcq_2bit_100iter_s99.bin

   这是仓库自带的 run-server.ps1 干的事，这里照做。用户自己设过同名变量时不覆盖。

2. **CUDA 运行时 DLL** —— 这套 exe 是 DLL 结构（llama-server.exe 只有 10 KB，
   真正实现在 llama-server-impl.dll 里），而且 ggml-cuda.dll 会直接引用
   ``cublas64_XX.dll`` 这类 CUDA 运行时。如果那些 DLL 既不在 exe 同目录、
   也不在 PATH 上，Windows 会以 0xC0000135（STATUS_DLL_NOT_FOUND）直接拒绝启动，
   而且**不给出任何提示**——只看到一个秒退的进程。

   所以启动前解析一次导入表，把问题变成一句人话；同时把找到的 CUDA 运行时目录
   补进**子进程**的 PATH（不动系统环境变量）。
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 码本
# --------------------------------------------------------------------------- #

# 环境变量 -> 仓库内相对路径（按顺序找第一个存在的）
CODEBOOKS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("TURBO_TCQ_CB", (
        "codebooks/3bit/cb_50iter_finetuned.bin",
        "codebooks/3bit/tcq_3bit_s99.bin",
    )),
    ("TURBO_TCQ_CB2", (
        "codebooks/2bit/tcq_2bit_100iter_s99.bin",
        "codebooks/2bit/tcq_2bit_50iter_s99.bin",
    )),
)


def repo_root(engine_path: str, exe: str = "") -> str:
    """从 engine_path（可能是仓库根、build/bin，或某个 exe）推出仓库根目录。"""
    p = str(engine_path or "").strip().strip('"')
    if not p and exe:
        p = os.path.dirname(str(exe))
    if not p:
        return ""
    if os.path.isfile(p):
        p = os.path.dirname(p)
    p = os.path.abspath(p)
    # build/bin -> 仓库根；build -> 仓库根
    for _ in range(3):
        base = os.path.basename(p).lower()
        if base == "bin" or base.startswith("build"):
            parent = os.path.dirname(p)
            if parent and parent != p:
                p = parent
                continue
        break
    return p


def codebook_paths(engine_path: str, exe: str = "") -> Dict[str, str]:
    """找仓库里存在的码本文件，返回 {环境变量: 绝对路径}。"""
    root = repo_root(engine_path or (os.path.dirname(exe) if exe else ""))
    out: Dict[str, str] = {}
    if not root:
        return out
    for env, rels in CODEBOOKS:
        for rel in rels:
            cand = os.path.join(root, *rel.split("/"))
            if os.path.isfile(cand):
                out[env] = cand
                break
    return out


def codebook_env(engine_path: str, exe: str = "") -> Dict[str, str]:
    """要注入子进程的码本环境变量（用户已经设过的同名变量不覆盖）。"""
    out: Dict[str, str] = {}
    for env, path in codebook_paths(engine_path, exe).items():
        if not os.environ.get(env):
            out[env] = path
    return out


def codebooks_missing(engine_path: str) -> List[str]:
    """仓库在，但码本文件缺失时返回缺的那几个。"""
    root = repo_root(engine_path)
    if not root or not os.path.isdir(root):
        return []
    found = codebook_paths(engine_path)
    return [env for env, _ in CODEBOOKS if env not in found]


# --------------------------------------------------------------------------- #
# CUDA 运行时目录探测
# --------------------------------------------------------------------------- #

def cuda_bin_dirs() -> List[str]:
    """本机所有 CUDA 安装的运行时 DLL 目录。

    CUDA 12.x 的 DLL 在 ``<CUDA>\\bin``，**13.x 才在** ``<CUDA>\\bin\\x64``，
    两边都要收，不然会出现「装了 CUDA 但引擎还是找不到 DLL」。
    """
    out: List[str] = []
    roots: List[str] = []
    for var in ("CUDA_PATH", "CUDA_HOME"):
        v = os.environ.get(var)
        if v:
            roots.append(v)
    for base in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                 r"C:\Program Files\NVIDIA Corporation\CUDA"):
        if os.path.isdir(base):
            try:
                for name in sorted(os.listdir(base)):
                    full = os.path.join(base, name)
                    if os.path.isdir(full):
                        roots.append(full)
            except OSError:
                pass
    env_path = os.environ.get("PATH") or os.environ.get("Path") or ""
    for part in env_path.split(os.pathsep):
        part = part.strip().strip('"')
        if part and "cuda" in part.lower() and os.path.isdir(part):
            roots.append(os.path.dirname(part.rstrip("\\/")) if
                         part.rstrip("\\/").lower().endswith("x64") else part)
    for r in roots:
        for sub in ("bin", os.path.join("bin", "x64")):
            d = os.path.join(r, sub)
            if os.path.isdir(d) and d not in out:
                out.append(d)
    return out


# --------------------------------------------------------------------------- #
# PE 导入表
# --------------------------------------------------------------------------- #

# 这些是 API set / 系统垫片，真正的实现在 System32 里，不能按文件名判缺失
_SKIP_PREFIX = ("api-ms-win-", "ext-ms-win-")
_SKIP_EXACT = frozenset((
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "user32.dll", "advapi32.dll",
    "shell32.dll", "ole32.dll", "oleaut32.dll", "ws2_32.dll", "crypt32.dll",
    "bcrypt.dll", "ncrypt.dll", "shlwapi.dll", "rpcrt4.dll", "gdi32.dll",
    "winmm.dll", "version.dll", "comdlg32.dll", "comctl32.dll", "iphlpapi.dll",
    "dbghelp.dll", "psapi.dll", "setupapi.dll", "dwmapi.dll", "uxtheme.dll",
    "secur32.dll", "wldap32.dll", "normaliz.dll", "winmmbase.dll",
))


def pe_imports(path: str, limit: int = 256) -> List[str]:
    """读取 PE 文件的导入 DLL 名列表。失败返回空表。"""
    names: List[str] = []
    try:
        with open(path, "rb") as f:
            f.seek(0)
            if f.read(2) != b"MZ":
                return []
            f.seek(0x3C)
            pe = struct.unpack("<I", f.read(4))[0]
            f.seek(pe)
            if f.read(4) != b"PE\0\0":
                return []
            f.read(2)                                    # Machine
            nsec = struct.unpack("<H", f.read(2))[0]
            f.seek(pe + 4 + 16)                          # SizeOfOptionalHeader
            opt_size = struct.unpack("<H", f.read(2))[0]
            opt = pe + 24                                # 可选头紧跟文件头
            f.seek(opt)
            magic = struct.unpack("<H", f.read(2))[0]
            dd = opt + (112 if magic == 0x20B else 96)
            f.seek(dd + 8)                               # 导入表是第 1 个目录项
            imp_rva, _ = struct.unpack("<II", f.read(8))
            if not imp_rva:
                return []
            secs = []
            f.seek(opt + opt_size)
            for _ in range(nsec):
                raw = f.read(40)
                if len(raw) < 40:
                    break
                vsize, va, rawsize, rawptr = struct.unpack("<IIII", raw[8:24])
                secs.append((va, max(vsize, rawsize), rawptr))

            def r2o(rva: int) -> Optional[int]:
                for va, size, rawptr in secs:
                    if va <= rva < va + size:
                        return rawptr + (rva - va)
                return None

            off = r2o(imp_rva)
            if off is None:
                return []
            for i in range(limit):
                f.seek(off + i * 20)
                ent = f.read(20)
                if len(ent) < 20:
                    break
                name_rva = struct.unpack("<I", ent[12:16])[0]
                if name_rva == 0:
                    break
                no = r2o(name_rva)
                if no is None:
                    continue
                f.seek(no)
                buf = f.read(260)
                end = buf.find(b"\0")
                if end <= 0:
                    continue
                names.append(buf[:end].decode("ascii", "replace").lower())
    except (OSError, struct.error):
        return []
    return names


def _index_dir(d: str, cache: Dict[str, set]) -> set:
    key = os.path.normcase(d)
    if key not in cache:
        try:
            cache[key] = {n.lower() for n in os.listdir(d)}
        except OSError:
            cache[key] = set()
    return cache[key]


def find_dll(name: str, dirs: Sequence[str],
             cache: Optional[Dict[str, set]] = None) -> str:
    """在给定目录里找 DLL（大小写不敏感）。"""
    cache = cache if cache is not None else {}
    low = name.lower()
    for d in dirs:
        if low in _index_dir(d, cache):
            return os.path.join(d, name)
    return ""


def _interesting(name: str) -> bool:
    low = name.lower()
    if low in _SKIP_EXACT:
        return False
    if any(low.startswith(p) for p in _SKIP_PREFIX):
        return False
    return True


def diagnose_exe(exe: str, extra_dirs: Sequence[str] = (),
                 depth: int = 3) -> Dict[str, object]:
    """启动前的依赖体检。

    返回::

        {"ok": bool,
         "missing": [(dll, 哪个文件要的, 层数)],
         "search":  [搜索过的目录],
         "note":    "给界面用的一句话"}

    只看能静态判定的部分：exe 同目录 + 追加目录 + 系统 PATH。
    找不到的会列出来，但不会自动去改用户的系统环境变量。
    """
    exe = str(exe or "")
    dirs: List[str] = []
    if exe and os.path.isdir(os.path.dirname(exe)):
        dirs.append(os.path.dirname(exe))
    for d in extra_dirs:
        if d and os.path.isdir(d) and d not in dirs:
            dirs.append(d)
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                          "System32")
    if os.path.isdir(sysdir):
        dirs.append(sysdir)

    if not exe or not os.path.isfile(exe):
        return {"ok": False, "missing": [], "search": dirs,
                "note": "找不到可执行文件：%s" % (exe or "(空)")}

    cache: Dict[str, set] = {}
    missing: List[Tuple[str, str, int]] = []
    seen = set()
    ok = True

    def walk(path: str, level: int) -> None:
        nonlocal ok
        for dll in pe_imports(path):
            if not _interesting(dll):
                continue
            key = dll
            if key in seen:
                continue
            seen.add(key)
            found = find_dll(dll, dirs, cache)
            if not found:
                ok = False
                missing.append((dll, os.path.basename(path), level))
            elif level < depth and found.lower() != path.lower():
                walk(found, level + 1)

    walk(exe, 0)
    return {"ok": ok, "missing": missing, "search": dirs, "note": ""}


def format_diagnosis(diag: Dict[str, object]) -> str:
    """把体检结果写成给人看的一段话。"""
    missing = list(diag.get("missing") or [])
    if diag.get("note"):
        return str(diag["note"])
    if not missing:
        return ""
    lines = ["缺少 %d 个运行库 DLL，引擎会以 0xC0000135 秒退：" % len(missing)]
    for dll, who, level in missing[:8]:
        lines.append("  · %s（%s 需要）" % (dll, who))
    if len(missing) > 8:
        lines.append("  · 还有 %d 个…" % (len(missing) - 8))
    names = {d.lower() for d, _, _ in missing}
    if any(n.startswith("cublas") or n.startswith("cudart") or
           n.startswith("cufft") or n.startswith("curand") or
           n.startswith("nvjitlink") for n in names):
        lines.append("看起来是 CUDA 运行时缺失。CUDA 12.x 的 DLL 在 "
                     "<CUDA>\\bin，13.x 在 <CUDA>\\bin\\x64；"
                     "把对应目录加进 PATH，或把那几个 DLL 拷到 exe 同目录。")
    return "\n".join(lines)


def engine_env(engine_path: str = "", exe: str = "",
               use_codebooks: bool = True,
               use_cuda_path: bool = True) -> Dict[str, str]:
    """要合并进子进程环境的变量。

    * 码本：只在文件真的存在、且用户没自己设过时注入
    * PATH：把本机的 CUDA 运行时目录补在前面（只影响这个子进程）
    """
    env: Dict[str, str] = {}
    if use_codebooks:
        env.update(codebook_env(engine_path, exe))
    if use_cuda_path:
        dirs = cuda_bin_dirs()
        if dirs:
            key = "PATH" if "PATH" in os.environ else "Path"
            cur = os.environ.get(key, "")
            parts = [p for p in cur.split(os.pathsep) if p]
            lowered = {os.path.normcase(p) for p in parts}
            add = [d for d in dirs if os.path.normcase(d) not in lowered]
            if add:
                env[key] = os.pathsep.join(add + parts)
    return env


def runtime_report(engine_path: str, exe: str = "") -> str:
    """启动前写进日志的一小段环境说明（没有可说的就返回空串）。"""
    bits: List[str] = []
    cbs = codebook_env(engine_path, exe)
    if cbs:
        bits.append("码本已注入：" + " ".join(
            "%s=%s" % (k, os.path.basename(v)) for k, v in cbs.items()))
    miss = codebooks_missing(engine_path)
    if miss:
        bits.append("警告：仓库里没找到码本文件 %s，"
                    "turbo*_tcq 档位会不可用。" % ",".join(miss))
    return "；".join(bits)


# --------------------------------------------------------------------------- #
# 默认引擎目录猜测
# --------------------------------------------------------------------------- #

# 这几个位置都放着一份 buun-llama-cpp 的构建；顺序即优先级。
DEFAULT_ENGINE_HINTS: Tuple[str, ...] = (
    r"E:\AID\buun-llama-cpp",
    r"D:\AID\buun-llama-cpp",
    r"C:\AID\buun-llama-cpp",
    r"E:\buun-llama-cpp",
)


def default_engine_candidates(extra: Sequence[str] = ()) -> List[str]:
    """猜一猜 buun 引擎装在哪：环境变量 → 常见盘位 → 调用方给的候选。"""
    out: List[str] = []

    def add(p: str) -> None:
        p = str(p or "").strip().strip('"')
        if p and p not in out:
            out.append(p)

    for var in ("BUUN_LLAMA_CPP", "BUUN_LLAMA_CPP_DIR"):
        add(os.environ.get(var, ""))
    for p in DEFAULT_ENGINE_HINTS:
        add(p)
    for p in extra:
        add(p)
    return out
