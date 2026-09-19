# -*- coding: utf-8 -*-
"""模型库：扫描目录、分类、能力探测、按列排序。

分类规则（LLMS / Embedding / Drafters）与能力图标（推理 / 多模态 / 工具调用）
都尽量基于 GGUF 元数据而不是文件名，文件名只作为兜底。
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import gguf as G

CATEGORIES: Tuple[str, ...] = ("LLMS", "Embedding", "Drafters")
CATEGORY_ZH = {"LLMS": "LLM", "Embedding": "Embedding", "Drafters": "Drafter"}

# 表格列：(列 id, 中文标题, 英文标题, 宽度, 对齐)
COLUMNS: Tuple[Tuple[str, str, str, int, str], ...] = (
    ("name", "模型名", "Model", 300, "w"),
    ("category", "LLM", "Kind", 78, "center"),
    ("arch", "架构", "Arch", 104, "w"),
    ("params", "参数量", "Params", 78, "e"),
    ("publisher", "发布者", "Publisher", 104, "w"),
    ("pip", "能力", "Capabilities", 96, "center"),
    ("quant", "量化规格", "Quant", 92, "w"),
    ("size", "尺寸", "Size", 78, "e"),
    ("mtime", "修改日期", "Modified", 116, "center"),
    ("params_set", "参数", "Preset", 62, "center"),
)

COLUMN_IDS = tuple(c[0] for c in COLUMNS)

# 「能力」列只显示两个：多模态（能读图）、MTP 预测头。
# 推理链 / 工具调用那两项用户要求去掉（它们只是聊天模板里的关键字，
# 对「这个模型能不能这么用」几乎没有指导意义）。
#
# ⚠️ 用**文字**而不是符号：原来是 ◈ / ✦ / ·，用户反馈「看不出是什么意思」，
#    而 · 在表格里只像一粒噪点。现在没有的能力直接**留空**。
CAP_VISION = "图像"      # 配了多模态投影，能读图
CAP_MTP = "MTP"          # 带 MTP 预测头，推测方式可以直接选 MTP
ICO_SET = "\u2605"       # ★ 已单独调参

LEGEND = ("图像 = 能读图（带多模态投影）    "
          "MTP = 带 MTP 预测头（推测解码可直接选 MTP）    "
          "空白 = 这两项都没有    ★ = 已单独调过参数")

_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})$")


def _split_part(path: str) -> Tuple[bool, int]:
    """(是否分片, 第几片)"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _SPLIT_RE.search(stem)
    if m:
        return True, int(m.group(1))
    return False, 1


def _tokens(path: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for sep in ("-", "_", ".", " "):
        stem = stem.replace(sep, "\x00")
    return [t for t in stem.split("\x00") if t]


def classify(path: str, info: Optional[Dict[str, Any]] = None) -> str:
    arch = str((info or {}).get("arch") or "").lower()
    toks = _tokens(path)
    joined = " ".join(toks)
    # 草稿模型有独立架构 dflash（实测：DSpark / DFlash2 那批全是它），
    # 这条比名字匹配可靠，所以先判。
    if arch == "dflash" or (info or {}).get("is_drafter"):
        return "Drafters"
    if arch in G.ARCH_EMBEDDING:
        return "Embedding"
    if any(t in joined for t in G.NAME_DRAFTER):
        return "Drafters"
    if any(k in joined for k in G.NAME_EMBEDDING):
        return "Embedding"
    return "LLMS"


def discover(dirs: Sequence[str], recursive: bool = True, depth: int = 3,
             progress: Optional[Callable[[int, int], None]] = None
             ) -> List[str]:
    """找出目录下所有 GGUF（跳过 mmproj 与分片的第 2 片之后）。"""
    out: List[str] = []
    for d in dirs:
        d = os.path.expanduser(str(d).strip().strip('"'))
        if not os.path.isdir(d):
            continue
        base_depth = d.rstrip("\\/").count(os.sep)
        for root, subdirs, files in os.walk(d):
            if not recursive:
                subdirs[:] = []
            elif depth > 0 and (root.count(os.sep) - base_depth) >= depth:
                subdirs[:] = []
            for fn in files:
                if not fn.lower().endswith(".gguf"):
                    continue
                if G.looks_like_mmproj(fn):
                    continue
                full = os.path.join(root, fn)
                split, part = _split_part(full)
                if split and part != 1:
                    continue
                out.append(os.path.normpath(full))
                if progress:
                    progress(len(out), len(out))
    return out


def _info_complete(info: Any) -> bool:
    """缓存里的 info 是不是「本版本该有的字段都齐」。

    INFO_TAG 忘了 +1 时的第二道保险：缺字段就当缓存无效、重读 GGUF。
    （踩过：加了 has_mtp 但 tag 没动，于是界面上一个 MTP 都不显示，
    而直接读 GGUF 又完全正常 —— 这种「只错在缓存路径上」的 bug 极难查。）
    """
    if not isinstance(info, dict):
        return False
    return all(k in info for k in G.INFO_REQUIRED)


def build_row(path: str, info: Dict[str, Any], mmproj: Optional[str],
              params_set: bool, depth: int = 99) -> Dict[str, Any]:
    row = dict(info)
    row["path"] = path
    row["mmproj"] = mmproj or ""
    row["has_vision"] = bool(info.get("has_vision") or mmproj)
    row["category"] = classify(path, info)
    row["params_set"] = ICO_SET if params_set else ""
    # MTP 用 GGUF 里的真值（{arch}.nextn_predict_layers），不再靠文件名猜
    row["has_mtp"] = bool(info.get("has_mtp"))
    row["nextn_layers"] = int(info.get("nextn_layers") or 0)
    row["mtp_name_hint"] = any(t in _tokens(path) for t in G.NAME_MTP)

    # 名称 / 发布者优先取目录结构（…\发布者\模型名\xxx.gguf），取不到再退回文件名
    file_name = os.path.basename(path)
    org, model_name, ok = G.split_repo(path, depth)
    if ok:
        row["name"] = model_name
        row["org"] = org
        row["name_from_path"] = True
        if org and org.lower() not in G.GENERIC_DIR_NAMES:
            row["publisher"] = G.org_display(org)
            row["org_display"] = org
        else:
            row["publisher"] = str(info.get("publisher") or "")
            row["org_display"] = ""
    else:
        row["name"] = os.path.splitext(file_name)[0]
        row["org"] = ""
        row["publisher"] = str(info.get("publisher") or "")
        row["org_display"] = ""
        row["name_from_path"] = False
    row["file_name"] = file_name

    row["pip"] = "   ".join(p for p in (
        CAP_VISION if row["has_vision"] else "",
        CAP_MTP if row["has_mtp"] else "",
    ) if p)          # 都没有 → 空字符串（界面上就是空白）
    row["mtime_text"] = time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(info.get("mtime") or 0))
    row["params_sort"] = int(info.get("params") or 0)
    row["sort_name"] = str(row["name"]).lower()
    row["sort_category"] = CATEGORIES.index(row["category"])
    row["sort_arch"] = str(info.get("arch") or "").lower()
    row["sort_publisher"] = str(row["publisher"] or "").lower()
    row["sort_pip"] = (1 if row["has_vision"] else 0) * 2 + \
                      (1 if row["has_mtp"] else 0)
    row["sort_quant"] = str(info.get("quant") or "").lower()
    row["sort_size"] = int(info.get("size") or 0)
    row["sort_mtime"] = float(info.get("mtime") or 0)
    row["sort_params_set"] = 1 if params_set else 0
    return row


def index_mmproj(dirs: Sequence[str], recursive: bool = True,
                 depth: int = 3) -> Dict[str, str]:
    """按目录 + 前缀索引 mmproj 文件，用于给多模态模型自动配上投影文件。"""
    found: Dict[str, str] = {}
    for d in dirs:
        d = os.path.expanduser(str(d).strip().strip('"'))
        if not os.path.isdir(d):
            continue
        base_depth = d.rstrip("\\/").count(os.sep)
        for root, subdirs, files in os.walk(d):
            if not recursive:
                subdirs[:] = []
            elif depth > 0 and (root.count(os.sep) - base_depth) >= depth:
                subdirs[:] = []
            for fn in files:
                if fn.lower().endswith(".gguf") and G.looks_like_mmproj(fn):
                    found.setdefault(os.path.normcase(root),
                                     os.path.join(root, fn))
    return found


_MMPROJ_GENERIC = frozenset((
    "mmproj", "gguf", "f16", "f32", "bf16", "q8", "vision", "projector",
    "clip", "model", "v", "1", "2", "3", "of", "audio",
))


def pick_mmproj(model_path: str, mmproj_by_dir: Dict[str, str]) -> Optional[str]:
    """同目录的 mmproj 只在该文件名与模型有共同特征词时才配对，
    避免一个目录里放了好几个模型的投影文件时全部误判成多模态。"""
    key = os.path.normcase(os.path.dirname(model_path))
    cand = mmproj_by_dir.get(key)
    if not cand:
        return None
    m_toks = set(_tokens(model_path))
    p_toks = set(_tokens(cand)) - _MMPROJ_GENERIC
    meaningful = p_toks - {"gguf"}
    if not meaningful:
        return cand
    if m_toks & meaningful:
        return cand
    return None


def scan(paths: Iterable[str], mmproj_by_dir: Optional[Dict[str, str]] = None,
         params_set_keys: Optional[Iterable[str]] = None,
         cache: Optional[Dict[str, Dict[str, Any]]] = None,
         progress: Optional[Callable[[int, int], None]] = None,
         use_cache: bool = True,
         roots: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """逐文件解析（带按 mtime+size 的缓存），返回表格行列表。

    ``roots`` 是本次扫描的根目录列表：相对根的层级决定了能否从目录结构里
    解析出「发布者 / 模型名」（根\\发布者\\模型名\\文件）。
    """
    mmproj_by_dir = mmproj_by_dir or {}
    keyset = set(params_set_keys or ())
    cache = cache if cache is not None else {}
    from .store import model_key as _mk
    norm_roots = []
    for d in roots or ():
        d = os.path.expanduser(str(d).strip().strip('"'))
        if d:
            norm_roots.append(os.path.normcase(os.path.normpath(d)))
    plist = list(paths)
    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(plist, 1):
        if progress:
            progress(i, len(plist))
        try:
            st = os.stat(path)
            # stamp 里带上缓存格式版本：GGUF 文件本身没变，但 analyse() 的
            # 返回字段变了（比如这次加的 has_mtp），老缓存会让新字段永远缺席。
            stamp = "%s:%d:%d" % (G.INFO_TAG, int(st.st_mtime), st.st_size)
        except OSError:
            continue
        entry = cache.get(path)
        if not (use_cache and entry and entry.get("stamp") == stamp
                and _info_complete(entry.get("info"))):
            info = G.analyse(path)
            if not info:
                continue
            entry = {"stamp": stamp, "info": info}
            cache[path] = entry
        info = entry["info"]
        rows.append(build_row(path, info,
                              pick_mmproj(path, mmproj_by_dir),
                              _mk(path) in keyset,
                              depth=G.depth_from_roots(path, norm_roots)))
    rows.sort(key=lambda r: r["sort_name"])
    return rows


def sort_rows(rows: List[Dict[str, Any]], column: str,
              reverse: bool = False) -> List[Dict[str, Any]]:
    """按列排序。column 用 COLUMNS 里的 id。"""
    fake = {"params": "params_sort", "mtime": "sort_mtime"}
    key = fake.get(column) or ("sort_" + column)
    if column not in COLUMN_IDS:
        key = "sort_name"
    rows.sort(key=lambda r: r.get(key, ""), reverse=reverse)
    return rows


def filter_rows(rows: List[Dict[str, Any]], category: str = "全部",
                text: str = "", only_set: bool = False,
                cap: str = "") -> List[Dict[str, Any]]:
    out = rows
    if category and category != "全部":
        want = {"LLM": "LLMS", "LLMS": "LLMS", "Embedding": "Embedding",
                "Drafter": "Drafters", "Drafters": "Drafters"}.get(category,
                                                                   category)
        out = [r for r in out if r["category"] == want]
    if only_set:
        out = [r for r in out if r.get("params_set")]
    if cap == "vision":
        out = [r for r in out if r.get("has_vision")]
    elif cap == "mtp":
        out = [r for r in out if r.get("has_mtp")]
    q = (text or "").strip().lower()
    if q:
        def hit(r):
            return (q in str(r.get("name", "")).lower()
                    or q in str(r.get("file_name", "")).lower()
                    or q in str(r.get("org", "")).lower()
                    or q in str(r.get("arch", "")).lower()
                    or q in str(r.get("publisher", "")).lower()
                    or q in str(r.get("quant", "")).lower())
        out = [r for r in out if hit(r)]
    return out
