# -*- coding: utf-8 -*-
"""GGUF 只读解析。

读取头部元数据(key/value) + 张量清单(name/shape/type)，不加载任何权重数据。
张量清单让我们能算出**真实参数量**（而不是从文件名猜），元数据里的对话模板则用来
判断模型是否支持推理链(thinking)与工具调用。
"""

from __future__ import annotations

import os
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

_STR, _ARR = 8, 9
_FIXED = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "IQ4_KS", 40: "IQ2_KS", 41: "IQ3_KS", 42: "IQ1_KT",
}

# 文件名里常见的量化标记（匹配时按长度倒序，避免 Q4_K 抢先于 Q4_K_M）
QUANT_TOKENS = (
    "IQ1_S", "IQ1_M", "IQ1_KT", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ2_KS", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ3_KS", "IQ3_KT",
    "IQ4_KSS", "IQ4_NL", "IQ4_XS", "IQ4_KS", "IQ5_KS", "IQ6_KS",
    "Q2_K_S", "Q2_K", "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
    "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_1", "Q4_0",
    "Q5_K_XL", "Q5_K_M", "Q5_K_S", "Q5_K", "Q5_1", "Q5_0",
    "Q6_K", "Q8_0", "Q8_K", "MXFP4_MOE", "MXFP4", "BF16", "F16", "F32",
    "FP16", "FP32", "TQ1_0", "TQ2_0", "IQ4_K", "IQ3_K", "IQ2_K", "IQ1_K",
    "Q4_0_4_4", "Q4_0_8_8",
)

ARCH_EMBEDDING = frozenset((
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3",
    "jina-embeddings-v3", "gte", "e5", "bge", "bge-m3", "bge-reranker",
    "bge-reranker-v2", "xlm-roberta", "roberta", "modernbert", "modern-bert",
    "qwen2-embedding", "qwen3-embedding", "gemma-embedding", "embeddinggemma",
))

NAME_EMBEDDING = (
    "embed", "bge", "gte-", "gte_", "e5-", "-e5", "nomic", "jina",
    "reranker", "rerank", "colbert", "mxbai", "arctic-embed", "text-embedding",
    "voyage", "minilm", "mpnet",
)

NAME_DRAFTER = (
    "draft", "dflash", "dspark", "eagle", "speculator", "assistant",
    "nextn", "medusa", "lookahead",
)
# 名字里带这些词说明是「带多 Token 预测头的主模型」，不是草稿模型
NAME_MTP = ("mtp", "nextn")

# 文件名前缀 -> 模型发布方
PUBLISHERS = (
    ("qwen", "Qwen"), ("qwq", "Qwen"),
    ("deepseek", "DeepSeek"),
    ("llama", "Meta"), ("meta-llama", "Meta"),
    ("mistral", "Mistral AI"), ("magistral", "Mistral AI"),
    ("devstral", "Mistral AI"), ("mixtral", "Mistral AI"),
    ("codestral", "Mistral AI"), ("pixtral", "Mistral AI"),
    ("gemma", "Google"), ("medgemma", "Google"),
    ("phi", "Microsoft"), ("granite", "IBM"),
    ("glm", "Zai/THUDM"), ("chatglm", "Zai/THUDM"),
    ("minimax", "MiniMax"), ("kimi", "Moonshot"), ("moonshot", "Moonshot"),
    ("yi-", "01.AI"), ("internlm", "Shanghai AI Lab"), ("baichuan", "Baichuan"),
    ("exaone", "LG AI"), ("nemotron", "NVIDIA"), ("olmo", "Allen AI"),
    ("command", "Cohere"), ("c4ai", "Cohere"), ("gpt-oss", "OpenAI"),
    ("lfm", "Liquid AI"), ("seed-oss", "ByteDance"), ("hunyuan", "Tencent"),
    ("ernie", "Baidu"), ("step-", "StepFun"), ("dots", "Rednote"),
    ("smollm", "HuggingFace"), ("zephyr", "HuggingFace"),
    ("solar", "Upstage"), ("pharia", "Aleph Alpha"), ("apertus", "Swiss AI"),
    ("grok", "xAI"), ("ring", "Ant Group"), ("ling", "Ant Group"),
    ("bailing", "Ant Group"), ("pangu", "Huawei"),
    ("telechat", "China Telecom"), ("gemma3", "Google"),
)

UPLOADERS = (
    "bartowski", "unsloth", "mradermacher", "thebloke", "ubergarm", "lmstudio",
    "ggml-org", "maziyarpanahi", "quantfactory", "tensorblock", "nightmedia",
    "duyntnet", "cognitivecomputations", "nisten", "worstplayer", "gguf",
)

# 这些目录名太通用，不适合当作「发布者」或「模型名」
GENERIC_DIR_NAMES = frozenset((
    "models", "model", "gguf", "ggufs", "llm", "llms", "llama", "main",
    "download", "downloads", "huggingface", "hub", "cache", "snapshots",
    "new", "quant", "quants", "local", "大模型", "模型", "模型库",
))

# 常见 HF 组织 id -> 显示名（没在这里的就用原始 id）
KNOWN_ORGS = {
    "qwen": "Qwen", "qwen-team": "Qwen",
    "deepseek-ai": "DeepSeek", "deepseek": "DeepSeek",
    "meta-llama": "Meta", "meta": "Meta",
    "mistralai": "Mistral AI", "mistral-community": "Mistral AI",
    "google": "Google", "google-gemma": "Google",
    "microsoft": "Microsoft", "ibm-granite": "IBM", "ibm": "IBM",
    "zai-org": "Zai/THUDM", "thudm": "Zai/THUDM", "zhipuai": "Zai/THUDM",
    "moonshotai": "Moonshot", "moonshot": "Moonshot",
    "01-ai": "01.AI", "internlm": "Shanghai AI Lab",
    "nvidia": "NVIDIA", "openai": "OpenAI", "allenai": "Allen AI",
    "cohere": "Cohere", "cohere-for-ai": "Cohere",
    "liquidai": "Liquid AI", "liquid4all": "Liquid AI",
    "tencent": "Tencent", "tencent-hunyuan": "Tencent", "baidu": "Baidu",
    "lgai-exaone": "LG AI", "exaone": "LG AI", "xai-org": "xAI",
    "baichuan-inc": "Baichuan", "minimax": "MiniMax",
    "stepfun-ai": "StepFun", "rednote-hilab": "Rednote",
    "bytedance": "ByteDance", "swiss-ai": "Swiss AI", "upstage": "Upstage",
    "inclusionai": "Ant Group", "openbmb": "OpenBMB",
}


def split_repo(path: str, depth: int = 99) -> Tuple[str, str, bool]:
    """从 HuggingFace 风格的目录结构里取 (组织 id, 模型名, 是否取到)。

    ``E:\\LM_models\\huihui-ai\\Huihui-Qwen3.8-27B-abliterated-GGUF\\xxx.gguf``
    相对扫描根有 3 层，返回 ``("huihui-ai", "Huihui-Qwen3.8-27B-abliterated", True)``。

    ``depth`` 是文件所在目录相对扫描根的层数（目录数）：

    * 0（文件直接躺在扫描根目录）→ 没有目录结构，返回 False
    * 1（根\\模型名\\文件）→ 只取模型名，没有发布者
    * >=2（根\\发布者\\模型名\\文件）→ 取发布者 + 模型名

    目录名太通用（models / gguf / 模型 之类）时同样不取。
    """
    if depth < 1:
        return "", "", False
    norm = os.path.normpath(path)
    parent_dir = os.path.dirname(norm)
    org_dir = os.path.dirname(parent_dir)
    parent = os.path.basename(parent_dir)
    org = os.path.basename(org_dir)
    if not parent or parent.lower() in GENERIC_DIR_NAMES:
        return "", "", False
    model_name = re.sub(r"[\s._-]*GGUF$", "", parent, flags=re.I).strip(" -_.")
    if not model_name or model_name.lower() in GENERIC_DIR_NAMES:
        return "", "", False
    if depth < 2:
        return "", model_name, True
    if (not org or org.lower() in GENERIC_DIR_NAMES
            or (len(org) <= 3 and org[1:2] == ":")):
        return "", model_name, True
    return org, model_name, True


def depth_from_roots(path: str, roots) -> int:
    """文件所在目录相对某个扫描根的层数（取最近的那个根）。"""
    parent = os.path.normpath(os.path.dirname(os.path.normpath(path)))
    best = None
    for r in roots or ():
        r = os.path.normpath(r)
        try:
            rel = os.path.relpath(parent, r)
        except ValueError:
            continue
        if rel == ".." or rel.startswith(".." + os.sep):
            continue
        n = 0 if rel == "." else len(rel.split(os.sep))
        best = n if best is None else min(best, n)
    return best if best is not None else 1


def org_display(org: str) -> str:
    """组织 id 转显示名；没登记过的保持原样（如 huihui-ai）。"""
    if not org:
        return ""
    return KNOWN_ORGS.get(org.lower(), org)


def _read(rf, fmt: str, size: int):
    buf = rf.read(size)
    if len(buf) != size:
        raise EOFError
    return struct.unpack(fmt, buf)[0]


def _read_string(rf) -> str:
    n = _read(rf, "<Q", 8)
    if n > 1 << 24:
        raise ValueError("bad string length")
    return rf.read(n).decode("utf-8", errors="replace")


def _read_value(rf, vtype: int) -> Any:
    if vtype == _STR:
        return _read_string(rf)
    if vtype == _ARR:
        etype = _read(rf, "<I", 4)
        count = _read(rf, "<Q", 8)
        if count > 8192 or etype == _ARR:
            if etype == _STR:
                for _ in range(count):
                    _read_string(rf)
            elif etype == _ARR:
                for _ in range(count):
                    _read_value(rf, _ARR)
            else:
                fmt, size = _FIXED.get(etype, ("<B", 1))
                rf.seek(size * count, os.SEEK_CUR)
            return None
        return [_read_value(rf, etype) for _ in range(count)]
    if vtype in _FIXED:
        fmt, size = _FIXED[vtype]
        return _read(rf, fmt, size)
    raise ValueError("unknown gguf value type %s" % vtype)


def read_header(path: str, want_tensors: bool = True, max_kv: int = 16384,
                max_tensors: int = 131072
                ) -> Tuple[Dict[str, Any], List[Tuple[str, Tuple[int, ...], int]]]:
    """解析 GGUF 头。返回 (元数据, [(张量名, 形状, 类型)])。"""
    kv: Dict[str, Any] = {}
    tensors: List[Tuple[str, Tuple[int, ...], int]] = []
    with open(path, "rb") as rf:
        if rf.read(4) != b"GGUF":
            raise ValueError("不是 GGUF 文件")
        version = _read(rf, "<I", 4)
        n_tensors = _read(rf, "<Q", 8)
        n_kv = _read(rf, "<Q", 8)
        for _ in range(min(n_kv, max_kv)):
            key = _read_string(rf)
            vtype = _read(rf, "<I", 4)
            try:
                kv[key] = _read_value(rf, vtype)
            except (EOFError, ValueError):
                break
        kv["__version__"] = version
        if not want_tensors or n_kv > max_kv:
            return kv, tensors
        for _ in range(min(int(n_tensors), max_tensors)):
            try:
                tname = _read_string(rf)
                ndim = _read(rf, "<I", 4)
                if ndim > 8:
                    break
                dims = tuple(int(_read(rf, "<Q", 8)) for _ in range(ndim))
                ttype = _read(rf, "<I", 4)
                _read(rf, "<Q", 8)  # offset
            except (EOFError, ValueError):
                break
            tensors.append((tname, dims, ttype))
    return kv, tensors


# --------------------------------------------------------------------------- #
# 推断辅助
# --------------------------------------------------------------------------- #

def quant_from_name(name: str) -> str:
    up = os.path.basename(name).upper()
    for tok in QUANT_TOKENS:
        if tok in up:
            return tok
    return ""


def publisher_from_name(name: str) -> Tuple[str, str]:
    """返回 (模型发布方, 转换/上传者)。"""
    low = os.path.basename(name).lower().replace("_", "-")
    uploader = ""
    for up in UPLOADERS:
        if up in low:
            uploader = up
            break
    publisher = ""
    for pref, pub in PUBLISHERS:
        if low.startswith(pref) or ("-" + pref) in low:
            publisher = pub
            break
    return publisher, uploader


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([BMT])(?![a-zA-Z])")
_ACTIVE_RE = re.compile(r"(?:^|[-_])a(\d+(?:\.\d+)?)b(?:[-_.]|$)", re.I)


def size_hint_from_name(name: str) -> Tuple[str, str]:
    """从文件名里提取 (总参数量提示, 激活参数量提示)，仅作展示兜底。"""
    base = os.path.basename(name)
    total = ""
    for m in _SIZE_RE.finditer(base):
        if m.group(2) in ("B", "T"):
            total = m.group(1) + m.group(2)
            break
    active = ""
    m = _ACTIVE_RE.search(base)
    if m:
        active = "A" + m.group(1) + "B"
    return total, active


def human_params(n: float) -> str:
    if not n:
        return ""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            val = n / div
            return ("%.0f %s" % (val, unit)) if val >= 100 else \
                   ("%.1f %s" % (val, unit))
    return "%d" % n


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return ("%.0f %s" % (num, unit)) if unit == "B" else \
                   ("%.2f %s" % (num, unit))
        num /= 1024.0
    return "%.2f TB" % num


_TEMPLATE_KEYS = ("tokenizer.chat_template",
                  "tokenizer.chat.ggml.chat_template", "chat_template")


def _find_template(kv: Dict[str, Any]) -> str:
    for key in _TEMPLATE_KEYS:
        val = kv.get(key)
        if isinstance(val, str) and val:
            return val
    for key, val in kv.items():
        if isinstance(val, str) and "chat_template" in key and val:
            return val
    return ""


def head_dim_of(kv: Dict[str, Any], arch: str, n_embd: Any,
                tensors: Sequence[Any] = ()) -> Tuple[Optional[int], str]:
    """推 K 的 head_dim（turbo / TCQ 的 KV block 是 128，要求它是 128 的倍数）。

    取值优先级：
      1. ``<arch>.attention.key_length``（新一点的 GGUF 直接写了）
      2. ``<arch>.attention.value_length``
      3. ``embedding_length / attention.head_count``
      4. 张量形状兜底：``blk.0.attn_k_norm.weight`` 或 ``attn_q_norm.weight``
         的第一维就是 head_dim；没有 norm 就用 ``attn_k.weight`` 的第二维
         除以 ``attention.head_count_kv``。
    """
    if arch:
        for suffix in ("attention.key_length", "attention.value_length"):
            v = kv.get("%s.%s" % (arch, suffix))
            if isinstance(v, int) and v > 0:
                return int(v), suffix.replace("attention.", "")
    if arch and isinstance(n_embd, int) and n_embd > 0:
        heads = kv.get("%s.attention.head_count" % arch)
        if isinstance(heads, int) and heads > 0 and n_embd % heads == 0:
            return int(n_embd // heads), "embedding_length/head_count"

    want = {"blk.0.attn_k_norm.weight": 0, "blk.0.attn_q_norm.weight": 0,
            "blk.0.attn_k.weight": 1}
    n_kv = kv.get("%s.attention.head_count_kv" % arch) if arch else None
    for name, dims, _t in (tensors or ()):
        idx = want.get(name)
        if idx is None or len(dims) <= idx:
            continue
        val = int(dims[idx])
        if name.endswith("attn_k.weight"):
            if isinstance(n_kv, int) and n_kv > 0 and val % n_kv == 0:
                return int(val // n_kv), "attn_k.weight/head_count_kv"
            continue
        if val > 0:
            return val, name
    return None, ""


def analyse(path: str, want_tensors: bool = True) -> Optional[Dict[str, Any]]:
    """把 GGUF 整理成界面需要的一行信息。读不出来返回 None。"""
    try:
        kv, tensors = read_header(path, want_tensors=want_tensors)
    except Exception:
        return None
    if not kv:
        return None

    arch = str(kv.get("general.architecture", "") or "")
    gguf_name = str(kv.get("general.name", "") or os.path.basename(path))
    block = kv.get("%s.block_count" % arch) if arch else None
    ctx_train = kv.get("%s.context_length" % arch) if arch else None
    experts = kv.get("%s.expert_count" % arch) if arch else None
    n_embd = kv.get("%s.embedding_length" % arch) if arch else None

    params = 0
    for _n, dims, _t in tensors:
        acc = 1
        for d in dims:
            acc *= int(d)
        params += acc
    hint_total, hint_active = size_hint_from_name(path)
    if not params and hint_total:
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
        params = int(float(hint_total[:-1]) * mult[hint_total[-1]])

    template = _find_template(kv)
    low_tpl = template.lower()
    vision_keys = any(k.startswith(("clip.", "vision.")) or k in (
        "clip.has_vision_encoder", "clip.has_audio_encoder") for k in kv)

    try:
        st = os.stat(path)
        size, mtime = st.st_size, st.st_mtime
    except OSError:
        size, mtime = 0, 0.0

    file_type = FILE_TYPES.get(kv.get("general.file_type"),
                               str(kv.get("general.file_type", "") or ""))
    quant = quant_from_name(path) or file_type
    publisher, uploader = publisher_from_name(path)
    author = str(kv.get("general.author", "") or
                 kv.get("general.organization", "") or "")

    try:
        version = int(kv.get("__version__") or 0)
    except (TypeError, ValueError):
        version = 0

    head_dim, head_src = head_dim_of(kv, arch, n_embd, tensors)

    return {
        "path": path,
        "name": os.path.basename(path),
        "gguf_name": gguf_name,
        "arch": arch,
        "layers": int(block) if isinstance(block, int) else None,
        "ctx_train": int(ctx_train) if isinstance(ctx_train, int) else None,
        "experts": int(experts) if isinstance(experts, int) else None,
        "n_embd": int(n_embd) if isinstance(n_embd, int) else None,
        "head_dim": head_dim,
        "head_dim_src": head_src,
        # turbo / TCQ / VBR 的 KV block 是 128 个值，head_dim 必须是 128 的倍数
        "block128_ok": (None if head_dim is None else head_dim % 128 == 0),
        "params": int(params) if params else None,
        "params_hint": hint_total,
        "active_hint": hint_active,
        "quant": quant,
        "file_type": file_type,
        "publisher": publisher or author,
        "author": author,
        "uploader": uploader,
        "size": size,
        "mtime": mtime,
        "n_tensors": len(tensors),
        "template": template,
        "has_think": ("think" in low_tpl) or ("reasoning" in low_tpl),
        "has_tools": ("tool" in low_tpl) or ("function" in low_tpl),
        "has_vision": bool(vision_keys),
        "size_text": human_size(size) if size else "",
        "params_text": human_params(params) if params else hint_total,
        "version": version,
        "split": bool(re.search(r"-\d{5}-of-\d{5}$",
                                os.path.splitext(os.path.basename(path))[0])),
    }


def looks_like_mmproj(filename: str) -> bool:
    low = os.path.basename(filename).lower()
    return low.startswith("mmproj") or "mmproj" in low
