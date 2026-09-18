# -*- coding: utf-8 -*-
"""把界面上的参数值翻译成命令行：定位可执行文件、拼参数、校验、生成 ini / .bat。"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import (Any, Dict, FrozenSet, Iterable, List, Optional,
                    Sequence, Tuple)

from . import schema as S

EXE_SUFFIX = ".exe" if os.name == "nt" else ""


# --------------------------------------------------------------------------- #
# 定位可执行文件
# --------------------------------------------------------------------------- #

def exe_names(mode: str) -> Tuple[str, ...]:
    base = S.MODES[mode]["exe"]
    return (base + EXE_SUFFIX, base)


def candidate_dirs(root: str) -> Tuple[str, ...]:
    return (
        root,
        os.path.join(root, "bin"),
        os.path.join(root, "build", "bin"),
        os.path.join(root, "build", "bin", "Release"),
        os.path.join(root, "out", "bin"),
    )


def resolve_exe(engine_path: str, mode: str) -> Optional[str]:
    """engine_path 可以是目录，也可以直接是某个 .exe。"""
    if not engine_path:
        return None
    p = os.path.expanduser(str(engine_path).strip().strip('"'))
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        return None
    for d in candidate_dirs(p):
        for name in exe_names(mode):
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
    try:
        for entry in os.scandir(p):
            if entry.is_dir():
                for name in exe_names(mode):
                    cand = os.path.join(entry.path, name)
                    if os.path.isfile(cand):
                        return cand
    except OSError:
        pass
    return None


def detect_exes(engine_path: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for mode in S.MODE_ORDER:
        p = resolve_exe(engine_path, mode)
        if p:
            found[mode] = p
    return found


# --------------------------------------------------------------------------- #
# 拼参数
# --------------------------------------------------------------------------- #

def _split_line(line: str) -> List[str]:
    """按空白拆分，支持双引号且不解释反斜杠（Windows 路径友好）。"""
    out: List[str] = []
    buf: List[str] = []
    in_quote = False
    for ch in line:
        if ch == '"':
            if in_quote:
                out.append("".join(buf))
                buf = []
                in_quote = False
            else:
                if buf:
                    out.append("".join(buf))
                    buf = []
                in_quote = True
        elif ch.isspace() and not in_quote:
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def numeric_ok(raw: str, kind: str) -> bool:
    """值能不能当数字传（挡位→数字框这类类型变更后用来兜底）。"""
    try:
        if kind == S.K_INT:
            int(raw)
        else:
            float(raw)
        return True
    except (TypeError, ValueError):
        return False


def _as_float_or_none(v: Any) -> Optional[float]:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_lora(raw: Any) -> List[Tuple[str, Optional[float]]]:
    """解析 LoRA 列表，返回 [(路径, 比例或 None)]。

    两种写法都认：
    * JSON（界面存的就是这种）：``[{"path":"a.gguf","scale":0.8}, ...]``
    * 纯文本：每行一个路径，行尾可以用 ``|`` 或 Tab 跟一个比例
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        return []
    out: List[Tuple[str, Optional[float]]] = []
    if text[:1] == "[":
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    path = str(item.get("path") or "").strip()
                    scale = _as_float_or_none(item.get("scale"))
                else:
                    path, scale = str(item).strip(), None
                if path:
                    out.append((path, scale))
            return out
    for line in text.splitlines():
        line = line.strip().strip('"')
        if not line:
            continue
        for sep in ("\t", "|"):
            if sep in line:
                head, _, tail = line.rpartition(sep)
                val = _as_float_or_none(tail)
                if val is not None:
                    line = head.strip()
                    out.append((line, val)) if line else None
                    break
        else:
            out.append((line, None))
    return out


def lora_args(raw: Any) -> List[str]:
    """一个 LoRA 条目展开成 --lora 或 --lora-scaled 两个/三个 token。"""
    out: List[str] = []
    for path, scale in parse_lora(raw):
        if scale is None or abs(scale - 1.0) < 1e-9:
            out += ["--lora", path]
        else:
            out += ["--lora-scaled", path, ("%g" % scale)]
    return out


def _merge_kwargs(argv: List[str]) -> List[str]:
    """``--chat-template-kwargs`` 只能出现一次：多个来源合并成一个 JSON。

    「推理强度」挡位和「模板附加参数」两个控件写的都是这个 flag，
    直接拼会产出两个同名参数（后者覆盖前者，挡位就白选了）。
    """
    flag = "--chat-template-kwargs"
    pos = [i for i, a in enumerate(argv) if a == flag]
    if len(pos) < 2:
        return argv
    merged: Dict[str, Any] = {}
    for i in pos:
        raw = argv[i + 1] if i + 1 < len(argv) else ""
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            merged.update(data)
    if not merged:
        return argv
    drop = set()
    for i in pos:
        drop.add(i)
        if i + 1 < len(argv):
            drop.add(i + 1)
    out: List[str] = []
    done = False
    for i, a in enumerate(argv):
        if i in drop:
            if a == flag and not done:
                done = True
                out += [flag, json.dumps(merged, ensure_ascii=False)]
            continue
        out.append(a)
    if not done:
        out += [flag, json.dumps(merged, ensure_ascii=False)]
    return out


def valid_choice(f: S.F, raw: str) -> str:
    """把「存档里遗留的旧挡位值」修正成当前挡位表里的值。

    挡位改名 / 增删之后，config.json 里可能还存着老名字（例如推理强度曾经叫
    「不限」，现在是 ``xhigh``）。老名字不在 choices 里，直接把它拼进命令行
    就会变成 ``--reasoning-budget 不限`` 这种引擎看不懂的参数。
    所以：认得就照用，认不得退回默认挡位，默认挡位也没有就干脆不传。
    """
    if not raw:
        return ""
    # multi_value（如 --spec-type 的逗号列表）：整体必然不在 choices 里，
    # 一律原样保留，交给 validate 逐个名字报「不认识」。
    if f.multi_value:
        return raw
    if not f.choices or raw in f.choices:
        return raw
    default = str(f.default_value() or "")
    if f.argmap:
        return default if default in f.argmap else ""
    return default if (not f.choices or default in f.choices) else ""


def _field_args(f: S.F, on: bool, value: Any) -> List[str]:
    if not on:
        return []
    raw = "" if value is None else str(value).strip()

    if f.kind == S.K_BOOL:
        return [f.flag] if f.flag else []

    # LoRA：一个条目可能展开成 --lora 或 --lora-scaled
    if f.key == "lora":
        return lora_args(raw)
    # --spec-draft-replace TARGET DRAFT 要两个值，界面上用空格分隔
    if f.key == "spec_draft_replace":
        parts = _two_value_split(raw)
        return ([f.flag] + parts) if len(parts) == 2 else []
    if f.kind == S.K_META:
        return []                  # 只用于界面组装，本身不产出参数

    # 纯界面设置（有挡位/勾选框，但没有对应的命令行参数）一律不产出任何 token。
    # 否则「运行方式」「空闲自动卸载」这种值会被当成位置参数塞进命令行。
    if not f.flag and not f.argmap and not f.positional \
            and f.kind not in (S.K_MULTI,):
        return []

    if f.kind == S.K_MULTI:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return []
        if f.flag:
            out: List[str] = []
            for ln in lines:
                out += [f.flag, ln]
            return out
        out = []
        for ln in lines:
            out += _split_line(ln)
        return out

    if not raw:
        return []

    # 数字类参数：值不是数字就干脆不传。挡位改成数字项（或反过来）之后，
    # 存档里遗留的中文挡位名会在这里被拦下，不会变成 `--reasoning-budget 不限`
    if f.kind in (S.K_INT, S.K_FLOAT) and not numeric_ok(raw, f.kind):
        return []

    if f.kind in S.CHOICE_KINDS:
        raw = valid_choice(f, raw)
        if not raw:
            return []
        if f.argmap:
            return list(f.argmap.get(raw, ()))
        return [f.flag, raw] if f.flag else [raw]

    return [f.flag, raw] if f.flag else [raw]


def _two_value_split(raw: str) -> List[str]:
    """把「A B」拆成两个 token（支持引号）。--spec-draft-replace 要两个值。"""
    parts = _split_line(raw)
    return parts[:2]


def build_argv(snapshot: Dict[str, Dict[str, Any]], mode: str,
               model: str = "", role: str = "llm",
               flags: Optional[set] = None) -> List[str]:
    """按参数声明顺序生成参数列表（不含可执行文件本身）。

    * ``model`` 非空时插入 ``-m <model>``（量化模式例外，它用位置参数）
    * 服务页只有一套参数（历史上按 LLM / Embedding 角色分开，现已合并成一个角色）
    * ``flags`` 给定时（来自 ``--help`` 探测），**当前 build 没有的 flag 一律不输出**，
      这样即使某个页面用户没打开过、快照里还留着旧值，也不会把不支持的参数喂给引擎

    界面里只保留**本 build 的 ``--help`` 里确实存在**的参数：上游删掉的
    （``--models-dir`` 路由组、``-a`` 别名、``-kvu``、旧 draft 组等）已经从
    参数表里移除；测速 / 量化 / imatrix 三个模式也已整体删除。
    """
    spec_on = bool((snapshot.get("spec_enable") or {}).get("on"))
    router = S.is_router(snapshot)
    # VBR 那一组参数只有在 KV 档位确实是 vbr 时才允许出现 ——
    # 引擎对「非 vbr 的 KV + --vbr-*」是硬报错：
    #   --vbr-* flags need a VBR cache side
    _k_eff, _v_eff = effective_kv_types(snapshot)
    vbr_on = "vbr" in (_k_eff, _v_eff)

    argv: List[str] = []
    if model and not router:
        # 路由模式下不能给路由器本身塞 -m：那会让它在启动时先加载一个模型，
        # 该模型一旦加载失败（例如 KV 档位跟 head_dim 不匹配）整个进程会直接
        # fail-fast 退出，连路由都起不来。模型一律由预置文件提供。
        argv += ["-m", model]

    for f in S.FIELDS:
        if mode not in f.modes:
            continue
        if f.page == "server" and role not in f.roles:
            continue
        # 路由相关的参数只在「多模型路由」下才输出，
        # 否则 --models-dir 会把 llama-server 悄悄切成路由模式，
        # 而网关这边还在按单模型抢锁换模型 —— 两边打架。
        if f.key in S.ROUTER_KEYS and not router:
            continue
        # KV 不是 vbr 时，VBR 那一组一律不输出（引擎会拒绝启动）
        if f.key in S.VBR_KEYS and not vbr_on:
            continue
        if f.page == "spec":
            if f.key == "spec_enable":
                continue               # 纯开关，本身不产生参数
            if not spec_on:
                continue
        if flags and f.all_flags() and \
                not any(c in flags for c in f.all_flags()):
            continue                   # 当前 build 不认识这个 flag
        st = snapshot.get(f.key) or {}
        argv += _field_args(f, bool(st.get("on")), st.get("value", ""))
    return _merge_kwargs(argv)


def _picked(snapshot: Dict[str, Dict[str, Any]], key: str) -> str:
    """某个字段实际会传出去的值（没勾选就是空串）。"""
    st = snapshot.get(key) or {}
    if not st.get("on"):
        return ""
    f = S.FIELD_BY_KEY.get(key)
    raw = str(st.get("value") or "").strip()
    if f is None:
        return raw
    if f.kind in S.CHOICE_KINDS:
        return valid_choice(f, raw)
    return raw


def vbr_floor_errors(snapshot: Dict[str, Dict[str, Any]]
                     ) -> List[Tuple[str, str]]:
    """--vbr-floor 与 --vbr-entry / --vbr-codec 的三条硬约束。

    都来自 common/arg.cpp 的实际行为：

    1. floor 高于 entry → **直接抛异常**（不是 clamp）：
       ``--vbr-floor (4.125 bits/value) cannot exceed --vbr-entry t2 (2 bits/value)``
       只在 K/V 两侧都可动时才检查，而这正是常规情形。
    2. ``--vbr-codec classic`` 时 turbo 档位非法 → ``unsupported VBR floor``。
       而 codec=auto 在模型不支持 turbo 时会退回 classic，同样会炸。
    3. floor 低于阶梯最低点 → 引擎**只警告并抬到最低点**，不报错。这里不拦，
       只在提示里说明。
    """
    errs: List[Tuple[str, str]] = []
    floor = _picked(snapshot, "vbr_floor")
    entry = _picked(snapshot, "vbr_entry")
    codec = _picked(snapshot, "vbr_codec")

    if floor and codec == "classic" and floor in S.VBR_TURBO_FLOORS:
        errs.append(("vbr_floor",
                     "「降级阶梯」选的是 classic，但最低档位下限写的是 turbo 档"
                     "（%s）—— classic 阶梯只认 q8_0 / q4_0，引擎会直接报 "
                     "「unsupported VBR floor」。\n"
                     "请把下限改成「跟随引擎」，或把阶梯改成 turbo / auto。"
                     % floor))

    if floor and entry:
        fb = S.VBR_BPV.get(floor)
        eb = S.VBR_BPV.get(entry)
        if fb is not None and eb is not None and fb > eb + 1e-9:
            errs.append(("vbr_floor",
                         "最低档位下限（%s = %.4g bits/value）不能高于起始档位"
                         "（%s = %.4g bits/value）。\n"
                         "引擎会直接报错：--vbr-floor cannot exceed --vbr-entry。\n"
                         "把起始档位调回 f16，或把下限压到不高于它。"
                         % (floor, fb, entry, eb)))
    return errs


# --------------------------------------------------------------------------- #
# 模型兼容性：turbo / TCQ / VBR 的 128-block 限制
# --------------------------------------------------------------------------- #

def chosen_kv_types(snapshot: Dict[str, Dict[str, Any]]) -> List[str]:
    """当前勾选的 KV 档位（统一档位 + K/V 两侧，去重）。"""
    out: List[str] = []
    for key in ("ct", "ctk", "ctv"):
        st = snapshot.get(key) or {}
        if not st.get("on"):
            continue
        val = valid_choice(S.FIELD_BY_KEY[key], str(st.get("value") or "").strip())
        if val and val not in out:
            out.append(val)
    return out


def effective_kv_types(snapshot: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    """算出 K / V 两侧**实际生效**的档位（含没显式勾选时的引擎默认）。

    buun 的 -ctk / -ctv 默认都是 ``vbr``；``-ct`` 会同时设两侧，
    但它写在前面，所以后出现的单项会覆盖自己那一侧。
    命令行顺序里 ct 在 ctk/ctv 之前，所以：
        k = ctk 有值 ? ctk : (ct 有值 ? ct : vbr)
    """
    def pick(key: str) -> str:
        st = snapshot.get(key) or {}
        if not st.get("on"):
            return ""
        return valid_choice(S.FIELD_BY_KEY[key], str(st.get("value") or "").strip())

    ct = pick("ct")
    k = pick("ctk") or ct or "vbr"
    v = pick("ctv") or ct or "vbr"
    return k, v


def model_head_dim(model: str) -> Tuple[Optional[int], str]:
    """读 GGUF 里的 head_dim（读不出来返回 (None, "")）。"""
    if not model or not os.path.isfile(model):
        return None, ""
    try:
        from . import gguf as _G          # 延迟导入，避免模块循环
        info = _G.analyse(model, want_tensors=True) or {}
    except Exception:                     # noqa: BLE001
        return None, ""
    hd = info.get("head_dim")
    return (int(hd) if isinstance(hd, int) else None,
            str(info.get("head_dim_src") or ""))


def kv_compat_errors(snapshot: Dict[str, Dict[str, Any]],
                     model: str) -> List[Tuple[str, str]]:
    """turbo / TCQ / VBR 档位的前置检查，返回会挡住启动的错误。

    buun 的 turbo 系列 KV block 固定 128 个值，所以模型的 n_embd_head_k
    必须能被 128 整除，否则引擎会直接拒绝启动：
    ``K cache type turbo3_tcq with block size 128 does not divide
    n_embd_head_k=64``。

    **注意最容易踩的一种**：本 build 的 -ctk/-ctv 默认就是 ``vbr``，
    而 vbr 会按模型几何自动选 turbo 阶梯。所以对 head_dim=64 的模型
    （bert 系、部分小模型），**什么都不设也会启动失败** —— 必须显式把
    两侧都设成常规量化。这里把隐式默认也算进来。
    """
    k_eff, v_eff = effective_kv_types(snapshot)
    need128 = [t for t in dict.fromkeys((k_eff, v_eff))
               if t in S._BLOCK128_TYPES]
    if not need128:
        return []
    errs: List[Tuple[str, str]] = []

    hd, src = model_head_dim(model)
    if hd is not None and hd % 128 != 0:
        explicit = bool(snapshot.get("ct", {}).get("on")
                        or snapshot.get("ctk", {}).get("on")
                        or snapshot.get("ctv", {}).get("on"))
        extra = "" if explicit else (
            "\n（你并没有显式设 KV 档位 —— k=%s / v=%s 是**本 build 的默认值**，"
            "它同样会走 turbo 阶梯。）" % (k_eff, v_eff))
        errs.append(("ct",
                     "这个模型 head_dim=%d（%s），不是 128 的倍数，用不了 %s 档位。\n"
                     "turbo / TCQ / VBR 的 KV block 固定是 128 个值，"
                     "引擎会直接拒绝启动。%s\n"
                     "请把「统一 KV 档位」（或 K/V 两侧）显式设成常规量化："
                     "f16 / q8_0 / q4_0 / iq4_nl 等。"
                     % (hd, src or "GGUF 元数据", "/".join(need128), extra)))

    fa_st = snapshot.get("fa") or {}
    fa_val = str(fa_st.get("value") or "").strip()
    if fa_st.get("on") and fa_val == "关闭":
        errs.append(("fa",
                     "选中了 %s 档位，但「快速注意力」被显式关掉了。\n"
                     "turbo / TCQ / VBR 都要求 -fa 开启（它们的 KV 走 FA 路径）。"
                     % "/".join(need128)))
    return errs


def kv_compat_notes(snapshot: Dict[str, Dict[str, Any]],
                    model: str) -> List[str]:
    """不挡启动、但值得在启动日志里说一句的提示。"""
    notes: List[str] = []
    k_eff, v_eff = effective_kv_types(snapshot)
    vbr_on = "vbr" in (k_eff, v_eff)
    if not vbr_on:
        picked_vbr = [f.key for f in S.FIELDS
                      if f.key in S.VBR_KEYS
                      and (snapshot.get(f.key) or {}).get("on")]
        if picked_vbr:
            notes.append(
                "KV 档位是 K=%s / V=%s（不是 vbr），所以 VBR 那一组参数"
                "（%s）不会传给引擎 —— 引擎对「非 vbr 的 KV + --vbr-*」"
                "是硬报错。想用 VBR 请把「统一 KV 档位」选成 vbr。"
                % (k_eff, v_eff, ", ".join(picked_vbr)))
    if vbr_on:
        ctx_raw = _picked(snapshot, "ctx")
        floor = _picked(snapshot, "vbr_floor")
        floor_desc = ("%s（%.4g bpv）" % (floor, S.VBR_BPV[floor])
                      if floor in S.VBR_BPV else "引擎默认")
        if ctx_raw:
            notes.append("上下文长度固定为 %s（手动指定），"
                         "VBR 下限 %s 只影响运行时的降级深度。"
                         % (ctx_raw, floor_desc))
        else:
            notes.append(
                "上下文长度交给引擎自动算：从模型训练上下文起算，按可用显存往下缩，"
                "KV 成本按 VBR 下限 %s 的单价计价（下限压得越低、能开的上下文越长）。"
                "实际算出来的值可以在「运行状态」里看。" % floor_desc)
    hd, src = model_head_dim(model)
    if hd is not None:
        notes.append("模型 head_dim=%d（%s），128 的倍数=%s；"
                     "实际生效的 KV 档位：K=%s / V=%s"
                     % (hd, src or "GGUF",
                        "是" if hd % 128 == 0 else "否", k_eff, v_eff))
    return notes


# --------------------------------------------------------------------------- #
# --models-preset 的 INI 生成
#
# 为什么优先用它、而不是 --models-dir：
#   --models-dir 的扫描规则很窄 —— 只认「顶层散落的 .gguf」或「一层子目录」
#   （工具层文件 common/preset.cpp 的 load_from_models_dir）。像
#   E:\LM_models\<发布者>\<模型名>\x.gguf 这种两层结构，它会一个都扫不到。
#   而 --models-preset 的段就是模型 id、`model = <任意深度的路径>`，既能覆盖
#   任意目录结构，又能顺手把「每个模型自己的加载参数」写进去。
#
# INI 格式（同样来自 common/preset.cpp）：
#   [*]                  ← 全局段，对所有模型生效
#   ctx-size = 32768
#   [模型 id]            ← 段名就是客户端请求里要写的 model 名
#   model = E:\...\x.gguf
#   cache-type-k = vbr
#   键可以写长参数名（去掉 --），也可以写环境变量名（LLAMA_ARG_XXX）。
# --------------------------------------------------------------------------- #

# 这些项属于「路由服务进程本身」或纯界面状态，不该写进模型预置
INI_SKIP: FrozenSet[str] = frozenset((
    "host", "port", "api_key", "webui", "auto_switch", "idle_unload",
    "run_kind", "models_dir", "models_max", "models_preset",
    "models_autoload", "spec_enable", "extra", "lora",
    # 推测解码的「总开关」不是参数，但它下面所有 spec_* 都只在服务进程层面有效，
    # 放进 per-model 段会让子进程各自带一份，反而乱
    "prompt", "sys_prompt", "single_turn",
    "g_prompt", "g_sys_prompt", "g_conversation", "g_interactive",
    "g_single_turn",
))


def _ini_value(f: S.F, st: Dict[str, Any]) -> Tuple[str, str]:
    """把一个界面参数项翻成 INI 的 (键, 值)。空键表示「不导出」。"""
    key = f.ini_key
    if not key or not st.get("on") or f.key in INI_SKIP:
        return "", ""
    raw = str(st.get("value") or "").strip()

    if f.kind == S.K_BOOL:
        # 无值开关：写 true 就会把 flag 带上；false 会被引擎丢弃（= 不传）
        return key, "true"

    if f.kind == S.K_GEAR:
        argv = list((f.argmap or {}).get(valid_choice(f, raw), ()))
        if len(argv) == 2 and argv[0] == f.flag:
            return key, argv[1]
        # 其余（自动 / 反向 flag）无法用「键 = 值」表达，交给命令行
        return "", ""

    if not raw:
        return "", ""
    if f.kind in (S.K_INT, S.K_FLOAT) and not numeric_ok(raw, f.kind):
        return "", ""
    if f.kind in S.CHOICE_KINDS:
        raw = valid_choice(f, raw) or raw
    return key, raw


def preset_ini(entries: Sequence[Tuple[str, str, Dict[str, Any]]],
               global_snap: Optional[Dict[str, Any]] = None,
               engine_hint: str = "") -> str:
    """生成 ``--models-preset`` 用的 INI 文本。

    :param entries: ``[(模型 id, 模型文件路径, 该模型的快照), ...]``
                    模型 id 就是客户端请求里要写的 model 名。
    :param global_snap: 写进 ``[*]`` 那一节的快照（服务级公共参数）。
    """
    out: List[str] = [
        "; buun-llama-cpp 多模型路由 · 模型预置文件",
        "; 由启动器自动生成 —— 手动改这里的内容会被下次导出覆盖。",
        ";",
        "; 段名 = 客户端请求里要写的 model 名（/v1/chat/completions 的 model 字段）",
        "; 键   = 去掉前导 -- 的长参数名，例如 ctx-size / n-gpu-layers / cache-type-k",
        "; [*] 是全局段，它里面的项对所有模型生效。",
        "",
    ]
    if global_snap:
        g_lines = []
        for f in S.FIELDS:
            st = global_snap.get(f.key) or {}
            k, v = _ini_value(f, st)
            if k:
                g_lines.append((k, v))
        if g_lines:
            out.append("[*]")
            for k, v in g_lines:
                out.append("%s = %s" % (k, v))
            out.append("")

    skipped: List[str] = []
    for name, path, snap in entries:
        out.append("[%s]" % name)
        out.append("model = %s" % path)
        wrote = 0
        for f in S.FIELDS:
            st = (snap or {}).get(f.key) or {}
            k, v = _ini_value(f, st)
            if not k or k == "model":
                continue
            out.append("%s = %s" % (k, v))
            wrote += 1
        out.append("")
        if not wrote:
            skipped.append(name)

    if engine_hint:
        out.append("; 引擎：%s" % engine_hint)
    if skipped:
        out.append("; 这些模型只写了路径（还没有自己的加载参数）：%s"
                   % ", ".join(skipped))
    out.append("; 无法用 INI 表达的勾选项（如 -mlock、-cpu-moe、LoRA）"
               "请留在服务页命令行上，它们对路由进程本身生效。")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #

def _as_int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def validate(snapshot: Dict[str, Dict[str, Any]], mode: str,
             exe: Optional[str], model: str = "", role: str = "llm"
             ) -> List[Tuple[str, str]]:
    """返回 [(字段 key, 错误说明)]，空列表表示通过。"""
    errs: List[Tuple[str, str]] = []

    if not exe:
        errs.append(("__engine__",
                     "没有找到可执行文件。请先在顶部「引擎目录」里选择 "
                     "buun-llama-cpp 的目录或某个 .exe。"))

    if not model:
        errs.append(("__model__",
                     "还没有选定模型。请先在「模型库」里选一个模型。"))
    elif not os.path.isfile(model):
        errs.append(("__model__", "模型文件不存在：%s" % model))

    for f in S.FIELDS:
        if mode not in f.modes:
            continue
        if f.page == "server" and role not in f.roles:
            continue
        st = snapshot.get(f.key) or {}
        if not st.get("on"):
            continue
        raw = str(st.get("value", "")).strip()
        if not raw:
            continue
        if f.kind == S.K_INT and _as_int(raw) is None:
            errs.append((f.key, "需要整数，当前是「%s」" % raw))
        if f.kind == S.K_FLOAT:
            try:
                float(raw)
            except ValueError:
                errs.append((f.key, "需要数字，当前是「%s」" % raw))

    if mode == "server":
        v = _as_int((snapshot.get("port") or {}).get("value", 0))
        if v is None or not (1 <= v <= 65535):
            errs.append(("port", "端口需要是 1~65535 之间的整数"))
        fit_st = snapshot.get("fit") or {}
        fit_on = bool(fit_st.get("on")) and \
            str(fit_st.get("value") or "") == "强制开启"
        if fit_on and (snapshot.get("ngl") or {}).get("on"):
            errs.append(("fit", "「自动适配显存」被强制开启时，"
                                "显式设的 -ngl 会让它失效（引擎会跳过 fit）。"
                                "请把 fit 改回「自动」，或取消 -ngl。"))

    if mode in ("server", "cli") and (snapshot.get("spec_enable") or {}).get("on"):
        st_st = snapshot.get("spec_type") or {}
        stype = str(st_st.get("value", "")).strip() if st_st.get("on") else ""
        names = S.spec_type_names(stype)
        md_st = snapshot.get("model_draft") or {}
        draft = str(md_st.get("value", "")).strip() if md_st.get("on") else ""
        need = [t for t in names if t in S.SPEC_NEEDS_DRAFT]
        if need and not draft:
            errs.append(("model_draft",
                         "推测方式「%s」需要指定草稿模型（-md）。"
                         "不需要草稿模型的写法是 draft-mtp 与 ngram-* / suffix。"
                         % ",".join(need)))
        unknown = [t for t in names if t not in S.SPEC_TYPES]
        if unknown:
            errs.append(("spec_type",
                         "本 build 不认识这些推测方式：%s。可选：%s"
                         % (",".join(unknown), ", ".join(S.SPEC_TYPES))))

    # turbo / TCQ / VBR 的 128-block 限制（读 GGUF 拿 head_dim）
    if model and os.path.isfile(model):
        errs += kv_compat_errors(snapshot, model)

    # VBR 下限与起始档位 / 阶梯的约束
    errs += vbr_floor_errors(snapshot)

    # 路由模式必须有模型来源，否则 llama-server 起来了也一个模型都没有
    if S.is_router(snapshot):
        md = snapshot.get("models_dir") or {}
        mp = snapshot.get("models_preset") or {}
        has_dir = bool(md.get("on") and str(md.get("value") or "").strip())
        has_ini = bool(mp.get("on") and str(mp.get("value") or "").strip())
        if not (has_dir or has_ini):
            errs.append(("models_dir",
                         "「多模型路由」需要一个模型来源：\n"
                         "· 填「路由模型目录」（目录结构必须是 <目录>/<模型名>/*.gguf "
                         "或顶层散放的 .gguf），或\n"
                         "· 用「导出路由预置 INI」生成一个预置文件填到"
                         "「路由预置文件」里 —— 它能覆盖任意深度的目录结构，"
                         "还能给每个模型带上自己的参数。"))
    return errs


def write_ini(path: str, text: str) -> bool:
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# 命令行文本
# --------------------------------------------------------------------------- #

_SAFE = re.compile(r'^[A-Za-z0-9_@%+=:,./\\-]+$')


def quote_arg(a: str) -> str:
    if a == "":
        return '""'
    if _SAFE.match(a) and not a.endswith("\\"):
        return a
    return '"' + a.replace('"', r'\"') + '"'


def format_cmdline(exe: str, argv: Sequence[str]) -> str:
    return " ".join([quote_arg(exe)] + [quote_arg(a) for a in argv])


def format_multiline(exe: str, argv: Sequence[str]) -> str:
    parts = [quote_arg(exe)] + [quote_arg(a) for a in argv]
    if not parts:
        return ""
    return " ^\n".join([parts[0]] + ["  " + p for p in parts[1:]])


def make_bat(exe: str, argv: Sequence[str], cwd: str = "") -> str:
    body = quote_arg(exe)
    if argv:
        body += " ^\n" + " ^\n".join("  " + quote_arg(a) for a in argv)
    lines = [
        "@echo off",
        "chcp 65001 >nul",
        "rem exported by buun-llama-cpp GUI launcher",
    ]
    if cwd:
        lines.append('cd /d "%s"' % cwd)
    lines.append(body)
    lines.append("echo.")
    lines.append("echo [process exited] press any key to close")
    lines.append("pause >nul")
    return "\n".join(lines) + "\n"
