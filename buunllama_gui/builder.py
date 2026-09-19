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


def lora_items(raw: Any) -> Tuple[List[str], List[Tuple[str, float]]]:
    """(比例=1.0 的路径列表, [(路径, 非 1.0 的比例)])。"""
    plain: List[str] = []
    scaled: List[Tuple[str, float]] = []
    for path, scale in parse_lora(raw):
        if scale is None or abs(scale - 1.0) < 1e-9:
            plain.append(path)
        else:
            scaled.append((path, float(scale)))
    return plain, scaled


def lora_args(raw: Any) -> List[str]:
    """LoRA 展开成参数。**buun 这里是单值逗号分隔形式**：

        --lora a.gguf,b.gguf              全部比例 1.0
        --lora-scaled p:0.8,q:0.5         带比例

    ⚠️ 不是上游 llama.cpp 的「--lora-scaled 路径 比例」两值写法。
    buun 的 common/arg.cpp 里 `--lora-scaled` 的 value_hint 是
    ``FNAME:SCALE,...``（单值），而引擎的 preset 层对**两值**参数会直接
    throw（"argument with 2 values is not yet supported"）——
    在路由模式下那会让整个 llama-server 起不来。所以这里必须用 CSV 形式。
    """
    plain, scaled = lora_items(raw)
    out: List[str] = []
    if plain:
        out += ["--lora", ",".join(plain)]
    if scaled:
        out += ["--lora-scaled",
                ",".join("%s:%g" % (p, s) for p, s in scaled)]
    return out


def lora_ini_lines(raw: Any) -> List[Tuple[str, str]]:
    """LoRA → preset INI 的 (键, 值)。可能一次给出两行。"""
    plain, scaled = lora_items(raw)
    out: List[Tuple[str, str]] = []
    if plain:
        out.append(("lora", ",".join(plain)))
    if scaled:
        out.append(("lora-scaled",
                    ",".join("%s:%g" % (p, s) for p, s in scaled)))
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

    # LoRA：buun 是单值 CSV 形式（--lora a,b / --lora-scaled p:0.8）
    if f.key == "lora":
        return lora_args(raw)
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
    # 「路由」只针对 llama-server：它的模型一律由预置文件提供，命令行不带 -m。
    # 对话 / 生成（llama-cli / llama-completion）仍是单模型单次运行，照旧带 -m。
    router = (mode == S.ROUTER_MODE)
    spec_method = S.spec_method_of(snapshot)
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
        # ⚠️ 路由模式下，**模型级参数一律不上路由器的命令行**。
        #    原因：路由会把自身的命令行转成 base_preset 再 merge 进每个模型的
        #    预置，而 merge 是「覆盖」—— 放上去就等于把所有模型都锁成同一套参数，
        #    各模型自己的 INI 段再也改不动它。
        #    模型级参数（load / chat / emb / lora）统一走 preset INI 的模型段。
        #    留在命令行上的只有服务进程自己的参数（scope=run）和纯界面状态（ui）。
        if router and f.scope not in ("run", "ui"):
            continue
        # 不属于当前推测方式的细项一律**不写命令**（值仍留在配置里）。
        # 引擎对「MTP 方式 + -md」这类组合要么报错要么打一堆无谓警告。
        if not S.field_visible_for_spec(f, spec_method):
            continue
        # 两值参数在服务模式下一律不输出：路由初始化会因它直接 throw
        # （见 schema.TWO_VALUE_FLAGS 的说明）。validate 那边也会报错拦下来，
        # 这里再兜一层，保证命令行预览里不会出现这个必定致命的参数。
        if router and f.flag in S.TWO_VALUE_FLAGS:
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
#
# ⚠️ 注意 LoRA **不在**这个表里了：它必须写进每个模型的段。
#    原因：路由模式下命令行上的参数会被 merge（覆盖）进所有模型，
#    而 LoRA 是「每个模型挂自己的适配器」——放命令行就等于所有模型共用同一套。
#    好在 buun 的 --lora / --lora-scaled 是单值 CSV 形式，INI 表达得了
#    （见 lora_ini_lines）。
INI_SKIP: FrozenSet[str] = frozenset((
    "host", "port", "api_key", "webui", "auto_switch",
    "models_preset", "models_max",
    "res_llm_max", "res_emb_max", "res_emb_uncounted",
    "res_llm_idle", "res_emb_idle",
    "spec_enable", "extra",
    # 单次运行（对话 / 生成）的参数，跟服务预置无关
    "prompt", "sys_prompt", "single_turn",
    "g_prompt", "g_sys_prompt", "g_conversation", "g_interactive",
    "g_single_turn",
))


def _ini_lines(f: S.F, st: Dict[str, Any],
               snap: Optional[Dict[str, Dict[str, Any]]] = None
               ) -> List[Tuple[str, str]]:
    """把一个界面参数项翻成 INI 的若干 (键, 值)。

    ``snap`` 给整个快照时会额外做三道闸（跟 build_argv 保持一致，
    **必须一致**：分叉过一次就出过 bug —— INI 里写了 MTP 方式不该有的
    草稿模型、以及 KV 不是 vbr 时的 ``--vbr-*``）：
      1. KV 不是 vbr 就不写 ``--vbr-*`` 那一组（引擎会拒绝启动）
      2. 不属于当前推测方式的细项不写（spec_only）
      3. LoRA 单值 CSV 拆成 lora / lora-scaled 两行
    """
    if not st.get("on") or f.key in INI_SKIP:
        return []
    if snap is not None:
        if f.key in S.VBR_KEYS:
            k_eff, v_eff = effective_kv_types(snap)
            if "vbr" not in (k_eff, v_eff):
                return []
        if not S.field_visible_for_spec(f, S.spec_method_of(snap)):
            return []
    # LoRA 一个条目可能同时产出 lora 和 lora-scaled 两行
    if f.key == "lora":
        return lora_ini_lines(st.get("value"))
    k, v = _ini_value(f, st)
    return [(k, v)] if k else []


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
        if len(argv) == 1 and argv[0].startswith("--no-"):
            # 反向开关（如「强制关闭 --no-spec-dspark-gpu-assist」）：
            # 写 ``键 = false``，引擎的 to_args() 见到 falsey 值就会换成
            # args_neg 那个反向 flag。没有这条，「关掉某项」在 INI 里就丢了
            # —— 而路由模式下命令行又送不进子模型。
            return key, "false"
        # 空 argmap（自动 / 默认档）＝跟引擎默认一样，不用写
        return "", ""

    if not raw:
        return "", ""
    if f.kind in (S.K_INT, S.K_FLOAT) and not numeric_ok(raw, f.kind):
        return "", ""
    if f.kind in S.CHOICE_KINDS:
        raw = valid_choice(f, raw) or raw
    return key, raw


def preset_aliases(name: str, path: str) -> List[str]:
    """这个模型除了段名以外，还该登记哪些别名。

    为什么需要：客户端手里的模型名不一定等于段名 ——

      * 本程序 ``GET /v1/models`` 给的是**段名**（= 模型库里的显示名）；
      * 第三方软件（Open WebUI / Cherry / 沉浸式翻译…）常常直接用**文件名**，
        而且会把 ``.gguf`` 一起带上；
      * 用户手抄、老客户端缓存的清单也都是文件名。

    引擎的 ``has_model()`` / ``get_meta()``（server-models.cpp:1024）**会一起
    匹配别名**，所以把「文件名」和「去掉扩展名的文件名」写进 ``alias``，
    各种写法都能落到同一个模型上，不用逼用户去改客户端的配置。

    返回空列表表示段名已经够用（路径为空或名字恰好相同）。
    """
    base = os.path.basename(str(path or "").replace("\\", "/"))
    if not base:
        return []
    stem = os.path.splitext(base)[0]
    out: List[str] = []
    for cand in (stem, base):
        if cand and cand != name and cand not in out and "," not in cand:
            out.append(cand)
    return out


def preset_ini(entries: Sequence[Tuple[str, str, Dict[str, Any]]],
               global_snap: Optional[Dict[str, Any]] = None,
               engine_hint: str = "") -> str:
    """生成 ``--models-preset`` 用的 INI 文本。

    :param entries: ``[(模型 id, 模型文件路径, 该模型的快照), ...]``
                    模型 id 就是客户端请求里要写的 model 名。
    :param global_snap: 写进 ``[*]`` 那一节的快照。**不要传某个模型的快照** ——
                    见下面的说明。

    ⚠️ 关于 ``[*]``：引擎会把命令行上的参数也 merge 进每个模型段（**覆盖**），
    而 ``[*]`` 同样是「对所有模型生效」。所以任何**模型级**参数都不能放这两处，
    否则多模型就退化成「所有模型共用一套参数」。模型级参数只写各自的段。
    """
    out: List[str] = [
        "; buun-llama-cpp 多模型路由 · 模型预置文件",
        "; 由启动器自动生成 —— 手动改这里的内容会被下次导出覆盖。",
        ";",
        "; 段名 = 客户端请求里要写的 model 名（/v1/chat/completions 的 model 字段）",
        "; 键   = 去掉前导 -- 的长参数名，例如 ctx-size / n-gpu-layers / cache-type-k",
        "; [*] 是全局段，它里面的项对所有模型生效（本程序默认不放模型级参数进去）。",
        "",
    ]
    if global_snap:
        g_lines: List[Tuple[str, str]] = []
        for f in S.FIELDS:
            if f.scope not in ("run", "ui"):
                continue           # 模型级参数不进 [*]
            g_lines += _ini_lines(f, global_snap.get(f.key) or {}, global_snap)
        out.append("[*]")
        for k, v in g_lines:
            out.append("%s = %s" % (k, v))
        out.append("")
    else:
        out.append("[*]")
        out.append("")

    # ⚠️ 模型段只写**模型级**参数（scope = load / chat / emb）。
    #    服务进程自己的参数（scope = run / ui）既在路由命令行上、又会跟模型段
    #    混在一起，写进来纯属噪音 —— 而且会让人误以为模型自带一套宿主参数。
    skipped: List[str] = []
    alias_seen: Dict[str, str] = {n: n for n, _p, _s in entries}
    for name, path, snap in entries:
        out.append("[%s]" % name)
        out.append("model = %s" % path)
        aliases = []
        for a in preset_aliases(name, path):
            owner = alias_seen.get(a)
            if owner and owner != name:
                continue           # 跟别的模型/别名撞了 → 这个模型就让出去
            alias_seen[a] = name
            aliases.append(a)
        if aliases:
            # 引擎按逗号拆（common/arg.cpp 的 --alias）
            out.append("alias = %s" % ",".join(aliases))
        wrote = 0
        safe = emb_guard(path, snap)
        for f in S.FIELDS:
            if f.scope not in ("load", "chat", "emb"):
                continue
            for k, v in _ini_lines(f, (safe or {}).get(f.key) or {}, safe):
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
    return "\n".join(out) + "\n"


def preset_section(name: str, path: str,
                   snap: Dict[str, Any]) -> str:
    """单个模型的 INI 段落文本（界面上「这个模型会写进预置的内容」预览）。"""
    lines: List[str] = ["[%s]" % name, "model = %s" % path]
    aliases = preset_aliases(name, path)
    if aliases:
        lines.append("alias = %s" % ",".join(aliases))
    if is_embedding_model(path):
        lines.append("; （embedding 类模型：KV 档位已自动钉成 f16）")
    safe = emb_guard(path, snap)
    for f in S.FIELDS:
        if f.scope not in ("load", "chat", "emb"):
            continue
        for k, v in _ini_lines(f, (safe or {}).get(f.key) or {}, safe):
            if not k or k == "model":
                continue
            lines.append("%s = %s" % (k, v))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 模型级快照的硬性纠正（进 INI 之前）
#
# 1) embedding 类模型的 KV 档位
#    ⚠️ buun 的 ``-ctk`` / ``-ctv`` **默认值就是 vbr**，而 vbr / turbo / TCQ
#       这一族要求 ``n_embd_head_k % 128 == 0``。bert 系（bge / gte / e5 /
#       qwen3-embedding / embeddinggemma …）的 head_dim 是 64 —— 于是
#       **什么都不设也会起不来**：
#         K cache type turbo4 with block size 128 does not divide
#         n_embd_head_k=64
#       界面上的 vbr 组默认是勾着的，会被写进每个模型段，所以这里必须对
#       embedding 类模型显式钉成 f16。
# 2) ``--embedding`` 不该由用户手动勾
#    embedding 类模型**天生**要带这个参数（不然它就只是个没用的 bert），
#    普通对话模型则一定不能带（带上就彻底不能聊天）。所以按 GGUF 架构自动定：
#    是 embedding 架构 → 强制开；不是 → 强制关并清掉整组 emb 参数。
# --------------------------------------------------------------------------- #

def is_embedding_model(path: str) -> bool:
    """是不是 embedding / reranker 类模型。

    用「模型库」那一套同一份判据（scan.classify）—— 因为光看架构不够：
    ``Qwen3-Embedding-0.6B`` 的 general.architecture 就是 ``qwen3``，
    只有从名字里的 embedding 关键字才认得出来。
    """
    from . import scan as _SC         # 延迟导入，避免模块循环
    return _SC.classify(path, {"arch": _arch_of(path)}) == "Embedding"


def emb_guard(path: str,
              snap: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按模型类别纠正 embedding 相关取值（返回一份副本）。"""
    out = dict(snap or {})
    emb = is_embedding_model(path)
    if emb:
        # 只有几何上真不支持的才钉 f16（head_dim 不是 128 的倍数）。
        # 读不出 head_dim 时保守起见也钉上 —— 钉了只是多占一点 KV，不会失败。
        hd, _src = model_head_dim(path)
        if not hd or hd % 128:
            for key in ("ct", "ctk", "ctv"):
                if key in out:
                    out[key] = {"on": True, "value": "f16"}
        if "emb_enable" in out:
            out["emb_enable"] = {"on": True, "value": ""}
    else:
        # 非 embedding 模型：整组 emb_* 一律关掉（存档里的旧值不算数）
        for f in S.FIELDS:
            if f.page == "emb" and f.key in out:
                out[f.key] = {"on": False, "value": f.default_value()}
    return out


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #

def _as_int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def validate(snapshot: Dict[str, Dict[str, Any]], mode: str,
             exe: Optional[str], model: str = "", role: str = "llm",
             model_checks: bool = True) -> List[Tuple[str, str]]:
    """返回 [(字段 key, 错误说明)]，空列表表示通过。

    ``model_checks=False``：跳过**针对这一个模型**的检查（文件在不在、
    KV 档位跟它的 head_dim 搭不搭）。路由模式下必须这么用 —— 路由器进程
    本身不加载任何模型（模型路径只写在预置文件里），拿某一个模型的参数去
    卡整个路由器的话，那个模型配错了就谁都起不来：

      * bert 系 embedding 模型的 head_dim=64，默认的 vbr 档位必然报错
        （预置文件里已经由 ``emb_guard`` 自动钉成 f16，但用户选中的
         「目标模型」不一定是 embedding）；
      * 目标模型的其它参数有问题时，同理想让别的模型也一起没法用。
    """
    errs: List[Tuple[str, str]] = []

    if not exe:
        errs.append(("__engine__",
                     "没有找到可执行文件。请先在顶部「引擎目录」里选择 "
                     "buun-llama-cpp 的目录或某个 .exe。"))

    if model_checks:
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
        method = S.spec_method_of(snapshot)
        if method not in S.SPEC_METHODS:
            errs.append(("spec_type",
                         "本程序只保留三种推测方式：%s。"
                         % "、".join(S.SPEC_METHODS)))
        elif method in S.SPEC_NEEDS_DRAFT:
            md_st = snapshot.get("model_draft") or {}
            draft = str(md_st.get("value", "")).strip() if md_st.get("on") else ""
            if not draft:
                errs.append(("model_draft",
                             "推测方式「%s」需要指定草稿模型（-md）。\n"
                             "只有 MTP 不需要 —— 它用的是主模型自带的 MTP 层。"
                             % method))
            elif model and os.path.isfile(model) \
                    and _model_has_mtp(model) is True:
                # 主模型自带 MTP 头，又配外部 DFlash/DSpark 草稿：内置 MTP 头会和
                # 外部草稿打架，引擎启动该模型子进程时会在 fit 阶段报
                # 「failed to measure a required speculative model/context」。
                errs.append(("spec_type",
                             "请改用 MTP 方式（spec-type=draft-mtp），"
                             "不需要额外草稿（-md 也清掉）。\n"
                             "原因：主模型自带 MTP 预测头（GGUF 里 %s.nextn_predict_layers "
                             ">0），选 %s 会跟内置 MTP 头冲突，触发引擎\n"
                             "「failed to measure a required speculative "
                             "model/context」这条 fit 警告。"
                             % (_arch_of(model), method)))
        else:
            # MTP：主模型必须真有 MTP 层，否则引擎只会打一行 warning 然后跳过
            if model and os.path.isfile(model):
                if not _model_has_mtp(model):
                    errs.append(("spec_type",
                                 "这个模型没有 MTP 预测头（GGUF 里 %s.nextn_predict_layers "
                                 "为空或 0），选 MTP 方式会被引擎静默跳过、完全不加速。\n"
                                 "请换一个带 MTP 的模型，或改用 DFlash / DSpark "
                                 "并指定草稿模型。" % _arch_of(model)))

    # ⚠️ 两值参数绝对不能进服务命令行：路由初始化会把命令行自身转成
    #    base_preset（common_params_to_map），遇到两值参数直接 throw，
    #    整个 llama-server 起不来（实测：failed to initialize router models）。
    if mode == "server":
        bad = [f.flag for f in S.FIELDS
               if f.flag in S.TWO_VALUE_FLAGS
               and (snapshot.get(f.key) or {}).get("on")]
        if bad:
            errs.append(("__engine__",
                         "这些参数是一次要两个值的，路由模式不支持：%s\n"
                         "（引擎的 preset 层会直接抛异常，服务起不来）"
                         % "、".join(bad)))

    # turbo / TCQ / VBR 的 128-block 限制（读 GGUF 拿 head_dim）
    if model_checks and model and os.path.isfile(model):
        errs += kv_compat_errors(snapshot, model)

    # VBR 下限与起始档位 / 阶梯的约束
    errs += vbr_floor_errors(snapshot)

    # 路由模式必须有模型来源，否则 llama-server 起来了也一个模型都没有
    if S.is_router(snapshot):
        mp = snapshot.get("models_preset") or {}
        has_ini = bool(mp.get("on") and str(mp.get("value") or "").strip())
        if not has_ini:
            errs.append(("models_preset",
                         "路由需要一个模型来源：本程序会自动生成预置文件"
                         "（config/router-preset.ini）并把路径填进「路由预置文件」。\n"
                         "如果这里是空的，说明生成失败了（一般是被杀毒软件锁住"
                         "或配置目录不可写）—— 检查一下 config 目录。"))
    return errs


# --------------------------------------------------------------------------- #
# 模型能力探测（MTP 判据）
# --------------------------------------------------------------------------- #

def _head_kv(path: str) -> Tuple[Dict[str, Any], str]:
    """读 GGUF 头，返回 (元数据, architecture)。读不出来给 ({}, "")。"""
    if not path or not os.path.isfile(path):
        return {}, ""
    try:
        from . import gguf as _G          # 延迟导入，避免模块循环
        kv, _ = _G.read_header(path, want_tensors=False)
    except Exception:  # noqa: BLE001
        return {}, ""
    if not isinstance(kv, dict):
        return {}, ""
    return kv, str(kv.get("general.architecture") or "")


def _arch_of(path: str) -> str:
    """从 GGUF 读出 general.architecture（读不出来就返回空串）。"""
    return _head_kv(path)[1]


def _model_has_mtp(path: str) -> Optional[bool]:
    """这个模型有没有 MTP（NextN）预测头。

    判据是 GGUF 元数据里的 ``{arch}.nextn_predict_layers`` —— 引擎自己就是用
    这个键决定要不要建 NextN 层的（见 llama-model.cpp:1462）。
    **不要拿文件名里有没有 mtp / nextn 去猜**：实测这一批 27B 里，
    ``Huihui-...-abliterated`` / ``SexyGPT`` / ``Signal`` 名字里都没有 MTP，
    但 nextn_predict_layers 都是 1。
    返回 None 表示读不出（文件不存在 / 不是 GGUF）。
    """
    kv, arch = _head_kv(path)
    if not arch:
        return None
    try:
        raw = kv.get("%s.nextn_predict_layers" % arch)
    except AttributeError:
        return None
    try:
        return int(raw) > 0
    except (TypeError, ValueError):
        return False


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
