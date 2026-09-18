# -*- coding: utf-8 -*-
"""参数模型：所有可调项、所属页面/小节、作用域、对应的命令行参数。

设计要点
--------
* 每个设置项由 :class:`F` 描述：中英双语标签、命令行 flag（以及 ini 用的长名）、
  控件类型、默认值、是否默认勾选、适用的运行模式、所属页面与小节。
* **三个作用域**（scope）决定参数存在哪里：
    - ``load``  加载参数 —— 跟着**模型**走（加载参数页 / 推测解码页）
    - ``chat``  对话参数 —— 跟着**模型**走（对话参数页）
    - ``run``   运行参数 —— 服务页只有一套（统一端口、一个角色），
      「对话」模式（cli）自己一份
* 未勾选的项 **完全不写入命令行**，所以即使某个 build 不认识这个参数也不会启动失败；
  再配合 ``--help`` 探测（probe.py）就能把不支持的项自动标出来。
* 页面不再是两排横向页签，而是一条竖向导航（见 ui.py 的 Sidebar）。

新增参数 = 在这里加一行 ``F(...)``，界面/命令行/ini 导出/参数对照表全部自动跟随。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 运行模式
# --------------------------------------------------------------------------- #

MODES: Dict[str, Dict[str, str]] = {
    "server": {
        "exe": "llama-server",
        "zh": "服务",
        "en": "Server",
        "tip": "启动 OpenAI 兼容的 HTTP 推理服务（llama-server）。\n"
               "统一端口：一个端口对外，按请求自动切换模型；也可以在服务页切到"
               "「多模型路由」，让 llama-server 自己驻留多个模型。",
    },
    "cli": {
        "exe": "llama-cli",
        "zh": "对话",
        "en": "Chat",
        "tip": "终端里多轮对话（llama-cli）。\n"
               "注意：本 build 的 llama-cli 默认就是对话模式，"
               "既不接受 -cnv，也没有 -i / -no-cnv（要这些请用「生成」模式）。",
    },
    "gen": {
        "exe": "llama-completion",
        "zh": "生成",
        "en": "Completion",
        "tip": "一次性生成：给一段提示词出结果然后退出（llama-completion）。\n"
               "适合跑批、测提示词；勾上「多轮对话」也能当交互终端用。",
    },
}

MODE_ORDER: Tuple[str, ...] = ("server", "cli", "gen")
ALL_MODES: FrozenSet[str] = frozenset(MODE_ORDER)

# 只跑 LLM：同一时刻只有一个 llama-server 进程、只驻留一个主模型。
# 历史上这里还有 Embedding 角色和「双进程」运行方式，已按用户要求全部移除。
ROLES: Dict[str, Dict[str, str]] = {
    "llm": {"zh": "LLM", "en": "LLM", "default_port": "1233",
            "tip": "聊天 / 补全用的主模型"},
}
ROLE_ORDER: Tuple[str, ...] = ("llm",)

# --------------------------------------------------------------------------- #
# 页面（左侧竖向导航）与小节
# --------------------------------------------------------------------------- #

PAGES: Tuple[Tuple[str, str, str, str], ...] = (
    ("lib", "模型库", "Models", "scan"),
    ("server", "服务", "Server", "run"),
    ("cli", "对话", "Chat", "run"),
    ("gen", "生成", "Completion", "run"),
    ("load", "加载参数", "Load", "load"),
    ("chat", "对话参数", "Chat params", "chat"),
    ("spec", "推测解码", "Spec", "load"),
)

PAGE_ZH = {p[0]: p[1] for p in PAGES}
PAGE_EN = {p[0]: p[2] for p in PAGES}
PAGE_KIND = {p[0]: p[3] for p in PAGES}
PAGE_ORDER = {p[0]: i for i, p in enumerate(PAGES)}

# 页面 -> 运行模式（编辑类页面按服务模式的参数集来拼命令行）
PAGE_MODE = {
    "server": "server", "cli": "cli", "gen": "gen",
    "load": "server", "chat": "server", "spec": "server",
}

# 导航分组：(组标题, 组英文, [页面 id])
NAV_GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("模型", "Models", ("lib",)),
    ("运行模式", "Run mode", ("server", "cli", "gen")),
    ("参数（跟随所选模型）", "Parameters", ("load", "chat", "spec")),
)

SECTIONS: Tuple[Tuple[str, str, str, str], ...] = (
    ("load", "basic", "基础", "Basic"),
    ("load", "lora", "LoRA 适配器", "LoRA Adapters"),
    ("load", "perf", "性能与显存", "Performance & VRAM"),
    ("load", "kv", "KV 缓存", "KV Cache"),
    ("load", "vbr", "VBR 动态量化", "VBR Dynamic KV"),
    ("load", "mem", "内存", "Memory"),
    ("load", "dev", "设备与放置", "Devices & Placement"),
    ("load", "misc", "其它", "Misc"),
    ("chat", "tpl", "模板与推理", "Template & Reasoning"),
    ("chat", "sample", "采样", "Sampling"),
    ("spec", "main", "方式与草稿模型", "Method & Draft"),
    ("spec", "draft", "草稿参数", "Draft params"),
    ("spec", "ngram", "n-gram 参数", "n-gram params"),
    ("spec", "res", "草稿模型资源", "Draft resources"),
)

SECTION_ZH: Dict[Tuple[str, str], str] = {(s[0], s[1]): s[2] for s in SECTIONS}
SECTION_EN: Dict[Tuple[str, str], str] = {(s[0], s[1]): s[3] for s in SECTIONS}

# 这些小节由专门的界面组件渲染（不是普通的一行行参数）
CUSTOM_SECTIONS: Dict[str, Tuple[str, ...]] = {"load": ("lora",)}

# 参数页顶部会被“选中模型”上下文影响的页面
MODEL_SCOPED_PAGES: Tuple[str, ...] = ("load", "chat", "spec")


def sections_for(page: str) -> Tuple[Tuple[str, str], ...]:
    return tuple((s[1], s[2]) for s in SECTIONS if s[0] == page)


# --------------------------------------------------------------------------- #
# 控件类型
# --------------------------------------------------------------------------- #
K_BOOL = "bool"
K_INT = "int"
K_FLOAT = "float"
K_TEXT = "text"
K_CHOICE = "choice"
K_CHOICE_EDIT = "choice_edit"
K_GEAR = "gear"                # 挡位：一排互斥按钮
K_META = "meta"                # 只用于界面与参数组装，不直接生成命令行
K_OPEN = "open_file"
K_SAVE = "save_file"
K_DIR = "dir"
K_MULTI = "multiline"

FILE_KINDS = (K_OPEN, K_SAVE, K_DIR)
CHOICE_KINDS = (K_CHOICE, K_CHOICE_EDIT, K_GEAR)
VALUED_KINDS = (K_INT, K_FLOAT, K_TEXT, K_CHOICE, K_CHOICE_EDIT, K_GEAR,
                K_OPEN, K_SAVE, K_DIR, K_MULTI)


def _default_threads() -> str:
    n = os.cpu_count() or 4
    n = max(1, n // 2)
    if n % 2:
        n += 1
    return str(n)


@dataclass
class F:
    key: str
    zh: str
    en: str
    page: str
    section: str = ""
    flag: str = ""
    kind: str = K_TEXT
    default: Any = ""
    on: bool = False
    choices: Sequence[str] = ()
    argmap: Optional[Dict[str, Sequence[str]]] = None
    ini: Optional[Dict[str, str]] = None      # 挡位/下拉 -> ini 取值
    long: str = ""                            # ini / --models-preset 用的长参数名
    hint: str = ""
    modes: FrozenSet[str] = ALL_MODES
    width: int = 16
    placeholder: str = ""
    roles: FrozenSet[str] = ALL_MODES         # 仅服务页有意义
    scope_override: str = ""
    positional: bool = False                  # 真正的「位置参数」（如量化的输入/输出）
    multi_value: bool = False                 # 允许逗号分隔的多值（如 --spec-type）

    # -------------------------------------------------------------- 派生
    def default_value(self) -> str:
        d = self.default
        if callable(d):
            d = d()
        return "" if d is None else str(d)

    @property
    def scope(self) -> str:
        if self.scope_override:
            return self.scope_override
        if self.page in ("load", "spec"):
            return "load"
        if self.page == "chat":
            return "chat"
        return "run"

    @property
    def ini_key(self) -> str:
        if self.long:
            return self.long
        if self.flag.startswith("--"):
            return self.flag[2:]
        return ""

    def all_flags(self) -> Tuple[str, ...]:
        out = []
        if self.flag:
            out.append(self.flag)
        for argv in (self.argmap or {}).values():
            out.extend(t for t in argv if t.startswith("-"))
        return tuple(dict.fromkeys(out))


# 常用模式集合
#
# 三个 exe 的公共参数并不完全一致（llama-completion 少了一批）：
#   _ALL_RUN   三个 exe 都认的项
#   _NO_GEN    只有 llama-completion 不认（--mmproj / -cram / --chat-template-kwargs ...）
#   _SERVER    只有 llama-server 认的项（路由组、KV 统一缓存、VBR 的缓存控制 ...）
_ALL_RUN = frozenset(MODE_ORDER)
_NO_GEN = frozenset({"server", "cli"})
_SERVER = frozenset({"server"})
_CLI_GEN = frozenset({"cli", "gen"})
_ROLE_ALL = frozenset(ROLE_ORDER)      # 现在只有 LLM 一个角色
_LLM_ONLY = frozenset({"llm"})

# buun-llama-cpp 的 KV 档位（-ct / -ctk / -ctv 的合法取值）。
# 前半段是 GGML 标准量化；中间三个是 fork 自带的 TurboQuant KV 编解码；
# 带 _tcq 后缀的是 TCQ 变体（需要 codebooks 里的码本文件，启动器会自动注入
# TURBO_TCQ_CB / TURBO_TCQ_CB2）；最后 vbr 是「按显存压力动态逐层降级」，
# 也是本 build 的出厂默认。
_KV_TYPES: Tuple[str, ...] = (
    "f16", "bf16", "f32",
    "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl",
    "turbo8", "turbo4", "turbo3",
    "turbo3_tcq", "turbo2_tcq", "turbo1_tcq",
    "vbr",
)

# 这些档位的 KV block 是 128 个值：要求模型 head_dim 能被 128 整除，
# 并且必须开 -fa（turbo / VBR 都依赖 FA 的 KV 路径）。
_BLOCK128_TYPES: FrozenSet[str] = frozenset(
    ("turbo8", "turbo4", "turbo3",
     "turbo3_tcq", "turbo2_tcq", "turbo1_tcq", "vbr"))

# 草稿模型的 KV 档位：和主模型同一套，但没有 vbr（草稿不做动态降级）。
_KV_TYPES_DRAFT: Tuple[str, ...] = tuple(
    t for t in _KV_TYPES if t != "vbr")

# --------------------------------------------------------------------------- #
# VBR 的「比特单价」表
#
# --vbr-entry / --vbr-floor / --vbr-reclaim-floor 都收同一套阶梯别名
# （common/arg.cpp 的 common_vbr_budget_to_type）。这里把别名换算成
# bits/value，用来做「floor 不能高于 entry」这类前置校验。
#
# ⚠️ 数值是**实测**的，不是从 help 里抄的整数 —— help 只给「聚合下限」的口径，
# 真实单价含量化块的 scale 开销。本机把 0.6B 模型逐档跑一遍，
# 从 /slots 的 kv_bpv 读回来的结果：
#     f16/bf16 = 16.0    q8_0 = 8.5     q5_1 = 6.0    q5_0 = 5.5
#     q4_1 = 5.0         q4_0 = 4.5     iq4_nl = 4.5
#     turbo8 = 8.125     turbo4 = 4.125 turbo3 = 3.5
#     turbo3_tcq = 3.25  turbo2_tcq = 2.25  turbo1_tcq = 1.25
# 这些也正是引擎内部 llama_vbr_type_bits_per_value() 的取值 ——
# 日志里 "floor 4.125 bits/value" / "floor 1.25 bits/value" 都对得上。
# --------------------------------------------------------------------------- #
VBR_BPV: Dict[str, float] = {
    "f16": 16.0, "fp16": 16.0, "16": 16.0,
    "turbo8": 8.125, "t8": 8.125, "8": 8.125,
    "turbo4": 4.125, "t4": 4.125, "4": 4.125,
    "turbo3_tcq": 3.25, "t3tcq": 3.25, "t3": 3.25, "3": 3.25,
    "turbo2_tcq": 2.25, "t2tcq": 2.25, "t2": 2.25, "2": 2.25,
    "turbo1_tcq": 1.25, "t1tcq": 1.25, "t1": 1.25, "1": 1.25,
    "q8_0": 8.5, "q8": 8.5, "q4_0": 4.5, "q4": 4.5,
    "bf16": 16.0, "q5_1": 6.0, "q5_0": 5.5, "q4_1": 5.0,
    "iq4_nl": 4.5, "turbo3": 3.5,
}

# 降级阶梯（从高到低）—— 用来把实测 bpv 反推回「最接近且不高于它」的档位
VBR_LADDER: Tuple[Tuple[float, str], ...] = (
    (16.0, "f16"), (8.5, "q8_0"), (8.125, "turbo8"), (6.0, "q5_1"),
    (5.5, "q5_0"), (5.0, "q4_1"), (4.5, "q4_0"), (4.125, "turbo4"),
    (3.5, "turbo3"), (3.25, "turbo3_tcq"), (2.25, "turbo2_tcq"),
    (1.25, "turbo1_tcq"),
)

# --vbr-floor 的挡位：(显示名, 传给引擎的别名)。空别名 = 不传，用引擎默认。
# 默认 turbo4 —— 比引擎在「显式 -ct vbr」时的隐式下限（t1 = 1.25 bpv）保守得多，
# 代价是能开的上下文会短一截（详见 README）。
VBR_FLOOR_GEARS: Tuple[Tuple[str, str], ...] = (
    ("跟随引擎", ""),
    ("turbo8", "turbo8"),
    ("turbo4", "turbo4"),
    ("turbo3_tcq", "turbo3_tcq"),
    ("turbo2_tcq", "turbo2_tcq"),
    ("turbo1_tcq", "turbo1_tcq"),
)
VBR_FLOOR_CHOICES: Tuple[str, ...] = tuple(k for k, _ in VBR_FLOOR_GEARS)
VBR_FLOOR_ARGMAP: Dict[str, Tuple[str, ...]] = {
    k: (("--vbr-floor", v) if v else ()) for k, v in VBR_FLOOR_GEARS}
# 只有这些挡位是 turbo 阶梯的（classic 阶梯不认）
VBR_TURBO_FLOORS: FrozenSet[str] = frozenset(
    v for _k, v in VBR_FLOOR_GEARS if v.startswith("turbo"))

# --------------------------------------------------------------------------- #
# 服务页的两种「运行方式」（互斥）
#
#   单模型网关   llama-server 只加载一个模型、绑内部端口；本程序做网关，
#               按请求里的 model 换模型（现有行为）。
#   多模型路由   llama-server 带 --models-dir / --models-preset 自己驻留多个
#               模型，按 model 名路由；此时网关只做纯透传。
#
# 两者不能叠加：路由模式下网关不再抢锁换模型。
# --------------------------------------------------------------------------- #
RUN_GATEWAY = "单模型网关"
RUN_ROUTER = "多模型路由"
RUN_KINDS: Tuple[str, ...] = (RUN_GATEWAY, RUN_ROUTER)

# 只有路由模式才该出现的参数项
ROUTER_KEYS: FrozenSet[str] = frozenset(
    ("models_dir", "models_max", "models_preset", "models_autoload"))

# ⚠️ 这些参数**只在 KV 档位确实是 vbr 时才允许输出**。
# 引擎对它们有硬校验（common/arg.cpp）：
#     --vbr-* flags need a VBR cache side: use -ctk vbr / -ctv vbr,
#     or drop the explicit non-vbr cache types
# 也就是说「-ct q8_0 + --vbr-floor turbo4」会直接拒绝启动。
# 因为 --vbr-floor 默认就是开启的，不加这道闸，任何非 vbr 的启动都会挂。
VBR_KEYS: FrozenSet[str] = frozenset((
    "vbr_codec", "vbr_entry", "vbr_floor", "vbr_vram",
    "vbr_reclaim_floor", "vbr_reset_keep_frac",
    "vbr_prompt_cache", "vbr_anchor_cache",
))


def run_kind_of(snapshot: Dict[str, Any]) -> str:
    """当前快照选的运行方式；认不出来就退回单模型网关。"""
    st = (snapshot or {}).get("run_kind") or {}
    if not st.get("on"):
        return RUN_GATEWAY
    val = str(st.get("value") or "").strip()
    return val if val in RUN_KINDS else RUN_GATEWAY


def is_router(snapshot: Dict[str, Any]) -> bool:
    return run_kind_of(snapshot) == RUN_ROUTER

# --------------------------------------------------------------------------- #
# 参数表
# --------------------------------------------------------------------------- #

FIELDS: Tuple[F, ...] = (
    # ===================================================================== #
    # 服务页
    # ===================================================================== #
    F("host", "监听地址", "Host (--host)", page="server", flag="--host",
      kind=K_CHOICE_EDIT, default="127.0.0.1", on=True,
      choices=("127.0.0.1", "0.0.0.0", "localhost"), modes=frozenset({"server"}),
      roles=_ROLE_ALL,
      hint="绑定的网卡地址。127.0.0.1 只允许本机访问；0.0.0.0 允许局域网访问"
           "（注意不要直接暴露到公网）。"),
    F("port", "统一端口", "Port (--port)", page="server", flag="--port",
      kind=K_INT, default="1233", on=True, modes=frozenset({"server"}),
      roles=_ROLE_ALL,
      hint="对外唯一的端口：控制 API、OpenAI 兼容接口（/v1/...）、WebUI "
           "都在这里。\n"
           "llama-server 自己会绑到一个内部端口（自动挑空闲口），由本网关转发，"
           "所以两者不会冲突；你只需要记住这一个端口。\n"
           "第一次绑不上时会自动往下找一个能用的端口并记住它。"),
    F("auto_switch", "按请求自动加载 / 切换", "Auto load & switch on request",
      page="server", kind=K_BOOL, on=True, modes=frozenset({"server"}),
      roles=_LLM_ONLY, scope_override="ui",
      hint="打开后，请求里 model 指定的模型没在跑时：卸载当前模型 → 加载目标模型 "
           "→ 再处理这个请求（要等加载完，模型越大越久）。\n"
           "期间其它请求排队等待；等这次切换结束后，排队的请求再按目标模型依次进行。\n"
           "关掉则只服务当前已加载的模型，请求别的模型直接返回 409。"),
    F("idle_unload", "空闲自动卸载（分钟）", "Idle auto-unload (minutes)",
      page="server", kind=K_INT, default="0", on=True,
      modes=frozenset({"server"}), roles=_LLM_ONLY, scope_override="ui",
      hint="连续这段时间没有收到任何推理请求，就自动卸载模型、把显存还回去；"
           "下次有请求时再自动加载。0 = 不自动卸载。"),
    F("parallel", "最大并发预测数", "Max Concurrent Predictions (-np)",
      page="server", flag="-np", kind=K_INT, default="1", on=True,
      modes=frozenset({"server"}), roles=_LLM_ONLY, long="parallel",
      hint="同时处理的请求槽位数量。每多一个槽位额外占用一份 KV 缓存，"
           "并在槽位之间均分上下文长度。"),
    F("api_key", "API 密钥", "API key (--api-key)", page="server",
      flag="--api-key", kind=K_TEXT, on=False, width=22,
      modes=frozenset({"server"}), roles=_ROLE_ALL,
      hint="设置后客户端需带上该密钥；留空表示不鉴权。"),
    F("webui", "内置 WebUI", "Web UI (--webui)", page="server", flag="--webui",
      kind=K_CHOICE, choices=("auto", "llamacpp", "none"), on=False,
      modes=frozenset({"server"}), roles=_ROLE_ALL,
      hint="auto=默认界面，llamacpp=新版界面，none=关闭网页界面只留 API。"),
    F("cache_ram", "对话缓存 (RAM)", "Prompt cache in RAM (-cram)",
      page="server", flag="-cram", kind=K_INT, on=False,
      modes=_NO_GEN, roles=_LLM_ONLY, long="cache-ram",
      hint="把历史对话的 KV 缓存保留在内存里的上限（MiB），再次命中时几乎不用重新"
           "处理提示词。内存紧张可设 0 关闭。默认 8192。"),
    F("no_warmup", "跳过预热", "No warmup (--no-warmup)", page="server",
      flag="--no-warmup", kind=K_BOOL, on=False, modes=frozenset({"server"}),
      roles=_ROLE_ALL,
      hint="启动时不跑一次空推理预热。首次请求会稍慢一点。"),

    # ---- 运行方式：单个模型走网关，还是让 llama-server 自己挂多个模型
    F("run_kind", "运行方式", "Run kind", page="server", kind=K_GEAR,
      default=RUN_GATEWAY, choices=RUN_KINDS,
      argmap={RUN_GATEWAY: (), RUN_ROUTER: ()},
      on=True, modes=frozenset({"server"}), roles=_ROLE_ALL,
      scope_override="ui", width=16,
      hint="两种互斥的运行方式：\n"
           "· 单模型网关 —— llama-server 只加载一个模型，绑内部端口；"
           "本程序在统一端口上做网关，请求别的模型时自动换模型。\n"
           "· 多模型路由 —— llama-server 带 --models-dir 自己驻留多个模型"
           "（同时最多 --models-max 个），按请求里的 model 名路由；"
           "此时网关只做透传，不再插手换模型。\n"
           "两者不要叠加：路由模式下「按请求自动加载 / 切换」会被忽略。"),
    F("models_dir", "路由模型目录", "Router models dir (--models-dir)",
      page="server", flag="--models-dir", kind=K_DIR, on=False, width=34,
      modes=frozenset({"server"}), roles=_ROLE_ALL,
      hint="多模型路由要扫描的目录（可递归发现 .gguf）。\n"
           "通常直接填「模型库」里那个根目录即可。"),
    F("models_max", "路由同时驻留上限", "Router max models (--models-max)",
      page="server", flag="--models-max", kind=K_INT, default="4", on=False,
      modes=frozenset({"server"}), roles=_ROLE_ALL,
      hint="路由服务同时加载几个模型（默认 4，0 = 不限制）。\n"
           "注意它们共享同一块显存，开太多会互相挤爆。"),
    F("models_preset", "路由预置文件", "Router preset INI (--models-preset)",
      page="server", flag="--models-preset", kind=K_OPEN, on=False, width=34,
      modes=frozenset({"server"}), roles=_ROLE_ALL,
      hint="INI 格式的每模型参数表。段名就是模型名，键是去掉 -- 的长参数名：\n"
           "  [*]            ← 全局，对所有模型生效\n"
           "  ctx-size = 32768\n"
           "  [Qwen3-8B]     ← 单个模型（支持 [名字:Q4_K_M] 这种量化后辍）\n"
           "  n-gpu-layers = 999\n"
           "本程序可以按「加载参数」页里保存的预设自动生成这个文件，"
           "见服务页的「导出路由预置 INI」。"),
    F("models_autoload", "路由自动加载", "Router autoload", page="server",
      kind=K_GEAR, default="开启", choices=("开启", "关闭"),
      argmap={"开启": (), "关闭": ("--no-models-autoload",)},
      ini={"开启": "true", "关闭": "false"}, long="models-autoload",
      on=False, modes=frozenset({"server"}), roles=_ROLE_ALL,
      hint="路由服务是否按需自动加载模型。关掉之后只有显式请求才加载。\n"
           "（默认就是开启，所以「开启」不产出参数。）"),
    # ===================================================================== #
    # 对话（llama-cli）
    #
    # 本 build 的 llama-cli 默认就是对话模式：没有 -cnv（不需要），
    # 也没有 -i / -no-cnv（要这些请走「生成」页的 llama-completion）。
    # 剩下的可控项就是首轮提示词、系统提示、是否只跑一轮。
    # ===================================================================== #
    F("prompt", "首轮提示词", "Prompt (-p)", page="cli", flag="-p",
      kind=K_TEXT, on=False, width=40, modes=frozenset({"cli"}),
      long="prompt",
      hint="进入对话时先替你发出去的第一句话。留空就直接等你输入。"),
    F("sys_prompt", "系统提示", "System prompt (-sys)", page="cli",
      flag="-sys", kind=K_TEXT, on=False, width=40, modes=frozenset({"cli"}),
      long="system-prompt",
      hint="对话开头插入的系统提示词（system role）。"),
    F("single_turn", "只跑一轮", "Single turn (-st)", page="cli",
      flag="-st", kind=K_BOOL, on=False, modes=frozenset({"cli"}),
      long="single-turn",
      hint="只完成一轮对话就退出，不等你继续输入。适合配合首轮提示词做单次问答。"),

    # ===================================================================== #
    # 生成（llama-completion）
    #
    # 这个 exe 默认是「纯补全」：给提示词 → 出结果 → 退出。它保留了
    # -cnv / -i / -st 这些开关，所以既能当一次性生成用，也能当交互终端。
    # ===================================================================== #
    F("g_prompt", "提示词", "Prompt (-p)", page="gen", flag="-p",
      kind=K_TEXT, on=False, width=40, modes=frozenset({"gen"}),
      long="prompt",
      hint="要补全 / 生成的提示词。给完就出结果然后退出。"),
    F("g_sys_prompt", "系统提示", "System prompt (-sys)", page="gen",
      flag="-sys", kind=K_TEXT, on=False, width=40, modes=frozenset({"gen"}),
      long="system-prompt",
      hint="对话模式下插入的系统提示词（不勾「多轮对话」时一般用不上）。"),
    F("g_conversation", "多轮对话", "Conversation (-cnv)", page="gen",
      flag="-cnv", kind=K_BOOL, on=False, modes=frozenset({"gen"}),
      long="conversation",
      hint="按对话模板套用会话格式（相当于临时变成 llama-cli 的行为）。"
           "不勾就是原始补全：把你的提示词原样喂进去。"),
    F("g_interactive", "交互模式", "Interactive (-i)", page="gen", flag="-i",
      kind=K_BOOL, on=False, modes=frozenset({"gen"}),
      long="interactive",
      hint="跑完提示词后不退出，继续等你输入。"),
    F("g_single_turn", "只跑一轮", "Single turn (-st)", page="gen",
      flag="-st", kind=K_BOOL, on=False, modes=frozenset({"gen"}),
      long="single-turn",
      hint="对话模式下调一次就退出，不进入交互循环。"),
    # ===================================================================== #
    # 加载参数（跟模型走）
    # ===================================================================== #
    F("lora", "LoRA 适配器", "LoRA adapters (--lora / --lora-scaled)",
      page="load", section="lora", flag="--lora", kind=K_META, on=True,
      modes=_ALL_RUN,
      hint="给这个模型挂 LoRA 适配器（.gguf / .safetensors），一个模型可以挂多个，\n"
           "按列表顺序依次应用。每个适配器可以有自己的比例（scale）:\n"
           "  · 比例 = 1.0 → --lora 路径\n"
           "  · 其它比例 → --lora-scaled 路径 比例（可以是负数做反向）\n"
           "列表按**模型**保存：切到别的模型会带出它自己的那份。\n"
           "这是加载期参数，改完要重启模型（或让它重新加载）才生效。"),
    F("mmproj", "多模态投影文件", "Vision projector (--mmproj)", page="load",
      section="basic", flag="--mmproj", kind=K_OPEN, on=False, width=34,
      modes=_NO_GEN, long="mmproj",
      hint="视觉模型需要的投影权重（mmproj-*.gguf）。\n"
           "模型库里如果同目录存在 mmproj，选中该模型时会自动填上。"),
    F("mmproj_gpu_swap", "投影按需进显存",
      "Projector GPU swap (--mmproj-gpu-swap)", page="load",
      section="basic", flag="--mmproj-gpu-swap", kind=K_BOOL, on=False,
      modes=_SERVER, long="mmproj-gpu-swap",
      hint="**投影文件先加载在内存（CPU），只有真的来了带图片的请求才把它换进显存**，\n"
           "媒体请求处理完再换回内存 —— 省显存，代价是首次看图时多一次搬运。\n\n"
           "源码里的原话：`Without a speculative model, the swap simply keeps "
           "mmproj on CPU until an image arrives.`\n"
           "启动日志会打 `loaded multimodal model on CPU (GPU swap enabled)`。\n\n"
           "要腾显存时会先把可重载的投机上下文（MTP / 外部 DFlash）换出去，\n"
           "用完再恢复。若草稿模型是**不可重载**的类型，这个功能不可用，\n"
           "引擎只会打一行警告并让两者都常驻。\n"
           "只有 llama-server 有这个参数；需要先指定「多模态投影文件」。"),
    F("mmproj_offload", "投影 GPU 卸载", "Projector GPU offload (--mmproj-offload)",
      page="load", section="basic", flag="--mmproj-offload", kind=K_GEAR,
      default="自动（默认已开启）",
      choices=("自动（默认已开启）", "强制关闭"),
      argmap={"自动（默认已开启）": (),
              "强制关闭": ("--no-mmproj-offload",)},
      long="mmproj-offload", on=False, modes=_NO_GEN,
      hint="投影文件是否允许放进显存。默认开启。\n"
           "强制关闭 = 投影**永远留在内存**（CPU），每次看图都在 CPU 上算，\n"
           "比「按需进显存」更省显存但更慢。\n"
           "和上面那个的区别：这里是「永不放显存」，"
           "上面是「用时放、用完撤」。"),
    F("mmproj_device", "投影所在设备", "Projector device (-mmdev)", page="load",
      section="basic", flag="-mmdev", kind=K_TEXT, on=False, width=16,
      placeholder="CUDA0 / none", modes=_NO_GEN, long="mmproj-device",
      hint="多模态投影放在哪块设备上。\n"
           "填 none = 不卸载到显存（等价于关掉投影 GPU 卸载）。\n"
           "留空 = 跟随 --device 的设置。用 --list-devices 可以看设备名。"),
    F("ctx", "上下文长度", "Context size (-c)", page="load", section="basic",
      flag="-c", kind=K_INT, default="", on=True, modes=_ALL_RUN,
      long="ctx-size",
      hint="提示词 + 生成内容的总长度（token）。显存/内存的主要消耗项。\n\n"
           "· **留空（默认）= 自动**：不传 -c，引擎自己算 —— 从模型的训练上下文\n"
           "  （本机多数模型是 262144）起算，按**可用显存**往下缩，\n"
           "  KV 成本按「最低档位下限」那一档的单价计价，下限是下面的"
           "「自动时的下限」。\n"
           "· 填数字 = 指定，直接传 -c N。\n\n"
           "实测（27B，可用显存 4875 MiB）自动算出的结果：\n"
           "  下限 turbo1_tcq → 262144、turbo8 → 156672、f16 → 80640。\n"
           "所以要开长上下文，就把「最低档位下限」压狠一点。\n"
           "自动算出来的实际值可以在「运行状态」里实时看到。\n"
           "多槽位并行（-np > 1）时会按槽位数均分。"),
    F("fit_ctx", "自动时的下限", "Fit min ctx (--fit-ctx)", page="load",
      section="basic", flag="--fit-ctx", kind=K_INT, default="4096",
      on=False, modes=_ALL_RUN, long="fit-ctx",
      hint="自动算上下文时，最多只允许缩到这么小（引擎默认 4096）。\n"
           "只有「上下文长度」留空（自动）时才起作用。\n"
           "设大 = 宁可少放几层也不肯缩上下文；设小 = 允许缩得更狠来换显存。"),
    F("ngl", "GPU 卸载层数", "GPU offload layers (-ngl)", page="load",
      section="basic", flag="-ngl", kind=K_INT, default="999", on=True,
      modes=_ALL_RUN, long="n-gpu-layers",
      hint="放到显存里的层数。填 999 表示能放多少放多少（引擎会按实际层数封顶）。\n"
           "显存不足就从模型层数往下调。"),
    F("threads", "生成线程数", "Threads (-t)", page="load", section="basic",
      flag="-t", kind=K_INT, default=_default_threads, on=True, modes=_ALL_RUN,
      long="threads",
      hint="Token 生成阶段的 CPU 线程数。建议等于物理核心数、且取偶数。"),
    F("threads_batch", "批处理线程数", "Threads batch (-tb)", page="load",
      section="basic", flag="-tb", kind=K_INT, default=_default_threads,
      on=False, modes=_ALL_RUN, long="threads-batch",
      hint="提示词处理（prefill）阶段的线程数。全量卸载到 GPU 时可以填小一点（如 2）。"),
    F("predict", "生成上限", "Predict (-n)", page="load", section="basic",
      flag="-n", kind=K_INT, default="-1", on=False, modes=_ALL_RUN,
      long="n-predict",
      hint="-1 = 不限制；-2 = 生成到塞满上下文。一般保持默认即可。"),
    F("keep", "保留提示前 N 个 token", "Keep (--keep)", page="load",
      section="basic", flag="--keep", kind=K_INT, on=False,
      modes=_ALL_RUN, long="keep",
      hint="上下文滚动时始终保留最前面 N 个 token（通常是系统提示）。"
           "-1 表示全部保留。"),

    F("batch", "评估批处理大小", "Batch size / logical (-b)", page="load",
      section="perf", flag="-b", kind=K_INT, default="2048", on=True,
      modes=_ALL_RUN, long="batch-size",
      hint="逻辑最大批大小。调大通常能提升提示词处理速度，但更吃显存。默认 2048。"),
    F("ubatch", "物理批处理大小", "Physical batch size (-ub)",
      page="load", section="perf", flag="-ub", kind=K_INT, default="512",
      on=True, modes=_ALL_RUN, long="ubatch-size",
      hint="物理最大批大小（官方参数名 Physical Batch Size）。\n"
           "实际一次计算的数据块上限，显存峰值主要由它决定。默认 512。\n"
           "提示词处理偏慢可以适当调大；CUDA OOM 就调小。"),
    F("fa", "快速注意力", "Flash Attention (-fa / --no-fa)", page="load",
      section="perf", flag="-fa", kind=K_GEAR, default="开启",
      choices=("开启", "关闭", "自动"),
      argmap={"开启": ("-fa", "1"), "关闭": ("-fa", "0"), "自动": ()},
      ini={"开启": "true", "关闭": "false"},
      long="flash-attn", on=True, modes=_ALL_RUN,
      hint="Flash Attention：提速并降低显存占用，多数 build 默认开启。\n"
           "本项会写成 -fa 1 / -fa 0。\n"
           "选「自动」＝不传参数，交给引擎决定（本 build 默认就是开）。\n"
           "注意：turbo / TCQ 系列 KV 档位和 VBR 都要求 FA 开启。"),
    F("fit", "自动适配显存", "Auto fit VRAM (--fit)", page="load",
      section="perf", flag="--fit", kind=K_GEAR, default="自动（默认已开启）",
      choices=("自动（默认已开启）", "强制开启", "强制关闭"),
      argmap={"自动（默认已开启）": (), "强制开启": ("--fit", "on"),
              "强制关闭": ("--fit", "off")},
      long="fit", on=False, modes=_ALL_RUN,
      hint="让引擎自己按可用显存调整「没显式设过的」参数，目标是别 OOM。\n"
           "本 build 默认就是开启的，所以一般留「自动」。\n"
           "它同时负责**自动算上下文长度**（见「上下文长度」那一项）：\n"
           "从模型训练上下文往下缩，缩到刚好塞进可用显存为止。\n"
           "注意：只要你显式设了 -ngl，--fit 就不会再改它；\n"
           "关掉 --fit 就等于关掉自动算上下文。"),
    F("fit_target", "显存保留余量", "Fit target margin (--fit-target)",
      page="load", section="perf", flag="--fit-target", kind=K_TEXT,
      on=False, width=14, placeholder="1024", modes=_ALL_RUN,
      long="fit-target",
      hint="--fit 给每块显卡保留多少空闲显存（MiB），逗号分隔，单个值广播到所有卡。"
           "默认 1024。\n"
           "余量越大 → 能留给 KV 的显存越少 → **自动算出的上下文越短**（或卸载层数越少）。\n"
           "加载时 CUDA OOM 就调大；显存没吃满、想开更长上下文就调小。"),
    F("repack", "权重重排", "Repack (--repack / -nr)", page="load",
      section="perf", flag="--repack", kind=K_GEAR, default="自动（默认已开启）",
      choices=("自动（默认已开启）", "强制开启", "强制关闭"),
      argmap={"自动（默认已开启）": (), "强制开启": ("--repack",),
              "强制关闭": ("--no-repack",)},
      long="repack", on=False, modes=_ALL_RUN,
      hint="加载时把权重重排成更高效的布局（相当于 ik 那边的 run-time repack）。\n"
           "本 build 默认就是开启的，所以一般留「自动」即可；\n"
           "关闭只在排查问题时用（`-nr` / `--no-repack`）。"),

    F("kvo", "KV 缓存卸载到 GPU", "KV cache offload (-nkvo)", page="load",
      section="kv", flag="-nkvo", kind=K_GEAR, default="开启",
      choices=("开启", "关闭"),
      argmap={"开启": (), "关闭": ("-nkvo",)},
      ini={"开启": "true", "关闭": "false"}, long="kv-offload",
      on=True, modes=_ALL_RUN,
      hint="KV 缓存放在显存里可大幅提速；显存不够或想省显存时选「关闭」，"
           "让 KV 留在内存（会慢一些）。"),

    # ---- 统一档位：一次同时设 K 和 V
    F("ct", "统一 KV 档位", "Both K and V (-ct)", page="load", section="kv",
      flag="-ct", kind=K_CHOICE_EDIT, default="vbr", on=False,
      choices=_KV_TYPES, long="cache-type", modes=_ALL_RUN,
      hint="一次设好 K 和 V 两侧的档位（等价于 -ctk X -ctv X）。\n"
           "写在 -ctk / -ctv 之前，后面的单项会覆盖对应那一侧。\n\n"
           "【本 build 的档位】\n"
           "· 常规：f32 / f16 / bf16 / q8_0 / q5_1 / q5_0 / q4_1 / q4_0 / iq4_nl\n"
           "· turbo：turbo8 / turbo4 / turbo3（不是标准 GGML 量化，"
           "是 buun 的 TurboQuant KV 编解码）\n"
           "· TCQ：turbo3_tcq / turbo2_tcq / turbo1_tcq（需要码本文件，"
           "启动器会自动注入 TURBO_TCQ_CB / TURBO_TCQ_CB2）\n"
           "· vbr：动态按显存压力逐 (层, 侧) 降级（本 build 的出厂默认）\n\n"
           "限制：turbo / TCQ 的 KV block 是 128 个值，要求模型的 "
           "n_embd_head_k 是 128 的倍数，且必须开 -fa。"),
    F("ctk", "K 缓存量化类型", "K cache type (-ctk)", page="load", section="kv",
      flag="-ctk", kind=K_CHOICE_EDIT, default="vbr", on=False,
      choices=_KV_TYPES, long="cache-type-k", modes=_ALL_RUN,
      hint="键（K）缓存的量化档位。\n"
           "本 build 的默认是 vbr（动态降级，隐式 t4 地板），"
           "所以界面上显示 vbr 却不出参数时就是引擎默认行为。\n"
           "只设 K 一侧会让 V 侧保持默认档位，要两侧一起改请用上面的「统一 KV 档位」。"),
    F("ctv", "V 缓存量化类型", "V cache type (-ctv)", page="load", section="kv",
      flag="-ctv", kind=K_CHOICE_EDIT, default="vbr", on=False,
      choices=_KV_TYPES, long="cache-type-v", modes=_ALL_RUN,
      hint="值（V）缓存的量化档位。V 对量化比 K 更敏感，"
           "固定档位时一般比 K 保守一档（例如 K=q8_0、V=f16）。"),
    F("kv_unified", "统一 KV 缓冲", "Unified KV (-kvu / -no-kvu)",
      page="load", section="kv", flag="-kvu", kind=K_GEAR, default="自动",
      choices=("自动", "开启", "关闭"),
      argmap={"自动": (), "开启": ("-kvu",), "关闭": ("-no-kvu",)},
      long="kv-unified", on=False, modes=_SERVER,
      hint="所有序列共用一个 KV 缓冲（unified KV）。\n"
           "VBR / turbo 都要求它开启；server 在单槽位时本来就会自动打开。\n"
           "多槽位并行时开启能让较长的对话借用其他槽位空闲的容量。\n"
           "只有 llama-server 认这个参数。"),
    F("kv_unified_per_slot", "每槽上下文上限", "Per-slot ctx (--kv-unified-per-slot)",
      page="load", section="kv", flag="--kv-unified-per-slot", kind=K_INT,
      on=False, modes=_SERVER, long="kv-unified-per-slot",
      hint="统一 KV 池下每个并行槽位的上下文上限。\n"
           "设了它又没设 -c 时，共享池会按 并发数 × 这个值 来分配。"),
    F("no_context_shift", "禁用上下文滚动",
      "No context shift (--no-context-shift)", page="load", section="kv",
      flag="--no-context-shift", kind=K_BOOL, on=False, modes=_ALL_RUN,
      long="no-context-shift",
      hint="上下文写满后不再丢弃最前面的内容、改为直接停止。"),
    F("cache_reuse", "KV 复用最小块", "Cache reuse (--cache-reuse)", page="load",
      section="kv", flag="--cache-reuse", kind=K_INT, on=False, modes=_SERVER,
      long="cache-reuse",
      hint="超过这个长度的相同片段就尝试从缓存里复用（靠 KV shifting 挪过来），"
           "而不是重新算一遍提示词。\n"
           "多轮对话里每次只加一点内容时收益明显；设太会拖慢首 token。"
           "只对 llama-server 有效。"),

    # ===================================================================== #
    # VBR 动态 KV 量化（buun 的看家功能）
    #
    # 只有把 -ct / -ctk / -ctv 选成 vbr 时，这一节才真正起作用。
    # 它按显存压力逐 (层, 侧) 把 KV 往下阶梯降级：
    #   f16 → turbo8 → turbo4 → turbo3_tcq → turbo2_tcq → turbo1_tcq
    # 降级顺序由 --vbr-codec 选的阶梯决定，最低点由 --vbr-floor 兜住。
    # ===================================================================== #
    F("vbr_codec", "降级阶梯", "VBR codec (--vbr-codec)", page="load",
      section="vbr", flag="--vbr-codec", kind=K_CHOICE_EDIT, default="auto",
      choices=("auto", "turbo", "classic"), on=False, modes=_ALL_RUN,
      long="vbr-codec",
      hint="VBR 往下降级时用哪套阶梯：\n"
           "· auto（默认）—— 每一层 KV 都支持 Turbo 时用 Turbo，否则退回 classic\n"
           "· turbo —— f16 → t8 → t4 → t3/t2/t1（压缩率最高）\n"
           "· classic —— f16 → q8_0 → q4_0（只在 BailingMoE3/Ling 这类模型上有）\n"
           "选 turbo 或 auto 时，模型必须满足 128 block 的要求。"),
    F("vbr_entry", "起始档位", "VBR entry tier (--vbr-entry)", page="load",
      section="vbr", flag="--vbr-entry", kind=K_CHOICE_EDIT, default="f16",
      choices=("f16", "t8", "t4", "t3", "t2", "t1", "q8_0", "q4_0"),
      on=False, modes=_ALL_RUN, long="vbr-entry",
      hint="从哪一档开始（默认 f16，质量优先）。\n"
           "把它设低就是主动「拿质量换容量」——起手就压缩，"
           "之后还会继续往「最低档位下限」降。\n"
           "q8_0 / q4_0 只有在 --vbr-codec classic 下才合法。\n"
           "约束：它不能低于「最低档位下限」（引擎会报 "
           "`--vbr-floor cannot exceed --vbr-entry`）。"),
    F("vbr_floor", "最低档位下限", "VBR floor (--vbr-floor)", page="load",
      section="vbr", flag="--vbr-floor", kind=K_GEAR, default="turbo4",
      choices=VBR_FLOOR_CHOICES, argmap=VBR_FLOOR_ARGMAP,
      on=True, modes=_ALL_RUN, long="vbr-floor",
      hint="降级阶梯的**最低点**：控制器降到不违反这个下限的那一档就停。\n"
           "各档的比特单价：turbo8 = 8、turbo4 = 4.125、turbo3_tcq = 3、\n"
           "turbo2_tcq = 2、turbo1_tcq = 1.25 bits/value。\n"
           "「跟随引擎」= 不传参数：显式 -ct vbr 时引擎的隐式下限是 "
           "turbo1_tcq（1.25 bpv，压缩最狠）。\n\n"
           "⚠️ 这个挡位**直接决定能开多长的上下文** —— 它同时是 fit 计算\n"
           "上下文容量时用的 KV 单价。实测同一台机器上 27B：\n"
           "  turbo1_tcq → 262144（撞模型上限）、turbo8 → 156672、f16 → 80640。\n"
           "所以下限压得越低，上下文能开越长、但 KV 精度越低。\n\n"
           "约束：不能高于 --vbr-entry（引擎会直接报错）；\n"
           "「降级阶梯」选 classic 时只能写 q8_0 / q4_0。\n"
           "只在 KV 档位是 vbr 时生效。另一个名字 --vbr-min-bits 是同一个参数。"),
    F("vbr_vram", "KV 显存预算", "VBR VRAM budget (--vbr-vram)", page="load",
      section="vbr", flag="--vbr-vram", kind=K_TEXT, on=False, width=12,
      placeholder="auto / 8G / 24576M", modes=_ALL_RUN, long="vbr-vram-budget",
      hint="留给 KV 的显存预算，用来驱动**运行时降级控制器**：\n"
           "占用的 KV 逼近这个数时，控制器按价格顺序把 (层, 侧) 往下转码。\n"
           "auto（默认）= 模型与开销之外剩下的全部显存，由 fit 算出来。\n"
           "写具体大小：8G / 24576M / 字节数。\n\n"
           "注意：**它不参与上下文长度的计算**。实测给它 16M（远低于全上下文"
           "成本）时引擎也只是警告一句、把最深的填充降级，n_ctx 不受影响。\n"
           "上下文长度由「可用显存 + --vbr-floor 单价」决定（见上面的说明）。\n"
           "另一个名字 --vbr-vram-budget 是同一个参数。"),
    F("vbr_reclaim_floor", "回收下限", "VBR reclaim floor (--vbr-reclaim-floor)",
      page="load", section="vbr", flag="--vbr-reclaim-floor", kind=K_TEXT,
      on=False, width=10, placeholder="t4", modes=_SERVER,
      long="vbr-reclaim-floor",
      hint="降级之前，先把空闲槽位的 KV 缓存清掉——但被清的部分不会低于这个档位。\n"
           "只有 llama-server 的动态 VBR 有效。"),
    F("vbr_reset_keep_frac", "保留比例",
      "VBR reset keep fraction (--vbr-reset-keep-frac)", page="load",
      section="vbr", flag="--vbr-reset-keep-frac", kind=K_FLOAT, on=False,
      width=10, placeholder="0.5", modes=_SERVER, long="vbr-reset-keep-frac",
      hint="已经降级的对话在重置后能复用不到这个比例时，就整体重来一遍。"
           "0~1 之间的小数。"),
    F("vbr_prompt_cache", "VBR 提示词缓存", "VBR prompt cache",
      page="load", section="vbr", flag="--vbr-prompt-cache", kind=K_GEAR,
      default="自动", choices=("自动", "开启", "关闭"),
      argmap={"自动": (), "开启": ("--vbr-prompt-cache",),
              "关闭": ("--no-vbr-prompt-cache",)},
      long="vbr-prompt-cache", on=False, modes=_SERVER,
      hint="缓存提示词的紧凑 VBR 副本，让重复的系统提示不用重新编码。\n"
           "只有 llama-server 有（cli / completion 没这个开关）。"),
    F("vbr_anchor_cache", "质量锚点缓存",
      "VBR anchor cache MiB (--vbr-anchor-cache-mib)", page="load",
      section="vbr", flag="--vbr-anchor-cache-mib", kind=K_INT, on=False,
      width=10, modes=_SERVER, long="vbr-anchor-cache-mib",
      hint="质量锚点缓存的容量（MiB）。0 = 只做紧凑缓存。\n"
           "需要先开「VBR 提示词缓存」。"),

    F("mlock", "保持模型在内存中", "Keep model in memory (--mlock)",
      page="load", section="mem", flag="--mlock", kind=K_BOOL, on=False,
      modes=_ALL_RUN, long="mlock",
      hint="锁定模型所在内存页，阻止系统把模型换到磁盘/压缩。内存充足时建议开启，"
           "能避免卡顿。"),
    F("no_mmap", "禁用内存映射", "Disable mmap (--no-mmap)", page="load",
      section="mem", flag="--no-mmap", kind=K_BOOL, on=False, modes=_ALL_RUN,
      long="no-mmap",
      hint="加载时把模型整个读进内存，而不是按需从磁盘映射。加载更慢，"
           "但能减少缺页导致的卡顿。"),

    # ---- 设备与张量放置（原 ik 专属小节的通用部分，ik 独有的 -mla/-fmoe/
    # ---- -gr/-sas/-mqkv/-ser/-vq 本 build 都没有，已整体移除）
    F("ot", "张量放置覆盖", "Override tensor placement (-ot)", page="load",
      section="dev", flag="-ot", kind=K_MULTI, on=False, modes=_ALL_RUN,
      long="override-tensor",
      hint="用正则指定某些张量放在哪个设备，一行一条，例如：\n"
           r"\.ffn_.*_exps\.=CPU" "\n" r"blk\.(3|4)\.ffn_.*=CUDA0" "\n"
           "MoE 模型想让专家层留在内存、其余进显存时最有用的开关。\n"
           "草稿模型对应的是 --override-tensor-draft（-otd）。"),
    F("cmoe", "MoE 权重全部留在 CPU", "Keep all MoE in CPU (--cpu-moe)",
      page="load", section="dev", flag="--cpu-moe", kind=K_BOOL, on=False,
      modes=_ALL_RUN, long="cpu-moe",
      hint="所有 MoE 权重都放内存。显存很小又想跑大 MoE 时的省事选择。"),
    F("ncmoe", "前 N 层 MoE 留在 CPU",
      "MoE in CPU for first N layers (--n-cpu-moe)", page="load", section="dev",
      flag="--n-cpu-moe", kind=K_INT, on=False, modes=_ALL_RUN,
      long="n-cpu-moe",
      hint="比 --cpu-moe 更细：只把前 N 层的专家权重留在内存。"),
    F("sm", "多卡拆分模式", "Split mode (-sm)", page="load", section="dev",
      flag="-sm", kind=K_CHOICE, choices=("none", "layer", "row", "tensor"),
      on=False, modes=_ALL_RUN, long="split-mode",
      hint="多张显卡时的拆分方式：\n"
           "· none —— 只用一张卡（用 -mg 指定）\n"
           "· layer（默认）—— 按层切分，流水线并行\n"
           "· row —— 按行切权重，并行计算\n"
           "· tensor —— 按张量切权重与 KV，实验特性"),
    F("ts", "张量拆分比例", "Tensor split (-ts)", page="load", section="dev",
      flag="-ts", kind=K_TEXT, on=False, width=14, placeholder="3,1",
      modes=_ALL_RUN, long="tensor-split",
      hint="各显卡分到多少模型，逗号分隔，例如 3,1 表示 75% / 25%。"),
    F("device", "指定使用的设备", "Devices (-dev)", page="load", section="dev",
      flag="-dev", kind=K_TEXT, on=False, width=18, placeholder="CUDA0,CUDA1",
      modes=_ALL_RUN, long="device",
      hint="只使用指定的设备。用 llama-server --list-devices 可以查看设备名。"),
    F("main_gpu", "主显卡序号", "Main GPU (-mg)", page="load", section="dev",
      flag="-mg", kind=K_INT, on=False, modes=_ALL_RUN, long="main-gpu",
      hint="拆分模式为 none 时模型放第几块卡；拆分模式为 row 时，"
           "中间结果与 KV 放哪块卡。"),

    F("special", "输出特殊 token", "Special tokens (-sp)", page="load",
      section="misc", flag="-sp", kind=K_BOOL, on=False, modes=_ALL_RUN,
      long="special",
      hint="打印时把特殊 token（如 <|im_start|>）显示出来，调试模板时有用。"),
    # 注意：ik_llama.cpp 的「只算内存不加载」是 -dr / --dry-run，
    # buun 这套 exe 里**没有** dry-run，而 -dr 被占给了 --docker-repo
    # （Docker Hub 模型仓库，还要跟一个值）。留着会变成一个会报错的陷阱项，
    # 所以整项删除。想估显存请用「自动适配显存」(--fit) 或直接看启动日志。
    F("extra", "额外参数（每行一个）", "Extra args (one per line)",
      page="load", section="misc", flag="", kind=K_MULTI, on=False,
      modes=ALL_MODES, positional=True,
      hint="本界面没有覆盖到的参数直接写在这里，一行一个，例如：\n"
           "--numa distribute" "\n" "--rope-scaling yarn"),

    # ===================================================================== #
    # 对话参数（跟模型走）
    # ===================================================================== #
    F("jinja", "Jinja 模板", "Jinja (--jinja)", page="chat", section="tpl",
      flag="--jinja", kind=K_BOOL, on=True, modes=_ALL_RUN, long="jinja",
      hint="使用模型内置的 Jinja 对话模板。要用工具调用（function calling）"
           "就必须开启。"),
    F("chat_template", "自定义对话模板", "Chat template (--chat-template)",
      page="chat", section="tpl", flag="--chat-template", kind=K_CHOICE_EDIT,
      on=False, width=20,
      choices=("chatml", "llama3", "llama4", "deepseek2", "deepseek3",
               "deepseek-ocr", "gemma", "command-r", "granite", "phi4", "phi3",
               "mistral-v7", "bailing", "bailing-think", "gpt-oss", "kimi-k2",
               "hunyuan-moe", "exaone4", "chatglm4", "minicpm", "vicuna",
               "zephyr", "openchat", "seed_oss", "smolvlm", "solar-open",
               "grok-2", "monarch", "orion", "pangu-embedded"),
      long="chat-template", modes=_ALL_RUN,
      hint="模型自带的模板有问题时，用内置模板名覆盖。\n"
           "本 build 一共内置了 60 多个（含 gemma / deepseek-ocr / hunyuan-moe / "
           "exaone4 / monarch 等），这里只列常用的，其余可以直接手输。\n"
           "注意：本 build 的模板表里**没有 qwen** 这一项，Qwen 系请用 chatml 或"
           "让模型自带的模板生效。"),
    F("chat_template_file", "模板文件",
      "Chat template file (--chat-template-file)", page="chat", section="tpl",
      flag="--chat-template-file", kind=K_OPEN, on=False,
      long="chat-template-file", modes=_ALL_RUN,
      hint="用外部 .jinja 文件替换模型里的模板，不必重新下载模型。"),
    F("reasoning", "推理开关", "Reasoning (-rea)", page="chat", section="tpl",
      flag="-rea", kind=K_GEAR, default="自动",
      choices=("自动", "开", "关"),
      argmap={"自动": ("-rea", "auto"), "开": ("-rea", "on"),
              "关": ("-rea", "off")},
      ini={"自动": "auto", "开": "on", "关": "off"}, long="reasoning",
      on=True, modes=_ALL_RUN,
      hint="是否让模型输出思考过程。选「自动」由模型的对话模板决定。"),
    F("reasoning_effort", "推理强度",
      "Reasoning effort (--reasoning-effort)",
      page="chat", section="tpl", flag="--reasoning-effort", kind=K_GEAR,
      default="high",
      choices=("不设置", "minimal", "low", "medium", "high", "xhigh", "max"),
      argmap={"不设置": (),
              "minimal": ("--reasoning-effort", "minimal"),
              "low": ("--reasoning-effort", "low"),
              "medium": ("--reasoning-effort", "medium"),
              "high": ("--reasoning-effort", "high"),
              "xhigh": ("--reasoning-effort", "xhigh"),
              "max": ("--reasoning-effort", "max")},
      long="reasoning-effort", on=True, modes=_ALL_RUN,
      hint="真正的推理强度挡位，**本 build 原生支持**（--reasoning-effort）。\n"
           "它会把这个变量交给模型的 Jinja 模板，让模板自己切换深浅\n"
           "（Qwen3.8 / gpt-oss 这类模板会据此改写系统提示）。\n"
           "可选：minimal / low / medium / high / xhigh / max，"
           "「不设置」= 保持模板默认（default）。\n"
           "它和下面的「思考 token 上限」是两回事：这里是让模型**想多深**，\n"
           "那个是到多少 token 就**硬截断**。"),
    F("reasoning_budget", "思考 token 上限",
      "Reasoning budget (--reasoning-budget)",
      page="chat", section="tpl", flag="--reasoning-budget", kind=K_INT,
      default="", on=False, long="reasoning-budget", modes=_ALL_RUN,
      hint="思考过程的 token 上限：填 0 = 立刻结束思考，-1 = 不限制，\n"
           "N>0 = 最多想 N 个 token 就强制收尾。\n"
           "这是**硬截断**，不改变模型想多深；要调深浅请用上面的「推理强度」。\n"
           "留空/不勾 = 用引擎默认（-1）。"),
    F("reasoning_format", "思考内容格式",
      "Reasoning format (--reasoning-format)", page="chat", section="tpl",
      flag="--reasoning-format", kind=K_CHOICE, on=False,
      choices=("none", "deepseek", "deepseek-legacy"),
      long="reasoning-format", modes=_ALL_RUN,
      hint="思考内容放哪里：none 留在 content 里，deepseek 放进 "
           "reasoning_content。"),
    F("chat_template_kwargs", "模板附加参数",
      "Chat template kwargs (--chat-template-kwargs)", page="chat",
      section="tpl", flag="--chat-template-kwargs", kind=K_TEXT, on=False,
      width=30, placeholder='{"reasoning_effort": "medium"}',
      long="chat-template-kwargs", modes=_NO_GEN,
      hint="传给 Jinja 模板的额外 JSON 参数。例如 gpt-oss 系列可以用 "
           '{"reasoning_effort": "low|medium|high"} 控制推理强度。\n'
           "llama-completion 不认这个参数，所以「生成」模式下不生效。"),
    F("reasoning_budget_message", "预算耗尽提示语",
      "Reasoning budget message (--reasoning-budget-message)", page="chat",
      section="tpl", flag="--reasoning-budget-message", kind=K_TEXT, on=False,
      width=30, long="reasoning-budget-message", modes=_ALL_RUN,
      hint="思考预算用完之后、在结束思考标记前插入的一段提示文字。"),

    F("temp", "温度", "Temperature (--temp)", page="chat", section="sample",
      flag="--temp", kind=K_FLOAT, default="0.8", on=False, long="temp",
      modes=_ALL_RUN,
      hint="采样随机性。越低越确定、越高越发散。"),
    F("top_k", "TopK 采样", "Top-K (--top-k)", page="chat", section="sample",
      flag="--top-k", kind=K_INT, default="40", on=False, long="top-k",
      modes=_ALL_RUN,
      hint="只从概率最高的 K 个候选里采样。0 或 -1 表示不限制。"),
    F("top_p", "Top P 采样", "Top-P (--top-p)", page="chat", section="sample",
      flag="--top-p", kind=K_FLOAT, default="0.95", on=False, long="top-p",
      modes=_ALL_RUN,
      hint="核采样：只在累计概率达到 P 的最小候选集合里采样。"),
    F("min_p", "最小 P 采样", "Min-P (--min-p)", page="chat", section="sample",
      flag="--min-p", kind=K_FLOAT, default="0.05", on=False, long="min-p",
      modes=_ALL_RUN,
      hint="丢弃概率低于「最高概率 × min_p」的候选。0 表示停用。"),
    F("repeat_penalty", "重复惩罚", "Repeat penalty (--repeat-penalty)",
      page="chat", section="sample", flag="--repeat-penalty", kind=K_FLOAT,
      default="1.1", on=False, long="repeat-penalty", modes=_ALL_RUN,
      hint="惩罚已出现过的 token，缓解复读。1.0 表示不改动。"),
    F("repeat_last_n", "惩罚回看长度", "Repeat last N (--repeat-last-n)",
      page="chat", section="sample", flag="--repeat-last-n", kind=K_INT,
      default="64", on=False, long="repeat-last-n", modes=_ALL_RUN,
      hint="参与重复惩罚的最近 token 数量。0 表示关闭，-1 表示全上下文。"),
    F("seed", "随机种子", "Seed (--seed)", page="chat", section="sample",
      flag="--seed", kind=K_INT, default="-1", on=False, long="seed",
      modes=_ALL_RUN,
      hint="-1 每次随机；固定数值可复现同一结果。"),
    F("samplers", "采样器链", "Sampler chain (--samplers)", page="chat",
      section="sample", flag="--samplers", kind=K_TEXT, on=False, width=34,
      default="dry;top_k;tfs_z;typical_p;top_p;min_p;xtc;top_n_sigma;"
              "temperature;adaptive_p",
      long="samplers", modes=_ALL_RUN,
      hint="按顺序执行的采样器列表，用分号分隔。只保留 min_p;temperature "
           "可以简化采样流程。"),

    # ===================================================================== #
    # 推测解码（跟模型走）
    #
    # buun 的推测解码和 ik 是两套完全不同的接口：
    #   --spec-type 收的是**逗号分隔的类型名列表**（可以同时挂多种），
    #   但**不接受** ik 那种 `dflash:n_max=4` 的内联写法；
    #   所有细项都是各自独立的 --spec-* 参数。
    # 所以这里每一项都是真参数，不再往 --spec-type 里合成。
    #
    # 另外本 build 只有 llama-server / llama-cli 支持推测解码，
    # llama-completion 一个 --spec-* 都不认（下面全部限定为 server+cli）。
    # ===================================================================== #
    F("spec_enable", "启用推测解码", "Speculative decoding", page="spec",
      section="main", flag="", kind=K_BOOL, on=False, modes=_NO_GEN,
      hint="总开关。关闭时这一页所有参数都不会写进命令行。"),
    F("spec_type", "推测方式", "Spec type (--spec-type)", page="spec",
      section="main", flag="--spec-type", kind=K_CHOICE_EDIT,
      default="none", on=False, width=24, modes=_NO_GEN,
      long="spec-type", multi_value=True,
      choices=("none", "draft-simple", "draft-eagle3", "draft-mtp",
               "draft-dflash", "draft-dspark", "dflash", "ngram-simple",
               "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
               "suffix", "copyspec", "recycle"),
      hint="**逗号分隔的类型名列表**，可以同时挂多种（如 "
           "ngram-simple,suffix）。本 build 认这些名字：\n"
           "· draft-simple —— 普通独立草稿模型（配合 -md）\n"
           "· draft-eagle3 —— EAGLE3 草稿头\n"
           "· draft-mtp —— 用主模型自带的 MTP 预测头，**不需要第二个模型**\n"
           "· draft-dflash / draft-dspark / dflash —— 块扩散草稿，一次并行出一整块\n"
           "· ngram-simple / ngram-map-k / ngram-map-k4v / ngram-mod / "
           "ngram-cache —— 纯统计推测，不需要额外模型\n"
           "· suffix —— 后缀匹配\n"
           "· copyspec / recycle —— 复制/回收式草稿\n\n"
           "注意：本 build **不支持** `dflash:n_max=4` 这种内联写法——\n"
           "所有细项都在下面各自的小节里，是独立的 --spec-* 参数。"),
    F("spec_default", "用默认推测配置", "Spec default (--spec-default)",
      page="spec", section="main", flag="--spec-default", kind=K_BOOL,
      on=False, modes=_NO_GEN, long="spec-default",
      hint="让引擎自己挑一套默认的推测解码配置（覆盖 --spec-type 的选择）。"),
    F("spec_dflash_default", "用 DFlash 默认配置",
      "Spec DFlash default (--spec-dflash-default)", page="spec",
      section="main", flag="--spec-dflash-default", kind=K_BOOL, on=False,
      modes=_NO_GEN, long="spec-dflash-default",
      hint="一键套用 DFlash 的推荐参数（需要先用 -md 指定草稿模型）。"),
    F("model_draft", "草稿模型", "Draft model (-md)", page="spec",
      section="main", flag="-md", kind=K_OPEN, on=False, width=38,
      modes=_NO_GEN, long="model-draft",
      hint="提供草稿的模型（draft-simple / draft-dflash / draft-dspark / "
           "draft-eagle3 需要）。\n"
           "draft-mtp 与 ngram-* / suffix 不需要。\n"
           "在模型库里把某个模型「用作推测草稿」就会自动填到这里。"),
    F("spec_draft_replace", "词汇翻译对",
      "Spec replace TARGET DRAFT (--spec-draft-replace)", page="spec",
      section="main", flag="--spec-draft-replace", kind=K_TEXT, on=False,
      width=30, placeholder="旧词 新词", modes=_NO_GEN, long="spec-draft-replace",
      hint="草稿模型和主模型的词表不一致时，把 TARGET 翻译成 DRAFT。\n"
           "填法：两个词用空格隔开，例如 `舊 旧`。"),
    F("spec_mtp_vocab_size", "MTP 词表上限",
      "Spec MTP vocab size (--spec-mtp-vocab-size)", page="spec",
      section="main", flag="--spec-mtp-vocab-size", kind=K_INT, on=False,
      width=10, placeholder="32768", modes=_SERVER, long="spec-mtp-vocab-size",
      hint="Qwen-27B 的 MTP 均衡词表大小：0 关闭，32768 启用。"),
    F("spec_synth_len", "合成接受长度",
      "Spec synth len (--spec-synth-len)", page="spec", section="main",
      flag="--spec-synth-len", kind=K_FLOAT, on=False, width=10,
      modes=_NO_GEN, long="spec-synth-len",
      hint="目标平均合成接受长度（含目标 token 本身）。**只用于基准测试**。"),
    F("spec_synth_rates", "合成接受概率",
      "Spec synth rates (--spec-synth-rates)", page="spec", section="main",
      flag="--spec-synth-rates", kind=K_TEXT, on=False, width=22,
      placeholder="0.8,0.6,0.4", modes=_NO_GEN, long="spec-synth-rates",
      hint="逗号分隔的、按位置排列的无条件合成接受概率。**只用于基准测试**。"),

    # ---- 草稿行为
    F("spec_draft_n_max", "单次最多草稿数", "Max draft tokens (--spec-draft-n-max)",
      page="spec", section="draft", flag="--spec-draft-n-max", kind=K_INT,
      default="3", on=False, modes=_NO_GEN, long="spec-draft-n-max",
      hint="一次推测最多草拟多少个 token。默认 3。\n"
           "草稿越多越省时间，但被拒时浪费也越多。"),
    F("spec_draft_n_min", "最少草稿数", "Min draft tokens (--spec-draft-n-min)",
      page="spec", section="draft", flag="--spec-draft-n-min", kind=K_INT,
      default="0", on=False, modes=_NO_GEN, long="spec-draft-n-min",
      hint="草稿数低于这个值就不提交验证，直接按普通方式生成。0 = 不由数量决定。"),
    F("spec_draft_p_min", "最低接受概率", "Draft p-min (--spec-draft-p-min)",
      page="spec", section="draft", flag="--spec-draft-p-min", kind=K_FLOAT,
      default="0.0", on=False, modes=_NO_GEN, long="spec-draft-p-min",
      hint="草稿 token 的概率低于该值就丢掉（贪心式过滤）。0 表示不设下限。"),
    F("spec_draft_p_split", "拆分概率", "Draft p-split (--spec-draft-p-split)",
      page="spec", section="draft", flag="--spec-draft-p-split", kind=K_FLOAT,
      default="0.10", on=False, modes=_NO_GEN, long="spec-draft-p-split",
      hint="推测解码的拆分概率。默认 0.10。调大 = 更早切回主模型。"),
    F("spec_draft_temp", "草稿采样温度", "Draft temp (--spec-draft-temp)",
      page="spec", section="draft", flag="--spec-draft-temp", kind=K_FLOAT,
      on=False, modes=_NO_GEN, long="spec-draft-temp",
      hint="草稿模型的采样温度。留空 = 跟随目标模型的设置；0 = 贪心。"),
    F("spec_draft_backend_sampling", "草稿采样放后端",
      "Draft backend sampling (--spec-draft-backend-sampling)", page="spec",
      section="draft", flag="--spec-draft-backend-sampling", kind=K_GEAR,
      default="自动（默认已开启）",
      choices=("自动（默认已开启）", "强制关闭"),
      argmap={"自动（默认已开启）": (),
              "强制关闭": ("--no-spec-draft-backend-sampling",)},
      long="spec-draft-backend-sampling", on=False, modes=_NO_GEN,
      hint="把草稿模型的采样放到后端设备上执行（默认开启）。\n"
           "草稿在主模型同卡上跑时开着通常更快；排查异常时可以关掉。"),
    F("spec_dspark_gpu_assist", "DSpark GPU 辅助",
      "DSpark GPU assist (--spec-dspark-gpu-assist)", page="spec",
      section="draft", flag="--spec-dspark-gpu-assist", kind=K_GEAR,
      default="自动（默认已开启）",
      choices=("自动（默认已开启）", "强制关闭"),
      argmap={"自动（默认已开启）": (),
              "强制关闭": ("--no-spec-dspark-gpu-assist",)},
      long="spec-dspark-gpu-assist", on=False, modes=_SERVER,
      hint="草稿主干留在 CPU、MoE 缓存没关时，把 DSpark 的轻量尾巴放到 GPU 上。"),

    # ---- n-gram 系列（不需要草稿模型，纯统计推测）
    # 注意：--spec-ngram-size-n / -size-m / -min-hits 这三个旧参数在本 build
    # 已经被上游删除，改成按算法分别命名 —— 所以下面每个算法各有自己的一份。
    F("spec_ngram_simple_size_n", "simple 查询长度",
      "ngram-simple size-n", page="spec", section="ngram",
      flag="--spec-ngram-simple-size-n", kind=K_INT, default="12", on=False,
      modes=_NO_GEN, long="spec-ngram-simple-size-n",
      hint="ngram-simple 用于查找的 n-gram 长度。默认 12。"),
    F("spec_ngram_simple_size_m", "simple 草稿长度",
      "ngram-simple size-m", page="spec", section="ngram",
      flag="--spec-ngram-simple-size-m", kind=K_INT, default="48", on=False,
      modes=_NO_GEN, long="spec-ngram-simple-size-m",
      hint="ngram-simple 草拟的 m-gram 长度。默认 48。"),
    F("spec_ngram_simple_min_hits", "simple 最少命中",
      "ngram-simple min-hits", page="spec", section="ngram",
      flag="--spec-ngram-simple-min-hits", kind=K_INT, default="1",
      on=False, modes=_NO_GEN, long="spec-ngram-simple-min-hits",
      hint="ngram-simple 至少要命中多少次才拿来做草稿。默认 1。"),
    F("spec_ngram_map_k_size_n", "map-k 查询长度",
      "ngram-map-k size-n", page="spec", section="ngram",
      flag="--spec-ngram-map-k-size-n", kind=K_INT, default="12", on=False,
      modes=_NO_GEN, long="spec-ngram-map-k-size-n",
      hint="ngram-map-k 用于查找的 n-gram 长度。默认 12。"),
    F("spec_ngram_map_k_size_m", "map-k 草稿长度",
      "ngram-map-k size-m", page="spec", section="ngram",
      flag="--spec-ngram-map-k-size-m", kind=K_INT, default="48", on=False,
      modes=_NO_GEN, long="spec-ngram-map-k-size-m",
      hint="ngram-map-k 草拟的 m-gram 长度。默认 48。"),
    F("spec_ngram_map_k_min_hits", "map-k 最少命中",
      "ngram-map-k min-hits", page="spec", section="ngram",
      flag="--spec-ngram-map-k-min-hits", kind=K_INT, default="1", on=False,
      modes=_NO_GEN, long="spec-ngram-map-k-min-hits",
      hint="ngram-map-k 至少要命中多少次才拿来做草稿。默认 1。"),
    F("spec_ngram_map_k4v_size_n", "map-k4v 查询长度",
      "ngram-map-k4v size-n", page="spec", section="ngram",
      flag="--spec-ngram-map-k4v-size-n", kind=K_INT, default="12", on=False,
      modes=_NO_GEN, long="spec-ngram-map-k4v-size-n",
      hint="ngram-map-k4v 用于查找的 n-gram 长度。默认 12。"),
    F("spec_ngram_map_k4v_size_m", "map-k4v 草稿长度",
      "ngram-map-k4v size-m", page="spec", section="ngram",
      flag="--spec-ngram-map-k4v-size-m", kind=K_INT, default="48", on=False,
      modes=_NO_GEN, long="spec-ngram-map-k4v-size-m",
      hint="ngram-map-k4v 草拟的 m-gram 长度。默认 48。"),
    F("spec_ngram_map_k4v_min_hits", "map-k4v 最少命中",
      "ngram-map-k4v min-hits", page="spec", section="ngram",
      flag="--spec-ngram-map-k4v-min-hits", kind=K_INT, default="1",
      on=False, modes=_NO_GEN, long="spec-ngram-map-k4v-min-hits",
      hint="ngram-map-k4v 至少要命中多少次才拿来做草稿。默认 1。"),
    F("spec_ngram_mod_n_min", "mod 最短 n-gram",
      "ngram-mod n-min", page="spec", section="ngram",
      flag="--spec-ngram-mod-n-min", kind=K_INT, default="48", on=False,
      modes=_NO_GEN, long="spec-ngram-mod-n-min",
      hint="ngram-mod 里用于推测的最短 n-gram 长度。默认 48。"),
    F("spec_ngram_mod_n_max", "mod 最长 n-gram",
      "ngram-mod n-max", page="spec", section="ngram",
      flag="--spec-ngram-mod-n-max", kind=K_INT, default="64", on=False,
      modes=_NO_GEN, long="spec-ngram-mod-n-max",
      hint="ngram-mod 里用于推测的最长 n-gram 长度。默认 64。"),
    F("spec_ngram_mod_n_match", "mod 查找长度",
      "ngram-mod n-match", page="spec", section="ngram",
      flag="--spec-ngram-mod-n-match", kind=K_INT, default="24", on=False,
      modes=_NO_GEN, long="spec-ngram-mod-n-match",
      hint="ngram-mod 的查找长度。默认 24。"),

    # ---- 草稿模型自己的资源与放置
    F("ngld", "草稿模型 GPU 层数", "Draft GPU layers (-ngld)", page="spec",
      section="res", flag="-ngld", kind=K_INT, on=False, modes=_NO_GEN,
      long="n-gpu-layers-draft",
      hint="草稿模型放到显存里的层数（用法同 -ngl）。"),
    F("cd", "草稿模型上下文", "Draft ctx (-cd)", page="spec", section="res",
      flag="-cd", kind=K_INT, on=False, modes=_NO_GEN,
      long="ctx-size-draft",
      hint="草稿模型自己的上下文长度。填 0 = 继承主模型的每槽容量；"
           "草稿一般只需要几百个 token。"),
    F("ctkd", "草稿 K 缓存类型", "Draft K cache (-ctkd)", page="spec",
      section="res", flag="-ctkd", kind=K_CHOICE_EDIT, on=False, width=12,
      choices=_KV_TYPES_DRAFT, long="cache-type-k-draft", modes=_ALL_RUN,
      hint="草稿模型 KV 缓存的 K 档位。默认 f16。\n"
           "注意草稿这边**没有 vbr**，但 turbo / tcq 系列都有。"),
    F("ctvd", "草稿 V 缓存类型", "Draft V cache (-ctvd)", page="spec",
      section="res", flag="-ctvd", kind=K_CHOICE_EDIT, on=False, width=12,
      choices=_KV_TYPES_DRAFT, long="cache-type-v-draft", modes=_ALL_RUN,
      hint="草稿模型 KV 缓存的 V 档位。默认 f16。"),
    F("td", "草稿生成线程数", "Draft threads (-td)", page="spec",
      section="res", flag="-td", kind=K_INT, on=False, modes=_NO_GEN,
      long="threads-draft",
      hint="草稿模型生成阶段的 CPU 线程数。"),
    F("tbd", "草稿批处理线程数", "Draft threads batch (-tbd)", page="spec",
      section="res", flag="-tbd", kind=K_INT, on=False, modes=_NO_GEN,
      long="threads-batch-draft",
      hint="草稿模型批处理阶段的 CPU 线程数。"),
    F("devd", "草稿模型设备", "Draft device (-devd)", page="spec",
      section="res", flag="-devd", kind=K_TEXT, on=False, width=18,
      placeholder="CUDA0", modes=_NO_GEN, long="device-draft",
      hint="草稿模型只放在哪些设备上。留空则跟随主模型的放置。"),
    F("otd", "草稿张量放置", "Draft override tensor (-otd)", page="spec",
      section="res", flag="-otd", kind=K_MULTI, on=False, modes=_NO_GEN,
      long="override-tensor-draft",
      hint="按正则覆盖草稿模型某些张量的存放位置，一行一条，"
           "格式与「张量放置覆盖」相同。"),
    F("cmoed", "草稿 MoE 全留 CPU", "Draft cpu-moe (-cmoed)", page="spec",
      section="res", flag="-cmoed", kind=K_BOOL, on=False, modes=_NO_GEN,
      long="cpu-moe-draft",
      hint="草稿模型的 MoE 权重全部留在内存。"),
    F("ncmoed", "草稿前 N 层 MoE 留 CPU",
      "Draft n-cpu-moe (-ncmoed)", page="spec", section="res",
      flag="-ncmoed", kind=K_INT, on=False, modes=_NO_GEN,
      long="n-cpu-moe-draft",
      hint="草稿模型前 N 层的 MoE 权重留在内存。"),
)

# --------------------------------------------------------------------------- #
# 查询辅助
# --------------------------------------------------------------------------- #

FIELD_BY_KEY: Dict[str, F] = {f.key: f for f in FIELDS}

# --spec-type 的合法类型名（buun 的 common_speculative_type_from_name_map）。
# 它收的是**逗号分隔的列表**，所以一项里可以同时写多个。
# 名字里的 draft / mtp 是 draft-simple / draft-mtp 的等价别名。
SPEC_TYPES: Tuple[str, ...] = tuple(FIELD_BY_KEY["spec_type"].choices)

# 需要独立草稿模型（-md）的类型。
# draft-mtp / mtp 不需要：它用的是主模型自带的 MTP 层
# （模型没有 MTP 层时引擎只会打一行 warning 然后跳过）。
# ngram-* / suffix / copyspec / recycle 都是纯统计，也不需要。
SPEC_NEEDS_DRAFT: FrozenSet[str] = frozenset(
    ("draft-simple", "draft", "draft-eagle3", "draft-dflash", "draft-dspark",
     "dflash"))


def spec_type_names(raw: str) -> Tuple[str, ...]:
    """把 --spec-type 的值拆成一个个小写的类型名。"""
    return tuple(t.strip().lower() for t in str(raw or "").split(",") if t.strip())


def fields_for(page: str) -> Tuple[F, ...]:
    return tuple(f for f in FIELDS if f.page == page)


def fields_in(page: str, section: str) -> Tuple[F, ...]:
    return tuple(f for f in FIELDS if f.page == page and f.section == section)


def fields_for_mode(mode: str) -> Tuple[F, ...]:
    return tuple(f for f in FIELDS if mode in f.modes)


def pages_with_fields() -> Tuple[str, ...]:
    return tuple(p[0] for p in PAGES if p[0] == "lib" or fields_for(p[0]))


def default_snapshot() -> Dict[str, Dict[str, Any]]:
    return {f.key: {"on": bool(f.on), "value": f.default_value()}
            for f in FIELDS}


def fields_by_scope(scope: str) -> Tuple[F, ...]:
    return tuple(f for f in FIELDS if f.scope == scope)


def preset_keys(page: str) -> Tuple[str, ...]:
    """页面上会被存进预设的字段（排除纯界面开关）。"""
    return tuple(f.key for f in FIELDS
                 if f.page == page and f.scope in ("load", "chat"))


def editable_keys(page: str) -> Tuple[str, ...]:
    return tuple(f.key for f in FIELDS if f.page == page)
