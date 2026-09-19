# -*- coding: utf-8 -*-
"""配置与预设的持久化。

目录结构（都在 <启动器目录>/config/ 下）::

    config/
      app.json                 软件级设置（引擎路径、模型目录、界面偏好、运行参数…）
      models/
        Huihui-Qwen3.8-27B-abliterated.json   每个「配过参数的模型」一个文件
        Qwen3-30B-A3B-Instruct-IQ4_KSS.json
      cache/
        scan.json              模型库扫描缓存（GGUF 元数据，可随时重建）
      router-preset.ini        多模型路由的预置文件（用时生成）
      selftest.log

为什么这么拆
------------
原来所有东西挤在一个 ``config.json`` 里，实测那个文件 **205.7 KB，其中 175 KB 是
扫描缓存**（GGUF 元数据 + 对话模板），真正的「设置」只有 17 KB。而 ``save()``
是全量重写的 —— 每次改一个勾选框都要把 200 KB 重新序列化写盘。

拆开之后：

* **每个模型一个文件** —— 好备份、好分享、好手动改；某个模型调崩了删它一个文件就行。
* **扫描缓存单独放** —— 它是可重建的派生数据，不该混进「设置」，
  也不该在每次改参数时被重写。
* **app.json 只装软件级设置** —— 通常几 KB，重写无感。

写入策略
--------
``save()`` 不靠脏标记（界面里到处在用 ``snapshot_ref()`` 返回的字典**就地改值**，
标不干净），而是**序列化后和上次写盘的内容比对，变了才写**。
16 个模型 × 约 2 KB 的比对是亚毫秒级的，换来的是「绝不会漏写」。

模型文件的命名
--------------
文件名 = 模型库里显示的名字（好认），文件内的 ``path`` 才是权威标识：

* 显示名里的非法字符（``\\ / : * ? " < > |``）会被替换成 ``-``；
* 重名自动加 ``-2`` / ``-3`` 后缀；
* 文件名存在条目里的 ``file`` 字段，**显示名改了会把文件跟着改名**（老的删掉）。

老配置迁移
----------
首次启动发现老的 ``config.json`` 会自动拆成新结构，老文件另存一份
``config.legacy.json``（不删）。迁移说明放在 :attr:`Store.migration_note`。
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional

CONFIG_VERSION = 4

# 软件级设置（会进 app.json）。`models` 与 `model_cache` 不在其中 —— 它们各自分文件。
DEFAULTS: Dict[str, Any] = {
    "version": CONFIG_VERSION,
    "engine_path": "",
    "engine_history": [],
    "models_dir": "",
    "models_extra_dirs": [],
    "models_recursive": True,
    "models_depth": 3,
    "nav": "lib",
    "last_mode": "server",
    "recent_models": [],
    "recent_drafts": [],
    "recent_loras": [],
    "prefs": {
        "show_english": False,
        "autoscroll": True,
        "cmd_multiline": False,
        "export_dir": "",
        "window": "",
        "table_sort": ["name", False],
        "category": "全部",
        "only_set": False,
        "cap": "",
    },
    "roles": {
        "llm": {"model": "", "server": {}},
    },
    "run": {"cli": {}, "gen": {}},
    "control_api": {"enabled": True, "host": "127.0.0.1", "port": 8090,
                    "token": ""},
}

APP_KEYS = tuple(DEFAULTS.keys())
# 模型级作用域：load（加载参数 + 推测解码 + LoRA）/ chat（对话参数）/
# emb（embedding 专属，只有 embedding 类模型用）
SCOPES = ("load", "chat", "emb")

# 一个模型文件里会写的字段
_MODEL_FIELDS = ("path", "name", "file", "category_override",
                 "load", "chat", "emb", "load_presets", "chat_presets",
                 "last_load_preset", "last_chat_preset")

# Windows 文件名里不能出现的字符 + 控制字符
_BAD_NAME = re.compile(r'[\\/:*?"<>|\r\n\t\x00-\x1f]')
_MAX_NAME = 80


def model_key(path: str) -> str:
    """用规范化绝对路径作为模型唯一标识。"""
    if not path:
        return ""
    p = os.path.expanduser(str(path).strip().strip('"'))
    try:
        p = os.path.abspath(p)
    except OSError:
        pass
    return os.path.normcase(p)


def safe_file_name(name: str) -> str:
    """把显示名变成一个能当文件名的字符串。"""
    out = _BAD_NAME.sub("-", str(name or "").strip())
    out = out.strip(" .-")
    if len(out) > _MAX_NAME:
        out = out[:_MAX_NAME].rstrip(" .-")
    return out or "model"


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _write_json(path: str, payload: Any) -> bool:
    """原子写：先写临时文件再 replace，避免半截文件把配置弄坏。"""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _name_of(entry: Dict[str, Any], key: str) -> str:
    name = str(entry.get("name") or "")
    if name:
        return name
    p = str(entry.get("path") or key)
    return os.path.splitext(os.path.basename(p))[0]


class Store:
    """配置读写。公开方法与拆文件之前完全一致，调用方不用改。"""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.config_dir = os.path.join(base_dir, "config")
        self.models_dir = os.path.join(self.config_dir, "models")
        self.cache_dir = os.path.join(self.config_dir, "cache")
        self.app_path = os.path.join(self.config_dir, "app.json")
        self.cache_path = os.path.join(self.cache_dir, "scan.json")
        self.legacy_path = os.path.join(self.config_dir, "config.json")
        # 兼容老名字（拆之前"单文件"时代的路径变量）
        self.path = self.app_path

        self.data: Dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.data["models"] = {}
        self.data["model_cache"] = {}
        self.migration_note = ""

        self._files: Dict[str, str] = {}          # key -> 文件名（不含目录）
        self._bases: Dict[str, str] = {}          # key -> 上次算文件名用的显示名
        self._written_files: Dict[str, str] = {}  # 上次实际写在哪个文件
        self._written_app: Optional[str] = None
        self._written_models: Dict[str, str] = {}
        self._written_cache: Optional[str] = None
        # 由界面注入：给定模型路径，返回模型库里显示的名字。
        # 有它才能让配置文件名跟着「界面里看到的名字」走。
        self.name_resolver = None
        self.first_run = False
        self.load()
        self._write_defaults_on_first_run()

    def _write_defaults_on_first_run(self) -> None:
        """**首次运行就把默认配置落到数据目录**（一般是 exe 所在目录）。

        不落盘的话，「设置在程序旁边」这件事要等用户动过某个开关才成立 ——
        用户看不出便携有没有生效，验收时也无从断言（而且 exe 目录里空空的，
        很像没写权限）。老配置（config.json）存在时不动它，交给上面的迁移流程。
        """
        if os.path.isfile(self.app_path) or os.path.isfile(self.legacy_path):
            return
        self.first_run = True
        self._written_app = None        # 强制 save() 认为「还没写过」
        self.save()

    # ------------------------------------------------------------- 路径
    def model_file_path(self, key: str) -> str:
        return os.path.join(self.models_dir, self._assign_file(key))

    def _assign_file(self, key: str) -> str:
        """给一个模型挑文件名：显示名优先，重名加后缀。

        显示名**改了要重算**——否则改名后文件还叫老名字，
        打开 config\\models 看到的和界面上对不上。
        """
        cur = self._files.get(key)
        entry = self.data.get("models", {}).get(key) or {}
        base = safe_file_name(_name_of(entry, key))
        if cur and self._bases.get(key) == base:
            return cur                       # 名字没动，沿用原来的文件
        used = {v.lower() for k, v in self._files.items() if k != key}
        cand = base + ".json"
        i = 2
        while cand.lower() in used:
            cand = "%s-%d.json" % (base, i)
            i += 1
        self._files[key] = cand
        self._bases[key] = base
        return cand

    # ------------------------------------------------------------- 载入
    def load(self) -> None:
        loaded = _read_json(self.app_path)
        if not isinstance(loaded, dict):
            loaded = self._migrate_from_single_file()
        self.data = _deep_merge(DEFAULTS, loaded if isinstance(loaded, dict)
                                else {})
        self.data["model_cache"] = {}
        self._load_models()
        self._load_cache()
        self.data.setdefault("roles", {}).setdefault(
            "llm", {"model": "", "server": {}})
        for mode in ("cli", "gen"):
            self.data.setdefault("run", {}).setdefault(mode, {})
        # 记下「刚读进来的样子」，第一次 save() 就不会白写一遍
        self._written_files = dict(self._files)
        self._written_app = self._dump_app()
        self._written_models = {k: self._dump_model(k)
                                for k in self.data["models"]}
        self._written_cache = self._dump_cache()

    def _migrate_from_single_file(self) -> Dict[str, Any]:
        """把老的 config.json 拆成新结构，返回其中 app 级的那部分。"""
        old = _read_json(self.legacy_path)
        if not isinstance(old, dict) or not old:
            return {}
        app = {k: copy.deepcopy(old[k]) for k in APP_KEYS if k in old}
        models = old.get("models") if isinstance(old.get("models"), dict) else {}
        cache = (old.get("model_cache")
                 if isinstance(old.get("model_cache"), dict) else {})
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass
        # 先把文件名都排好（处理重名），再逐个写
        planned: Dict[str, str] = {}
        used: set = set()
        for key, entry in models.items():
            if not isinstance(entry, dict):
                continue
            base = safe_file_name(_name_of(entry, key))
            cand = base + ".json"
            i = 2
            while cand.lower() in used:
                cand = "%s-%d.json" % (base, i)
                i += 1
            used.add(cand.lower())
            planned[key] = cand
        n_models = 0
        for key, entry in models.items():
            if key not in planned:
                continue
            out = {k: entry[k] for k in _MODEL_FIELDS if k in entry}
            out["file"] = planned[key]
            if _write_json(os.path.join(self.models_dir, planned[key]), out):
                n_models += 1
        if cache:
            _write_json(self.cache_path, cache)
        _write_json(self.app_path, app)
        # 老文件改名留退路（不删，用户自己决定要不要清掉）。
        # 留着原名的 config.json 会让目录里出现两个「配置」看着迷惑，
        # 所以是真的 rename；rename 失败（被占用等）才退化为复制。
        try:
            os.replace(self.legacy_path,
                       os.path.join(self.config_dir, "config.legacy.json"))
        except OSError:
            try:
                shutil.copyfile(self.legacy_path,
                                os.path.join(self.config_dir,
                                             "config.legacy.json"))
            except OSError:
                pass
        self.migration_note = (
            "老配置已拆分：config.json → app.json + models/%d 个模型"
            "+ cache/scan.json（老文件另存为 config.legacy.json）" % n_models)
        return app

    def _load_models(self) -> None:
        models: Dict[str, Any] = {}
        self._files = {}
        self._bases = {}
        try:
            names = sorted(os.listdir(self.models_dir))
        except OSError:
            names = []
        for fn in names:
            if not fn.endswith(".json"):
                continue
            entry = _read_json(os.path.join(self.models_dir, fn))
            if not isinstance(entry, dict):
                continue
            key = model_key(str(entry.get("path") or ""))
            if not key:
                key = "?" + os.path.splitext(fn)[0].lower()
            models[key] = entry
            self._files[key] = str(entry.get("file") or fn)
            self._bases[key] = safe_file_name(_name_of(entry, key))
        self.data["models"] = models

    def _load_cache(self) -> None:
        cache = _read_json(self.cache_path)
        self.data["model_cache"] = cache if isinstance(cache, dict) else {}

    # --------------------------------------------------------- 序列化
    def _dump_app(self) -> str:
        payload = {k: self.data.get(k, copy.deepcopy(DEFAULTS.get(k)))
                   for k in APP_KEYS}
        payload["version"] = CONFIG_VERSION
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _dump_model(self, key: str) -> str:
        entry = self.data.get("models", {}).get(key) or {}
        out = {k: entry[k] for k in _MODEL_FIELDS if k in entry}
        out["file"] = self._files.get(key, "")
        return json.dumps(out, ensure_ascii=False, sort_keys=True)

    def _dump_cache(self) -> str:
        return json.dumps(self.data.get("model_cache") or {},
                          ensure_ascii=False, sort_keys=True)

    def save(self) -> None:
        """只写真正变过的那几份文件。"""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass

        app_now = self._dump_app()
        if app_now != self._written_app:
            if _write_json(self.app_path, json.loads(app_now)):
                self._written_app = app_now

        for key in list(self.data.get("models") or {}):
            entry = self.data["models"][key]
            # 显示名由界面提供（模型库里的名字）。变了就让文件名跟着换。
            if self.name_resolver is not None:
                try:
                    nm = str(self.name_resolver(str(entry.get("path") or ""))
                             or "")
                except Exception:  # noqa: BLE001
                    nm = ""
                if nm and nm != entry.get("name"):
                    entry["name"] = nm
                    self._files.pop(key, None)
            entry["file"] = self._assign_file(key)
            now = self._dump_model(key)
            if now == self._written_models.get(key) and \
                    self._written_files.get(key) == entry["file"]:
                continue
            if not _write_json(
                    os.path.join(self.models_dir, entry["file"]),
                    json.loads(now)):
                continue
            old = self._written_files.get(key)
            if old and old != entry["file"]:
                try:                       # 显示名改了 → 删掉老文件
                    os.remove(os.path.join(self.models_dir, old))
                except OSError:
                    pass
            self._written_files[key] = entry["file"]
            self._written_models[key] = now

        cache_now = self._dump_cache()
        if cache_now != self._written_cache:
            if _write_json(self.cache_path, json.loads(cache_now)):
                self._written_cache = cache_now

    # ------------------------------------------------------- 简单字段
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def pref(self, key: str, default: Any = None) -> Any:
        return (self.data.get("prefs") or {}).get(key, default)

    def set_pref(self, key: str, value: Any) -> None:
        self.data.setdefault("prefs", {})[key] = value

    def push_unique(self, key: str, value: str, limit: int = 16) -> None:
        if not value:
            return
        seq = [x for x in (self.data.get(key) or []) if x != value]
        seq.insert(0, value)
        self.data[key] = seq[:limit]

    # --------------------------------------------------------- 模型条目
    def ensure_model(self, path: str, create: bool = True,
                     name: str = "") -> Optional[Dict[str, Any]]:
        """取（必要时新建）一个模型条目。

        ``name`` 是模型库里的显示名 —— 传了就会用来决定配置文件名。
        """
        k = model_key(path)
        if not k:
            return None
        models = self.data.setdefault("models", {})
        if k not in models:
            if not create:
                return None
            models[k] = {
                "path": os.path.abspath(
                    os.path.expanduser(str(path).strip().strip('"'))),
                "name": str(name or ""),
                "file": "",
                "category_override": "",
                "load": {}, "chat": {}, "emb": {},
                "load_presets": {}, "chat_presets": {},
                "last_load_preset": "", "last_chat_preset": "",
            }
            self._assign_file(k)
        entry = models[k]
        for key, dv in (("load", {}), ("chat", {}), ("emb", {}),
                        ("load_presets", {}), ("chat_presets", {})):
            entry.setdefault(key, copy.deepcopy(dv))
        if name and not entry.get("name"):
            # 之前不知道显示名，现在知道了 → 记下来（文件名下次保存时跟上）
            entry["name"] = str(name)
        return entry

    def set_model_name(self, path: str, name: str) -> None:
        """记住模型库里显示的名字（决定配置文件名）。"""
        if not name:
            return
        entry = self.ensure_model(path)
        if entry is not None and entry.get("name") != name:
            entry["name"] = str(name)
            self._files.pop(model_key(path), None)   # 让它重新挑文件名

    def get_model(self, path: str) -> Optional[Dict[str, Any]]:
        return self.ensure_model(path, create=False)

    def model_path_of(self, key: str) -> str:
        entry = (self.data.get("models") or {}).get(key)
        return str((entry or {}).get("path") or "")

    def configured_models(self) -> List[str]:
        """有自定义参数（加载或对话）或命名预设的模型路径。"""
        out = []
        for entry in (self.data.get("models") or {}).values():
            if entry.get("load") or entry.get("chat") or entry.get("emb") \
                    or entry.get("load_presets") or entry.get("chat_presets"):
                p = entry.get("path")
                if p:
                    out.append(p)
        return out

    def set_category_override(self, path: str, category: str) -> None:
        entry = self.ensure_model(path)
        if entry is not None:
            entry["category_override"] = category

    # --------------------------------------------------- 按模型的作用域
    def state(self, path: str, scope: str) -> Optional[Dict[str, Any]]:
        entry = self.ensure_model(path, create=False)
        if entry and entry.get(scope):
            return entry[scope]
        return None

    def snapshot_ref(self, path: str, scope: str) -> Dict[str, Any]:
        """返回可以被**就地修改**的快照字典（界面边改边预览用）。"""
        entry = self.ensure_model(path)
        if entry is None:
            return {}
        entry.setdefault(scope, {})
        return entry[scope]

    def set_state(self, path: str, scope: str, snapshot: Dict[str, Any]) -> None:
        entry = self.ensure_model(path)
        if entry is not None:
            entry[scope] = copy.deepcopy(snapshot)

    def clear_state(self, path: str, scope: str) -> None:
        entry = self.ensure_model(path, create=False)
        if entry is not None:
            entry[scope] = {}

    # ------------------------------------------------------------ 预设
    def list_presets(self, path: str, scope: str) -> List[str]:
        entry = self.ensure_model(path, create=False)
        if not entry:
            return []
        return sorted((entry.get(scope + "_presets") or {}).keys())

    def get_preset(self, path: str, scope: str,
                   name: str) -> Optional[Dict[str, Any]]:
        entry = self.ensure_model(path, create=False)
        if not entry:
            return None
        snap = (entry.get(scope + "_presets") or {}).get(name)
        return copy.deepcopy(snap) if snap else None

    def save_preset(self, path: str, scope: str, name: str,
                    snapshot: Dict[str, Any]) -> None:
        entry = self.ensure_model(path)
        if entry is None or not name:
            return
        entry.setdefault(scope + "_presets", {})[name] = copy.deepcopy(snapshot)
        entry["last_" + scope + "_preset"] = name

    def delete_preset(self, path: str, scope: str, name: str) -> None:
        entry = self.ensure_model(path, create=False)
        if not entry:
            return
        (entry.get(scope + "_presets") or {}).pop(name, None)
        if entry.get("last_" + scope + "_preset") == name:
            entry["last_" + scope + "_preset"] = ""

    def last_preset(self, path: str, scope: str) -> str:
        entry = self.ensure_model(path, create=False)
        return str((entry or {}).get("last_" + scope + "_preset") or "")

    def set_last_preset(self, path: str, scope: str, name: str) -> None:
        entry = self.ensure_model(path)
        if entry is not None:
            entry["last_" + scope + "_preset"] = name

    # ------------------------------------------------------------ 角色
    def role(self, name: str) -> Dict[str, Any]:
        return self.data.setdefault("roles", {}).setdefault(
            name, {"model": "", "server": {}})

    # 「角色默认模型」（role_model / set_role_model）已删除 —— 用户要求去掉
    # 「设置默认模型」功能。现在只有一个「目标模型」= 模型库里选中的那行，
    # 重启后从 recent_models（最近使用）恢复。旧配置里的 roles.<名>.model
    # 字段会被忽略，不影响加载。

    def role_state(self, name: str) -> Dict[str, Any]:
        return self.role(name).setdefault("server", {})

    def set_role_state(self, name: str, snapshot: Dict[str, Any]) -> None:
        self.role(name)["server"] = copy.deepcopy(snapshot)

    # ------------------------------------------------------- 运行模式状态
    def run_state(self, mode: str) -> Dict[str, Any]:
        return self.data.setdefault("run", {}).setdefault(mode, {})

    def set_run_state(self, mode: str, snapshot: Dict[str, Any]) -> None:
        self.data.setdefault("run", {})[mode] = copy.deepcopy(snapshot)

    # ------------------------------------------------------------ 控制 API
    def control(self) -> Dict[str, Any]:
        return self.data.setdefault("control_api", {
            "enabled": True, "host": "127.0.0.1", "port": 8090, "token": ""})

    # ------------------------------------------------------------ 扫描缓存
    def cache(self) -> Dict[str, Any]:
        return self.data.setdefault("model_cache", {})

    def prune_cache(self, alive_paths) -> None:
        alive = set(alive_paths)
        cache = self.cache()
        for p in [p for p in list(cache) if p not in alive]:
            cache.pop(p, None)
