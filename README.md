# buun-llama-cpp 控制台

给 [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) 用的图形化启动器。
只用 Python 标准库的 tkinter，**不需要 pip 安装任何东西**；也可以打包成单文件 exe 便携运行。

![模型库](docs/preview-1-library.png)
![服务页](docs/preview-2-server.png)
![KV / VBR](docs/preview-3-kv-vbr.png)
![VBR 下限挡位](docs/preview-6-vbr-floor.png)
![运行状态条](docs/preview-7-status-bar.png)
![推测解码](docs/preview-4-spec.png)

> ⚠️ **这是 ik_llama.cpp 启动器的另一个独立副本，两份互不影响、也不共用代码。**
> 两套引擎的参数集差别很大（buun 独有的 VBR / TurboQuant / TCQ 一整套，
> ik 独有的 `-mla` / `-fmoe` / `-rtr` 那一批），所以**不要在两个项目之间互相拷
> `schema.py`** —— 拷过去会立刻出现「界面勾了、引擎不认」的启动失败。

---

## 它是什么

一个「把命令行参数做成界面 + 管住 llama-server 进程」的桌面工具：

- **模型库**：扫描 GGUF、按目录结构认发布者和模型名、读元数据算真实参数量、
  标出能力（多模态 / 工具 / 思考 / 草稿），选中即可加载。
- **服务**：OpenAI 兼容的 HTTP 服务。对外只有**一个端口**，
  控制 API、`/v1/*`、内置 WebUI 全在这里。
- **对话 / 生成**：终端里跑 `llama-cli`（多轮对话）或 `llama-completion`（一次性生成）。
- **参数页**：加载参数、对话参数、推测解码（三者都**跟模型走**）；
  服务页与两个运行模式各自一套运行参数。

## 运行

```bat
:: 源码运行（需要带 tkinter 的 Python；本机是 E:\Python\Python311）
start_gui.bat               :: 正常启动
start_gui_debug.bat         :: 带控制台，方便看报错

:: 自检（不开窗口，跑一遍核心逻辑，末行 OK）
E:\Python\Python311\python.exe buun_launcher.pyw --selftest

:: 打包成便携 exe（产物在 dist\）
E:\Python\Python311\python.exe build_portable.py
```

> 解释器必须是 `E:\Python\Python311\python.exe`（3.11.8，自带 tkinter 8.6）。
> 托管 Python 3.13 没有 tkinter，跑不了本程序。

配置放在程序目录下的 `config/`——**便携**：不写注册表、不碰 `%APPDATA%`。
目录不可写时才回退到 `%LOCALAPPDATA%\buun_llama_gui`，原因会写进启动日志。

### config 目录长这样

```
config/
  app.json                 软件级设置：引擎路径、模型目录、界面偏好、控制 API、运行参数
  models/
    Huihui-Qwen3.8-27B-abliterated.json     每个「配过参数的模型」一个文件
    Qwen3-30B-A3B-Instruct-IQ4_KSS.json
  cache/
    scan.json              模型库扫描缓存（GGUF 元数据，可随时删掉重建）
  router-preset.ini        多模型路由的预置文件（用「导出路由预置 INI」生成）
  selftest.log             自检输出
```

- **文件名 = 模型库里的显示名**（非法字符替换成 `-`，重名自动加 `-2`）。
  文件内的 `path` 才是权威标识，所以**给模型改名不会丢设置** —— 下次保存时
  文件名会跟着改过去。
- 单个模型的文件通常几 KB，好备份、好分享、好手动改；某个模型调崩了删它那一个就行。
- 老的单文件 `config.json` 会在首次启动时**自动拆分**，原件改名成
  `config.legacy.json` 留退路（内容一字不改，想回滚直接改回 `config.json`）。
  拆分结果会写进启动日志。

---

## 两种运行方式（服务页）

| | 单模型网关（默认） | 多模型路由 |
|---|---|---|
| llama-server | 只加载一个模型、绑内部端口 | 带 `--models-dir` / `--models-preset`，自己驻留多个 |
| 换模型 | 本程序做网关：抢锁 → 卸旧的 → 装新的 → 等就绪 | llama-server 自己按 `model` 名路由 |
| 网关的角色 | 模型名翻译 + 自动切换 + 控制 API | **纯透传**（不再插手换模型） |
| 适合 | 「主要用一个模型，偶尔换」 | 「一次会话里要切好几个模型」 |

两种方式互斥：路由模式下「按请求自动加载 / 切换」会被忽略，界面里也会说明。

### 路由模式的模型来源：优先用预置文件

**推荐 `--models-preset`，不要用 `--models-dir`。** 后者的扫描规则很窄
（见 `common/preset.cpp` 的 `load_from_models_dir`）：

- 只认**顶层散放的 `.gguf`**，或**一层子目录**（`<目录>/<模型名>/*.gguf`，目录名 = 模型名）；
- 像 `E:\LM_models\<发布者>\<模型名>\x.gguf` 这种**两层结构，它一个都扫不到**。

而预置 INI 里 `model = <任意深度的路径>`，段名就是客户端请求里要写的模型名，
还能顺手把「每个模型自己的加载参数」写进去。

服务页右上角有 **「导出路由预置 INI」** 按钮，生成的内容长这样：

```ini
; 段名 = 请求里 model 字段要写的名字
[*]                                  ← 全局段，对所有模型生效
ctx-size = 32768
n-gpu-layers = 999
cache-type-k = vbr

[Qwen3.8-27B-Uncensored-Heretic-v3]  ← 段名就是模型 id
model = E:\LM_models\...\xxx.gguf
ctx-size = 100000
cache-type-k = turbo3_tcq
```

键写去掉 `--` 的长参数名（也认 `LLAMA_ARG_XXX` 环境变量名）。导出后把路径填进
「路由预置文件」，启动即可。

> **重要**：路由模式下命令行上的参数**不会传给各个模型**（子进程的 args 是从预置
> 生成的，只有 `[*]` 全局段会合并进去）。所以模型参数必须写进预置文件 ——
> 导出功能就是干这个的。

---

## buun 这套引擎特有的地方

### KV 缓存档位（buun 的立身之本）

`-ctk` / `-ctv` / `-ct` 的档位比上游多一批：

| 类别 | 档位 | 说明 |
|---|---|---|
| 常规 | `f32` `f16` `bf16` `q8_0` `q5_1` `q5_0` `q4_1` `q4_0` `iq4_nl` | 和上游一样 |
| Turbo | `turbo8` `turbo4` `turbo3` | fork 自带的 TurboQuant KV 编解码 |
| TCQ | `turbo3_tcq` `turbo2_tcq` `turbo1_tcq` | 需要码本文件（见下） |
| 动态 | `vbr` | 按显存压力逐 (层, 侧) 降级，**也是本 build 的默认** |

**三个必须知道的坑：**

1. **默认值就是 `vbr`**，不是 `f16`。界面上「统一 KV 档位」显示 `vbr` 但没勾选时，
   就是引擎的默认行为（隐式 t4 地板）。
2. **turbo / TCQ / VBR 的 KV block 是 128 个值**，要求模型 `n_embd_head_k`
   能被 128 整除，并且必须开 `-fa`。
   - `Qwen3` 系、`Gemma4`、多数 7B+ 模型 head_dim 是 128 / 256 → 没问题；
   - `bert` 系（如 bge-small，head_dim=64）和部分小模型 → 会直接报
     `K cache type turbo4 with block size 128 does not divide n_embd_head_k=64`；
   - **而且因为默认就是 vbr，什么都不设也会踩到。** 启动器会读 GGUF 里的
     head_dim 提前拦住，并告诉你把 K/V 两侧显式设成 `q8_0` 之类。
3. **TCQ 档位需要码本**：`codebooks/3bit/cb_50iter_finetuned.bin` 与
   `codebooks/2bit/tcq_2bit_100iter_s99.bin`，通过环境变量传给子进程：

   ```
   TURBO_TCQ_CB  = <引擎目录>\codebooks\3bit\cb_50iter_finetuned.bin
   TURBO_TCQ_CB2 = <引擎目录>\codebooks\2bit\tcq_2bit_100iter_s99.bin
   ```

   启动器会自动推导引擎目录、找到码本并注入（你自己设过同名变量则不覆盖）。
   仓库里缺文件时会写警告，此时 `turbo*_tcq` 档位不可用。

### VBR 动态量化（加载参数页 → VBR 动态量化）

只有把 KV 档位选成 `vbr` 时这一节才真正起作用。降级阶梯：

```
f16 → turbo8 → turbo4 → turbo3_tcq → turbo2_tcq → turbo1_tcq
```

**「最低档位下限」（`--vbr-floor`）做成了挡位按钮**：
`跟随引擎 / turbo8 / turbo4（默认）/ turbo3_tcq / turbo2_tcq / turbo1_tcq`。
默认 `turbo4` —— 比引擎在「显式 `-ct vbr`」时的隐式下限（`turbo1_tcq` = 1.25 bpv）
保守得多。

各档的**真实比特单价**（本机逐档实测，从 `/slots` 的 `kv_bpv` 读回来的，
含量化块的 scale 开销，不是整数）：

| 档位 | bpv | | 档位 | bpv |
|---|---|---|---|---|
| f16 / bf16 | 16.0 | | turbo4 | 4.125 |
| q8_0 | 8.5 | | turbo3 | 3.5 |
| turbo8 | 8.125 | | turbo3_tcq | 3.25 |
| q5_1 | 6.0 | | turbo2_tcq | 2.25 |
| q5_0 | 5.5 | | **turbo1_tcq** | **1.25** |
| q4_1 | 5.0 | | q4_0 / iq4_nl | 4.5 |

其他可调项：`--vbr-codec`（auto/turbo/classic）、`--vbr-entry`（起始档）、
`--vbr-vram`（显存预算）、以及 server 专属的 `--vbr-prompt-cache` /
`--vbr-anchor-cache-mib` / `--vbr-reclaim-floor` / `--vbr-reset-keep-frac`。

**三条会挡住启动的约束**（启动器都会提前校验并说明）：

1. 下限**不能高于**起始档位 —— 引擎直接报错
   `--vbr-floor (4.125 bits/value) cannot exceed --vbr-entry t2 (2.25 bits/value)`。
2. `--vbr-codec classic` 时 turbo 档位非法（classic 阶梯只认 `q8_0` / `q4_0`）。
3. 下限低于阶梯最低点时引擎**只警告并抬到最低点**，不报错。

> ⚠️ **KV 档位不是 `vbr` 时，这一整组参数会被自动丢弃。**
> 引擎对「非 vbr 的 KV + `--vbr-*`」是硬报错
> （`--vbr-* flags need a VBR cache side`），而这组参数默认就带着
> `--vbr-floor turbo4`，所以启动器加了一道闸：只有 KV 确实是 vbr 才输出它们，
> 被忽略时会在日志里说明原因。

### 上下文长度：留空 = 引擎自动算

「上下文长度」（`-c`）**默认留空**，意思是交给引擎自己算：

1. 起点是**模型的训练上下文**（本机多数模型是 262144）；
2. 按**可用显存**（扣掉模型与显存余量）往下缩；
3. KV 成本按 **「最低档位下限」那一档的单价**计价；
4. 缩到刚好装下为止，下限是「自动时的下限」（`--fit-ctx`，默认 4096）。

**实测（27B，可用显存 4875 MiB）** —— 这条链把「下限」和「能开多长」绑在一起：

| `--vbr-floor` | 引擎自动算出的 n_ctx |
|---|---|
| `turbo1_tcq`（1.25 bpv） | **262144**（撞模型上限） |
| `turbo4`（4.125） | **262144**（容量约 31 万，仍撞上限） |
| `turbo8`（8.125） | **156672** |
| `f16`（16） | **80640** |

所以想要长上下文，就把下限压狠一点；想要 KV 精度，就得接受上下文变短。

相关的两个参数：**显存保留余量**（`--fit-target`，默认 1024 MiB，余量越大
留给 KV 的越少 → 自动算出的上下文越短）和**自动时的下限**（`--fit-ctx`）。

> 常见误解：**`--vbr-vram` 不参与上下文长度计算**。实测给它 16 MiB
> （远低于全上下文的成本）时 n_ctx 一点没变，引擎只警告一句
> `the KV budget (16 MiB) is below the full-context cost ...`。
> 它只驱动**运行时降级控制器**。

### 实时运行状态（底部状态条）

启动后会常驻在命令行/日志上方，数据全部来自引擎自己暴露的接口，
**不需要加任何引擎参数**：

| 显示 | 来源 |
|---|---|
| 生成 / 提示处理速度 | 每次转发响应体里的 `timings.predicted_per_second` / `prompt_per_second`（**默认就带**） |
| 当前 KV 档位 | `/slots[].kv_bpv` 与 `/props.vbr` 反查（没降级时按 `--vbr-floor` 单价显示） |
| 上下文长度与已用比例 | `/props.default_generation_settings.n_ctx`、`/slots[].n_prompt_tokens` |
| VBR 状态 | `/props.vbr` → `codec` / `entry_type_k\|v` / `floor_bpv` / `capacity_floor_bpv` / `realized_bpv` / `vram_budget_bytes` |
| 缓存命中 | `timings.cache_n` |

例：`运行状态：生成 365 t/s · 提示 399 t/s · KV vbr·当前 f16 · 未降级（下限 4.12 bpv） · 上下文 60 / 16384 · 空闲`

> 降级发生后 `realized_bpv` 才有值，状态条会改成「已降级」。
> 混合状态（例如 4.25 bpv = t4 布局夹几个单元高一档）显示成 `≈turbo4（4.25 bpv）`。

### 多模态投影：按需进显存

`--mmproj-gpu-swap` 让投影文件**先加载在内存（CPU），只有真的来了带图片的请求
才换进显存**，媒体请求处理完再换回内存 —— 省显存，代价是首次看图多一次搬运。
启动日志会打 `loaded multimodal model on CPU (GPU swap enabled)`。

配套还有两项：**投影 GPU 卸载**（`--no-mmproj-offload` = 投影永不进显存，
每次看图都在 CPU 算）和**投影所在设备**（`-mmdev`，填 `none` 等价于不卸载）。

> 要腾显存时会先把可重载的投机上下文（MTP / 外部 DFlash）换出去，用完恢复。
> 若草稿模型是**不可重载**的类型，这个功能不可用，引擎只会打一行警告并让两者都常驻：
> `mmproj GPU swap is unavailable for this external draft type; keeping both resident`。
> 只有 `llama-server` 有这个参数。

### 推测解码：和 ik 完全不是一套接口

- `--spec-type` 收的是**逗号分隔的类型名列表**，可以同时挂多种；
  **不支持** ik 那种 `--spec-type dflash:n_max=4` 的内联写法。
  认这些名字：`draft-simple` `draft-eagle3` `draft-mtp` `draft-dflash`
  `draft-dspark` `dflash` `ngram-simple` `ngram-map-k` `ngram-map-k4v`
  `ngram-mod` `ngram-cache` `suffix` `copyspec` `recycle`
  （`draft`=`draft-simple`、`mtp`=`draft-mtp` 是等价别名）。
- 所有细项都是**独立 flag**：`--spec-draft-n-max` / `--spec-draft-n-min` /
  `--spec-draft-p-min` / `--spec-draft-p-split` / `--spec-draft-temp`；
  n-gram 参数按算法分别命名（`--spec-ngram-simple-size-n` 等）。
  旧的 `--spec-ngram-size-n` / `-size-m` / `-min-hits` 在本 build 已被上游删除。
- `draft-mtp` **不需要**草稿模型（用主模型自带的 MTP 层，没有就只打一行警告跳过）；
  `draft-simple` / `draft-eagle3` / `draft-dflash` / `draft-dspark` / `dflash`
  需要 `-md`。启动器按这个规则提前校验。
- 推测解码只有 `llama-server` / `llama-cli` 支持，`llama-completion` 一个都不认。

### 推理强度是原生参数

本 build 有 `--reasoning-effort LEVEL`（`minimal`/`low`/`medium`/`high`/`xhigh`/`max`），
不需要 ik 那边「塞进 `--chat-template-kwargs` 走模板变量」的做法。
请求体里的顶层 `reasoning_effort` 引擎也原生支持，网关原样透传。

---

## 两个启动器的差异（为什么参数表不能共用）

| | ik_llama.cpp | buun-llama-cpp |
|---|---|---|
| `llama-server` flag 数 | 406（**和 cli 完全相同**） | 461 |
| `llama-cli` flag 数 | 406 | 375（少了整批推测解码 / 多模态） |
| `llama-completion` | 无 | 272，**它有 `-cnv` / `-i`，而 cli 没有** |
| 本启动器的参数项 | 96 | 119 |

buun 独有的：VBR 全组、turbo/TCQ 档位、`-ct` 统一档位、`--models-dir` 路由组、
`--repack`、`--agent` / `--tools` / `--mcp-servers-*`、`--rerank` / `--embeddings`、
`--mmproj-auto`，以及上游新加的 `--spec-ngram-*`、CPU 亲和组等。

ik 独有（本启动器里已全部删除）：`-mla` `-fmoe` `-gr` `-sas` `-mqkv` `-ser`
`-vq` `-khad` `-vhad` `-dkvc` `-rtr` `-amb` `--fit-margin` `--chunks`
`--minilog` `-crs` `-ptcall` `--spec-autotune` `--spec-ckpt-mode`
`--mtp-requantize-output-tensor` `--draft-params` `-dr`(=dry-run)。

> 例外一：`-rtr`（run-time repack）在本 build 里对应 `--repack` / `--no-repack`，
> 语义相同，这一项保留了下来（界面叫「权重重排」）。
>
> 例外二（陷阱）：`-dr` 在 ik 里是 `--dry-run`，在 buun 里却是 `--docker-repo`
> （Docker Hub 模型仓库，还要跟一个值）。所以「只算内存不加载」这一项**删掉了** ——
> 留着就是一个会报错的坑。自检里有专门一条断言在守这类「同名不同义」。

---

## 代码结构

```
buun_launcher.pyw          入口（双击 / --selftest）
buunllama_gui/
  schema.py                参数单一事实来源：122 个参数项、7 个页面、3 个运行模式
  engine.py                引擎层：TCQ 码本推导与环境注入、CUDA 运行时依赖体检
  builder.py               参数 → 命令行、校验（VBR/turbo 约束）、preset INI、.bat
  probe.py                 跑 --help 解析当前 build 认哪些 flag
  gguf.py                  GGUF 只读解析（含 head_dim，用于 turbo 档位前置校验）
  runtime.py               运行时状态：轮询 /props + /slots，反查 KV 档位与已用比例
  scan.py                  模型库扫描与分类
  manager.py / process.py  进程生死 + 命令总线（不跨线程碰 tkinter）
  unified.py               单模型网关的调度（抢锁 / 换模型 / 等就绪 / 空闲卸载）
  control_api.py           控制 API + /v1 网关（路由模式下退化为纯透传；
                           转发时顺手抓 timings 喂给运行时状态）
  ui.py / widgets.py       界面（含底部运行状态条）
  store.py / theme.py      配置持久化（app.json + 每模型一份）/ 配色
```

### 三条铁律（改代码前先读）

1. **任何启动路径都必须走 `App.build_argv_checked()`** —— 它会按 `--help` 结果
   剔除当前 build 不认识的 flag。自检里有源码级断言守着这件事。
2. **界面专用字段绝不能产出裸值**：没有 `flag`、没有 `argmap`、
   `positional=False` 的字段一律 `return []`，否则「运行方式」这种值会被
   当成位置参数塞进命令行。
3. **删参数 / 挡位改名后要处理存档里的旧值**：`builder.valid_choice()` 负责把
   老名字退回默认挡位；`multi_value` 的字段（`--spec-type`）例外 ——
   逗号列表整体必然不在 `choices` 里，一律原样保留，由 `validate` 逐个名字报错。

## 验证

```bat
E:\Python\Python311\python.exe buun_launcher.pyw --selftest
```

末行必须是 `OK`。自检覆盖：7 个页面与 121 行参数、3 套运行模式的命令行、
模型扫描与分类、**参数 flag 与真实 exe 全量对齐**、**短参数语义对齐**、
推测解码（类型列表 / 独立 flag / 草稿模型校验）、**turbo 128-block 兼容性**、
**VBR 下限挡位与三条约束**、**上下文「留空=自动 / 填值=指定」**、
**运行时状态（轮询桩后端 + 转发抓 timings + bpv 反查档位）**、
**多模态投影三项的运行模式归属**、**路由模式与 preset INI 键名校验**、
**配置拆分（app.json + 每模型一份 + 缓存分文件 + 选择性写入 + 老配置迁移）**、
LoRA、预设、`--help` 解析、统一端口网关端到端
（短名翻译 / 自动切换 / 409 / 并发排队 / 真起进程 / 鉴权）。

自检用临时配置目录，不碰真实 `config/`；找不到 buun 引擎时会跳过
「与真实 exe 对齐」那两条，并在日志里写明。

## 打包后的验收

把 `dist\buun_llama_gui_debug.exe` **拷到一个干净空目录**再跑：

```bat
buun_llama_gui_debug.exe --selftest
```

断言三件事：退出码 0、末行 `OK`、**该目录下出现 `config\`** —— 这才算便携成立。
