# 当前项目代码中文注释规范化计划

> 制定日期：2026-09-05
> 状态：**已实施并验证（2026-09-05）：WP0–WP6 完成——24 文件中文注释改写后全仓
> 扫描清零、可执行 AST 与改写前逐位一致；`python smoke_test.py`（CPU smoke 通过）与
> `python pre_smoke_test.py`（59/59 PASS）已在本地 `diafno` 环境（torch 2.4.1+cpu）
> 复跑通过。复审后同日修复 6 处确认缺陷（`axis=` 裁定非问题），修复清单与归档产物
> 勘误见 `docs/project/CHANGELOG.md`**
> 约束来源：用户提供的仓库根目录 `项目注释要求.md`。该文件只作为规范来源，其中的
> 示例不视为本项目实施任务。
> 实施目标：在不改变运行行为的前提下，把当前项目代码中的自然语言注释和文档字符串
> 统一为规范中文，并为复杂数据结构、张量流、算法和框架行为补齐高密度说明。

## 1. 目标与硬约束

### 1.1 必须达到

1. 所有自然语言行内注释、块注释、模块/类/函数 docstring 均使用中文。
2. 注释解释代码本身不能充分表达的信息：设计意图、输入输出契约、约束、副作用、异常、
   性能原因、兼容历史和易误用行为。
3. 任何涉及数据结构的代码必须说明输入、输出和关键中间量，包括 shape、轴语义、dtype、
   device、单位/数值范围、掩膜语义，以及 view/copy/广播/连续性/梯度状态等关键变化。
4. 核心算法必须说明局部难点、关键转换和不宜随意修改的原因；不能只把代码逐句翻译成中文。
5. 注释必须与当前实际代码、实验协议和 checkpoint 兼容规则一致；不能根据名称猜测语义。
6. 实施过程保持“注释与 docstring 为主”的最小变更，不顺手重构、改名或修复算法。

### 1.2 允许保留英文的内容

以下内容不是“英文注释”违规项，但应嵌入中文语境或原样保留：

- 代码标识符、环境变量、文件名、路径、API 名、库名和模型名；
- shape 符号与公式，例如 `(B, L, 2, H, W, Z)`、`rfftn`、`einsum`、`sigma_data`；
- 论文标题、作者名、URL 和正式引用信息；
- `# noqa`、`# type: ignore`、shebang、编码声明等工具指令；
- 程序输出、异常文本和测试期望字符串——它们不是代码注释，本计划不翻译。

英文术语不得单独构成自然语言注释。例如 `# window-scoped RNG` 应改为
`# 每个窗口使用独立 RNG 状态，避免 batch 划分改变采样轨迹`。

### 1.3 明确不做

- 不修改模型结构、损失、采样器、数据切分、训练参数和评估口径；
- 不翻译变量名、函数名、类名、日志内容或 checkpoint/NPZ 字段；
- 不把架构文档全文复制到代码中；全局流程仍放文档，局部契约和陷阱放代码附近；
- 不为简单 getter、显然的赋值或无额外语义的转发函数强行添加长 docstring；
- 不保留失效的注释代码、调试 `print` 注释或无归属的 TODO；确认无用后直接删除注释代码。

## 2. 当前基线

扫描范围为仓库中的 24 个 Python 文件，不含 checkpoint、缓存和生成产物：

| 指标 | 当前值 | 说明 |
|---|---:|---|
| Python 文件 | 24 | 根目录、`scripts/`、`presentation/` |
| 代码总行数 | 约 9,633 | 含空行、注释和 docstring |
| Python 注释 token | 925 | 使用标准库 `tokenize` 统计，包含行尾注释 |
| 纯英文注释 token | 895 | 含英文自然语言，且不含中文字符 |
| 已有中文注释 token | 8 | 主要位于 `presentation/make_figures.py` |
| 可承载 docstring 的节点 | 352 | module/class/function/method，含嵌套定义 |
| 已有 docstring | 115 | 不要求剩余 237 个简单节点全部补写 |
| 英文 docstring | 114 | 需逐条翻译并校对实际契约 |
| 中文 docstring | 1 | 位于 presentation 脚本 |

注释工作量最大的文件：

| 文件 | 行数 | 注释 token | 主要风险 |
|---|---:|---:|---|
| `pre_smoke_test.py` | 2,078 | 315 | 测试数据结构多，容易把步骤说明写成流水账 |
| `pre_trainer.py` | 806 | 148 | DDP、AMP、多步反馈、resume 与 checkpoint 状态耦合 |
| `pre_evaluate.py` | 560 | 89 | rho/native 双网格、多个基线和累计器 shape 密集 |
| `pre_config.py` | 761 | 70 | 环境变量、兼容策略、进度线程和元数据约束集中 |
| `diffusion.py` | 305 | 57 | EDM 预条件、噪声 schedule、Heun 更新和广播 |
| `scripts/diag_uv_predictability.py` | 508 | 43 | 流式统计、分层/分区聚合和输出 schema |
| `pre_dataset.py` | 358 | 36 | 时间窗口、轴换位、归一化、memmap 和掩膜语义 |
| `IAFNO.py` | 346 | 33 | 频域复数运算、patch 展开/还原和时间调制 |

以上数字是实施前基线。最终验收以标准库扫描器的结果为准，不以人工抽样代替。

## 3. 模块、调用者与数据流地图

```text
原生 ROMS C-grid u/v + mask + ocean_time
        │
        ├─ scripts/preprocess_align_uv.py ──> rho 网格 memmap / 双变量 mask / 时间缓存
        │        └─ scripts/profile_preprocess_align_uv.py（只写 scratch，复用生产 kernel）
        │
        └─> pre_dataset.py ──> cond / target / native truth / mask tensor
                    │
                    ├─> pre_trainer.py <─ pre_config.py
                    │       ├─> pre_models.py ─> IAFNO.py
                    │       ├─> diffusion.py ──> IAFNO.py
                    │       ├─> pre_rollout.py（detached 多步反馈）
                    │       └─> pre_metrics.py
                    │
                    ├─> pre_evaluate.py <─ pre_config.py
                    │       ├─> pre_models.py / diffusion.py / IAFNO.py
                    │       ├─> pre_rollout.py（ensemble 自回归）
                    │       └─> pre_metrics.py（rho -> native + 累计指标）
                    │
                    └─> scripts/diag_*.py（复用正式数据、模型、rollout 和指标口径）

pre_smoke_test.py ──> 覆盖上述 PRE 共享模块及关键回归约束
smoke_test.py / trainer.py / utilities3.py ──> 原始 DiAFNO/兼容路径
presentation/make_figures.py ──> 只读归档 NPZ，生成汇报图并核对 RESULTS 数字
```

注释必须优先放在上述边界处：网格转换、数据布局转换、模型包装、反馈窗口、指标累计、
checkpoint 重建和框架状态切换。`pre_trainer.py`、`pre_evaluate.py` 与部分诊断脚本是
module-top-level 入口，不能为了复用注释检查而 import；注释中必须明确这一副作用边界。

## 4. 统一注释格式

### 4.1 shape 符号

各模块首次使用时按实际需要声明，禁止同一文件内一符多义：

| 符号 | 含义 |
|---|---|
| `B` | batch 中的窗口数 |
| `T` | 原始时间轴长度或绝对日数 |
| `K` | 训练允许采样的最大 lead |
| `L` | 当前 target/rollout 的 lead 天数 |
| `E` | ensemble member 数 |
| `C` | 通道数；必须注明是否 day-major `u/v` 交错 |
| `H, W` | rho 或原生 staggered 网格空间轴；必须注明所属网格 |
| `Z` | sigma 层数；单层 preset 为 1，full3d 为 30 |
| `N` | 有效样本/网格点计数；使用时注明聚合范围 |
| `P_h,P_w,P_z` | patch 在三个空间轴上的尺寸 |

### 4.2 模块级 docstring

核心模块统一回答四件事：

```python
"""模块职责：……

不负责：……
关键约束：……
依赖关系：……
"""
```

模块 docstring 只描述稳定职责和边界，不罗列会频繁变化的执行步骤。

### 4.3 类/函数 docstring

公共接口、核心算法、复杂数据转换和有副作用的函数使用以下字段的必要子集：

```python
"""功能：将 rho 网格预测映射回原生 staggered u/v 网格。

参数：
- rho_pred（np.ndarray）：shape 为 `(B,L,2,H,W,Z)`；通道 0/1 分别为 u/v，单位为 m/s。

返回：
- u_native：shape 为 `(B,L,H,W-1,Z)`。
- v_native：shape 为 `(B,L,H-1,W,Z)`。

关键转换：
- u 沿 xi 轴对相邻 rho 点求均值；v 沿 eta 轴处理；不做 east/north 旋转。

异常 / 前置条件：
- 输入必须为六维且变量通道数为 2，否则断言失败。
"""
```

字段无内容时省略，不保留空标题。内部简单 helper 可使用一到两句短 docstring。

### 4.4 局部注释

只在关键转换或框架陷阱附近添加，采用“动作 + 原因/后果”的陈述句：

```python
# rfftn 只压缩最后一个频率轴：Zp -> floor(Zp/2)+1；后续 mode 裁剪只作用该轴。

# broadcast_to 返回只读 view；显式复制为 C-contiguous 数组，避免 torch.from_numpy
# 共享只读内存并在原地操作时失败。

# 反馈 forward 必须关闭 autocast 权重缓存；否则 no_grad 产生的 detached fp16 权重
# 会被最终梯度 forward 复用，导致 DDP 参数 hook 不触发。
```

禁止 `# 遍历 batch`、`# 计算 loss`、`# 保存结果` 等字面复述。

### 4.5 TODO / FIXME / NOTE

- 当前扫描未发现注释形式的 TODO/FIXME；实施时不得新建裸 TODO。
- TODO 必须包含负责人或任务归属、具体动作和触发条件。
- FIXME 必须说明问题、影响范围和风险。
- NOTE 只用于稳定但易误解的约束，不能用“这里很坑”等情绪化表达。

## 5. 数据结构代码的强制说明清单

凡代码创建、读取、转换或累计以下对象，至少覆盖适用项：`torch.Tensor`、
`np.ndarray`、memmap、Dataset sample、mask、checkpoint dict、NPZ payload、统计 accumulator、
list/dict 分组结构。

1. **身份**：对象代表条件窗口、真值、预测、mask、误差和、元数据还是缓存。
2. **类型**：Python 类型、dtype；Tensor 还要说明 CPU/CUDA device。
3. **shape 与轴顺序**：逐轴给出语义，不只写“5D tensor”。
4. **数值语义**：物理单位、归一化范围、NaN/land fill、mask 中 0/1 的含义。
5. **输入输出**：调用前需要什么，返回顺序是否稳定，是否允许空值或 legacy 字段缺失。
6. **关键中间量**：只记录发生语义变化的节点，例如：
   - `u/v (T,Z,H,W*) -> rho (T,Z,H,W)`；
   - `(L,2,H,W,Z) -> (2L,H,W,Z)` day-major flatten；
   - `(B,C,H,W,Z) -> (B,Hp,Wp,Zp,E)` patch token；
   - `(B,E,L,2,H,W,Z) -> (B,L,2,H,W,Z)` ensemble mean；
   - error sum/count `(L,2,Z) -> pooled scalar RMSE`。
7. **存储与所有权**：说明 memmap 是否只读、`np.asarray` 是否物化、`expand`/`broadcast_to`
   是否为 view、`permute` 后是否 contiguous、`torch.from_numpy` 是否共享内存。
8. **梯度与随机性**：说明 `no_grad`、detach、autocast、GradScaler、per-window seed 和
   DDP rank 间是否共享状态。
9. **副作用**：文件覆盖/拒绝覆盖、cache 写入、进度线程、随机种子、全局配置和输出目录。
10. **失败语义**：shape/mask/fingerprint 不一致为何必须 fail-fast，legacy fallback 会产生
    什么告警，哪些失败可恢复。

## 6. 分阶段实施计划

每个工作包只处理列出的文件；完成后单独 review，再进入下一包。这样出现注释误解时可在
较小 diff 内定位，不把 9,000 多行代码混成一次审查。

### WP0：术语、扫描器与行为基线

文件：本计划、拟新增的 `scripts/check_comment_language.py`、现有测试入口。

- 冻结第 4 节 shape 符号和以下中文术语：条件窗口、目标、持续性基线、原生 staggered
  网格、rho 网格、双变量掩膜、逐窗口种子、分离式多步反馈、预条件网络输出。
- 用 Python 标准库 `tokenize` 扫描 comment token，用 `ast` 扫描 module/class/function
  docstring；不增加第三方依赖。
- 允许列表只覆盖 shebang、编码声明、lint/type pragma 和引用信息；“含英文技术名但无
  中文说明”的自然语言注释仍判失败。
- 保存实施前的 smoke 输出；建立去除 docstring 后的 AST 对比，确保后续包没有改变
  可执行语法树。

验收：扫描器能报告当前 895 个纯英文注释 token 和 114 个英文 docstring，并按文件定位。

### WP1：数据源、网格与指标契约

文件：

- `scripts/preprocess_align_uv.py`
- `scripts/profile_preprocess_align_uv.py`
- `pre_dataset.py`
- `pre_metrics.py`

必须补清：

- ROMS C-grid 的原生 u/v shape、rho 共定位 stencil、边界单侧处理和“不旋转”语义；
- mask 是权威有效性来源，动态缺测与静态陆地值的不同失败/丢弃策略；
- `mmap read -> H2D -> mask -> colocation -> D2H -> memmap write` 的设备和所有权变化；
- train-only stats 的流式两遍/直方图逻辑、clip/min-max/pooled sigma 的计算空间；
- `PREUVDataset` 的绝对日索引、split 不越界、day-major channel flatten 和 target lead 对齐；
- `NativeUVReader` 把 sigma 轴移到末尾后的统一布局；
- rho 到原生网格的不可逆误差、mask 后 error sum 与 pooled RMSE 的聚合范围；
- profiler 只写 scratch、CUDA 计时必须同步、生产 kernel 不重复实现的原因。

验收：从原生文件到正式 native-grid 指标的每次 shape/单位/掩膜变化都可沿注释追踪。

### WP2：IAFNO、EDM 与确定性残差模型

文件：

- `IAFNO.py`
- `diffusion.py`
- `pre_models.py`

必须补清：

- `IAFNODiff` 的 noisy target、动态条件和可选静态 mask 如何沿 channel 轴拼接；
- `PatchEmbed` 的 channel-last/channel-first 转换、3D Conv patch 化和 token 网格 shape；
- AFNO 中 `rfftn` 半谱 shape、embedding 分 block、复数权重拆为实部/虚部的 `einsum`
  规则、mode 截断、soft-shrink 和 `irfftn` 还原；
- implicit/explicit block 的残差缩放、double skip、位置 embedding 和 patch head 还原；
- 非整除空间尺寸的零信息 padding、还原后裁剪，以及 padding 轴不能写错的原因；
- EDM 的 `[0,1] -> [-1,1]`、sigma 广播、`c_skip/c_out/c_in/c_noise` 预条件、训练
  loss weighting、采样 schedule、churn、Euler/Heun 更新；
- 框架中的 `x_self_cond` 实际承载 14/16 通道外部 condition，这是历史接口兼容而非
  常规 self-conditioning，禁止按名称“修正”；
- persistence-residual 的 `prediction = last_day + residual`、zero-init head 的首步梯度
  行为、constant time embedding 和 masked MSE 分母。

验收：读者能从 `(B,C,H,W,Z)` 跟踪到频域 block、patch token、重建输出，并理解扩散与
确定性两条路径为何共享同一 backbone。

### WP3：rollout、配置、训练与正式评估

文件：

- `pre_rollout.py`
- `pre_config.py`
- `pre_trainer.py`
- `pre_evaluate.py`

必须补清：

- ensemble 的 `(B,E,L,2,H,W,Z)` 布局、member 独立状态、逐窗口 RNG 和最终均值；
- 自回归窗口“丢最旧 2 通道、追加自身预测”的结构变化，static condition 不进入滑窗；
- detached feedback 只让第 `J` 步携带梯度，前 `J-1` 步的显存、clamp 和 rf0 语义；
- autocast + no_grad 权重缓存为何会切断 DDP 梯度 hook，以及反馈阶段关闭 autocast 的边界；
- PRESET/config/checkpoint dict 的字段、legacy fallback、fingerprint、resume/weights-only init
  和 objective/horizon/static-mask 互斥规则；
- `ProgressReporter` 的 rank-0、heartbeat thread、锁、终止状态和单行可解析输出约束；
- DDP rank/world-size、DistributedSampler、`drop_last`、per-device batch、AMP skipped update、
  scheduler step、early-stop 计数和 checkpoint 原子语义；
- 评估时 checkpoint 驱动模型重建、输出拒绝覆盖、rho 预测反归一化、native truth 读取、
  model/persistence/zero/oracle 四套 `(L,2,Z)` accumulator 和 figure u/v 配对检查；
- module-top-level 入口的 import 副作用，明确禁止把 `pre_trainer.py`/`pre_evaluate.py`
  当共享库导入。

验收：单卡、DDP、恢复训练和正式评估的状态变化均有中文注释；关键 Tensor/metadata 的
输入输出契约完整。

### WP4：诊断、检查与汇报脚本

文件：

- `scripts/diag_uv_predictability.py`
- `scripts/diag_leadtime_residual.py`
- `scripts/diag_region_breakdown.py`
- `scripts/analyze_pre_dataset.py`
- `scripts/analyze_checkpoint_results.py`
- `scripts/inspect_raw_nc.py`
- `scripts/inspect_pre_dataset.py`
- `scripts/make_handoff_figures.py`
- `presentation/make_figures.py`

必须补清：

- 在线 moments/extrema、定步长 quantile sample、分 band/变量/lead 的聚合 key 和 shape；
- bias、variance ratio、spatial correlation 的统计范围，以及“先累计再 finalize”的原因；
- coastal/offshore mask 的 dilation 语义、尾部 `Z` 轴切片陷阱；
- NetCDF/xarray/NPZ 输入 schema、mmap 采样范围、输出文件和拒绝覆盖副作用；
- 汇报图从哪份归档 NPZ 取数、pooled 公式与 RESULTS 断言如何防止选错 checkpoint。

验收：每个脚本的输入数据、输出文件、只读/写入边界和统计口径在模块 docstring 中明确。

### WP5：原始 DiAFNO 兼容代码与测试

文件：

- `trainer.py`
- `utilities3.py`
- `smoke_test.py`
- `pre_smoke_test.py`

必须补清：

- `trainer.py` 是原始/legacy 训练入口，不是当前 PRE 正式入口；placeholder 数据和旧 AMP
  API 不得被注释成现行协议；
- `MatReader` 对旧 MATLAB 与 HDF5 v7.3 的轴反转、normalizer 的统计轴、Lp/Hs loss 的
  reduction 和 relative/absolute 语义；
- `torch.load(weights_only=True)` 的安全边界与 verified legacy checkpoint fallback；
- smoke fixture 的小尺寸 Tensor 每一轴代表什么、测试验证的 invariant 和失败含义；
- `pre_smoke_test.py` 按数据/指标、模型、rollout、配置、进度、DDP/AMP、checkpoint 分组；
  每个非平凡测试使用简短中文 docstring 说明“防止哪类回归”，不逐行描述 arrange/act/assert。

验收：测试注释能说明维护价值和数据结构，但不会把 2,078 行测试写成逐句中文旁白。

### WP6：全仓复核与收尾

- 运行中文注释扫描器，清零未豁免的纯英文自然语言注释/docstring；
- 人工抽查所有含 `reshape`、`rearrange`、`permute`、`transpose`、`moveaxis`、`view`、
  `expand`、`broadcast_to`、`stack`、`cat`、`einsum` 的位置；
- 人工抽查所有含 mask、memmap、checkpoint、DDP、autocast、no_grad、seed、thread/lock 的位置；
- 删除重复、复述代码或已过时的旧注释；统一中文术语和 shape 记法；
- 更新本计划状态与 Changelog，记录实际覆盖文件、扫描结果和回归测试结果。

## 7. 实施纪律与审查边界

1. 每个工作包开始前先读完整文件及其调用者，不做全局搜索替换式翻译。
2. 原注释若与代码或当前文档冲突，以可执行代码和已冻结协议为证据；仍无法确定时记录为
   待核问题，不写一个看似确定但未经验证的中文解释。
3. 注释改写发现潜在 bug 时，不在同一变更内修复；单独报告位置、证据和影响，避免行为
   修改混入注释 diff。
4. 行内 shape 注释只放在轴顺序、聚合范围或存储语义发生变化的位置；连续若干简单算子
   可用一条块前注释概括。
5. 保留作者归属和论文/参考实现信息，但将说明性前缀改为中文。
6. 现有英文 section banner 改为简短中文标题，例如 `# 模型构建`、`# 自回归评估与指标累计`；
   删除纯装饰性超长 `#####` 分隔线。
7. 所有 docstring 必须描述当前行为，不能写“未来可能”“理论上”等不可验证措辞。

## 8. 验证方案

### 8.1 每个工作包

1. `python scripts/check_comment_language.py <本包文件>`：无未豁免的英文自然语言注释。
2. 去除 docstring 后对比修改前后 `ast.dump(..., include_attributes=False)`：可执行 AST 相同。
3. `python -m py_compile <本包文件>`：语法与编码通过。
4. `git diff --check`：无行尾空格和冲突标记。
5. 人工检查 diff：中文含义准确、shape 与当前代码一致、无代码改动。

### 8.2 核心里程碑

完成 WP1–WP3 后运行：

```bash
python smoke_test.py
python pre_smoke_test.py
```

预期结果：现有 59 项 PRE smoke 全部通过；无 CUDA 的环境只允许既有 CUDA 专用分支按
原逻辑跳过。注释改写不得改变测试数量、日志协议或输出文件。

### 8.3 最终人工数据流验收

逐段核对以下四条链路：

1. 原生 C-grid → rho 共定位 → Dataset day-major condition/target；
2. condition/noisy target → IAFNO patch/frequency blocks → next-day u/v；
3. detached/ensemble rollout → 滑动窗口 → 多 lead 预测；
4. rho 预测 → native u/v → error sum/count → pooled RMSE/诊断 NPZ。

每条链路必须能只靠附近中文注释回答：输入是什么、输出是什么、shape/类型如何变化、
为什么这样变、错误修改会造成什么后果。

## 9. 完成判定

只有同时满足以下条件，才能把计划状态改为“已完成”：

- 24 个 Python 文件全部审查，未豁免的英文自然语言注释和 docstring 数量为 0；
- 所有公共接口、核心算法和有副作用入口均有中文契约说明；
- 所有关键数据结构转换都覆盖 shape、轴、dtype/device、数值/掩膜和所有权语义中的适用项；
- DDP/AMP、随机数、memmap、广播 view、checkpoint 兼容和 rho/native 转换等框架陷阱
  均有贴近代码的中文说明；
- 不存在逐句翻译代码、情绪化注释、裸 TODO、失效注释或用注释掩盖命名问题；
- AST 行为一致性、编译、smoke、Markdown/Changelog 记录全部通过；
- 注释变更按工作包完成审查，未混入模型或数据语义修改。

## 10. 建议的审查批次

建议按下列边界提交供审查；这只是未来实施时的拆分建议，本计划本身不执行提交：

1. 数据、网格与指标契约；
2. IAFNO、EDM 与 persistence-residual；
3. rollout、配置、训练和评估；
4. 诊断、检查与汇报脚本；
5. legacy 工具与全部测试；
6. 中文扫描器、全仓复核和文档收尾。

每批都应保持可独立回退、可独立验证，避免一次审查近千条注释而失去准确性。
