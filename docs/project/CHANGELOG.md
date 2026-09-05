# 项目 Changelog

本文件记录本地适配仓库中与 PRE 模型、训练和评估有关的实际变更与已确认计划。

- **已完成**：代码或文档已经存在于当前工作区，并有相应验证或状态证据；
- **计划中**：只完成设计，尚未修改代码或运行实验；
- 每次实施后将条目从“计划中”迁移到对应日期，并补充验证命令与结果；
- 未执行、失败或缺少产物的实验不会记为完成。

## Unreleased — 计划中

### Proposed

- full3d 重启前置项（Path B，待独立预算落实后执行）：单步峰值显存（22.6 GB）与
  逐 epoch 评估成本（val h15 ≈2 h05m）压缩方案；per-band 归一化复核（底层归一化
  std ≈ 海面 1/3，见实验 06 RESULTS）。
- 分支再评估触发项：若未来工作专攻 u 分量 d15 rebound（surface u 0.906）并产生
  新的 u/v 不对称证据，loss weighting 可按新预注册重开（方向文档 §6）。
- 服务器端归档重生成：含 `norm_p1_p99_width` 键的诊断归档 NPZ/CSV/MD（60/60 为
  负值符号错误）需重跑 `scripts/diag_uv_predictability.py` 重生成（修复见
  2026-09-05 复审修复条目）。

## 2026-09-05 — 已完成（代码中文注释规范化 WP0–WP5 实施）

- **WP0**：新增 `scripts/check_comment_language.py`（仅标准库 tokenize/ast；缺省扫描
  根目录/scripts/presentation 共 24 个业务文件，豁免 shebang/编码声明/pragma/URL 引用/
  注释掉的可执行代码）。基线核对：注释 token 925（与计划一致）、英文 docstring 114
  （一致）；895 个英文自然语言注释中 20 个为豁免项（13 shebang、4 编码声明、1 noqa、
  1 URL、1 URL 引用），875 个为待清零违规。另建仓库外 AST 对比工具（去除 docstring 后
  对比 git HEAD 的 `ast.dump`）作为行为基线。
- **WP1–WP5**：24 个 Python 文件全部完成注释/docstring 中文规范化（仅注释与 docstring，
  零代码改动）。按计划 4/5/7 节执行：模块 docstring 四字段（职责/不负责/关键约束/
  依赖关系）、公共接口契约 docstring、数据结构强制清单（shape 逐轴/dtype/device/
  掩膜语义/所有权/梯度与随机性/副作用/失败语义）、统一中文术语与 shape 记号、
  装饰性分隔线改中文短标题、删除 `# print(emb.shape)` 等 1 处调试注释。
  diffusion.py 被 AGENTS.md 保护的自条件禁用注释代码块原样保留。
- **验证**：全仓 24 文件 + 扫描器 `python scripts/check_comment_language.py` →
  `TOTAL: english_comments=0 english_docstrings=0`；30 个被跟踪 .py 全部
  `AST-STRIPPED IDENTICAL`（与 HEAD 对比）；`python -m py_compile` 全部通过；
  `git diff --check` 干净；diff 通读抽查（IAFNO padding、diffusion mask 契约、
  pre_trainer autocast 反馈帧、pre_dataset 切分/交错通道）与代码及 AGENTS.md 一致。
- **遗留（原待服务器端，已闭环）**：计划 8.2 的 `python smoke_test.py` /
  `python pre_smoke_test.py` 已在本地 `diafno` 环境（torch 2.4.1+cpu）复跑，
  分别为 CPU smoke passed 与 59/59 PASS（见下一条目）。
- **实施中发现并仅报告、未修复的疑似问题**（详见各文件注释）：IAFNO.py `ex_layer!=1 & nlayer==1`
  链式比较优先级陷阱（当前配置休眠）；diffusion.py `sample_using_dpmpp` 初始 shape
  缺 Z 维且无调用方；scripts/diag_uv_predictability.py `norm_p1_p99_width` 值为
  负（p1-p99，键名语义相反）；utilities3.py `MatReader` 裸 `except:` 与
  LpLoss.abs 的 h 因子假设；trainer.py `torch.amin/amax(axis=)` 关键字与
  `count` 先赋值再被 200 覆盖的死赋值；scripts/profile_preprocess_align_uv.py
  输出 shape 硬编码常量（部分网格 chunk 时才暴露）。
  → **同日复审后 6 处确认缺陷已全部修复，见下一条目；`axis=` 复审裁定不构成
  问题（torch 2.4.1 实测可用），仅保留跨版本兼容注释。**

## 2026-09-05 — 已完成（注释规范化复审修复：6 处确认缺陷 + 本地 smoke 闭环）

- **复审裁定**：注释实施报告的 7 处疑似问题中 6 处确认存在并修复；`trainer.py`
  的 `amin/amax(axis=)` 在 torch 2.4.1（本仓锁定版本，+cpu/+cu124）实测可用，
  不记为故障，仅落"跨版本若报 TypeError 改回 dim="的注释。真正污染既有产物的
  只有诊断宽度符号问题，其余为闲置/legacy/特定输入触发。
- **修复清单（均同步中文注释）**：
  1. `scripts/diag_uv_predictability.py`：`norm_p1_p99_width` 改为
     `(p99-p1)/(hi-lo)`，键回归正的归一化 p1–p99 宽度。**产物影响**：已归档
     NPZ/CSV/MD 中该键 60/60 为负值（-0.3273..-0.0695），幅值仍是真宽度；
     归档产物按勘误纪律不复用，需在服务器重跑该脚本重生成。
  2. `diffusion.py sample_using_dpmpp`：初始噪声补 z 轴为 `(b,c,h,w,z)`，并把
     `self_cond` 逐步透传给网络（旧实现漏传）。小网格 CPU 验证：输出 shape 与
     Heun `sample()` 一致，条件变化输出随之变化。仍无生产调用方（正式评估走 Heun）。
  3. `IAFNO.py forward_features`：`ex_layer!=1 & nlayer==1` 链式比较修为
     `(ex_layer!=1) and (nlayer==1)`；真值表确认行为差异仅限"ex_layer!=1 且
     nlayer 为奇数>1"（如 (2,3)/(4,3)），全部既有配置（nlayer=2/4、冒烟 1/1）
     新旧同支，行为不变。
  4. `utilities3.py MatReader._load_file`：裸 `except:` 收窄为
     `except NotImplementedError`（scipy 判定 v7.3 的唯一信号）；文件缺失现在
     原样抛 `FileNotFoundError`，权限/损坏/中断不再被吞。
  5. `utilities3.py LpLoss.abs`：h 因子改按展平后总点数 N 取 `1/(N-1)`，并改用
     `reshape`（非连续输入安全）；实测 (1,6)、(1,2,3)、(1,3,2) 三种布局结果一致
     （0.145558），转置不再翻转。仓库无 `abs()` 调用方（只走 `rel()`）。
  6. `trainer.py`：删除 `count = data.shape[1]` 死赋值（随即被 `count=200`
     覆盖、从未读取）；`axis=` 注释记录实测结论。
  7. `scripts/profile_preprocess_align_uv.py`：`colocate_u/colocate_v` 输出
     shape 从输入推导（末/倒数第二轴 +1），scratch memmap shape 与读写字节
     统计同步改为按实际数组；main() 的完整网格断言保留（当前支持路径自洽）。
- **验证**：仓库外 scratch 脚本 5 组针对性检查全部通过（IAFNO 真值表 /
  dpmpp 5D+条件消费 / LpLoss.abs 转置不变 / MatReader 错误传播 / profiler
  全网格等价+部分网格）；全仓扫描器 `TOTAL: english_comments=0
  english_docstrings=0`；与 HEAD 的 AST 对比——6 个修复文件按预期不同，其余
  24 文件逐位一致；`python smoke_test.py`（CPU smoke passed）与
  `python pre_smoke_test.py`（59/59 PASS）在本地 `diafno` 环境
  （torch 2.4.1+cpu）全部通过——计划 8.2 里程碑闭环。

## 2026-09-05 — 已完成（入口文档同步与实验 11 Ep4 正式产物归档）

- 同步项目交接概要、文档/实验索引、runbook、实验 06 状态和 `AGENTS.md`：middle
  正式 Ep4 test 0.851、gate 5 边缘缺陷已接受；full3d 已选 Path B；六分支均 No-Go，
  当前无待执行实验。
- 从权威运行目录归档 experiment 11 middle Ep4 正式 test NPZ、12 张 figures、val
  结构诊断 NPZ/PNG 及两份日志到对应 `checkpoints/PRE/` 路径；五个单文件的本地
  SHA-256 与服务器端一致，两份日志均以 `status=completed` 结束。
- 将仓库根 `README.md` 全文改写为中文，并把 PRE 当前状态更新到 2026-09-05 的
  已裁定结论；命令、路径、变量名、数学口径和论文引用保持不变。

## 2026-09-04 — 已完成（分支准入评估：六分支全部不满足 + full3d Path B 定案）

- **评估方法（零 GPU/零训练/零新评估）**：只读归档 NPZ（surface MS10 Ep2、单步
  Ep10、middle MS5 Ep4、bottom MS5 Ep5 的 test eval NPZ 与 val `leadtime_diag`
  NPZ），scratch heredoc 复算判据；单步 Ep10 旧格式 diag NPZ 按 2026-09-02
  勘误纪律不复用，引实验 07 RESULTS 归档数字。
- **关键证据**：u/v 不对称已被 MS 大幅消解——单步 Ep10 test overall u/v
  1.014/1.031（长 lead v 更差：d14 1.188 vs 1.145）→ MS 后 surface 0.842/0.824、
  middle 0.851/0.850、bottom 0.811/0.820（gap ≤0.018，逐 lead |v−u| ≤0.058 且
  三层方向不一致）；detached 已稳定越过长 lead persistence（三层 test 全 15 天
  ratio <1，最差 0.906/0.979/0.880 均在 d15；val 无 crossover，corr 除 middle
  d15 边缘 −0.002/−0.006 外全 lead 占优）；残余缺陷两变量共有（val var_ratio@d15
  surface u 0.337/v 0.425、middle 0.262/0.372、bottom 0.305/0.371；bias 最大为
  middle u −0.050@d15，相对 persistence 误差 0.218 仍小）。
- **判定**：§6 六分支（loss weighting / direct multi-horizon head / TBPTT / 额外
  输入 / residual diffusion / full BPTT）全部**不满足准入条件**，无新立项；
  loss weighting 留"u d15 rebound 新证据"再评估触发条件。详见方向文档 §6。
- **full3d Path B 定案（用户决策）**：冻结 full3d 待独立正式预算（≈5 天/50 epoch、
  峰值 22.6 GB）；重启前置项 = 显存/评估成本压缩方案 + per-band 归一化复核
  （方向文档 §5）。当前无待执行实验。
- 文档同步：方向文档头部（三项待办闭环）/§5（已决策）/§6（已评估判定表）重写；
  本文件 Unreleased Proposed 改为 full3d 重启前置项与分支再评估触发项。

## 2026-09-04 — 已完成（实验 11 middle 勘误修正执行：正式 Ep4 test + 结构诊断）

- **正式 Ep4 test（h15、stride 7、154 窗、seed 123、churn 0、rf0、batch 4，
  单卡 RTX 4090 GPU 5，rollout 247 s，`status=completed`，无异常/NaN/OOM）**：
  `pre_evaluate.py` 零修改（scratch driver 内存补丁常量）。结果：day-1 ratio
  0.665（0.0483 vs 0.0727 m/s）、15-day overall **0.851**（0.1149 vs 0.1351；
  u 0.851 / v 0.850 各自 < 1.0）、d10–15 每日 0.862/0.881/0.883/0.916/0.956/0.978
  全部 < 1.0、全程无 crossover（最差 lead 0.978 @ d15）——test 门槛全部通过。
  正式选型 Ep4 的 test（0.851）略差于探索性 Ep2（0.830），与 val overall 方向一致
  （0.820 vs 0.814），属合规选型下的真实结果。产物：
  `eval_test_h15_ch0_e1_s123_rf0_ckptEp4_test15.npz`、`eval_midms5_test15_ep4.log`、
  `figures_h15_ch0_e1_s123_rf0_ckptEp4_test15/`。
- **正式 Ep4 val 结构诊断**（`diag_mid_ms5_ep4_val.log`、
  `leadtime_diag_ckptEp4.{npz,png}`，与 Ep2 同协议 stride 14 / 77 窗 / seed 123）：
  无 crossover、pooled ratio 单调 0.62→0.93；corr 在 lead 1–14 对 u/v 均占优，但
  **d15 边缘低于 persistence（u 0.428 vs 0.430、v 0.417 vs 0.423）**——gate 5
  按预注册字面未过，middle 不记"全门槛 Go"，裁定转入方向文档决策项。
  另：u bias −0.050@d15（Ep2 为 +0.043，符号相反）；var_ratio u 0.262@d15。
- 文档同步：实验 11 `RESULTS.md`（勘误修正执行节 + 选型/门槛/对比/诊断表 + 结论）、
  `EXPERIMENT.md` 状态、方向文档头部/§2/§3.5/§4。
- **gate-5 裁定（同日）**：接受边缘结构缺陷——middle 记"gate 1–4 + test 全过、
  gate 5 边缘未过"，不通过事后放宽容差改写预注册门槛；不改选 Ep5（预注册选型
  规则未定义 fallback）、不回退 Ep2（day-1 未过门槛）；既存事实转入 full3d/
  分支决策（方向文档 §4）。

## 2026-09-04 — 已完成（S1–S3 修复包：静态 mask 评估、绘图、文档同步、Windows 回归、early-stop 计数）

- **`pre_evaluate.py` 支持静态 mask checkpoint（修复：原实现固定 14 条件通道，
  无法加载 `_MSK` checkpoint）**：按 checkpoint `config.static_mask_input`/
  `model_cond_chans` 经新 `pre_config.static_mask_from_checkpoint` 重建（legacy 缺
  字段 → 14 通道；扩散+静态 mask、通道数矛盾 → 拒绝）；静态 checkpoint 构造
  `static_cond`（双变量 rho mask）并传入每个 rollout step；输出 tag 增加 `msk1`、
  NPZ 记录 `static_mask_input`/`model_cond_chans`。验收：归档实验 08
  `..._RES_MSK/Ep10.pth` 重建成功并完成最小 CPU rollout（新增
  `test_archived_msk_checkpoint_minimal_cpu_rollout`）。
- **`pre_evaluate.py` 绘图修复**：`tight_layout/savefig/close` 原在 u/v 循环外
  （u 图从未写出、figure 泄漏），移回循环内；新增成对校验——每个选定
  lead/layer 必须同时产出 `_u.png` 与 `_v.png`，缺失或不成对直接 RuntimeError。
- **`pre_smoke_test.py` Windows 修复**：`test_dataset_horizon5_split_and_alignment`
  持有 `PREUVDataset` 的 `numpy.memmap` 句柄导致临时目录清理报 WinError 32；
  新增测试内 `_close_mmaps` 辅助（不改生产数据集生命周期）。本机直跑 59 项
  全 PASS、exit 0（56 项存量 + 3 项新增）。
- **early-stop 计数随 checkpoint 保存/恢复**：`pre_trainer.py` checkpoint 新增
  `worse_epochs`，resume 经 `pre_config.restore_worse_epochs` 恢复（legacy 缺字段
  默认 0）；新增 `test_worse_epochs_checkpoint_roundtrip`。一次 worsening 后保存
  再恢复，下一次 worsening 按原语义在 2 次时停止。
- **实验 11 口径勘误（S2-6，不改写已执行决定）**：middle probe day-1 原记录
  0.754/0.803/0.770 在当前仓库可用的归档 NPZ/日志中均不可复现（真值
  0.569/0.645/**0.582**，
  RMSE 0.0517）；据此 middle MS5 的 day-1 门槛应为 RMSE ≤0.05269（原记录误用
  ratio 0.785）——复核后仅 Ep4/Ep5 过门槛，预注册规则会选 Ep4（val h15 overall
  0.8202）而非已执行的 Ep2（0.0551 未过门槛）；bottom 行复算无误（0.550/0.624/
  0.568）。RESULTS.md 表格全部改为可从 NPZ 独立复算的 overall (u/v) 值，
  `d10–15 max` 拆为独立列；新增"勘误与影响"节；EXPERIMENT.md/交接概要/实验索引/
  AGENTS.md 同步标注。**影响**：按原预注册规则正式改选 Ep4 并补 test；已执行的
  Ep2 test 仅保留为探索性结果；MS5 长时效修复效果本身不受影响。
- **checkpoint 指向警示（S2-7）**：MS10 run 目录的 `best.pth` 按训练期
  `val_masked_relL2` 对应 **Ep3**，正式选型是 **Ep2**——在实验 10 RESULTS、
  Runbook §4、交接概要、实验索引统一醒目标注；正式评估一律显式用
  `MS10/Ep2.pth`（`best.pth` 保留，不重写历史 checkpoint）。
- **方向文档重写（S1-3）**：`CURRENT_CHALLENGES_AND_NEXT_STEPS.md` 只保留任务语义、
  当前证据（链接各 RESULTS.md，不复制结果表）、未解决问题、middle Ep4 正式补测及
  两条待决策（full3d K3/预算 A/B/C、§6 分支准入）与执行约定；实施前算法/文件计划/旧运行入口
  原文归档至 `archive/MULTISTEP_PLAN_20260901.md`（带历史状态说明横幅）；
  全文不再含"multi-step 尚不可执行 / DDP2 未做 / full3d 尚未运行"等过期表述。
- **Runbook 更新（S1-4）**：补 detached multi-step 操作协议（`DIAFNO_TRAIN_HORIZON`、
  `DIAFNO_INIT_CHECKPOINT`、`MS_DEFAULTS`、`_MS{K}` run tag、resume guard、smoke
  的 `max_lead_seen` 门槛、DDP autocast 约束与示例命令）；§4 选型协议改写为
  "单步 = day-1 RMSE；multi-step = day-1 守门 + validation 15-day overall"，并固定
  当前正式模型 = `MS10/Ep2.pth`；§1 文件清单预设数改四套、pre_trainer/pre_config
  描述补 MS 能力；§5 full3d 补实测资源与 K3 阻塞状态及 middle/bottom preset 状态；
  §4 输出 tag 补 `[_msk1]` 与静态 mask 自动重建说明。
- **元数据漂移（S3-9）**：`docs/README.md`、`docs/experiments/README.md`、
  交接概要索引日期更新至 2026-09-04；交接概要测试计数改 59 项；full3d 单样本
  ≈296 MB 明确为 condition-only（完整 K1 样本约 340 MB）；runbook "两个 preset"
  改四套；归档计划中 `55/55` 等表述保留原文（历史记录），现状以本条目为准。
- **文档残项复核**：h1 窗口数改为 NPZ 元数据对应的 156（h15 仍为 154）；实验 06
  修正 condition 标签与 middle Ep10 ratio；失效的工作包引用改指向历史实施计划，
  现行分支引用统一为 §6；涉及未传输服务器产物的断言限定为当前仓库可用归档范围。
- **验证**：`python -m py_compile`（pre_config/pre_trainer/pre_evaluate/pre_smoke_test）
  通过；`python smoke_test.py` exit 0；`python pre_smoke_test.py` 59 项全 PASS、
  exit 0（Windows 本机，CPU 环境）。

## 2026-09-04 — 已完成（full3d 大权重分卷归档）

- full3d 4 个 ~342 MB 权重（RES best/Ep1、SMOKE best/Ep1）超过 GitHub 100 MB 单文件
  硬限制、LFS 免费额度也不足，改用 `split -b 90M -d` 切成 16 个 <100 MB 分卷归档至
  `checkpoints/PRE/full3d_pth_split/`（分两批 commit 推送）；该目录 `README.md`
  记录原件 md5 与 `cat ... >` 重组命令。
- 完整性验证：4 组分卷 `cat` 重组后 md5 与磁盘原件逐项一致（管道校验，不落盘）。
- 权威原件仍在服务器磁盘 `~/checkpoints/PRE/<run_dir>/`，训练/评估直接用原件，
  不在仓库目录内重组；`torch.load` 前须先重组并过 md5。
- `AGENTS.md` 归档例外同步更新：>100 MB 文件由"绝不入库"改为"按分卷约定入库"。

## 2026-09-03 — 已完成（文档同步：交接概要与归档约定成文）

- `PROJECT_HANDOFF_SUMMARY.md` 更新至 2026-09-03 现状：当前最优改为 MS10 Ep2
  （test overall ratio 0.838，crossover 消除），实验表补 10/11 行、06 行改为
  "部分执行"；困难与下一步改写为 full3d K3/预算决策与 §10 分支准入；已解决项
  （exposure bias、垂向证据、资源实测、NPZ key 缺陷）相应移出。
- `AGENTS.md` Conventions 把实验归档例外成文：`checkpoints/` 默认不提交，但工作包
  结束时以 `git add -f` 专门 `归档…` commit 入库（先例 7cf959e / 04ef0f0 /
  78b7a66..cbc5c61）；硬规则 = >100MB 文件绝不入库（full3d 342MB `.pth` 仅留磁盘）、
  不重新加入先前清理 commit 有意移除的文件；并注明仓库内 `checkpoints/PRE/` 是磁盘
  `~/checkpoints/PRE/` 的拷贝快照而非符号链接。

## 2026-09-03 — 已完成（工作包 5：代表层 + 工作包 6 第 1–4 步）

实验 11（`docs/experiments/11_representative_layers/`）与实验 06 恢复，详细数字见
各自 RESULTS.md。

- **代码**：`pre_config.py` 新增 `middle_smoke`（depth_index=14）/`bottom_smoke`
  （depth_index=0）preset（架构/预算与 surface_smoke 全同，单变量 = depth）；
  `pre_smoke_test.py` 新增 `test_representative_layer_presets` → 56/56。
  trainer/evaluate/dataset 的 depth_index 链路零改动。
- **代表层单步 probe**（并行 2 卡，10 epochs）：middle day-1 ratio **0.770** ✅、
  bottom **0.568** ✅（均过"day-1 < 层 persistence"门槛）；但两层单步均有长时效
  失效（middle test overall 1.183、crossover d5；bottom 0.930、v d15 1.15）。
- **代表层 MS5**（各层 probe Ep10 weights-only 初始化，5 epochs）：**全门槛 Go**——
  middle Ep2（test overall **0.830**，单步 1.183→修复 crossover）、bottom Ep5
  （test overall **0.813**，v d15 1.15→0.880 修复）。垂向泛化成立，难度排序与
  WP1 画像一致。
- **full3d（实验 06）**：资源 probe 实测（单步 0.97 s/步、峰值 22.6 GB、
  1 epoch ≈ 2.3 h → 50 epoch ≈ 5 天）；K1 smoke PASS；1-epoch single-step pilot
  训练健康（2h08m、22.2 GB、无 OOM/skip）但**逐层 day-1 信号未出现**（60 个
  ratio 全部 ≈1.000）→ **K3 按预注册条件阻塞**，候选路径（加 epochs/冻结/调参）
  留待决策；`EPOCH_OVERRIDES` 临时设 1 已还原 `{}`。
- 执行方式：tmux 值守 + 监控 subagent；`pre_evaluate.py`/诊断脚本零修改（scratch
  驱动器补 PRESET/BATCH_SIZE patch）。

## 2026-09-03 — 已完成（DDP2 smoke：根因定位与修复后通过）

- **根因（首次尝试 3 连败的 "Expected to have finished reduction in the prior
  iteration"）**：detached 反馈 forward 原本在 autocast 上下文内执行。autocast 会把
  Linear 族权重缓存为 fp16 副本；no_grad 下的 forward 使缓存中的 fp16 权重
  **detached**（与 fp32 参数断开），同批次的最终带梯度 forward 复用缓存后，损失图
  与这些参数断连、DDP hook 永不触发——恰好解释缺失梯度的 27 个参数
  （patch_embed/mlp/time_mlp/head/up/downproj）而 AFNO 频域层（einsum/FFT 走 fp32、
  不进缓存）完好。
- **修复**：`pre_trainer.py` 的 `lead>1` 分支用嵌套
  `autocast(device_type, enabled=False)` 包裹 `detached_feedback_window`——反馈
  推理以 fp32 执行（数值上严格更接近确定性 rollout 的底层数学），不污染 autocast
  权重缓存；`pre_rollout.detached_feedback_window` docstring 记录该 AUTOCAST
  CAVEAT。`pre_smoke_test.py` 55/55 保持通过（CPU 路径不触发该问题）。
- **验证**：tmux 值守脚本（卡况轮询 + 自动执行）在 GPU 0/3 上
  **SMOKE PASS**（4 updates/rank、无 AMP skip、`max_lead=3` 证明 J>1 批次在 DDP
  下真实执行、checkpoint 齐全）；有效 batch=8。日志
  `checkpoints/PRE/train_ms5_smoke_ddp2_20260903_003529.log`。
- 注：per-rank 峰值显存 ~17.5-20 GB，DDP smoke 需要两块各 ≥19.5 GB 空闲的卡
  （值守脚本按此门槛自动选卡）。

## 2026-09-02 — 已完成（工作包 3+4：MS5/MS10 训练、选型与 test）

实验 10（`docs/experiments/10_multistep_deterministic/`）两条臂全部完成，单变量 =
训练 horizon；详细数字见该实验 `RESULTS.md`，此处只记代码/流程事实。

- MS5 smoke：`DIAFNO_TRAIN_HORIZON=5` + `DIAFNO_INIT_CHECKPOINT`（实验 07 Ep10）
  单卡 real-data smoke → **SMOKE PASS**（4 updates、无 AMP skip、`max_lead=3`
  证明 J>1 批次真实执行、weights-only init 生效）。
- MS5 短训：5 epochs（MS 默认 lr 1e-4），~30 min/epoch，峰值显存 20.1 GB 平稳；
  逐 epoch val 15-day 选型（day-1 守门 + overall 排名）→ **Ep4**；结构诊断
  crossover 无、corr 全 lead 占优；**test 一次**：overall ratio **0.871**
  （单步基线 1.018），day-1 0.843，最差 lead 0.963。**Go**。
- MS10 短训：`EPOCH_OVERRIDES` 临时设 3（已还原 `{}`），K=10、从 MS5 Ep4
  weights-only 初始化，~38 min/epoch；选型 → **Ep2**（与 Ep3 overall 差 0.06%，
  按规则取更低者）；结构诊断晚段 bias 稳定（+0.017 vs MS5 的 -0.071）；**test
  一次**：overall ratio **0.838**，day-1 0.833，最差 lead 0.894。
- 评估执行方式：`pre_evaluate.py` 零修改（scratch 驱动器内存补丁常量，评估 tag
  带 `OUTPUT_TAG` 隔离 val/test 的 figures 目录）；`scripts/diag_leadtime_residual.py`
  通过 import + 属性覆写调用（该脚本 WP2 已重构为 `main()` 守卫结构）。
- 遗留：DDP2 smoke 未做（无双卡同时空闲）；方差塌缩与 d15 ratio 回升保留为
  §10 分支的指向证据。

## 2026-09-01 — 已完成（工作包 2：detached multi-step 代码与 CPU 回归）

实现 `docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md` §5 的 detached
autoregressive multi-step；`python pre_smoke_test.py` 55/55 全过（47 项存量 +
9 项新增，其中 1 项为 NPZ 修复回归）。

- `pre_config.py`：新增 `DIAFNO_TRAIN_HORIZON`（`train_horizon()`，默认 1 = 历史单步）、
  `DIAFNO_INIT_CHECKPOINT`（`init_checkpoint()`，weights-only 初始化源）、纯函数
  lead schedule `lead_for_batch`/`lead_schedule_str`（MS5 = `1,2,1,3,1,4,1,5`，
  50% day-1 anchor，无 RNG，DDP 各 rank 一致）、multi-step 冻结默认
  `MS_DEFAULTS`（lr 1e-4 / 5 epochs，仅 K>1 生效，smoke 仍 1 epoch）、resume 守卫
  `check_multistep_config`（拒绝跨 horizon/schedule 恢复；legacy checkpoint 仅可
  K=1）；`run_tag_for`/`training_run_tag` 支持 `_MS{K}` 后缀（K=1 不变）。
- `pre_rollout.py`：新增训练侧 `detached_feedback_window(step_fn, cond, lead)`——
  J-1 步 `no_grad` 自回归回灌（clamp [0,1]、rf0、滑窗丢最旧一天与正式 rollout
  逐位一致），返回最终 condition 窗口；J=1 原样返回。
- `pre_trainer.py`：K>1 仅允许 `persistence_residual` 且无静态 mask（显式拒绝）；
  `DIAFNO_CHECKPOINT` 与 `DIAFNO_INIT_CHECKPOINT` 互斥；train dataset
  `horizon=K`（val 保持单步）；训练循环按 batch schedule 选 lead J，前 J-1 步
  detached 反馈、只对第 J 步反传（K=1 路径逐位保持历史代码）；smoke 门禁新增
  "实际执行过 J>1 batch" 检查；checkpoint config 记录 `train_horizon`/
  `lead_schedule`/`feedback_detach`/`init_checkpoint`/`init_weights_only`；
  weights-only init 只载模型权重并校验 objective/preset/mask/time_sigma/
  归一化指纹，optimizer/scheduler/scaler/历史全部全新。
- `pre_dataset.py`：`PREUVDataset.__init__` docstring 补 multi-step 用法与
  `target[:, J-1]` 索引约定（horizon 能力与不跨 split 保证本已存在，无功能改动）。
- `pre_smoke_test.py` 新增测试：schedule 模式/分布/DDP 一致性与环境变量解析、
  MS 默认超参与 `_MS{K}` tag 隔离、K=1 与历史单步逐位一致、detached 反馈的
  调用计数/无梯度图/滑窗对齐、未训练模型 multi-step 恒等于 persistence、
  dataset horizon=5 不跨 split 且 target 对齐、checkpoint 元数据 roundtrip 与
  resume 守卫、`build_npz_payload` key 无碰撞。
- 验证：`python pre_smoke_test.py` → `pre_smoke_test passed`（55 项）。
  real-data smoke 属工作包 3，尚未运行。

## 2026-09-01 — 已完成（工作包 1：全层零训练画像 + NPZ 修复）

- 新增只读诊断脚本 `scripts/diag_uv_predictability.py`（无模型/无 GPU，mmap 分块
  流式）：train 逐层×逐变量精确 mean/std/min/max/有效计数/越界计数 + stride 子采样
  分位数（p0.1/p1/p50/p99/p99.9）、train 一日增量统计、validation day 1–15
  persistence RMSE/MAE（协议一致的窗口集 s∈[val_lo, val_hi−22]，rho 网格物理
  单位，注明非正式 native 协议）、coastal/offshore（陆地 5 格内，与
  `diag_region_breakdown.py` 同口径）与 bottom/middle/upper band 聚合、统一
  min-max 归一化的逐层压缩度量。门禁四项：时间连续性（复用
  `verify_daily_time`）、mask 形状/版本、掩码内 finite、逐层有效计数。
- 运行结果（产物：`/data2/user/zyq/checkpoints/PRE/diag_uv_predictability_20260901/`，
  耗时 6285 s）：**门禁四项全 PASS（OVERALL PASS）**——0 动态缺失格、逐层有效
  计数 u≥134,921,520 / v≥134,964,225。关键数字：val persistence d1 RMSE
  u bottom/middle/upper = 0.068/0.105/0.137 m/s（d15 = 0.130/0.204/0.281），
  v = 0.039/0.054/0.087（d15 = 0.066/0.087/0.149）——surface（k=29）仍是各
  lead 最难层，底层最易；统一 min-max 无截断（clip_frac=0），底层归一化 std
  仅为海面的约 1/3（u L0 0.022 vs L29 0.067）。
- 修复 `scripts/diag_leadtime_residual.py` NPZ key 覆盖缺陷（历史
  `f"{field}_{var}"` key 使 persistence 数组覆盖 model）：key 改为
  `f"{m|p}_{field}_{var}"`，抽出可单测的 `build_npz_payload`，脚本改为
  `main()` 守卫结构（逻辑不变）；修复前的归档 PNG/终端统计仍有效，NPZ 不可复用。

以上条目均**已实施并验证**。

## 2026-09-01 — 已完成（文档职责重构）

- 将 `PROJECT_HANDOFF_SUMMARY.md` 重写为面向新成员/agent 的精简项目概要，只保留项目
  目标、当前证据、困难、下一步摘要和接手入口。
- 将原 multi-step 计划改名并重构为 `CURRENT_CHALLENGES_AND_NEXT_STEPS.md`，集中维护
  当前困难、工作包、准入门槛、待办和事后回顾表；移除活跃项目文档中的 Phase 编号。
- 将原实验 07 中追加的两项 A/B 拆成实验 08（静态 mask 输入）和实验 09（remask
  feedback），使一个实验目录只回答一个问题；实验 07 恢复为确定性基线本身。
- 从实验 07 `RESULTS.md` 移除静态 mask 代码实现与 smoke 回归清单；代码/文档实现结果
  继续以本 Changelog 为唯一事实源，实验结果只保留运行事实、科学指标、问题与分析。
- 同步文档/实验索引、runbook 和 `AGENTS.md`，并明确实验文档的职责与更新规则。

## 2026-09-01 — 已完成（确定性 multi-step 方向初版）

- 新增现名为 `docs/project/CURRENT_CHALLENGES_AND_NEXT_STEPS.md` 的方向文档：明确首要科学
  目标是确定性 u/v 点预测，训练仍为自回归，但首轮仅对选定 lead 的最后一步反传；
  固定 MS5 lead schedule，保留 50% day-1 anchor，不首轮启用 full BPTT。
- 冻结 validation 准入门槛：day-1 ≤0.1031 m/s、15-day overall ratio <0.941、
  u/v 各自 <1、day 10–15 各自 <1；test 仅在配置冻结后运行一次。
- 将全 30 层零训练画像、代表层实验和 full3d 资源/K1/K3 probe 纳入同一决策链；
  residual diffusion、TBPTT、新变量和 loss weighting 保留为条件分支。
- 同步 `PROJECT_HANDOFF_SUMMARY.md`、文档/实验索引、实验结果页与 `AGENTS.md`；
  所有新增开关、训练命令和实验目录均明确标记为计划状态，未声称已经可运行。

## 2026-09-01 — 已完成（文档归档与交接同步）

- `docs/project/CODE_MODIFICATION_PLAN.md` 归档为
  `docs/project/archive/CODE_MODIFICATION_PLAN_20260830.md`（补执行完毕状态头；
  后续方向另立文档，现名为 `CURRENT_CHALLENGES_AND_NEXT_STEPS.md`）。
- `PROJECT_HANDOFF_SUMMARY.md` 更新至 2026-09-01 现状：一句话结论改为
  persistence-residual 基线 Go；§5 mask 输入建议改写为 A/B 结论（不保留）；
  §6/§7 并入基线成绩与 day-2 7.7% 声明更正；新增第 9 节（基线/诊断/Phase 5
  决策/Phase 6 含义）；未完成项清单 8 条逐项标注完成状态。
- `docs/README.md`、`docs/experiments/README.md`（实验 07 状态与决策树）、
  `docs/operations/PRE_runbook.md`（`DIAFNO_STATIC_MASK`/`_MSK` 约定与两个
  诊断脚本）、`AGENTS.md`（static mask 约定）同步。

## 2026-09-01 — 已完成（Phase 5② remask_feedback A/B，评估-only）

- 同一 checkpoint（A 臂 Ep10）在 validation 15 天确定性 rollout 下对比
  rf0（历史整帧回灌）与 rf1（每步预测重应用海洋 mask 后回灌）；
  `pre_evaluate.py` 原生支持，统一 `OUTPUT_TAG="rfab"` 避免与 test 图目录冲突；
  两臂 exit=0，产物 `eval_val_h15_..._rf{0,1}_..._rfab.npz`。
- **决策：默认维持 rf0（历史行为）**——rf1 分段效应：day 2-8 改善
  （-0.5%~-7.9%），day 9-15 转差（+1.5%~+7.8%），overall 持平略差
  （0.2183 vs 0.2180）；不满足"稳定改善才保留"。
- **HANDOFF 未完成项 5 复现结论**：远端"day-2 改善约 7.7%"未在 day-2 复现
  （实际 -0.49%）；同量级改善实际位于 day 4-7（-5.9%~-7.9%），方向一致、
  数值归属更正（详见实验 09 RESULTS.md）。

## 2026-08-31 — 已完成（Phase 5① 双静态 mask 输入 A/B）

### Static mask input support（arm B 代码路径）

- `pre_config.py`：`DIAFNO_STATIC_MASK` 开关（`static_mask_input()`）、
  `STATIC_MASK_CHANNELS=2`、run tag `_MSK` 后缀（B 臂绝不与 A 臂共用目录）。
- `pre_models.py`：`PersistenceResidualIAFNO.forward/sample` 增加可选
  `static_cond`——仅拼入 backbone 的 x_self_cond；DYNAMIC 窗口保持纯 14 通道，
  persistence base 语义不变；批次广播与形状/通道错误显式拒绝。
- `pre_rollout.py`：`ensemble_rollout/_rollout_one/_sample` 增加 `static_cond`
  透传（None 时逐位保持历史行为，EDM 调用签名不受影响；滑窗切片不变）。
- `pre_trainer.py`：objective 守卫（static mask 仅限 persistence_residual）、
  `MODEL_COND_CH` 16 通道建模、零初始化 identity 探针含静态通道、train/val
  传入 `static_cond`、checkpoint 记录 `static_mask_input`/`model_cond_chans`、
  resume 结构守卫拒绝跨配置续训。
- `pre_evaluate.py`：按 checkpoint `config.static_mask_input` 自动重建（元数据
  驱动，不读环境变量）、输出 tag `msk{0|1}`、npz 元数据记录该字段。
- `pre_smoke_test.py` 新增 2 项测试（wrapper 静态拼接/恒等/错误拒绝、rollout
  static_cond 透传与滑窗纯度），47 项全部 PASS；legacy `smoke_test.py` 通过。

### A/B 执行与决策

- B 臂 smoke `SMOKE PASS`；10/10 epochs 训练 3 h 36 min（best 0.40038@ep10，
  全程单调改善）；validation day-1 选型 10 个 checkpoint 全部 exit=0。
- **决策：不保留静态 mask 输入**——A 最优 0.1011 < B 最优 0.1024，10 epoch 中
  9 个 A 领先；区域分解（coastal/offshore × u/v）4 项全部 A 优。B 臂产物保留
  于 `..._RES_MSK/` 供复核。
- 新增 `scripts/diag_region_breakdown.py`（coastal = 距陆地 ≤5 格的区域分解
  评估；coastal 改善幅度小于 offshore 的观察记录在案）。

### 边界

- 未执行 Phase 5②（remask A/B）；未修改 `IAFNO.py`/`diffusion.py`/
  `pre_dataset.py`/`pre_metrics.py`；`pre_evaluate.py` 选型用临时改动已恢复，
  工作区余 Phase 5① 代码路径、2 个诊断脚本与文档更新。

## 2026-08-31 — 已完成（persistence-residual 真实数据 smoke、短训练与 Phase 3 Go）

### Real-data smoke（Phase 2）

- 单卡（GPU 1，默认 smoke 模式，`DIAFNO_OBJECTIVE=persistence_residual`）：`SMOKE PASS`；
  4 updates/rank、无 AMP skip、零初始化 identity 自检通过；产物
  `surface_smoke_..._S4_C7_SD2_RES_SMOKE/`。
- DDP world size 2（GPU 1+2，`torchrun --standalone`）：`SMOKE PASS`；每 rank 4 updates、
  skipped 0；进度行 `scope=rank0_shard_of_2`、仅 rank 0 写 checkpoint；产物
  `..._RES_SMOKE_DDP2/`。smoke 末行 `lr=0` 为 cosine T_max=4 退火到底的设计行为。

### Phase 3 surface 短训练

- 单卡 10 epochs 跑满（未触发 early stop），3 h 35 min，~1.63 step/s；
  `val_masked_relL2` 从 0.58275 单调降至 0.40325（仅 ep4 一次波动），
  checkpoint 落盘 `surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/`（Ep1~Ep10 + best + loss.dat）。

### Validation day-1 选型（Phase 3 Go/No-Go）

- 逐个 `Ep{n}.pth`（禁用 `best.pth`）以 `SPLIT="val"`、`ROLLOUT_DAYS=1` 运行
  `pre_evaluate.py`（156 窗口，确定性评估），10 轮全部 exit=0，产物
  `eval_val_h1_ch0_e1_s123_rf0_ckptEp{n}.npz`。
- **Go**：Ep10 validation day-1 native RMSE `0.1011 m/s` 严格优于 persistence
  `0.1294 m/s`（ratio 0.781），并优于 ridge probe 参考 `0.1177 m/s`；
  Ep2 起所有 epoch ratio < 1 且随训练单调下降。
- 实验记录已补录：`docs/experiments/07_residual_baseline/{EXPERIMENT,RESULTS}.md`。

### Phase 4 test 报告（同日完成）

- 配置冻结：Ep10（validation day-1 选出）、`SPLIT="test"`、`ROLLOUT_DAYS=15`、
  154 窗口（stride 7）、确定性评估；单次运行 `status=completed`（~6.7 min，零异常）。
- **test day-1 `0.0973 m/s` 优于 persistence `0.1167`（ratio 0.833，Phase 4 第一目标达成）**；
  15-day overall `0.2136` vs persistence `0.2098`（ratio 1.018，基本持平，长时效自回归
  误差累积仍未解决）。
- 对照扩散：SD2 diffusion test d1 `0.2568` / overall `0.3442`；本基线分别改善约
  2.6 倍 / 1.6 倍。产物 `eval_test_h15_ch0_e1_s123_rf0_ckptEp10.npz` + figures。
- 备注（透明记录）：test 评估进程未设 `CUDA_VISIBLE_DEVICES`，落在 GPU 0 与他人任务
  共存（~1.3G 显存，确定性评估数值不受影响）。

### 长时效误差诊断（同日追加）

- 新增 `scripts/diag_leadtime_residual.py`：复用官方 rollout 协议在 77 个
  stride-14 test 窗口上重放 Ep10 的 15 天确定性 rollout，补齐评估 NPZ 不含的
  逐 lead day bias / 方差比 / 逐窗口空间相关；产物
  `leadtime_diag_ckptEp10.npz/.png`（run 目录）。
- 结论：**方差塌缩主导**（u 方差比 d1 0.87 → d7 起 ~0.55，模糊化）；空间相关
  d7 起低于 persistence（0.48 vs 0.57，d15 0.39 vs 0.61）；bias 漂移且变号
  （u: -0.005 → -0.11 → +0.065）。d1-3 为优势期（ratio 0.85-0.93）。
- 含义：为“恢复方差”的后续假设提供直接证据；当前方向优先检验训练—rollout 暴露偏差。
  长时效主诊断见实验 07，后续 mask/remask 对照分别见实验 08、09。

### 边界

- 未执行 Phase 5 A/B 与 full3d；未声称 15-day overall 优于 persistence。
- 评估用的 `pre_evaluate.py` 常量改动（CHECKPOINT/SPLIT/ROLLOUT_DAYS）已全部恢复，
  工作区仅余文档更新与新增诊断脚本。

## 2026-08-30 — 已完成（persistence-residual 基线代码实施）

### Training / models

- 新增 `pre_models.py`：`PersistenceResidualIAFNO`（condition-only 确定性基线，
  预测 = 条件第 7 天 persistence + 零初始化残差头输出；未训练时严格等于 persistence）
  与 `masked_mse_loss`（逐样本有效格点均值，与 EDM masked loss 语义一致）。
- `pre_config.py`：objective 配置（`OBJECTIVES`/`validate_objective`/`objective_from_checkpoint`/
  `ensure_objective_compatible`、`MASK_SCHEME`、`RESIDUAL_TIME_SIGMA`）；`run_tag_for`/
  `training_run_tag` 支持 objective（persistence_residual 追加 `_RES`，绝不与扩散实验共用目录）；
  新增共享进度辅助 `ProgressReporter`/`format_progress`（交互 tqdm 条；非交互 ≥30s 一条可解析
  `PROGRESS key=value` 状态行，start/completed/failed 必发）。
- `pre_trainer.py`：`DIAFNO_OBJECTIVE` 选择训练目标（默认 `diffusion`，行为不变）；
  residual 走 `masked_mse_loss`；启动时零初始化 == persistence 严格自检；checkpoint config
  记录 `objective/cond_chans/target_ch/mask_scheme`（扩散另存 sigma 字段，residual 另存
  `residual_base/time_sigma`）；断点续训校验 objective 与关键结构参数（跨模型类/结构变化拒绝）；
  sigma 尺度决策仅适用于扩散；rank-0 每 epoch train/val 进度条 + run 级
  start/completed/failed 状态行；保留全部关键 epoch summary 与 `SMOKE PASS` 文本。
- `pre_rollout.py`：`remask_feedback`/`ocean_mask` 可选开关（启用时每步预测重应用海洋 mask
  后再回灌；默认 False 保持历史行为）；docstring 明确模型 duck-type 对确定性模型的兼容
  （无 RNG 消耗、seed 无关、成员相同）。
- `pre_evaluate.py`：按 checkpoint `config.objective` 重建 diffusion 或确定性模型（legacy →
  diffusion 并提示）；确定性评估强制 `ENSEMBLE_SIZE=1`、采样参数在 `sampler`/`sampler_note`
  中显式记为不适用（`sigma_data=nan`、`sampling_steps=-1`）；`REMASK_FEEDBACK` 配置 + `rf{0|1}`
  输出 tag + metadata（`objective/residual_base/remask_feedback/sampler/sampler_note/time_sigma`）；
  评估进度条（running day-1 RMSE/比值 postfix）与 failed/completed 状态行。

### Tests / verification

- `pre_smoke_test.py` 新增 9 项 CPU 回归测试：零初始化 persistence identity（shape/通道序/
  clamp/忽略采样步数）、一次 optimizer step（head 移动、全部参数有 `.grad`——DDP 兼容性质）、
  masked MSE 语义（参考实现、陆地不变性、全零 mask→0、广播）、checkpoint roundtrip +
  objective 守卫、rollout remask 开/关行为与 mask 必填、deterministic rollout（seed 无关、
  成员相同）、objective 配置辅助、run tag objective 后缀、`ProgressReporter` 行格式与间隔门控。
- 验证命令与结果（本地 `diafno` env，torch 2.4.1+cpu，无 CUDA）：
  - `python -m py_compile pre_models.py pre_config.py pre_rollout.py pre_trainer.py pre_evaluate.py pre_smoke_test.py` 通过；
  - `python pre_smoke_test.py` 通过：41 项测试函数全部 PASS（含 9 项新增；4 项含 CUDA 专属分支
    本机按设计 SKIP；既有 32 项结果不变，legacy diffusion 覆盖未回归）；
  - `python smoke_test.py`（legacy 路径）通过。

### Documentation

- `docs/operations/PRE_runbook.md`：文件表新增 `pre_models.py`；objective/`_RES`/元数据/续训守卫、
  `REMASK_FEEDBACK`+`rf` tag、validation day-1 native RMSE 选型协议、新增第 7 节终端进度与监控约定。
- 新增 `docs/experiments/07_residual_baseline/`（EXPERIMENT = 设计与门槛；RESULTS = 未执行，
  仅记录代码实施与验证状态）；实验索引与本索引条目同步更新。
- `AGENTS.md` 同步 PRE 路径约定（objective、`pre_models.py`、remask、`_RES`、进度约定）。

### 边界

- 本轮未运行任何真实数据训练/评估；未声称任何模型精度改善；Go/No-Go 数值留待实验执行后记录。
- 未修改 `IAFNO.py`/`diffusion.py`/`pre_dataset.py`/`pre_metrics.py`/`utilities3.py`。

### Code review fixes（评审跟进，2026-08-30 同日）

针对首轮实现评审的 3 项 P1 与进度计数 P2，按最小化原则修复（未引入新监控框架）：

- **心跳线程化（P1）**：`ProgressReporter` 的周期状态行改为**时间驱动**——非交互模式下由守护
  心跳线程按间隔补发 `status=running`，单个 batch/rollout 步阻塞超过间隔不再静默；所有发射
  经 `threading.Lock` 串行化，`close()` 停止线程。
- **生命周期语义（P1）**：引入稳定状态词汇表——`start`/`running`/`phase_done`（本阶段结束，
  reporter `close()` 的新默认）/`failed`；`completed` 只由入口脚本在全部产物落盘后输出
  （评估端移到 NPZ + 汇总 + 全部图之后），每个 epoch 不再输出误导性的 `completed`；
  新增 `install_progress_failure_hook`（`sys.excepthook` 兜底）为初始化/数据/模型/pre-flight/
  后处理等不受 guarded 块保护的异常输出标准 `status=failed`（`stage=setup|run|data_model|
  rollout|postprocess` 标明位置），与 guarded handler 通过 `mark_progress_failed()` 去重；
  `_progress_value` 把一切空白（含多行异常的换行/制表符）替换为 `_`，状态行永不被错误信息打断。
- **checkpoint 语义指纹（P1）**：新 checkpoint 记录 `norm_lo`/`norm_hi`/`mask_version`；
  续训与评估重建经 `check_norm_fingerprint` 校验归一化范围与 mask 版本（不一致拒绝，legacy 缺字段
  告警），residual 另经 `check_residual_time_sigma` 校验 `time_sigma`（缺失/不一致拒绝），
  residual 续训还校验 `stats_sigma`（无迁移策略）。
- **进度计数（P2）**：评估 reporter 改为按**真实窗口数**计数（total=`len(eval_ds)`、每 batch 按
  实际窗口数推进、`sample_per_s`=窗口×`ROLLOUT_DAYS`），不足 batch 的尾批吞吐不再失真；
  DDP 下 train/val 进度行显式标注 `scope=rank0_shard_of_<n>`（单卡 `whole_split`），分片计数
  不再被误读为全局。
- **文档状态（P3）**：`CODE_MODIFICATION_PLAN.md` 状态头更新为"首轮代码已实施"；runbook 第 7 节
  重写状态词汇表/心跳/兜底 hook/scope 约定，第 1 节与第 4 节补充指纹校验与 `best.pth` 选型警告
  （`best.pth` 按 `val_masked_relL2` 产生，禁止直接用于 day-1 native RMSE 选型）。
- `pre_smoke_test.py` 新增 4 项测试：时间驱动心跳（零 update 仍发射、close 后停止）、多行错误
  清洗、失败 hook 去重与 stage 读取、归一化/mask/time_sigma 指纹校验。
- 验证：`py_compile` 通过；`pre_smoke_test.py` 45 项测试函数全部 PASS（新增 4 项；4 项含 CUDA
  分支本机按设计 SKIP）；`smoke_test.py` 通过；`git diff --check` 干净。
- 未处理（评审 P2，按最小化原则留待后续）：coastal/open-ocean 与空间相关性指标、
  validation 选型流程自动化。

## 2026-08-30 — 已完成

### Documentation

- 新增 `docs/project/CODE_MODIFICATION_PLAN.md`，固化下一轮代码修改、烟测、单卡/DDP 和全量训练门槛。
- 新增本 changelog，并加入项目文档索引。
- 本次更新未修改 Python 源码、训练配置或模型参数。

### Repository state

- 本地 `adapt-weather-ocean` 已安全 fast-forward 到远端 `43f9813`，当前与 `origin/adapt-weather-ocean` 为 `0 ahead / 0 behind`。
- 保留 fast-forward 前的保险 stash：`codex-safe-ff-origin-adapt-weather-ocean-20260830`；尚未删除。
- 远端归档文档已按现有 `docs/` 分类结构整理；项目仍保留未提交的工作区修改。

### Training infrastructure

- `pre_config.py` 增加 smoke/full 训练 profile 和隔离的 run tag 规则。
- `pre_trainer.py` 支持真实数据 smoke、单卡训练和 `torchrun` DDP；仅 rank 0 写 checkpoint，训练状态记录 world size/profile。
- `pre_smoke_test.py` 增加训练配置相关回归覆盖。

### Verification

- Python 编译检查通过。
- legacy `smoke_test.py` 通过。
- PRE `pre_smoke_test.py` 通过；本机无 CUDA，4 项 CUDA 专属检查跳过。
- `git diff --check` 与 Markdown 本地链接检查通过。

> 注：上述训练基础设施为当前工作区已有改动，尚未据此声称模型效果改善；真实 GPU smoke 和全量训练仍需在服务器环境执行。
