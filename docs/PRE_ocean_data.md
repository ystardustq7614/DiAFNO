# PRE_ocean_data 数据集详细说明

> 数据源：`/data/PRE_ocean_data`（在 `~/datasets/PRE` 有软链接指向同一目录）
> 本文档基于 2026-08-24 的实测核对（运行 `scripts/inspect_pre_dataset.py`、`scripts/inspect_raw_nc.py`、`scripts/analyze_pre_dataset.py` 生成）。

---

## 1. 概述

该数据集实为**粤港澳大湾区近海（GBA）COAWST/ROMS 模型逐日平均输出**，目录名沿用 `PRE_ocean_data`（PRE 为模式实验名）。

- **模型**：ROMS/TOMS nonlinear model (COAWST)，包含泥沙悬浮测试（"Suspended Sediment Test in an Estuary"）
- **时间**：1994-01-01T12:00 ~ 2022-12-30T12:00，逐日一个时次，共 **10591** 个时次，无缺档
- **空间**：约 112.3°E–115.7°E，20.9°N–23.1°N，水平分辨率 ~1 km；垂直 30 层地形追随 sigma 坐标
- **总量**：约 **4.1 TB**（raw 2.6 TB + processed 1.5 TB）
- 格式：原始为 NetCDF-4/HDF5（`.nc`，CF-1.4 / SGRID-0.3 规范）；已处理为 NumPy `.npy`

## 2. 目录结构

```text
/data/PRE_ocean_data/
├── READ.md                        # 数据集说明（标题写的是 GBA_ocean_data）
├── metadata.json                  # 机器可读元数据（raw/ 下）
├── docs/
│   └── data_dic.md                # 数据字典（描述与实测略有出入，见 §6）
├── raw/
│   ├── PRE-90921-V2.nc            # 静态网格文件（Gridpak，2016 年生成）
│   └── dyn/
│       └── coawst_avg_00001..10591.nc   # 逐日动态场，10591 个文件，2.6 TB
├── processed/
│   ├── dyn_var/                   # 12 个动态变量 .npy（大文件 209 GB 级）
│   └── stat_var/                  # 27 个静态量 .npy
├── scripts/                       # process.py / lon_lat_interpolation.py / plot_examples.py
└── examples/
    └── quick_start.ipynb          # 快速上手
```

## 3. 原始数据 raw/

### 3.1 动态场 `raw/dyn/coawst_avg_*.nc`（10591 个）

- 每个文件恰含 **1 个时次**（`ocean_time` 维度=1），文件名序号即时间顺序（`coawst_avg_00001.nc` = 1994-01-01）
- 单文件约 253 MB（265,314,517 字节）
- 维度：`s_rho=30, s_w=31, eta_rho=400, xi_rho=441, eta_u=400, xi_u=440, eta_v=399, xi_v=441`
- **三维/二维场变量**（float32）：

| 变量 | 形状 | 说明 |
|---|---|---|
| `temp` / `salt` / `rho` | (1, 30, 400, 441) | 温度/盐度/密度（ρ 点，全水平覆盖） |
| `u` / `v` | (1,30,400,440) / (1,30,399,441) | 网格坐标原始流速（C 网格交错） |
| `u_eastward` / `v_northward` | (1,30,400,441) | 旋转到正东/正北的流速分量 |
| `omega` / `w` | (1,31,400,441) | 垂直速度（w 点，量级 ~1e-3 m/s） |
| `ubar` / `vbar` | (1,400,440) / (1,399,441) | 深度平均流速 |
| `ubar_eastward` / `vbar_northward` | (1,400,441) | 旋转后的深度平均流速 |
| `zeta` | (1,400,441) | 自由面高度 |
| `AKv` / `AKt` / `AKs` | (1,31,400,441) | 湍流扩散/粘性系数（无 NaN） |

> 注意：**`w` 和 `omega` 只在原始 nc 中存在**，`processed/dyn_var/` 未导出。

### 3.2 静态网格 `raw/PRE-90921-V2.nc`

- Gridpak 网格文件（"Teignmouth Grid converted from Delft3D"，2016-07-18）
- 提供：`h`(水深 1–92.57 m)、`f`、`pm/pn`(网格度量)、`angle`(旋转角)、`mask_rho/u/v/psi`、`x/y_rho/psi/u/v`、`lat/lon_psi/u/v` 及投影参数
- **只含 psi/u/v 网格经纬度，不含 lat_rho/lon_rho**（rho 经纬度由 `scripts/lon_lat_interpolation.py` 从 psi 四点插值得到）

## 4. 已处理数据 processed/

### 4.1 动态变量 `processed/dyn_var/`（实测核对）

所有变量 float32，NaN 率与 rho 网格陆地比例（29.96%）一致。大文件用 mmap 读取。

| 变量 | 形状 | 磁盘大小 | NaN% | min | max | mean | std |
|---|---|---|---|---|---|---|---|
| `temp` | (10591,30,400,441) | 208.8 GiB | 29.96 | 10.15 | 33.59 | 22.28 | 3.881 |
| `salt` | (10591,30,400,441) | 208.8 GiB | 29.96 | 1.945 | 40.27 | 30.89 | 7.893 |
| `rho` | (10591,30,400,441) | 208.8 GiB | 29.96 | **-3.791** | 28.11 | 21.06 | 6.37 |
| `u_eastward` | (10591,30,400,441) | 208.8 GiB | 29.96 | -1.718 | 2.692 | -0.045 | 0.203 |
| `v_northward` | (10591,30,400,441) | 208.8 GiB | 29.96 | -1.462 | 0.821 | -0.023 | 0.120 |
| `u` | (10591,30,400,440) | 208.3 GiB | 30.69 | -1.724 | **7.009** | -0.045 | 0.218 |
| `v` | (10591,30,399,441) | 208.3 GiB | 30.73 | -1.326 | 0.946 | -0.007 | 0.105 |
| `ubar_eastward` | (10591,400,441) | 7.0 GiB | 29.96 | -1.051 | 0.965 | -0.041 | 0.185 |
| `vbar_northward` | (10591,400,441) | 7.0 GiB | 29.96 | -1.283 | 0.475 | -0.021 | 0.103 |
| `ubar` | (10591,400,440) | 6.9 GiB | 30.69 | -1.044 | 1.652 | -0.041 | 0.204 |
| `vbar` | (10591,399,441) | 6.9 GiB | 30.73 | -0.787 | 0.530 | -0.006 | 0.075 |
| `zeta` | (10591,400,441) | 7.0 GiB | 29.96 | 0.161 | 1.923 | 0.712 | 0.193 |

说明：
- 维度含义：`T=10591`(天), `s=30`(sigma 层), `η=400`(行), `ξ=441`(列)；u/v 类在 C 网格上错开一列/一行
- 数值取自时间抽样（默认 200 时次 + 空间 1/4 抽样），非全量统计，仅作量级参考
- `rho` 出现负值、`u` 出现 7.0 等极大值，可能是边界/近岸奇异点，使用时需注意
- 陆地格点为 **NaN**

### 4.2 静态量 `processed/stat_var/`（实测核对）

| 变量 | 形状 | 说明 |
|---|---|---|
| `mask_rho/u/v` | (400,441)/(400,440)/(399,441) | 海陆掩膜（1=海，0=陆） |
| `h` | (400,441) | 水深 3.8–92.6 m（湿点，均值 36.9 m） |
| `lon_rho` / `lat_rho` | (400,441) | rho 点经纬度 |
| `lon_psi` / `lat_psi` | (399,440) | psi 点经纬度 |
| `angle` / `f` / `pm` / `pn` | (400,441) | 旋转角 / 科氏参数 / 度量因子 |
| `x_rho` `y_rho` `x_u` `y_u` `x_v` `y_v` | 对应网格 | 投影坐标 (m) |
| `s_rho` / `Cs_r` | (30,) | sigma 层坐标与拉伸函数（-1..-0.983 / -0.999..-0.965） |
| `Cs_w` | (31,) | w 点拉伸函数（-1..0） |
| `hc` / `Tcline` | () | 1.0 / 100.0（Vtransform=2 型坐标） |
| `theta_s` / `theta_b` | () | 5.0 / 1.0（表层加密） |
| `s_w` | (31,) | **文件损坏（空数组，无法读取）** |
| `meta` | object | 混杂对象数组，无法 mmap（可用 `allow_pickle=True` 读） |

### 4.3 海陆掩膜分析

- 湿格点：**123265 / 176400 = 69.9%**（陆地 53135）
- 湿行：396/400（首行 0、末行 395 均有湿点）；湿列：441/441（全列有湿点）
- 全域经纬度：112.309–115.678°E，20.896–23.126°N
- 湿区经纬度：112.315–115.678°E，20.896–23.028°N
- 水深：湿点 3.8–92.6 m（近岸浅、东南深）

## 5. 与 DiAFNO 模型的适配要点

模型 `trainer.py` / `IAFNO.py` 的关键约束与数据现状对比：

| 项目 | 模型要求 | PRE 数据现状 | 差距 |
|---|---|---|---|
| 数据形态 | 单个 .npy：`[trainset_num, nt, x, y, z, c]`，c=3 | 每变量独立 .npy：`[10591, s, η, ξ]` | 需重组、切片、选 3 通道 |
| 网格 | 硬编码 64×66×32（`dim`），64×65×32（`dim_f`），patch 2³ | 400×441×30（sigma） | 需水平降到 64×65、垂直转成 32 层 |
| 通道 | 3 通道（原为湍流 u,v,w） | processed 无 w；raw 有 `w`/`omega`(31 层) | 需选 (u,v,w)/(u,v,temp)/… |
| 有效域 | 无掩膜周期盒 | 70% 湿点、陆地为 NaN | NaN 会污染 loss，需填 0 或裁剪 |
| 内存 | 单文件小、`count`≤200 | 单个 3D 变量 209 GiB 无法整体加载 | 需 mmap 分块或先构建子集 |
| 轨迹 | 多轨迹（bs×nt） | 单条 10591 天连续轨迹 | 需按段切块构造样本 |

参考（`AGENTS.md`）：
- 模型张量为 `bs c x y z`，原始数据为 `bs x y z c`，`trainer.py` 用 einops `rearrange` 转换
- 归一化为按通道 min-max，缓存文件名 `ts{trainset_num}_c{count}_iw{InferenceWidth}_ii{InitialInterval}.npy`（内含 sigma 作为扩散模型 `sigma_data`）
- `trainer.py` 中有 3 处占位符必须替换：`np.load('your dataset')`、`info_folder_path`、`parent_dir`

## 6. 深度核查（变量清单 / mask 语义 / 趋势分布 / 网格几何）

本节为 `scripts/analyze_pre_dataset.py` 的实测结果，回答"每个变量是什么、每个值代表什么、趋势分布如何、网格多大"。

### 6.1 原始 NetCDF 完整变量清单

**动态文件 `coawst_avg_00001.nc`（共 85 个变量：17 个场 + 68 个标量/配置参数）**

| 变量 | 形状 | 含义 | 单位 |
|---|---|---|---|
| `zeta` | (1,400,441) | 自由面高度 | m |
| `ubar`/`vbar` | (1,400,440)/(1,399,441) | 深度平均流速（C 网格原始方向） | m/s |
| `ubar_eastward`/`vbar_northward` | (1,400,441) | 深度平均流速（旋转到正东/正北） | m/s |
| `u`/`v` | (1,30,400,440)/(1,30,399,441) | 三维流速（C 网格原始方向） | m/s |
| `u_eastward`/`v_northward` | (1,30,400,441) | 三维流速（正东/正北分量） | m/s |
| `omega`/`w` | (1,31,400,441) | 垂直速度（w 点） | m/s |
| `temp`/`salt`/`rho` | (1,30,400,441) | 温度/盐度/密度 | °C/PSU/kg·m⁻³ |
| `AKv`/`AKt`/`AKs` | (1,31,400,441) | 湍流粘性/扩散系数（无 NaN） | m²/s |

68 个标量/配置变量：模式运行参数（dt、nHIS、边界条件开关、数值格式等）及垂向坐标（`theta_s=5, theta_b=1, Tcline=100, hc=1, Cs_r(30), Cs_w(31)`）。**标量不含空间信息，非科学场。**

**静态网格文件 `PRE-90921-V2.nc`（共 42 个变量：26 个场 + 16 个投影/标量）**：`h`(1–92.57 m)、`f`、`pm/pn`、`dndx/dmde`、`mask_rho/u/v/psi`、`x/y_rho/psi/u/v`、`lat/lon_psi/u/v`、`angle`、`hraw`；16 个投影参数（JPRJ/PLAT/ROTA 等）。

### 6.2 mask 语义验证（0=陆地，1=海洋 ✓）

- `mask_rho` 唯一取值 `{0, 1}`，与文档一致：**1=海洋，0=陆地**
- 交叉验证：`temp[0,29]` 在陆地格点 100% 为 NaN、海洋格点 0% 为 NaN，**NaN 位置与 mask==0 完全吻合（100%）**
- 水深 `h`：陆地格点恒为 1.0（名义值，勿当真实水深），海洋格点 3.77–92.6 m
- 图 `plots/01_field_mask_sanity.png`：mask、h、temp 原始(陆地=白) 与掩膜后对比，肉眼可直接核对

### 6.3 时间演化与分布（代表性深水点 115.17°E, 21.75°N, h≈92 m）

- `zeta`：10591 天全有值，范围 0.116–1.271 m（均值 0.663 m）；年际平均图见 `plots/02_zeta_trend.png`
- `temp`：表层(层29)季节变化 15.9–29.2 °C、底层(层0) 14.4–26.6 °C（表层变化大、底层稳定，符合物理）；见 `plots/03_temp_trend.png`
- 分布（湿点抽样 400 天）：temp 表层主峰 ~22–27 °C；salt 表层 2–40 PSU（河口混合）；`u_eastward` 表层 mean -0.06, std 0.30 m/s（幅度 3.28 m/s）；见 `plots/04_distributions.png`

> ⚠️ **垂直层索引约定（易错）**：`s_rho` 从 -0.983（**idx0=底层**）到 -0.017（**idx29=表层**）。即 **level 0 = 海底，level 29 = 海面**，与直觉相反，后续取"表层"一律用 `[...,-1,...]`。

### 6.4 时空维度 / 网格大小 / 区域范围

| 项 | 值 |
|---|---|
| 水平维度 | rho 网格 **400 × 441**（不是 400×400；η=400 行、ξ=441 列） |
| 垂直维度 | 30 层 sigma（w 点 31 层） |
| 时间 | 10591 天（1994-01-01 ~ 2022-12-30，~29 年） |
| 网格尺寸(几何) | `dx = 1/pm` 中位 **758 m**（216–995 m）；`dy = 1/pn` 中位 **407 m**（169–824 m）→ 各向异性 ~0.76 km × 0.41 km |
| 网格尺寸(经纬度) | 经向 0.00631° / 纬向 0.00318°（≈131 列/度、188 行/度） |
| 区域范围(湿区) | 经度 112.315–115.678°E（**3.36°** ≈ **347 km**）；纬度 20.896–23.028°N（**2.13°** ≈ **237 km**） |

**注意**：分辨率约 **0.006°（~0.6 km）**，远细于常说的 0.25° 格点；整个区域约 3.4°×2.1°（~350×240 km），约为 131×188 格点/度。网格为曲线正交、各向异性，降采样到 DiAFNO 的 64×65 时不能简单按等度距取点。

## 7. 已知问题 / 与文档不符处

1. **`s_w.npy` 损坏**：128 字节、数组为 0 个元素，`np.load` 报 "cannot reshape array of size 0 into shape (31,)"。w 层坐标可从 `Cs_w.npy`(31 元素) 代替。
2. **`meta.npy` 为 object 数组**：无法 mmap，需 `allow_pickle=True`；用途待确认（疑为导出时残留）。
3. **`docs/data_dic.md` 中示例路径写的是 `/data0/GBA_ocean_data`**，与当前 `/data/PRE_ocean_data` 不符，且提及 lat/lon 文件名（`lat_rho.npy`/`lon_rho.npy`）与 `lon_lat_interpolation.py` 输出名（`lat.npy`/`lon.npy`）不一致——以实测 `lon_rho.npy`/`lat_rho.npy` 为准。
4. **`rho` 存在负值、`u` 存在 7.0 m/s 量级极值**：可能为近岸/边界奇异点，统计和归一化时应关注。
5. `raw/dyn` 文件创建时间均为 2023-12-22，属批量落盘，时间元数据在 `ocean_time` 内，文件名即序。

## 8. 数据读取脚本

本项目下已保存（可反复使用，均为只读）：

| 脚本 | 用途 |
|---|---|
| `scripts/inspect_pre_dataset.py` | 全数据集体检：文件清点、形状/dtype、NaN 率、逐变量统计、掩膜/水深、sigma 坐标。大文件 mmap 抽样读取，安全。`--full` 用 10% 时次做统计 |
| `scripts/inspect_raw_nc.py` | 检查单个原始 NetCDF（动态场或静态网格）的维度/变量/字段统计 |
| `scripts/analyze_pre_dataset.py` | 深度核查：变量清单、mask 语义验证、趋势/分布图、网格几何与区域范围；图输出到 `plots/` |

常用命令：

```bash
python3 scripts/inspect_pre_dataset.py          # 快速体检
python3 scripts/inspect_pre_dataset.py --full   # 全量统计（读 ~22 GB/大文件，耗时较长）
python3 scripts/inspect_raw_nc.py /data/PRE_ocean_data/raw/dyn/coawst_avg_00001.nc
python3 scripts/inspect_raw_nc.py /data/PRE_ocean_data/raw/PRE-90921-V2.nc
python3 scripts/analyze_pre_dataset.py          # 深度核查 + 生成 plots/*.png
```
