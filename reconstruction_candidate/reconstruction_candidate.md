# reconstruction_candidate — 电场重建与能量重建模块

*2026-06-22T03:14:33Z by Showboat 0.6.1*
<!-- showboat-id: b0b62fea-9366-403d-bf54-51418d836e11 -->

## 文件职责

`reconstruction_candidate.py` 是 GRAND 实验的 **离线电场重建与能量重建** 模块。它提供两个顶层入口函数：

- **`efield_recons_from_efield_PWF(root_path, event_index)`** — 电场重建：从 ROOT 文件中读取触发事例的 ADC 波形，经过因果一致性筛选、平面波前拟合（PWF）、频域 Voc 转换、电场重建（最小二乘），最终落盘每个 DU 的重建电场时域波形和 fluence。
- **`energy_restruction(root_path, event_index)`** — 能量重建：读取电场重建落盘的 fluence 数据，执行球面波前拟合（SWF）定位 Xmax、ADF 拟合确定簇射方向、切伦科夫分布拟合，最终计算辐射能量。

两个函数独立运行，通过磁盘文件传递数据（电场重建写 `*reconstruction_results.txt`，能量重建读取它）。

## 模块级依赖与常量

模块在 import 时执行以下初始化（已优化为惰性加载）：

### 外部依赖

来自同目录 `reconstruction_candidate/` 的辅助模块：

| 模块 | 用途 |
|------|------|
| `equivalent.CEL` | 天线有效长度计算 |
| `galacticnoise_get.gala` | 银河噪声频谱获取 |
| `Time_domain_Shower_Edata_get.time_data_get` | 时域簇射电场数据 |
| `vad2voc.vad2voc` | ADC → Voc 频域转换 |
| `interpolation_.inter` | 天线方向图插值 |
| `complex_expansion.expan` | 复数系数展开 |
| `AiresInfoFunctions` (as `aires`) | 大气折射率计算 |
| `coordinatesystems` (as `cs`) | v×B 坐标系变换 |
| `helper` | 球面/笛卡尔坐标转换、海拔查询 |
| `models` (as `atm`) | 大气密度模型 |
| `_optimized_read_matching_times_graph` | 图论因果一致性筛选 |

### 全局常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `c` | 299792458 | 光速 (m/s) |
| `tukey_wind` | Tukey(200) | 时域加窗函数 |
| `time_wind` | 100 | 时间窗口大小 |
| `flower, fupper` | 30, 201 | 频率范围 (MHz) |
| `thetaList, phiList` | linspace(90,180,6), linspace(0,360,6) | PWF 初始角度网格 |
| `xList, yList, zList` | 4×4×4 网格 | SWF Xmax 搜索网格 |
| `N, f0, f1` | 2000, 1.0, linspace(0,1000,1001) | 频率点数与范围 |
| `magnetic_field_vector` | 球面→笛卡尔 | GRAND 站地磁场单位矢量 |
| `efield_rec_dir` | `./Reconstruction/` | 结果输出根目录 |

## 工具函数

### `gps_to_lst(gps_times)`

GPS 时间 → 地方恒星时（LST）。公式：
- GPS → Unix: `unix = gps - 18`
- Unix → JD → GMST(度) → GMST(小时) → LST = GMST + lon/15
- 经度固定为 `93.97111533°`（敦煌站）

### `rotation_m(theta, phi)`

球面 → 笛卡尔坐标旋转矩阵（3×3），用于天线有效长度的坐标转换。

### `_ensure_output_dir()` / `_load_du_position()`

惰性初始化函数，避免 import 时触发文件 IO：
- `_ensure_output_dir` 惰性创建 `./Reconstruction/` 目录
- `_load_du_position` 惰性加载 `_gp65_rtksort_2002_DU7.txt` 并预计算旋转后坐标

## 电场重建核心函数

### `e_recons(Voc, Lce, Vnoise)`

**频域最小二乘电场重建**。对每个频率点（30–200 MHz），执行：

1. 构建天线响应矩阵 `A`（3×2，假设只有 θ/φ 分量）
2. 构建噪声协方差矩阵 `sig_V`（对角阵，噪声幅度平方为方差）
3. 计算重建矩阵：`A_ = (A^H · sig_V⁻¹ · A)⁻¹ · A^H · sig_V⁻¹`
4. `[E_θ, E_φ] = A_ · V` 得到该频点的电场分量
5. 同时保存信息矩阵的逆对角线作为不确定度 `invM_Ett`, `invM_Epp`

返回 `(Et_rec, Ep_rec, E_inv)`，各为 171 个频点的复数数组。

> 使用 `np.linalg.solve` 代替 `np.linalg.inv` 以避免病态矩阵放大数值误差。

### `get_lec_grand(e_theta, e_phi, N, f0)`

获取 GRAND 天线在特定方向的有效长度矩阵。
- 从 `EFL_data_3.2m_XYZ_20250313.mat` 加载天线响应数据
- 插值获取指定 `(θ, φ)` 方向的辐射方向图
- 展开复数系数（30–250 MHz）
- 笛卡尔 → 球面坐标转换

返回 `Lce_sphere`（N×3×3 复数矩阵）。

## 到达方向拟合（PWF / SWF）

### `PWF` 类 + `PWF_fit()`

**平面波前拟合**，用于重建簇射到达方向 `(θ, φ)`。

- `PWF.chi2(theta, phi, rc)` 计算所有 DU 对之间的时间差与平面波模型预测的残差：
  `fmin = Δx·sinθ·cosφ + Δy·sinθ·sinφ + Δz·cosθ - rc·c·Δt`
- 残差归一化：除以 `6ns · c`（时间分辨率 × 光速）
- `PWF_fit` 使用 `iminuit` 最小化 chi2，固定 `rc=1`

### `SWF` 类 + `SWF_fit()`

**球面波前拟合**，用于定位簇射极大值位置 `(xs, ys, zs)`。

- `SWF.chi2(xs, ys, zs, rc)` 基于球面波传播模型：
  `fmin = (t·c/n_eff - t_s) - d`
  其中 `n_eff` 由 `aires.GetZHSEffectiveRefractionIndex` 计算
- `SWF_fit` 先用 `simplex` 粗略搜索，再用 `migrad` + `hesse` 精确拟合

### `find_core(x, y, z, theta, phi, z_det)`

计算簇射轴与探测器平面的交点（芯位）及 Xmax 到芯位的距离。

## ADF 拟合（角分布函数）

### `ADF` 类 + `_adf_fit_impl()` / `ADF_fit()` / `ADF_fit_fixdw()`

**切伦科夫辐射角分布拟合**，包含地磁不对称修正。

`ADF.chi2(theta, phi, dw, A, B, rc)` 的计算流程：

1. **坐标变换** → v×B 坐标系（`coordinatesystems.cstrafo`）
2. **芯位计算** → `find_core` 得到 shower core
3. **视角计算** → 每个天线与簇射轴的夹角 ω
4. **地磁修正** → `f_GeoM = 1 + B·sin²(α)·cos(η)`
5. **距离衰减** → `early_late = (d_ant2Xmax / dXmax)²`
6. **切伦科夫函数** → `f_Cerenkov = 1 / (1 + 4·((tan(ω)/tan(wc))² - 1)² / dw²)`
7. **预测值** → `adf = A · f_Cerenkov · f_GeoM / early_late`
8. **归一化残差** → `(p - adf) / (0.1·p + 0.01·max(p))`

当 DU 数量 > 50 时，wc 从数据中动态更新（取最大 fluence_decay 对应的 ω）。

`_adf_fit_impl` 是统一的内部实现，`fix_dw` 参数控制是否固定 dw：
- `ADF_fit` → `fix_dw=False`（自由拟合 dw）
- `ADF_fit_fixdw` → `fix_dw=True`（固定 dw，用于 dw 超出合理范围时）

## 电场重建主流程: `efield_recons_from_efield_PWF(root_path, event_index)`

这是模块的第一个顶层入口。完整执行流程如下：

### 阶段 0：初始化与数据加载

1. 惰性创建输出目录、加载 DU 位置
2. 构造 `event_dir = ./Reconstruction/<filename>_event_<index>/`
3. `uproot.open` 读取 ROOT 文件 `teventadc` Tree（使用 `try/finally` 确保关闭）
4. 提取 `event_number, du_id, gps_time, du_nanoseconds, trace_1/2/3`
5. 计算 LST 时间

### 阶段 1：因果一致性筛选

1. 构造 `detector_positions` 字典（DU ID → 坐标）
2. 生成临时因果文件（格式：`gps_time: [('du_id', ns), ...]`）
3. 调用 `optimized_read_matching_times_graph` 执行图论因果筛选（最少 4 个 DU）
4. 应用掩码更新所有数组（trace、位置等）
5. 若通过筛选的 DU < 4，返回 -1

### 阶段 2：平面波前拟合（PWF）

1. 6×6 网格搜索（`thetaList` × `phiList`）最佳 `(θ, φ)`
2. 每个初始猜测调用 `PWF_fit`，选择 chi2/dof 最小者
3. 角度变换：`θ → 180-θ`, `φ → φ+180`

### 阶段 3：逐 DU 电场重建

对每个通过筛选的 DU 执行：

1. **频域处理**：RFFT → 120 MHz 低通滤波 → IRFFT
2. **Voc 转换**：
   - 在 1 MHz 网格上插值 ADC 频域数据
   - `vad2voc` 转换为开路电压 Voc
   - 对原始频带信号直接乘以硬件响应 `H_real_freq` 和窗函数
   - IRFFT 得到 Voc 时域（单位 μV）
3. **信号加窗**：
   - 寻找 SNR > 5 的触发通道
   - 以峰值位置为中心，Tukey 窗（200 点）加窗
   - 边界安全处理（不超出数组范围）
4. **电场重建**：
   - 加窗信号 → RFFT → 插值到 30–250 MHz
   - 获取天线有效长度 `Lce_sphere` + 银河噪声
   - 调用 `e_recons` 执行最小二乘重建
   - IRFFT 得到时域电场（500 点，2 ns 间隔）
5. **一致性验证**：
   - 从重建电场反算 Voc：`Voc_recalc = Lce · E_rec + Vnoise`
   - 绘制原始 Voc vs 反算 Voc 对比图
6. **保存结果**：`saved_data[i]` 记录 15 列（event_num, gps_time, LST, du_id, PWF 角度, chi2, du_ns, 位置, Fluence 及分量）

### 阶段 4：落盘

- `*reconstruction_results.txt` — 每个 DU 一行，15 列
- `*E_rec_time_all_DUs.txt` — 重建电场时域波形（2D 展平）

### 输出图表（每个 DU 6 张）

| 图表 | 内容 |
|------|------|
| `*_ADC_Traces.png` | 三通道滤波后 ADC 时域波形 |
| `*_ADC_Interpolation_Comparison.png` | 频域插值前后对比 |
| `*_Voc_Frequency_Comparison.png` | Voc 频域（1MHz vs 原生频带） |
| `*_Voc_Time_Domain.png` | Voc 时域波形 |
| `*_Efield_Reconstruction_Frequency.png` | 重建电场频域 |
| `*_Efield_Reconstruction_Time.png` | 重建电场时域 |
| `*_Voc_Recalculated_Time.png` | 反算 Voc 时域 |
| `*_Voc_Consistency_Check.png` | 原始 vs 反算 Voc 三通道对比 |
| `*_Triggered_DUs_Position.png` | 触发 DU 位置图（含 PWF 方向箭头） |

## 能量重建主流程: `energy_restruction(root_path, event_index)`

这是模块的第二个顶层入口。它读取电场重建的落盘文件，执行以下流程：

### 阶段 1：读取电场重建结果

1. 从 `*reconstruction_results.txt` 读取 15 列数据
2. 从 `*E_rec_time_all_DUs.txt` 读取时域电场波形，reshape 为 (N_DU, 500, 2)
3. 提取：du_ids, PWF 角度, chi2, 到达时间, 位置, Fluence 及分量

### 阶段 2：联合重建（SWF + ADF）

调用 `recons_angle()`，执行两步网格搜索：

1. **SWF 网格搜索**：在 4×4×4 的 `(xList, yList, zList)` 网格上调用 `SWF_fit`，选择 chi2 最小者作为 Xmax 位置
2. **ADF 网格搜索**：以 PWF 结果为中心，在 3×3×4×3 的 `(θ, φ, dw, A)` 网格上调用 `ADF_fit`
3. **dw 越界处理**：若拟合的 dw 超出 [0.101, 3.499]，使用参数化公式重新计算 dw 并固定它重新拟合

### 阶段 3：几何分析与绘图

1. 计算 Xmax 海拔高度（`helper.get_local_altitude`）
2. 计算大气密度（`atm.get_density`）
3. 计算地磁角 α（shower 轴与地磁场夹角）
4. 计算芯位（`find_core`）
5. 绘制 shower core 位置图（含 Fluence 色标和切伦科夫角等值线）

### 阶段 4：地磁修正与视角计算

1. 球面 → 笛卡尔 → v×B 坐标系的 Fluence 变换
2. 计算每个天线的视角 ω
3. 从时域电场计算地磁修正后的能量 fluence `f_geo_pos`
4. 绘制 shower plane 能量 fluence 分布图

### 阶段 5：切伦科夫拟合与能量计算

1. 调用 `che_fit` 拟合切伦科夫分布（网格搜索初始值 + iminuit 最小化）
2. 数值积分计算地磁辐射能量 `E_rad`
3. 密度和地磁角修正（`density_and_alpha_correction`）
4. 非对称 ADF 双重积分计算 `E_em`
5. 误差传播：分别扰动 A、wc、dw 参数计算能量误差
6. 绘制切伦科夫角分布拟合图

### 输出打印

三行关键输出（被 `run_parallel_reconstruction.py` 解析）：
- `"Reconstructed Xmax coordinates (m) and chi2: x y z chi2"`
- `"Reconstructed shower direction (theta, phi in degrees and errors): θ dθ φ dφ"`
- `"Reconstructed ADF parameters (A, wc, dw, fmin/DOF): A wc dw fmin"`

## 辅助拟合函数

### `recons_angle(x, y, z, t, p, init_input, obs_altitude)`

联合 SWF + ADF 重建的编排函数。输入为天线位置、到达时间、信号幅度、PWF 初始结果和观测高度。内部执行：
1. 4×4×4 SWF 网格搜索 → Xmax 位置
2. 3×3×4×3 ADF 网格搜索 → 方向 + 切伦科夫参数
3. dw 越界时自动回退到固定 dw 的 ADF 拟合

### `che_fit(omega_, data, amp0, w_c0, domega0)`

切伦科夫分布拟合。在 5×5×5 网格上搜索初始值，每个初始值调用 `che_fit_chi2`（iminuit 最小化），选择 chi2 最小者。

### `density_and_alpha_correction(E_em_GeV, rho, alpha)`

密度和地磁角修正公式（参考 arXiv:2507.06698v1），将地磁辐射能量修正为真实簇射能量。

### `f_Che(par, omega_)` / `f_adf(par, omega_, alpha_, phi_)`

切伦科夫函数模型（对称 / 带地磁不对称修正），用于拟合和积分。

## 数据流与文件约定

### 输入

| 来源 | 格式 | 内容 |
|------|------|------|
| `root_path` | ROOT (`.root`) | `teventadc` Tree，含 event_number, du_id, gps_time, du_nanoseconds, trace_1/2/3 |
| `_gp65_rtksort_2002_DU7.txt` | 文本 | DU 位置数据（ID, x, y, z） |
| `EFL_data_3.2m_XYZ_20250313.mat` | MATLAB | GRAND 天线有效长度数据 |

### 电场重建输出（`efield_recons_from_efield_PWF`）

所有文件写入 `{efield_rec_dir}/{root_basename}_event_{index}/`：

| 文件 | 格式 | 内容 |
|------|------|------|
| `*_reconstruction_results.txt` | 空格分隔 | 每 DU 一行 15 列 |
| `*_E_rec_time_all_DUs.txt` | 空格分隔 | 每 DU 一行，展平的时域电场 |
| `*_Triggered_DUs_Position.png` | PNG | 触发 DU 位置 + PWF 方向 |
| `*_DU_{id}_ADC_Traces.png` | PNG | ADC 时域波形 |
| `*_DU_{id}_ADC_Interpolation_Comparison.png` | PNG | 频域插值检查 |
| `*_DU_{id}_Voc_Frequency_Comparison.png` | PNG | Voc 频域对比 |
| `*_DU_{id}_Voc_Time_Domain.png` | PNG | Voc 时域波形 |
| `*_DU_{id}_Efield_Reconstruction_Frequency.png` | PNG | 重建电场频域 |
| `*_DU_{id}_Efield_Reconstruction_Time.png` | PNG | 重建电场时域 |
| `*_DU_{id}_Voc_Recalculated_Time.png` | PNG | 反算 Voc |
| `*_DU_{id}_Voc_Consistency_Check.png` | PNG | Voc 一致性对比 |

### 能量重建输出（`energy_restruction`）

| 文件 | 格式 | 内容 |
|------|------|------|
| `shower_core_position.png` | PNG | Shower core 位置 + Fluence 分布 |
| `fluence_map_shower_plane.png` | PNG | Shower plane 能量 fluence 分布 |
| `che_fit_figure.png` | PNG | 切伦科夫角分布拟合 |
| stdout | 文本 | Xmax 坐标、方向、ADF 参数（被上游解析） |

### `*reconstruction_results.txt` 列映射（N_SAVED_COLS = 15）

| 列 | 含义 |
|----|------|
| 0 | event_num |
| 1 | gps_time |
| 2 | LST |
| 3 | du_id |
| 4 | theta_plane (PWF) |
| 5 | phi_plane (PWF) |
| 6 | chi2Plane/dof |
| 7 | du_nanoseconds |
| 8 | pos_x (rotated) |
| 9 | pos_y (rotated) |
| 10 | pos_z (rotated) |
| 11 | Fluence |
| 12 | Fluence_theta |
| 13 | Fluence_phi |
| 14 | Fluence_theta_phi |

## 错误处理与返回码

### `efield_recons_from_efield_PWF`

| 返回值 | 含义 |
|--------|------|
| 0 | 成功完成所有 DU 的电场重建 |
| -1 | 因果筛选后无 DU 通过（`matching_times` 为空）或通过 DU < 4 |

- `uproot.open` 使用 `try/finally` 确保文件关闭
- 因果筛选的临时文件在读取后立即删除

### `energy_restruction`

| 返回值 | 含义 |
|--------|------|
| 0 | 成功完成能量重建 |

- 依赖电场重建已落盘的文件；若文件不存在会由 `glob` / `np.loadtxt` 抛出异常
- ADF 拟合中 dw 越界时自动回退到固定 dw 的拟合
- wc 计算失败时捕获异常并打印警告（不中断流程）

## 线性工作原理总览

### 电场重建（`efield_recons_from_efield_PWF`）

1. **初始化** — 惰性创建输出目录、加载 DU 位置、构造 event_dir
2. **读取 ROOT** — `uproot.open` → `teventadc` Tree → 提取 event_number, du_id, gps_time, du_nanoseconds, trace_1/2/3
3. **DU 位置匹配** — 遍历 du_ids_all，从 DU_position 查找每个 DU 的 (x, y, z)
4. **因果一致性筛选** — 构造临时因果文件 → `optimized_read_matching_times_graph` → 应用掩码 → 若通过 DU < 4 则返回 -1
5. **PWF 拟合** — 6×6 网格搜索 `(θ, φ)` → `PWF_fit`（iminuit）→ 角度变换
6. **逐 DU 电场重建**（对每个触发 DU）：
   - ADC RFFT → 120 MHz 低通滤波 → IRFFT
   - 频域插值 → `vad2voc` → Voc 时域
   - 信号加窗（Tukey 窗，SNR > 5 触发）
   - 获取 Lce_sphere + 银河噪声 → `e_recons` 最小二乘重建
   - IRFFT → 时域电场 → 反算 Voc 一致性验证
   - 记录 Fluence 及分量到 `saved_data`
7. **落盘** — `*reconstruction_results.txt`（15 列）+ `*E_rec_time_all_DUs.txt`

### 能量重建（`energy_restruction`）

1. **读取电场结果** — `*reconstruction_results.txt` + `*E_rec_time_all_DUs.txt`
2. **联合重建** — `recons_angle`：
   - SWF 4×4×4 网格搜索 → Xmax 位置
   - ADF 3×3×4×3 网格搜索 → 方向 + 切伦科夫参数
   - dw 越界时固定 dw 重新拟合
3. **几何计算** — Xmax 海拔、大气密度、地磁角 α、芯位
4. **地磁修正** — 球面→笛卡尔→v×B 坐标变换 → `f_geo_pos` 计算
5. **切伦科夫拟合** — `che_fit` 网格搜索 + iminuit → 拟合参数
6. **能量计算** — 数值积分（对称 + 非对称 ADF）→ 密度/地磁修正 → 误差传播
7. **输出** — 打印 Xmax 坐标、方向、ADF 参数（供上游解析）
