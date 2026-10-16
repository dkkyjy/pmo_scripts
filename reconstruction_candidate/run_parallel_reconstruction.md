# run_parallel_reconstruction.py — 并行重建调度脚本线性工作原理

*2026-06-22T05:11:07Z by Showboat 0.6.1*
<!-- showboat-id: 9e8621c9-51ee-4ca2-aaeb-2804b794ee05 -->

## 文件职责

`run_parallel_reconstruction.py` 是 **GRAND 实验 Dunhuang 站点数据处理流水线中的并行重建调度脚本**。它负责：

1. 从候选事例列表（txt 或 candidates.yaml）中读取待处理任务
2. 利用多进程并行对每个候选事例执行完整的电场重建（平面波前拟合 PWF）和能量重建（球面波前拟合 SWF + 侧向分布函数拟合 ADF）
3. 将所有事例的重建结果汇总为一个结构化 YAML 文件

该脚本本身不实现重建算法，而是作为调度层调用 `reconstruction_candidate.py` 中的 `efield_recons_from_efield_PWF()` 和 `energy_restruction()`。

## 模块依赖

脚本通过 `sys.path.insert` 将自己所在目录加入模块搜索路径，然后从 `reconstruction_candidate` 导入以下核心函数：

| 函数 | 来源 | 职责 |
|---|---|---|
| `efield_recons_from_efield_PWF` | `reconstruction_candidate.py:298` | 基于平面波前拟合（PWF）的电场重建 |
| `energy_restruction` | `reconstruction_candidate.py:1349` | 能量重建（SWF + ADF 拟合） |
| `efield_rec_dir` | `reconstruction_candidate` | 电场重建结果的落盘目录 |
| `helper` | `reconstruction_candidate` | 辅助模块（含坐标转换等工具函数） |

另外使用了 `logger_config.logger`（loguru 封装）进行日志输出，确保在多进程环境下日志安全（`enqueue=True`）。

## 主流程（main 函数）

`main()` 是该脚本的唯一入口，执行路径如下：

### 1. CLI 参数解析

`argparse` 定义以下参数：

- `--input` 与 `--candidates`：**互斥组**（`add_mutually_exclusive_group`），二者必选其一
  - `--input`：候选事例 txt 列表，每行格式 `<root_file> <event_index>`
  - `--candidates`：candidates.yaml 路径，自动提取 file/index
- `--output`：汇总 YAML 输出路径，默认 `reconstruction_summary.yaml`
- `--ncores`：并行进程数，默认 20
- `--base-path`：ROOT 文件基础目录，默认 `TD`

### 2. 任务列表加载

根据用户选择的输入模式，调用对应的加载函数：

- 若提供了 `--candidates` → `load_tasks_from_candidates_yaml()`
- 否则 → `load_tasks_from_txt()`

两者均返回 `List[Tuple[str, int]]`，即 `(file_path, event_index)` 列表。

### 3. 并行执行

使用 `multiprocessing.Pool` 的 `pool.map()` 将 `reconstruct_one` 映射到任务列表。`pool.map` 保证结果顺序与输入一致。

### 4. 结果汇总

`assemble_summary_yaml()` 收集所有 worker 返回的 dict，去重后写入 YAML。

### 5. 统计报告

最终打印成功/电场失败/能量失败计数。

## 路径自动推导

脚本通过两个工具函数实现 ROOT 文件路径的自动补全：

### `_extract_date_from_filename(filename)`

用正则 `(\d{8})` 从文件名中提取 YYYYMMDD 日期，返回 `(year, month, day)` 三元组。

- 例如 `Trigger_20260604_RUN10379_candidates.yaml` → `("2026", "06", "04")`
- 若文件名不含 8 位连续数字，抛出 `ValueError`

### `_build_root_path(filename, base_path)`

- 如果 `filename` 已是绝对路径或包含目录部分 → 直接返回
- 否则拼接为 `base_path/YYYY/MM/DD/filename`

这意味着用户只需提供 ROOT 文件名，脚本会自动定位到 `TD/2026/06/04/Trigger_xxx.root`。

## 任务加载函数

### `load_tasks_from_txt(txt_path, base_path)`

读取纯文本任务列表，每行格式：

```
Trigger_xxx.root 196
# 注释行
Trigger_yyy.root 203
```

处理逻辑：
1. 跳过空行和 `#` 开头的注释行
2. 每行按空白分割，取第一列为文件名、第二列为 event_index
3. 对每行的文件名调用 `_build_root_path` 补全路径
4. 格式不正确的行发出 warning 后跳过

### `load_tasks_from_candidates_yaml(candidate_yaml_path, base_path)`

读取 candidates YAML，结构为：

```yaml
event_key_1:
  event_number: 12345
  file: Trigger_xxx.root
  index: 196
  ...
```

处理逻辑：
1. 遍历顶层 dict 的每个 event
2. 提取 `payload["file"]` 和 `payload["index"]`
3. 缺失 file 或 index 的条目发出 warning 后跳过
4. 对 file 调用 `_build_root_path` 补全路径

## Worker：`reconstruct_one(args)`

这是每个子进程执行的单位工作函数，接收 `(root_path, event_index)` 元组。

### 默认返回结构

函数首先构造一个带有所有字段默认值（`None` 或空列表）的 dict。这使得即使重建失败，也能返回完整的结构化信息。

`_status` 是内部字段：
- `0` = 成功
- `1` = 电场重建失败
- `2` = 能量重建失败

### 步骤 1：电场重建（PWF）

1. 调用 `efield_recons_from_efield_PWF(root_path, event_index)`
   - 该函数在 `reconstruction_candidate.py:298` 定义
   - 执行平面波前拟合，将结果落盘到 `efield_rec_dir/Trigger_xxx_event_N/`
2. 调用 `_read_efield_txt()` 读取落盘的 `*reconstruction_results.txt`
   - 该文件包含 15 列：event_num, gps_time, LST, du_id, theta_plane, phi_plane, chi2Plane/dof, du_nanoseconds, pos_x, pos_y, pos_z, Fluence, Fluence_theta, Fluence_phi, Fluence_theta_phi
3. 将 PWF 结果（θ, φ, χ²/dof）和天线信息（du_ids, energy_flux, 位置）填入 entry

若任何步骤抛出异常，设置 `_status=1` 并直接返回。

### 步骤 2：能量重建（SWF + ADF）

1. 调用 `energy_restruction(root_path, event_index)`
   - 该函数在 `reconstruction_candidate.py:1349` 定义
   - 内部依次执行：球面波前拟合（SWF）→ 侧向分布函数拟合（ADF）
2. 将返回的 SWF 结果（x_xmax, y_xmax, z_xmax, chi2Sph）和 ADF 结果（θ_sph, φ_sph, A, wc, dw, fmin_per_dof）填入 entry
3. 额外计算每个天线的视角 `omega`：
   - 将球面重构方向 `(θ_s, φ_s)` 转为笛卡尔单位向量 `shower_axis`
   - 对每个天线计算 `(Xmax → 天线)` 的单位向量 `u_ant`
   - `ω = arccos(shower_axis · u_ant)`

若任何步骤抛出异常，设置 `_status=2` 并返回。

### 一致性校验

检查 `du_ids`、`omega_rad`、`energy_flux` 三个数组长度是否一致：
- 不一致 → warning
- 一致 → info 日志（含 event_num, N_DU, θ, φ）

## 结果汇总：`assemble_summary_yaml(results, output_path)`

将 worker 返回的 dict 列表组装为最终的汇总 YAML。

### 处理逻辑

1. 遍历每个 worker 结果
2. `pop("_status")` 移除内部状态字段，同时记录到 `statuses` 列表
3. `pop("event_num")` 作为顶层 key
   - 若 `event_num` 为 `None` → 用 `unknown_<filename>_<index>` 作为后备 key
4. 重复 key 处理：若同一 event_num 出现多次，追加 `_dupN` 后缀
5. 用 `yaml.dump` 写入文件（`default_flow_style=False` 保证块风格输出，`sort_keys=False` 保留插入顺序）

### 返回值

`(summary_dict, statuses_list)` — summary 字典用于后续可能的处理，statuses 列表用于统计。

## 辅助工具函数

### `_read_efield_txt(root_path, event_number_int)`

读取电场重建落盘的 `*reconstruction_results.txt`：

1. 根据 root_path 和 event_number 构造事件目录名 `Trigger_xxx_event_N`
2. 在 `efield_rec_dir/<event_dir>/` 下 glob 匹配 `*reconstruction_results.txt`
3. 用 `np.loadtxt` 读取；若为 1D 数组则扩展为 2D
4. 找不到文件时抛出 `FileNotFoundError`

### `_safe(v)` / `_safe_list(arr)` / `_safe_int_list(arr)`

类型转换工具，确保 numpy 标量转为 Python 原生类型（兼容 YAML 序列化）：
- `_safe`: numpy float/int → Python float/int，`NaN` → `None`
- `_safe_list`: 对数组每个元素调用 `_safe`
- `_safe_int_list`: 转为 Python int 列表（不做 NaN 检查，用于 du_ids）

## 输出 YAML 结构

最终输出的 YAML 文件结构如下：

```yaml
12345:                        # event_num（顶层 key）
  file_path: /path/to/TD/2026/06/04/Trigger_xxx.root
  event_index: 196
  PWF:
    rec_theta_plane: 150.12
    rec_phi_plane: 235.47
    chi2_per_dof: 1.23
  SWF:
    rec_x_xmax: -12345.6
    rec_y_xmax: 23456.7
    rec_z_xmax: 15000.0
    chi2_per_dof: 0.98
  ADF:
    rec_theta_sph: 150.34
    rec_phi_sph: 235.61
    rec_A: 123000.0
    rec_wc: 0.031
    rec_dw: 2.10
    fmin_per_dof: 0.87
  antennas:
    du_ids: [101, 203, 305]
    omega_rad: [0.031, 0.028, 0.035]
    energy_flux: [12300.0, 9870.0, 15600.0]
```

- `NaN` 值在 YAML 中显示为 `null`
- 重复 event_num 会自动重命名为 `<num>_dupN`

## 分支逻辑

脚本的核心分支：

| 决策点 | 位置 | 分支行为 |
|---|---|---|
| 输入模式 | `main()` L380-384 | `--candidates` → YAML 解析；否则 → txt 解析 |
| 路径补全 | `_build_root_path()` L98 | 绝对路径/含目录 → 直接返回；纯文件名 → 自动拼接日期路径 |
| 电场重建 | `reconstruct_one()` L242-262 | 成功 → 继续能量重建；异常 → `_status=1`，跳过能量重建 |
| 能量重建 | `reconstruct_one()` L264-299 | 成功 → 计算 ω 并校验；异常 → `_status=2` |
| 数组一致性 | `reconstruct_one()` L302-315 | 不一致 → warning；一致 → info |
| event_num 重复 | `assemble_summary_yaml()` L341-346 | 唯一 → 直接使用；重复 → 追加 `_dupN` 后缀 |
| event_num 缺失 | `assemble_summary_yaml()` L337-338 | 存在 → 直接使用；`None` → 用 `unknown_<file>_<index>` 替代 |

## 错误处理策略

脚本采用 **per-event 容错** 策略，即单个事例失败不影响其他事例：

- 电场重建失败 → 该事例的 PWF/SWF/ADF 字段均为 `null`，但仍保留在汇总 YAML 中
- 能量重建失败 → PWF 字段有值，SWF/ADF 字段为 `null`
- 所有异常都会通过 `logger.error()` 输出完整 traceback，便于事后排查
- 最终统计报告会分别列出成功数、电场失败数、能量失败数

这种设计确保在批量处理时，少数异常事例不会阻塞整个流水线。

## 使用示例

### txt 输入模式

```bash
python run_parallel_reconstruction.py     --input candidate_list.txt     --output reconstruction_summary.yaml     --base-path /path/to/TD     --ncores 20
```

### candidates YAML 输入模式

## 使用示例

### txt 输入模式

```bash
python run_parallel_reconstruction.py \
    --input candidate_list.txt \
    --output reconstruction_summary.yaml \
    --base-path /path/to/TD \
    --ncores 20
```

### candidates YAML 输入模式

```bash
python run_parallel_reconstruction.py \
    --candidates ../Reco_Dir/Trigger_20260617_RUN10386_candidates.yaml \
    --output reconstruction_summary.yaml \
    --base-path /path/to/TD \
    --ncores 20
```

ROOT 文件路径会自动生成为 `base-path/YYYY/MM/DD/filename`，其中 YYYY/MM/DD 从输入文件名中提取。

## 线性工作原理总览

`run_parallel_reconstruction.py` 的完整执行路径总结如下：

1. **CLI 解析** — `main()` 通过 `argparse` 解析 `--input` 或 `--candidates`（互斥）、`--output`、`--ncores`、`--base-path`
2. **任务加载** — 根据输入类型调用 `load_tasks_from_candidates_yaml()` 或 `load_tasks_from_txt()`，对每个 ROOT 文件名调用 `_build_root_path()` 自动补全为 `base_path/YYYY/MM/DD/filename`
3. **并行调度** — 使用 `multiprocessing.Pool(pool.map)` 将 `reconstruct_one` 分发到 N 个进程
4. **单事例重建**（每个 worker 独立执行）：
   - 4a. 构造带默认值的返回 dict（`_status=0`）
   - 4b. 调用 `efield_recons_from_efield_PWF()` → 读取落盘 txt → 填充 PWF 字段和天线信息
   - 4c. 若 4b 异常 → `_status=1`，跳过后续步骤
   - 4d. 调用 `energy_restruction()` → 填充 SWF 和 ADF 字段 → 计算每个天线的视角 ω
   - 4e. 若 4d 异常 → `_status=2`
   - 4f. 校验 `du_ids`/`omega_rad`/`energy_flux` 数组长度一致性
5. **结果汇总** — `assemble_summary_yaml()` 收集所有 worker 结果，处理重复/缺失 event_num，写入 YAML
6. **统计报告** — 输出成功/电场失败/能量失败计数

### 关键设计决策

- **Per-event 容错**：单个事例异常不阻塞其他事例，所有结果都保留在汇总 YAML 中
- **路径自动推导**：从文件名提取 YYYYMMDD 日期，自动拼接 `TD/YYYY/MM/DD/filename`，减少用户输入
- **YAML 安全序列化**：所有 numpy 类型通过 `_safe()` 转为 Python 原生类型，NaN → null
- **顺序保持**：`pool.map` 保证输出顺序与输入一致，`sort_keys=False` 保留插入顺序
