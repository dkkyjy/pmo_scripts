# pmo_scripts

敦煌 DU 阵列数据分析脚本：从 ROOT 文件读取事件数据，完成时间戳/时间的提取、
时间/位置标定、因果律（causality）匹配、时间差指纹过滤与定制化筛选，
并用平面波模型（PWM）与球面波模型（SWM）拟合出射方向，最后输出拟合参数与
可视化图表。

按数据来源分三种输入布局（由 `loop.py --reader` 选择）：

| reader | 输入（`--base-path`） | TTree / 布局 | 说明 |
| --- | --- | --- | --- |
| `trigger`（默认） | 目录 `base-path/<date>/Trigger*.root` | `teventadc` | 真实触发数据，含 `gps_time`、`trace_0..3`（F/X/Y/Z）；需给 `start`/`end` 日期范围 |
| `coreas` | 目录，递归扫描 `**/adc_*.root` | `tadc` | CoREAS/Dunhuang 模拟，`trace_ch` 形状 `(n_du, 3, n_time)`（X/Y/Z），无 `gps_time`，逐事件内嵌 `det_pos` |
| `starshape` | 目录（递归扫描 `*stshp*.root`）或单个 `*stshp*.root` 文件 | 三个 collection | StarShape（ZHAireS）模拟，无 `gps_time`/`du_id`/`du_nanoseconds`，逐事件内嵌 `det_pos` |

> `coreas` / `starshape` **仅由 `--base-path` 限定分析文件**，不需要日期参数
> （即使传了也会被忽略），输出分别固定落在 `coreas/`、`starshape/` 子目录
> （不按日期分目录）。`trigger` 仍需 `start`/`end` 日期范围。

trigger 的 TTree 类型由 `readroot/common.py` 的 `find_event_tree_key` 自动识别；
`coreas` / `starshape` 由各自的专用 reader（`readroot/read_coreas.py`、
`readroot/read_starshape.py`）读取，均把逐事件探测器布局写入 matching YAML 的
`det_pos` 字段。

## 目录结构

```
.
├── loop.py                     # 批量驱动：按时间范围生成并执行 read + main 命令
├── main.py                     # 单文件分析主入口：calibration → matching → fingerprint → PWM → 定制化筛选 → SWM → PWM
├── logger_config.py            # 统一日志配置（loguru）
├── time_position_offset.py     # 时间/位置标定子程序（apply_calibration，默认启用）
├── du_pair_theoretical.py      # 由坐标文件生成/加载 DU 对理论时间差
├── readroot/
│   ├── common.py               # TTree 识别、时间映射、YAML 读写等公共函数
│   ├── read_header.py          # trigger：读取 du_id / gps_time / du_nanoseconds → <file>.yaml
│   ├── read_trace.py           # trigger：读取波形并计算 [left:right] 内 ADC 峰值 → <file>_<ch>.yaml
│   ├── read_coreas.py          # coreas：读取 tadc 并合并 companion 文件 → 逐事件 det_pos/truth YAML
│   └── read_starshape.py       # starshape：读取 SimShower/SimEfield 等 collection → 逐事件 det_pos/truth YAML
├── find_event/
│   ├── io.py                   # 读取 DU 坐标文件
│   ├── matching_times_graph.py # 因果律匹配（causality matching，主用）
│   ├── matching_times.py       # 因果律匹配（旧实现，保留）
│   ├── time_difference_fingerprint.py  # 时间差指纹过滤（剔除固定源）
│   ├── plane_wave_model_gradient.py    # 平面波模型（PWM）拟合（主用）
│   ├── spherical_wave_model_nopenal.py # 球面波模型（SWM）拟合（主用）
│   ├── plane_wave_model.py / spherical_wave_model.py  # 早期模型实现（保留）
│   ├── shower_footprint_filter.py      # PWM 与 SWM 之间的定制化筛选（footprint + ring cut）
│   ├── time_offset.py          # DU 时间偏移配置
│   ├── estimation.py / plotting.py / utils.py / background_rejection.py  # 估计、绘图与通用工具
│   └── ...
├── position/                   # 探测器坐标文件
│   ├── _gp65_rtksort_DU7_1029.txt  # trigger 默认探测器坐标（main.py 内默认）
│   ├── _gp65_rtksort_2002_DU7.txt  # trigger 备用坐标
│   ├── _gp65_rtksort.txt
│   ├── du_coords_CoreasDC2.txt     # CoREAS DC2 坐标（逗号分隔）
│   └── *_du_pair_theoretical.csv   # 各坐标文件对应的 DU 对理论时间差缓存
├── calibration/
│   └── jul24_locked_xyz_additive_utc.csv  # 时间/位置标定表（时间窗 + 每 DU 偏移）
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：`numpy`、`scipy`、`numba`、`matplotlib`、`scienceplots`、`loguru`、
`uproot`、`pyyaml`；测试依赖 `pytest`、`pytest-cov`。绘图到 PDF 合并需要
`Pillow`（`loop.py` 中可选，缺失时跳过 PDF 生成）。

## 使用

### 1. 批量驱动 `loop.py`

默认只打印将要执行的命令，加 `--run` 才会真正执行：

```bash
# 只预览命令（trigger：需给日期范围）
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59

# 真正执行（trigger 数据）
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59 --run --jobs 4

# 处理 CoREAS 模拟数据（无需日期，--base-path 指向目录）
python loop.py --reader coreas --base-path /path/to/sim \
    --out-dir-base ../Reco_Dir --run

# 处理 StarShape 模拟数据（无需日期；--base-path 可为目录或单个 ROOT 文件）
python loop.py --reader starshape \
    --base-path /path/to/Proton_EeV_Mazongshan_stshp_2100Events.root \
    --out-dir-base ../Reco_StarShape --max-events 10 --run
```

`loop.py` 对每个 ROOT 文件依次执行：

1. read 阶段：trigger 用 `readroot/read_header.py`（`--with-signal` 时改用
   `readroot/read_trace.py`）；`--reader coreas` 用 `readroot/read_coreas.py`，
   `--reader starshape` 用 `readroot/read_starshape.py`；
2. `main.py`；
3. 把当日产生的 PNG 合并为同名 PDF（需 Pillow）。

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `start` `end` | 无（`trigger` 必填） | 时间范围，格式 `YYYY-MM-DDThh:mm:ss`；仅 `trigger` 使用，coreas/starshape 忽略 |
| `--reader` | `trigger` | 输入布局：`trigger` / `coreas` / `starshape` |
| `--base-path` | `/mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD` | trigger：含 `base-path/<date>/Trigger*.root` 的目录；coreas：递归扫描 `adc_*.root` 的目录；starshape：递归扫描 `*[sS]tshp*.root`（大小写不敏感，兼容 `Stshp_...` / `..._stshp_...`）的目录，或单个 `*[sS]tshp*.root` 文件 |
| `--out-dir-base` | `../Reco_Dir` | 输出根目录（trigger 按 `yyyy/mm/dd` 组织；coreas/starshape 分别固定落在 `coreas/`、`starshape/`） |
| `--run` | 关 | 实际执行，否则仅打印 |
| `--jobs` | `1` | 并行 worker 数 |
| `--limit` | `0` | 限制处理文件数（0=全部） |
| `--skip-read` | 关 | 跳过 read 阶段 |
| `--only-read` | 关 | 只跑 read 阶段，不跑 `main.py` |
| `--with-signal` | 关 | 读取波形/幅值，并让 `main.py` 处理信号 |
| `--left` / `--right` | `0` / `512` | `read_trace`（trigger）与 coreas reader 的采样窗口区间；starshape 无窗口概念，不透传 |
| `--channel` | `XY` | 匹配文件通道后缀：`F`/`X`/`Y`/`Z`/`XY` |
| `--det-pos` | 省略（`main.py` 内默认 `position/_gp65_rtksort_DU7_1029.txt`） | 探测器坐标文件；coreas/starshape 忽略此项（布局逐事件内嵌于 matching YAML），trigger 不传时用 `main.py` 默认 |
| `--skip-matching` / `--no-skip-matching` | 见下 | 跳过因果律匹配阶段 |
| `--skip-fingerprint` / `--no-skip-fingerprint` | 见下 | 跳过指纹过滤阶段 |
| `--force-recompute` | 关 | 忽略已有 `_matched`/`_PWM`/`_SWM` 缓存并重算 |
| `--max-events` | `0` | 转发给 coreas/starshape reader，只读前 N 个事件（0=全部） |
| `--min-detectors` | `5` | 匹配阶段最少探测器数，转发给 `main.py` |
| `--skip-footprint` | coreas/starshape 下默认关 | 跳过定制化 footprint 形状筛选（trigger 默认启用；sim 默认跳过，见下文差异说明） |
| `--skip-ringcut` | coreas/starshape 下默认关 | 跳过定制化 d_max ring cut（trigger 默认启用；sim 默认跳过，见下文差异说明） |
| `--skip-offset` | 关 | 跳过时间/位置标定（默认启用） |
| `--offset-csv` | `./calibration/jul24_locked_xyz_additive_utc.csv` | 标定 CSV 路径 |

`--skip-matching` 与 `--skip-fingerprint` 默认值随 `--reader` 推导：
**coreas / starshape 默认跳过**（模拟事件本身就是信号），**trigger 默认不跳过**。
显式传入 `--skip-matching`/`--no-skip-matching`（指纹同理）可覆盖该默认。

> 说明：这两个阶段是面向真实触发数据设计的——因果律匹配用于剔除偶然符合，
> 指纹过滤用于剔除固定重复源。CoREAS/StarShape 模拟数据无此类噪声，默认跳过可省去开销。

`--det-pos` 默认不传：`loop.py` 不再按 `--reader` 推导坐标文件，省略时
`main.py` 回退到自身默认 `position/_gp65_rtksort_DU7_1029.txt`。
**coreas / starshape 的布局逐事件内嵌于 matching YAML，`main.py` 会优先使用逐事件坐标**，
故静态 `--det-pos` 只对 trigger 布局有意义；显式传入 `--det-pos` 总是优先生效。

### 2. 单文件分步执行

`read_header`（trigger，读取 du_id / gps_time / du_nanoseconds）：

```bash
python readroot/read_header.py /path/to/Trigger_xxx.root \
    --date 2025/01/01 --out_dir_base ../Reco_Dir
# 输出 ../Reco_Dir/2025/01/01/Trigger_xxx.yaml
```

`read_trace`（trigger，读取波形并计算区间内 ADC 峰值）：

```bash
python readroot/read_trace.py /path/to/Trigger_xxx.root \
    --left 0 --right 512 --channel XY \
    --date 2025/01/01 --out_dir_base ../Reco_Dir
# 输出 ../Reco_Dir/2025/01/01/Trigger_xxx_XY.yaml
```

`read_coreas` / `read_starshape`（模拟数据，可选 `--with-signal`、`--max-events`）：

```bash
python readroot/read_coreas.py /path/to/adc_0001-0100_L1_1.root \
    --date 2025/01/01 --out_dir_base ../Reco_Dir --with-signal
python readroot/read_starshape.py /path/to/*_stshp_*Events.root \
    --out_dir_base ../Reco_StarShape --max-events 10
```

`main.py`（对匹配文件做方向重建）：

```bash
python main.py ../Reco_Dir/2025/01/01/Trigger_xxx.yaml \
    --fig_name Trigger_xxx \
    --det-pos position/_gp65_rtksort_DU7_1029.txt
```

`main.py` 参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `matching_file` | 必填 | read 阶段产出的 YAML 文件 |
| `--fig_name` | 输入文件名（去 `.yaml`） | 图表输出前缀 |
| `--det-pos` | `position/_gp65_rtksort_DU7_1029.txt` | 探测器坐标文件（模拟布局时会被逐事件坐标取代） |
| `--with-signal` | 关 | 启用信号幅值相关处理 |
| `--force-recompute` | 关 | 忽略缓存重算 |
| `--skip-matching` | 关 | 跳过匹配阶段，直接使用已有 `_matched.yaml` 缓存 |
| `--skip-fingerprint` | 关 | 跳过指纹阶段，直接加载源 YAML |
| `--min-detectors` | `5` | 匹配阶段最少探测器数 |
| `--skip-footprint` | 关 | 跳过 PWM→SWM 之间的 footprint 形状筛选 |
| `--skip-ringcut` | 关 | 跳过 PWM→SWM 之间的 d_max ring cut |
| `--skip-offset` | 关 | 跳过时间/位置标定 |
| `--offset-csv` | `./calibration/jul24_locked_xyz_additive_utc.csv` | 标定 CSV 路径 |

## 三者（trigger / coreas / starshape）刻意差异

三类输入共享 `loop.py` 的同一套 read→`main.py`→合并 PDF 流程，但以下差异是**刻意设计**，
并非缺陷：

| 行为 | trigger | coreas / starshape |
| --- | --- | --- |
| `start` / `end` 位置参数 | **必填**（缺则报错退出） | 可选并忽略（模拟数据无 wall-clock 日期） |
| `--base-path` 含义 | 只作**日期前缀**，实际文件由 `base-path/<yyyy>/<mm>/<dd>/Trigger*.root` + 日期筛选决定 | **完全限定**分析文件集合：coreas 递归 `adc_*.root`；starshape 递归 `*[sS]tshp*.root` 或单文件 |
| 输出目录 | `--out-dir-base` 下按 `yyyy/mm/dd` 分目录 | 固定子目录 `coreas/`、`starshape/` |
| read 命令 | `read_header.py`；`--with-signal` 时改 `read_trace.py`（两条独立命令） | `read_coreas.py` / `read_starshape.py` |
| `--left` / `--right` | 透传（`read_trace` 采样窗口） | coreas 透传（其 ADC 树型有窗口概念）；**starshape 不透传**（无窗口概念） |
| `--skip-matching` / `--skip-fingerprint` | 默认 **False**（真实数据需做匹配/指纹） | 默认 **True**（模拟事件全视为信号） |
| `--skip-footprint` / `--skip-ringcut` | 默认 **False**（启用定制化 cut） | 默认 **True**（跳过；见下） |
| `--det-pos` | 不传则用 `main.py` 默认阵列文件 | 不传（坐标逐事件内嵌于 matching YAML，`main.py` 优先用逐事件坐标） |
| 时间/位置标定 | 默认启用 | 自动跳过（无真实 `gps_time`） |

**关于 sim 默认跳过 footprint/ringcut**：`main.py` 的定制化 footprint/ringcut 只读取
**静态** `--det-pos` 文件，不会使用 coreas/starshape 的逐事件坐标。sim 既不设
`--det-pos`，若启用 cut 就会拿 `main.py` 默认阵列文件比较、把**所有**模拟事件丢弃
（实测触发 `n_du < 4`）。因此 sim 默认跳过这两个 cut。注意
`--skip-footprint`/`--skip-ringcut` 是 `store_true`，**当前无 CLI 开关为 sim 重新启用**
cut；即便显式给 `--det-pos` 也不会重新启用。

## 时间/位置标定

`time_position_offset.py` 读取 `calibration/jul24_locked_xyz_additive_utc.csv`
（列：`window_start_utc, window_end_utc, DU_ID, time_offset_add_ns, x_offset_m,
y_offset_m, z_offset_m`），对 trigger 数据施加标定：

- **时间**：逐事例按 `gps_time`（`gps_time - GPS_UTC_OFFSET`，`GPS_UTC_OFFSET=18`）
  选择最接近的时间窗，对该 DU 的触发时间**正向**加 `time_offset_add_ns` 并取整；
- **坐标**：每个 DU 取所有窗口位置偏移的均值，正向加在坐标上。

`apply_calibration(matching_file, det_pos_file, offset_csv)` 产出
`<matching>.toffset.yaml` 与 `<det_pos>.toffset.txt` 并作为后续阶段的输入
（缓存签名含文件路径，标定开关切换时缓存自动失效）。

标定默认启用，且**仅对 trigger 布局生效**：coreas/starshape 的坐标逐事件内嵌于
matching YAML、`gps_time` 为合成值，`main.py` 会自动跳过标定。`--skip-offset`
可强制跳过，`--offset-csv` 可指定其它 CSV。

## 数据流与产物

以输入 `Trigger_xxx.yaml` 为例，`main.py` 依次产出（均与输入同目录）：

```
Trigger_xxx.yaml                    # read 阶段产出（含 time、du_id）
  └─ Trigger_xxx_matched.yaml       # 因果律匹配结果（--skip-matching 时跳过）
      └─ Trigger_xxx_fingerprint.yaml  # 指纹过滤结果（--skip-fingerprint 时跳过）
          ├─ <fig_prefix>_PWM_rec.png      # PWM 拟合：重建位置
          ├─ <fig_prefix>_PWM_overtime.png # PWM 拟合：随时间参数
          │   （PWM 与 SWM 之间：定制化筛选，默认启用）
          ├─ <fig_prefix>_SWM_rec.png      # SWM 拟合：重建位置
          ├─ <fig_prefix>_SWM_overtime.png # SWM 拟合：随时间参数
          └─ （第二次 PWM 刷新 <fig_prefix>_PWM_*）
```

- `<fig_prefix>` 即 `--fig_name`（默认去 `.yaml` 的文件名）。
- trigger 布局下，标定会先生成 `Trigger_xxx.toffset.yaml` / `<det_pos>.toffset.txt`。
- `--skip-footprint` / `--skip-ringcut` 均开启时跳过定制化筛选；否则
  `find_event/shower_footprint_filter.py` 写出诊断 CSV
  (`<fig_prefix>_footprint_filter_diagnostics.csv`) 与散点图
  (`<fig_prefix>_footprint_ratio_diagnostic.png`)。
- `loop.py --run` 会在每份 `main.py` 完成后把当日 PNG 合并为 `Trigger_xxx.pdf`
  （`--with-signal` 时仅合并 `*_with_signal.png`）。

## 输入文件格式

**ROOT 事件数据**

- `teventadc`（trigger）：`run_number`、`event_number`、`du_id`、`gps_time`、
  `du_nanoseconds`、`trace_0`~`trace_3`（F/X/Y/Z）。
- `tadc`（coreas）：`run_number`、`event_number`、`du_id`、`du_nanoseconds`、
  `trace_ch`（形状 `(n_du, 3, n_time)`，X/Y/Z；无 F 口，F 通道回退为 X）。
  无 `gps_time`，由运行目录时间戳合成单调时间轴；配套 `shower`/`run`/`showersim`/
  `runefieldsim` companion 文件提供 `truth` / `efield_config` / 逐事件 `det_pos`。
- StarShape（starshape）：`SimuCollection/SimShowerBranch`（shower truth）、
  `EfieldCollection/SimEfieldBranch`（`Detectors_det_id`、`Detectors_det_pos_shc`、
  `Detectors_t_0` 逐天线布局与到达时间）。无 `gps_time`/`du_id`/`du_nanoseconds`，
  `t_0` 为浮点 ns；文件名不含日期，逐事件内嵌 `det_pos`。

> 约定：每个 DU 在一次事件内最多触发一次，因此事件内 `du_id` 唯一，
> 时间映射为一一对应的标量 `{du: t}`。coreas/starshape 的 YAML 另含逐事件
> `det_pos`（`{du_id: [x,y,z]}`），`main.py` 据此判别布局并跳过标定。

**探测器坐标文件**（`--det-pos` / `--det-pos` 未指定时的默认文件）
每行一个探测器，`#` 开头为注释：

```
ID x y z
1032  -265.726  -522.760  7.422
1029  -546.161  -1027.024 12.317
```

`position/du_coords_CoreasDC2.txt` 为逗号分隔版本（CoREAS DC2 布局）。
`position/*_du_pair_theoretical.csv` 为 `du_pair_theoretical.py` 生成的
DU 对理论时间差缓存（字段 `du_a, du_b, distance_m, theoretical_delta_ns`）。

## 参考

- 代码内各模块的 docstring 记录了三类 TTree / collection 的 schema 差异与算法细节。
- 阈值、通道选择等会影响匹配/拟合结果，请结合具体数据调整 `--left`/`--right`、
  `--channel`、`--det-pos`、`--min-detectors` 等参数。
