# pmo_scripts

敦煌 DU 阵列数据分析脚本：从 ROOT 文件读取事件数据，完成时间戳/时间的提取、
因果律（causality）匹配、时间差指纹过滤，并用平面波模型（PWM）与球面波模型
（SWM）拟合出射方向，最后输出拟合参数与可视化图表。

按数据来源分两种输入布局（由 `loop.py --reader` 选择）：

| reader | 输入文件 | TTree | 说明 |
| --- | --- | --- | --- |
| `trigger`（默认） | `base-path/<date>/Trigger*.root` | `teventadc` | 真实触发数据，含 `gps_time`、`trace_0..3`（F/X/Y/Z） |
| `coreas` | 递归扫描 `base-path/**/adc_*.root` | `tadc` | CoREAS/Dunhuang 模拟，`trace_ch` 形状 `(n_du, 3, n_time)`（X/Y/Z），无 `gps_time` |

TTree 类型由 `readroot/common.py` 的 `find_event_tree_key` 自动识别，两个 reader
共用同一套读取逻辑。

## 目录结构

```
.
├── loop.py                     # 批量驱动：按时间范围生成并执行 read + main 命令
├── main.py                     # 单文件分析主入口：matching → fingerprint → PWM → SWM
├── logger_config.py            # 统一日志配置（loguru）
├── readroot/
│   ├── common.py               # TTree 识别、时间映射、YAML 读写等公共函数
│   ├── read_header.py          # 读取 du_id / gps_time / du_nanoseconds → <file>.yaml
│   └── read_trace.py           # 读取波形并计算 [left:right] 内 ADC 峰值 → <file>_<ch>.yaml
├── find_event/
│   ├── io.py                   # 读取 DU 坐标文件
│   ├── matching_times_graph.py # 因果律匹配（causality matching）
│   ├── time_difference_fingerprint.py  # 时间差指纹过滤（剔除固定源）
│   ├── plane_wave_model_gradient.py    # 平面波模型（PWM）拟合
│   ├── spherical_wave_model_nopenal.py # 球面波模型（SWM）拟合
│   ├── time_offset.py          # DU 时间偏移配置
│   ├── estimation.py / plotting.py / utils.py  # 估计、绘图与通用工具
│   └── ...
├── du_pair_theoretical.py      # 由坐标文件生成/加载 DU 对理论时间差
├── position/                   # 探测器坐标文件
│   ├── _gp65_rtksort_DU7_1029.txt  # trigger 默认探测器坐标（main.py 内默认）
│   ├── _gp65_rtksort_2002_DU7.txt  # trigger 备用坐标
│   ├── _gp65_rtksort.txt
│   └── du_coords_CoreasDC2.txt     # CoREAS DC2 坐标（逗号分隔）
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：`numpy`、`scipy`、`matplotlib`、`scienceplots`、`loguru`、`uproot`、
`pyyaml`。绘图到 PDF 合并需要 `Pillow`（`loop.py` 中可选，缺失时跳过 PDF 生成）。

## 使用

### 1. 批量驱动 `loop.py`

默认只打印将要执行的命令，加 `--run` 才会真正执行：

```bash
# 只预览命令
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59

# 真正执行（trigger 数据）
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59 --run --jobs 4

# 处理 CoREAS 模拟数据
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59 \
    --reader coreas --base-path /path/to/sim \
    --out-dir-base ../Reco_Dir --run
```

`loop.py` 对每个 ROOT 文件依次执行：

1. `readroot/read_header.py`（默认）或 `readroot/read_trace.py`（`--with-signal` 时）；
2. `main.py`；
3. 把当日产生的 PNG 合并为同名 PDF（需 Pillow）。

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `start` `end` | 必填 | 时间范围，格式 `YYYY-MM-DDThh:mm:ss` |
| `--reader` | `trigger` | 输入布局：`trigger` / `coreas` |
| `--base-path` | `/mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD` | 输入根目录 |
| `--out-dir-base` | `../Reco_Dir` | 输出根目录（按 `yyyy/mm/dd` 组织） |
| `--run` | 关 | 实际执行，否则仅打印 |
| `--jobs` | `1` | 并行 worker 数 |
| `--limit` | `0` | 限制处理文件数（0=全部） |
| `--skip-read` | 关 | 跳过 read 阶段 |
| `--only-read` | 关 | 只跑 read 阶段，不跑 `main.py` |
| `--with-signal` | 关 | 用 `read_trace` 读取波形幅值，并让 `main.py` 处理信号 |
| `--left` / `--right` | `0` / `512` | `read_trace` 的采样区间 |
| `--channel` | `XY` | 匹配文件通道后缀：`F`/`X`/`Y`/`Z`/`XY` |
| `--det-pos` | 随 `--reader` 推导 | 探测器坐标文件：coreas 默认 `position/du_coords_CoreasDC2.txt`，trigger 默认 `position/_gp65_rtksort_DU7_1029.txt`（`main.py` 内默认） |
| `--skip-matching` / `--no-skip-matching` | 见下 | 跳过因果律匹配阶段 |
| `--skip-fingerprint` / `--no-skip-fingerprint` | 见下 | 跳过指纹过滤阶段 |
| `--force-recompute` | 关 | 忽略已有 `_matched`/`_PWM`/`_SWM` 缓存并重算 |

`--skip-matching` 与 `--skip-fingerprint` 默认值随 `--reader` 推导：
**coreas 默认跳过**（模拟事件本身就是信号），**trigger 默认不跳过**。
显式传入 `--skip-matching`/`--no-skip-matching`（指纹同理）可覆盖该默认。

> 说明：这两个阶段是面向真实触发数据设计的——因果律匹配用于剔除偶然符合，
> 指纹过滤用于剔除固定重复源。CoREAS 模拟数据无此类噪声，默认跳过可省去开销。

`--det-pos` 默认值同样随 `--reader` 推导：**coreas 默认 `position/du_coords_CoreasDC2.txt`**、
**trigger 默认 `position/_gp65_rtksort_DU7_1029.txt`**（后者为 `main.py` 的默认）。
CoREAS 的 DU 编号从 0 开始、阵列布局与真实阵列不同，用错坐标会导致
`missing detector position for DU xxx` 并跳过事件。显式传入 `--det-pos` 可覆盖该默认。

### 2. 单文件分步执行

`read_header`（读取 du_id / gps_time / du_nanoseconds）：

```bash
python readroot/read_header.py /path/to/Trigger_xxx.root \
    --date 2025/01/01 --out_dir_base ../Reco_Dir
# 输出 ../Reco_Dir/2025/01/01/Trigger_xxx.yaml
```

`read_trace`（读取波形并计算区间内 ADC 峰值）：

```bash
python readroot/read_trace.py /path/to/Trigger_xxx.root \
    --left 0 --right 512 --channel XY \
    --date 2025/01/01 --out_dir_base ../Reco_Dir
# 输出 ../Reco_Dir/2025/01/01/Trigger_xxx_XY.yaml
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
| `--det-pos` | `position/_gp65_rtksort_DU7_1029.txt` | 探测器坐标文件 |
| `--with-signal` | 关 | 启用信号幅值相关处理 |
| `--skip-matching` | 关 | 跳过匹配阶段，直接使用已有 `_matched.yaml` 缓存 |
| `--skip-fingerprint` | 关 | 跳过指纹阶段，直接加载源 YAML |
| `--force-recompute` | 关 | 忽略缓存重算 |

## 数据流与产物

以输入 `Trigger_xxx.yaml` 为例，`main.py` 依次产出（均与输入同目录）：

```
Trigger_xxx.yaml                    # read_header 产出（含 time、du_id）
  └─ Trigger_xxx_matched.yaml       # 因果律匹配结果（--skip-matching 时跳过）
      └─ Trigger_xxx_fingerprint.yaml  # 指纹过滤结果（--skip-fingerprint 时跳过）
          ├─ <fig_prefix>_PWM.png   # 平面波模型拟合图
          └─ <fig_prefix>_SWM.png   # 球面波模型拟合图
```

`loop.py --run` 会在每份 `main.py` 完成后把当日 PNG 合并为 `Trigger_xxx.pdf`。

## 输入文件格式

**ROOT 事件数据**

- `teventadc`（trigger）：`run_number`、`event_number`、`du_id`、`gps_time`、
  `du_nanoseconds`、`trace_0`~`trace_3`（F/X/Y/Z）。
- `tadc`（coreas）：`run_number`、`event_number`、`du_id`、`du_nanoseconds`、
  `trace_ch`（形状 `(n_du, 3, n_time)`，X/Y/Z；无 F 口，F 通道回退为 X）。
  无 `gps_time`，由运行目录时间戳合成单调时间轴。

> 约定：每个 DU 在一次事件内最多触发一次，因此事件内 `du_id` 唯一，
> 时间映射为一一对应的标量 `{du: t}`。

**探测器坐标文件**（`--det-pos` / `--det-pos` 未指定时的默认文件）
每行一个探测器，`#` 开头为注释：

```
ID x y z
1032  -265.726  -522.760  7.422
1029  -546.161  -1027.024 12.317
```

`position/du_coords_CoreasDC2.txt` 为逗号分隔版本（CoREAS DC2 布局）。

## 参考

- 代码内各模块的 docstring 记录了三类 TTree 的 schema 差异与算法细节。
- 阈值、通道选择等会影响匹配/拟合结果，请结合具体数据调整 `--left`/`--right`、
  `--channel`、`--det-pos` 等参数。
