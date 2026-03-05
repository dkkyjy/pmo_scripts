# PMO Scripts (GRAND)

面向 GRAND/PMO 的 Trigger ROOT 数据重建流水线。

该项目用于批量处理 `Trigger*.root`，生成中间 YAML，执行事件匹配与方向重建（PWM/SWM），并产出统计和图像结果。

> [!IMPORTANT]
> `loop.py` 默认是 dry-run，只打印计划命令；加上 `--run` 才会真正执行。

## 概览

- **批处理调度**：按日期范围扫描 `--base-path/yyyy/mm/dd`。
- **两种读取模式**：常规时间模式（`read_header.py`）和信号幅值模式（`read_trace.py`）。
- **分阶段缓存**：输出 `*_matched.yaml`、`*_PWM.yaml`、`*_SWM.yaml`，便于断点续跑。
- **后处理能力**：支持按天合并、触发统计与覆盖率测试脚本。

## 核心流程

```text
loop.py
  ├─ read_header/read_header.py（常规时间模式）
  ├─ read_header/read_trace.py（--with-signal）
  ├─ main.py
  │    ├─ find_event/matching_times.py
  │    ├─ find_event/estimation.py
  │    └─ find_event/plotting.py
  └─ merge.py（可选，按天汇总）
```

## 目录结构

```text
.
├── loop.py
├── main.py
├── merge.py
├── stats_trigger.py
├── stats_lookback.py
├── stats_du_ns.py
├── read_header/
│   ├── read_header.py
│   └── read_trace.py
├── find_event/
├── scripts/
│   └── run_tests.sh
├── tests/
└── docs/
```

## 环境准备

- Python 3.8+
- 依赖安装：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1) 安全试跑（仅打印命令）

```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 \
  --base-path /path/to/TD \
  --out-dir-base ../Reco_Dir \
  --limit 1 --jobs 1
```

### 2) 实际执行重建

```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 \
  --base-path /path/to/TD \
  --out-dir-base ../Reco_Dir \
  --run --jobs 4
```

### 3) 信号幅值模式

```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 \
  --base-path /path/to/TD \
  --out-dir-base ../Reco_Dir \
  --run --with-signal --left 0 --right 512 --channel X
```

> [!TIP]
> 建议先用 `--limit 1 --jobs 1` 验证输出，再放大任务规模。

## 常用命令

### 单文件调试

```bash
python main.py ../Reco_Dir/2026/02/14/Trigger_xxx.yaml \
  --fig_name Trigger_xxx \
  --det-pos _gp65_rtksort.txt
```

### 日级合并

```bash
python merge.py 2026/02/14 -o ../Reco_Dir
```

### 统计分析（示例）

```bash
python stats_trigger.py ../Reco_Dir/Trigger_20260214_merged.yaml --avg-window=60 --no-plot
python stats_lookback.py ../Reco_Dir/Trigger_20260214_merged.yaml --lookback 10
```

### 运行全部测试并统计覆盖率

```bash
./scripts/run_tests.sh
```

## 主要输出

单个输入文件通常会生成：

- `Trigger_xxx.yaml`（读取阶段输出）
- `Trigger_xxx_matched.yaml`
- `Trigger_xxx_PWM.yaml`
- `Trigger_xxx_SWM.yaml`
- `Trigger_xxx*.png`

按天合并可生成：

- `Trigger_yyyymmdd_merged.yaml`

## 配置与约定

### 日志

```bash
export LOG_LEVEL=INFO
export LOG_FILE=loop.log
```

### 时间参数格式

`loop.py` 支持：

- `YYYY-MM-DDTHH:MM:SS`
- `YYYYMMDDHHMMSS`
- `YYYY-MM-DDHHMM`
- `YYYY/MM/DDHHMM`

> [!NOTE]
> `main.py` 在 SWM 绘图后有显式 `exit()`，其后的 background rejection 分支默认不可达。

> [!IMPORTANT]
> 不要修改缓存后缀规则（`*_matched.yaml`、`*_PWM.yaml`、`*_SWM.yaml`），下游流程依赖这些命名。

## 故障排查

- **找不到输入文件**：检查 `--base-path/yyyy/mm/dd` 目录格式和 `Trigger*.root` 文件名。
- **过滤后事件数为 0**：可能是正常现象，匹配阶段使用了较严格的一致性筛选。
- **图像或 PDF 未生成**：确认事件有效性、依赖安装和输出路径权限。
- **探测器位置文件缺失**：使用 `main.py --det-pos <path>` 指定文件。

## 更多文档

- `docs/loop.md`
- `docs/main.md`
- `docs/merge.md`
- `docs/stats_trigger.md`
