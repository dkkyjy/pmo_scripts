# run_daily_pipeline.py — 单日数据处理完整流水线

*2026-06-20T09:30:00Z by Showboat 0.6.1*
<!-- showboat-id: 611e41c7-c69e-4552-9f91-074a36028aaa -->

## 文件职责

`run_daily_pipeline.py` 是 GRAND Dunhuang 实验**单日数据处理的主调度脚本**。它按顺序编排 5 个处理阶段（含 Step 0 预检查），将原始 ROOT 文件全天一次性重建后，进行 signal 信息补全 + slope 事例筛选和每日数据合并，最终输出合并后的 YAML 文件及候选 YAML。

流水线通过 `SKIP_*` 环境变量和 `DRY_RUN` 模式实现灵活的跳过和试运行控制。重建、enrich+filter 阶段通过统一的 `--jobs` 参数控制并行核数。

## CLI 接口

```bash
python scripts/run_daily_pipeline.py 20260617
python scripts/run_daily_pipeline.py 2026-06-17
python scripts/run_daily_pipeline.py 2026/06/17
python scripts/run_daily_pipeline.py 20260617 --jobs 80
```

### 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `date` | 日期（YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD） | 必填 |
| `--base-path` | TD ROOT 文件根目录 | 环境变量 `BASE_PATH` 或 `/mnt/sdb2/users/m/mapx/DunhuangData/ROOTFile/TD` |
| `--reco-dir` | 重建输出目录 | 环境变量 `RECO_DIR` 或 `../Reco_Dir`（相对仓库根目录） |
| `--python` | Python 解释器 | 环境变量 `PYTHON` 或 `python` |
| `--jobs` | 并行核数（Step 2/3/4 统一使用） | 环境变量 `JOBS` 或 `40` |

### 环境变量

| 变量 | 作用 |
|---|---|
| `SKIP_INVENTORY=1` | 跳过文件清点 |
| `SKIP_LOOP=1` | 跳过 loop.py 重建 |
| `SKIP_ENRICH=1` | 跳过 signal 信息补全 + slope 候选筛选 |
| `SKIP_MERGE=1` | 跳过数据合并 |
| `DRY_RUN=1` | 只打印命令不执行 |
| `JOBS=N` | 并行核数（与 `--jobs` 等效） |

## 函数说明

### `parse_date(raw: str) -> str`
统一日期格式。支持 `YYYYMMDD`、`YYYY-MM-DD`、`YYYY/MM/DD` 三种输入，输出统一为 `YYYYMMDD`。格式不匹配时抛出 `ValueError`。

### `resolve_repo_root() -> Path`
返回仓库根目录（`scripts/` 的父目录），用于动态解析子脚本的绝对路径。

### `run_cmd(desc, cmd, *, dry_run, date_label) -> int`
命令执行包装器。打印命令描述和完整命令行；`dry_run=True` 时跳过执行直接返回 0；否则通过 `subprocess.run` 执行并返回退出码。失败时打印错误信息。

### `main() -> None`
流水线主入口。完整流程见下一节。

## 执行流程

### 初始化阶段

1. 调用 `parse_date()` 解析命令行日期参数
2. 从日期中拆分 `year`、`month`、`day`，构建 `yyyy/mm/dd` 路径格式
3. 解析 `repo_root`、`reco_dir`、`td_day_dir`（`BASE_PATH/yyyy/mm/dd`）、`reco_day_dir`（`reco_dir/yyyy/mm/dd`）
4. 读取环境变量确定 `dry_run` 和各 `SKIP_*` 状态
5. 打印配置摘要（TD dir、Reco dir、Day subdir、Python、Jobs、Dry run）

### Step 0：检查 TD 目录

- 检查 `td_day_dir`（即 `BASE_PATH/yyyy/mm/dd`）是否存在
- 不存在 → `sys.exit(1)`，提示数据可能尚未到达
- 存在 → 统计 `Trigger*.root` 和 `Calibration*.root` 文件数，计算总大小（`_human_size()` 辅助格式化）
- 若 `trigger_count == 0` → `sys.exit(0)`（仅有 Calibration 或无数据，流水线无法继续）

### Step 1：文件清点（可跳过）

调用 `stats_root_file_inventory.py`，传入 `--base-path`（TD 根目录的父目录）、`--date`（`YYYY-MM-DD` 格式）、`--output-dir` 和 `--daily-summary --hourly-summary` 标志。输出写入 `reco_dir/root_file_inventory/yyyy/mm/dd/`。失败时 `sys.exit(rc)`。

### Step 2：loop.py 当天完整重建（可跳过，并行）

直接调用 `loop.py`，传入 `YYYY-MM-DDTHH:MM:SS` 格式的起止时间（当天 00:00:00 ~ 23:59:59），加上 `--base-path`、`--out-dir-base`、`--run`、`--jobs`。`loop.py` 内部使用 `multiprocessing.Pool` 并行处理全天所有 `Trigger*.root` 文件。失败时 `sys.exit(rc)`。

### Step 3：signal 信息补全 + slope 候选筛选（可跳过，并行）

调用 `run_enrich_and_filter.py`，传入 `--reco-dir`、`--date`（`YYYYMMDD`）、`--overwrite`、`--min-du-count 6`、`--jobs`。该脚本扫描 `reco_dir/yyyy/mm/dd/` 下所有 `Trigger_*_SWM.yaml`，以指定并发数调用 `enrich_and_filter_swm.py` 为每个 SWM 文件补全 XY signal 信息并筛选 slope>0 候选事例，输出 `_SWM_with_signal.yaml` 和 `_candidates.yaml`。失败时 `sys.exit(rc)`。

### Step 4：数据合并（可跳过）

调用 `run_merge_date.py`，传入 `--target all`、`--out-dir-base`、`--run` 和日期（`YYYYMMDD`）。合并 header、trace、results 三类 YAML 为每日汇总文件（`Trigger_{yyyymmdd}_RUN*_merged*.yaml`）。失败时 `sys.exit(rc)`。

### 完成阶段

- 列出 `reco_dir` 下的 `Trigger_{yyyymmdd}_RUN*_merged*.yaml` 文件及其大小
- 检查并显示 `Trigger_{yyyymmdd}_RUN*_candidates.yaml` 文件数量

## 分支逻辑

### 试运行模式（`DRY_RUN=1`）
所有 `run_cmd()` 调用只打印命令行不执行，始终返回 `rc=0`，流水线完整走完但无副作用。

### 跳过模式（`SKIP_*=1`）
每个 Step 有独立的 `SKIP_*` 环境变量。设为 1 时跳过该步骤，打印 `[SKIP]` 标记，流水线继续执行后续步骤。

### 早期退出
- Step 0 中 TD 目录不存在 → `exit(1)`
- Step 0 中无 Trigger 文件 → `exit(0)`
- 任一步骤 `run_cmd()` 返回非零 → `exit(rc)`

### 依赖关系
流水线严格串行，后续步骤依赖前一步的输出文件：
- Step 3 依赖 Step 2 产生的 `Trigger_*_SWM.yaml`
- Step 4 合并 Step 2-3 的所有输出

## 用法示例

### 试运行（只打印命令）

```bash
DRY_RUN=1 python scripts/run_daily_pipeline.py 20260617
```

### 正式运行（默认 40 核）

```bash
python scripts/run_daily_pipeline.py 20260617
```

### 指定并行核数

```bash
python scripts/run_daily_pipeline.py 20260617 --jobs 80
```

### 部分运行（跳过已完成的步骤）

```bash
SKIP_INVENTORY=1 SKIP_LOOP=1 python scripts/run_daily_pipeline.py 20260617
```

### 指定自定义路径

```bash
python scripts/run_daily_pipeline.py 2026-06-17 \
  --base-path /data/TD \
  --reco-dir /data/Reco_Dir \
  --python python3 \
  --jobs 60
```

## 线性工作原理总览

1. **解析日期参数**：`parse_date()` 将 YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD 统一为 YYYYMMDD，提取 year/month/day
2. **解析路径**：计算 `repo_root`、`reco_dir`、`td_day_dir`（`BASE_PATH/yyyy/mm/dd`）、`reco_day_dir`（`reco_dir/yyyy/mm/dd`）
3. **读取环境变量**：`DRY_RUN`、`SKIP_INVENTORY`、`SKIP_LOOP`、`SKIP_ENRICH`、`SKIP_MERGE`
4. **打印配置摘要**：TD dir、Reco dir、Day subdir、Python、Jobs、Dry run 状态
5. **Step 0 — TD 目录检查**：验证 `td_day_dir` 存在且包含 Trigger 文件，统计文件数和总大小；无数据则退出（`exit 0`），目录不存在则 `exit 1`
6. **Step 1 — 文件清点**：`stats_root_file_inventory.py --base-path <parent> --date YYYY-MM-DD --output-dir <dir> --daily-summary --hourly-summary`（可通过 `SKIP_INVENTORY=1` 跳过）
7. **Step 2 — 全天重建（并行）**：`loop.py YYYY-MM-DDT00:00:00 YYYY-MM-DDT23:59:59 --base-path <path> --out-dir-base <dir> --run --jobs <N>`，loop.py 内部 Pool 并行处理全天 Trigger 文件（可通过 `SKIP_LOOP=1` 跳过）
8. **Step 3 — signal 补全 + slope 候选筛选（并行）**：`run_enrich_and_filter.py --reco-dir <dir> --date YYYYMMDD --overwrite --min-du-count 6 --jobs <N>`，以 N 并发为当天所有 SWM 文件补全 XY signal 并筛选 slope>0 候选（可通过 `SKIP_ENRICH=1` 跳过）
9. **Step 4 — 数据合并**：`run_merge_date.py --target all --out-dir-base <reco_dir> --run YYYYMMDD`（可通过 `SKIP_MERGE=1` 跳过）
10. **完成报告**：列出 `Trigger_{yyyymmdd}_RUN*_merged*.yaml` 文件及大小，检查 `Trigger_{yyyymmdd}_RUN*_candidates.yaml`
11. **早期退出路径**：TD 目录不存在（`exit 1`）、无 Trigger 文件（`exit 0`）、任一步骤失败（`exit rc`）
12. **DRY_RUN 模式**：所有 `run_cmd()` 只打印命令不执行，流水线完整走完但无副作用
13. **并行控制**：`--jobs` 参数（默认 40）统一传给 Step 2/3，也可通过 `JOBS` 环境变量设置
