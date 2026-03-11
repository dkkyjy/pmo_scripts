# PMO Scripts (GRAND)

A Trigger ROOT data reconstruction pipeline for GRAND/PMO.

This project batch-processes `Trigger*.root` files, generates intermediate YAML files, performs event matching and direction reconstruction (PWM/SWM), and produces statistics and figures.

> [!IMPORTANT]
> `loop.py` runs in dry-run mode by default. It only prints planned commands. Add `--run` to execute.

## Overview

- **Batch orchestration**: Scans `--base-path/yyyy/mm/dd` over a date range.
- **Two reader modes**: Time mode (`read_header.py`) and signal-amplitude mode (`read_trace.py`).
- **Stage-level cache**: Outputs `*_matched.yaml`, `*_PWM.yaml`, `*_SWM.yaml` for resumable runs.
- **Post-processing**: Supports daily merge, trigger statistics, and coverage test scripts.

## Core pipeline

```text
loop.py
  ├─ read_header/read_header.py (time mode)
  ├─ read_header/read_trace.py (--with-signal)
  ├─ main.py
  │    ├─ find_event/matching_times.py
  │    ├─ find_event/estimation.py
  │    └─ find_event/plotting.py
  └─ merge.py (optional, daily aggregation)
```

## Project structure

```text
.
├── loop.py
├── main.py
├── merge.py
├── merge_trace.py
├── stats/
│   ├── common.py
│   ├── stats_trigger.py
│   ├── stats_lookback.py
│   └── stats_du_pairs.py
├── read_header/
│   ├── read_header.py
│   └── read_trace.py
├── find_event/
├── scripts/
│   └── run_tests.sh
├── tests/
└── docs/
```

## Environment setup

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

### 1) Safe trial run (print commands only)

```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 \
  --base-path /path/to/TD \
  --out-dir-base ../Reco_Dir \
  --limit 1 --jobs 1
```

### 2) Actual reconstruction run

```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 \
  --base-path /path/to/TD \
  --out-dir-base ../Reco_Dir \
  --run --jobs 4
```

### 3) Signal-amplitude mode

```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 \
  --base-path /path/to/TD \
  --out-dir-base ../Reco_Dir \
  --run --with-signal --left 0 --right 512 --channel X
```

> [!TIP]
> Start with `--limit 1 --jobs 1` to validate outputs, then scale up.

## Common commands

### Single-file debug

```bash
python main.py ../Reco_Dir/2026/02/14/Trigger_xxx.yaml \
  --fig_name Trigger_xxx \
  --det-pos _gp65_rtksort.txt
```

### Daily merge

```bash
python merge.py 2026/02/14 -o ../Reco_Dir
```

### Statistics (examples)

```bash
python -m stats.stats_trigger ../Reco_Dir/Trigger_20260214_merged.yaml --avg-window=60 --no-plot
python -m stats.stats_lookback ../Reco_Dir/Trigger_20260214_merged.yaml --lookback 10
```

### Run all tests with coverage

```bash
./scripts/run_tests.sh
```

## Main outputs

Typical outputs for one input file:

- `Trigger_xxx.yaml` (reader-stage output)
- `Trigger_xxx_matched.yaml`
- `Trigger_xxx_PWM.yaml`
- `Trigger_xxx_SWM.yaml`
- `Trigger_xxx*.png`

Daily merge output:

- `Trigger_yyyymmdd_merged.yaml`

## Configuration and conventions

### Logging

```bash
export LOG_LEVEL=INFO
export LOG_FILE=loop.log
```

### Supported datetime formats

`loop.py` accepts:

- `YYYY-MM-DDTHH:MM:SS`
- `YYYYMMDDHHMMSS`
- `YYYY-MM-DDHHMM`
- `YYYY/MM/DDHHMM`

> [!NOTE]
> `main.py` has an explicit `exit()` right after SWM plotting; the background-rejection branch below is unreachable by default.

> [!IMPORTANT]
> Do not change cache suffix conventions (`*_matched.yaml`, `*_PWM.yaml`, `*_SWM.yaml`), as downstream stages depend on them.

## Troubleshooting

- **No input files found**: Check `--base-path/yyyy/mm/dd` layout and `Trigger*.root` naming.
- **Zero events after filtering**: This can be normal; matching uses strict consistency filters.
- **Missing figures or PDF**: Check event validity, dependencies, and output path permissions.
- **Missing detector position file**: Pass `main.py --det-pos <path>`.

## More docs

- `docs/loop.md`
- `docs/main.md`
- `docs/merge.md`
- `docs/merge_trace.md`
- `docs/stats_trigger.md`
