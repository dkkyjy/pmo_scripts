# PMO Scripts (GRAND)

Event reconstruction pipeline for GRAND/PMO Trigger ROOT files.

This repository reads `Trigger*.root` files, builds per-file matching YAMLs, reconstructs event directions with PWM/SWM models, and saves diagnostic figures for analysis.

> [!IMPORTANT]
> `loop.py` is **dry-run by default**. It only prints commands unless you pass `--run`.

## Overview

Core pipeline:

1. `loop.py` scans date folders (`yyyy/mm/dd`) under `--base-path` and orchestrates jobs.
2. `read_header/_read_header.py` converts ROOT header timing to `Trigger*.yaml`.
3. Optional signal mode uses `read_header/read_header_alltraceADCsquare.py` to produce `*_F/X/Y/Z/XY.yaml`.
4. `main.py` runs matching + reconstruction (`find_event/*`) and writes cached stage outputs:
	 - `_matched.yaml`
	 - `_PWM.yaml`
	 - `_SWM.yaml`
	 - plot images (`*.png`)
5. `merge.py` merges one day of outputs into `Trigger_yyyymmdd_{matched|PWM|SWM}.yaml`.

## Repository Structure

```text
.
├── loop.py                 # Date-range orchestration, multiprocessing, dry-run/run switch
├── main.py                 # Single matching YAML reconstruction entrypoint
├── merge.py                # Daily merged YAML generator
├── logger_config.py        # loguru setup (LOG_LEVEL / LOG_FILE)
├── read_header/            # ROOT -> initial YAML converters
└── find_event/             # Filtering, fitting, plotting, utilities
```

## Prerequisites

- Python 3.9+
- Access to ROOT Trigger files readable by `uproot`
- Detector position file (default `_gp65_rtksort.txt`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Dry-run a small window (safe)

```bash
python loop.py 2025-11-01T00:00:00 2025-11-01T23:59:59 \
	--base-path /path/to/ROOT \
	--out-dir-base ../Reco_Dir \
	--limit 1 --jobs 1
```

### 2) Execute reconstruction

```bash
python loop.py 2025-11-01T00:00:00 2025-11-01T23:59:59 \
	--base-path /path/to/ROOT \
	--out-dir-base ../Reco_Dir \
	--run --jobs 4
```

### 3) Run signal-amplitude mode

```bash
python loop.py 2025-11-01T00:00:00 2025-11-01T23:59:59 \
	--base-path /path/to/ROOT \
	--out-dir-base ../Reco_Dir \
	--run --with-signal --left 0 --right 512 --channel X
```

## Common Workflows

### Single-file debug

```bash
python main.py ../Reco_Dir/2025/11/01/Trigger_xxx.yaml \
	--fig_name Trigger_xxx \
	--det-pos _gp65_rtksort.txt
```

### Merge one day outputs

```bash
python merge.py 2025/11/01 -o ../Reco_Dir
```

Merged files are written as:

- `../Reco_Dir/Trigger_20251101_matched.yaml`
- `../Reco_Dir/Trigger_20251101_PWM.yaml`
- `../Reco_Dir/Trigger_20251101_SWM.yaml`

## Configuration

Logging is configured through environment variables:

```bash
export LOG_LEVEL=DEBUG
export LOG_FILE=logs/loop.log
```

Supported datetime input formats for `loop.py` include:

- `YYYY-MM-DDTHH:MM:SS`
- `YYYYMMDDHHMMSS`
- `YYYY-MM-DDHHMM`
- `YYYY/MM/DDHHMM`

## Notes and Caveats

> [!NOTE]
> In `main.py`, code after SWM plotting is currently unreachable because of an explicit `exit()`.

> [!TIP]
> Keep output filename conventions unchanged (`*_matched.yaml`, `*_PWM.yaml`, `*_SWM.yaml`), because downstream stages rely on these exact patterns.

## Troubleshooting

- **No files processed**: verify input directory layout is `--base-path/yyyy/mm/dd` and filenames contain both `Trigger` and `.root`.
- **No events after filtering**: this can be expected; `find_event/matching_times.py` applies strict speed-of-light consistency filtering.
- **Missing detector file**: pass `--det-pos` to `main.py` if `_gp65_rtksort.txt` is not in the working directory.
