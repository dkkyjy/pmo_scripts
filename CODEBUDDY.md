# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Commands

### Safe dry-run (print commands only)
```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 --base-path /path/to/TD --out-dir-base ../Reco_Dir --limit 1 --jobs 1
```
Always start with dry-run before adding `--run`. Use `--limit 1 --jobs 1` for validation.

### Actual reconstruction
```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 --base-path /path/to/TD --out-dir-base ../Reco_Dir --run --jobs 4
```

### Signal-amplitude mode (trace-based)
```bash
python loop.py 2026-02-14T00:00:00 2026-02-14T02:00:00 --base-path /path/to/TD --out-dir-base ../Reco_Dir --run --with-signal --left 0 --right 512 --channel X
```

### Single-file debug / stage-specific execution
```bash
python main.py ../Reco_Dir/2026/02/14/Trigger_xxx.yaml --fig_name Trigger_xxx --det-pos _gp65_rtksort.txt
python main.py ../Reco_Dir/2026/02/14/Trigger_xxx.yaml --run-matching
python main.py ../Reco_Dir/2026/02/14/Trigger_xxx.yaml --run-pwm
python main.py ../Reco_Dir/2026/02/14/Trigger_xxx.yaml --run-swm --with-signal
```
When none of `--run-matching`, `--run-pwm`, `--run-swm` is given, `main.py` runs the full chain.

### Daily merge
```bash
python -m merge.merge_header 2026/02/14 -o ../Reco_Dir
python -m merge.merge_trace 2026/02/14 -o ../Reco_Dir
python -m merge.merge_results 2026/02/14 -o ../Reco_Dir
```

### Date-range merge orchestration (dry-run first)
```bash
python scripts/run_merge_date.py 2026-02-14 2026-02-16 --out-dir-base ../Reco_Dir
python scripts/run_merge_date.py 2026-02-14 2026-02-16 --target both --out-dir-base ../Reco_Dir --run
```

### Statistics
```bash
python -m stats.stats_trigger ../Reco_Dir/Trigger_20260214_merged.yaml --avg-window=60 --no-plot
python -m stats.stats_lookback ../Reco_Dir/Trigger_20260214_merged.yaml --lookback 10
python -m stats.stats_du_pairs ../Reco_Dir/Trigger_20260214_merged.yaml
```

### Run all tests with coverage
```bash
./scripts/run_tests.sh
```
This runs `pytest` with `--cov` across all major modules and writes annotated coverage reports to `cov_annotate/`.

### Slurm submission (HPC cluster)
```bash
scripts/submit_run_main_date.sh -d 20260501 --end-date 20260503 --dry-run
scripts/submit_run_main_date.sh -d 20260501 --end-date 20260503 --run
```

## Architecture

### Project purpose
This is the **GRAND (Giant Radio Array for Neutrino Detection)** Dunhuang site data processing pipeline. It batch-processes `Trigger*.root` files from cosmic-ray detectors, performs causal-consistency matching to identify real physics events, and reconstructs incoming shower directions using both plane-wave and spherical-wave models.

### High-level data flow

```
loop.py (orchestrator, scans by date range)
  ├── readroot/read_header.py  →  extracts event timestamps from ROOT TTree
  │   (or readroot/read_trace.py for signal-amplitude mode)
  │   Output: Trigger_xxx.yaml  (per ROOT file)
  │
  └── main.py  (single-file reconstruction pipeline, called as subprocess)
        ├── Stage 1: find_event/matching_times_graph.py  →  causality filter
        │     Uses du_pair_theoretical.py for DU-pair max time differences.
        │     Output: _matched.yaml
        ├── Stage 2: find_event/time_difference_fingerprint.py  →  fixed-source dedup
        │     Output: _fingerprint.yaml
        ├── Stage 3: find_event/plane_wave_model_gradient.py  →  PWM fit
        │     Output: _PWM.yaml + PNG charts
        └── Stage 4: find_event/spherical_wave_model_nopenal.py  →  SWM fit
              Output: _SWM.yaml + PNG charts

Post-processing (optional):
  merge/ package  →  daily aggregation of per-file YAMLs into merged YAMLs
  stats/ package  →  trigger-rate stats, DU-pair stats, lookback analysis
```

### Module roles

**`loop.py`** — The entry point. It scans `--base-path/yyyy/mm/dd` for `Trigger*.root` files over a date range, builds subprocess commands for `readroot/read_header.py` (or `read_trace.py`) followed by `main.py`, and optionally runs them in parallel via `multiprocessing.Pool`. Dry-run by default (`--run` to execute). Also handles optional PDF merge of output charts. The key function is `build_tasks()` which generates `TaskSpec` dataclass instances; `execute_tasks()` dispatches them to workers.

**`main.py`** — Runs the full 4-stage reconstruction pipeline on a single YAML input file. Each stage has its own `run_<stage>_stage()` function that delegates to the appropriate `find_event/` module. Stages are **cached**: if a stage output YAML exists and the input file signature hasn't changed, the stage is skipped. `--force-recompute` bypasses cache. The shared state dict (`_new_stage_state()`) carries `times`, `signals`, `du_ids`, `directions`, `chi_squares` between stages. Stage 2 (fingerprint) is only run when the input YAML contains signal amplitude data (trace mode). After SWM plotting, `main.py` intentionally `exit()`s — code below that point is unreachable by design.

**`readroot/`** — Reads ROOT files via `uproot`. `read_header.py` extracts `run_number`, `event_number`, `du_id`, `gps_time`, `du_nanoseconds` from the `teventadc` TTree. `read_trace.py` additionally reads 4 trace channels (F/X/Y/Z) and computes per-channel ADC maxima and XY combined amplitude. `common.py` provides shared helpers for TTree key lookup, time-map construction, and YAML writing.

**`find_event/`** — The physics core. Contains the reconstruction and filtering algorithms:
- `matching_times_graph.py`: Builds a causality graph where DU pairs are connected if their time difference ≤ distance/c × tolerance. Uses greedy maximum-clique search followed by post-processing purification (hard filter). This is the primary event-selection algorithm.
- `time_difference_fingerprint.py`: Computes time-difference fingerprints for events and removes duplicates caused by fixed background sources using sliding-window comparison.
- `plane_wave_model_gradient.py`: Fits plane-wave direction (θ, φ) using L-BFGS-B with analytical gradients and Numba JIT acceleration. Multiple initial guesses are tried.
- `spherical_wave_model_nopenal.py`: Fits spherical-wave source position using (u, v) parameterization to avoid azimuthal periodic boundary issues. Iteratively prunes high-χ² DUs. Handles both 5-parameter (≥5 DUs) and 3-parameter (4 DUs) fits.
- `io.py`: Detector coordinate loading and event result filtering/writing.
- `plotting.py`: All visualization — azimuth/zenith time-series, 3D direction scatter, detector position maps, signal LDF fits.
- `utils.py`: GPS/UTC conversion, coordinate rotation, angular calculations.
- Older versions (`matching_times.py`, `plane_wave_model.py`, `spherical_wave_model.py`) are retained but superseded by the graph/gradient/nopenal versions.

**`merge/`** — Post-processing aggregation. `merge_header.py` merges header-type YAMLs by `event_number`, deduplicating and combining DU info. `merge_trace.py` does the same for trace-type YAMLs (F/X/Y/Z/XY channels). `merge_results.py` concatenates `_matched`, `_PWM`, `_SWM` YAMLs. All use `common.py` for file discovery and traceability header generation.

**`stats/`** — Statistical analysis on merged YAMLs. `stats_trigger.py` computes per-event DU counts, event rates, DU trigger rates, and generates plots. `stats_du_pairs.py` compares observed vs theoretical DU-pair time differences. `stats_lookback.py` does lookback-window analysis. `common.py` provides YAML reading, DU time-offset loading, and GPS time-axis formatting.

**`du_pair_theoretical.py`** — Pre-computes theoretical maximum time differences (`distance / c`) for all DU pairs based on detector geometry. Results are cached as CSV (`_gp65_rtksort_du_pair_theoretical.csv`) and consumed by `matching_times_graph.py`.

**`toymc/`** — Toy Monte Carlo simulations for signal, background, and mixed event studies. Used for validating the causality-matching approach.

**`scripts/`** — Shell scripts for batch execution, Slurm job submission, test running, and various stats batch runners. Notable: `run_merge_date.py` (date-range merge orchestrator), `run_loop_hourly.py` (hourly loop runner), `submit_run_main_date.sh` (Slurm submit for main.py).

**`logger_config.py`** — Configures `loguru`-based logging with console + file sinks, log rotation, and `LOG_LEVEL`/`LOG_FILE` environment variable overrides. All modules import `from logger_config import logger`.

### Key conventions (from copilot-instructions.md)

- **Cache filename suffixes are load-bearing**: `*_matched.yaml`, `*_fingerprint.yaml`, `*_PWM.yaml`, `*_SWM.yaml`. Downstream stages and merge scripts depend on these exact suffixes.
- **YAML schema keys** (`du_ns`, `du_id`, `gps_time`, `event_number`, `index`) must be preserved because downstream logic assumes them.
- **Date directories** use `yyyy/mm/dd` format under `--base-path`.
- **Dry-run by default**: `loop.py` prints planned commands unless `--run` is passed.
- **Logging**: Use `logger_config.logger` for runtime logs; avoid `print()` diagnostics.
- **Plotting style**: Before plotting, import `scienceplots` and apply `plt.style.use(["science", "grid", "notebook"])`.
- **Stats scripts**: Row records must use `NamedDefaultDict`-style models; access fields by name only (`row["du_id"]`), never by index. Use `DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"` consistently.
- **Detector position file**: Most scripts assume `_gp65_rtksort.txt` is present; pass `--det-pos` to override.

### Pitfalls

- `matching_times_graph.py` applies strict speed-of-light consistency filtering with tight tolerances; do not relax thresholds without explicit request.
- `main.py` `exit()`s after SWM plotting — code after that point is unreachable by design.
- PDF merge in `loop.py` depends on Pillow (`PIL`) and gracefully skips if unavailable.
- Plotting failures should not block core data generation; keep them isolated.
- `loop.py` top comment may reference `read_header/...` (legacy paths); actual scripts live under `readroot/`.
- Do not change cache suffix conventions or YAML schema keys — downstream stages and merge/scripts depend on them.
