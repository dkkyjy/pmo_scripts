# AGENTS.md

Python analysis pipeline for the Dunhuang DU array: reads ROOT event data, does
time/position calibration, causality matching, fingerprint filtering, and plane/spherical
wave-model direction reconstruction. Deep details live in `README.md` — read it before
changing CLI behavior; this file is the quick agent map.

## Layout

- `loop.py` — batch driver: enumerates ROOT files, runs read → `main.py` → PDF merge.
- `main.py` — single-file reconstruction entry point (calibration → matching → fingerprint → PWM → cuts → SWM → PWM).
- `readroot/` — ROOT readers: `read_header.py`/`read_trace.py` (trigger), `read_coreas.py`, `read_starshape.py`, `common.py` (TTree detect + YAML IO).
- `find_event/` — algorithms (`matching_times_graph.py`, `time_difference_fingerprint.py`, `plane_wave_model_gradient.py`, `spherical_wave_model_nopenal.py`, `shower_footprint_filter.py`, …). Files without `_graph`/`_gradient`/`_nopenal` are older kept-for-reference implementations.
- `reconstruction_candidate/` — ADF/SWF/PWF 方向与能量重建链（CoREAS 输入），
  权威实现是 `reconstruction_candidate_fluence_error_fixed.py` +
  `run_parallel_reconstruction_fixed.py`；`restoration/` 是旧副本，不要改。
- `position/` — detector coordinate files and DU-pair theoretical delta cache.
- `calibration/` — time/position offset CSV.
- `logger_config.py` — loguru setup; use `from logger_config import logger` everywhere.

## Commands

```bash
pip install -r requirements.txt

# Batch (dry-run by default; add --run to execute)
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59 --run --jobs 4
python loop.py --reader coreas --base-path /path/to/sim --run
python loop.py --reader starshape --base-path /path/to/*_stshp_*.root --max-events 10 --run

# Single file
python readroot/read_header.py FILE.root --date 2025/01/01 --out_dir_base ../Reco_Dir
python main.py ../Reco_Dir/2025/01/01/Trigger_xxx.yaml --fig_name Trigger_xxx

# Tests (pytest is a declared dep; tests/ is gitignored and currently empty)
pytest
```

`loop.py` only builds/prints commands unless `--run` is passed. `--reader` selects the
input layout and drives the defaults below.

## Conventions

- Python 3.10, plain scripts + packages. No packaging/type-check config; only ruff cache is present.
- Comments/docstrings are mixed Chinese/English — match the file you edit.
- Intermediate artifacts are YAML siblings of the input (`_matched`, `_fingerprint`, `*.toffset.yaml`). Preserve this naming and the cache-signature logic (including the offset file path) so recompute invalidation keeps working.
- Keep the three-reader structural differences (see README "刻意差异"); they are intentional, not bugs. trigger needs `start`/`end`; sim readers ignore dates and skip matching/fingerprint/footprint by default.

## Gotchas

- Calibration applies to trigger layouts only; coreas/starshape embed per-event `det_pos` and synthetic `gps_time`, so `main.py` skips it.
- footprint/ringcut use the *static* `--det-pos` file, not per-event sim coordinates — that is why sim defaults to skipping them.
- `--min-detectors` default is 5 across the whole chain (loop → main → matching).
- Outputs (`*.png/pdf/txt/log/csv/json/yaml`) are gitignored; don't commit them.
