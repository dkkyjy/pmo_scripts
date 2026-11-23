# Repository Guidelines

Python analysis pipeline for the Dunhuang DU array: ROOT event data → calibration →
causality matching → fingerprint filter → plane/spherical wave direction reconstruction
→ cuts → fluence/energy reconstruction. Deep behavior lives in `README.md` (Chinese); this
file is the agent map: what breaks, what is intentional, what will bite you.

## Project Overview

- Inputs: real trigger data (`teventadc`), CoREAS sim (`tadc`/`tefield`/`tvoltage`),
  StarShape/ZHAireS sim (collections). Selected by `loop.py --reader {trigger,coreas,starshape}`.
- Outputs: sibling YAML stage caches, PNG figures, merged PDF, and (optional) a
  `*_reconstruction_summary.yaml` from the ADF/PWF/SWF chain.
- No package, no service, no CI: plain scripts run from the repo root.

## Architecture & Data Flow

```
ROOT ──readroot/read_*.py──► <out>/<date|coreas|starshape>/<stem>[_CH].yaml
      ──(trigger only) time_position_offset.apply_calibration──► <stem>.toffset.yaml
                                                                <det_pos>.toffset.txt
      ──main.py──► _matched.yaml ──► _matched_fingerprint.yaml
                ──► _matched_fingerprint_PWM.yaml
                ──► (in-memory cuts) ──► _matched_fingerprint_SWM.yaml ──► PWM again
      ──loop.py──► <stem>.pdf (PNG merge, needs Pillow)
      ──run_parallel_reconstruction_fixed.py --candidates <PWM/SWM yaml>
                ──► ./Reconstruction/<stem>_event_<n>/… + *_reconstruction_summary.yaml
```

Stage table (functions inside `main.py` `main()`):

| # | Stage | Call site | Artifact |
|---|-------|-----------|----------|
| 0 | layout detect (`det_pos` per event? `truth` block = simulation) | `main.py` `_load_det_pos_by_event` | — |
| 0b | calibration (trigger only) | `apply_calibration` inside `main()` | `*.toffset.yaml`, `*.toffset.txt` |
| 1 | causality matching | `run_matching_stage` | `_matched.yaml` |
| 2 | fingerprint | `run_fingerprint_stage` | `_matched_fingerprint.yaml` |
| 3 | PWM + PWF refine | `run_pwm_stage` | `_PWM.yaml` |
| 4 | footprint + ring cut (memory only) | `apply_customized_cuts` | diagnostics CSV/PNG |
| 5 | SWM | `run_swm_stage` | `_SWM.yaml` |
| 6 | PWM again (re-plot) | `run_pwm_stage` (second call) | same `_PWM.yaml` |

Invariants that are load-bearing:

- **The layout discriminator is `det_pos` per event *or* a `truth` block** (`main.py` `_load_det_pos_by_event`).
  StarShape embeds `det_pos`; CoREAS embeds `truth` only and takes its coordinates from
  `--det-pos` (`loop.py` defaults that to `position/du_coords_CoreasDC2.txt`). Add a
  `det_pos` block to trigger YAML and calibration silently turns off for the whole pipeline.
- **Only an embedded `det_pos` reaches the PWM/SWM caches.** Trigger and CoREAS rows carry
  none (the static table stays the single source), so a `_PWM.yaml` from those readers is
  not self-contained — pass `--det-pos` to standalone cut tools.
- **Artifact suffixes are the API.** Paths come from `str.replace(".yaml", "_X.yaml")`
  (`run_matching_stage`, `run_fingerprint_stage`, `run_pwm_stage`, `run_swm_stage`) and are consumed by the reconstruction driver. Renaming a
  suffix breaks caches, plotting and downstream tools at once.
- **Cache validity = exact dict equality** of `{path, exists, size, mtime_ns}` signatures
  (`_build_file_signature` + `_build_stage_cache_meta`). The *absolute path* is in the signature — that is why calibration can
  switch inputs without touching stage code, and why moving files invalidates everything.
  Any new stage input must be added to `expected_meta`, and `schema_version` is not enforced
  on read.
- **Caches are write-mostly on a full run.** `run_fingerprint_stage` returns
  `fingerprint_computed = True` unconditionally and `run_pwm_stage` does
  `use_cache = … and not fingerprint_computed`; SWM gates on `pwm_computed`
  the same way. Alias: the 8th positional arg of `run_pwm_stage` is *not* an "enabled" flag.
- **Reader-derived defaults** (`loop.py:896-920`): `coreas`/`starshape` default
  `--skip-matching`/`--skip-fingerprint` on (`BooleanOptionalAction`, explicit flag wins) and
  force `--skip-footprint`/`--skip-ringcut` on — those cuts only read the *static* `--det-pos`
  file and would drop every sim event. There is **no CLI switch to re-enable the cuts for sims**.
- **Skip semantics differ by stage**: `--skip-matching` loads *source* metadata
  (`_load_event_metadata_map`), `--skip-fingerprint` prefers `<stem>_matched.yaml` when it exists
  (`skip_fingerprint_stage`). Both skip helpers are called before their `if skip_*` test, so their
  YAML read happens even when the real stage runs.
- Trigger date filtering is **filename-based** (`basename.split("_")[1]`, `loop.py:113`), not
  mtime; sim readers ignore dates for *file selection* but still write into the fixed
  `coreas/`/`starshape/` output dirs.

## Key Directories

| Path | Purpose |
|------|---------|
| `loop.py`, `main.py` | batch driver; single-file stage machine — both are CLI-first |
| `readroot/` | `common.py` (TTree detect, time maps, YAML IO), `read_header.py`/`read_trace.py` (trigger), `read_coreas.py`, `read_starshape.py` |
| `find_event/` | **current**: `matching_times_graph.py`, `time_difference_fingerprint.py`, `plane_wave_model_gradient.py`, `pwf_refine.py`, `spherical_wave_model_nopenal.py`, `shower_footprint_filter.py`, `plotting.py`, `io.py`, `utils.py` |
| `find_event/` (legacy, unreferenced) | `matching_times.py`, `plane_wave_model.py`, `spherical_wave_model.py`, `background_rejection.py` — only `matching_times` and `background_rejection` even appear in the package `__all__`; nothing imports them |
| `find_event/` (legacy, import-load) | `estimation.py`, `time_offset.py` — `main.py:22` (`from find_event import estimation as fe_est`) imports `estimation`, which imports `time_offset` (`estimation.py:14`); `fe_est` is never dereferenced, but neither module is deletable without editing `main.py` first |
| `reconstruction_candidate/` | authoritative `reconstruction_candidate_fluence_error_fixed.py` (three orchestrators only) + `reconstruction_common.py` / `reconstruction_antenna.py` / `reconstruction_io.py` / `reconstruction_direction.py` + `run_parallel_reconstruction_fixed.py`; legacy twins without `_fixed`; `tests/` |
| `position/` | DU layout files (`ID x y z`, `#` = disabled DU, space **or** comma separated) + `*_du_pair_theoretical.csv` cache |
| `calibration/` | `jul24_locked_xyz_additive_utc.csv` (time windows + per-DU xyz offsets) |
| `time_position_offset.py`, `du_pair_theoretical.py` | calibration subroutine; DU-pair theoretical-Δt cache builder |
| `restoration/` | old snapshot of the recon chain — do not edit |
| `docs/solutions/` | documented solutions to past problems (bugs, workflow patterns), one file per solved problem, organized by category with YAML frontmatter (`module`, `tags`, `problem_type`); gitignored, local-only |

## Development Commands

All commands assume **CWD = repo root** (see Runtime section).

```bash
pip install -r requirements.txt        # insufficient alone: also needs iminuit, h5py, awkward

# batch (dry run by default — commands are only logged)
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59
python loop.py 2025-01-01T00:00:00 2025-01-01T23:59:59 --run --jobs 4
python loop.py --reader coreas --base-path /path/sim --out-dir-base ../Reco_Dir --run
python loop.py --reader starshape --base-path /path/*_stshp_*.root --max-events 10 --run
# note: --base-path takes one value — that glob must expand to a single file; pass a directory for many

# single file
python readroot/read_header.py F.root --date 2025/01/01 --out_dir_base ../Reco_Dir
python main.py ../Reco_Dir/2025/01/01/Trigger_x.yaml --fig_name Trigger_x
python -m find_event.shower_footprint_filter Trigger_x_PWM.yaml --det-pos position/_gp65_rtksort.txt
python du_pair_theoretical.py position/_gp65_rtksort_2002_DU7.txt [--force-recompute]

# tests / syntax gate
pytest reconstruction_candidate/tests -q
python -m py_compile reconstruction_candidate/reconstruction_candidate_fluence_error_fixed.py \
                        reconstruction_candidate/reconstruction_common.py \
                        reconstruction_candidate/reconstruction_antenna.py \
                        reconstruction_candidate/reconstruction_io.py \
                        reconstruction_candidate/reconstruction_direction.py \
                        reconstruction_candidate/run_parallel_reconstruction_fixed.py

# ADF driver (no dry run; resume-by-default)
python reconstruction_candidate/run_parallel_reconstruction_fixed.py \
    --candidates Trigger_x_fingerprint_SWM.yaml --ncores 8 --det-pos position/du_coords_CoreasDC2.txt
```

Useful bounding flags for a first run: `--limit`, `--max-events`, `--only-read`,
`--skip-read`, `--force-recompute`, `--skip-pwf-refine` (main.py only — `loop.py` never
forwards it). `--left`/`--right` are per-reader: the trigger `read_trace.py` path receives
them under `--with-signal` (`loop.py:597-599`), the coreas reader under `--with-signal`
(`loop.py:403-404`), and the trigger `read_header.py` and starshape paths define no such
options; `--max-radius` is `coreas`-only.

## Code Conventions & Common Patterns

- **Scripts, not a package.** Modules import top-level names (`from logger_config import logger`,
  `from find_event.io import …`); `readroot/__init__.py` and `find_event/__init__.py` only declare `__all__`.
- **Logging:** loguru only. `import logger_config` configures sinks at import time
  (`logger_config.py:82`) — a file sink `loop.log` at DEBUG in the CWD, `enqueue=True`.
  Every command, even a dry run, appends to `loop.log`. Override with `LOG_LEVEL`/`LOG_FILE`.
- **Stage contract:** `stage(...) -> (state, computed_flag)`; `state` is a dict of parallel
  dicts keyed by the canonical event key (`_new_stage_state`, `_canonicalize_state_keys`).
  Cuts must truncate *every* field (`shower_footprint_filter.py:387-389`).
- **Error handling:** broad `except` → `logger.warning`/`error` → return `(state, False)` or a
  status int; per-file reader failures never abort a batch. Plotting failures are swallowed so
  caches still get written (`_validate_stage_cache_payload`).
- **Comments/docstrings:** identifiers and module docstrings English; section comments and the
  algorithm modules (`matching_times_graph`, `spherical_wave_model_nopenal`,
  `reconstruction_candidate*`) Chinese. Match the file you edit.
- **Type normalization:** `du_id` is `[str]` in YAML but detector maps are keyed by `int`;
  every consumer converts (`plane_wave_model_gradient.py:171`, `pwf_refine._position_of`).
- **Units:** times are ns in `find_event`, **seconds** in the ADF `t` argument; `c` appears as
  `299792458.0/1e9/1.000259` in several files and `/1.000` in a legacy one — do not "unify"
  casually.
- **Physics quirks that are intentional:** PWM exports the arrival vector `(-dx,-dy,+dz)`;
  SWM YAML `x,y,z` is a *source point* in metres while `zenith/azimuth` are copied from PWM;
  cuts use `atan2(-y,-x)` and plots `atan2(y,x)`. `dw_err = NaN` is a sentinel for
  "not computed", not a bug.
- `main()` reads the **module-global `args`** (module-level `args = None`, then `args.<field>` inside `main()`), set only under
  `if __name__ == "__main__"` — calling `main.main(...)` programmatically raises `NameError`.

## Important Files

| File | Why it matters |
|------|----------------|
| `loop.py` | subprocess orchestration, reader-gated flag forwarding (`make_*_command`), PDF merge, reader defaults |
| `main.py` | ordered stages, cache signatures, cut wiring; the file to read before changing CLI behavior |
| `readroot/common.py` | tree detection order `teventadc → tadc → tefield → tvoltage → SimuCollection`, YAML schema |
| `find_event/matching_times_graph.py` | causality graph; effective ratio bound is hard-coded `1.02` — the `speed_of_light_tolerance` arg is dead |
| `find_event/plane_wave_model_gradient.py` | PWM; `<4` DU events skipped, σ = 6 ns in χ² |
| `find_event/spherical_wave_model_nopenal.py` | SWM; `<5` DU skipped, `rho ∈ [2e3,1e6] m`, `theta ∈ [5,95]°` |
| `find_event/shower_footprint_filter.py` | cut thresholds (`min_du_fraction_in_ellipse=0.8`, `cone_half_angle_deg=2.0`, `dmax_cut_threshold(800, 300, 75)` …) |
| `reconstruction_candidate/reconstruction_direction.py` | PWF/SWF/ADF fitters, the direction entry points and the 20-field direction contract (`_TUPLE_LEN == 20` asserted at import); no ROOT I/O, no `parameter_files` |
| `reconstruction_candidate/reconstruction_candidate_fluence_error_fixed.py` | only the three orchestrators `efield_recons_from_efield_PWF` / `energy_restruction` / `adf_angles_from_voc`; imports `reconstruction_common` / `reconstruction_antenna` / `reconstruction_io` / `reconstruction_direction` one-way, defines no I/O or antenna response itself |
| `reconstruction_candidate/reconstruction_common.py` | path constants (`efield_rec_dir`, a CWD-relative literal; `DU_POS_TABLE` / `DU_ROTATION_FILE` / `_PARAM_DIR` with env overrides), plus `gps_to_lst` / `rotation_m` / `calculate_energy_fluence_and_error`; imports no sibling layer |
| `reconstruction_candidate/reconstruction_antenna.py` | antenna response (EFL / ADC→Voc), the `REfile_grand` cache (loaded lazily), `tukey_wind`, and the DU rotation-table reader `read_du_rotation_angles`; the 30–250 MHz band is written inline, there is no band constant |
| `reconstruction_candidate/reconstruction_io.py` | ROOT reading (`read_root_file_from_path`, `readroot` fallback), `_read_du_position_table`, `causality_filter` (writes a per-event-per-process temp file), `_COREAS_TRUTH_TREES` |
| `time_position_offset.py` | `apply_calibration`; `DEFAULT_OFFSET_CSV` is module-anchored (CWD-independent) |

Docs status: `README.md` is the authoritative deep dive (Chinese) — read it before changing
CLI behavior — but it is stale in places: `--offset-csv` is `None` in both CLIs with the real
default applied late (`time_position_offset.py:32`); its flag table prints
`coreas/starshape 下默认关` for `--skip-footprint`/`--skip-ringcut` (`README.md:127`) while its
own difference table says 默认 **True** (`README.md:211`) and `loop.py:918-920` forces both on
for sim readers; and the documented `adf_angles_from_voc(gps-style)` signature is the old one
(`adf_angles_from_voc` in `reconstruction_candidate/reconstruction_candidate_fluence_error_fixed.py`).
`reconstruction_candidate/*.md` describe the **legacy** non-`_fixed` modules (15/16-column
results, 108-start ADF grid) — do not trust them for the current chain (18 columns, 36 starts).

## Runtime/Tooling Preferences

- **Python 3.10** observed (`__pycache__/*.cpython-310*`); ≥3.9 is the syntax floor
  (`argparse.BooleanOptionalAction`, `loop.py:812`). No `pyproject.toml`, no pins, no venv committed.
- `requirements.txt` is **incomplete**: `iminuit`, `h5py`, `awkward` are imported but undeclared
  (importing `reconstruction_candidate_fluence_error_fixed.py` fails on a bare install);
  `Pillow` (PDF merge) and `radiotools` are optional/lazy. `scienceplots` and `matplotlib` are
  hard import-time deps of `main.py` (`find_event/plotting.py:11-13`).
- **Run everything from the repo root.** Verified: `python readroot/read_header.py` fails with
  `ModuleNotFoundError: logger_config` when the CWD is not the repo root, because neither
  `logger_config` nor `readroot.*` is a package path. `loop.py` invokes `<sys.executable>
  readroot/...` relatively for the same reason.
- Config is CLI flags + hard-coded constants; YAML is data interchange, not configuration.
  Env vars are the override mechanism: `LOG_LEVEL`, `LOG_FILE`, `RECON_EFL_MAT`,
  `RECON_DU_POS_TABLE`, `RECON_DU_ROTATION_FILE`, `RECON_COREAS_DET_POS`,
  `RECON_PARAM_DIR`, `RECON_SKIP_PLOTS`, `PYTEST_COREAS_ROOT`;
  `RECON_GALACTIC_MAT`, `RECON_PG_MAT` and `RECON_ZANT_CSV` override galactic-noise inputs
  (normally taken relative to `RECON_PARAM_DIR`), and `RECON_ROOT`/`RECON_EVENT`/`RECON_ADF_VOC_WINDOW`
  only drive the module's own `__main__` demo.
- Machine-specific defaults: `--base-path /mnt/sdb2/users/m/mapx/…` (`loop.py:723`),
  `--out-dir-base ../Reco_Dir`, `--det-pos position/_gp65_rtksort_DU7_1029.txt` (CWD-relative, `main.py`
  `parse_args` default), `./Reconstruction/` for ADF outputs. Several code defaults point at files that
  do not exist: `reconstruction_candidate/parameter_files/` (EFL `.mat`, rotation table) and
  `<repo>/_gp65_rtksort.txt` (the real file is `position/_gp65_rtksort.txt`).
- `.gitignore` is blanket (`*.yaml *.json *.csv *.txt *.png *.pdf *.log *.mat`) and also hides
  `.github/`, `docs/`, root `/tests/`. Required inputs (`position/*.txt`, `calibration/*.csv`)
  exist only because they are force-tracked — never commit run artifacts.

## Testing & QA

- Suite: `reconstruction_candidate/tests/` (tracked; root `tests/` does not exist and is ignored).
  `pytest reconstruction_candidate/tests -q` → **88 collected, 61 passed, 27 skipped** without
  ROOT data, ~33 s (25 skips are real-input ADF cases gated by `PYTEST_COREAS_ROOT`; 2 are
  placeholders needing the absent `parameter_files/` EFL table).
  With `PYTEST_COREAS_ROOT=<CoREAS-AN event dir>` the real-input tests run and add a cross-input
  direction agreement gate (`< 3°`).
- Coverage is the ADF/PWF/SWF chain plus the driver around it (`--cov` on the package reports
  46% for `reconstruction_candidate_fluence_error_fixed.py`; the rest is EFL-blocked physics):
  `reconstruction_candidate_fluence_error_fixed.py` and `run_parallel_reconstruction_fixed.py`
  (task loading, resume short-circuit, `reconstruct_one` statuses 0–3, summary assembly, and a
  real-subprocess end-to-end that exercises only the resume path because the physics entries need
  `parameter_files/`). `loop.py`, `main.py`, `readroot/`, `find_event/`, `restoration/` have
  **zero tests**. Changing the CLI, a stage suffix or a reader is verified by running it, not by
  the suite.
- Use `python -m py_compile <file>` as the syntax gate (the project's own practice); there is no
  lint/format config — the stale `.ruff_cache/` is not enforcement. No CI: `.github/` is gitignored.
- The suite encodes contracts that are easy to violate: 20-tuple indexing confined to
  `direction_result_from_tuple` (source-regex guard), `dw_err = NaN`, `_COREAS_TRUTH_TREES`
  single literal, and "direction-only entry points write no files". Tests import the module flat
  (`import reconstruction_candidate_fluence_error_fixed as m`) via `tests/conftest.py`, which
  injects both `reconstruction_candidate/` and the repo root into `sys.path`.
- Not usable as shipped: `tests/run_adf_equiv.py` needs a `--baseline` module that is not
  committed; `_write_synthetic_tadc` self-skips on uproot/awkward versions that cannot write
  3-D jagged branches.
- Manual ladder when no test covers the change: dry-run `loop.py` (`--limit 1`), run one reader
  to YAML, run `main.py` on that YAML (writes artifacts in place), then diff the YAML fields you
  touched. `run_parallel_reconstruction_fixed.py` has **no** dry-run flag.
- Never commit: `*.yaml`/`*.json`/`*.csv`/`*.png`/`*.pdf`/`*.log`, `loop.log`, `.pytest_cache/`,
  `Reconstruction/`, `docs/`. Prefer `--force-recompute` over deleting caches by hand when a
  signature looks stale.
