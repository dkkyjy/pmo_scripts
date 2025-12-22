PMO Scripts (grand)

Quickstart
----------
This repository processes ROOT Trigger files into reconstructed event YAMLs and diagnostic plots.

1. Create and activate a conda environment:

```bash
pip install -r requirements.txt
```

2. Configure logger verbosity (optional):

```bash
export LOG_LEVEL=DEBUG
export LOG_FILE=logs/loop.log
```

3. Run for a single day (dry-run):

```bash
python3 pmo_scripts/loop.py 2025-11-01T00:00:00 2025-11-01T23:59:59 --base-path /path/to/ROOT --out-dir-base ../Reco_Dir
```

4. Run actual processing with concurrency:

```bash
python3 pmo_scripts/loop.py 2025-11-01T00:00:00 2025-11-01T23:59:59 --base-path /path/to/ROOT --out-dir-base ../Reco_Dir --run --jobs 4
```

Notes
-----
- The time format supports the options: '2025-11-01T00:00:00', '2025-11-01', '2025/11/01', or '20251101'.
- Merged YAML outputs created by `merge.py` include a comment header listing source files and timestamp for traceability.
