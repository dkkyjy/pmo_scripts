#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running pytest for all tests with coverage"
python -m pytest tests \
  --cov=main \
  --cov=merge.merge_header \
  --cov=merge.merge_trace \
  --cov=merge.merge_results \
  --cov=stats.stats_trigger \
  --cov=stats.stats_du_pairs \
  --cov=stats.stats_lookback \
  --cov=loop \
  --cov=du_pair_theoretical \
  --cov=readroot.read_header \
  --cov=readroot.read_trace \
  --cov-report=term-missing \
  --cov-report=annotate:cov_annotate

echo "Coverage runs completed successfully."
echo "Annotated reports directory: $ROOT_DIR/cov_annotate"
