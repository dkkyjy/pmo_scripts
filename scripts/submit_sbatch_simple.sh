#!/usr/bin/env bash

set -euo pipefail

LOG_DIR="./slurm_logs"
DRY_RUN=0

show_help() {
    cat <<'EOF'
Usage:
  scripts/submit_sbatch_simple.sh [--dry-run] "<command>"

Example:
  scripts/submit_sbatch_simple.sh "python -V"
  scripts/submit_sbatch_simple.sh --dry-run "echo hello"
EOF
}

if [[ $# -eq 0 ]]; then
    show_help
    exit 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $# -ne 1 ]]; then
    echo "Error: provide exactly one command string." >&2
    show_help >&2
    exit 2
fi

USER_CMD="$1"
mkdir -p "${LOG_DIR}"

JOB_NAME="quick_job"
SBATCH_CMD=(
    sbatch
    --job-name "${JOB_NAME}"
    # --mem 128G
    -n 200
    --mem-per-cpu 4G
    --output "${LOG_DIR}/${JOB_NAME}_%j.out"
    --error "${LOG_DIR}/${JOB_NAME}_%j.err"
    --wrap "${USER_CMD}"
)

printf 'Prepared command:\n'
printf ' %q' "${SBATCH_CMD[@]}"
printf '\n'

if ((DRY_RUN == 1)); then
    exit 0
fi

"${SBATCH_CMD[@]}"
