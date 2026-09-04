#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
if [[ -n "${VBENCH_PYTHON:-}" ]]; then PYTHON_BIN=${VBENCH_PYTHON}; fi
source "${SCRIPT_DIR}/conda_env.sh"
require_conda_python

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_vbench.sh --videos PATH --vbench-root PATH --name NAME [options]

Options:
  --output PATH       Default: eval/<name>.
  --gpus N            VBench evaluation workers. Default: 8.
  --dimensions ...    Optional dimension subset; must be the final argument.
  --local-models      Use/download checkpoints in VBENCH_CACHE_DIR.
  -h, --help          Show this help.
EOF
}

VIDEOS=
VBENCH_ROOT=
NAME=
OUTPUT=
GPUS=8
LOCAL_MODELS=0
DIMENSIONS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos) VIDEOS=$2; shift 2 ;;
    --vbench-root) VBENCH_ROOT=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --local-models) LOCAL_MODELS=1; shift ;;
    --dimensions) shift; DIMENSIONS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${VIDEOS}" && -d "${VIDEOS}" ]] || { echo "--videos must be a directory" >&2; exit 2; }
[[ -n "${VBENCH_ROOT}" && -f "${VBENCH_ROOT}/vbench/VBench_full_info.json" ]] || {
  echo "--vbench-root must point to a VBench clone" >&2
  exit 2
}
[[ -n "${NAME}" ]] || { echo "--name is required" >&2; exit 2; }
[[ "${GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be positive" >&2; exit 2; }
OUTPUT=${OUTPUT:-${REPO_ROOT}/eval/${NAME}}

ARGS=(
  --videos-path "${VIDEOS}"
  --full-info "${VBENCH_ROOT}/vbench/VBench_full_info.json"
  --output-dir "${OUTPUT}"
  --name "${NAME}"
)
if [[ "${LOCAL_MODELS}" -eq 1 ]]; then ARGS+=(--local-models); fi
if [[ ${#DIMENSIONS[@]} -gt 0 ]]; then ARGS+=(--dimensions "${DIMENSIONS[@]}"); fi

export PYTHONPATH="${VBENCH_ROOT}:${PYTHONPATH:-}"
if [[ "${GPUS}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_vbench.py" "${ARGS[@]}"
else
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPUS}" \
    "${REPO_ROOT}/scripts/run_vbench.py" "${ARGS[@]}"
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_vbench.py" \
  --input "${OUTPUT}" \
  --output "${OUTPUT}/score_summary.json"
