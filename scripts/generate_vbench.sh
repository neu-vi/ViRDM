#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/conda_env.sh"
require_conda_python

usage() {
  cat <<'EOF'
Usage:
  bash scripts/generate_vbench.sh --checkpoint PATH --vbench-root PATH [options]

Options:
  --recipe causal4|causal2|causal1|bid4  Default: causal4.
  --checkpoint PATH                       ViRDM generator .pt or model.pt. Required.
  --vbench-root PATH                      Clone of Vchitect/VBench. Required.
  --output PATH                           Default: outputs/vbench/<recipe>.
  --extended-prompts PATH                 Optional 946/944-row generation text;
                                          the matched release protocol uses
                                          prompts/vbench/all_dimension_extended.txt.
  --gpus N                                Prompt-parallel inference workers. Default: 8.
  --seed N                                Default: 0.
  --samples-per-prompt N                  Default: 1; use 5 for standard VBench.
  -h, --help                              Show this help.

The generated filenames always use the original official VBench prompts, even
when --extended-prompts supplies longer text to the model.
EOF
}

RECIPE=causal4
CHECKPOINT=
VBENCH_ROOT=
OUTPUT=
EXTENDED_PROMPTS=
GPUS=8
SEED=0
SAMPLES_PER_PROMPT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recipe) RECIPE=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --vbench-root) VBENCH_ROOT=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --extended-prompts) EXTENDED_PROMPTS=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --samples-per-prompt) SAMPLES_PER_PROMPT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${CHECKPOINT}" && -f "${CHECKPOINT}" ]] || { echo "--checkpoint must be a file" >&2; exit 2; }
[[ -n "${VBENCH_ROOT}" && -f "${VBENCH_ROOT}/vbench/VBench_full_info.json" ]] || {
  echo "--vbench-root must point to a VBench clone" >&2
  exit 2
}
[[ "${SAMPLES_PER_PROMPT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--samples-per-prompt must be positive" >&2
  exit 2
}
OUTPUT=${OUTPUT:-${REPO_ROOT}/outputs/vbench/${RECIPE}}
PROMPT_DIR=${OUTPUT}/prompt_suite
mkdir -p "${PROMPT_DIR}"

PREPARE_ARGS=(
  --full-info "${VBENCH_ROOT}/vbench/VBench_full_info.json"
  --output "${PROMPT_DIR}/official.txt"
  --receipt "${PROMPT_DIR}/receipt.json"
)
INFER_PROMPT_ARGS=()
if [[ -n "${EXTENDED_PROMPTS}" ]]; then
  [[ -f "${EXTENDED_PROMPTS}" ]] || { echo "extended prompt file not found: ${EXTENDED_PROMPTS}" >&2; exit 2; }
  PREPARE_ARGS+=(--extended-source "${EXTENDED_PROMPTS}" --extended-output "${PROMPT_DIR}/extended.txt")
  INFER_PROMPT_ARGS+=(--extended_prompt_path "${PROMPT_DIR}/extended.txt")
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_vbench_prompts.py" "${PREPARE_ARGS[@]}"

bash "${REPO_ROOT}/scripts/infer.sh" \
  --recipe "${RECIPE}" \
  --checkpoint_path "${CHECKPOINT}" \
  --prompt_path "${PROMPT_DIR}/official.txt" \
  --output_folder "${OUTPUT}/videos" \
  --num_samples_per_prompt "${SAMPLES_PER_PROMPT}" \
  --naming raw_prompt_index \
  --seed "${SEED}" \
  --gpus "${GPUS}" \
  "${INFER_PROMPT_ARGS[@]}"

printf 'ok\n' > "${OUTPUT}/generation.done"
echo "VBench videos ready: ${OUTPUT}/videos"
