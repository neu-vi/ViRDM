#!/usr/bin/env bash
set -euo pipefail
set +x

GPU_COUNT=${1:?usage: train.sh 1|8 chunk4|chunk4+2|chunk4+1|bid4 [dynamic|nodynamic]}
RECIPE=${2:?usage: train.sh 1|8 chunk4|chunk4+2|chunk4+1|bid4 [dynamic|nodynamic]}
DYNAMIC_MODE=${3:-nodynamic}
case "${GPU_COUNT}" in 1|8) ;; *) echo "GPU count must be 1 or 8" >&2; exit 2 ;; esac
case "${RECIPE}" in chunk4|chunk4+2|chunk4+1|bid4) ;; *) echo "invalid recipe: ${RECIPE}" >&2; exit 2 ;; esac
case "${DYNAMIC_MODE}" in dynamic) DYNAMIC_FLAG=--dynamic ;; nodynamic) DYNAMIC_FLAG= ;; *) echo "invalid dynamic mode: ${DYNAMIC_MODE}" >&2; exit 2 ;; esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
source "${SCRIPT_DIR}/conda_env.sh"
require_conda_python
BASE_CONFIG=${BASE_CONFIG:-${REPO_ROOT}/config_virdm_bs64_1x8.yaml}
export VIRDM_ARTIFACT_ROOT=${VIRDM_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}
export VIRDM_WAN_MODEL_ROOT=${VIRDM_WAN_MODEL_ROOT:-${VIRDM_ARTIFACT_ROOT}/wan/Wan2.1-T2V-1.3B}

RECIPE_SLUG=${RECIPE//+/_plus_}
TOPOLOGY_SUFFIX=
if [[ "${GPU_COUNT}" == 1 ]]; then
  TOPOLOGY_SUFFIX=_1gpu
fi
RUN_NAME=${RUN_NAME_OVERRIDE:-virdm_${RECIPE_SLUG}_${DYNAMIC_MODE}_bs64${TOPOLOGY_SUFFIX}_lr2e6_step20}
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/runs/${RUN_NAME}}
LAUNCH_CONFIG=${RUN_DIR}/launch_config.yaml
WANDB_MODE=${WANDB_MODE:-online}

test -x "${PYTHON_BIN}"
test -s "${BASE_CONFIG}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_artifacts.py" \
  --config "${BASE_CONFIG}" ${DYNAMIC_FLAG}

mkdir -p "${RUN_DIR}/wandb"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_recipe.py" \
  --base "${BASE_CONFIG}" \
  --rollout "${RECIPE}" \
  --world-size "${GPU_COUNT}" \
  ${DYNAMIC_FLAG} \
  --output "${LAUNCH_CONFIG}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIRDM_RUN_NAME=${RUN_NAME}

WANDB_ARGS=()
if [[ "${WANDB_MODE}" == disabled ]]; then
  WANDB_ARGS+=(--disable-wandb)
fi

cd "${REPO_ROOT}"
set +e
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPU_COUNT}" train.py \
  --config_path "${LAUNCH_CONFIG}" \
  --logdir "${RUN_DIR}" \
  --wandb-save-dir "${RUN_DIR}/wandb" \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee -a "${RUN_DIR}/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
exit "${TRAIN_STATUS}"
