#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/conda_env.sh"
require_conda_python
require_conda_cuda 12.4

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ViRDM requires Python 3.10; got {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import sysconfig

include_dirs = [Path(sysconfig.get_path("include"))]
for variable in ("CPATH", "CPLUS_INCLUDE_PATH"):
    include_dirs.extend(
        Path(path) for path in os.environ.get(variable, "").split(os.pathsep) if path
    )
headers = [directory / "Python.h" for directory in include_dirs]
if not any(header.is_file() for header in headers):
    raise SystemExit(
        f"Missing Python.h in {include_dirs}. Recreate the Conda environment from its YAML. "
        "Use "
        "a Python 3.10 distribution that includes development headers."
    )
PY
VIRDM_BUILD_ROOT=${VIRDM_BUILD_ROOT:-${CONDA_PREFIX}/.build/virdm}
mkdir -p "${VIRDM_BUILD_ROOT}/tmp" "${VIRDM_BUILD_ROOT}/pip-cache"
export TMPDIR=${VIRDM_BUILD_ROOT}/tmp
export PIP_CACHE_DIR=${VIRDM_BUILD_ROOT}/pip-cache

# FlashAttention's release-wheel bootstrap uses an atomic rename. Keeping pip's
# temporary and cache directories on the same filesystem makes setup reliable
# when /tmp, the checkout, and the user's default cache are separate mounts.
"${PYTHON_BIN}" -m pip install --upgrade \
  pip setuptools wheel packaging ninja==1.13.0 psutil==7.2.2
"${PYTHON_BIN}" -m pip install \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
VIRDM_FLASH_BUILD_JOBS=${VIRDM_FLASH_BUILD_JOBS:-8}
VIRDM_FLASH_NVCC_THREADS=${VIRDM_FLASH_NVCC_THREADS:-4}
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=${VIRDM_FLASH_BUILD_JOBS} \
NVCC_THREADS=${VIRDM_FLASH_NVCC_THREADS} \
"${PYTHON_BIN}" -m pip install --no-build-isolation -r "${REPO_ROOT}/requirements.txt"
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import torch
import virdm_integration
from wan.modules.attention import require_flash_attention_2

attention_backend = require_flash_attention_2("2.8.3.post1")
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Training attention: {attention_backend}")
print(f"ViRDM import: {virdm_integration.__file__}")
PY

echo "Environment ready. Activate it with: conda activate ${CONDA_DEFAULT_ENV}"
