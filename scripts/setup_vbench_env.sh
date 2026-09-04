#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VBENCH_ROOT=${VBENCH_ROOT:-${REPO_ROOT}/external/VBench}
source "${SCRIPT_DIR}/conda_env.sh"
require_conda_python
require_conda_cuda 12.1
VBENCH_REVISION=${VBENCH_REVISION:-fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490}

"${PYTHON_BIN}" - <<'PY'
import pathlib
import sys
import sysconfig
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"VBench environment requires Python 3.10; got {sys.version.split()[0]}")
header = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.is_file():
    raise SystemExit(
        f"VBench's Detectron2 build requires Python.h, but {header} is missing. "
        "Install the Python 3.10 development headers (for example "
        "python3.10-dev on Debian/Ubuntu) or set PYTHON_BIN to a Python 3.10 "
        "Conda environment that includes headers."
    )
PY

if [[ ! -d "${VBENCH_ROOT}/.git" ]]; then
  mkdir -p "$(dirname -- "${VBENCH_ROOT}")"
  git clone https://github.com/Vchitect/VBench.git "${VBENCH_ROOT}"
fi
git -C "${VBENCH_ROOT}" fetch --all --tags
git -C "${VBENCH_ROOT}" checkout --detach "${VBENCH_REVISION}"

# decord 0.6.0, required by the pinned VBench revision, ships a legacy wheel
# tag that newer pip versions reject during `pip check` even though its Linux
# extension imports and decodes correctly. Keep pip 24.0 in this isolated
# evaluator environment; the training environment is unaffected. VBench's
# pinned OpenAI CLIP dependency also still imports pkg_resources, so retain the
# final setuptools release that provides that compatibility module.
"${PYTHON_BIN}" -m pip install --upgrade 'pip==24.0' 'setuptools==80.9.0' wheel
"${PYTHON_BIN}" -m pip install \
  torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
"${PYTHON_BIN}" -m pip install --no-build-isolation "${VBENCH_ROOT}"
"${PYTHON_BIN}" -m pip install --no-build-isolation \
  'detectron2@git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453'

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import decord
import detectron2
import pkg_resources
import torch
from vbench import VBench
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"decord: {decord.__version__}")
print(f"Detectron2: {detectron2.__version__}")
print(f"VBench import: {VBench.__module__}")
PY

echo "VBench ready. Activate it with: conda activate ${CONDA_DEFAULT_ENV}"
echo "VBench source: ${VBENCH_ROOT}"
