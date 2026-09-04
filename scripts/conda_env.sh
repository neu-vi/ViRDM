#!/usr/bin/env bash
require_conda_python() {
  if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == base ]]; then
    echo "Activate a project Conda environment first (see README.md); do not use base." >&2
    return 1
  fi
  PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}
  PYTHON_BIN=$(command -v "${PYTHON_BIN}") || return 1
  "${PYTHON_BIN}" - <<'CHECK' || return
import os
from pathlib import Path
import sys
if Path(sys.prefix).resolve() != Path(os.environ['CONDA_PREFIX']).resolve():
    raise SystemExit('Selected Python is outside the active Conda environment. Unset PYTHON_BIN or activate the correct environment.')
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f'Python 3.10 required; got {sys.version.split()[0]}. Create the environment from its YAML.')
CHECK
  export PYTHON_BIN
}
require_conda_cuda() {
  local expected=$1
  export CUDA_HOME="${CONDA_PREFIX}"
  export CUDA_PATH="${CONDA_PREFIX}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Missing Conda nvcc; recreate the environment from its YAML." >&2
    return 1
  fi
  if ! "${CUDA_HOME}/bin/nvcc" --version | grep -Fq "release ${expected},"; then
    echo "Expected CUDA toolkit ${expected} in ${CUDA_HOME}; activate the matching environment." >&2
    return 1
  fi
  if [[ -z "${CXX:-}" ]] || ! command -v "${CXX}" >/dev/null; then
    echo "Missing Conda C++ compiler; recreate and reactivate the environment." >&2
    return 1
  fi
  echo "Python: ${PYTHON_BIN}; CUDA: ${CUDA_HOME}; compiler: ${CXX}"
}
