#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/conda_env.sh"
require_conda_python

cd "${REPO_ROOT}"
for script in scripts/*.sh; do
  bash -n "${script}"
done
"${PYTHON_BIN}" -m compileall -q \
  model pipeline scripts third_party trainer utils virdm_integration wan \
  inference.py train.py
"${PYTHON_BIN}" -m unittest discover -s tests -v
printf 'ViRDM release checks passed\n'
