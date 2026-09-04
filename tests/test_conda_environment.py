"""Environment guards run without model downloads or GPU allocation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/conda_env.sh'

class CondaEnvironmentTests(unittest.TestCase):
    def run_guard(self, command, **overrides):
        env = dict(os.environ)
        for key in ('CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'PYTHON_BIN', 'CXX'):
            env.pop(key, None)
        env.update(overrides)
        return subprocess.run(['bash', '-c', 'source "' + str(HELPER) + '"; ' + command],
                              env=env, text=True, capture_output=True)

    def test_requires_activation(self):
        result = self.run_guard('require_conda_python')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Activate', result.stderr)

    def test_rejects_base(self):
        result = self.run_guard('require_conda_python', CONDA_PREFIX=sys.prefix,
                                CONDA_DEFAULT_ENV='base')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('base', result.stderr)

    def test_rejects_python_from_other_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_guard('require_conda_python', CONDA_PREFIX=directory,
                                    CONDA_DEFAULT_ENV='virdm', PYTHON_BIN=sys.executable)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('outside', result.stderr)

    def test_python_version_contract(self):
        result = self.run_guard('require_conda_python', CONDA_PREFIX=sys.prefix,
                                CONDA_DEFAULT_ENV='virdm', PYTHON_BIN=sys.executable)
        self.assertEqual(result.returncode == 0, sys.version_info[:2] == (3, 10))

    def test_missing_toolkit_does_not_fall_back_to_system(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_guard('require_conda_cuda 12.4', CONDA_PREFIX=directory)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing Conda nvcc', result.stderr)

    def test_wrong_toolkit_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            nvcc = Path(directory) / 'bin' / 'nvcc'
            nvcc.parent.mkdir()
            nvcc.write_text('#!/bin/sh\necho "Cuda compilation tools, release 13.0, V13.0"\n')
            nvcc.chmod(0o755)
            result = self.run_guard('require_conda_cuda 12.4', CONDA_PREFIX=directory)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Expected CUDA toolkit 12.4', result.stderr)

if __name__ == '__main__':
    unittest.main()
