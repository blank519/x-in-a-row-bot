"""Pytest configuration for the test suite.

The project modules live at the repository root (flat layout, e.g.
``x_in_a_row_sb3_env.py``), so make sure that directory is importable when the
tests run, regardless of pytest's import mode or the working directory.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
