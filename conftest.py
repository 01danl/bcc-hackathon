"""Pytest configuration.

Makes the repository root importable so that ``app.*`` packages resolve
when tests are run from anywhere (``pytest`` or ``python -m pytest``).
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
