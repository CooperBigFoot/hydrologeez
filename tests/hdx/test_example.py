from __future__ import annotations

import runpy
from pathlib import Path


def test_hdx_quickstart_runs() -> None:
    repo_root = Path(__file__).parents[2]
    runpy.run_path(str(repo_root / "docs" / "examples" / "hdx_quickstart.py"))
