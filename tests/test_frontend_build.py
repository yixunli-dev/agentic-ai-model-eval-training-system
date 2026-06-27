import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_build_passes_when_dependencies_are_installed():
    frontend_dir = Path("frontend")
    if shutil.which("npm") is None:
        pytest.skip("npm is not installed")
    if not (frontend_dir / "node_modules").exists():
        pytest.skip("frontend dependencies are not installed")

    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=frontend_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
