import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hcx_is_only_an_optional_registry_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    uv = pyproject["tool"]["uv"]

    assert project["optional-dependencies"]["hcx"] == ["hcx>=0.1.0"]
    assert "hcx" not in project["dependencies"]
    assert "hcx" not in pyproject["dependency-groups"]["dev"]
    assert "default-extras" not in uv
    assert "hcx" not in uv.get("sources", {})
    assert all("hcx" not in pyproject["dependency-groups"][group] for group in uv["default-groups"])

    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    hcx = next(package for package in lock["package"] if package["name"] == "hcx")
    assert hcx["source"] == {"registry": "https://pypi.org/simple"}

    hydrologeez = next(package for package in lock["package"] if package["name"] == "hydrologeez")
    assert hydrologeez["optional-dependencies"]["hcx"] == [{"name": "hcx"}]
    assert all(requirement["name"] != "hcx" for requirement in hydrologeez["dev-dependencies"]["dev"])


def test_core_and_adapter_import_without_hcx() -> None:
    code = textwrap.dedent(
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def import_without_hcx(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "hcx" or name.startswith("hcx."):
                raise ModuleNotFoundError("hcx intentionally unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_hcx
        import hydrologeez
        import hydrologeez.hcx
        assert "hcx" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
