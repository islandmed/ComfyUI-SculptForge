# SculptForge - ComfyUI node tests: shared fixtures.
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os
import stat
import sys

import pytest

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK_DIR)

from sculptforge_comfy import cli, paths  # noqa: E402


@pytest.fixture(autouse=True)
def comfy_dirs(tmp_path, monkeypatch):
    """Point the input/output/temp folders at a fresh directory for every test."""
    def fallback(name):
        path = tmp_path / "comfy" / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr(paths, "folder_paths", None)
    monkeypatch.setattr(paths, "_fallback", fallback)
    return tmp_path / "comfy"


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Install tests/fake_cli.py as $SCULPTFORGE_CLI; returns a helper to read its call log."""
    if sys.platform == "win32":
        pytest.skip("the fake CLI is a shebang script")
    exe = tmp_path / "bin" / "sculptforge-cli"
    exe.parent.mkdir()
    with open(os.path.join(os.path.dirname(__file__), "fake_cli.py"), encoding="utf-8") as f:
        exe.write_text(f"#!{sys.executable}\n" + f.read())
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv(cli.CLI_ENV, str(exe))
    monkeypatch.setenv("FAKE_SF_LOG", str(log))

    class Calls:
        path = str(exe)

        def __call__(self):
            if not log.exists():
                return []
            return [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]

    return Calls()
