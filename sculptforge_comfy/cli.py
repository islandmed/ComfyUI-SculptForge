# SculptForge - ComfyUI nodes: locating and running sculptforge-cli.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Every node is a short sculptforge-cli script. File paths never go through the script
# text: they are passed as --set variables ($IN, $OUT...), so spaces, quotes and
# backslashes in paths need no escaping. The CLI runs without a shell.
import os
import shutil
import subprocess
import sys

CLI_ENV = "SCULPTFORGE_CLI"
APP_ENV = "SCULPTFORGE_APP"
TIMEOUT_ENV = "SCULPTFORGE_TIMEOUT"
DEFAULT_TIMEOUT = 600

_EXE = ".exe" if sys.platform == "win32" else ""


class SculptForgeError(RuntimeError):
    """A SculptForge command failed (or the CLI could not be found)."""


def _install_dirs():
    """Folders the official packages install SculptForge into, most likely first."""
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        dirs = []
        for var in ("ProgramFiles", "ProgramW6432", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                dirs.append(os.path.join(base, "SculptForge"))
                dirs.append(os.path.join(base, "SculptForge", "bin"))
                dirs.append(os.path.join(base, "Programs", "SculptForge"))
        return dirs
    if sys.platform == "darwin":
        return [
            "/Applications/SculptForge.app/Contents/MacOS",
            os.path.join(home, "Applications/SculptForge.app/Contents/MacOS"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ]
    return [
        "/usr/bin",
        "/usr/local/bin",
        os.path.join(home, ".local/bin"),
        "/opt/sculptforge/bin",
        "/opt/SculptForge/bin",
        "/app/bin",  # Flatpak
    ]


def _find(name, env_var):
    override = os.environ.get(env_var, "").strip().strip('"')
    if override:
        if os.path.isdir(override):
            override = os.path.join(override, name + _EXE)
        if os.path.isfile(override):
            return override
        raise SculptForgeError(f"{env_var} is set to '{override}', but no such file exists.")
    found = shutil.which(name)
    if found:
        return found
    for folder in _install_dirs():
        candidate = os.path.join(folder, name + _EXE)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_cli():
    """Path of sculptforge-cli: $SCULPTFORGE_CLI, then PATH, then the usual install folders."""
    path = _find("sculptforge-cli", CLI_ENV)
    if not path:
        raise SculptForgeError(
            "sculptforge-cli was not found. Install SculptForge (https://github.com/islandmed/SculptForge) "
            f"and put sculptforge-cli on PATH, or set the {CLI_ENV} environment variable to its full path "
            "before starting ComfyUI."
        )
    return path


def find_app():
    """Path of the SculptForge desktop app: $SCULPTFORGE_APP, next to the CLI, PATH, install folders."""
    try:
        cli = find_cli()
    except SculptForgeError:
        cli = None
    if not os.environ.get(APP_ENV) and cli:
        sibling = os.path.join(os.path.dirname(cli), "sculptforge" + _EXE)
        if os.path.isfile(sibling):
            return sibling
    path = _find("sculptforge", APP_ENV)
    if not path:
        raise SculptForgeError(
            f"The SculptForge app was not found. Put sculptforge on PATH or set {APP_ENV} to its full path."
        )
    return path


def _timeout():
    try:
        return float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def run(script, variables=None):
    """Run a command script (statements separated by ';'), returning the CLI's stdout.

    variables maps names to values substituted for $NAME in the script (use them for paths).
    Raises SculptForgeError with the CLI's own message when a command fails.
    """
    cmd = [find_cli()]
    for name, value in (variables or {}).items():
        cmd += ["--set", f"{name}={value}"]
    cmd += ["-c", script]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=_timeout(), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise SculptForgeError(f"sculptforge-cli timed out after {_timeout():g} s running: {script}") from None
    except OSError as e:
        raise SculptForgeError(f"Could not run {cmd[0]}: {e}") from None
    if proc.returncode != 0:
        message = (proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}")
        raise SculptForgeError(f"SculptForge: {message}")
    return proc.stdout


def launch_app(path):
    """Open a file in the SculptForge desktop app without waiting for it."""
    app = find_app()
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([app, path], **kwargs)
    return app
