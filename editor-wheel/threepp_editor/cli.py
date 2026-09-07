"""Launcher for the threepp editor executable.

The editor EMBEDS CPython for its scripting console, which creates the three
environment problems this launcher exists to solve — a bare double-click on
the exe inside site-packages would hit all of them:

1. python3XX.dll: the exe links the runtime DLL of the Python it was built
   for. In a venv that DLL lives in the BASE installation, not the venv, and
   nothing puts it on a child process's search path by default.
2. stdlib + site-packages: the embedded interpreter initialises from
   PYTHONHOME/PYTHONPATH. Point PYTHONHOME at the base installation (stdlib)
   and PYTHONPATH at the RUNNING environment's site-packages, so an editor
   script can `import numpy` from the venv the user actually installed into.
3. type stubs: the editor's "Edit in VS Code" hands Pylance the UNION stubs
   (threepp + the play-session physics handles). THREEPP_PYTHON_STUBS is the
   documented override; ship the stubs in the wheel and point it there.

Everything else — argv, cwd, exit code — passes straight through.
"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def exe_path() -> Path:
    """Path of the bundled editor executable (exists() is the install check)."""
    name = "threepp_editor.exe" if os.name == "nt" else "threepp_editor"
    return _PKG / "bin" / name


def _child_env() -> dict:
    env = os.environ.copy()

    # 1. The runtime DLL's home. In a venv, base_exec_prefix is the real
    # installation; outside one it equals exec_prefix, so this is always right.
    if os.name == "nt":
        env["PATH"] = sys.base_exec_prefix + os.pathsep + env.get("PATH", "")

    # 2. The embedded interpreter's world. PYTHONHOME gives it the stdlib of
    # the base installation; PYTHONPATH layers the running environment's
    # site-packages on top (sysconfig reports the venv's purelib when running
    # inside one). A user-set PYTHONPATH stays, appended after ours.
    env["PYTHONHOME"] = sys.base_prefix
    site_packages = sysconfig.get_paths()["purelib"]
    extra = env.get("PYTHONPATH")
    env["PYTHONPATH"] = site_packages + ((os.pathsep + extra) if extra else "")

    # 3. Stubs for the editor's VS Code integration — only if the user hasn't
    # pointed the variable somewhere deliberate (e.g. a source tree).
    env.setdefault("THREEPP_PYTHON_STUBS", str(_PKG / "stubs"))

    return env


def main() -> int:
    exe = exe_path()
    if not exe.is_file():
        print(f"threepp-editor: bundled executable not found at {exe}", file=sys.stderr)
        return 1
    result = subprocess.run([str(exe), *sys.argv[1:]], env=_child_env())
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
