"""The run manifest: what produced the files in a run folder, written by every script of the harness.

manifest.json in the run folder holds one entry per script (its file name without .py), so the capture, the
training runs and the scoring of one run sit side by side:
  commit hash and dirty flag of every git repo involved, the versions of Python, numpy, torch, gsplat (each as
  far as the running interpreter has it), the CUDA toolkit (nvcc), the driver and the GPU (nvidia-smi), the full
  argv, the seed, a UTC timestamp, and whatever the script adds under "extra".

    from manifest import write_manifest
    write_manifest(run_dir, repos=[threepp_root, twin_root], seed=args.seed, extra={...})
"""
import datetime
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THREEPP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def _run(cmd, cwd=None, strip=True):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return (r.stdout.strip() if strip else r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def repo_state(path):
    """{path name, commit, branch, dirty} of the git repo at path (dirty: any tracked change; untracked files are
    listed by count only)."""
    root = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if root is None:
        return {"path": os.path.basename(os.path.abspath(path)), "git": None}
    status = _run(["git", "status", "--porcelain"], cwd=root, strip=False) or ""   # porcelain lines start with a space
    lines = [s for s in status.splitlines() if s]
    return {"repo": os.path.basename(root),
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=root),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
            "dirty": any(not s.startswith("??") for s in lines),
            "changed_tracked": sorted(s[3:] for s in lines if not s.startswith("??")),
            "untracked_count": sum(s.startswith("??") for s in lines)}


def _version(mod):
    if mod not in sys.modules:
        try:
            importlib.import_module(mod)
        except Exception:                                          # noqa: BLE001  (absent here: say so)
            return None
    return getattr(sys.modules[mod], "__version__", "unknown")


def gpu_state():
    q = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    out = {"nvidia_smi": q}
    nvcc = None
    for var in ["CUDA_HOME", "CUDA_PATH"] + sorted((k for k in os.environ if k.startswith("CUDA_PATH_V")), reverse=True):
        exe = os.path.join(os.environ.get(var, ""), "bin", "nvcc.exe" if os.name == "nt" else "nvcc")
        if os.environ.get(var) and os.path.isfile(exe):
            nvcc = exe
            break
    nvcc = nvcc or shutil.which("nvcc")
    if nvcc:
        v = _run([nvcc, "--version"]) or ""
        out["cuda_toolkit"] = next((ln.strip() for ln in v.splitlines() if "release" in ln), v[-80:])
    return out


def write_manifest(run_dir, repos=(), seed=None, extra=None, name=None, modules=("numpy", "torch", "gsplat", "scipy", "PIL")):
    """Add this script's entry to run_dir/manifest.json and return it."""
    os.makedirs(run_dir, exist_ok=True)
    name = name or os.path.splitext(os.path.basename(sys.argv[0]))[0]
    roots = [THREEPP_ROOT] + [r for r in repos if r]
    seen, states = set(), []
    for r in roots:
        st = repo_state(r)
        key = st.get("commit"), st.get("repo")
        if key not in seen:
            seen.add(key)
            states.append(st)
    entry = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {m: _version(m) for m in modules},
        "repos": states,
        "gpu": gpu_state(),
        "extra": extra or {},
    }
    path = os.path.join(run_dir, "manifest.json")
    data = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[name] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return entry
