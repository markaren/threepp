"""Where the turbine demo writes its renders, clips and survey data.

Default: an `out/` folder beside these scripts (gitignored). Set TURBINE_OUT to put it elsewhere.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TURBINE_OUT") or os.path.join(_HERE, "out")


def out_dir(sub=""):
    """ROOT/sub, created on demand: 'site', 'fleet', 'jobs', 'film'."""
    path = os.path.join(ROOT, sub) if sub else ROOT
    os.makedirs(path, exist_ok=True)
    return path


def latest(sub, stem, ext="png"):
    """The highest-numbered ROOT/sub/<stem>_vNN.<ext>, or None."""
    folder = os.path.join(ROOT, sub)
    if not os.path.isdir(folder):
        return None
    hits = sorted(f for f in os.listdir(folder) if f.startswith(stem + "_v") and f.endswith("." + ext))
    return os.path.join(folder, hits[-1]) if hits else None
