"""Shared utility helpers for threepp Python demos."""
import pathlib
import shutil
import urllib.request


def fetch_file(url, cache_dir, name=None, *, user_agent="threepp"):
    """Download *url* to *cache_dir*/*name* once; return the local path as str.

    *name* defaults to the last path component of the URL.
    The directory is created if absent.  An existing non-empty file is reused
    without re-downloading.

    Example::

        path = fetch_file("https://example.com/noon_grass_2k.hdr",
                          "~/.cache/threepp/hdri", "noon_grass_2k.hdr")
    """
    cache_dir = pathlib.Path(cache_dir).expanduser()
    if name is None:
        name = url.rsplit("/", 1)[-1].split("?")[0] or "file"
    dest = cache_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)
    return str(dest)
