"""Regenerate the `threepp` type stubs.

Driven by the `threepp_stubs` CMake target (see ../CMakeLists.txt), but also
runnable standalone once the native module has been built into the package dir:

    python python/scripts/gen_stubs.py

Output layout — a PEP 561 stub *package* mirroring the native module's own
package structure:

    python/threepp/threepp/__init__.pyi   # the threepp.threepp module
    python/threepp/threepp/imgui.pyi      # the threepp.threepp.imgui submodule

A single flat `threepp.pyi` cannot express the `imgui` submodule, which is why
the package dir is the layout this repo standardises on. The dir sits next to
`threepp.<abi>.pyd`/`.so` but does NOT shadow it at import time: it holds no
`__init__.py`, so it is only a namespace-package candidate, and Python's
FileFinder prefers the extension-module loader within the same path entry.

This wrapper exists so the generation is reproducible rather than
version-dependent: it pins the generator, normalises the flags, repairs the one
construct pybind11-stubgen cannot express, and *validates* the result — the
plain `pybind11-stubgen` CLI exits 0 even when it writes a stub that no type
checker can parse.
"""

from __future__ import annotations

import argparse
import ast
import keyword
import os
import re
import subprocess
import sys
from pathlib import Path

# Pinned so the emitted layout/format does not drift under us. Keep in step with
# python/requirements-stubs.txt and the README's regeneration recipe.
PINNED_STUBGEN = "2.5.5"

MODULE = "threepp.threepp"

# Files the generator is expected to produce, relative to the package dir. An
# exact match is required: a new native submodule (or a dropped one) should
# surface here as a hard failure rather than as a silently half-updated stub.
EXPECTED = {"__init__.pyi", "imgui.pyi"}

# pybind11 bakes a function's signature at def() time, so a parameter whose C++
# type is registered later in the binding TU is rendered as a raw C++ name
# ("threepp::Matrix4"). stubgen degrades those annotations to `...` and logs an
# ERROR. Silence the known set so --exit-code can turn anything *new* into a
# build failure. Shrinking this regex means fixing binding declaration order.
IGNORE_INVALID_EXPRESSIONS = r"threepp::|<threepp\.threepp\."

# AnimationBlendMode is used as a default argument value before stubgen can
# infer where the enum class lives; point it at the right module.
ENUM_CLASS_LOCATIONS = f"AnimationBlendMode:{MODULE}"

# A handful of bound names are Python keywords, so they cannot appear literally
# in a stub (or in user code). Today: the `Blending.None` enum value and
# `damp()`'s `lambda` parameter — see the "keyword-named bindings" note in
# ../README.md. Repaired here rather than left to break the whole file.
_ANNOTATED_MEMBER = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s")
_DEF_LINE = re.compile(r"^(\s*def\s+[A-Za-z_][A-Za-z0-9_]*\()(.*)(\).*:)\s*$")


def _check_version() -> None:
    try:
        from importlib.metadata import version

        found = version("pybind11-stubgen")
    except Exception:
        print("warning: could not determine the installed pybind11-stubgen version", file=sys.stderr)
        return
    if found != PINNED_STUBGEN:
        print(
            f"warning: pybind11-stubgen {found} is installed, but this repo pins {PINNED_STUBGEN}.\n"
            f"         The emitted stub layout/format may differ. Install the pin with:\n"
            f"             pip install -r python/requirements-stubs.txt",
            file=sys.stderr,
        )


def _run_stubgen(pkg_dir: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(pkg_dir), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    cmd = [
        sys.executable,
        "-m",
        "pybind11_stubgen",
        MODULE,
        "--output-dir",
        str(pkg_dir),
        "--enum-class-locations",
        ENUM_CLASS_LOCATIONS,
        "--ignore-invalid-expressions",
        IGNORE_INVALID_EXPRESSIONS,
        # Exit non-zero on any error stubgen did not expect, and skip writing
        # rather than emit a half-broken stub.
        "--exit-code",
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=pkg_dir, env=env, check=True)


def _repair_keyword_names(path: Path) -> int:
    """Repair bound names that are Python keywords.

    Members (`None: ClassVar[Blending]`) are commented out; parameters
    (`lambda`) are renamed and made positional-only.

    Line-based, because the file does not parse yet — that is the whole point.
    Triple-quote state is tracked so a docstring that happens to contain a
    `Something:` line is never rewritten.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fixed = 0
    in_docstring = False
    for i, line in enumerate(lines):
        was_in_docstring = in_docstring
        if line.count('"""') % 2:
            in_docstring = not in_docstring
        if was_in_docstring or in_docstring:
            continue

        m = _ANNOTATED_MEMBER.match(line)
        if m and keyword.iskeyword(m.group(2)):
            indent, name = m.group(1), m.group(2)
            lines[i] = (
                f"{indent}# {line.strip()}\n"
                f"{indent}# NOTE: '{name}' is a Python keyword — unusable as an attribute and\n"
                f"{indent}# not expressible in a stub. Reach it via getattr() if you need it.\n"
            )
            fixed += 1
            continue

        repaired = _fix_keyword_params(line)
        if repaired is not None:
            lines[i] = repaired
            fixed += 1
    if fixed:
        path.write_text("".join(lines), encoding="utf-8")
    return fixed


def _split_params(params: str) -> list[str]:
    """Split a signature's parameter list on top-level commas.

    Bracket-aware: annotations such as
    `typing.Annotated[numpy.typing.ArrayLike, numpy.float32]` contain commas.
    """
    out, depth, start = [], 0, 0
    for i, ch in enumerate(params):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(params[start:i])
            start = i + 1
    out.append(params[start:])
    return out


def _param_name(param: str) -> str:
    return param.strip().lstrip("*").split(":")[0].split("=")[0].strip()


def _fix_keyword_params(line: str) -> str | None:
    """Rename keyword-named parameters and make them positional-only.

    `def damp(x, y, lambda, dt)` cannot be written in Python. Renaming alone
    would lie — `damp(lambda_=...)` is a TypeError at runtime — so a PEP 570 `/`
    is inserted after the offending parameter. That is exactly the truth: the
    argument can only ever be passed positionally.
    """
    m = _DEF_LINE.match(line)
    if not m:
        return None
    head, params, tail = m.groups()
    if not params.strip():
        return None
    parts = _split_params(params)
    last_kw = -1
    for i, part in enumerate(parts):
        name = _param_name(part)
        if keyword.iskeyword(name):
            parts[i] = part.replace(name, name + "_", 1)
            last_kw = i
    if last_kw < 0:
        return None
    # Only add the marker if a `/` is not already in effect for that position.
    if not any(p.strip() == "/" for p in parts[last_kw + 1 :]):
        parts.insert(last_kw + 1, " /")
    return f"{head}{','.join(parts)}{tail}\n"


def _validate(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as e:
        raise SystemExit(f"error: generated stub {path} does not parse: line {e.lineno}: {e.msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="the directory containing the `threepp` package (default: <repo>/python)",
    )
    args = parser.parse_args()

    pkg_dir: Path = args.package_dir.resolve()
    stub_dir = pkg_dir / "threepp" / "threepp"

    _check_version()
    _run_stubgen(pkg_dir)

    produced = {p.name for p in stub_dir.glob("*.pyi")} if stub_dir.is_dir() else set()
    if produced != EXPECTED:
        raise SystemExit(
            f"error: expected stubs {sorted(EXPECTED)} in {stub_dir}, got {sorted(produced)}.\n"
            f"       If the native module gained or lost a submodule, update EXPECTED in {__file__}."
        )

    for name in sorted(produced):
        path = stub_dir / name
        repaired = _repair_keyword_names(path)
        _validate(path)
        note = f" ({repaired} keyword-named binding(s) repaired)" if repaired else ""
        print(f"  {path.relative_to(pkg_dir)}: {len(path.read_text(encoding='utf-8').splitlines())} lines{note}")

    legacy = pkg_dir / "threepp" / "threepp.pyi"
    if legacy.exists():
        print(
            f"warning: the legacy flat stub {legacy} still exists and now shadows nothing;\n"
            f"         it is superseded by {stub_dir} and should be deleted.",
            file=sys.stderr,
        )

    print("stubs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
