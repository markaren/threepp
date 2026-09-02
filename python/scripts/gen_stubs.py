"""Regenerate the `threepp` type stubs.

Driven by the `threepp_stubs` CMake target (see ../CMakeLists.txt), but also
runnable standalone once the native module has been built into the package dir:

    python python/scripts/gen_stubs.py

Output layout — a PEP 561 stub *package* mirroring the native module's own
package structure:

    python/threepp/threepp/__init__.pyi   # the threepp.threepp module
    python/threepp/threepp/imgui.pyi      # the threepp.threepp.imgui submodule
    python/threepp/threepp/editor.pyi     # HAND-MAINTAINED — see HAND_MAINTAINED

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
EXPECTED = {"__init__.pyi", "editor.pyi", "imgui.pyi"}

# HAND-MAINTAINED stubs: restored verbatim after stubgen runs, never generated.
#
# editor.pyi is the one file the "stubs are a superset of the module" rule runs
# backwards on. `threepp.editor` is served by TWO modules built from the same
# binding sources: the wheel (python/CMakeLists.txt) and the editor app's
# embedded interpreter (apps/editor/CMakeLists.txt + scripting/ScriptModule.cpp).
# Only the editor compiles bind_editor_physics.cpp — RigidBody, SoftBody,
# Articulation and the three *_from_object lookups are handles onto a live
# PhysicsPlaySession, which nothing outside a running editor has. So the wheel
# has the spline half and the editor has both, and the stub has to describe the
# UNION for a script author's completion to be right.
#
# stubgen can only introspect the wheel, so regenerating this file silently
# deletes the physics half (41 symbols; _check_no_regression catches it). Edit it
# BY HAND to match bind_editor_physics.cpp, as commit c70e92be did.
HAND_MAINTAINED = {"editor.pyi"}

# pybind11 bakes a function's signature at def() time, so a parameter whose C++
# type is registered later in the binding TU is rendered as a raw C++ name
# ("threepp::Matrix4"). stubgen degrades those annotations to `...` and logs an
# ERROR. Silence the known set so --exit-code can turn anything *new* into a
# build failure. Shrinking this regex means fixing binding declaration order.
IGNORE_INVALID_EXPRESSIONS = r"threepp::|<threepp\.threepp\."

# An enum used as a DEFAULT ARGUMENT VALUE reaches stubgen only as its repr
# ("<CurveType.centripetal: 0>"), which carries no module or owning class. Every
# such enum needs a mapping here, or stubgen reports an invalid expression and
# --exit-code aborts the run before a single file is written. Nested enums take
# the owning class in the path.
# Every enum used as a py::arg DEFAULT needs an entry here. pybind11 renders
# such a default into the docstring as `<Enum.Value: 0>`, which stubgen cannot
# resolve to a type without being told where the enum lives — and it treats that
# as a hard error, so the whole stub is skipped. Enums that only ever appear as
# def_readwrite members or plain parameters do not need one.
ENUM_CLASS_LOCATIONS = [
    f"AnimationBlendMode:{MODULE}",
    f"CurveType:{MODULE}.CatmullRomCurve3",
    f"LeafShape:{MODULE}",
    f"BarkStyle:{MODULE}",
    f"SplatPoseSet:{MODULE}",
]

# A bound name that is a Python keyword cannot appear literally in a stub (or in
# user code), and one such name makes the whole file unparseable. No binding
# needs this today — `Blending.None` and `damp()`'s `lambda` were renamed at the
# py::arg/.value site — so this is a standing guard against the next one, which
# pybind11 will happily emit from any C++ identifier that collides with a Python
# keyword (`from`, `in`, `is`, `lambda`, `None`, ...).
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
        *[arg for loc in ENUM_CLASS_LOCATIONS for arg in ("--enum-class-locations", loc)],
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
    (`lambda`) are renamed and made positional-only. Expected to be a no-op
    against the current bindings — see the note by `_ANNOTATED_MEMBER`.

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

    A signature like `def f(x, lambda, dt)` cannot be written in Python. Renaming
    alone would lie — `f(lambda_=...)` raises TypeError when the binding really
    is `py::arg("lambda")` — so a PEP 570 `/` is inserted after the offending
    parameter. That is exactly the truth: it can only be passed positionally.

    The honest fix is at the binding site (`py::arg("lambda_")`), which makes the
    argument keyword-callable and leaves nothing here to repair.
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


def _strip_self_qualification(path: Path) -> int:
    """Drop the module's own `threepp.threepp.` prefix from its annotations.

    pybind11 spells every type fully qualified, and stubgen leaves that spelling
    alone inside a composed annotation (`Optional[...]`, `Union[...]`) — so
    `__init__.pyi` ends up referring to `threepp.threepp.Color` while importing
    no `threepp` at all. pyright resolves that to **Unknown**, silently: no
    completion, no checking, no error to notice it by.

    Bare names are right under both layouts the stubs serve — the wheel, where
    this file is `threepp.threepp`, and the editor's `stubPath`, where the very
    same file is imported as `threepp` (see doc/editor.md). Every type so
    qualified is declared in this file, so the prefix is pure self-reference.

    Docstrings are left alone: prose naming `threepp.threepp.Foo` means it.
    """
    prefix = f"{MODULE}."
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fixed = 0
    in_docstring = False
    for i, line in enumerate(lines):
        was_in_docstring = in_docstring
        if line.count('"""') % 2:
            in_docstring = not in_docstring
        if was_in_docstring or in_docstring or prefix not in line:
            continue
        lines[i] = line.replace(prefix, "")
        fixed += line.count(prefix)
    if fixed:
        path.write_text("".join(lines), encoding="utf-8")
    return fixed


def _validate(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as e:
        raise SystemExit(f"error: generated stub {path} does not parse: line {e.lineno}: {e.msg}")


def _symbols(path: Path) -> set[str] | None:
    """Qualified names a stub declares: `Class`, `Class.member`, `function`.

    None if the file is absent or unparseable (nothing to compare against).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None

    def bound(body) -> set[str]:
        out = set()
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out.add(node.target.id)
            elif isinstance(node, ast.Assign):
                out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        return out

    names = bound(tree.body)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.update(f"{node.name}.{n}" for n in bound(node.body))
    return names


def _snapshot(stub_dir: Path) -> dict[str, set[str]]:
    return {p.name: s for p in stub_dir.glob("*.pyi") if (s := _symbols(p)) is not None}


def _check_no_regression(before: dict[str, set[str]], stub_dir: Path, allow_removals: bool) -> None:
    """Fail if regeneration dropped symbols the committed stubs already had.

    The trap this exists for: the stubs must be generated from a module built
    with the SAME feature set as the one they were generated from last time. A
    default or GL-only rebuild drops VulkanRenderer, the PhysX world and every
    sensor binding — and the result still parses, so every other check here is
    happy while the committed stubs silently lose thousands of symbols.

    Comparing against what is already committed keeps this feature-set agnostic:
    a deliberate reduction is fine, it just has to be stated with
    --allow-removals.
    """
    removed: dict[str, set[str]] = {}
    added = 0
    for name, old in before.items():
        new = _symbols(stub_dir / name)
        if new is None:
            removed[name] = old
            continue
        if gone := old - new:
            removed[name] = gone
        added += len(new - old)

    if removed and not allow_removals:
        lines = [
            "error: regeneration REMOVED symbols that the committed stubs declare.",
            "",
            "       This usually means the module was rebuilt with a smaller feature",
            "       set than the stubs were last generated from (e.g. a GL-only build",
            "       dropping Vulkan/PhysX/sensor bindings). Rebuild threepp_py with",
            "       the full feature set and regenerate.",
            "",
        ]
        for name, gone in sorted(removed.items()):
            shown = sorted(gone)
            lines.append(f"       {name}: {len(gone)} removed")
            lines.extend(f"         - {s}" for s in shown[:15])
            if len(shown) > 15:
                lines.append(f"         ... and {len(shown) - 15} more")
        lines += ["", "       If the reduction is intended, re-run with --allow-removals."]
        raise SystemExit("\n".join(lines))

    if removed:
        total = sum(len(g) for g in removed.values())
        print(f"  (--allow-removals: {total} symbol(s) dropped)")
    if added:
        print(f"  ({added} symbol(s) added)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="the directory containing the `threepp` package (default: <repo>/python)",
    )
    parser.add_argument(
        "--allow-removals",
        action="store_true",
        help="permit regeneration to drop symbols the committed stubs declare "
        "(needed only when deliberately generating from a reduced-feature build)",
    )
    args = parser.parse_args()

    pkg_dir: Path = args.package_dir.resolve()
    stub_dir = pkg_dir / "threepp" / "threepp"

    _check_version()
    # Snapshot before stubgen overwrites, so a feature-set regression is visible.
    before = _snapshot(stub_dir) if stub_dir.is_dir() else {}
    # Verbatim bytes of the hand-maintained stubs — stubgen overwrites every file
    # it emits, and for these its output is the wrong (wheel-only) answer.
    preserved = {n: (stub_dir / n).read_bytes() for n in HAND_MAINTAINED if (stub_dir / n).is_file()}
    _run_stubgen(pkg_dir)

    produced = {p.name for p in stub_dir.glob("*.pyi")} if stub_dir.is_dir() else set()
    if produced != EXPECTED:
        raise SystemExit(
            f"error: expected stubs {sorted(EXPECTED)} in {stub_dir}, got {sorted(produced)}.\n"
            f"       If the native module gained or lost a submodule, update EXPECTED in {__file__}."
        )

    # After the EXPECTED check, so a hand-maintained stub whose submodule vanished
    # from the module still fails loudly rather than being restored over silence.
    for name, blob in preserved.items():
        (stub_dir / name).write_bytes(blob)

    for name in sorted(produced):
        path = stub_dir / name
        if name in preserved:
            print(f"  {path.relative_to(pkg_dir)}: hand-maintained, left as committed")
            _validate(path)
            continue
        repaired = _repair_keyword_names(path)
        unqualified = _strip_self_qualification(path)
        _validate(path)
        notes = []
        if repaired:
            notes.append(f"{repaired} keyword-named binding(s) repaired")
        if unqualified:
            notes.append(f"{unqualified} self-qualified name(s) unqualified")
        note = f" ({', '.join(notes)})" if notes else ""
        print(f"  {path.relative_to(pkg_dir)}: {len(path.read_text(encoding='utf-8').splitlines())} lines{note}")

    # After repair/validation: compare only stubs that are known to parse.
    _check_no_regression(before, stub_dir, args.allow_removals)

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
