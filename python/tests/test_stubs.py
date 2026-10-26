"""The committed type stubs declare every name the native module binds.

python/threepp/threepp/__init__.pyi is generated from a full-featured build
(python/scripts/gen_stubs.py, the threepp_stubs target) and committed. A binding
added without regenerating it is invisible to type checkers and editor
completion, and nothing else notices: in October 2026 twelve VulkanRenderer
members (the shadow-map and displacement settings) were bound for days with no
stub.

The check runs against any build. One with fewer features binds fewer names,
and every name it does bind must be declared; the other direction (a stub name
the module lacks) is a feature difference, not drift.
"""
import ast
import inspect
import types
from pathlib import Path

import threepp as tp

STUB = Path(__file__).resolve().parents[1] / "threepp" / "threepp" / "__init__.pyi"


def _declared(body, prefix, out):
    """Names a stub body binds, qualified by `prefix`; recurses into classes."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(prefix + node.name)
        elif isinstance(node, ast.ClassDef):
            out.add(prefix + node.name)
            _declared(node.body, f"{prefix}{node.name}.", out)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(prefix + node.target.id)
        elif isinstance(node, ast.Assign):
            out.update(prefix + t.id for t in node.targets if isinstance(t, ast.Name))
    return out


def _bound(namespace, prefix, owner, out):
    """Public names the module binds, qualified like _declared's."""
    for name, obj in namespace.items():
        if name.startswith("_") or isinstance(obj, types.ModuleType):
            continue
        out.add(prefix + name)
        # Only classes defined here: an attribute aliasing another class
        # would otherwise be walked twice, or forever.
        if inspect.isclass(obj) and obj.__qualname__ == (f"{owner}.{name}" if owner else name):
            _bound(vars(obj), f"{prefix}{name}.", obj.__qualname__, out)
    return out


def test_stub_declares_every_bound_name():
    declared = _declared(ast.parse(STUB.read_text(encoding="utf-8")).body, "", set())
    bound = _bound(vars(tp._native), "", "", set())
    missing = sorted(bound - declared)
    assert not missing, (
        f"{len(missing)} bound name(s) missing from {STUB.name}; regenerate the stubs "
        f"(build the threepp_stubs target, or run python/scripts/gen_stubs.py):\n  "
        + "\n  ".join(missing[:40])
        + (f"\n  ... and {len(missing) - 40} more" if len(missing) > 40 else "")
    )
