"""threepp-editor: the threepp scene editor, installed as a Python package.

Run it as `threepp-editor` (console script) or `python -m threepp_editor`.
The package is a launcher around a native executable — see cli.py for the
environment contract between the two.
"""
from .cli import exe_path, main

__all__ = ["exe_path", "main"]
