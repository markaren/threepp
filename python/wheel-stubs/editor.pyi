# The WHEEL's stub for threepp.editor — hand-maintained, like the union stub.
#
# threepp.editor is served by two different binaries built from the same
# binding sources, and they do not expose the same names:
#
#   the wheel (python/CMakeLists.txt)  compiles bind_editor.cpp only
#       -> SplinePath, spline_from_object. That is this file.
#   the editor app's embedded interpreter (apps/editor/CMakeLists.txt) also
#       compiles bind_editor_physics.cpp / _sensors / _camera / _authoring
#       -> the union described by threepp/threepp/editor.pyi, which the editor
#          hands to Pylance via THREEPP_EDITOR_PYTHON_STUBS.
#
# The wheel ships py.typed, so type checkers TRUST whatever stub is installed:
# shipping the union here would bless ~40 names that raise AttributeError at
# runtime. python/CMakeLists.txt therefore installs THIS file into the wheel as
# threepp/editor.pyi and excludes the union.
#
# Keeping it in sync: a name added to bind_editor.cpp (the wheel TU) belongs in
# BOTH stubs; a name added to the editor-only TUs belongs ONLY in the union.
# The class/function text below is copied verbatim from the union stub — edit
# it there first, then mirror here.
"""
The runtime face of editor-authored data.
"""
from __future__ import annotations
import typing
import threepp
__all__: list[str] = ['SplinePath', 'spline_from_object']
class SplinePath:
    def get_point_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> threepp.Vector3:
        """
        WORLD-SPACE point at fraction u in [0, 1] of the arc length.
        """
    def get_tangent_at(self, u: typing.SupportsFloat | typing.SupportsIndex) -> threepp.Vector3:
        """
        WORLD-SPACE unit tangent at fraction u of the arc length.
        """
    def get_length(self) -> float:
        """
        Arc length in the spline's LOCAL space.
        """
    def refresh(self) -> None:
        """
        Re-capture the control points and config. The world transform is live regardless.
        """
    @property
    def closed(self) -> bool:
        ...
    @property
    def tension(self) -> float:
        ...
    @property
    def curve_type(self) -> threepp.CatmullRomCurve3.CurveType:
        ...
    @property
    def curve(self) -> threepp.CatmullRomCurve3:
        """
        The captured LOCAL-SPACE CatmullRomCurve3.
        """
def spline_from_object(object: threepp.Object3D | None) -> SplinePath | None:
    """
    The SplinePath an authored spline describes, or None when `object` is not a spline or has fewer than two control points.
    """
