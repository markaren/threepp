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
