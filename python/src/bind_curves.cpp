// Curves. CatmullRomCurve3 is the one the editor's spline authoring writes and
// a play-mode script reads back, so it is the one bound here.
//
// Bound as a CONCRETE LEAF with no registered base: Curve<Vector3> is an
// abstract class template, and registering intermediate bases is exactly the
// shape pybind11 mishandles in this codebase (see the note at the top of
// bind_objects.cpp). Every inherited method is therefore exposed through a
// handler taking `CatmullRomCurve3&`, which is always a correct self pointer.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"

#include <sstream>

using namespace threepp;

namespace threepp_py {

    void init_curves(py::module_& m) {

        py::class_<CatmullRomCurve3, std::shared_ptr<CatmullRomCurve3>> curve(m, "CatmullRomCurve3");

        py::enum_<CatmullRomCurve3::CurveType>(curve, "CurveType")
                .value("centripetal", CatmullRomCurve3::centripetal)
                .value("chordal", CatmullRomCurve3::chordal)
                .value("catmullrom", CatmullRomCurve3::catmullrom)
                .export_values();

        curve.def(py::init([](std::vector<Vector3> points, bool closed,
                              CatmullRomCurve3::CurveType type, float tension) {
                      return std::make_shared<CatmullRomCurve3>(std::move(points), closed, type, tension);
                  }),
                  py::arg("points") = std::vector<Vector3>{}, py::arg("closed") = false,
                  py::arg("curve_type") = CatmullRomCurve3::centripetal, py::arg("tension") = 0.5f)
                // A list copy in both directions (pybind11's stl caster): assign
                // the property to change the points, mutating what the getter
                // returned changes nothing.
                .def_readwrite("points", &CatmullRomCurve3::points)
                .def_readwrite("closed", &CatmullRomCurve3::closed)
                .def_readwrite("curve_type", &CatmullRomCurve3::curveType)
                // Applies to curve_type == catmullrom only, as in three.js.
                .def_readwrite("tension", &CatmullRomCurve3::tension)
                // Segments used to build the arc-length table get_point_at and
                // get_length work from.
                .def_readwrite("arc_length_divisions", &CatmullRomCurve3::arcLengthDivisions)
                .def(
                        "get_point", [](const CatmullRomCurve3& c, float t) {
                            Vector3 v;
                            c.getPoint(t, v);
                            return v;
                        },
                        py::arg("t"), "Point at curve parameter t in [0, 1] (NOT arc length).")
                .def(
                        "get_point_at", [](const CatmullRomCurve3& c, float u) {
                            Vector3 v;
                            c.getPointAt(u, v);
                            return v;
                        },
                        py::arg("u"), "Point at fraction u in [0, 1] of the ARC LENGTH — equidistant.")
                .def(
                        "get_tangent", [](const CatmullRomCurve3& c, float t) {
                            Vector3 v;
                            c.getTangent(t, v);
                            return v;
                        },
                        py::arg("t"), "Unit tangent at curve parameter t.")
                .def(
                        "get_tangent_at", [](const CatmullRomCurve3& c, float u) {
                            Vector3 v;
                            c.getTangentAt(u, v);
                            return v;
                        },
                        py::arg("u"), "Unit tangent at fraction u of the arc length.")
                .def(
                        "get_points", [](const CatmullRomCurve3& c, unsigned int divisions) {
                            return c.getPoints(divisions);
                        },
                        py::arg("divisions") = 5, "divisions + 1 points, evenly spaced in t.")
                .def(
                        "get_spaced_points", [](const CatmullRomCurve3& c, unsigned int divisions) {
                            return c.getSpacedPoints(divisions);
                        },
                        py::arg("divisions") = 5, "divisions + 1 points, evenly spaced along the curve.")
                .def(
                        "get_length", [](const CatmullRomCurve3& c) { return c.getLength(); },
                        "Total arc length, from the cached table.")
                .def(
                        "get_lengths", [](const CatmullRomCurve3& c) { return c.getLengths(); },
                        "Cumulative segment lengths, arc_length_divisions + 1 of them.")
                .def(
                        "update_arc_lengths", [](CatmullRomCurve3& c) { c.updateArcLengths(); },
                        "Rebuild the arc-length table after editing points in place.")
                .def("__repr__", [](const CatmullRomCurve3& c) {
                    std::ostringstream o;
                    o << "CatmullRomCurve3(points=" << c.points.size()
                      << ", closed=" << (c.closed ? "True" : "False")
                      << ", tension=" << c.tension << ")";
                    return o.str();
                });
    }

}// namespace threepp_py
