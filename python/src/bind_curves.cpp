// Curves and 2-D shapes.
//
// Curve<Vector2> / Curve<Vector3> are abstract and registered as the bases
// Curve2 / Curve3, so the sampling API is bound once and TubeGeometry accepts
// any Curve3. This is plain non-virtual inheritance; the virtual-base caveat
// in bind_objects.cpp does not apply.
#include "bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/extras/core/Shape.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/extras/curves/SplineCurve.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace threepp;

namespace threepp_py {

    namespace {

        using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

        // ── Array forms of the point API ───────────────────────────────────────
        // A curve through N points costs N Python objects each way through the
        // list-based API (a Vector3 per point in, a Vector3 per sample out). For
        // a 500-point polyline that marshalling dwarfs the curve maths, so the
        // constructors and the sampling calls also come in (N, D) float32 array
        // form: Vector2 rows are (x, y), Vector3 rows (x, y, z).
        template<class P>
        constexpr py::ssize_t dim_of() {
            return std::is_same_v<P, Vector3> ? 3 : 2;
        }

        template<class P>
        void writeRow(float* row, const P& v) {
            row[0] = v.x;
            row[1] = v.y;
            if constexpr (dim_of<P>() == 3) row[2] = v.z;
        }

        template<class P>
        P readRow(const float* row) {
            if constexpr (dim_of<P>() == 3) {
                return P(row[0], row[1], row[2]);
            } else {
                return P(row[0], row[1]);
            }
        }

        template<class P>
        py::array_t<float> pointsToArray(const std::vector<P>& points) {
            constexpr auto D = dim_of<P>();
            py::array_t<float> out({static_cast<py::ssize_t>(points.size()), D});
            float* d = out.mutable_data();
            for (std::size_t i = 0; i < points.size(); ++i) writeRow(d + i * D, points[i]);
            return out;
        }

        template<class P>
        std::vector<P> arrayToPoints(const FloatArray& a, const char* what) {
            constexpr auto D = dim_of<P>();
            if (a.ndim() != 2 || a.shape(1) != D) {
                throw std::invalid_argument(std::string(what) + ": expected an (N, " + std::to_string(D) +
                                            ") float array");
            }
            std::vector<P> points;
            points.reserve(static_cast<std::size_t>(a.shape(0)));
            const float* d = a.data();
            for (py::ssize_t i = 0; i < a.shape(0); ++i) points.push_back(readRow<P>(d + i * D));
            return points;
        }

        // One sample per entry of the 1-D fraction array `u`, as an (len(u), D) array.
        template<class T, class P, class Sample>
        py::array_t<float> sampleAt(const T& curve, const FloatArray& u, const char* what, Sample sample) {
            constexpr auto D = dim_of<P>();
            if (u.ndim() != 1) {
                throw std::invalid_argument(std::string(what) + ": u must be a 1-D array of arc-length fractions");
            }
            const py::ssize_t n = u.shape(0);
            py::array_t<float> out({n, D});
            float* d = out.mutable_data();
            const float* uu = u.data();
            P v;
            for (py::ssize_t i = 0; i < n; ++i) {
                sample(curve, uu[i], v);
                writeRow(d + i * D, v);
            }
            return out;
        }

        // Sampling API shared by every curve (threepp::Curve<T>). get_point takes
        // the curve parameter t; get_point_at takes a fraction of the arc length.
        template<class Cls>
        void bind_curve_api(Cls& c) {
            using T = typename Cls::type;
            using P = typename decltype(std::declval<T>().getPoints(1u))::value_type;

            c.def_readwrite("arc_length_divisions", &T::arcLengthDivisions,
                            "Segments used to build the arc-length table get_point_at and get_length work from.")
                    .def(
                            "get_point", [](const T& curve, float t) {
                                P v;
                                curve.getPoint(t, v);
                                return v;
                            },
                            py::arg("t"), "Point at curve parameter t in [0, 1] (NOT arc length).")
                    .def(
                            "get_point_at", [](const T& curve, float u) {
                                P v;
                                curve.getPointAt(u, v);
                                return v;
                            },
                            py::arg("u"), "Point at fraction u in [0, 1] of the ARC LENGTH — equidistant.")
                    .def(
                            "get_tangent", [](const T& curve, float t) {
                                P v;
                                curve.getTangent(t, v);
                                return v;
                            },
                            py::arg("t"), "Unit tangent at curve parameter t.")
                    .def(
                            "get_tangent_at", [](const T& curve, float u) {
                                P v;
                                curve.getTangentAt(u, v);
                                return v;
                            },
                            py::arg("u"), "Unit tangent at fraction u of the arc length.")
                    .def(
                            "get_points", [](const T& curve, unsigned int divisions) { return curve.getPoints(divisions); },
                            py::arg("divisions") = 5, "divisions + 1 points, evenly spaced in t.")
                    .def(
                            "get_spaced_points", [](const T& curve, unsigned int divisions) { return curve.getSpacedPoints(divisions); },
                            py::arg("divisions") = 5, "divisions + 1 points, evenly spaced along the curve.")
                    // Array forms: the same samples as one (N, D) float32 array, with no
                    // per-point Python object. get_points_at / get_tangents_at take the
                    // arc-length fractions as a 1-D array (get_spaced_points is
                    // get_points_at on an even grid).
                    .def(
                            "get_points_array", [](const T& curve, unsigned int divisions) {
                                return pointsToArray(curve.getPoints(divisions));
                            },
                            py::arg("divisions") = 5, "get_points as one (divisions + 1, D) float32 array.")
                    .def(
                            "get_spaced_points_array", [](const T& curve, unsigned int divisions) {
                                return pointsToArray(curve.getSpacedPoints(divisions));
                            },
                            py::arg("divisions") = 5, "get_spaced_points as one (divisions + 1, D) float32 array.")
                    .def(
                            "get_points_at", [](const T& curve, const FloatArray& u) {
                                return sampleAt<T, P>(curve, u, "get_points_at",
                                                      [](const T& c, float f, P& v) { c.getPointAt(f, v); });
                            },
                            py::arg("u"),
                            "get_point_at for every arc-length fraction in the 1-D array u, as one (len(u), D) float32 array.")
                    .def(
                            "get_tangents_at", [](const T& curve, const FloatArray& u) {
                                return sampleAt<T, P>(curve, u, "get_tangents_at",
                                                      [](const T& c, float f, P& v) { c.getTangentAt(f, v); });
                            },
                            py::arg("u"),
                            "get_tangent_at for every arc-length fraction in the 1-D array u, as one (len(u), D) float32 array.")
                    .def(
                            "get_length", [](const T& curve) { return curve.getLength(); },
                            "Total arc length, from the cached table.")
                    .def(
                            "get_lengths", [](const T& curve) { return curve.getLengths(); },
                            "Cumulative segment lengths, arc_length_divisions + 1 of them.")
                    .def(
                            "update_arc_lengths", [](T& curve) { curve.updateArcLengths(); },
                            "Rebuild the arc-length table after editing the curve in place.");
        }

    }// namespace

    void init_curves(py::module_& m) {

        // ---- 3-D curves ------------------------------------------------------
        // Abstract: no constructor.
        py::class_<Curve3, std::shared_ptr<Curve3>> curve3(m, "Curve3");
        bind_curve_api(curve3);

        py::class_<CatmullRomCurve3, Curve3, std::shared_ptr<CatmullRomCurve3>> catmullRom(m, "CatmullRomCurve3");

        py::enum_<CatmullRomCurve3::CurveType>(catmullRom, "CurveType")
                .value("centripetal", CatmullRomCurve3::centripetal)
                .value("chordal", CatmullRomCurve3::chordal)
                .value("catmullrom", CatmullRomCurve3::catmullrom)
                .export_values();

        catmullRom
                .def(py::init([](std::vector<Vector3> points, bool closed,
                                 CatmullRomCurve3::CurveType type, float tension) {
                         return std::make_shared<CatmullRomCurve3>(std::move(points), closed, type, tension);
                     }),
                     py::arg("points") = std::vector<Vector3>{}, py::arg("closed") = false,
                     py::arg("curve_type") = CatmullRomCurve3::centripetal, py::arg("tension") = 0.5f)
                // The constructor with the points as an (N, 3) float array. A named
                // factory rather than an overload so that a list of Vector3 and an
                // array never compete in overload resolution.
                .def_static(
                        "from_array", [](const FloatArray& points, bool closed,
                                         CatmullRomCurve3::CurveType type, float tension) {
                            return std::make_shared<CatmullRomCurve3>(
                                    arrayToPoints<Vector3>(points, "CatmullRomCurve3.from_array"), closed, type, tension);
                        },
                        py::arg("points"), py::arg("closed") = false,
                        py::arg("curve_type") = CatmullRomCurve3::centripetal, py::arg("tension") = 0.5f,
                        "The constructor with `points` as an (N, 3) float array instead of a list of Vector3.")
                // A list copy in both directions (pybind11's stl caster): assign
                // the property to change the points, mutating what the getter
                // returned changes nothing.
                .def_readwrite("points", &CatmullRomCurve3::points)
                .def_readwrite("closed", &CatmullRomCurve3::closed)
                .def_readwrite("curve_type", &CatmullRomCurve3::curveType)
                // Applies to curve_type == catmullrom only, as in three.js.
                .def_readwrite("tension", &CatmullRomCurve3::tension)
                .def("__repr__", [](const CatmullRomCurve3& c) {
                    std::ostringstream o;
                    o << "CatmullRomCurve3(points=" << c.points.size()
                      << ", closed=" << (c.closed ? "True" : "False")
                      << ", tension=" << c.tension << ")";
                    return o.str();
                });

        py::class_<LineCurve3, Curve3, std::shared_ptr<LineCurve3>>(m, "LineCurve3")
                .def(py::init([](const Vector3& v1, const Vector3& v2) {
                         return std::make_shared<LineCurve3>(v1, v2);
                     }),
                     py::arg("v1"), py::arg("v2"))
                .def_readwrite("v1", &LineCurve3::v1)
                .def_readwrite("v2", &LineCurve3::v2)
                .def("__repr__", [](const LineCurve3& c) {
                    std::ostringstream o;
                    o << "LineCurve3((" << c.v1.x << ", " << c.v1.y << ", " << c.v1.z
                      << ") -> (" << c.v2.x << ", " << c.v2.y << ", " << c.v2.z << "))";
                    return o.str();
                });

        // ---- 2-D curves ------------------------------------------------------
        py::class_<Curve2, std::shared_ptr<Curve2>> curve2(m, "Curve2");
        bind_curve_api(curve2);

        py::class_<LineCurve, Curve2, std::shared_ptr<LineCurve>>(m, "LineCurve")
                .def(py::init([](const Vector2& v1, const Vector2& v2) {
                         return std::make_shared<LineCurve>(v1, v2);
                     }),
                     py::arg("v1"), py::arg("v2"))
                .def_readwrite("v1", &LineCurve::v1)
                .def_readwrite("v2", &LineCurve::v2)
                .def("__repr__", [](const LineCurve& c) {
                    std::ostringstream o;
                    o << "LineCurve((" << c.v1.x << ", " << c.v1.y << ") -> (" << c.v2.x << ", " << c.v2.y << "))";
                    return o.str();
                });

        py::class_<SplineCurve, Curve2, std::shared_ptr<SplineCurve>>(m, "SplineCurve")
                .def(py::init([](std::vector<Vector2> points) {
                         return std::make_shared<SplineCurve>(std::move(points));
                     }),
                     py::arg("points") = std::vector<Vector2>{})
                .def_static(
                        "from_array", [](const FloatArray& points) {
                            return std::make_shared<SplineCurve>(arrayToPoints<Vector2>(points, "SplineCurve.from_array"));
                        },
                        py::arg("points"), "The constructor with `points` as an (N, 2) float array instead of a list of Vector2.")
                .def_readwrite("points", &SplineCurve::points)
                .def("__repr__", [](const SplineCurve& c) {
                    return "SplineCurve(points=" + std::to_string(c.points.size()) + ")";
                });

        // ---- Path / Shape ----------------------------------------------------
        // CurvePath<Vector2>: a chain of 2-D curve segments sampled as one curve.
        py::class_<Path, Curve2, std::shared_ptr<Path>> path(m, "Path");
        path.def(py::init([](const std::vector<Vector2>& points) { return std::make_shared<Path>(points); }),
                 py::arg("points") = std::vector<Vector2>{})
                .def_readwrite("current_point", &Path::currentPoint)
                .def_readwrite("auto_close", &Path::autoClose)
                .def("set_from_points", &Path::setFromPoints, py::arg("points"), py::return_value_policy::reference_internal)
                .def("move_to", &Path::moveTo, py::arg("x"), py::arg("y"), py::return_value_policy::reference_internal)
                .def("line_to", &Path::lineTo, py::arg("x"), py::arg("y"), py::return_value_policy::reference_internal)
                .def("quadratic_curve_to", &Path::quadraticCurveTo,
                     py::arg("cpx"), py::arg("cpy"), py::arg("x"), py::arg("y"), py::return_value_policy::reference_internal)
                .def("bezier_curve_to", &Path::bezierCurveTo,
                     py::arg("cp1x"), py::arg("cp1y"), py::arg("cp2x"), py::arg("cp2y"), py::arg("x"), py::arg("y"),
                     py::return_value_policy::reference_internal)
                .def("spline_thru", &Path::splineThru, py::arg("points"), py::return_value_policy::reference_internal)
                .def("absarc", &Path::absarc,
                     py::arg("x"), py::arg("y"), py::arg("radius"), py::arg("start_angle"), py::arg("end_angle"),
                     py::arg("clockwise") = false, py::return_value_policy::reference_internal)
                .def("close_path", &Path::closePath);

        // Closed outline with optional holes; the input to ExtrudeGeometry / ShapeGeometry.
        py::class_<Shape, Path, std::shared_ptr<Shape>>(m, "Shape")
                .def(py::init([](const std::vector<Vector2>& points) { return std::make_shared<Shape>(points); }),
                     py::arg("points") = std::vector<Vector2>{})
                .def_property_readonly("uuid", &Shape::uuid)
                // A list copy of the shared Path handles: editing a Path from the list edits
                // the hole; appending to the list does not add one (assign the property).
                .def_readwrite("holes", &Shape::holes)
                .def("extract_points", [](const Shape& s, unsigned int divisions) {
                    const auto points = s.extractPoints(divisions);
                    return py::make_tuple(points.shape, points.holes);
                }, py::arg("divisions") = 12,
                   "(outline, holes) as point lists, sampling every curve segment with `divisions` steps.")
                .def("__repr__", [](const Shape& s) {
                    return "Shape(curves=" + std::to_string(s.curves.size()) +
                           ", holes=" + std::to_string(s.holes.size()) + ")";
                });
    }

}// namespace threepp_py
