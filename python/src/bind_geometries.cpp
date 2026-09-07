// Geometry factories. Each is a BufferGeometry subclass constructed through its
// float-argument `create(...)` overload, with defaults copied verbatim from the
// C++ headers so Python construction matches three.js.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/geometries.hpp"
// Not aggregated by geometries.hpp:
#include "threepp/geometries/ConvexGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

using namespace threepp;

namespace threepp_py {

    void init_geometries(py::module_& m) {

        py::class_<BoxGeometry, BufferGeometry, std::shared_ptr<BoxGeometry>>(m, "BoxGeometry")
                .def(py::init([](float w, float h, float d, unsigned ws, unsigned hs, unsigned ds) {
                    return BoxGeometry::create(w, h, d, ws, hs, ds);
                }),
                     py::arg("width") = 1.f, py::arg("height") = 1.f, py::arg("depth") = 1.f,
                     py::arg("width_segments") = 1, py::arg("height_segments") = 1, py::arg("depth_segments") = 1)
                .def_readonly("width", &BoxGeometry::width)
                .def_readonly("height", &BoxGeometry::height)
                .def_readonly("depth", &BoxGeometry::depth);

        py::class_<SphereGeometry, BufferGeometry, std::shared_ptr<SphereGeometry>>(m, "SphereGeometry")
                .def(py::init([](float radius, unsigned ws, unsigned hs, float phiStart, float phiLength, float thetaStart, float thetaLength) {
                    return SphereGeometry::create(radius, ws, hs, phiStart, phiLength, thetaStart, thetaLength);
                }),
                     py::arg("radius") = 1.f, py::arg("width_segments") = 16, py::arg("height_segments") = 12,
                     py::arg("phi_start") = 0.f, py::arg("phi_length") = math::TWO_PI,
                     py::arg("theta_start") = 0.f, py::arg("theta_length") = math::PI)
                .def_readonly("radius", &SphereGeometry::radius);

        py::class_<PlaneGeometry, BufferGeometry, std::shared_ptr<PlaneGeometry>>(m, "PlaneGeometry")
                .def(py::init([](float w, float h, unsigned ws, unsigned hs) {
                    return PlaneGeometry::create(w, h, ws, hs);
                }),
                     py::arg("width") = 1.f, py::arg("height") = 1.f,
                     py::arg("width_segments") = 1, py::arg("height_segments") = 1)
                .def_readonly("width", &PlaneGeometry::width)
                .def_readonly("height", &PlaneGeometry::height);

        py::class_<CylinderGeometry, BufferGeometry, std::shared_ptr<CylinderGeometry>>(m, "CylinderGeometry")
                .def(py::init([](float rt, float rb, float h, unsigned rs, unsigned hs, bool open, float ts, float tl) {
                    return CylinderGeometry::create(rt, rb, h, rs, hs, open, ts, tl);
                }),
                     py::arg("radius_top") = 1.f, py::arg("radius_bottom") = 1.f, py::arg("height") = 1.f,
                     py::arg("radial_segments") = 16, py::arg("height_segments") = 1, py::arg("open_ended") = false,
                     py::arg("theta_start") = 0.f, py::arg("theta_length") = math::TWO_PI);

        py::class_<ConeGeometry, BufferGeometry, std::shared_ptr<ConeGeometry>>(m, "ConeGeometry")
                .def(py::init([](float radius, float h, unsigned rs, unsigned hs, bool open, float ts, float tl) {
                    return ConeGeometry::create(radius, h, rs, hs, open, ts, tl);
                }),
                     py::arg("radius") = 1.f, py::arg("height") = 1.f,
                     py::arg("radial_segments") = 16, py::arg("height_segments") = 1, py::arg("open_ended") = false,
                     py::arg("theta_start") = 0.f, py::arg("theta_length") = math::TWO_PI);

        py::class_<CapsuleGeometry, BufferGeometry, std::shared_ptr<CapsuleGeometry>>(m, "CapsuleGeometry")
                .def(py::init([](float radius, float length, unsigned cs, unsigned rs) {
                    return CapsuleGeometry::create(radius, length, cs, rs);
                }),
                     py::arg("radius") = 0.5f, py::arg("length") = 1.f,
                     py::arg("cap_segments") = 8, py::arg("radial_segments") = 16)
                .def_readonly("radius", &CapsuleGeometry::radius)
                .def_readonly("length", &CapsuleGeometry::length);

        py::class_<TorusGeometry, BufferGeometry, std::shared_ptr<TorusGeometry>>(m, "TorusGeometry")
                .def(py::init([](float radius, float tube, unsigned rs, unsigned ts, float arc) {
                    return TorusGeometry::create(radius, tube, rs, ts, arc);
                }),
                     py::arg("radius") = 1.f, py::arg("tube") = 0.4f,
                     py::arg("radial_segments") = 20, py::arg("tubular_segments") = 64, py::arg("arc") = math::TWO_PI);

        py::class_<TorusKnotGeometry, BufferGeometry, std::shared_ptr<TorusKnotGeometry>>(m, "TorusKnotGeometry")
                .def(py::init([](float radius, float tube, unsigned ts, unsigned rs, unsigned p, unsigned q) {
                    return TorusKnotGeometry::create(radius, tube, ts, rs, p, q);
                }),
                     py::arg("radius") = 1.f, py::arg("tube") = 0.4f,
                     py::arg("tubular_segments") = 64, py::arg("radial_segments") = 16,
                     py::arg("p") = 2, py::arg("q") = 3);

        py::class_<CircleGeometry, BufferGeometry, std::shared_ptr<CircleGeometry>>(m, "CircleGeometry")
                .def(py::init([](float radius, unsigned segments, float thetaStart, float thetaLength) {
                    return CircleGeometry::create(radius, segments, thetaStart, thetaLength);
                }),
                     py::arg("radius") = 1.f, py::arg("segments") = 16,
                     py::arg("theta_start") = 0.f, py::arg("theta_length") = math::TWO_PI);

        py::class_<RingGeometry, BufferGeometry, std::shared_ptr<RingGeometry>>(m, "RingGeometry")
                .def(py::init([](float inner, float outer, unsigned ts, unsigned ps, float thetaStart, float thetaLength) {
                    return RingGeometry::create(inner, outer, ts, ps, thetaStart, thetaLength);
                }),
                     py::arg("inner_radius") = 0.5f, py::arg("outer_radius") = 1.f,
                     py::arg("theta_segments") = 16, py::arg("phi_segments") = 2,
                     py::arg("theta_start") = 0.f, py::arg("theta_length") = math::TWO_PI);

        py::class_<IcosahedronGeometry, BufferGeometry, std::shared_ptr<IcosahedronGeometry>>(m, "IcosahedronGeometry")
                .def(py::init([](float radius, unsigned detail) {
                    return IcosahedronGeometry::create(radius, detail);
                }),
                     py::arg("radius") = 1.f, py::arg("detail") = 0);

        py::class_<OctahedronGeometry, BufferGeometry, std::shared_ptr<OctahedronGeometry>>(m, "OctahedronGeometry")
                .def(py::init([](float radius, unsigned detail) {
                    return OctahedronGeometry::create(radius, detail);
                }),
                     py::arg("radius") = 1.f, py::arg("detail") = 0);

        // ---- ConvexGeometry --------------------------------------------------
        // Convex hull of an input point set (pass a list of Vector3).
        // contains_point() is a cheap point-in-hull test; tolerance < 0 uses the
        // geometry's default.
        py::class_<ConvexGeometry, BufferGeometry, std::shared_ptr<ConvexGeometry>>(m, "ConvexGeometry")
                .def(py::init([](const std::vector<Vector3>& points) {
                    return ConvexGeometry::create(points);
                }),
                     py::arg("points"))
                .def("contains_point", &ConvexGeometry::containsPoint, py::arg("point"), py::arg("tolerance") = -1.f);

        // ---- BufferGeometry merge / simplify utilities (free functions) ------
        // Merge static scene geometry into one draw, or weld / decimate imported
        // meshes for faster GPU sim. Each returns a new BufferGeometry.
        m.def("merge_buffer_geometries",
              [](const std::vector<std::shared_ptr<BufferGeometry>>& geometries, bool use_groups) {
                  return mergeBufferGeometries(geometries, use_groups);
              },
              py::arg("geometries"), py::arg("use_groups") = false);
        m.def("merge_vertices",
              [](const BufferGeometry& geometry, float tolerance) { return mergeVertices(geometry, tolerance); },
              py::arg("geometry"), py::arg("tolerance") = 1e-4f);
        m.def("simplify_geometry",
              [](const BufferGeometry& geometry, float ratio, float error) { return simplifyGeometry(geometry, ratio, error); },
              py::arg("geometry"), py::arg("ratio"), py::arg("error") = 1e-2f);
    }

}// namespace threepp_py
