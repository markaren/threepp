// Cameras. Both concrete cameras derive from Camera, which derives from
// Object3D (so position/lookAt come from the core bindings). `near`/`far` are
// exposed under their three.js names in addition to the threepp field names.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Raycaster.hpp"

#include <limits>
#include <sstream>
#include <utility>
#include <vector>

using namespace threepp;

namespace threepp_py {

    void init_cameras(py::module_& m) {

        // ---- Camera (abstract base) -----------------------------------------
        py::class_<Camera, Object3D, std::shared_ptr<Camera>>(m, "Camera")
                .def_readwrite("zoom", &Camera::zoom)
                .def_readwrite("near_plane", &Camera::nearPlane)
                .def_readwrite("far_plane", &Camera::farPlane)
                .def_property("near", [](const Camera& c) { return c.nearPlane; }, [](Camera& c, float v) { c.nearPlane = v; })
                .def_property("far", [](const Camera& c) { return c.farPlane; }, [](Camera& c, float v) { c.farPlane = v; })
                // Intrinsics / extrinsics, returned as Matrix4 (use .to_numpy() for a
                // (4,4) array). The getters refresh the matrices first so a freshly
                // moved/reconfigured camera reads correctly without an explicit update:
                //   projection_matrix      — clip-from-view (the projection; pinhole K)
                //   matrix_world_inverse   — view matrix (world -> camera)
                //   matrix_world (Object3D)— camera pose (camera -> world, extrinsics)
                .def_property_readonly("projection_matrix", [](Camera& c) {
                    c.updateProjectionMatrix();
                    return c.projectionMatrix;
                })
                .def_property_readonly("projection_matrix_inverse", [](Camera& c) {
                    c.updateProjectionMatrix();
                    return c.projectionMatrixInverse;
                })
                .def_property_readonly("matrix_world_inverse", [](Camera& c) {
                    c.updateMatrixWorld(true);
                    return c.matrixWorldInverse;
                })
                .def("update_projection_matrix", &Camera::updateProjectionMatrix);

        // ---- PerspectiveCamera ----------------------------------------------
        py::class_<PerspectiveCamera, Camera, std::shared_ptr<PerspectiveCamera>>(m, "PerspectiveCamera")
                .def(py::init([](float fov, float aspect, float near, float far) {
                    return PerspectiveCamera::create(fov, aspect, near, far);
                }),
                     py::arg("fov") = 60.f, py::arg("aspect") = 1.f,
                     py::arg("near") = 0.1f, py::arg("far") = 2000.f)
                .def_readwrite("fov", &PerspectiveCamera::fov)
                .def_readwrite("aspect", &PerspectiveCamera::aspect)
                .def_readwrite("focus", &PerspectiveCamera::focus)
                .def_readwrite("film_gauge", &PerspectiveCamera::filmGauge)
                .def_readwrite("film_offset", &PerspectiveCamera::filmOffset)
                .def("set_focal_length", &PerspectiveCamera::setFocalLength, py::arg("focal_length"))
                .def("get_focal_length", &PerspectiveCamera::getFocalLength)
                .def("update_projection_matrix", &PerspectiveCamera::updateProjectionMatrix);

        // ---- OrthographicCamera ---------------------------------------------
        py::class_<OrthographicCamera, Camera, std::shared_ptr<OrthographicCamera>>(m, "OrthographicCamera")
                .def(py::init([](float left, float right, float top, float bottom, float near, float far) {
                    return OrthographicCamera::create(left, right, top, bottom, near, far);
                }),
                     py::arg("left") = -1.f, py::arg("right") = 1.f,
                     py::arg("top") = 1.f, py::arg("bottom") = -1.f,
                     py::arg("near") = 0.1f, py::arg("far") = 2000.f)
                .def_readwrite("left", &OrthographicCamera::left)
                .def_readwrite("right", &OrthographicCamera::right)
                .def_readwrite("top", &OrthographicCamera::top)
                .def_readwrite("bottom", &OrthographicCamera::bottom)
                .def("update_projection_matrix", &OrthographicCamera::updateProjectionMatrix);

        // ---- Raycaster -------------------------------------------------------
        // Bound here rather than in init_core: set_from_camera takes a Camera, and
        // pybind11 resolves the type names in a signature (and so in the generated
        // stub) when the method is defined, so Camera must already be registered.
        py::class_<Intersection>(m, "Intersection",
                                 "One ray hit, as returned by Raycaster.intersect_object(s).")
                .def_readonly("distance", &Intersection::distance, "Distance from the ray origin to the hit (world units).")
                .def_property_readonly("point", [](const Intersection& i) { return i.point; }, "The hit point in world space.")
                .def_property_readonly("object", py::cpp_function([](const Intersection& i) { return i.object; }, py::return_value_policy::reference),
                                       "The intersected object: the same Python object you added to the scene. It "
                                       "must still be alive when this is read.")
                .def_property_readonly("face_index", [](const Intersection& i) { return i.faceIndex; }, "Triangle index for a mesh hit, else None.")
                .def_property_readonly("instance_id", [](const Intersection& i) { return i.instanceId; }, "Instance index for an InstancedMesh hit, else None.")
                .def_property_readonly("index", [](const Intersection& i) { return i.index; }, "Vertex/segment index for a Points or Line hit, else None.")
                .def_property_readonly("uv", [](const Intersection& i) { return i.uv; }, "Interpolated uv at the hit when the geometry has uvs, else None.")
                .def_property_readonly("distance_to_ray", [](const Intersection& i) { return i.distanceToRay; }, "For Points hits: the point's distance to the ray, else None.")
                .def("__repr__", [](const Intersection& i) {
                    std::ostringstream ss;
                    ss << "Intersection(distance=" << i.distance;
                    if (i.object) ss << ", object=<" << i.object->type() << " name='" << i.object->name << "'>";
                    ss << ")";
                    return ss.str();
                });

        py::class_<Raycaster>(m, "Raycaster",
                              "Ray picking (three.js Raycaster). Intersections are sorted nearest first. World "
                              "matrices are read as they are: after moving objects or the camera, rendering a frame "
                              "or calling update_matrix_world() on them brings them up to date.")
                .def(py::init<const Vector3&, const Vector3&, float, float>(),
                     py::arg("origin") = Vector3(), py::arg("direction") = Vector3(),
                     py::arg("near") = 0.f, py::arg("far") = std::numeric_limits<float>::infinity(),
                     "A ray from origin along direction (normalized), reporting hits between near and far.")
                .def("set", &Raycaster::set, py::arg("origin"), py::arg("direction"),
                     "Re-aim the ray: origin and a normalized direction.")
                .def("set_from_camera", [](Raycaster& r, const Vector2& coords, Camera& camera) {
                    r.setFromCamera(coords, camera);
                }, py::arg("coords"), py::arg("camera"), py::keep_alive<1, 3>(),
                   "Aim the ray from the camera through coords in NDC [-1, 1] (x right, y up). From a "
                   "window pixel (x, y): nx = 2*x/w - 1, ny = 1 - 2*y/h.")
                .def("set_from_camera", [](Raycaster& r, const std::pair<float, float>& coords, Camera& camera) {
                    r.setFromCamera(Vector2(coords.first, coords.second), camera);
                }, py::arg("coords"), py::arg("camera"), py::keep_alive<1, 3>(),
                   "Same, with coords as an (x, y) pair in NDC.")
                .def("intersect_object", [](Raycaster& r, const py::handle& object, bool recursive) {
                    return r.intersectObject(*as_object3d(object), recursive);
                }, py::arg("object"), py::arg("recursive") = false,
                   "Hits on object (and its descendants when recursive), nearest first; [] for a miss.")
                .def("intersect_objects", [](Raycaster& r, const py::sequence& objects, bool recursive) {
                    std::vector<std::shared_ptr<Object3D>> held;
                    std::vector<Object3D*> ptrs;
                    held.reserve(py::len(objects));
                    for (const auto& o : objects) {
                        held.push_back(as_object3d(o));
                        ptrs.push_back(held.back().get());
                    }
                    return r.intersectObjects(ptrs, recursive);
                }, py::arg("objects"), py::arg("recursive") = false,
                   "Hits on any of objects (and their descendants when recursive), nearest first.")
                .def_readwrite("near", &Raycaster::nearPlane, "Hits closer than this are ignored.")
                .def_readwrite("far", &Raycaster::farPlane, "Hits farther than this are ignored.")
                .def_readwrite("ray", &Raycaster::ray, "The ray itself (a live view: edits re-aim the raycaster).")
                .def_readwrite("layers", &Raycaster::layers, "Only objects sharing a layer channel with this are tested.")
                .def_property("line_threshold",
                              [](const Raycaster& r) { return r.params.lineThreshold; },
                              [](Raycaster& r, float v) { r.params.lineThreshold = v; },
                              "Hit tolerance for Line objects (world units).")
                .def_property("points_threshold",
                              [](const Raycaster& r) { return r.params.pointsThreshold; },
                              [](Raycaster& r, float v) { r.params.pointsThreshold = v; },
                              "Hit tolerance for Points objects (world units).");
    }

}// namespace threepp_py
