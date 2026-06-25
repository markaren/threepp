// Math value types: Vector2/3/4, Color, Euler, Quaternion, Matrix3/4, Box3.
//
// These mirror three.js' math classes. They are plain value types (no shared
// holder); pybind11 copies them by value across the boundary, except when
// reached as a member of an Object3D (e.g. mesh.position), where def_readwrite
// hands back a reference so `mesh.position.x = 1` mutates in place.
#include "bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"// complete type for Vector3::project/unproject
#include "threepp/math/Box3.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/core/Object3D.hpp"// complete type for Box3::setFromObject

#include <algorithm>
#include <sstream>

using namespace threepp;

namespace threepp_py {

    void init_math(py::module_& m) {

        // ---- Vector2 ---------------------------------------------------------
        py::class_<Vector2>(m, "Vector2")
                .def(py::init([](float x, float y) { return Vector2(x, y); }),
                     py::arg("x") = 0.f, py::arg("y") = 0.f)
                .def_readwrite("x", &Vector2::x)
                .def_readwrite("y", &Vector2::y)
                .def("set", &Vector2::set, py::arg("x"), py::arg("y"), py::return_value_policy::reference_internal)
                .def("copy", &Vector2::copy, py::arg("v"), py::return_value_policy::reference_internal)
                .def("clone", [](const Vector2& v) { return Vector2(v.x, v.y); })
                .def("length", &Vector2::length)
                .def("normalize", &Vector2::normalize, py::return_value_policy::reference_internal)
                .def("dot", &Vector2::dot, py::arg("v"))
                .def("distance_to", &Vector2::distanceTo, py::arg("v"))
                .def("add", &Vector2::add, py::arg("v"), py::return_value_policy::reference_internal)
                .def("sub", &Vector2::sub, py::arg("v"), py::return_value_policy::reference_internal)
                .def("multiply_scalar", &Vector2::multiplyScalar, py::arg("s"), py::return_value_policy::reference_internal)
                .def(py::self + py::self)
                .def(py::self - py::self)
                .def(py::self * float())
                .def("__eq__", [](const Vector2& a, const Vector2& b) { return a == b; }, py::is_operator())
                .def("__repr__", [](const Vector2& v) {
                    std::ostringstream o;
                    o << "Vector2(" << v.x << ", " << v.y << ")";
                    return o.str();
                });

        // ---- Vector3 ---------------------------------------------------------
        py::class_<Vector3>(m, "Vector3")
                .def(py::init([](float x, float y, float z) { return Vector3(x, y, z); }),
                     py::arg("x") = 0.f, py::arg("y") = 0.f, py::arg("z") = 0.f)
                .def_readwrite("x", &Vector3::x)
                .def_readwrite("y", &Vector3::y)
                .def_readwrite("z", &Vector3::z)
                .def("set", &Vector3::set, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("copy", &Vector3::copy, py::arg("v"), py::return_value_policy::reference_internal)
                .def("clone", [](const Vector3& v) { return Vector3(v.x, v.y, v.z); })
                .def("length", &Vector3::length)
                .def("length_sq", &Vector3::lengthSq)
                .def("normalize", &Vector3::normalize, py::return_value_policy::reference_internal)
                .def("negate", &Vector3::negate, py::return_value_policy::reference_internal)
                .def("dot", &Vector3::dot, py::arg("v"))
                .def("cross", py::overload_cast<const Vector3&>(&Vector3::cross), py::arg("v"), py::return_value_policy::reference_internal)
                .def("distance_to", &Vector3::distanceTo, py::arg("v"))
                .def("add", &Vector3::add, py::arg("v"), py::return_value_policy::reference_internal)
                .def("add_scaled_vector", &Vector3::addScaledVector, py::arg("v"), py::arg("s"), py::return_value_policy::reference_internal)
                .def("sub", &Vector3::sub, py::arg("v"), py::return_value_policy::reference_internal)
                .def("multiply_scalar", &Vector3::multiplyScalar, py::arg("s"), py::return_value_policy::reference_internal)
                .def("apply_matrix4", &Vector3::applyMatrix4, py::arg("m"), py::return_value_policy::reference_internal)
                .def("apply_quaternion", &Vector3::applyQuaternion, py::arg("q"), py::return_value_policy::reference_internal)
                // World point -> NDC ([-1,1]^3) and back. In-place (mutates self,
                // returns self, like three.js). NDC.x/y map to the viewport with
                // x_px=(ndc.x*0.5+0.5)*W, y_px=(1-(ndc.y*0.5+0.5))*H. The camera's
                // matrices must be current — call render(), or update_matrix_world()
                // + update_projection_matrix(), before projecting.
                .def("project", [](Vector3& v, const Camera& cam) -> Vector3& { return v.project(cam); }, py::arg("camera"), py::return_value_policy::reference_internal)
                .def("unproject", [](Vector3& v, const Camera& cam) -> Vector3& { return v.unproject(cam); }, py::arg("camera"), py::return_value_policy::reference_internal)
                .def("lerp", &Vector3::lerp, py::arg("v"), py::arg("alpha"), py::return_value_policy::reference_internal)
                .def(py::self + py::self)
                .def(py::self - py::self)
                .def(py::self * float())
                .def("__eq__", [](const Vector3& a, const Vector3& b) { return a == b; }, py::is_operator())
                .def("__repr__", [](const Vector3& v) {
                    std::ostringstream o;
                    o << "Vector3(" << v.x << ", " << v.y << ", " << v.z << ")";
                    return o.str();
                });

        // ---- Vector4 ---------------------------------------------------------
        py::class_<Vector4>(m, "Vector4")
                .def(py::init([](float x, float y, float z, float w) { return Vector4(x, y, z, w); }),
                     py::arg("x") = 0.f, py::arg("y") = 0.f, py::arg("z") = 0.f, py::arg("w") = 1.f)
                .def_readwrite("x", &Vector4::x)
                .def_readwrite("y", &Vector4::y)
                .def_readwrite("z", &Vector4::z)
                .def_readwrite("w", &Vector4::w)
                .def("set", &Vector4::set, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"), py::return_value_policy::reference_internal)
                .def("__repr__", [](const Vector4& v) {
                    std::ostringstream o;
                    o << "Vector4(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
                    return o.str();
                });

        // ---- Color -----------------------------------------------------------
        // Accepts a hex int (0xff0000), three floats, or a CSS/name string.
        // The int and str overloads are registered as implicit conversions, so
        // `material.color = 0xff0000` and `AmbientLight("white")` just work.
        py::class_<Color>(m, "Color")
                .def(py::init<>())
                .def(py::init([](unsigned int hex) { return Color(hex); }), py::arg("hex"))
                .def(py::init([](float r, float g, float b) { return Color(r, g, b); }),
                     py::arg("r"), py::arg("g"), py::arg("b"))
                .def(py::init([](const std::string& style) {
                    Color c;
                    c.setStyle(style);
                    return c;
                }),
                     py::arg("style"))
                .def_readwrite("r", &Color::r)
                .def_readwrite("g", &Color::g)
                .def_readwrite("b", &Color::b)
                .def("set_hex", [](Color& c, unsigned int hex) -> Color& { return c.setHex(hex); }, py::arg("hex"), py::return_value_policy::reference_internal)
                .def("set_rgb", [](Color& c, float r, float g, float b) -> Color& { return c.setRGB(r, g, b); }, py::arg("r"), py::arg("g"), py::arg("b"), py::return_value_policy::reference_internal)
                .def("set_style", [](Color& c, const std::string& s) -> Color& { return c.setStyle(s); }, py::arg("style"), py::return_value_policy::reference_internal)
                .def("get_hex", [](const Color& c) { return c.getHex(); })
                .def("get_hex_string", [](const Color& c) { return c.getHexString(); })
                .def("copy", &Color::copy, py::arg("color"), py::return_value_policy::reference_internal)
                .def("clone", [](const Color& c) { return Color(c.r, c.g, c.b); })
                .def("lerp", &Color::lerp, py::arg("color"), py::arg("alpha"), py::return_value_policy::reference_internal)
                .def("__eq__", [](const Color& a, const Color& b) { return a == b; }, py::is_operator())
                .def("__repr__", [](const Color& c) {
                    std::ostringstream o;
                    o << "Color(" << c.r << ", " << c.g << ", " << c.b << ")";
                    return o.str();
                });
        py::implicitly_convertible<int, Color>();
        py::implicitly_convertible<std::string, Color>();

        // ---- Euler -----------------------------------------------------------
        // x/y/z are float_view (assignment fires the Object3D euler->quaternion
        // sync); expose them as properties so the callback still runs.
        py::enum_<Euler::RotationOrders>(m, "RotationOrder")
                .value("XYZ", Euler::RotationOrders::XYZ)
                .value("YZX", Euler::RotationOrders::YZX)
                .value("ZXY", Euler::RotationOrders::ZXY)
                .value("XZY", Euler::RotationOrders::XZY)
                .value("YXZ", Euler::RotationOrders::YXZ)
                .value("ZYX", Euler::RotationOrders::ZYX);

        py::class_<Euler>(m, "Euler")
                .def(py::init([](float x, float y, float z) { return Euler(x, y, z); }),
                     py::arg("x") = 0.f, py::arg("y") = 0.f, py::arg("z") = 0.f)
                .def_property("x", [](const Euler& e) { return float(e.x); }, [](Euler& e, float v) { e.x = v; })
                .def_property("y", [](const Euler& e) { return float(e.y); }, [](Euler& e, float v) { e.y = v; })
                .def_property("z", [](const Euler& e) { return float(e.z); }, [](Euler& e, float v) { e.z = v; })
                .def_property("order", &Euler::getOrder, &Euler::setOrder)
                .def("set", [](Euler& e, float x, float y, float z) -> Euler& { return e.set(x, y, z); }, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("__repr__", [](const Euler& e) {
                    std::ostringstream o;
                    o << "Euler(" << float(e.x) << ", " << float(e.y) << ", " << float(e.z) << ")";
                    return o.str();
                });

        // ---- Quaternion ------------------------------------------------------
        py::class_<Quaternion>(m, "Quaternion")
                .def(py::init([](float x, float y, float z, float w) { return Quaternion(x, y, z, w); }),
                     py::arg("x") = 0.f, py::arg("y") = 0.f, py::arg("z") = 0.f, py::arg("w") = 1.f)
                .def_property("x", [](const Quaternion& q) { return float(q.x); }, [](Quaternion& q, float v) { q.x = v; })
                .def_property("y", [](const Quaternion& q) { return float(q.y); }, [](Quaternion& q, float v) { q.y = v; })
                .def_property("z", [](const Quaternion& q) { return float(q.z); }, [](Quaternion& q, float v) { q.z = v; })
                .def_property("w", [](const Quaternion& q) { return float(q.w); }, [](Quaternion& q, float v) { q.w = v; })
                .def("set", &Quaternion::set, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"), py::return_value_policy::reference_internal)
                .def("set_from_euler", [](Quaternion& q, const Euler& e) -> Quaternion& { return q.setFromEuler(e); }, py::arg("euler"), py::return_value_policy::reference_internal)
                .def("set_from_axis_angle", &Quaternion::setFromAxisAngle, py::arg("axis"), py::arg("angle"), py::return_value_policy::reference_internal)
                .def("normalize", &Quaternion::normalize, py::return_value_policy::reference_internal)
                .def("invert", &Quaternion::invert, py::return_value_policy::reference_internal)
                .def("slerp", &Quaternion::slerp, py::arg("qb"), py::arg("t"), py::return_value_policy::reference_internal)
                .def("__repr__", [](const Quaternion& q) {
                    std::ostringstream o;
                    o << "Quaternion(" << float(q.x) << ", " << float(q.y) << ", " << float(q.z) << ", " << float(q.w) << ")";
                    return o.str();
                });

        // ---- Matrix3 / Matrix4 ----------------------------------------------
        py::class_<Matrix3>(m, "Matrix3")
                .def(py::init<>())
                .def("identity", &Matrix3::identity, py::return_value_policy::reference_internal)
                .def("invert", &Matrix3::invert, py::return_value_policy::reference_internal)
                .def("transpose", &Matrix3::transpose, py::return_value_policy::reference_internal)
                .def("determinant", &Matrix3::determinant)
                .def("elements", [](const Matrix3& m) { return std::vector<float>(m.elements.begin(), m.elements.end()); })
                // (3,3) row-major float32 — the standard math layout, so `M @ v`
                // works directly. (`elements()` is the raw column-major buffer.)
                .def("to_numpy", [](const Matrix3& m) {
                    py::array_t<float> a({3, 3});
                    auto* d = a.mutable_data();
                    for (int r = 0; r < 3; ++r)
                        for (int c = 0; c < 3; ++c) d[r * 3 + c] = m.elements[c * 3 + r];
                    return a;
                });

        py::class_<Matrix4>(m, "Matrix4")
                .def(py::init<>())
                .def("identity", &Matrix4::identity, py::return_value_policy::reference_internal)
                .def("invert", &Matrix4::invert, py::return_value_policy::reference_internal)
                .def("transpose", &Matrix4::transpose, py::return_value_policy::reference_internal)
                .def("determinant", &Matrix4::determinant)
                .def("copy", &Matrix4::copy, py::arg("m"), py::return_value_policy::reference_internal)
                .def("make_translation", py::overload_cast<float, float, float>(&Matrix4::makeTranslation), py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("make_rotation_x", &Matrix4::makeRotationX, py::arg("theta"), py::return_value_policy::reference_internal)
                .def("make_rotation_y", &Matrix4::makeRotationY, py::arg("theta"), py::return_value_policy::reference_internal)
                .def("make_rotation_z", &Matrix4::makeRotationZ, py::arg("theta"), py::return_value_policy::reference_internal)
                .def("make_scale", &Matrix4::makeScale, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("compose", &Matrix4::compose, py::arg("position"), py::arg("quaternion"), py::arg("scale"), py::return_value_policy::reference_internal)
                .def("set_position", py::overload_cast<float, float, float>(&Matrix4::setPosition), py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("elements", [](const Matrix4& m) { return std::vector<float>(m.elements.begin(), m.elements.end()); })
                // (4,4) row-major float32 — the standard math layout, so `M @ v`
                // works directly (e.g. K, or a world / view / projection matrix).
                // (`elements()` is the raw column-major buffer three.js stores.)
                .def("to_numpy", [](const Matrix4& m) {
                    py::array_t<float> a({4, 4});
                    auto* d = a.mutable_data();
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c) d[r * 4 + c] = m.elements[c * 4 + r];
                    return a;
                });

        // ---- Box3 ------------------------------------------------------------
        py::class_<Box3>(m, "Box3")
                .def(py::init<>())
                .def(py::init<const Vector3&, const Vector3&>(), py::arg("min"), py::arg("max"))
                .def("min", &Box3::min, py::return_value_policy::reference_internal)
                .def("max", &Box3::max, py::return_value_policy::reference_internal)
                .def("is_empty", &Box3::isEmpty)
                .def("get_center", py::overload_cast<>(&Box3::getCenter, py::const_))
                .def("get_size", py::overload_cast<>(&Box3::getSize, py::const_))
                .def("contains_point", &Box3::containsPoint, py::arg("point"))
                // Route the object through as_object3d: a direct &Box3::setFromObject
                // pointer corrupts the argument for virtual-Object3D types
                // (Mesh/Points/Line) → "no RTTI data" crash. Group/Scene worked by
                // luck (non-virtual base); this makes any Object3D safe.
                .def("set_from_object", [](Box3& b, const py::object& obj, bool precise) -> Box3& {
                    return b.setFromObject(*as_object3d(obj), precise);
                }, py::arg("object"), py::arg("precise") = false, py::return_value_policy::reference_internal);

        // ---- MathUtils free functions (module-level) -------------------------
        // Plain scalar helpers from threepp::math, reimplemented constantly in
        // Python today (angle conversions, RL action/state math). randFloat is
        // overloaded, so split into rand_float() [0,1] and rand_float_range(min,max).
        m.def("deg_to_rad", &math::degToRad, py::arg("degrees"));
        m.def("rad_to_deg", &math::radToDeg, py::arg("radians"));
        m.def("map_linear", &math::mapLinear, py::arg("x"), py::arg("a1"), py::arg("a2"), py::arg("b1"), py::arg("b2"));
        m.def("inverse_lerp", &math::inverseLerp, py::arg("x"), py::arg("y"), py::arg("value"));
        m.def("lerp", &math::lerp, py::arg("x"), py::arg("y"), py::arg("t"));
        m.def("damp", &math::damp, py::arg("x"), py::arg("y"), py::arg("lambda"), py::arg("dt"));
        m.def("euclidean_modulo", &math::euclideanModulo, py::arg("n"), py::arg("m"));
        // clamp has no MathUtils equivalent; provide it via std::clamp.
        m.def("clamp", [](float value, float low, float high) { return std::clamp(value, low, high); },
              py::arg("value"), py::arg("low"), py::arg("high"));
        m.def("rand_int", &math::randInt, py::arg("low"), py::arg("high"));
        m.def("rand_float", py::overload_cast<>(&math::randFloat));
        m.def("rand_float_range", py::overload_cast<float, float>(&math::randFloat), py::arg("min"), py::arg("max"));
        m.def("rand_float_spread", &math::randFloatSpread, py::arg("range"));
        m.def("generate_uuid", &math::generateUUID);
        m.def("is_power_of_two", &math::isPowerOfTwo, py::arg("value"));
    }

}// namespace threepp_py
