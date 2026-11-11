// CPU cloth: threepp::Cloth and threepp::ClothMesh from extras/cloth, exposed
// to Python.
//
// Cloth is the XPBD solver (particles, triangles, pins, colliders, wind);
// ClothMesh welds a BufferGeometry's vertices into a Cloth and writes positions
// and normals back after each step, so to both renderers the geometry is a
// plain animated mesh. From Python it is the C++ usage line for line:
//
//     geo = tp.PlaneGeometry(1.1, 0.8, 22, 16)
//     flag = tp.ClothMesh(geo)
//     flag.cloth.pin_many(flag.particles_where(x_max=-0.549))
//     flag.cloth.set_wind(tp.Vector3(6, 0, 0))
//     ...each frame: flag.update(dt)
//
// Everything per particle crosses the boundary as a numpy array in one call,
// so a host never loops over particles in Python: positions and normals come
// out as (N, 3) float32 copies, and a moving hoist is re-pinned with one
// pin_many(indices, targets) per frame. The wind field callback is the one
// exception, and it is opt-in: it is called once per particle per step, which
// is fine for a few hundred particles and a reason to prefer set_wind (a
// uniform wind with the solver's own gusts) for anything per-frame.
#include "bindings.hpp"

#include "threepp/extras/cloth/Cloth.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <limits>
#include <optional>
#include <stdexcept>

namespace threepp_py {

    namespace {

        using namespace threepp;

        using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
        using IndexArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

        // (N, 3) float32 copy of a vector of Vector3. A copy, not a view: the
        // solver owns the storage and a view would outlive nothing it can see.
        py::array_t<float> toArray(const std::vector<Vector3>& v) {
            py::array_t<float> out({static_cast<py::ssize_t>(v.size()), py::ssize_t{3}});
            float* o = out.mutable_data();
            for (size_t i = 0; i < v.size(); ++i) {
                o[i * 3] = v[i].x;
                o[i * 3 + 1] = v[i].y;
                o[i * 3 + 2] = v[i].z;
            }
            return out;
        }

        // Any (N, 3) float array, or a flat array of 3N, as Vector3s.
        std::vector<Vector3> toVectors(const FloatArray& a, const char* what) {
            if (a.size() % 3 != 0 || (a.ndim() == 2 && a.shape(1) != 3) || a.ndim() > 2) {
                throw std::invalid_argument(std::string(what) + ": expected an (N, 3) array");
            }
            const float* p = a.data();
            std::vector<Vector3> out(static_cast<size_t>(a.size() / 3));
            for (size_t i = 0; i < out.size(); ++i) out[i].set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
            return out;
        }

        // A flat (or (M, 3)) integer array as particle indices, checked
        // against the particle count so a bad index raises instead of
        // writing out of bounds.
        std::vector<size_t> toIndices(const IndexArray& a, size_t count, const char* what) {
            const std::int64_t* p = a.data();
            std::vector<size_t> out(static_cast<size_t>(a.size()));
            for (size_t i = 0; i < out.size(); ++i) {
                if (p[i] < 0 || static_cast<size_t>(p[i]) >= count) {
                    throw std::out_of_range(std::string(what) + ": particle index " + std::to_string(p[i]) + " out of range");
                }
                out[i] = static_cast<size_t>(p[i]);
            }
            return out;
        }

        py::array_t<std::int64_t> indexArray(const std::vector<size_t>& v) {
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(v.size()));
            std::int64_t* o = out.mutable_data();
            for (size_t i = 0; i < v.size(); ++i) o[i] = static_cast<std::int64_t>(v[i]);
            return out;
        }

    }// namespace

    void init_cloth(py::module_& m) {

        auto cloth = py::class_<Cloth>(m, "Cloth",
                                       "XPBD cloth on the CPU: particles, triangles, pins, sphere/capsule/plane "
                                       "colliders and wind as pressure on each triangle. Works in whatever space "
                                       "its positions are in: gravity, wind, pins and colliders are in that space. "
                                       "Single-threaded and order-fixed, so the same inputs and dt sequence give "
                                       "bit-identical positions. Usually reached through ClothMesh.cloth.");

        py::class_<Cloth::Params>(cloth, "Params")
                .def(py::init<>())
                .def_readwrite("gravity", &Cloth::Params::gravity, "m/s2, in the solver's space")
                .def_readwrite("substeps", &Cloth::Params::substeps,
                               "Substeps per step(); each runs one pass over the constraints.")
                .def_readwrite("max_step", &Cloth::Params::maxStep,
                               "A step() longer than this (s) is clamped, so a hitch does not launch the cloth.")
                .def_readwrite("areal_density", &Cloth::Params::arealDensity,
                               "Mass per area, kg/m2; flag polyester is about 0.1-0.15. Read at construction.")
                .def_readwrite("stretch_compliance", &Cloth::Params::stretchCompliance,
                               "m/N; 0 = inextensible as far as the substeps allow.")
                .def_readwrite("bend_compliance", &Cloth::Params::bendCompliance, "m/N; larger bends more easily.")
                .def_readwrite("damping", &Cloth::Params::damping, "Fraction of velocity lost per second.")
                .def_readwrite("air_density", &Cloth::Params::airDensity, "kg/m3")
                .def_readwrite("normal_drag", &Cloth::Params::normalDrag, "Pressure coefficient across the sheet.")
                .def_readwrite("tangent_drag", &Cloth::Params::tangentDrag, "Friction coefficient along the sheet.")
                .def_readwrite("flapping", &Cloth::Params::flapping,
                               "Amplitude of the flapping pressure wave, as a fraction of the dynamic pressure "
                               "of the flow along the sheet.")
                .def_readwrite("gustiness", &Cloth::Params::gustiness,
                               "0 = steady wind; 1 = speed +-50 % and direction +-25 degrees, in waves that "
                               "travel downwind. Applies to set_wind, not to a wind field.")
                .def_readwrite("collider_slip", &Cloth::Params::colliderSlip,
                               "Tangential velocity kept on contact (1 = no friction).")
                .def_readwrite("thickness", &Cloth::Params::thickness, "m of padding on every collider.");

        cloth.def(py::init([](const FloatArray& positions, const IndexArray& indices, const Cloth::Params& params) {
                      std::vector<unsigned int> idx(static_cast<size_t>(indices.size()));
                      const std::int64_t* p = indices.data();
                      for (size_t i = 0; i < idx.size(); ++i) {
                          if (p[i] < 0 || p[i] > std::numeric_limits<unsigned int>::max()) {
                              throw std::invalid_argument("Cloth: negative or oversized index");
                          }
                          idx[i] = static_cast<unsigned int>(p[i]);
                      }
                      return std::make_unique<Cloth>(toVectors(positions, "Cloth"), std::move(idx), params);
                  }),
                  py::arg("positions"), py::arg("indices"), py::arg("params") = Cloth::Params{},
                  "positions: (N, 3) float; indices: three per triangle, flat or (M, 3).")
                .def_readwrite("params", &Cloth::params,
                               "Live parameters; areal_density only matters at construction.")
                .def("particle_count", &Cloth::particleCount)
                .def("triangle_count", &Cloth::triangleCount)
                .def("stretch_constraint_count", &Cloth::stretchConstraintCount)
                .def("bend_constraint_count", &Cloth::bendConstraintCount)
                .def_property_readonly("positions", [](const Cloth& c) { return toArray(c.positions()); },
                                       "(N, 3) float32 copy of the particle positions.")
                .def_property_readonly("velocities", [](const Cloth& c) { return toArray(c.velocities()); },
                                       "(N, 3) float32 copy of the particle velocities, m/s.")
                .def_property_readonly("normals", [](const Cloth& c) { return toArray(c.normals()); },
                                       "(N, 3) float32 copy of the area-weighted vertex normals, refreshed by step().")
                .def_property_readonly("indices", [](const Cloth& c) {
                    const auto& t = c.indices();
                    py::array_t<std::uint32_t> out({static_cast<py::ssize_t>(t.size() / 3), py::ssize_t{3}});
                    std::copy(t.begin(), t.end(), out.mutable_data());
                    return out; }, "(M, 3) uint32 copy of the triangles' particle indices.")
                .def_property_readonly("pinned", [](const Cloth& c) {
                    py::array_t<bool> out(static_cast<py::ssize_t>(c.particleCount()));
                    bool* o = out.mutable_data();
                    for (size_t i = 0; i < c.particleCount(); ++i) o[i] = c.isPinned(i);
                    return out; }, "(N,) bool: which particles are pinned.")
                // ── Pins ──────────────────────────────────────────────────
                .def("pin", py::overload_cast<size_t>(&Cloth::pin), py::arg("i"),
                     "Hold particle i where it is now.")
                .def("pin", py::overload_cast<size_t, const Vector3&>(&Cloth::pin), py::arg("i"), py::arg("target"),
                     "Hold particle i at target; called each frame with a moving target, the particle is "
                     "carried there over the next step's substeps.")
                .def("unpin", &Cloth::unpin, py::arg("i"))
                .def("is_pinned", &Cloth::isPinned, py::arg("i"))
                .def("pin_many", [](Cloth& c, const IndexArray& indices, std::optional<FloatArray> targets) {
                         const auto idx = toIndices(indices, c.particleCount(), "pin_many");
                         if (!targets) {
                             for (const size_t i : idx) c.pin(i);
                             return;
                         }
                         const auto t = toVectors(*targets, "pin_many");
                         if (t.size() != idx.size()) throw std::invalid_argument("pin_many: one target per index");
                         for (size_t k = 0; k < idx.size(); ++k) c.pin(idx[k], t[k]);
                     },
                     py::arg("indices"), py::arg("targets") = py::none(),
                     "pin(i) or pin(i, target) for many particles in one call: indices a flat integer array, "
                     "targets None (hold where they are) or an (N, 3) array, one per index. Call it each frame "
                     "with the targets of a moving hoist.")
                .def("unpin_many", [](Cloth& c, const IndexArray& indices) {
                         for (const size_t i : toIndices(indices, c.particleCount(), "unpin_many")) c.unpin(i);
                     },
                     py::arg("indices"))
                // ── Wind ──────────────────────────────────────────────────
                .def("set_wind", &Cloth::setWind, py::arg("wind"),
                     "Air velocity, m/s, in the solver's space. The air each triangle feels is this (with "
                     "params.gustiness) minus the triangle's own velocity.")
                .def("wind", &Cloth::wind)
                .def("set_wind_field", &Cloth::setWindField, py::arg("field"),
                     "Replace set_wind (and its gusts) by field(position: Vector3, time: float) -> Vector3, "
                     "called once per particle per step(), from Python: fine for a few hundred particles, "
                     "costly beyond. None goes back to set_wind.")
                // ── Colliders ─────────────────────────────────────────────
                .def("add_sphere", &Cloth::addSphere, py::arg("center"), py::arg("radius"),
                     "Returns a handle for set_sphere.")
                .def("add_capsule", &Cloth::addCapsule, py::arg("a"), py::arg("b"), py::arg("radius"),
                     "Returns a handle for set_capsule (spheres and capsules share one list).")
                .def("set_sphere", &Cloth::setSphere, py::arg("handle"), py::arg("center"), py::arg("radius"))
                .def("set_capsule", &Cloth::setCapsule, py::arg("handle"), py::arg("a"), py::arg("b"), py::arg("radius"))
                .def("add_plane", &Cloth::addPlane, py::arg("point"), py::arg("normal"),
                     "The cloth stays on the side normal points to. Returns a handle for set_plane.")
                .def("set_plane", &Cloth::setPlane, py::arg("handle"), py::arg("point"), py::arg("normal"))
                .def("clear_colliders", &Cloth::clearColliders)
                // ── Stepping ──────────────────────────────────────────────
                .def_property_readonly("time", &Cloth::time, "Simulated seconds so far.")
                // No call_guard release here: a wind field calls back into
                // Python on this thread, and the step is a fraction of a
                // millisecond for a flag, shorter than a release is worth.
                .def("step", &Cloth::step, py::arg("dt"), "Advance by dt seconds (clamped to params.max_step).");

        py::class_<ClothMesh>(m, "ClothMesh",
                              "A Cloth bound to a BufferGeometry: vertices that share a position become one "
                              "particle, and update(dt) steps the cloth and writes positions and normals back. "
                              "The cloth works in the geometry's own space, so keep the mesh unrotated and "
                              "unscaled (or give gravity, wind, pins and colliders in its frame).")
                .def(py::init([](std::shared_ptr<BufferGeometry> geometry, const Cloth::Params& params, float weld) {
                         if (!geometry) throw std::invalid_argument("ClothMesh: geometry is None");
                         return std::make_unique<ClothMesh>(*geometry, params, weld);
                     }),
                     py::arg("geometry"), py::arg("params") = Cloth::Params{}, py::arg("weld_distance") = 1e-5f,
                     // The ClothMesh writes into the geometry every update:
                     // the geometry lives at least as long as it does.
                     py::keep_alive<1, 2>())
                .def_property_readonly("cloth", py::overload_cast<>(&ClothMesh::cloth),
                                       py::return_value_policy::reference_internal)
                .def("particle_of", &ClothMesh::particleOf, py::arg("vertex"),
                     "The particle a geometry vertex was welded into.")
                .def("particles_where", [](const ClothMesh& c, const std::function<bool(const Vector3&)>& predicate) {
                         return indexArray(c.particlesWhere(predicate));
                     },
                     py::arg("predicate"),
                     "Indices (int64 array) of the particles whose present position satisfies "
                     "predicate(Vector3) -> bool; called once per particle, so a set-up call, not a per-frame one.")
                .def("particles_where", [](const ClothMesh& c, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax) {
                         return indexArray(c.particlesWhere([&](const Vector3& p) {
                             return p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax && p.z >= zMin && p.z <= zMax;
                         }));
                     },
                     py::arg("x_min") = -std::numeric_limits<float>::infinity(),
                     py::arg("x_max") = std::numeric_limits<float>::infinity(),
                     py::arg("y_min") = -std::numeric_limits<float>::infinity(),
                     py::arg("y_max") = std::numeric_limits<float>::infinity(),
                     py::arg("z_min") = -std::numeric_limits<float>::infinity(),
                     py::arg("z_max") = std::numeric_limits<float>::infinity(),
                     "Indices (int64 array) of the particles inside a box, bounds inclusive; for pinning an "
                     "edge or a corner without knowing the vertex layout. For any other predicate, test "
                     "cloth.positions with numpy.")
                .def("update", &ClothMesh::update, py::arg("dt"), "Step the cloth by dt and write the geometry.")
                .def("write", &ClothMesh::write,
                     "Write the cloth's positions and normals into the geometry without stepping.");
    }

}// namespace threepp_py
