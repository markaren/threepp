// BVH — mesh-vs-mesh queries: intersection, intersection points, distance.
// Every query releases the GIL; the *_many forms run the candidate x target
// grid in parallel inside C++.
#include "bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Ray.hpp"
#include "threepp/utils/BVH.hpp"
#include "threepp/utils/Parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace threepp;

namespace threepp_py {

    namespace {

        using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
        using IndexArray = py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>;
        using BVHList = std::vector<std::shared_ptr<BVH>>;

        std::vector<float> toPositions(const FloatArray& a) {
            if (a.ndim() == 2 && a.shape(1) != 3) {
                throw std::invalid_argument("positions: a 2-D array must be (N, 3)");
            }
            if (a.ndim() != 1 && a.ndim() != 2) {
                throw std::invalid_argument("positions: expected a flat or (N, 3) float array");
            }
            if (a.size() % 3 != 0) {
                throw std::invalid_argument("positions: element count must be a multiple of 3");
            }
            return {a.data(), a.data() + a.size()};
        }

        std::vector<unsigned int> toIndices(const std::optional<IndexArray>& a) {
            if (!a) return {};
            if (a->size() % 3 != 0) {
                throw std::invalid_argument("indices: element count must be a multiple of 3");
            }
            return {a->data(), a->data() + a->size()};
        }

        // Copies positions + index out of the geometry so the BVH does not keep a
        // pointer into a Python-owned object (BVH::build(const BufferGeometry&) does).
        void buildFromGeometry(BVH& bvh, const BufferGeometry& geometry) {
            const auto* pos = geometry.getAttribute<float>("position");
            if (!pos) throw std::invalid_argument("BVH.build: geometry has no 'position' attribute");

            const auto& array = pos->array();
            std::vector<float> positions(array.begin(), array.end());

            std::vector<unsigned int> indices;
            if (const auto* index = geometry.getIndex()) {
                const auto& ia = index->array();
                indices.assign(ia.begin(), ia.end());
            }

            py::gil_scoped_release release;
            bvh.build(positions, indices);
        }

        py::array_t<float> pointsToArray(const std::vector<Vector3>& points) {
            py::array_t<float> out({static_cast<py::ssize_t>(points.size()), py::ssize_t{3}});
            auto* d = out.mutable_data();
            for (std::size_t i = 0; i < points.size(); ++i) {
                d[i * 3] = points[i].x;
                d[i * 3 + 1] = points[i].y;
                d[i * 3 + 2] = points[i].z;
            }
            return out;
        }

        // Parallel-for over the C x T grid; `fn(i, j, cell)` writes only its own cell.
        template<class Fn>
        void forEachPair(std::size_t candidates, std::size_t targets, Fn fn) {
            if (candidates == 0 || targets == 0) return;
            std::vector<std::size_t> cells(candidates * targets);
            std::iota(cells.begin(), cells.end(), std::size_t{0});
            parallelForEach(cells.begin(), cells.end(), [&](std::size_t k) {
                fn(k / targets, k % targets, k);
            });
        }

        void requireBuilt(const BVHList& list, const char* what) {
            for (const auto& b : list) {
                if (!b) throw std::invalid_argument(std::string(what) + ": list contains None");
            }
        }

    }// namespace

    void init_bvh(py::module_& m) {

        py::class_<BVH::RayHit>(m, "RayHit")
                .def_readonly("distance", &BVH::RayHit::distance)
                .def_readonly("triangle_index", &BVH::RayHit::triangleIndex)
                .def_readonly("point", &BVH::RayHit::point)
                .def_readonly("normal", &BVH::RayHit::normal,
                              "Geometric face normal, flipped to face the ray origin.")
                .def("__repr__", [](const BVH::RayHit& h) {
                    return "RayHit(distance=" + std::to_string(h.distance) +
                           ", triangle_index=" + std::to_string(h.triangleIndex) + ")";
                });

        py::class_<BVH, std::shared_ptr<BVH>> bvh(m, "BVH");

        bvh.def(py::init<int, int>(),
                py::arg("max_triangles_per_node") = 8, py::arg("max_subdivisions") = 10,
                "max_triangles_per_node is the leaf size; smaller gives a deeper tree that prunes harder.")
                .def(
                        "build", [](BVH& self, const BufferGeometry& geometry) { buildFromGeometry(self, geometry); },
                        py::arg("geometry"),
                        "Build from a BufferGeometry (indexed or a raw soup). The triangles are "
                        "COPIED, so the geometry may be dropped afterwards.")
                .def(
                        "build_arrays", [](BVH& self, const FloatArray& positions, const std::optional<IndexArray>& indices) {
                            auto p = toPositions(positions);
                            auto i = toIndices(indices);
                            py::gil_scoped_release release;
                            self.build(p, i);
                        },
                        py::arg("positions"), py::arg("indices") = py::none(),
                        "Build from raw arrays: `positions` flat or (N, 3) float, `indices` a flat "
                        "or (M, 3) uint32 triangle list, or None for a soup (three consecutive vertices "
                        "per triangle) — the layout of BufferGeometry.get_attribute('position') / get_index().")
                .def_property_readonly("triangle_count", &BVH::triangleCount)
                .def("bounding_box", &BVH::boundingBox, "Root bounds, in the BVH's own space.")
                .def(
                        "collect_boxes", [](const BVH& self, bool leaves_only) {
                            std::vector<BVHBox3> boxes;
                            self.collectBoxes(boxes);
                            std::vector<Box3> out;
                            out.reserve(boxes.size());
                            for (const auto& b : boxes) {
                                if (!leaves_only || b.isLeaf()) out.emplace_back(b.min(), b.max());
                            }
                            return out;
                        },
                        py::arg("leaves_only") = false,
                        "Every node's box, for debug visualisation.")
                .def(
                        "raycast", [](const BVH& self, const Ray& ray, float max_distance) {
                            py::gil_scoped_release release;
                            return self.raycast(ray, max_distance);
                        },
                        py::arg("ray"), py::arg("max_distance") = std::numeric_limits<float>::infinity(),
                        "Closest hit, or None. The ray is in the BVH's own space.")
                .def(
                        "raycast_any", [](const BVH& self, const Ray& ray, float max_distance) {
                            py::gil_scoped_release release;
                            return self.raycastAny(ray, max_distance);
                        },
                        py::arg("ray"), py::arg("max_distance") = std::numeric_limits<float>::infinity(),
                        "Early-out occlusion query: is anything hit within max_distance?")
                .def(
                        "intersect_box", [](const BVH& self, const Box3& box, const Matrix4& matrix) {
                            py::gil_scoped_release release;
                            return self.intersect(box, matrix);
                        },
                        py::arg("box"), py::arg("matrix") = Matrix4(),
                        "Indices of the triangles whose boxes overlap `box`.")
                .def("__repr__", [](const BVH& self) {
                    return "<threepp.BVH triangles=" + std::to_string(self.triangleCount()) + ">";
                });

        // ---- pair queries ----------------------------------------------------
        // Static: a and b are symmetric. m1 / m2 place each BVH in a common frame;
        // identity is detected and skipped.
        bvh.def_static(
                   "intersects", [](const BVH& a, const BVH& b, const Matrix4& m1, const Matrix4& m2) {
                       py::gil_scoped_release release;
                       return BVH::intersects(a, b, m1, m2);
                   },
                   py::arg("a"), py::arg("b"), py::arg("m1") = Matrix4(), py::arg("m2") = Matrix4(),
                   "Do the two surfaces touch? Exact at the triangle level, early-exiting on the first hit.")
                .def_static(
                        "intersect", [](const BVH& a, const BVH& b, const Matrix4& m1, const Matrix4& m2, bool accurate) {
                            std::vector<BVH::IntersectionResult> results;
                            {
                                py::gil_scoped_release release;
                                results = BVH::intersect(a, m1, b, m2, accurate);
                            }
                            std::vector<Vector3> points;
                            points.reserve(results.size());
                            for (const auto& r : results) points.push_back(r.position);
                            return pointsToArray(points);
                        },
                        py::arg("a"), py::arg("b"), py::arg("m1") = Matrix4(), py::arg("m2") = Matrix4(),
                        py::arg("accurate") = true,
                        "(N, 3) float32 points of intersection, one per intersecting triangle pair. "
                        "With accurate=True each point lies on both surfaces; accurate=False is the "
                        "cheap conservative form that reports the centres of overlapping leaf boxes.")
                .def_static(
                        "intersect_pairs", [](const BVH& a, const BVH& b, const Matrix4& m1, const Matrix4& m2, bool accurate) {
                            std::vector<BVH::IntersectionResult> results;
                            {
                                py::gil_scoped_release release;
                                results = BVH::intersect(a, m1, b, m2, accurate);
                            }
                            const auto n = static_cast<py::ssize_t>(results.size());
                            py::array_t<std::int32_t> idxA(n), idxB(n);
                            std::vector<Vector3> points;
                            points.reserve(results.size());
                            for (std::size_t i = 0; i < results.size(); ++i) {
                                idxA.mutable_data()[i] = results[i].idxA;
                                idxB.mutable_data()[i] = results[i].idxB;
                                points.push_back(results[i].position);
                            }
                            return py::make_tuple(idxA, idxB, pointsToArray(points));
                        },
                        py::arg("a"), py::arg("b"), py::arg("m1") = Matrix4(), py::arg("m2") = Matrix4(),
                        py::arg("accurate") = true,
                        "As intersect(), plus which triangles met: (idx_a, idx_b, points). The "
                        "indices are -1 when accurate=False, which reports nodes rather than triangles.")
                .def_static(
                        "distance", [](const BVH& a, const BVH& b, const Matrix4& m1, const Matrix4& m2, float max_distance) {
                            py::gil_scoped_release release;
                            return BVH::distance(a, b, m1, m2, max_distance);
                        },
                        py::arg("a"), py::arg("b"), py::arg("m1") = Matrix4(), py::arg("m2") = Matrix4(),
                        py::arg("max_distance") = std::numeric_limits<float>::infinity(),
                        "Smallest surface-to-surface distance: 0 when the meshes intersect, inf when "
                        "either is empty or nothing is closer than max_distance. A finite max_distance "
                        "seeds the pruning bound and is much cheaper than an exact search.");

        // ---- batch queries ---------------------------------------------------
        // Every candidate against every target, in parallel with the GIL released.
        bvh.def_static(
                   "intersects_many", [](const BVHList& candidates, const BVHList& targets) {
                       requireBuilt(candidates, "intersects_many");
                       requireBuilt(targets, "intersects_many");
                       const auto c = candidates.size(), t = targets.size();
                       // std::vector<bool> is a bitfield, and concurrent writes to
                       // neighbouring bits are a data race; char is not.
                       std::vector<char> hits(c * t, 0);
                       {
                           py::gil_scoped_release release;
                           forEachPair(c, t, [&](std::size_t i, std::size_t j, std::size_t k) {
                               hits[k] = BVH::intersects(*candidates[i], *targets[j]) ? 1 : 0;
                           });
                       }
                       py::array_t<bool> out({static_cast<py::ssize_t>(c), static_cast<py::ssize_t>(t)});
                       std::copy(hits.begin(), hits.end(), out.mutable_data());
                       return out;
                   },
                   py::arg("candidates"), py::arg("targets"),
                   "(C, T) bool array: does candidate i touch target j?")
                .def_static(
                        "distance_many", [](const BVHList& candidates, const BVHList& targets, float max_distance) {
                            requireBuilt(candidates, "distance_many");
                            requireBuilt(targets, "distance_many");
                            const auto c = candidates.size(), t = targets.size();
                            std::vector<float> distances(c * t, std::numeric_limits<float>::infinity());
                            {
                                py::gil_scoped_release release;
                                forEachPair(c, t, [&](std::size_t i, std::size_t j, std::size_t k) {
                                    distances[k] = BVH::distance(*candidates[i], *targets[j], Matrix4(), Matrix4(), max_distance);
                                });
                            }
                            py::array_t<float> out({static_cast<py::ssize_t>(c), static_cast<py::ssize_t>(t)});
                            std::copy(distances.begin(), distances.end(), out.mutable_data());
                            return out;
                        },
                        py::arg("candidates"), py::arg("targets"),
                        py::arg("max_distance") = std::numeric_limits<float>::infinity(),
                        "(C, T) float32 array of surface-to-surface distances; 0 where they intersect, "
                        "inf beyond max_distance.")
                .def_static(
                        "intersect_many", [](const BVHList& candidates, const BVHList& targets, bool accurate) {
                            requireBuilt(candidates, "intersect_many");
                            requireBuilt(targets, "intersect_many");
                            const auto c = candidates.size(), t = targets.size();
                            std::vector<std::vector<Vector3>> points(c * t);
                            {
                                py::gil_scoped_release release;
                                forEachPair(c, t, [&](std::size_t i, std::size_t j, std::size_t k) {
                                    const auto results = BVH::intersect(*candidates[i], Matrix4(), *targets[j], Matrix4(), accurate);
                                    points[k].reserve(results.size());
                                    for (const auto& r : results) points[k].push_back(r.position);
                                });
                            }
                            py::list out;
                            for (std::size_t i = 0; i < c; ++i) {
                                py::list row;
                                for (std::size_t j = 0; j < t; ++j) row.append(pointsToArray(points[i * t + j]));
                                out.append(row);
                            }
                            return out;
                        },
                        py::arg("candidates"), py::arg("targets"), py::arg("accurate") = true,
                        "C x T nested list of (N, 3) float32 point arrays — the intersection points "
                        "of every candidate/target pair. An empty (0, 3) array means no contact.");
    }

}// namespace threepp_py
