// Point-cloud algorithms: voxel spatial index, ICP registration, marching cubes.
// Primary interface uses numpy (N,3) float32 arrays for speed; Vector3 lists
// also accepted via pybind11/stl.h.
#include "bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/pointcloud/Icp.hpp"
#include "threepp/extras/pointcloud/MarchingCubes.hpp"
#include "threepp/extras/pointcloud/VoxelGrid.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

using namespace threepp;
namespace py = pybind11;

namespace {

    // (N,3) float32 numpy array → std::vector<Vector3>
    std::vector<Vector3> numpyToPoints(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 2 || arr.shape(1) != 3)
            throw std::invalid_argument("point array must be shape (N, 3)");
        auto r = arr.unchecked<2>();
        std::vector<Vector3> pts(static_cast<std::size_t>(r.shape(0)));
        for (py::ssize_t i = 0; i < r.shape(0); ++i)
            pts[i] = Vector3(r(i, 0), r(i, 1), r(i, 2));
        return pts;
    }

    // std::vector<Vector3> → (N,3) float32 numpy array
    py::array_t<float> pointsToNumpy(const std::vector<Vector3>& pts) {
        py::array_t<float> arr({static_cast<py::ssize_t>(pts.size()), static_cast<py::ssize_t>(3)});
        auto r = arr.mutable_unchecked<2>();
        for (std::size_t i = 0; i < pts.size(); ++i) {
            r(static_cast<py::ssize_t>(i), 0) = pts[i].x;
            r(static_cast<py::ssize_t>(i), 1) = pts[i].y;
            r(static_cast<py::ssize_t>(i), 2) = pts[i].z;
        }
        return arr;
    }

}// namespace

namespace threepp_py {

    void init_pointcloud(py::module_& m) {

        // ---- VoxelGrid -------------------------------------------------------
        py::class_<VoxelGrid>(m, "VoxelGrid",
                "Voxel-hash spatial index. O(1) insert, nearest-neighbour queries over the "
                "27 surrounding voxels. Use voxelSize >= search radius for exact results.")
                .def(py::init<float, std::size_t, float>(),
                     py::arg("voxel_size"),
                     py::arg("max_points_per_voxel") = 20,
                     py::arg("min_spacing") = 0.f)
                .def("clear", &VoxelGrid::clear)
                .def_property_readonly("empty", &VoxelGrid::empty)
                .def_property_readonly("size", &VoxelGrid::size)
                .def_property_readonly("voxel_size", &VoxelGrid::voxelSize)
                .def_property_readonly("voxel_count", &VoxelGrid::voxelCount)
                .def("insert", &VoxelGrid::insert, py::arg("point"),
                     "Insert a single Vector3. Returns True if stored (passed cap + spacing).")
                .def("insert_array", [](VoxelGrid& self, const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
                         auto pts = numpyToPoints(arr);
                         std::size_t n = 0;
                         for (const auto& p : pts) n += self.insert(p) ? 1 : 0;
                         return n;
                     }, py::arg("points"),
                     "Insert an (N,3) float32 array. Returns number of points actually stored.")
                .def("nearest", [](const VoxelGrid& self, const Vector3& query, float max_dist) -> py::object {
                         Vector3 out;
                         if (self.nearest(query, max_dist, out)) return py::cast(out);
                         return py::none();
                     }, py::arg("query"), py::arg("max_dist"),
                     "Nearest stored point within max_dist. Returns Vector3 or None.")
                .def("collect", [](const VoxelGrid& self) {
                         std::vector<Vector3> pts;
                         self.collect(pts);
                         return pointsToNumpy(pts);
                     }, "Return all stored points as (N,3) float32 numpy array.")
                .def("collect_voxel_centers", [](const VoxelGrid& self) {
                         std::vector<Vector3> pts;
                         self.collectVoxelCenters(pts);
                         return pointsToNumpy(pts);
                     }, "Return one centre point per occupied voxel as (N,3) float32 numpy array.");

        // ---- voxel_downsample -----------------------------------------------
        m.def("voxel_downsample",
              [](const py::array_t<float, py::array::c_style | py::array::forcecast>& arr, float voxel_size) {
                  auto pts = numpyToPoints(arr);
                  return pointsToNumpy(voxelDownsample(pts, voxel_size));
              },
              py::arg("points"), py::arg("voxel_size"),
              "Voxel-downsample an (N,3) float32 point array. Returns (M,3) float32.");

        // ---- IcpOptions / IcpResult ------------------------------------------
        py::class_<IcpOptions>(m, "IcpOptions")
                .def(py::init<>())
                .def_readwrite("max_iterations", &IcpOptions::maxIterations)
                .def_readwrite("max_correspondence_distance", &IcpOptions::maxCorrespondenceDistance)
                .def_readwrite("min_correspondence_distance", &IcpOptions::minCorrespondenceDistance)
                .def_readwrite("robust_sigma", &IcpOptions::robustSigma);

        py::class_<IcpResult>(m, "IcpResult")
                .def_readonly("iterations", &IcpResult::iterations)
                .def_readonly("correspondences", &IcpResult::correspondences)
                .def_readonly("converged", &IcpResult::converged)
                .def("__repr__", [](const IcpResult& r) {
                    return "<IcpResult iters=" + std::to_string(r.iterations) +
                           " corr=" + std::to_string(r.correspondences) +
                           " converged=" + (r.converged ? "True" : "False") + ">";
                });

        // ---- icp_point_to_point ----------------------------------------------
        m.def("icp_point_to_point",
              [](const py::array_t<float, py::array::c_style | py::array::forcecast>& source,
                 const VoxelGrid& target, Matrix4& pose, const IcpOptions& opts) {
                  auto pts = numpyToPoints(source);
                  return icpPointToPoint(pts, target, pose, opts);
              },
              py::arg("source"), py::arg("target"), py::arg("pose"), py::arg("opts") = IcpOptions{},
              "Register source (N,3) float32 array against a VoxelGrid target. "
              "pose (Matrix4) is updated in place; seed it with an initial guess first.");

        // ---- ScalarField -----------------------------------------------------
        py::class_<ScalarField>(m, "ScalarField")
                .def(py::init<>())
                .def_readonly("nx", &ScalarField::nx)
                .def_readonly("ny", &ScalarField::ny)
                .def_readonly("nz", &ScalarField::nz)
                .def_readonly("origin", &ScalarField::origin)
                .def_readonly("cell_size", &ScalarField::cellSize)
                .def_property_readonly("empty", &ScalarField::empty)
                .def("at", &ScalarField::at, py::arg("x"), py::arg("y"), py::arg("z"))
                .def("data_numpy", [](const ScalarField& f) {
                         py::array_t<float> arr({f.nz, f.ny, f.nx});
                         std::memcpy(arr.mutable_data(), f.data.data(), f.data.size() * sizeof(float));
                         return arr;
                     }, "Return field data as (nz, ny, nx) float32 numpy array.");

        // ---- IsoMesh --------------------------------------------------------
        py::class_<IsoMesh>(m, "IsoMesh")
                .def(py::init<>())
                .def_property_readonly("empty", &IsoMesh::empty)
                .def_property_readonly("positions", [](const IsoMesh& m) {
                         return pointsToNumpy(m.positions);
                     }, "(N,3) float32 vertex positions (3 per triangle, unwelded).")
                .def_property_readonly("normals", [](const IsoMesh& m) {
                         return pointsToNumpy(m.normals);
                     }, "(N,3) float32 per-vertex normals.");

        // ---- splat_points_to_field / marching_cubes -------------------------
        m.def("splat_points_to_field",
              [](const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                 float cell_size, float radius, std::size_t max_nodes) {
                  auto pts = numpyToPoints(arr);
                  py::gil_scoped_release release;
                  return splatPointsToField(pts, cell_size, radius, max_nodes);
              },
              py::arg("points"), py::arg("cell_size"), py::arg("radius"),
              py::arg("max_nodes") = std::size_t(8'000'000),
              "Build a union-of-balls scalar field from an (N,3) point array.");

        m.def("marching_cubes",
              [](const ScalarField& f, float isolevel) {
                  py::gil_scoped_release release;
                  return marchingCubes(f, isolevel);
              },
              py::arg("field"), py::arg("isolevel") = 0.5f,
              "Extract an isosurface mesh from a ScalarField. Returns an IsoMesh.");

        // ---- iso_mesh_to_geometry -------------------------------------------
        m.def("iso_mesh_to_geometry",
              [](const IsoMesh& iso) {
                  auto geom = BufferGeometry::create();
                  const auto n = iso.positions.size();
                  std::vector<float> pos(n * 3), nrm(n * 3);
                  for (std::size_t i = 0; i < n; ++i) {
                      pos[i * 3 + 0] = iso.positions[i].x;
                      pos[i * 3 + 1] = iso.positions[i].y;
                      pos[i * 3 + 2] = iso.positions[i].z;
                      nrm[i * 3 + 0] = iso.normals[i].x;
                      nrm[i * 3 + 1] = iso.normals[i].y;
                      nrm[i * 3 + 2] = iso.normals[i].z;
                  }
                  geom->setAttribute("position", FloatBufferAttribute::create(std::move(pos), 3));
                  geom->setAttribute("normal", FloatBufferAttribute::create(std::move(nrm), 3));
                  return geom;
              },
              py::arg("iso_mesh"),
              "Convert an IsoMesh to a BufferGeometry ready for tp.Mesh(geom, material).");
    }

}// namespace threepp_py
