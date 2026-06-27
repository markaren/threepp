#include "bindings.hpp"

#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/Texture.hpp"

#include <pybind11/stl.h>

namespace threepp_py {

    void init_vegetation(py::module_& m) {

        using namespace threepp;
        using namespace threepp::vegetation;

        py::enum_<CrownShape>(m, "CrownShape")
                .value("Sphere",     CrownShape::Sphere)
                .value("Ellipsoid",  CrownShape::Ellipsoid)
                .value("Cone",       CrownShape::Cone)
                .value("Hemisphere", CrownShape::Hemisphere)
                .value("Cylinder",   CrownShape::Cylinder)
                .export_values();

        py::enum_<LeafStyle>(m, "LeafStyle")
                .value("Quad",      LeafStyle::Quad)
                .value("Cluster",   LeafStyle::Cluster)
                .value("CrossQuad", LeafStyle::CrossQuad)
                .value("Blob",      LeafStyle::Blob)
                .export_values();

        py::class_<TreeParams>(m, "TreeParams")
                .def(py::init<>())
                .def_readwrite("seed",               &TreeParams::seed)
                .def_readwrite("trunk_height",       &TreeParams::trunkHeight)
                .def_readwrite("trunk_radius",       &TreeParams::trunkRadius)
                .def_readwrite("crown_shape",        &TreeParams::crownShape)
                .def_readwrite("crown_radius_x",     &TreeParams::crownRadiusX)
                .def_readwrite("crown_radius_z",     &TreeParams::crownRadiusZ)
                .def_readwrite("crown_height",       &TreeParams::crownHeight)
                .def_readwrite("attractor_count",    &TreeParams::attractorCount)
                .def_readwrite("influence_distance", &TreeParams::influenceDistance)
                .def_readwrite("kill_distance",      &TreeParams::killDistance)
                .def_readwrite("segment_length",     &TreeParams::segmentLength)
                .def_readwrite("max_iterations",     &TreeParams::maxIterations)
                .def_readwrite("randomness",         &TreeParams::randomness)
                .def_readwrite("tropism",            &TreeParams::tropism)
                .def_readwrite("radius_exponent",    &TreeParams::radiusExponent)
                .def_readwrite("min_branch_radius",  &TreeParams::minBranchRadius)
                .def_readwrite("radial_segments",    &TreeParams::radialSegments)
                .def_readwrite("leaf_style",         &TreeParams::leafStyle)
                .def_readwrite("leaf_size",          &TreeParams::leafSize)
                .def_readwrite("leaf_density",       &TreeParams::leafDensity)
                .def_readwrite("leaves_per_cluster", &TreeParams::leavesPerCluster)
                .def_readwrite("leaf_spread",        &TreeParams::leafSpread)
                .def_readwrite("leaf_clumping",      &TreeParams::leafClumping)
                .def_property("bark_color",
                        [](const TreeParams& p) {
                            return std::vector<float>(p.barkColor.begin(), p.barkColor.end());
                        },
                        [](TreeParams& p, const std::vector<float>& c) {
                            if (c.size() >= 3) p.barkColor = {c[0], c[1], c[2]};
                        })
                .def_property("leaf_color",
                        [](const TreeParams& p) {
                            return std::vector<float>(p.leafColor.begin(), p.leafColor.end());
                        },
                        [](TreeParams& p, const std::vector<float>& c) {
                            if (c.size() >= 3) p.leafColor = {c[0], c[1], c[2]};
                        });

        py::class_<TreeGenerator>(m, "TreeGenerator")
                .def(py::init<unsigned int>(), py::arg("seed") = 1337u)
                .def("reseed", &TreeGenerator::reseed, py::arg("seed"))
                .def_property_readonly("seed",       &TreeGenerator::seed)
                .def_property_readonly("node_count", &TreeGenerator::nodeCount)
                .def("build_skeleton", [](TreeGenerator& self, const TreeParams& tp) {
                         py::gil_scoped_release release;
                         self.buildSkeleton(tp);
                     }, py::arg("params"))
                .def("make_trunk_geometry", [](TreeGenerator& self, const TreeParams& tp) {
                         py::gil_scoped_release release;
                         return self.makeTrunkGeometry(tp);
                     }, py::arg("params"))
                .def("make_leaf_geometry", [](TreeGenerator& self, const TreeParams& tp) {
                         py::gil_scoped_release release;
                         return self.makeLeafGeometry(tp);
                     }, py::arg("params"))
                // Convenience: build skeleton then return trunk geometry in one call.
                .def("create_trunk_geometry", [](TreeGenerator& self, const TreeParams& tp) {
                         py::gil_scoped_release release;
                         return self.createTrunkGeometry(tp);
                     }, py::arg("params"))
                // Convenience: return leaf geometry (requires build_skeleton called first).
                .def("create_leaf_geometry", [](TreeGenerator& self, const TreeParams& tp) {
                         py::gil_scoped_release release;
                         return self.createLeafGeometry(tp);
                     }, py::arg("params"));

        // apply_preset(0..3, params) — Oak / Pine / Birch / Willow
        m.def("apply_tree_preset", [](int preset, TreeParams& p) {
                  applyPreset(preset, p);
              }, py::arg("preset"), py::arg("params"),
              "Apply species preset: 0=Oak, 1=Pine, 2=Birch, 3=Willow.");

        // ── Procedural textures ──────────────────────────────────────────

        m.def("make_leaf_texture",
              [](unsigned int size, unsigned int seed, const std::vector<float>& color) -> std::shared_ptr<Texture> {
                  std::array<float, 3> c = {color.size() > 0 ? color[0] : 0.26f,
                                            color.size() > 1 ? color[1] : 0.45f,
                                            color.size() > 2 ? color[2] : 0.14f};
                  py::gil_scoped_release release;
                  return makeLeafClusterTexture(size, seed, c);
              },
              py::arg("size") = 256u, py::arg("seed") = 1337u,
              py::arg("base_color") = std::vector<float>{0.26f, 0.45f, 0.14f},
              "RGBA leaf-cluster alpha-cutout DataTexture. Use mat.alpha_test = 0.5.");

        m.def("make_bark_textures",
              [](unsigned int size, unsigned int seed, const std::vector<float>& color)
                      -> py::tuple {
                  std::array<float, 3> c = {color.size() > 0 ? color[0] : 0.34f,
                                            color.size() > 1 ? color[1] : 0.24f,
                                            color.size() > 2 ? color[2] : 0.16f};
                  std::shared_ptr<DataTexture> alb, nrm;
                  {
                      py::gil_scoped_release release;
                      std::tie(alb, nrm) = makeBarkTextures(size, seed, c);
                  }// GIL reacquired before constructing Python objects
                  std::shared_ptr<Texture> a = alb, n = nrm;
                  return py::make_tuple(a, n);
              },
              py::arg("size") = 256u, py::arg("seed") = 1337u,
              py::arg("base_color") = std::vector<float>{0.34f, 0.24f, 0.16f},
              "Returns (albedo, normal) tiling bark Textures.");

        m.def("make_flower_texture",
              [](unsigned int size, unsigned int seed) -> std::shared_ptr<Texture> {
                  py::gil_scoped_release release;
                  return makeFlowerTexture(size, seed);
              },
              py::arg("size") = 128u, py::arg("seed") = 1337u,
              "RGBA wildflower alpha-cutout Texture. seed % 5 selects petal colour.");
    }

}// namespace threepp_py
