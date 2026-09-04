#include "bindings.hpp"

#include "threepp/constants.hpp"
#include "threepp/extras/terrain/GeoScene.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/Texture.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>

namespace threepp_py {

    void init_terrain(py::module_& m) {

        using namespace threepp;
        using namespace threepp::terrain;

        py::enum_<NoiseType>(m, "NoiseType")
                .value("fBm",    NoiseType::fBm)
                .value("Ridged", NoiseType::Ridged)
                .value("Hybrid", NoiseType::Hybrid);

        py::enum_<Falloff>(m, "TerrainFalloff")
                .value("Off",    Falloff::None)   // 'None' is a Python keyword
                .value("Radial", Falloff::Radial);

        py::enum_<ErosionType>(m, "ErosionType")
                .value("Off",       ErosionType::None)  // 'None' is a Python keyword
                .value("Hydraulic", ErosionType::Hydraulic)
                .value("Thermal",   ErosionType::Thermal)
                .value("Both",      ErosionType::Both);

        // ── TerrainParams ─────────────────────────────────────────────────────
        py::class_<TerrainParams>(m, "TerrainParams")
                .def(py::init<>())
                .def_readwrite("seed",              &TerrainParams::seed)
                .def_readwrite("world_size",        &TerrainParams::worldSize)
                .def_readwrite("resolution",        &TerrainParams::resolution)
                .def_readwrite("noise_type",        &TerrainParams::noiseType)
                .def_readwrite("feature_scale",     &TerrainParams::featureScale)
                .def_readwrite("octaves",           &TerrainParams::octaves)
                .def_readwrite("lacunarity",        &TerrainParams::lacunarity)
                .def_readwrite("gain",              &TerrainParams::gain)
                .def_readwrite("amplitude",         &TerrainParams::amplitude)
                .def_readwrite("warp",              &TerrainParams::warp)
                .def_readwrite("ridge_sharpness",   &TerrainParams::ridgeSharpness)
                .def_readwrite("height_exponent",   &TerrainParams::heightExponent)
                .def_readwrite("terraces",          &TerrainParams::terraces)
                .def_readwrite("falloff",           &TerrainParams::falloff)
                .def_readwrite("falloff_start",     &TerrainParams::falloffStart)
                .def_readwrite("erosion",           &TerrainParams::erosion)
                .def_readwrite("droplets",          &TerrainParams::droplets)
                .def_readwrite("droplet_lifetime",  &TerrainParams::dropletLifetime)
                .def_readwrite("inertia",           &TerrainParams::inertia)
                .def_readwrite("sediment_capacity", &TerrainParams::sedimentCapacity)
                .def_readwrite("min_slope",         &TerrainParams::minSlope)
                .def_readwrite("erode_speed",       &TerrainParams::erodeSpeed)
                .def_readwrite("deposit_speed",     &TerrainParams::depositSpeed)
                .def_readwrite("evaporation",       &TerrainParams::evaporation)
                .def_readwrite("gravity",           &TerrainParams::gravity)
                .def_readwrite("erosion_radius",    &TerrainParams::erosionRadius)
                .def_readwrite("talus_angle",       &TerrainParams::talusAngle)
                .def_readwrite("thermal_iterations",&TerrainParams::thermalIterations)
                .def_readwrite("thermal_rate",      &TerrainParams::thermalRate)
                .def_readwrite("snow_line",         &TerrainParams::snowLine)
                .def_readwrite("snow_noise_amp",    &TerrainParams::snowNoiseAmp)
                .def_readwrite("snow_slope_max",    &TerrainParams::snowSlopeMax)
                .def_readwrite("slope_grass_max",   &TerrainParams::slopeGrassMax)
                .def_readwrite("slope_rock_min",    &TerrainParams::slopeRockMin)
                .def_readwrite("band_edge",         &TerrainParams::bandEdge)
                .def_readwrite("ao_strength",       &TerrainParams::aoStrength)
                .def_readwrite("ao_max",            &TerrainParams::aoMax)
                .def_property("rock_color",
                        [](const TerrainParams& p) { return std::vector<float>(p.rockColor.begin(),  p.rockColor.end()); },
                        [](TerrainParams& p, const std::vector<float>& c) { if (c.size() >= 3) p.rockColor  = {c[0],c[1],c[2]}; })
                .def_property("grass_color",
                        [](const TerrainParams& p) { return std::vector<float>(p.grassColor.begin(), p.grassColor.end()); },
                        [](TerrainParams& p, const std::vector<float>& c) { if (c.size() >= 3) p.grassColor = {c[0],c[1],c[2]}; })
                .def_property("scree_color",
                        [](const TerrainParams& p) { return std::vector<float>(p.screeColor.begin(), p.screeColor.end()); },
                        [](TerrainParams& p, const std::vector<float>& c) { if (c.size() >= 3) p.screeColor = {c[0],c[1],c[2]}; })
                .def_property("snow_color",
                        [](const TerrainParams& p) { return std::vector<float>(p.snowColor.begin(),  p.snowColor.end()); },
                        [](TerrainParams& p, const std::vector<float>& c) { if (c.size() >= 3) p.snowColor  = {c[0],c[1],c[2]}; });

        // ── TerrainGenerator ──────────────────────────────────────────────────
        py::class_<TerrainGenerator>(m, "TerrainGenerator")
                .def(py::init<unsigned int>(), py::arg("seed") = 1337u)
                .def("reseed", &TerrainGenerator::reseed, py::arg("seed"))
                .def_property_readonly("seed", &TerrainGenerator::seed)
                .def_property_readonly("dim",  &TerrainGenerator::dim)
                // Step 1: build noise field
                .def("build_field", [](TerrainGenerator& g, const TerrainParams& p) {
                         py::gil_scoped_release r;
                         g.buildField(p);
                     }, py::arg("params"))
                // Step 2: erode in place
                .def("erode", [](TerrainGenerator& g, const TerrainParams& p) {
                         py::gil_scoped_release r;
                         g.erode(p);
                     }, py::arg("params"))
                // Step 3: bake field → BufferGeometry
                .def("make_geometry", [](TerrainGenerator& g, const TerrainParams& p) {
                         py::gil_scoped_release r;
                         return g.makeGeometry(p);
                     }, py::arg("params"))
                .def("displace_to", [](TerrainGenerator& g, BufferGeometry& geo, const TerrainParams& p) {
                         py::gil_scoped_release r;
                         g.displaceTo(geo, p);
                     }, py::arg("geometry"), py::arg("params"))
                // Convenience: build [+ erode] → geometry in one call
                .def("create_geometry", [](TerrainGenerator& g, const TerrainParams& p, bool withErosion) {
                         py::gil_scoped_release r;
                         return g.createGeometry(p, withErosion);
                     }, py::arg("params"), py::arg("with_erosion") = false)
                // Analytic height query (pre-erosion, useful for physics/placement)
                .def("height_at", &TerrainGenerator::heightAt,
                     py::arg("wx"), py::arg("wz"), py::arg("params"))
                // Raw field as float32 numpy array of shape (dim, dim)
                .def("get_field", [](const TerrainGenerator& g) {
                         const auto& f = g.getField();
                         const int d = g.dim();
                         py::array_t<float> arr({d, d});
                         if (!f.empty()) std::memcpy(arr.mutable_data(), f.data(), f.size() * sizeof(float));
                         return arr;
                     },
                     "Height field as float32 numpy array of shape (dim, dim), values in [0,1].")
                // Bake slope/altitude splat → RGBA8 numpy array of shape (dim, dim, 4)
                .def("bake_splat_colors", [](TerrainGenerator& g, const TerrainParams& p) {
                         std::vector<unsigned char> buf;
                         const int d = g.dim();
                         {
                             py::gil_scoped_release r;
                             buf = g.bakeSplatColors(p);
                         }
                         py::array_t<uint8_t> arr({d, d, 4});
                         if (!buf.empty()) std::memcpy(arr.mutable_data(), buf.data(), buf.size());
                         return arr;
                     }, py::arg("params"),
                     "Bake slope/altitude splat into RGBA8 numpy array of shape (dim, dim, 4).")
                // Convenience: bake splat → sRGB DataTexture (ready to assign to material.map)
                .def("bake_splat_texture", [](TerrainGenerator& g, const TerrainParams& p) -> std::shared_ptr<Texture> {
                         std::vector<unsigned char> buf;
                         const int d = g.dim();
                         {
                             py::gil_scoped_release r;
                             buf = g.bakeSplatColors(p);
                         }
                         auto tex = DataTexture::create(ImageData(std::move(buf)),
                                                        static_cast<unsigned int>(d),
                                                        static_cast<unsigned int>(d));
                         tex->colorSpace  = ColorSpace::sRGB;
                         tex->magFilter   = Filter::Linear;
                         tex->minFilter   = Filter::Linear;
                         tex->needsUpdate();
                         std::shared_ptr<Texture> t = tex;
                         return t;
                     }, py::arg("params"),
                     "Bake splat colours into a sRGB DataTexture ready for material.map.");

        // ── GeoScene ──────────────────────────────────────────────────────────
        // Real-world geodata terrain (Kartverket DTM region pack) as ONE Group:
        // tiles + LOD, band structure sets, cliff shell, canopy forest and
        // synthetic bathymetry, all wired with the norway_terrain defaults.
        // Holder is shared_ptr and the base is Group, so it drops straight into
        // scene.add() and behaves like any other Object3D from Python.
        py::class_<GeoScene, Group, std::shared_ptr<GeoScene>>(m, "GeoScene")
                .def(py::init([](const std::string& packDir, bool bands, bool cliffShell,
                                 bool forest, float forestExtent,
                                 const std::array<float, 3>& forestFocus, int forestCap,
                                 bool scatter, float shellLevelStep, float shellExtent,
                                 bool bathymetry, float shoreSlope, float maxDepth,
                                 unsigned int seed) {
                         GeoSceneOptions o;
                         o.packDir = packDir;
                         o.bands = bands;
                         o.cliffShell = cliffShell;
                         o.forest = forest;
                         o.forestExtent = forestExtent;
                         o.forestCap = forestCap;
                         o.focus.set(forestFocus[0], forestFocus[1], forestFocus[2]);
                         o.scatter = scatter;
                         o.shellLevelStep = shellLevelStep;
                         o.shellExtent = shellExtent;
                         o.bathymetry = bathymetry;
                         o.shoreSlope = shoreSlope;
                         o.maxDepth = maxDepth;
                         o.seed = seed;
                         // Loading a 4 km 1 m pack is seconds of I/O + a distance
                         // transform + shell/forest baking: hold no GIL for it,
                         // or a threaded caller stalls for the whole load.
                         py::gil_scoped_release r;
                         return GeoScene::create(o);
                     }),
                     py::arg("pack_dir"), py::arg("bands") = true,
                     py::arg("cliff_shell") = true, py::arg("forest") = true,
                     py::arg("forest_extent") = 1100.f,
                     py::arg("forest_focus") = std::array<float, 3>{0.f, 0.f, 0.f},
                     py::arg("forest_cap") = 40000, py::arg("scatter") = true,
                     py::arg("shell_level_step") = 2.f, py::arg("shell_extent") = 1200.f,
                     py::arg("bathymetry") = true, py::arg("shore_slope") = 0.35f,
                     py::arg("max_depth") = 180.f, py::arg("seed") = 4242u,
                     "Load a geodata region pack and build the whole terrain scene. "
                     "Raises RuntimeError if the pack directory is missing or malformed.")
                .def("update", [](GeoScene& g, const Vector3& p) { g.update(p); }, py::arg("pos"),
                     "Once per frame, with the ACTIVE camera position: tile LOD + scatter.")
                .def("update", [](GeoScene& g, const std::array<float, 3>& p) {
                         g.update(Vector3(p[0], p[1], p[2]));
                     }, py::arg("pos"))
                .def("height_at", &GeoScene::heightAt, py::arg("x"), py::arg("z"),
                     "Terrain height (m) at a world XZ — DEM + road carve + relief + bathymetry.")
                .def_property_readonly("pack_world_size", &GeoScene::packWorldSize)
                .def_property_readonly("sea_level", &GeoScene::seaLevel)
                .def_property_readonly("height_min", [](const GeoScene& g) { return g.pack().region.heightMin; })
                .def_property_readonly("height_max", [](const GeoScene& g) { return g.pack().region.heightMax; })
                .def_property_readonly("attribution", [](const GeoScene& g) { return g.pack().region.attribution; })
                .def_property_readonly("stats", [](const GeoScene& g) {
                         const auto s = g.stats();
                         py::dict d;
                         d["tiles"] = s.tiles;
                         d["baking"] = s.baking;
                         d["shell_tris"] = s.shellTris;
                         d["forest_sites"] = s.forestSites;
                         d["forest_cells"] = s.forestCells;
                         d["load_seconds"] = s.loadSeconds;
                         return d;
                     },
                     "dict: tiles, baking, shell_tris, forest_sites, forest_cells, load_seconds.");

        // ── Free functions ────────────────────────────────────────────────────
        m.def("apply_terrain_preset",
              [](int preset, TerrainParams& p) { applyPreset(preset, p); },
              py::arg("preset"), py::arg("params"),
              "Apply named preset: 0=Alpine, 1=Rolling Hills, 2=Desert Mesa, 3=Volcanic.");

        m.def("terrain_to_json",
              [](const TerrainParams& p) { return toJson(p); },
              py::arg("params"), "Serialise TerrainParams to a JSON string.");

        m.def("terrain_from_json",
              [](const std::string& json, TerrainParams& p) { return fromJson(json, p); },
              py::arg("json"), py::arg("params"),
              "Deserialise TerrainParams from a JSON string. Unknown keys keep current value.");

        m.def("terrain_save_config",
              [](const std::string& path, const TerrainParams& p) { return saveConfig(path, p); },
              py::arg("path"), py::arg("params"));

        m.def("terrain_load_config",
              [](const std::string& path, TerrainParams& p) { return loadConfig(path, p); },
              py::arg("path"), py::arg("params"));
    }

}// namespace threepp_py
