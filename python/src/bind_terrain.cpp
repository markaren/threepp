#include "bindings.hpp"

#include "threepp/constants.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/Texture.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
                         tex->colorSpace = ColorSpace::sRGB;
                         tex->needsUpdate();
                         std::shared_ptr<Texture> t = tex;
                         return t;
                     }, py::arg("params"),
                     "Bake splat colours into a sRGB DataTexture ready for material.map.");

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
