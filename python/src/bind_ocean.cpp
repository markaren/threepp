// FFT-displaced ocean: DisplacedMesh + the ready-made Ocean convenience type.
//
// Gated on the Vulkan backend (THREEPP_PY_HAS_VULKAN). The FFT/displace/foam
// pipeline that turns these surfaces into water lives entirely in the Vulkan
// renderer, and the lib only compiles DisplacedMesh/Ocean when
// THREEPP_WITH_VULKAN is ON — so on a GL-only build tp.Ocean / tp.DisplacedMesh
// are simply absent, the same convention as tp.VulkanRenderer (check
// tp.HAS_VULKAN before constructing one).
//
// Surface exposed (the hero-free core): the wave Params, the adaptive-density
// MeshWarp, world-space foam disturbances, and the CPU height sampler. The
// vessel-only wake / hull-exclusion fields are intentionally left unbound — they
// only mean anything with a boat, which this Python surface deliberately omits.
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_VULKAN

#include <pybind11/stl.h>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/DisplacedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Ocean.hpp"

using namespace threepp;

namespace threepp_py {

    void init_ocean(py::module_& m) {

        // ---- DisplacedMesh ---------------------------------------------------
        // Bound as a subclass of the already-registered Mesh: Mesh is a
        // non-virtual base of DisplacedMesh, so the concrete Object3D API bound
        // on Mesh (position/rotation/add/...) is inherited safely — same pattern
        // as InstancedMesh (see bind_objects.cpp).
        auto displaced = py::class_<DisplacedMesh, Mesh, std::shared_ptr<DisplacedMesh>>(m, "DisplacedMesh");

        // The three-cascade wave field knobs (DisplacedMesh::Params).
        py::class_<DisplacedMesh::Params>(displaced, "Params")
                .def_readwrite("tile_size_0", &DisplacedMesh::Params::tileSize0,
                               "Cascade-0 world tile (m); the largest swell wavelength. Must be > 0.")
                .def_readwrite("tile_size_1", &DisplacedMesh::Params::tileSize1, "Cascade-1 tile (m); 0 disables.")
                .def_readwrite("tile_size_2", &DisplacedMesh::Params::tileSize2, "Cascade-2 tile (m); 0 disables.")
                .def_readwrite("wind_theta", &DisplacedMesh::Params::windTheta, "Wind direction (radians, 0 = +X).")
                .def_readwrite("wind_speed", &DisplacedMesh::Params::windSpeed,
                               "Wind speed (m/s); the dominant wave-height lever (Phillips amplitude ~ V^4).")
                .def_readwrite("wave_scale", &DisplacedMesh::Params::waveScale, "Global wave-height multiplier; 1.0 = physical.")
                .def_readwrite("choppiness", &DisplacedMesh::Params::choppiness,
                               "Horizontal pull / crest sharpness; ~0.45 realistic, >=0.8 folds crests.")
                .def_readwrite("texture_size_0", &DisplacedMesh::Params::textureSize0, "Cascade-0 FFT resolution (power of two).")
                .def_readwrite("texture_size_1", &DisplacedMesh::Params::textureSize1, "Cascade-1 FFT resolution (power of two).")
                .def_readwrite("texture_size_2", &DisplacedMesh::Params::textureSize2, "Cascade-2 FFT resolution (power of two).");

        // Adaptive vertex-density warp (DisplacedMesh::MeshWarp).
        py::class_<DisplacedMesh::MeshWarp>(displaced, "MeshWarp")
                .def_readwrite("center_x", &DisplacedMesh::MeshWarp::centerX)
                .def_readwrite("center_z", &DisplacedMesh::MeshWarp::centerZ)
                .def_readwrite("half_range", &DisplacedMesh::MeshWarp::halfRange, "Half-extent the warp covers; 0 = uniform grid (disabled).")
                .def_readwrite("coef_a", &DisplacedMesh::MeshWarp::coefA, "1 = uniform; lower = denser centre (~0.1 ≈ 10 cm centre / 2.7 m edge).");

        displaced
                .def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                         return std::make_shared<DisplacedMesh>(std::move(g), as_material(mat));
                     }),
                     py::arg("geometry"), py::arg("material"),
                     "Low-level constructor. Most callers want Ocean instead, which builds the "
                     "plane + water material + cascade defaults for you.")
                // Mutable sub-objects — `mesh.params.wind_speed = 8` writes through.
                .def_property_readonly("params", [](DisplacedMesh& o) { return &o.params; },
                                       py::return_value_policy::reference_internal)
                .def_property_readonly("warp", [](DisplacedMesh& o) { return &o.warp; },
                                       py::return_value_policy::reference_internal)
                // World-space foam splats (boat waterline, splashes, anything). Clear
                // and repopulate each frame before render(); decays ~1.4 s half-life.
                .def("clear_foam_disturbances", &DisplacedMesh::clearFoamDisturbances)
                .def("add_foam_disturbance", &DisplacedMesh::addFoamDisturbance,
                     py::arg("world_x"), py::arg("world_z"), py::arg("radius"), py::arg("intensity"),
                     "Splat a gaussian foam blob at a world XZ (radius m, intensity in [0,1]).")
                // CPU mirror of the GPU wave height — for buoyancy / placing floats.
                // Valid after a Vulkan render() has filled the height fields.
                .def("sample_height", &DisplacedMesh::sampleHeight,
                     py::arg("world_x"), py::arg("world_z"), py::arg("cascade_mask") = 0b111u,
                     "Combined wave height (m) at a world XZ. cascade_mask selects cascades "
                     "(bit i = cascade i). Returns 0 until a Vulkan render() has run.");

        // ---- Ocean -----------------------------------------------------------
        // The one-liner: builds the plane geometry, the tuned transmissive water
        // material, and a sensible 3-cascade spectrum.
        py::class_<Ocean, DisplacedMesh, std::shared_ptr<Ocean>>(m, "Ocean")
                .def(py::init([](float size, unsigned int resolution, float wind_speed, float wind_theta,
                                 float choppiness, float wave_scale, float tile_size_1, float tile_size_2,
                                 unsigned int fft_size) {
                         Ocean::Options o;
                         o.size = size;
                         o.resolution = resolution;
                         o.windSpeed = wind_speed;
                         o.windTheta = wind_theta;
                         o.choppiness = choppiness;
                         o.waveScale = wave_scale;
                         o.tileSize1 = tile_size_1;
                         o.tileSize2 = tile_size_2;
                         o.fftSize = fft_size;
                         return Ocean::create(o);
                     }),
                     py::arg("size") = 1000.0f, py::arg("resolution") = 512u,
                     py::arg("wind_speed") = 10.0f, py::arg("wind_theta") = 0.6f,
                     py::arg("choppiness") = 0.55f, py::arg("wave_scale") = 1.0f,
                     py::arg("tile_size_1") = 100.0f, py::arg("tile_size_2") = 8.0f,
                     py::arg("fft_size") = 1024u,
                     "A ready-to-use FFT ocean. Add it to a Scene and render with the Vulkan "
                     "renderer. size is the tile extent (m); resolution is the vertex grid per side.")
                .def("warp_toward", &Ocean::warpToward,
                     py::arg("world_x"), py::arg("world_z"), py::arg("coef_a") = 0.1f,
                     "Pack vertex density toward a world-space focus point (e.g. the camera). "
                     "Call each frame before render().")
                .def("set_wind", &Ocean::setWind, py::arg("speed"), py::arg("theta"),
                     "Set wind speed (m/s) and direction (radians).");
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_VULKAN not defined — GL-only build

namespace threepp_py {

    // The FFT ocean needs the Vulkan backend; tp.Ocean / tp.DisplacedMesh are
    // absent on a GL-only build (mirrors init_vulkan's no-op fallback).
    void init_ocean(py::module_&) {}

}// namespace threepp_py

#endif
