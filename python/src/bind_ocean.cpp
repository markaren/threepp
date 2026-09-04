// FFT-displaced ocean: DisplacedMesh + the ready-made Ocean convenience type.
//
// Gated on the Vulkan backend (THREEPP_PY_HAS_VULKAN). The FFT/displace/foam
// pipeline that turns these surfaces into water lives entirely in the Vulkan
// renderer, and the lib only compiles DisplacedMesh/Ocean when
// THREEPP_WITH_VULKAN is ON — so on a GL-only build tp.Ocean / tp.DisplacedMesh
// are simply absent, the same convention as tp.VulkanRenderer (check
// tp.HAS_VULKAN before constructing one).
//
// Surface exposed: the wave Params, the adaptive-density MeshWarp, world-space
// foam disturbances, the CPU height sampler, and — since a Python boat is now a
// real use case (python/examples/warp_sailboat.py) — the vessel pair:
// HullExclusion (the hull displaces the water, on its own waterline plane) and
// VesselWake (Kelvin V + bow bump + foam trail) with the trail bookkeeping done
// in C++ so Python never loops over records per frame.
#include "bindings.hpp"

#ifdef THREEPP_PY_HAS_VULKAN

#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>

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
                .def_readwrite("fetch", &DisplacedMesh::Params::fetch,
                               "Fetch (m of open water upwind). 0 = fully developed Phillips/PM sea (long swell, "
                               "peak ~8 V^2/g). Finite = JONSWAP young sea: shorter, steeper waves with more "
                               "energy at the 10-40 m scale and less swell; 20e3-40e3 reads as a coastal sea. "
                               "Live-tunable; saturates at ~1600 V^2.")
                .def_readwrite("wave_scale", &DisplacedMesh::Params::waveScale, "Global wave-height multiplier; 1.0 = physical.")
                .def_readwrite("choppiness", &DisplacedMesh::Params::choppiness,
                               "Horizontal pull / crest sharpness; ~0.45 realistic, >=0.8 folds crests.")
                .def_readwrite("foam_amount", &DisplacedMesh::Params::foamAmount,
                               "Natural whitecap foam scale, live-tunable (1 = ocean whitewater, 0 = none; "
                               "wake/disturbance foam unaffected). Ocean auto-derives ~size/300.")
                .def_readwrite("texture_size_0", &DisplacedMesh::Params::textureSize0, "Cascade-0 FFT resolution (power of two).")
                .def_readwrite("texture_size_1", &DisplacedMesh::Params::textureSize1, "Cascade-1 FFT resolution (power of two).")
                .def_readwrite("texture_size_2", &DisplacedMesh::Params::textureSize2, "Cascade-2 FFT resolution (power of two).");

        // Adaptive vertex-density warp (DisplacedMesh::MeshWarp).
        py::class_<DisplacedMesh::MeshWarp>(displaced, "MeshWarp")
                .def_readwrite("center_x", &DisplacedMesh::MeshWarp::centerX)
                .def_readwrite("center_z", &DisplacedMesh::MeshWarp::centerZ)
                .def_readwrite("half_range", &DisplacedMesh::MeshWarp::halfRange, "Half-extent the warp covers; 0 = uniform grid (disabled).")
                .def_readwrite("coef_a", &DisplacedMesh::MeshWarp::coefA, "1 = uniform; lower = denser centre (~0.1 ≈ 10 cm centre / 2.7 m edge).");

        // ---- the vessel pair -------------------------------------------------
        // Hull exclusion. The footprint is plan-form tapered (fine bow, ~75 %
        // beam at the transom) and fades into the wave field over ~2 m outside
        // the hull edge; inside it the surface sits on the vessel's waterline
        // PLANE, so a hull riding a swell carries her waterline with her.
        py::class_<DisplacedMesh::HullExclusion>(displaced, "HullExclusion")
                .def_readwrite("center_x", &DisplacedMesh::HullExclusion::centerX,
                               "World X of the hull's centre (the exclusion origin).")
                .def_readwrite("center_z", &DisplacedMesh::HullExclusion::centerZ)
                .def_readwrite("half_length", &DisplacedMesh::HullExclusion::halfLength,
                               "Half the vessel's length (m). 0 DISABLES hull exclusion and the wake.")
                .def_readwrite("half_beam", &DisplacedMesh::HullExclusion::halfBeam, "Half the beam (m).")
                .def_readwrite("sin_yaw", &DisplacedMesh::HullExclusion::sinYaw,
                               "sin/cos of the heading: forward = (sin_yaw, cos_yaw), starboard = (cos_yaw, -sin_yaw).")
                .def_readwrite("cos_yaw", &DisplacedMesh::HullExclusion::cosYaw)
                .def_readwrite("center_y", &DisplacedMesh::HullExclusion::centerY,
                               "World Y of the hull's DESIGN WATERLINE at (center_x, center_z) — "
                               "the height the water should meet the hull at, not the deck. "
                               "0 (the default) = the ocean rest plane, i.e. the pre-2026-08 behaviour.")
                .def_readwrite("pitch", &DisplacedMesh::HullExclusion::pitch,
                               "Waterline-plane pitch (rad), POSITIVE = bow up. Clamped to +/-1 rad.")
                .def_readwrite("roll", &DisplacedMesh::HullExclusion::roll,
                               "Waterline-plane roll (rad), POSITIVE = starboard up. Clamped to +/-1 rad.")
                .def("set_pose",
                     [](DisplacedMesh::HullExclusion& h, float x, float z, float y,
                        float yaw, float pitch, float roll, float half_length, float half_beam) {
                         h.centerX = x;
                         h.centerZ = z;
                         h.centerY = y;
                         h.sinYaw = std::sin(yaw);
                         h.cosYaw = std::cos(yaw);
                         h.pitch = pitch;
                         h.roll = roll;
                         if (half_length >= 0.f) h.halfLength = half_length;
                         if (half_beam >= 0.f) h.halfBeam = half_beam;
                     },
                     py::arg("x"), py::arg("z"), py::arg("y") = 0.f,
                     py::arg("yaw") = 0.f, py::arg("pitch") = 0.f, py::arg("roll") = 0.f,
                     py::arg("half_length") = -1.f, py::arg("half_beam") = -1.f,
                     "One-call per-frame update: world XZ, waterline height, heading (rad, 0 = +Z) "
                     "and the two plane angles. half_length/half_beam are left alone when negative, "
                     "so the usual pattern is to set them once and then call set_pose(x, z, y, yaw, "
                     "pitch, roll) every frame.");

        // One snapshot of the vessel's pose; the Kelvin V-wake is the envelope
        // over the trail, which is why the wake bends through her turns.
        py::class_<DisplacedMesh::WakeSample>(displaced, "WakeSample")
                .def(py::init<>())
                .def_readwrite("world_x", &DisplacedMesh::WakeSample::worldX)
                .def_readwrite("world_z", &DisplacedMesh::WakeSample::worldZ)
                .def_readwrite("sin_yaw", &DisplacedMesh::WakeSample::sinYaw)
                .def_readwrite("cos_yaw", &DisplacedMesh::WakeSample::cosYaw)
                .def_readwrite("speed", &DisplacedMesh::WakeSample::speed, "m/s along +heading at emission.")
                .def_readwrite("age", &DisplacedMesh::WakeSample::age, "Seconds since emission.")
                .def("__repr__", [](const DisplacedMesh::WakeSample& s) {
                    return "<WakeSample (" + std::to_string(s.worldX) + ", " + std::to_string(s.worldZ) +
                           ") speed=" + std::to_string(s.speed) + " age=" + std::to_string(s.age) + ">";
                });

        py::class_<DisplacedMesh::VesselWake>(displaced, "VesselWake")
                .def_readwrite("forward_speed", &DisplacedMesh::VesselWake::forwardSpeed,
                               "m/s along +heading. The whole wake fades out below ~0.5 m/s; 0 disables it.")
                .def_readwrite("enabled", &DisplacedMesh::VesselWake::enabled)
                .def_readwrite("trail", &DisplacedMesh::VesselWake::trail,
                               "Historical pose snapshots (list of WakeSample). Reading COPIES and "
                               "writing REPLACES — mutating the returned list does not write through. "
                               "Use mesh.add_wake_sample()/age_wake() for the per-frame path.");

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
                .def_property_readonly("hull_exclusion", [](DisplacedMesh& o) { return &o.hullExclusion; },
                                       py::return_value_policy::reference_internal,
                                       "The vessel's footprint + waterline plane; set each frame before "
                                       "render(). half_length = 0 (the default) disables it AND the wake.")
                .def_property_readonly("wake", [](DisplacedMesh& o) { return &o.wake; },
                                       py::return_value_policy::reference_internal,
                                       "Kelvin V-wake / bow bump / foam trail. Shares the hull_exclusion "
                                       "pose, so set that first.")
                // Trail bookkeeping in C++: a Python loop over 64 records every
                // frame is both slower and (because `wake.trail` copies) wrong.
                .def("add_wake_sample",
                     [](DisplacedMesh& o, float x, float z, float sin_yaw, float cos_yaw,
                        float speed, size_t max_samples) {
                         auto& t = o.wake.trail;
                         if (max_samples > 0)
                             while (t.size() >= max_samples) t.erase(t.begin());// oldest out
                         DisplacedMesh::WakeSample s{};
                         s.worldX = x;
                         s.worldZ = z;
                         s.sinYaw = sin_yaw;
                         s.cosYaw = cos_yaw;
                         s.speed = speed;
                         s.age = 0.f;
                         t.push_back(s);
                     },
                     py::arg("x"), py::arg("z"), py::arg("sin_yaw"), py::arg("cos_yaw"),
                     py::arg("speed"), py::arg("max_samples") = 64,
                     "Emit one wake snapshot at the vessel's current pose (age 0), dropping the "
                     "oldest once the trail is full. The renderer's hard cap is 64 samples; "
                     "overflow beyond it is dropped silently on upload. The C++ showcase's "
                     "cadence is 10 Hz OR every 1 m travelled, whichever fires first.")
                .def("age_wake",
                     [](DisplacedMesh& o, float dt, float max_age, size_t max_samples) {
                         auto& t = o.wake.trail;
                         for (auto& s : t) s.age += dt;
                         t.erase(std::remove_if(t.begin(), t.end(),
                                                [max_age](const DisplacedMesh::WakeSample& s) {
                                                    return s.age > max_age;
                                                }),
                                 t.end());
                         if (max_samples > 0 && t.size() > max_samples)
                             t.erase(t.begin(), t.end() - static_cast<long long>(max_samples));
                         return t.size();
                     },
                     py::arg("dt"), py::arg("max_age") = 6.0f, py::arg("max_samples") = 64,
                     "Age every trail sample by dt, drop anything older than max_age, and keep at "
                     "most max_samples (newest). Returns the surviving count. Call once per frame.")
                .def("clear_wake", [](DisplacedMesh& o) { o.wake.trail.clear(); },
                     "Drop the whole trail (e.g. after teleporting the vessel, so the wake does "
                     "not stretch across the map).")
                .def("sample_wake_height", &DisplacedMesh::sampleWakeHeight,
                     py::arg("world_x"), py::arg("world_z"),
                     "CPU mirror of the shader's wake height (bow bump + bow V-wedge + the "
                     "trail-summed Kelvin V) at a world XZ. 0 with no active vessel or below the "
                     "speed gate. Add to sample_height() to make a buoy bob through a passing wake.")
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
                                 unsigned int fft_size, float size_z, unsigned int resolution_z,
                                 const std::string& look, float fetch) {
                         Ocean::Options o;
                         o.size = size;
                         o.sizeZ = size_z;
                         o.resolution = resolution;
                         o.resolutionZ = resolution_z;
                         o.windSpeed = wind_speed;
                         o.windTheta = wind_theta;
                         o.fetch = fetch;
                         o.choppiness = choppiness;
                         o.waveScale = wave_scale;
                         o.tileSize1 = tile_size_1;
                         o.tileSize2 = tile_size_2;
                         o.fftSize = fft_size;
                         if (look == "ocean") o.look = Ocean::Look::Ocean;
                         else if (look == "pond") o.look = Ocean::Look::Pond;
                         else if (look == "fjord") o.look = Ocean::Look::Fjord;
                         else if (look != "auto")
                             throw py::value_error("look must be 'auto', 'ocean', 'pond' or 'fjord'");
                         return Ocean::create(o);
                     }),
                     py::arg("size") = 1000.0f, py::arg("resolution") = 512u,
                     py::arg("wind_speed") = 10.0f, py::arg("wind_theta") = 0.6f,
                     py::arg("choppiness") = 0.55f, py::arg("wave_scale") = 1.0f,
                     py::arg("tile_size_1") = -1.0f, py::arg("tile_size_2") = -1.0f,
                     py::arg("fft_size") = 1024u,
                     py::arg("size_z") = 0.0f, py::arg("resolution_z") = 0u,
                     py::arg("look") = "auto", py::arg("fetch") = 0.0f,
                     "A ready-to-use FFT ocean. Add it to a Scene and render with the Vulkan "
                     "renderer. size is the local-X extent (m); size_z=0 makes a square, >0 a "
                     "rectangle (vertices only where the water is — the wave field is unaffected). "
                     "resolution is the vertex grid along X; resolution_z=0 keeps cells square-ish. "
                     "fft_size caps the per-cascade FFT resolutions (band-passed cascades auto-size "
                     "below it). tile_size_1/2 default to -1 = auto: scaled from the larger extent "
                     "(a 1000 m ocean gets the classic 127/9.3 bands, a 16 m pond gets dm-scale "
                     "ripples); 0 disables a cascade, >0 pins it. Ponds also want wind_speed 2-5. "
                     "look picks the water material: 'auto' = pond recipe under 100 m, ocean above; "
                     "'ocean'/'pond' pin it regardless of scale; 'fjord' is the glacial recipe whose turquoise "
                     "comes from volume SCATTERING (Vulkan deferred only). fetch (m) = 0 is a fully developed sea "
                     "(long swell); 20e3-40e3 gives the shorter, steeper JONSWAP chop of a coastal sea "
                     "(see Params.fetch).")
                .def("warp_toward", &Ocean::warpToward,
                     py::arg("world_x"), py::arg("world_z"), py::arg("coef_a") = 0.1f,
                     "Pack vertex density toward a world-space focus point (e.g. the camera). "
                     "Call each frame before render().")
                .def("set_wind", &Ocean::setWind, py::arg("speed"), py::arg("theta"),
                     "Set wind speed (m/s) and direction (radians). Live: the renderer "
                     "regenerates the spectra next frame and the sea morphs smoothly into "
                     "the new state.");
    }

}// namespace threepp_py

#else// THREEPP_PY_HAS_VULKAN not defined — GL-only build

namespace threepp_py {

    // The FFT ocean needs the Vulkan backend; tp.Ocean / tp.DisplacedMesh are
    // absent on a GL-only build (mirrors init_vulkan's no-op fallback).
    void init_ocean(py::module_&) {}

}// namespace threepp_py

#endif
