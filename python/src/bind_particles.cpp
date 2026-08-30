// ParticleField: weather and granular fields whose particle count the CPU never
// walks, exposed to Python.
//
// The type is VULKAN ONLY at render time (see the header's banner): on the GL
// backend a field is a valid, zero-area Mesh that draws nothing. The class
// itself is built into the library unconditionally, so tp.ParticleField exists
// on a GL-only build too — it just renders no particles there.
//
// Bound as a subclass of the already-registered Mesh (Mesh is a non-virtual
// base of ParticleField, so the concrete Object3D API bound on Mesh comes
// through safely) — the same pattern as InstancedMesh, DisplacedMesh and Flock.
// Materials still arrive through as_material(): pybind11 corrupts a pointer
// up-cast across threepp's `virtual Material` base.
//
// The API divides in half by Ownership, and calling the wrong half THROWS in
// C++ (std::runtime_error -> RuntimeError in Python) rather than silently
// rendering an empty field:
//
//   Ownership.Renderer  set_emitter / set_emitter_time / set_follow_center
//   Ownership.HostRing  submit / set_live_count
//   Ownership.Interop   set_live_count (positions come from a foreign device API)
//
// Weather (rain, snow, embers, dust) is Ownership.Renderer: a closed-form
// device emitter, reproducible, seekable, and free on the bus.
#include "bindings.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/textures/Texture.hpp"

#include <pybind11/numpy.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace threepp_py {

    void init_particles(py::module_& m) {

        auto cls = py::class_<ParticleField, Mesh, std::shared_ptr<ParticleField>>(m, "ParticleField");

        // ── enums ───────────────────────────────────────────────────────────
        py::enum_<ParticleField::Ownership>(cls, "Ownership")
                .value("Interop", ParticleField::Ownership::Interop,
                       "A foreign device API (CUDA/PhysX) writes the positions. NOT reproducible.")
                .value("HostRing", ParticleField::Ownership::HostRing,
                       "The CPU owns the positions; feed them with submit().")
                .value("Renderer", ParticleField::Ownership::Renderer,
                       "The GPU owns the positions, written from the closed-form EmitterParams. "
                       "This is what weather uses.");

        py::enum_<ParticleField::WSemantic>(cls, "WSemantic")
                .value("InvMass", ParticleField::WSemantic::InvMass,
                       "w is PhysX's inverse mass; the radius comes from Config.uniform_radius.")
                .value("Radius", ParticleField::WSemantic::Radius,
                       "w IS the world radius (so emitter size_jitter is free per-particle variety).");

        // ── Config ──────────────────────────────────────────────────────────
        py::class_<ParticleField::Config>(cls, "Config")
                .def(py::init<>())
                .def_readwrite("capacity", &ParticleField::Config::capacity,
                               "Fixed for life — a field is created ONCE at its final capacity and "
                               "never resized (creating/destroying one is a structural scene change: "
                               "entry re-expansion, device idle, TAA history clear). Park a field with "
                               "set_live_count(0) instead.")
                .def_readwrite("ownership", &ParticleField::Config::ownership, "Who writes the positions.")
                .def_readwrite("w_semantic", &ParticleField::Config::wSemantic, "What the w channel means.")
                .def_readwrite("uniform_radius", &ParticleField::Config::uniformRadius,
                               "World radius the MeshRepr proxy geometry is authored at; also the "
                               "billboard/density size reference.")
                .def_readwrite("orientations", &ParticleField::Config::orientations,
                               "Allocate the snorm16x4 per-particle orientation buffer.")
                .def_readwrite("attributes", &ParticleField::Config::attributes,
                               "Allocate the per-particle vec4 appearance buffer (rgb = linear HDR "
                               "radiance, a reserved). It rides the POSITIONS' path exactly: under "
                               "Ownership.Interop it is a second exported allocation handed back by "
                               "the same enable_particle_field_interop call and snapshotted by the "
                               "same per-frame copy, so positions and colours can never diverge. "
                               "With it on, BillboardRepr uses attribute.rgb INSTEAD of the "
                               "color_hot/color_cool ramp — one scheme or the other, never a blend.")
                .def_readwrite("host_stable_slots", &ParticleField::Config::hostStableSlots,
                               "HostRing only: the host promises index i is the SAME particle in "
                               "every submit (fixed pool, dead slots left at w < 0, no compaction). "
                               "That makes the previous ring slot a real prevPositions buffer, which "
                               "is what BillboardRepr.stretch_seconds — the velocity streak — needs; "
                               "without the promise the stretch stays off on a host field. A frame "
                               "the host skips falls back to round sprites rather than smearing over "
                               "two steps, and a freshly spawned slot streaks from its predecessor "
                               "for one frame (bounded by stretch_max; spend it under a fade-in).");

        // ── MeshRepr (lit proxy per particle) ───────────────────────────────
        // geometry/material are set through set_mesh_repr, which also keeps the
        // field's own Mesh material in step — mutate the rest here.
        py::class_<ParticleField::MeshRepr>(cls, "MeshRepr")
                .def_readwrite("lod_far", &ParticleField::MeshRepr::lodFar,
                               "Metres beyond which the proxy collapses to zero area and the billboard "
                               "is expected to take over (BillboardRepr.lod_near). 0 = no LOD.")
                .def_readwrite("lod_fade", &ParticleField::MeshRepr::lodFade,
                               "Metres of soft shrink below lod_far, so the swap cross-dissolves.")
                .def_readwrite("near_cull", &ParticleField::MeshRepr::nearCull,
                               "Shrink particles closer than this to the camera; caps how big the "
                               "nearest one may get. 0 = off.")
                .def_readwrite("enabled", &ParticleField::MeshRepr::enabled);

        // ── BillboardRepr (additive, unlit, post-upscaler) ──────────────────
        py::class_<ParticleField::BillboardRepr>(cls, "BillboardRepr")
                .def_readwrite("texture", &ParticleField::BillboardRepr::texture,
                               "Optional sprite that MODULATES the procedural quad (rgb tint x alpha). "
                               "None binds a 1x1 white default.")
                .def_readwrite("size_scale", &ParticleField::BillboardRepr::sizeScale,
                               "Multiplies the particle radius.")
                .def_readwrite("color_hot", &ParticleField::BillboardRepr::colorHot,
                               "Linear HDR radiance at age 0.")
                .def_readwrite("color_cool", &ParticleField::BillboardRepr::colorCool,
                               "Linear HDR radiance at end of life.")
                .def_readwrite("intensity", &ParticleField::BillboardRepr::intensity,
                               "HDR scale on both colours. Additive over a field the camera stands "
                               "INSIDE — rain wants ~0.07, not ~0.5.")
                .def_readwrite("softness", &ParticleField::BillboardRepr::softness,
                               "0 = a tight spark, 1 = a broad glow.")
                .def_readwrite("fade_power", &ParticleField::BillboardRepr::fadePower,
                               "Brightness over life: (1 - age_frac)^fade_power. 0 = no fade (rain).")
                .def_readwrite("bright_jitter", &ParticleField::BillboardRepr::brightJitter,
                               "Per-particle brightness spread, hashed.")
                .def_readwrite("size_taper", &ParticleField::BillboardRepr::sizeTaper,
                               "Radius over life: r *= (1 - size_taper * age_frac).")
                .def_readwrite("stretch_seconds", &ParticleField::BillboardRepr::stretchSeconds,
                               "Seconds of travel to smear the quad over, along the particle's own "
                               "analytic velocity. 0 = a round sprite; rain without this reads as HAIL.")
                .def_readwrite("stretch_max", &ParticleField::BillboardRepr::stretchMax,
                               "Streak cap in multiples of the radius (WORLD units).")
                .def_readwrite("stretch_max_screen", &ParticleField::BillboardRepr::stretchMaxScreen,
                               "Streak cap as a fraction of the frame HEIGHT (NDC). 0 = off; ~0.045 "
                               "stops the nearest drop painting a bar across the frame.")
                .def_readwrite("near_fade", &ParticleField::BillboardRepr::nearFade,
                               "Fade the sprite out below this camera distance (m). 0 = off.")
                .def_readwrite("lod_near", &ParticleField::BillboardRepr::lodNear,
                               "Quads CLOSER than this collapse — MeshRepr.lod_far is the complementary "
                               "gate on the same field. 0 = off.")
                .def_readwrite("lod_fade", &ParticleField::BillboardRepr::lodFade,
                               "Metres of ramp above lod_near over which the quad fades IN.")
                .def_readwrite("glow", &ParticleField::BillboardRepr::glow,
                               "> 0 gives this field its own bloom pyramid. 0 skips the whole chain "
                               "(what weather wants — 300k rain streaks have nothing to bloom).")
                .def_readwrite("glow_threshold", &ParticleField::BillboardRepr::glowThreshold,
                               "Bright-pass knee for this field's own pyramid. 0 = no bright pass.")
                .def_readwrite("splash_ring_width", &ParticleField::BillboardRepr::splashRingWidth,
                               "Annulus width as a fraction of the splash ring's radius (1 = a filled "
                               "disc). Only means anything with emitter.surface.splash_seconds > 0.")
                // ── 4c: the sprite slice ────────────────────────────────────
                .def_readwrite("alpha_over", &ParticleField::BillboardRepr::alphaOver,
                               "Composite premultiplied SRC_ALPHA-over instead of additive, so a "
                               "sprite OCCLUDES what is behind it. Nothing is sorted: draws go in "
                               "field order and, within a field, in SLOT order, so submit "
                               "back-to-front for correct blending.")
                .def_readwrite("lit", &ParticleField::BillboardRepr::lit,
                               "Per-particle radiance = colour x (ambient + sun x HG phase), from "
                               "the scene's own sun. One lobe per particle — no shadow ray, no "
                               "cluster walk.")
                .def_readwrite("lit_phase_g", &ParticleField::BillboardRepr::litPhaseG,
                               "Henyey-Greenstein asymmetry for that lobe. ~0.35 = the forward-ish "
                               "scattering of a water parcel; 0 = isotropic.")
                .def_readwrite("lit_ambient", &ParticleField::BillboardRepr::litAmbient,
                               "Ambient radiance FLOOR added to the scene's summed AmbientLights, "
                               "in the same linear units — how dark the shaded side of a sprite is "
                               "allowed to get. An IBL-lit scene carries no AmbientLight, so a "
                               "scale on it would be a scale on zero.")
                .def_readwrite("opacity", &ParticleField::BillboardRepr::opacity,
                               "Coverage scale in alpha_over mode, before the texture/procedural "
                               "falloff. Ignored when additive.")
                // ── Volumetric transport (plans/particle-volumetric-sprites) ─
                .def_readwrite("volume_extinction",
                               &ParticleField::BillboardRepr::volumeExtinction,
                               "Dim each sprite by the transmittance of the field's OWN DensityRepr "
                               "volume between it and the camera, as pow(T_cam, this). 1 is the "
                               "physically honest answer; >1 is a 'more dust' grade that does not "
                               "disturb sigma_per_particle (which the deferred fog march also "
                               "reads). 0 is an EXACT no-op — the shader takes a uniform branch "
                               "around the march. This is what puts DUST LANES across a nebula: "
                               "additive blending is orderless and therefore carries zero occlusion "
                               "information, and this is the occlusion, from the medium the same "
                               "particles collectively are. Needs DensityRepr on.")
                .def_readwrite("volume_shadow", &ParticleField::BillboardRepr::volumeShadow,
                               "Mix each sprite toward T_sun * (volume_ambient + volume_sun_gain * "
                               "HG(V.L)) — the transmittance from the sprite to the sun through the "
                               "same volume, times one phase lobe. 0 = unshadowed (the pre-change "
                               "look and an exact no-op), 1 = fully replace. This is the LIT RIM "
                               "and the self-shadowed interior. Needs DensityRepr on.")
                .def_readwrite("volume_ambient", &ParticleField::BillboardRepr::volumeAmbient,
                               "Floor under the sun term, so the shadowed side of the volume does "
                               "not go black. Only read when volume_shadow > 0.")
                .def_readwrite("volume_sun_gain", &ParticleField::BillboardRepr::volumeSunGain,
                               "Scale on the sun's own contribution through T_sun. Unitless — it "
                               "multiplies the HG lobe (whose asymmetry is lit_phase_g), not a "
                               "radiance, because the sprite's colour already IS its radiance.")
                .def_readwrite("enabled", &ParticleField::BillboardRepr::enabled);

        // ── DensityRepr (world-anchored sigma_t volume the froxel pass reads) ─
        py::class_<ParticleField::DensityRepr>(cls, "DensityRepr")
                .def_readwrite("sigma_per_particle", &ParticleField::DensityRepr::sigmaPerParticle,
                               "Extinction (1/m) one particle contributes. Total optical mass is "
                               "N * sigma — a 300k weather field wants hundredths, not units.")
                .def_readwrite("albedo", &ParticleField::DensityRepr::albedo,
                               "Scattering albedo of THIS medium. Snow is bright; a rain curtain is dark.")
                .def_readwrite("anisotropy", &ParticleField::DensityRepr::anisotropy, "HG g for this medium.")
                .def_readwrite("center", &ParticleField::DensityRepr::center, "World centre of the volume box.")
                .def_readwrite("half_extent", &ParticleField::DensityRepr::halfExtent, "World half-size per axis.")
                .def_readonly("resolution", &ParticleField::DensityRepr::resolution,
                              "Voxels per axis — LATCHED at the first enable (set it via set_density_repr).")
                .def_readwrite("emissive_intensity", &ParticleField::DensityRepr::emissiveIntensity,
                               "HDR radiance scale of the analytic blackbody flame ramp. 0 = the exact "
                               "no-op (pure dust).")
                .def_readwrite("temp_bottom_k", &ParticleField::DensityRepr::tempBottomK)
                .def_readwrite("temp_top_k", &ParticleField::DensityRepr::tempTopK)
                .def_readwrite("temp_falloff", &ParticleField::DensityRepr::tempFalloff)
                .def_readwrite("enabled", &ParticleField::DensityRepr::enabled);

        // ── EmitterParams (Ownership.Renderer's closed form) ────────────────
        auto emitter = py::class_<ParticleField::EmitterParams>(cls, "EmitterParams");

        py::class_<ParticleField::EmitterParams::Surface>(emitter, "Surface")
                .def(py::init<>())
                .def_readwrite("enabled", &ParticleField::EmitterParams::Surface::enabled,
                               "Solve each slot's landing against a top-down height bake of the scene, "
                               "inside the same closed form. Snow rests; rain splashes.")
                .def_readwrite("extent", &ParticleField::EmitterParams::Surface::extent,
                               "Half-size of the square bake footprint (field-local m). 0 = use "
                               "spawn_half_extent.xz, which is also the toroidal wrap period.")
                .def_readwrite("resolution", &ParticleField::EmitterParams::Surface::resolution,
                               "Texels per axis, clamped to [16, 1024]. Sampled NEAREST by design.")
                .def_readwrite("search_top", &ParticleField::EmitterParams::Surface::searchTop)
                .def_readwrite("search_bottom", &ParticleField::EmitterParams::Surface::searchBottom,
                               "Vertical search band, field-local. top == bottom = derive it.")
                .def_readwrite("bias", &ParticleField::EmitterParams::Surface::bias,
                               "Metres above the baked surface the particle CENTRE rests; ~its own size.")
                .def_readwrite("rest_seconds", &ParticleField::EmitterParams::Surface::restSeconds)
                .def_readwrite("rest_jitter", &ParticleField::EmitterParams::Surface::restJitter)
                .def_readwrite("fade_seconds", &ParticleField::EmitterParams::Surface::fadeSeconds)
                .def_readwrite("splash_seconds", &ParticleField::EmitterParams::Surface::splashSeconds,
                               "> 0 makes a landed drop an expanding flat RING for this long instead "
                               "of resting. Drawn by the billboard representation.")
                .def_readwrite("splash_grow", &ParticleField::EmitterParams::Surface::splashGrow,
                               "Ring radius at the end of the splash, in multiples of the drop radius.");

        emitter
                .def(py::init<>())
                .def_readwrite("spawn_center", &ParticleField::EmitterParams::spawnCenter,
                               "Birth region centre, FIELD-LOCAL.")
                .def_readwrite("spawn_half_extent", &ParticleField::EmitterParams::spawnHalfExtent,
                               "Birth region half-size. A thin slab is an emission plane — author "
                               "snow/rain as a slab at the TOP with lifetime = height / speed, not as "
                               "a box the size of the volume.")
                .def_readwrite("velocity", &ParticleField::EmitterParams::velocity, "Initial velocity (m/s).")
                .def_readwrite("speed_spread", &ParticleField::EmitterParams::speedSpread,
                               "Isotropic per-particle perturbation of the velocity (m/s).")
                .def_readwrite("accel", &ParticleField::EmitterParams::accel,
                               "Constant acceleration (m/s^2). Falling snow and rain use ZERO — they "
                               "are at terminal velocity.")
                .def_readwrite("wind", &ParticleField::EmitterParams::wind,
                               "Uniform horizontal drift (m/s), summed with velocity. Usually animated.")
                .def_readwrite("drift_amplitude", &ParticleField::EmitterParams::driftAmplitude, "metres")
                .def_readwrite("drift_frequency", &ParticleField::EmitterParams::driftFrequency,
                               "Hz of the slowest term.")
                .def_readwrite("drift_growth", &ParticleField::EmitterParams::driftGrowth,
                               "0 = constant, 1 = ramps in over the life.")
                .def_readwrite("drift_scale", &ParticleField::EmitterParams::driftScale,
                               "Metres of spatial wavelength (turns wobble into travelling gusts); "
                               "0 = per-slot phase only.")
                .def_readwrite("lifetime", &ParticleField::EmitterParams::lifetime,
                               "Slot repeat period (s). It has to CONTAIN the whole story: fall, land, "
                               "rest/splash, fade.")
                .def_readwrite("lifetime_jitter", &ParticleField::EmitterParams::lifetimeJitter)
                .def_readwrite("duty_cycle", &ParticleField::EmitterParams::dutyCycle,
                               "Alive fraction of the period, (0,1].")
                .def_readwrite("size", &ParticleField::EmitterParams::size,
                               "Per-particle radius in METRES, written into w. Sizes here are metres, "
                               "not pixels — a 2.4 cm proxy centimetres from the lens paints 150 px.")
                .def_readwrite("size_jitter", &ParticleField::EmitterParams::sizeJitter)
                .def_readwrite("follow", &ParticleField::EmitterParams::follow,
                               "Wrap the field toroidally into a lateral box centred on "
                               "set_follow_center() — weather instead of a patch. The wrap PERIOD is "
                               "2 * spawn_half_extent.xz, so author the slab accordingly. Y is never "
                               "wrapped.")
                .def_readwrite("follow_snap", &ParticleField::EmitterParams::followSnap,
                               "Lattice (m) the follow centre is snapped to; 0 = no snapping. Choose an "
                               "integer number of density voxels when the field also carries a "
                               "DensityRepr, or the haze visibly swims.")
                .def_readwrite("surface", &ParticleField::EmitterParams::surface,
                               "Analytic landing (rest / splash) against a baked height map.")
                .def_readwrite("seed", &ParticleField::EmitterParams::seed);

        // ── the field itself ────────────────────────────────────────────────
        cls
                .def_static("create", &ParticleField::create, py::arg("config"),
                            "Create a field at its FINAL capacity (never resized — see Config.capacity). "
                            "Raises ValueError on capacity == 0.")
                .def_property_readonly("config", &ParticleField::config)
                .def_property_readonly("capacity", &ParticleField::capacity)
                .def_property_readonly("live_count", &ParticleField::liveCount)
                .def("set_live_count", &ParticleField::setLiveCount, py::arg("n"),
                     "Park a field with 0 (it stays in the scene and costs one entry, and the emit "
                     "dispatch is skipped) or, on a HostRing field, cap the live prefix.")

                // ── Ownership.Renderer half ─────────────────────────────────
                .def("set_emitter", &ParticleField::setEmitter, py::arg("params"),
                     "Install the closed-form trajectory the device emitter evaluates. Free to call "
                     "every frame (O(1) bytes, published as push constants), so animating the wind "
                     "costs nothing. Raises on a HostRing / Interop field.")
                .def_property_readonly("emitter", &ParticleField::emitter,
                                       "A COPY of the current parameters — mutate it and hand it back "
                                       "to set_emitter().")
                .def("set_emitter_time", &ParticleField::setEmitterTime,
                     py::arg("time_sec"), py::arg("dt_sec"),
                     "Advance the emitter to ABSOLUTE time (not a delta): the trajectory is closed "
                     "form in t, so any t is valid in any order and a capture may seek with no "
                     "warm-up. dt_sec is the interval the motion vectors are taken over — pass the "
                     "frame's own delta, or 0 to freeze the field for a still.")
                .def_property_readonly("emitter_time", &ParticleField::emitterTime)
                .def_property_readonly("emitter_dt", &ParticleField::emitterDt)
                .def("set_follow_center", &ParticleField::setFollowCenter, py::arg("world_center"),
                     "Move the centre of the toroidal follow box — pass the CAMERA's world position "
                     "(weather that follows anything else is not weather). Snapped to "
                     "emitter.follow_snap here; read the snapped point back from follow_center.")
                .def_property_readonly("follow_center", &ParticleField::followCenter,
                                       "The SNAPPED centre — use this, not the raw camera position, to "
                                       "place anything that must agree with the wrap box (above all "
                                       "density_repr.center).")

                // ── representations ─────────────────────────────────────────
                .def("set_mesh_repr",
                     [](ParticleField& f, std::shared_ptr<BufferGeometry> geometry, const py::object& material) {
                         f.setMeshRepr(std::move(geometry), as_material(material));
                     },
                     py::arg("geometry"), py::arg("material"),
                     "Draw every live particle as a lit proxy in the G-buffer — ONE indirect draw. "
                     "`material` also becomes the field's Mesh material, which is what keeps the "
                     "shading in step. Vulkan only.")
                .def_property_readonly("mesh_repr", [](ParticleField& f) { return &f.meshRepr(); },
                                       py::return_value_policy::reference_internal)
                .def("set_billboard_repr",
                     [](ParticleField& f, const Color& hot, const Color& cool, float intensity, float size_scale) {
                         f.setBillboardRepr(hot, cool, intensity, size_scale);
                     },
                     py::arg("hot"), py::arg("cool"), py::arg("intensity") = 1.f, py::arg("size_scale") = 1.f,
                     "Draw every live particle as one camera-facing ADDITIVE quad, composited after "
                     "the upscalers (so it is outside TAA and is NOT exposed by auto-exposure — scale "
                     "intensity with the scene exposure by hand). Vulkan only.")
                .def_property_readonly("billboard_repr", [](ParticleField& f) { return &f.billboardRepr(); },
                                       py::return_value_policy::reference_internal)
                .def("set_density_repr",
                     [](ParticleField& f, const Vector3& center, const Vector3& half_extent,
                        float sigma_per_particle, std::uint32_t resolution) {
                         f.setDensityRepr(center, half_extent, sigma_per_particle, resolution);
                     },
                     py::arg("center"), py::arg("half_extent"), py::arg("sigma_per_particle"),
                     py::arg("resolution") = 128u,
                     "Scatter the field once per frame into a world-anchored extinction volume every "
                     "view's froxel pass then samples — the haze a snowfall or a rain curtain adds. "
                     "`resolution` is LATCHED the frame the volume is allocated. Vulkan only.")
                .def_property_readonly("density_repr", [](ParticleField& f) { return &f.densityRepr(); },
                                       py::return_value_policy::reference_internal)

                // ── Ownership.HostRing half ─────────────────────────────────
                .def("submit",
                     [](ParticleField& f, py::array_t<float, py::array::c_style | py::array::forcecast> data,
                        float dt) {
                         if (data.ndim() != 2 || data.shape(1) != 4)
                             throw std::runtime_error("submit: expected an (n, 4) float32 array of "
                                                      "(x, y, z, w) positions");
                         const auto n = static_cast<std::uint32_t>(data.shape(0));
                         f.submit(data.data(), n, dt);
                     },
                     py::arg("positions"), py::arg("dt") = 0.f,
                     "Point a HostRing field at n positions as an (n, 4) float32 array — xyz plus w, "
                     "which is the radius under WSemantic.Radius and is the DEAD sentinel when "
                     "negative under either. One memcpy, no per-particle loop. n > capacity is "
                     "clamped; also sets the live count. Raises on a Renderer / Interop field. "
                     "`dt` is the step this submit advanced the pool over, read only by the "
                     "velocity stretch under Config.host_stable_slots (0 = assume 1/60 s).")
                .def("set_orientations",
                     [](ParticleField& f, py::array_t<float, py::array::c_style | py::array::forcecast> quats) {
                         if (quats.ndim() != 2 || quats.shape(1) != 4)
                             throw std::runtime_error("set_orientations: expected an (n, 4) float32 array "
                                                      "of (x, y, z, w) quaternions");
                         f.setOrientations(quats.data(), static_cast<std::uint32_t>(quats.shape(0)));
                     },
                     py::arg("quaternions"),
                     "Per-particle orientation as (n, 4) float32 quaternions in (x, y, z, w) order. "
                     "Requires Config.orientations. WRITE-ONCE by contract: the device buffer is not "
                     "ringed, so this is authored with the field, not animated.")
                .def("set_attributes",
                     [](ParticleField& f, py::array_t<float, py::array::c_style | py::array::forcecast> rgba) {
                         if (rgba.ndim() != 2 || rgba.shape(1) != 4)
                             throw std::runtime_error("set_attributes: expected an (n, 4) float32 "
                                                      "array of (r, g, b, a)");
                         f.setAttributes(rgba.data(), static_cast<std::uint32_t>(rgba.shape(0)));
                     },
                     py::arg("rgba"),
                     "Per-particle appearance as (n, 4) float32 (r, g, b, a): rgb is LINEAR HDR "
                     "radiance in the same domain BillboardRepr.color_hot is authored in, a is "
                     "reserved for the phase-2 alpha-over opacity. Requires Config.attributes.\n\n"
                     "WRITE-ONCE by contract, exactly like set_orientations: the device buffer is "
                     "not ringed, so rewriting it while frames are in flight is a host write to "
                     "memory the GPU may be reading. A sim that needs colours EVERY frame wants the "
                     "interop leg — enable_particle_field_interop hands back a second handle for "
                     "this buffer and the foreign kernel writes it device-to-device.\n\n"
                     "THROWS on an Ownership.Interop field that is not in host_fallback().")
                .def_property_readonly("host_fallback", &ParticleField::hostFallback,
                                       "True when an Interop field had to fall back to the host path "
                                       "(this device cannot export memory to a foreign API), which "
                                       "makes submit() legal on it.")
                .def("__repr__", [](const ParticleField& f) {
                    return "<threepp.ParticleField name='" + f.name + "' capacity=" +
                           std::to_string(f.capacity()) + " live=" + std::to_string(f.liveCount()) + ">";
                });
    }

}// namespace threepp_py
