// A mesh whose vertex positions are GPU-displaced every frame from
// height/displacement textures. Designed for use with the Vulkan path
// tracer: the renderer recognises the type, runs the displacement
// compute pass, and rebuilds the BLAS in-place every frame.
//
// Typical usage (see examples/projects/Ocean):
//
//   auto plane = PlaneGeometry::create(40.f, 40.f, 255, 255);
//   plane->rotateX(-math::PI / 2.f);
//   auto mat   = MeshPhysicalMaterial::create();
//   mat->setIor(1.33f);
//   mat->transmission = 1.f;
//   mat->roughness    = 0.05f;
//   auto ocean = DisplacedMesh::create(plane, mat);
//   scene->add(ocean);
//
// The renderer looks up the FFT cascade textures by (currently) a single
// known channel — this header exposes only the public-facing knobs;
// renderer-side state lives in VulkanRenderer.cpp's DisplacedMeshState.

#ifndef THREEPP_DISPLACEDMESH_HPP
#define THREEPP_DISPLACEDMESH_HPP

#include "threepp/objects/Mesh.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace threepp {

    class DisplacedMesh : public Mesh {

    public:
        struct Params {
            // World-space tile sizes for each cascade (the FFT texture wraps
            // once per tile). 0 disables that cascade; cascade 0 must be > 0.
            float tileSize0 = 40.0f;
            float tileSize1 =  0.0f;
            float tileSize2 =  0.0f;

            // Wind direction (radians, 0 = +X) and strength (m/s). Affects
            // wave anisotropy and overall amplitude via the Phillips spectrum.
            float windTheta = 0.0f;
            float windSpeed = 12.0f;

            // Fetch: metres of open water the wind has blown across. 0 (the
            // default) = a FULLY DEVELOPED sea — the plain Phillips /
            // Pierson-Moskowitz spectrum, whose peak sits at ~8 V²/g (80 m
            // at 9.5 m/s, 260 m at 17) with most of the height in that
            // swell. A finite fetch gives the JONSWAP young sea a coastal
            // vessel actually sails in: the peak moves to shorter waves
            // (∝ fetch^(2/3)), the short-wave tail gains energy
            // (∝ fetch^-0.22) and the peak sharpens (γ → 3.3). 20–40 km
            // puts a 9.5 m/s sea's peak at 20–35 m with ~1 m significant
            // height instead of 80 m / 2 m — the same wind, a far busier
            // surface at the scale of a 10 m hull. Saturates at the fully
            // developed fetch (~1600 V² m, i.e. 145 km at 9.5 m/s). Live-
            // tunable like windSpeed: the renderer regenerates the spectra.
            float fetch = 0.0f;

            // Global Y-displacement multiplier. 1.0 is physical; higher values
            // exaggerate wave height.
            float waveScale = 1.0f;

            // Horizontal-displacement multiplier ("choppiness"). 0 is a pure
            // height-field, 1 is full Tessendorf horizontal pull (sharp crests).
            // Real ocean falls around 0.4 — anything ≥ 0.8 starts producing
            // wave-folding artefacts (white spike crests).
            float choppiness = 0.45f;

            // Natural whitecap (Jacobian-fold) foam scale, live-tunable.
            // 1 = full ocean whitewater, 0 = none. Wake trails and explicit
            // foam disturbances are NOT scaled by this. The Tessendorf fold
            // measure is scale-free, so without this a pond's cm-ripples foam
            // like a gale — Ocean::create derives ~size/300 (capped at 1) so
            // small water reads calm by default.
            float foamAmount = 1.0f;

            // FFT texture resolution per cascade (must be power of two).
            // Band-passed cascades need only enough texels to resolve their
            // own wavelength band (~10 samples per shortest wavelength) — the
            // renderer's band-pass caps cascade 0 at λ ≥ tileSize1 and
            // cascade 1 at λ ≥ tileSize2, so running them at cascade-2-like
            // resolutions is pure waste. Typical trio for 1 km / 127 m /
            // 9.3 m tiles: {128, 256, 512} (Ocean::create derives exactly
            // this from its tile sizes).
            uint32_t textureSize0 = 256;
            uint32_t textureSize1 = 256;
            uint32_t textureSize2 = 256;
        };

        Params params;

        // Hull exclusion zone: suppresses wave displacement inside a
        // world-space oriented rectangle so the ocean doesn't clip through
        // a vessel's deck. Set each frame before render().
        //
        // Inside the (plan-form tapered) footprint the surface is pulled onto
        // the VESSEL'S WATERLINE PLANE and fades back into the wave field over
        // ~2 m outside the hull edge — i.e. the hull really displaces the
        // water instead of the sea passing through it. The plane is
        //
        //     y(localX, localZ) = centerY + tan(pitch)·localZ + tan(roll)·localX
        //
        // with localZ the length axis (positive toward the BOW, along
        // (sinYaw, cosYaw)) and localX the beam axis (positive to STARBOARD,
        // along (cosYaw, -sinYaw)) — the same vessel-local frame the wake
        // formulas use. So:
        //   centerY  world height of the hull's own design-waterline plane at
        //            (centerX, centerZ); the height the water should meet the
        //            hull at, NOT the deck.
        //   pitch    radians, POSITIVE = bow up   (the plane rises toward +localZ)
        //   roll     radians, POSITIVE = starboard up (rises toward +localX)
        // Both angles are clamped to ±1 rad in the shader.
        //
        // Defaults (centerY = pitch = roll = 0) reproduce the historical
        // behaviour exactly: the patch flattens onto the ocean rest plane.
        struct HullExclusion {
            float centerX    = 0.f;
            float centerZ    = 0.f;
            float halfLength = 0.f;   // 0 = disabled
            float halfBeam   = 0.f;
            float sinYaw     = 0.f;
            float cosYaw     = 1.f;
            float centerY    = 0.f;   // world y of the hull's waterline plane
            float pitch      = 0.f;   // rad, +bow up
            float roll       = 0.f;   // rad, +starboard up
        };
        HullExclusion hullExclusion;

        // Adaptive resolution: instead of laying vertices on a uniform
        // grid, the compute shader remaps each (u,v) ∈ [-1,1]² through
        // a cubic blend before scaling so density packs toward the
        // `center`. Topology stays fixed (same triangle indices, same
        // vertex count, same BLAS shape) — only positions move, so the
        // ocean's BLAS-rebuild + per-vertex history paths are unchanged.
        // Pair with the world-space foam texture so foam doesn't drag
        // along with vertex indices as the warp follows the camera.
        //
        // Mapping per axis: `x = halfRange * (coefA·u + (1-coefA)·u³)`
        //   - coefA = 1: uniform (no warp).
        //   - coefA = 0: pure cubic; vertices cluster tightly at the
        //     centre but the edge spacing balloons by 3× the average.
        //   - coefA ≈ 0.1: ~10 cm centre / ~2.7 m edge at 512² verts
        //     over a 1 km tile, with a ~28× density ratio.
        //
        // Set `halfRange = 0` to disable the warp (default).
        struct MeshWarp {
            float centerX   = 0.f;
            float centerZ   = 0.f;
            float halfRange = 0.f;   // 0 = disabled (uniform grid)
            float coefA     = 1.f;   // 1 = linear (no warp); lower = denser centre
        };
        MeshWarp warp;

        // Vessel wake: per-frame state used by water_displace.comp to inject a
        // Kelvin V-wake (geometric height), a bow bump, and a foam trail behind
        // the vessel. Reuses the HullExclusion pose (centerX/Z, sin/cosYaw,
        // halfLength, halfBeam) for the immediate-pose wake formulas; the
        // historical trail uses snapshot samples for the Kelvin V-wake so the
        // wake actually traces the boat's past path through turns instead of
        // snapping to the current heading.
        //
        // Sample emission cadence + trail length are owned by the application
        // (see examples/vulkan/vulkan_ocean.cpp). Compile-time cap matches
        // the renderer's allocated trail SSBO; overflow is dropped silently.
        struct WakeSample {
            float worldX   = 0.f;
            float worldZ   = 0.f;
            float sinYaw   = 0.f;
            float cosYaw   = 1.f;
            float speed    = 0.f;
            float age      = 0.f;       // seconds since emission
            float _pad0    = 0.f;
            float _pad1    = 0.f;
        };
        struct VesselWake {
            float forwardSpeed = 0.f;   // m/s along +heading; 0 disables wake
            bool  enabled      = true;
            std::vector<WakeSample> trail;
        };
        VesselWake wake;

        // Several vessels on one ocean. Vessel 0 is `hullExclusion` + `wake`
        // above, the pair every single-vessel scene sets; vessels
        // 1..kMaxVessels-1 live in `extraVessels`. Each is a footprint on its
        // own waterline plane plus its own wake (bow bump, V-wedge over its
        // own trail, foam trail), and takes part while its hull.halfLength
        // > 0. vesselHull(i) / vesselWake(i) address them all by index.
        // Every footprint also flattens the OTHER vessels' wakes, so a ship's
        // V-wedge does not rise through a smaller boat's deck.
        static constexpr uint32_t kMaxVessels = 8;
        struct Vessel {
            HullExclusion hull;
            VesselWake    wake;
        };
        std::array<Vessel, kMaxVessels - 1> extraVessels;

        HullExclusion& vesselHull(uint32_t i) {
            return i == 0 ? hullExclusion : extraVessels.at(i - 1).hull;
        }
        [[nodiscard]] const HullExclusion& vesselHull(uint32_t i) const {
            return i == 0 ? hullExclusion : extraVessels.at(i - 1).hull;
        }
        VesselWake& vesselWake(uint32_t i) {
            return i == 0 ? wake : extraVessels.at(i - 1).wake;
        }
        [[nodiscard]] const VesselWake& vesselWake(uint32_t i) const {
            return i == 0 ? wake : extraVessels.at(i - 1).wake;
        }

        // Per-frame point sources of foam — splatted by water_displace.comp
        // into the per-vertex foam buffer with a gaussian falloff, persisted
        // via the existing decay (~1.4 s half-life). Use for boat-hull
        // waterline contacts, propeller wash, dynamic splashes — anything
        // not captured by the analytical hull/wake. Clear and repopulate
        // each frame before render(); the renderer uploads on demand.
        //
        // Capacity is bounded by a fixed compile-time max (currently 64);
        // overflow is dropped silently. Tile-local — worldXZ should be in
        // the same frame as the ocean mesh (typically world space).
        struct FoamDisturbance {
            float worldX = 0.f;
            float worldZ = 0.f;
            float radius = 1.0f;    // metres
            float intensity = 1.0f; // [0,1] foam value at centre
        };
        std::vector<FoamDisturbance> foamDisturbances;

        void clearFoamDisturbances() { foamDisturbances.clear(); }
        void addFoamDisturbance(float worldX, float worldZ,
                                float radius, float intensity) {
            foamDisturbances.push_back({worldX, worldZ, radius, intensity});
        }

        // ── Wake field ────────────────────────────────────────────────────
        // What something moving through the water leaves ON it: a propeller's
        // race, a waterjet's wash, a bow thruster, a paddle's puddle, a
        // hull's stern. The foam accumulator above cannot carry these: it is
        // ONE cascade-0 tile repeated over the sheet (a boat's foam comes back
        // every tileSize0) at about a metre a texel.
        //
        // The wake field is a set of PATCHES: square windows of simulated
        // water surface, each `resolution` texels a side, placed in the world
        // by the application (usually trailing one vessel) and moved as it
        // likes. A patch is world-anchored and does not repeat: what is
        // deposited stays where it was put while the patch slides over it
        // (the origin snaps to whole texels, so nothing is resampled), and
        // what the patch leaves behind is dropped. Each texel carries
        //
        //   foam        the white film on the surface                 (0..1)
        //   aeration    bubbles under the surface, the pale water     (0..1)
        //   turbulence  rms turbulent speed of the water (m/s): it stirs
        //               everything here and ruffles the surface
        //   lane        how far the short wind waves are damped       (0..1)
        //   velocity    the water's own mean flow (m/s, world x/z)
        //
        // and every frame the field is carried by its own velocity plus a
        // turbulent flow scaled by `turbulence` (so a deposit is drawn out
        // into streaks and eddies by motion; no structure is painted), spread
        // by an eddy diffusivity, and decayed at each quantity's own rate.
        //
        // What makes a mark is a SOURCE: a capsule on the surface (where the
        // producer was last frame to where it is now, so a coarse time step
        // leaves no gaps), a radius, the water velocity it imparts and how
        // much of each quantity it puts in. Nothing here knows what a
        // propeller is: the caller turns thrust, depth and speed into these
        // numbers (python/examples/usv_rig.py does, per kind of producer).
        // Clear and repopulate the sources each frame before render().
        //
        // RIPPLES. A patch can also carry the waves its producers make: the
        // fan a hull throws, the rings round a paddle. They are not painted
        // and not an analytic V: each patch holds the spectrum of a small
        // linear sea (`rippleResolution` squared wavenumbers over the patch's
        // side), every wavenumber turning at its own deep-water rate
        // (omega^2 = g k), so short waves travel slower than long ones and a
        // moving source leaves the Kelvin pattern for its speed, whatever
        // that is, bent where it turned. What forces it is a source's `push`:
        // the force (N) it bears down on the water with, over its radius (a
        // floating hull bears down with its weight; a blade with its pull).
        // Only the waves are shown: the still dent under a weight at rest is
        // taken out. The shade adds their slope to the water's normal, and
        // what a pixel is too coarse to hold goes to the roughness.
        // The small sea is periodic over its side, so it is shown through a
        // window of that side and its waves are made to die before they have
        // come round: at least 1.5 x speed / side a second for the fastest of
        // the patch's pushing sources. That side need not be the patch's: a
        // patch can name a smaller window for its ripples (WakePatch::
        // rippleSize), which is a finer sea for the same resolution, as a
        // small slow boat's short waves need.
        //
        // Off by default (resolution 0): no memory, no pass, and the water
        // shade is byte-identical. `resolution` and `patches` are LATCHED when
        // the renderer first sees the mesh; the other knobs are live.
        static constexpr uint32_t kMaxWakePatches = 8;
        static constexpr uint32_t kMaxWakeSources = 256;
        struct WakeField {
            uint32_t resolution = 0;              // texels a side per patch; 0 = off (latched)
            uint32_t patches    = kMaxWakePatches;// layers allocated (latched)
            // e-folding times (s). Foam's is for a thick film; a thin one
            // goes `foamThinning` times faster, so a patch dies from its
            // edges and breaks up instead of dimming as a whole.
            float foamLife       = 5.0f;
            float foamThinning   = 3.0f;
            // Bubbles a propeller takes down are back at the surface in
            // seconds; the few that are small enough to stay are not seen.
            float aerationLife   = 4.5f;
            float turbulenceLife = 5.0f;
            float laneLife       = 40.0f;
            float velocityLife   = 2.5f;
            // Large eddies of the turbulent flow, in texels, for a patch that
            // does not name its own size (WakePatch::eddy).
            float eddyTexels     = 12.0f;
            // Eddy diffusivity K = spread * turbulence * eddy size (m2/s).
            float spread         = 0.03f;
            // Ripples: wavenumbers a side of each patch's small sea (a power
            // of two, 128..1024); 0 = none (latched with resolution).
            uint32_t rippleResolution = 0;
            float rippleLife      = 14.0f;// e-folding time (s) of a wave, where the patch's side allows it
            float rippleViscosity = 2e-4f;// m2/s: a wave of wavenumber k also dies at 2 nu k^2 (the shortest first)
            float rippleGain      = 1.0f; // on the slope the shade reads; 1 = what linear theory gives
        };
        WakeField wakeField;

        struct WakePatch {
            float centerX = 0.f;
            float centerZ = 0.f;
            float size    = 0.f;// side (m); 0 = this patch is off
            // Size (m) of the large eddies of the turbulent flow in this patch:
            // about the width of what stirs it (a race, a hull). 0 = the
            // field's eddyTexels texels.
            float eddy    = 0.f;
            // Its ripples' own window: side (m) and centre. 0 = the patch's.
            float rippleSize    = 0.f;
            float rippleCenterX = 0.f;
            float rippleCenterZ = 0.f;
        };
        std::array<WakePatch, kMaxWakePatches> wakePatches;

        // 64 bytes, memcpy'd to the GPU (mirrors WakeSource in ocean_wake.glsl).
        struct WakeSource {
            float x0 = 0.f, z0 = 0.f;// where the producer was at the last frame (world)
            float x1 = 0.f, z1 = 0.f;// where it is now
            float vx = 0.f, vz = 0.f;// water velocity it imparts (m/s, world, over the ground)
            float radius     = 0.3f; // m, gaussian
            float foam       = 0.f;
            float aeration   = 0.f;
            float turbulence = 0.f;  // m/s
            float lane       = 0.f;
            // Ripples: the force (N) it bears down on the water with, spread
            // as a gaussian of `radius` drawn out to a line that reaches
            // (ax, az) to either side of the point (a hull's half length along
            // her heading; 0, 0 = round). A source with only a push leaves no
            // foam, air, turbulence or lane.
            float push       = 0.f;
            float ax = 0.f, az = 0.f;
            // The patch whose ripples it forces; -1 = the first that holds it.
            // (The other quantities go into every patch the source lies in.)
            int32_t patch    = -1;
            float _pad       = 0.f;
        };
        std::vector<WakeSource> wakeSources;

        void clearWakeSources() { wakeSources.clear(); }
        void addWakeSource(const WakeSource& s) { wakeSources.push_back(s); }

        DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                      const std::shared_ptr<Material>& material);

        [[nodiscard]] const std::string& type() const override;

        static std::shared_ptr<DisplacedMesh> create(
                const std::shared_ptr<BufferGeometry>& geometry,
                const std::shared_ptr<Material>& material) {
            return std::make_shared<DisplacedMesh>(geometry, material);
        }

        // Bumped each render() so the renderer's dirty-detect always treats
        // this mesh as moved (it is — FFT textures advance every frame).
        // Internal; do not mutate from user code.
        uint64_t frameTick = 0;

        // Grid topology hint for the renderer's displacement pass, which
        // reconstructs rest positions from vertex index (the displaced buffer
        // is rewritten in place, so rest can't be read back). 0 = derive a
        // SQUARE grid from sqrt(vertexCount) — the historical default.
        // Ocean::create always fills these; hand-built rectangular grids
        // (gridWidth ≠ gridDepth) must set them or state creation fails.
        uint32_t gridWidth = 0;// vertices along local X
        uint32_t gridDepth = 0;// vertices along local Z

        // Sticky opt-in for the CPU height mirror. Set by sampleHeight() on
        // first use; the Vulkan renderer skips the per-frame GPU→host cascade
        // copies (and the mirror memcpy) entirely until then, so scenes that
        // never query wave height pay nothing for it. Internal; do not mutate
        // from user code.
        mutable bool wantsHeightReadback = false;

        // Renderer-filled spatial-domain wave fields, one per cascade,
        // copied back from GPU each frame after the IFFT pass (only while
        // wantsHeightReadback is set). Both are row-major RG32F with cell
        // (ix, iz) at index `(iz*dim + ix)*2`:
        //   data — R = vertical displacement, G = unused
        //   disp — R = horizontal displacement x, G = horizontal displacement z
        //          (in the cascade's sample domain; cascade 1 is rotated)
        // Values are unnormalized IFFT output: scale by 1/tileSize.
        // mutable so const-method `sampleHeight` can be called on a const
        // DisplacedMesh while the renderer keeps the data fresh.
        struct CascadeField {
            std::vector<float> data;
            std::vector<float> disp;
            uint32_t dim      = 0;
            float    tileSize = 0.f;
        };
        mutable CascadeField heightFields[3];

        // Combined wave height of the rendered surface over (worldX, worldZ).
        // Mirrors water_displace.comp's sampleDisplacement(): cascades 0 and 1
        // are B-spline filtered, cascade 2 bilinear, and each contributes
        // height * (1/tileSize) * waveScale.
        //
        // The wave field is defined over rest (undisplaced) positions: a
        // surface point with rest position q is drawn at q + D(q), where D is
        // the horizontal displacement scaled by `choppiness`. The height over
        // a world XZ is therefore H(q) for the q whose displaced image is that
        // XZ, not H(worldXZ). sampleHeight solves x = q + D(q) by fixed-point
        // iteration (the same inversion foam_world.comp does). Without it the
        // result is off by the local chop displacement, up to metres in a
        // steep sea. `cascadeMask` applies to both D and H, so a masked query
        // returns the height of the surface built from those cascades alone.
        //
        // `cascadeMask` is a bitmask (bit i selects cascade i); the default
        // 0b111 sums all three. Use a narrower mask for hull-scale buoyancy
        // queries on long vessels — sampling the fine cascade at the bow
        // and stern of a 28 m hull picks up 4-8 m chop that wouldn't
        // physically induce pitch on a hull that long ("car on rocks"
        // feel). Masking out cascade 2 for the buoyancy sample (visual
        // remains unchanged — GPU still renders all cascades) restores
        // the integrating behaviour of a long hull.
        //
        // Latency and determinism (Vulkan): the value comes from a CPU
        // mirror of the GPU height field, refreshed inside render() right
        // after the frame's fence wait from a per-frame-in-flight readback
        // ring — so it is the field as of a FIXED number of renderer frames
        // ago (kFramesInFlight = 2; the synchronous first build of a new
        // mesh is current), never "whatever the GPU had finished copying
        // when the host looked". The same pinned renderer sim-time sequence
        // therefore gives the same heights flat out and with sleeps between
        // frames. (Until 2026-08 it was a memcpy out of one buffer the
        // in-flight frames were still copying into: 1-2 frames old and
        // torn, depending on GPU pacing.)
        float sampleHeight(float worldX, float worldZ,
                           uint32_t cascadeMask = 0b111u) const;

        // CPU mirror of the wake-height block in water_displace.comp —
        // bow bump (B), bow V-wedge (D), and the bow-anchored Kelvin
        // V-wake (C, summed over `wake.trail`). Mirrors the shader
        // formulas closely enough that buoys spring-damping against
        // (sampleHeight + sampleWakeHeight) bob through a passing
        // V-wake the way the rendered mesh does. Returns 0 if there is
        // no active vessel or the speed gate hasn't engaged.
        //
        // With several vessels it is the SUM of their wakes, each faded
        // only inside its own footprint: the water a hull rides on, not the
        // rendered surface (which another footprint flattens). So a boat
        // crossing a ship's wake that samples this under her own hull feels
        // the ship's wake.
        float sampleWakeHeight(float worldX, float worldZ) const;
    };

}// namespace threepp

#endif//THREEPP_DISPLACEDMESH_HPP
