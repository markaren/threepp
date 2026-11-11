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

    // Default of DisplacedMesh::Params::measuredSpreading: false unless the
    // environment sets THREEPP_OCEAN_SPREAD to 1, on or true; 0 is off (read
    // once per process).
    bool defaultMeasuredSpreading();

    // Default of DisplacedMesh::Params::chopNormals: false unless the
    // environment sets THREEPP_OCEAN_CHOP_NORMALS to 1, on or true; 0 is off
    // (read once per process).
    bool defaultChopNormals();

    class DisplacedMesh : public Mesh {

    public:
        struct Params {
            // World-space tile sizes for each cascade (the FFT texture wraps
            // once per tile). 0 disables that cascade; cascade 0 must be > 0.
            float tileSize0 = 40.0f;
            float tileSize1 =  0.0f;
            float tileSize2 =  0.0f;

            // Wind direction and strength (m/s). windTheta is the direction
            // the wind blows TOWARD, in radians in the world XZ plane:
            // (cos windTheta, sin windTheta), 0 = toward +X, pi/2 = toward +Z.
            // With measuredSpreading the waves run that way (downwind); with
            // the symmetric cos^2 only the crest axis follows it, so toward
            // and from drew the same sea. Speed sets the overall amplitude
            // via the Phillips spectrum.
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

            // Hs calibration switch (default OFF, the historical sea). The
            // plain spectrum's constant (A = 0.01, set by eye) and Phillips
            // peak exponent give a significant wave height of only ~0.52x
            // Pierson-Moskowitz (Hs ~ 0.0115 U^2 against 0.021-0.025 U^2;
            // ~0.55x JONSWAP at a finite fetch). ON uses Pierson and
            // Moskowitz's own alpha = 8.1e-3 and beta = 0.74: Hs x1.85 at
            // every wind and fetch, the peak at 1.16x the wavelength (Tp
            // x1.08, onto PM's and Hasselmann's). Every scene tuned its
            // waveScale against the OFF sea. The environment variable
            // THREEPP_OCEAN_HS_FIX = 1 / 0 forces it on / off for every
            // ocean. Live-tunable like windSpeed.
            bool hsFix = true;

            // Choppy Wave Model switch (default OFF, the historical chop).
            // ON draws the surface as the first-order Lagrangian sea of
            // Nouguier, Guerin and Chapron (2009), x = q + D(q) with the full
            // horizontal displacement D, i.e. choppiness 1 whatever the field
            // below says, so crests are as peaked and troughs as broad as
            // linear Lagrangian theory makes them. A per-vertex limiter keeps
            // the Jacobian of the drawn map at or above kCwmMinJacobian by
            // taking the largest choppiness <= 1 that satisfies it there, so
            // the mesh never loops; vertex normals are the displaced
            // surface's own. The environment variable THREEPP_OCEAN_CWM =
            // 1 / 0 forces it on / off for every ocean. Live-tunable.
            bool cwm = false;

            // Second-order bound waves (default OFF). ON adds the narrow-band
            // second-order harmonic of Longuet-Higgins (1963) and Tayfun
            // (1980), eta2 = (k_m / 2)(eta^2 - |R[eta]|^2), to cascades 0 and
            // 1, k_m the band's mean wavenumber and R the Riesz transform,
            // the Hilbert transform's 2D form (Felsberg and Sommer 2001),
            // which is minus the unit-choppiness horizontal displacement the
            // cascade already carries: crests higher and sharper, troughs
            // flatter, height skewness 3 sigma k_m, for waves from any
            // direction. Normals follow. THREEPP_OCEAN_BOUND_WAVES = 1 / 0
            // forces it. Live-tunable.
            bool boundWaves = false;

            // Breaking-crest whitecaps (default OFF, the historical Jacobian
            // foam accumulator). ON draws natural foam only where the current
            // crest is breaking, from the instantaneous surface: the local
            // steepness of the Lagrangian map (the most compressive eigenvalue
            // of the choppiness-1 displacement gradient, a k for a single
            // wave at its crest) above kBreakingSteepness. It rides the crest
            // as it moves, has no texture and no lag, and is lit with the
            // foam albedo Whitlock et al. (1982) measured. Vessel and splat
            // foam are unchanged. THREEPP_OCEAN_BREAKING_CRESTS = 1 / 0
            // forces it. Live-tunable.
            bool breakingCrests = false;

            // Crest foam (default OFF). ON replaces the natural whitecaps
            // with the standard recipe for a spectral sea, kept where the
            // water is: foam is laid down where the surface folds (the
            // breakingCrests measure, from the exact derivative fields of
            // cascades 0 and 1, past kOceanCrestFoamThreshold), into a
            // texture over the water's REST coordinates (the Lagrangian label
            // q of x = q + D(q)), so it stays on the water it was laid on
            // with nothing to carry it, rides the crest that makes it, and
            // fades behind it in seconds, squeezed on crests and stretched in
            // troughs by the water's own orbital motion. What the foam looks
            // like is read off a photograph of sea foam laid on the same
            // coordinates: holes open in it as it thins, then a lace, then
            // threads. With a wake field, what the vessels leave on the
            // water is drawn the same way. Needs chopNormals (the
            // derivative fields are its) and takes the UV channel
            // breakingCrests would use, so it wins when both are on. Works
            // best with shortWaves: the waves that break are short.
            // THREEPP_OCEAN_CREST_FOAM = 1 / 0 forces it. Live-tunable.
            bool crestFoam = true;

            // Wave reflections (default OFF, the historical mirror). The
            // water's mirror ray passes THROUGH any other part of the water
            // it meets and shades whatever lies beyond, usually the seabed:
            // wherever a wave's face cannot see the sky past its neighbour,
            // the reflection is the bottom's, a black spot or a dark slit
            // along a crest, and more of them the steeper the sea. ON shades
            // the wave the ray meets as water: its Fresnel share mirrors on
            // to the sky above it, the rest is the deep body colour, the way
            // every other ray in the renderer already sees water.
            // THREEPP_OCEAN_WAVE_REFLECTIONS = 1 / 0 forces it. Live-tunable.
            bool waveReflections = false;

            // Short waves as geometry (default OFF, the historical bands).
            // The middle cascade is the finest the mesh draws, and it has
            // the texels for waves a few metres long, but the historical
            // split hands everything shorter than the fine cascade's tile
            // (9.3 m on the default ocean) to the fine cascade, which only
            // tilts the shading: the sea has no steep short waves to see,
            // and the 3 to 9 m waves are the fine tile's first few modes, a
            // pattern that repeats every tile. ON moves the hand-off to
            // kShortWaveHandoff of the middle tile (3 m on the default
            // ocean): those waves become geometry, with their chop, and the
            // fine cascade keeps what is shorter, down to six of its texels
            // a wave (historically its roll-off sat near thirty, which left
            // it a third of its slope variance). Read when the renderer
            // first sees the mesh. THREEPP_OCEAN_SHORT_WAVES = 1 / 0 forces it.
            bool shortWaves = true;

            // The measured short-wave spectrum (default OFF, the historical
            // Phillips tail alone). The k^-3 tail carries a constant slope
            // variance per octave and no dependence on the wind, while the
            // real short waves get steeper as the wind rises: at 16 m/s the
            // drawn sea had a quarter of Cox and Munk's (1954) mean square
            // slope. ON adds the short-wave curvature spectrum of Elfouhaily,
            // Chapron, Katsaros and Vandemark (1997), "A unified directional
            // spectrum for long and short wind-driven waves", J. Geophys. Res.
            // 102(C7), eq. 40-44: B_h = (alpha_m / 2)(c_m / c) F_m, its
            // saturation level alpha_m set by the friction velocity, its
            // gravity-capillary peak at k_m = 370 rad/m, c_m = 0.23 m/s. It is
            // faded in from twice to sixteen times the spectral peak's
            // wavenumber, so the long waves and Hs keep their spectrum, and
            // what lies beyond the fine cascade's roll-off and Nyquist (the
            // capillary peak) is not dropped: its slope variance goes to the
            // shade's roughness through the fine cascade's moment chain
            // (needs chopNormals for that part). Read when the renderer first
            // sees the mesh. THREEPP_OCEAN_SHORT_WAVE_SPECTRUM = 1 / 0 forces it.
            bool shortWaveSpectrum = false;

            // Hull contact (default off): the sea meets a vessel's side
            // instead of easing onto her waterline plane over a margin of up
            // to 2 m around her footprint. The blend to the plane keeps to a
            // narrow band (0.2 x her half beam, 0.15 to 0.6 m), so the waves
            // and the wake field's own bow wave (WakeField::rippleHeights)
            // stand at full height against her; inside the footprint the
            // water is still her waterline plane, so none comes up through
            // her. THREEPP_OCEAN_HULL_CONTACT = 1 / 0 forces it.
            bool hullContact = false;

            // Crest sharpness (default 0, off). A LOOK CONTROL, not physics:
            // a sum of linear waves, even drawn as the Lagrangian sea of the
            // Choppy Wave Model, keeps rounded crests, and the sharp, short-
            // crested chop of a real wind sea comes from nonlinear steepening
            // this surface does not model. s > 0 draws the horizontal
            // displacement of cascades 0 and 1 with a choppiness of 1 + s
            // instead of 1, so the crests pinch into ridges and the troughs
            // widen, under the CWM limiter (needs cwm; nothing happens
            // without it): where the Jacobian of the drawn map would fall
            // under kCwmMinJacobian the vertex takes the largest choppiness
            // that keeps it there, so a crest becomes a ridge and never loops.
            // Heights are untouched: the wave height, and what a sensor
            // ranges to at a given rest point, stay as they are; only where a
            // crest is drawn moves. The vertex normals, the crest foam's rest
            // points and sampleHeight's inversion all use the drawn
            // choppiness; the fine cascade keeps choppiness 1.
            // THREEPP_OCEAN_CREST_SHARPNESS = <s> forces it for every ocean.
            // Live-tunable.
            float crestSharpness = 0.f;

            // Fold foam (default OFF: the crest foam deposits on the crest
            // steepness eps of cascades 0 and 1 against a fixed threshold).
            // That threshold does not carry between setups: eps is dominated
            // by the shortest waves in those cascades, and where cascade 1
            // ends depends on its tile (3 m on a 127 m tile, 1.1 m on the
            // Norvasundet twin's 47 m one), so one value dotted the twin's
            // strait at 10 m/s and left the open sea at 16 m/s bare. ON
            // changes two things in foam_crest.comp. The waves that count are
            // the same in metres everywhere: cascade 1 is read through its
            // mipped copies (needs shortWaves) at the level that keeps the
            // waves longer than kFoldFoamLength. And the measure is the drawn
            // surface's own: the Jacobian J = det(I + l A) of the map the
            // mesh draws, l the choppiness the CWM limiter allows (1 + crest
            // sharpness, less where it binds), so foam forms where the drawn
            // crest is about to fold, the more the closer J comes to the
            // limiter's floor (kCwmMinJacobian): the deposit starts under
            // kFoldFoamJacobian and is full kFoldFoamRamp below it.
            // Needs crestFoam. THREEPP_OCEAN_FOLD_FOAM = 1 / 0 forces it;
            // THREEPP_OCEAN_FOLD_FOAM_J / _RAMP / _LENGTH move the constants.
            // Read when the crest foam is first made.
            bool foldFoam = true;

            // Crest-line foam (default OFF: fold foam lands where the
            // measure peaks, and on a short-crested sum of linear waves its
            // peaks are blobs a few metres across, not caps along a crest).
            // ON, foam_crest.comp takes the largest fold measure met along
            // the local crest line (perpendicular to the most compressive
            // direction of grad D) within kCrestLineFoamReach times the
            // fold-foam length either way, so a cap runs along its crest as
            // far as the crest nearly breaks over that length and does not
            // grow across it. Needs foldFoam. THREEPP_OCEAN_CREST_LINE_FOAM
            // = 1 / 0 forces it; THREEPP_OCEAN_CREST_LINE_FOAM_REACH moves
            // the reach. Read when the crest foam is first made.
            bool crestLineFoam = true;
            // Directional spreading of the wave energy. true = the measured
            // distribution of Donelan, Hamilton & Hui (1985) with Banner's
            // (1990) short-wave form: near the spectral peak the energy
            // runs in a narrow fan downwind (long crests travelling with the
            // wind), the short waves spread wider, and a little energy runs
            // against the wind. false (the default) = the historical
            // symmetric cos^2, as much energy upwind as downwind (standing,
            // short-crested slosh). Both carry the same energy per
            // wavelength in the continuum; on the cascades' discrete grids
            // the significant wave height agrees within about 10%, and
            // depends on the wind angle where the spectral peak sits near a
            // tile's fundamental, because the narrow lobe then falls on a
            // handful of modes. Live-tunable like windSpeed: the renderer
            // regenerates the spectra. THREEPP_OCEAN_SPREAD=1 makes true the
            // default.
            bool measuredSpreading = defaultMeasuredSpreading();

            // Optional swell: a second, narrow wave system from a distant
            // storm, added to the wind sea with its own height, period and
            // direction (a JONSWAP peak with gamma = 6 and cos^(2s)(theta/2)
            // spreading; sources in phillips_spectrum.comp). swellHeight is
            // its significant height in metres; 0 (the default) = no swell.
            // It lives on cascade 0 only, so its peak wavelength g Tp^2 / 2 pi
            // must lie between tileSize1 and tileSize0 / 2 (9 to 18 s on the
            // default 1 km ocean, 9.5 to 17.5 s with a margin); outside that
            // it is not drawn and the renderer says so once. swellTheta is
            // the direction it travels toward (radians, as windTheta).
            // swellSpread is s: ~25 for swell from a nearby storm, ~75 from a
            // far one (Goda 2000); values below 1 are taken as 1. Live-
            // tunable: the renderer regenerates the spectra.
            float swellHeight = 0.0f;
            float swellPeriod = 12.0f;
            float swellTheta  = 0.0f;
            float swellSpread = 50.0f;

            // Global Y-displacement multiplier. 1.0 is physical; higher values
            // exaggerate wave height.
            float waveScale = 1.0f;

            // Horizontal-displacement multiplier ("choppiness"). 0 is a pure
            // height-field, 1 is full Tessendorf horizontal pull (sharp crests).
            // Real ocean falls around 0.4 — anything ≥ 0.8 starts producing
            // wave-folding artefacts (white spike crests).
            float choppiness = 0.45f;

            // How much of the sea's own whitecaps (the crest foam) is laid
            // down, live-tunable: 1 = all of it, 0 = none. Wake trails and
            // explicit foam disturbances are NOT scaled by this. Ocean::create
            // derives ~size/300 (capped at 1), so small water stays calm.
            float foamAmount = 1.0f;

            // Chop normals: shade the surface the choppy displacement actually
            // draws. Exact slope and displacement-gradient fields (more inverse
            // FFTs of the same spectrum) give the vertex normal as the cross
            // product of the displaced surface's tangents, and the fine
            // cascade's per-pixel slope comes from a mip chain of slope moments
            // (mean tilts the normal, the variance a pixel cannot hold goes to
            // roughness), read as two placements of the cascade, turned,
            // scaled and shifted against each other and summed, so its tile's
            // period does not show from above. Near a fold the chop correction fades back to the
            // rest slope and its remainder goes to roughness. false = the
            // height-only finite-difference normals, the renderer's original
            // path bit for bit: the feature's images, pipelines and shade
            // branch are built only the first time a mesh turns it on.
            // Live-tunable. THREEPP_OCEAN_CHOP_NORMALS=1 makes true the default.
            bool chopNormals = defaultChopNormals();

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

        // The wave-model switches as the renderer applies them: the Params
        // field, unless its environment variable (THREEPP_OCEAN_<SWITCH> =
        // 1 / 0, read once, named with the field) forces it for every ocean.
        [[nodiscard]] bool cwmOn() const;
        [[nodiscard]] bool boundWavesOn() const;
        [[nodiscard]] bool breakingCrestsOn() const;
        // Crest foam as asked for; the renderer draws it only while the chop
        // normals are on too (see Params::crestFoam).
        [[nodiscard]] bool crestFoamOn() const;
        [[nodiscard]] bool waveReflectionsOn() const;
        [[nodiscard]] bool shortWavesOn() const;
        [[nodiscard]] bool shortWaveSpectrumOn() const;
        [[nodiscard]] bool hullContactOn() const;
        // The horizontal-displacement scale the surface is drawn with: 1 under
        // the Choppy Wave Model, else params.choppiness.
        [[nodiscard]] float effectiveChoppiness() const;
        // Params::crestSharpness as drawn: its environment override, else the
        // field, held to [0, 4]; 0 without the Choppy Wave Model (whose
        // limiter it needs).
        [[nodiscard]] float effectiveCrestSharpness() const;
        [[nodiscard]] bool foldFoamOn() const;
        [[nodiscard]] bool crestLineFoamOn() const;
        // Params::foldFoam's look constants (see there; env overrides).
        // 0.78 puts Monahan and O'Muircheartaigh's (1980) whitecap cover
        // (4.4 %) on the open sea at 16 m/s and none on the Norvasundet
        // strait at 10 m/s, crest sharpness 0; at sharpness 1 the same holds
        // near 0.53 (ocean_metrics' crest-foam row; the twin from the drone).
        static constexpr float kFoldFoamJacobian = 0.78f;
        static constexpr float kFoldFoamRamp     = 0.2f;
        static constexpr float kFoldFoamLength   = 3.0f;// metres
        // Params::crestLineFoam's reach along a crest, either way, in units
        // of kFoldFoamLength (the shortest wave the fold foam counts).
        static constexpr float kCrestLineFoamReach = 1.0f;
        // ... and the J it starts under (replaces kFoldFoamJacobian while it
        // is on): 0.76 puts 4.7 % cover on the open sea at 16 m/s and none
        // at 10 m/s over 1.5 km of fetch, crest sharpness 0 (ocean_metrics).
        static constexpr float kCrestLineFoamJacobian = 0.76f;

        // The CWM limiter's floor on the Jacobian of the drawn map.
        static constexpr float kCwmMinJacobian = 0.1f;
        // The breaking threshold on the local steepness of the choppiness-1
        // map (cascades 0 and 1, differenced at the mesh's grid spacing).
        // Mirrors kOceanBreakingSteepness (vulkan_shared.h), where its
        // calibration is written down.
        static constexpr float kBreakingSteepness = 0.15f;
        // Params::shortWaves: the cascade 1 -> 2 hand-off wavelength as a
        // fraction of the middle cascade's tile (3 m of 127 m), never under
        // five of its texels.
        static constexpr float kShortWaveHandoff = 3.0f / 127.0f;

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
            // Its bow bump, its V-wedge and the V-wedge's ridge foam. false =
            // a scene whose wake field makes her waves (VesselWash with
            // ripples): the analytic ones would come on top of them. The
            // stern foam trail stays either way.
            bool  analyticWaves = true;
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
        // numbers (python/examples/rigs/usv_rig.py does, per kind of producer).
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
        // THREEPP_WAKE_HEIGHTS from the environment, read once: "0" = false,
        // anything else or unset = true.
        static bool defaultRippleHeights();
        // THREEPP_WAKE_BREAKING, the same way, for WakeField::rippleBreaking.
        static bool defaultRippleBreaking();
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
            // The ripples' HEIGHT is part of the ocean's geometry (water_displace
            // adds it, so the ray-traced surface, lidar and radar see the waves
            // the shade draws), and the waves that reach the edge of a ripple
            // window are taken up by an absorbing layer there instead of every
            // wave being made to die before it can wrap round the window, so a
            // wake keeps its transverse waves. false = slopes only, every wave
            // decaying at the window's crossing rate (the model before
            // 2026-11). That decay leaves about a fifth of a wave to come
            // back in at the opposite edge, so a fast vessel meets her own
            // wake ahead of her and on both sides. On by default; the
            // environment's THREEPP_WAKE_HEIGHTS=0 (read once) or this field
            // turns it off. Live.
            bool  rippleHeights   = defaultRippleHeights();
            // The ripples' own waves BREAK where they are steep, and the
            // broken water is foam and air in the field, which then carries,
            // tears and thins it like any other: so a hull's bow wave is white
            // on its crest where it is steep enough and nowhere else, and the
            // white leaves her along the crest it broke on. A crest of slope
            // |grad eta| starts to spill at rippleBreakSlope[0] and is fully
            // broken at [1] (a Stokes wave's limit is about 0.44; ship bow
            // waves are seen to break from about 0.3). Off by default until it
            // is judged: THREEPP_WAKE_BREAKING=1 or this field turns it on.
            // The amounts are LOOK constants, held under a closed film so the
            // water shows through.
            bool  rippleBreaking      = defaultRippleBreaking();
            float rippleBreakSlope[2] = {0.25f, 0.45f};
            float rippleBreakFoam     = 0.30f;
            float rippleBreakAir      = 0.20f;
            // Scales the light the film and the bubbles turn back (their albedo);
            // 1 = as the water shade has them. A scene whose camera opens wide
            // at night turns it down: under a moon, or the night sky's own light,
            // a race is otherwise a pale band on a sea that mirrors a dark sky.
            float albedo          = 1.0f;
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
            // The push's line ends gently: it is the line above (half length
            // a) convolved with one of half length t = taper x a along it, a
            // trapezoid, full where |s| < a - t and nothing past a + t.
            // 0 = the line ends as a step, rounded only by the gaussian.
            float taper      = 0.f;
            // The push's depth: a hull carries her weight on the water down
            // to her draft, not on its surface, and a load at depth z makes
            // waves of wavenumber k weaker by e^(-kz) (Havelock). Taken as
            // spread evenly over 0..draft, the push's waves are weakened by
            // (1 - e^(-k draft)) / (k draft). 0 = a load on the surface (the
            // waves of a pressure patch).
            float draft      = 0.f;
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

        // The surface the RAYS see: the renderer builds and refits the mesh's
        // acceleration structure over every s-th grid line, s chosen so the
        // traced grid has at most this many cells a side (the raster always
        // draws the full grid; the per-vertex normals are shared). Reflection
        // and refraction rays, GI and shadow tests then meet the water at that
        // coarser spacing. 0 = trace the full grid. Read when the renderer
        // first sees the mesh. The per-frame refit of a 1024 x 1024 grid
        // (2.09 M triangles) cost 5.0 ms on an RTX 4070 against 0.36 ms for
        // 512 x 512: the cost is not linear in the triangle count.
        uint32_t rtProxyCells = 512;

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
            // The band's mean wavenumber (rad/m) the bound harmonic uses;
            // 0 = no bound harmonic on this cascade.
            float    boundKm  = 0.f;
            // Params::shortWaveSpectrum, the finest cascade only: the slope
            // variance (both axes) of the short-wave spectrum this grid does
            // not hold, which the shade adds to its roughness. 0 otherwise.
            float    bankedMss = 0.f;
        };
        mutable CascadeField heightFields[3];

        // The site's wave climate (default none: one sea everywhere, as
        // before). A small world-anchored map, baked by the caller from the
        // site's land, seabed and wind, whose channel c (0, 1, 2) scales
        // cascade c's height and horizontal displacement at a water parcel's
        // rest point; channel 3 is the caller's own (e.g. the depth) and is
        // not read. So the lee of land can be calm while the open reach keeps
        // its sea, and waves can die out on a beach: water_displace.comp
        // scales the cascades' fields per vertex, the shade the fine
        // cascade's per-pixel slope (by s, its variance by s^2), the crest
        // foam forms at the scaled sea, and sampleHeight follows the drawn
        // surface. The map varies slowly, so its own gradient is left out of
        // the normals. Texel (i, j) is the value at the centre of the cell
        // [originX + i sizeX / width, ...) x [originZ + j sizeZ / height, ...),
        // in the ocean's own XZ (the frame of its rest points); read
        // bilinearly between centres, clamped to the edge texels outside.
        // Row-major, x fastest: data[(j * width + i) * 4 + c].
        // The renderer uploads it when the map changes; the crest foam's
        // levels (water::CrestFoam) are made when the crest foam is first
        // made, so a map set after that scales the sea but not the foam.
        struct WaveClimate {
            std::vector<float> data;
            uint32_t width   = 0;
            uint32_t height  = 0;
            float    originX = 0.f;
            float    originZ = 0.f;
            float    sizeX   = 0.f;
            float    sizeZ   = 0.f;
            // Bumped by every set / clear: the renderer re-uploads on a change.
            uint64_t version = 0;
        };
        // Set the map (width x height texels of 4 floats, as WaveClimate
        // describes; width at least 2). Throws std::invalid_argument on a
        // size mismatch or an empty extent.
        void setWaveClimate(std::vector<float> data, uint32_t width, uint32_t height,
                            float originX, float originZ, float sizeX, float sizeZ);
        // Back to one sea everywhere.
        void clearWaveClimate();
        [[nodiscard]] const WaveClimate& waveClimate() const { return waveClimate_; }
        [[nodiscard]] bool hasWaveClimate() const { return waveClimate_.width > 0u; }
        // The map's three cascade gains at a rest point (1, 1, 1 without a
        // map): the same bilinear read the shaders make.
        [[nodiscard]] std::array<float, 3> waveClimateGain(float x, float z) const;

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

    private:
        WaveClimate waveClimate_;
    };

}// namespace threepp

#endif//THREEPP_DISPLACEDMESH_HPP
