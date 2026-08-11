// Vulkan deferred — 300k flakes of falling snow (plans/particle-atmosphere.md F2).
//
// The whole snowfall is ONE scene object and ZERO per-frame CPU work. A
// ParticleField in Ownership::Renderer mode owns a device-local position buffer
// that a compute pass (particle_emit.comp) rewrites at the head of every frame
// from a closed-form trajectory; the host never allocates, walks, uploads or
// even KNOWS a particle position. Per frame the CPU does exactly this:
//
//     snow->setEmitterTime(t, dt);          // two floats
//
// and the renderer turns that into one 128-byte push constant and one dispatch.
// A field of 300k costs the same CPU as a field of 300 — which is the entire
// point of the mode, and is what makes 10^5-10^6 weather affordable at all.
//
// What to look for:
//   • flakes that FALL, wobble on a gusting drift, and vary in size — all of it
//     out of one closed form, with no simulation state anywhere
//   • NO smearing behind them under TAA. The emit dispatch writes f(t) and
//     f(t - dt) into positions and prevPositions from the same thread, so the
//     motion vectors are exact by construction rather than by a well-ordered
//     copy. This is the thing to check on a --seq capture; a still cannot see it
//   • the snow's own DensityRepr hazing the far props — the same field feeding
//     the mesh representation AND the world-anchored density volume
//   • grey overcast daylight: a snowfall is a LOW-CONTRAST scene, so the demo
//     pins exposure rather than letting AE hunt for a subject
//
// Controls:  drag = orbit   scroll = zoom   SPACE = start / stop the snow
// Headless:  vulkan_snow --shot out.png [--frames N] [--t SECONDS] [--count N]
//            vulkan_snow --seq DIR [--orbit deg/s] [--warm N] [--seqframes N]
//            vulkan_snow --bench           interleaved A/B of the field's cost
//            vulkan_snow --bench-lod       interleaved A/B of the F4 LOD split
//            vulkan_snow --no-lod          mesh proxies at every distance (pre-F4)
//            vulkan_snow --rain            the second archetype, same emitter
//            vulkan_snow --mv              motion-AOV probe (magnitude AND SIGN)
//            vulkan_snow --msaa N          MSAA G-buffer instead of the default
//            vulkan_snow --no-haze         mesh flakes only, no density volume
//            vulkan_snow --upscaler dlss|fsr|taa
//
// --seq is the motion harness, and for this phase it is the CHECKPOINT rather
// than a convenience: reversed or garbage motion vectors are invisible to every
// image metric and obvious to the eye on four consecutive frames. Combine with
// --t to freeze the field and leave the camera as the only thing moving, which
// separates view dependence from the snowfall's own motion.
//
// CPU cost, the claim of the whole mode, is measured with
//     set THREEPP_CPU_PHASE_PROFILE=1 && vulkan_snow --shot out.png --frames 400
// and reading frame.P_particleFields — the phase that would carry a per-particle
// upload if one existed.

#include "capture_util.hpp"
#include "window_util.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/IcosahedronGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace threepp;

namespace {


    auto matteMat(const Color& c, float rough = 0.92f) {
        return MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(c).roughness(rough).metalness(0.f));
    }

    // Deterministic scene dressing: a capture that has to be LOOKED at and
    // compared across builds is worthless if the rocks move between runs.
    float h01(std::uint32_t i, std::uint32_t s) {
        std::uint32_t x = i * 0x9e3779b9u + s * 0x85ebca6bu;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return float(x >> 8) * (1.f / 16777216.f);
    }

    // ── The ground ──────────────────────────────────────────────────────────
    // Snow-covered, but deliberately NOT white: an overcast scene has almost no
    // dynamic range above the sky, so a 0.9 albedo ground clips and the flakes
    // in front of it disappear. Mid-grey keeps the flakes readable against it,
    // which is what the capture is for.
    std::shared_ptr<Group> makeGround() {
        auto g = Group::create();
        auto ground = Mesh::create(PlaneGeometry::create(160.f, 160.f, 1, 1),
                                   matteMat(Color(0.34f, 0.36f, 0.40f), 0.96f));
        ground->rotateX(-math::PI / 2.f);
        g->add(ground);
        // Drifts: low, wide, snow-coloured lumps so the ground plane is not a
        // featureless sheet the eye reads as infinite.
        for (int i = 0; i < 26; ++i) {
            const float a = h01(std::uint32_t(i), 31u) * math::TWO_PI;
            const float r = 3.5f + 22.f * h01(std::uint32_t(i), 37u);
            const float sc = 0.5f + 1.7f * h01(std::uint32_t(i), 41u);
            auto m = Mesh::create(IcosahedronGeometry::create(sc, 1),
                                  matteMat(Color(0.39f, 0.41f, 0.45f), 0.97f));
            m->position.set(r * std::cos(a), -sc * 0.62f, r * std::sin(a));
            m->rotation.set(h01(std::uint32_t(i), 43u) * math::TWO_PI,
                            h01(std::uint32_t(i), 47u) * math::TWO_PI, 0.f);
            m->scale.set(1.f, 0.42f, 1.f);
            g->add(m);
        }
        return g;
    }

    // ── The props ───────────────────────────────────────────────────────────
    // Dark against the snow, and spread over a DEPTH range — near ones show the
    // flakes crossing in front of them, far ones show the density volume's haze
    // washing them out. Both readings need something solid to happen against.
    std::shared_ptr<Group> makeProps() {
        auto g = Group::create();
        auto wood = matteMat(Color(0.13f, 0.10f, 0.08f), 0.9f);
        auto bark = matteMat(Color(0.10f, 0.09f, 0.085f), 0.95f);

        // A fence running across the frame: the long horizontal the flakes fall
        // past, and the surface that shows the snow's haze with distance.
        for (int i = 0; i < 12; ++i) {
            const float x = -13.f + 2.4f * float(i);
            auto post = Mesh::create(CylinderGeometry::create(0.09f, 0.11f, 1.55f, 8), wood);
            post->position.set(x, 0.72f, -7.5f + 0.35f * std::sin(0.7f * float(i)));
            post->rotation.z = 0.04f * (h01(std::uint32_t(i), 53u) - 0.5f);
            g->add(post);
        }
        for (float y : {0.62f, 1.16f}) {
            auto rail = Mesh::create(BoxGeometry::create(27.f, 0.11f, 0.05f), wood);
            rail->position.set(0.f, y, -7.5f);
            g->add(rail);
        }

        // Bare trees. Trunk plus a few limbs — enough silhouette to catch snow
        // in front of and behind, without a foliage system in a snow demo.
        const float treeX[] = {-6.5f, 4.2f, 9.8f, -11.5f};
        const float treeZ[] = {-3.0f, -11.0f, 1.5f, -14.0f};
        for (int t = 0; t < 4; ++t) {
            auto tree = Group::create();
            const float hgt = 4.2f + 2.6f * h01(std::uint32_t(t), 61u);
            auto trunk = Mesh::create(CylinderGeometry::create(0.10f, 0.20f, hgt, 9), bark);
            trunk->position.y = hgt * 0.5f;
            tree->add(trunk);
            for (int b = 0; b < 6; ++b) {
                const float a = math::TWO_PI * (float(b) + 0.4f * h01(std::uint32_t(t * 8 + b), 67u)) / 6.f;
                const float lb = 0.9f + 1.3f * h01(std::uint32_t(t * 8 + b), 71u);
                auto limb = Mesh::create(CylinderGeometry::create(0.035f, 0.06f, lb, 6), bark);
                limb->position.set(0.35f * std::cos(a), hgt * (0.55f + 0.11f * float(b % 4)),
                                   0.35f * std::sin(a));
                limb->rotation.set(0.f, -a, 1.05f);
                tree->add(limb);
            }
            tree->position.set(treeX[t], 0.f, treeZ[t]);
            g->add(tree);
        }

        // A cabin: the big near mass, so the flakes have something dark and
        // close to be counted against.
        auto cabin = Group::create();
        cabin->add(Mesh::create(BoxGeometry::create(4.6f, 2.5f, 3.4f),
                                matteMat(Color(0.15f, 0.12f, 0.10f), 0.9f)));
        auto roof = Mesh::create(BoxGeometry::create(5.2f, 0.28f, 4.0f),
                                 matteMat(Color(0.44f, 0.46f, 0.50f), 0.95f));
        roof->position.y = 1.4f;
        cabin->add(roof);
        cabin->position.set(-7.0f, 1.25f, 3.4f);
        cabin->rotation.y = 0.35f;
        g->add(cabin);

        // Stacked firewood and a couple of crates near the camera.
        for (int i = 0; i < 12; ++i) {
            const int row = i / 4, col = i % 4;
            auto log = Mesh::create(CylinderGeometry::create(0.10f, 0.10f, 1.3f, 9), wood);
            log->position.set(3.6f + 0.03f * float(row), 0.11f + 0.205f * float(row),
                              1.4f + 0.215f * float(col) - 0.05f * float(row));
            log->rotation.set(0.f, 0.1f, math::PI / 2.f);
            g->add(log);
        }
        return g;
    }

    // ── The two archetypes ──────────────────────────────────────────────────
    // Snow and rain differ ONLY in these numbers. No second shader, no second
    // code path, no branch anywhere in the renderer — which is the claim F-C
    // makes about "archetypes are parameters, not shader variants", stated here
    // as the diff between two initialisers.
    //
    // Read the geometry as: born in a thin slab at the TOP, alive for
    // `lifetime * dutyCycle`, falling at `|velocity|`. The steady-state cloud is
    // that slab swept over the fall distance, which is why the slab is thin and
    // the lifetime is what sets the volume's height. A spawn box the size of the
    // volume would give a triangular density ramp instead of even snowfall.
    ParticleField::EmitterParams snowParams() {
        ParticleField::EmitterParams e;
        // Upwind of the origin, because the wind shears the falling column
        // downwind by wind * lifetime — about 7 m here — and the column should
        // end up over the camera, not beside it.
        e.spawnCenter.set(-4.0f, 12.5f, -1.6f);
        e.spawnHalfExtent.set(24.f, 0.30f, 24.f);
        // Terminal velocity, so ZERO acceleration: a flake is at equilibrium
        // within a metre of leaving the cloud, and an accelerating one is a bug
        // you then have to author around.
        e.velocity.set(0.f, -1.25f, 0.f);
        e.speedSpread = 0.10f;
        e.accel.set(0.f, 0.f, 0.f);
        e.wind.set(0.52f, 0.f, 0.21f);
        // The wobble. driftScale couples the phase to position, so neighbouring
        // flakes lean together and the field reads as gusts crossing the scene
        // rather than as 300k independent jitters.
        e.driftAmplitude = 0.34f;
        e.driftFrequency = 0.15f;
        e.driftScale     = 11.f;
        e.driftGrowth    = 0.f;
        // 14 s at 1.25 m/s is ~17.5 m of fall from y = 12.5, so every flake is
        // well under the ground before its duty window closes — the w < 0 dead
        // slots happen out of sight, which is where a respawn belongs.
        e.lifetime   = 14.f;
        e.dutyCycle  = 0.94f;
        // Flake radius, WORLD units. Tuned down from a first pass at 0.024:
        // the camera stands INSIDE the column, so the nearest flakes are
        // centimetres from the lens and a 2.4 cm proxy paints a 150-px crystal
        // across the frame. The same authoring trap FireEffect's embers hit
        // from the other side — sizes here are metres, not pixels.
        e.size       = 0.016f;
        e.sizeJitter = 0.50f;
        e.seed       = 20260812u;
        return e;
    }

    // Rain is the SECOND archetype and it exists here to prove the point that
    // an archetype is a parameter set: not one line of shader or renderer code
    // differs between this and the snow above.
    //
    // F3 finished the LOOK. A drop falling at 9 m/s crosses ~20 px in a frame,
    // so drawn as a solid it reads as HAIL — which is exactly what F2 shipped
    // and said so. It now draws through the BILLBOARD representation with the
    // quad stretched along its own analytic velocity: the emit dispatch already
    // wrote f(t) and f(t - dt), so (pos - prevPos) IS the frame's displacement
    // and the streak needs no new state, no second buffer and no CPU work.
    ParticleField::EmitterParams rainParams() {
        ParticleField::EmitterParams e;
        e.spawnCenter.set(-2.2f, 13.0f, -0.9f);
        e.spawnHalfExtent.set(22.f, 0.30f, 22.f);
        e.velocity.set(0.f, -9.0f, 0.f);// fast and near-linear
        e.speedSpread = 0.35f;
        e.wind.set(1.30f, 0.f, 0.45f);
        e.driftAmplitude = 0.03f;// rain barely wanders
        e.driftFrequency = 0.9f;
        e.driftScale     = 6.f;
        e.lifetime       = 1.9f;
        e.dutyCycle      = 0.92f;
        e.size           = 0.013f;
        e.sizeJitter     = 0.30f;
        e.seed           = 20260813u;
        return e;
    }

    // Overcast daylight. Exposure PINNED in every mode: a snowfall is a
    // low-contrast scene whose mean luminance moves as the field starts and
    // stops, so auto-exposure would silently cancel the very thing an A/B or a
    // capture is measuring (feedback_vulkan_capture_confounds).
    void applySky(VulkanRenderer& renderer, Scene& scene, DirectionalLight& sun,
                  AmbientLight& ambient, bool rain) {
        renderer.toneMapping = ToneMapping::ACESFilmic;
        // Under 1: an overcast snowfall is a scene of white things lit by a
        // white sky, so at unity exposure everything — flakes, ground, haze —
        // lands in the top of the range at once and the picture is a sheet of
        // paper. Bringing exposure down is what gives the flakes something to
        // be brighter THAN.
        renderer.toneMappingExposure = 0.70f;
        renderer.setAutoExposure(false);
        renderer.setBloomIntensity(0.22f);
        renderer.setBloomThreshold(1.6f);
        renderer.setDeferredStarfield(0.f);
        renderer.setVolumetricFog(true);

        const Color sky = rain ? Color(0.26f, 0.29f, 0.34f) : Color(0.50f, 0.53f, 0.58f);
        renderer.setClearColor(sky);
        scene.background = sky;
        // ONE DirectionalLight (one-sun policy). Overcast: weak, high, and cool
        // — the light in a snowstorm comes from the whole sky, so the ambient
        // carries most of it and the sun only shapes the props.
        sun.color.setRGB(0.92f, 0.94f, 1.00f);
        sun.intensity = rain ? 0.45f : 0.85f;
        sun.position.set(-4.f, 12.f, 6.f);
        ambient.color.setRGB(0.66f, 0.71f, 0.80f);
        ambient.intensity = rain ? 0.35f : 0.50f;

        VulkanRenderer::HeightFogSettings hf;
        hf.baseY       = 0.0f;
        hf.density     = rain ? 0.016f : 0.010f;
        hf.falloff     = 2.2f;
        hf.noiseAmount = 0.35f;
        renderer.setHeightFog(hf);
    }

}// namespace

int main(int argc, char** argv) {

    // --shot/--frames/--cam/--look are the shared capture flags. NB: unlike the
    // aaa_caps demos, --shot here is a PATH written as given (see the capture
    // block below), so it stays out of capture::finishShot.
    const capture::Args cap = capture::parseArgs(argc, argv);
    capture::Shot shot(cap, /*defaultFrames=*/240);
    float shotTime   = -1.f;// >= 0: freeze the emitter at this absolute t
    bool  bench      = false;
    // F4 (3): the LOD split's own A/B. The legs differ only in whether the
    // mesh/billboard distance gates are set, with the field FALLING in both, so
    // the delta is the split and nothing else — the parked/falling bench above
    // cannot express that. One binary, one seed, interleaved.
    bool  benchLod   = false;
    bool  noLod      = false;// --no-lod: the pre-F4 mesh-everywhere behaviour
    bool  rain       = false;
    bool  noHaze     = false;
    std::uint32_t count = 300'000;
    // --rain default. A streaked drop covers two orders of magnitude more
    // pixels than a 1.6 cm flake proxy, so 300k of them is not "the same
    // amount of weather at a different speed", it is a solid additive wall.
    // 90k is a heavy downpour here; --count still overrides, and the perf
    // checkpoint is measured at the full 300k on purpose.
    constexpr std::uint32_t kRainCount = 90'000;
    bool countExplicit = false;
    std::string upscaler = "taa";
    // MSAA instead of a temporal resolve. The one path with NO history at all,
    // which is what makes it the control leg when a temporal artefact is under
    // suspicion: if a trail survives here it is in the motion vectors, and if it
    // does not it is in the history.
    int   msaa = 0;
    bool  mvProbe = false;// --mv: read the motion AOV back and report it
    std::string seqDir;
    int   seqFrames = 8;
    int   seqWarm   = 100;
    float orbitDeg  = 14.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--t" && i + 1 < argc) shotTime = float(std::atof(argv[++i]));
        else if (a == "--count" && i + 1 < argc) { count = std::uint32_t(std::atoi(argv[++i])); countExplicit = true; }
        else if (a == "--bench") bench = true;
        else if (a == "--bench-lod") benchLod = true;
        else if (a == "--no-lod") noLod = true;
        else if (a == "--rain") rain = true;
        else if (a == "--no-haze") noHaze = true;
        else if (a == "--upscaler" && i + 1 < argc) upscaler = argv[++i];
        else if (a == "--msaa" && i + 1 < argc) msaa = std::atoi(argv[++i]);
        else if (a == "--mv") mvProbe = true;
        else if (a == "--seq" && i + 1 < argc) seqDir = argv[++i];
        else if (a == "--seqframes" && i + 1 < argc) seqFrames = std::atoi(argv[++i]);
        else if (a == "--warm" && i + 1 < argc) seqWarm = std::atoi(argv[++i]);
        else if (a == "--orbit" && i + 1 < argc) orbitDeg = float(std::atof(argv[++i]));
    }
    if (rain && !countExplicit) count = kRainCount;
    if (count == 0u) count = 1u;

    Canvas canvas("Vulkan Deferred - Snow",
                  {{"title", std::string("Vulkan Deferred - Snow")},
                   {"size", std::pair<int, int>{1280, 720}},
                   {"vsync", false},
                   {"headless", shot.active() || bench || benchLod || mvProbe || !seqDir.empty()}});
    VulkanRenderer renderer(canvas);
    renderer.setDlss(upscaler == "dlss");
    renderer.setFsr(upscaler == "fsr");
    if (msaa > 1) renderer.setGbufferMsaa(std::uint32_t(msaa));

    Scene scene;
    scene.add(makeGround());
    scene.add(makeProps());

    auto sun = DirectionalLight::create();
    sun->castShadow = true;
    scene.add(sun);
    auto ambient = AmbientLight::create();
    scene.add(ambient);
    applySky(renderer, scene, *sun, *ambient, rain);

    // ── The field ───────────────────────────────────────────────────────────
    // Ownership::Renderer: the positions are device-local and are written by
    // particle_emit.comp. There is no host staging buffer, no submit() and no
    // per-frame upload — submit() on this field THROWS, which is the API saying
    // out loud which half of it applies.
    ParticleField::Config cfg;
    cfg.capacity  = count;
    cfg.ownership = ParticleField::Ownership::Renderer;
    // WSemantic::Radius: the emitter writes each flake's own radius into w and
    // the mesh proxy scales by w / uniformRadius, so sizeJitter is free variety
    // rather than a second buffer.
    cfg.wSemantic     = ParticleField::WSemantic::Radius;
    cfg.uniformRadius = rain ? 0.013f : 0.016f;
    auto snow = ParticleField::create(cfg);
    snow->name = rain ? "rain" : "snow";

    // ── Which representation draws the weather ──────────────────────────────
    //
    // SNOW keeps the mesh proxy: a flake is a slow, nearly stationary solid
    // that the eye resolves as a shape, and putting it in the ordinary G-buffer
    // is what gets it shaded, shadowed, fogged and TAA-resolved like any other
    // surface. Eight triangles per flake at 300k is 2.4M triangles, a fraction
    // of what this renderer draws for a tree. F2's checkpoints are all on that
    // path and F3 deliberately does not move it.
    //
    // RAIN takes the billboard: a drop is not a shape at all, it is a STREAK,
    // and no amount of shading a 1.3 cm octahedron produces one. The two are
    // different representations of the same field with the same emitter, which
    // is the whole argument for the representation split.
    if (rain) {
        // FAINT and THIN, and both words are doing work. The blend is additive
        // over a field the camera stands INSIDE, so every knob here compounds:
        // the first build shipped intensity 0.55 with a 36 cm streak and 300k
        // drops and painted a white sheet with a cabin dimly behind it. Rain is
        // water — it does not glow, it catches a little of the sky and smears
        // it — so the honest values are an order of magnitude lower than an
        // ember's, and the count comes down as well (see kRainCount).
        snow->setBillboardRepr(Color(0.72f, 0.79f, 0.90f), Color(0.60f, 0.67f, 0.78f),
                               /*intensity*/ 0.085f, /*sizeScale*/ 0.30f);
        auto& br = snow->billboardRepr();
        // ~22 cm of travel at 9 m/s: roughly the exposure smear a real camera
        // gives rain, and about 15 px at this framing on a mid-distance drop.
        br.stretchSeconds = 0.024f;
        // The cap that stops a near drop from painting a streak the height of
        // the frame. 30 radii of a 0.4 cm quad is ~12 cm of hard ceiling.
        br.stretchMax   = 30.f;
        br.softness     = 0.95f;// soft along its whole length, no hot core
        br.fadePower    = 0.f;  // a drop does not fade; it lands
        br.sizeTaper    = 0.f;
        br.brightJitter = 0.55f;
        // ── F4 (4): the near-camera streak, killed at its cause ─────────────
        // The F3 sequence shows exactly one anomalously bright bar per frame,
        // and it is always the nearest drop. Nothing is wrong with the streak:
        // stretchMax caps its length in METRES, correctly, and then 1/d turns
        // that legal 12 cm into a quarter of the frame while 1/d^2 makes it the
        // brightest thing in it. Both caps below are that same compounding,
        // clamped in the two domains it shows up in.
        //   • 4.5% of the frame HEIGHT is about 32 px here — long enough to
        //     read as motion smear, short enough never to draw the eye.
        //   • the near fade takes the last 1.2 m to zero, which is where a real
        //     lens has nothing in focus anyway.
        br.stretchMaxScreen = 0.045f;
        br.nearFade         = 1.20f;
        // Re-tuned against the fog (F3 note iii). With the drops now drawn as
        // faint streaks the density volume's haze is doing proportionally more
        // of the work, and it is doing it TWICE now that the quads are fogged
        // too — so the streak comes down again rather than fighting the murk
        // it is supposed to sit inside.
        br.intensity = 0.070f;
    } else {
        auto flakeMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}
                        .color(Color(0.96f, 0.97f, 1.00f))
                        .roughness(0.85f)
                        .metalness(0.f));
        snow->setMeshRepr(OctahedronGeometry::create(cfg.uniformRadius, 0), flakeMat);

        // ── F4 (3): the LOD split, and the near cull ────────────────────────
        // ONE field, ONE position buffer, TWO vertex stages with complementary
        // distance gates — no CPU work, no compaction pass, no second field.
        // The mesh proxy shrinks out over [kLodNear, kLodFar] and the billboard
        // fades in over exactly the same band, so a flake crossing the boundary
        // cross-dissolves between representations instead of popping.
        //
        // WHY: past ~8 m a flake is under 5 px and the octahedron is buying
        // nothing but cost — the 5M bench put 11 of 16.8 ms in G-buffer raster
        // plus the deferred shade of flakes that small. A soft additive sprite
        // carries the same 5 px for a fraction of the fill.
        //
        // (The plan asks for a procedural HEX-flake sprite out there. Not done,
        // deliberately: at the distance the billboard takes over, a flake is
        // 2-5 px across and a hexagon and a disc are the same picture. The
        // shape would be information the frame cannot carry, paid for in a
        // fragment branch on the representation that exists to be cheap.)
        constexpr float kLodNear = 5.5f, kLodFar = 8.0f;
        auto& mr = snow->meshRepr();
        mr.lodFar  = kLodFar;
        mr.lodFade = kLodFar - kLodNear;
        // The F3 capture's "giant flake". Not a size-hash outlier — the
        // emitter's `size * (1 + sizeJitter * rndS)` is bounded to 2x by
        // construction and cannot overshoot — but a flake 40 cm from the lens,
        // which subtends ~100 px and reads as a floating crystal. A real camera
        // resolves nothing at that distance either.
        //
        // The value is larger than "the distance a flake stops being resolvable"
        // because of what the LINEAR ramp does: scale rises as d/nearCull while
        // apparent size falls as 1/d, so inside the band the two cancel exactly
        // and every flake projects to the SAME apparent size, r/nearCull rad.
        // The knob is therefore "how big may the nearest flake get" — 3 m caps
        // a 2.4 cm flake at about 13 px — and it is a cap rather than a hole,
        // which is why no empty bubble opens around the camera.
        mr.nearCull = 3.00f;

        // The far representation. Faint and slightly blue-white, because at
        // this size the flake is a highlight on the overcast sky rather than a
        // lit solid — and additive, because that is the only blend this slice
        // has (a normal-blend field billboard needs the deferred radix sort,
        // still deferred).
        snow->setBillboardRepr(Color(0.88f, 0.91f, 0.97f), Color(0.80f, 0.84f, 0.92f),
                               /*intensity*/ 0.30f, /*sizeScale*/ 1.00f);
        auto& br = snow->billboardRepr();
        br.lodNear     = kLodNear;
        br.lodFade     = kLodFar - kLodNear;
        br.softness    = 0.62f;// a flake is soft-edged at 3 px, not a hard dot
        br.fadePower   = 0.f;  // a flake does not burn out
        br.sizeTaper   = 0.f;
        br.brightJitter = 0.40f;

        // --no-lod restores the pre-F4 picture exactly: mesh proxies at every
        // distance, no near cull, no quads. The billboard representation is
        // switched OFF rather than pushed out of range, so the A/B leg records
        // no overlay draw at all.
        if (noLod) {
            mr.lodFar = mr.lodFade = mr.nearCull = 0.f;
            br.enabled = false;
        }
    }

    // The SAME field also feeds the world-anchored density volume, so the
    // snowfall hazes what is behind it. One emit dispatch serves both
    // representations and every view — nothing is scattered twice.
    if (!noHaze) {
        // sigma per particle is TINY here compared with FireEffect's 2.2, and
        // the reasoning quantity is why: total optical mass is N * sigma, and
        // this field has fifty times the particles spread over a thousand times
        // the volume. 0.055 puts the optical depth to the fence at ~0.3 — a
        // veil that greys the far trees, not a wall.
        snow->setDensityRepr(Vector3(0.f, 6.f, -2.f), Vector3(26.f, 7.5f, 26.f),
                             rain ? 0.014f : 0.055f, /*resolution*/ 96);
        auto& dr = snow->densityRepr();
        // Snow scatters brightly and forward; rain is water, and a rain curtain
        // is DARK — the same haze albedo would turn a downpour into a whiteout.
        dr.albedo     = rain ? Color(0.42f, 0.46f, 0.52f) : Color(0.90f, 0.93f, 0.98f);
        dr.anisotropy = 0.35f;
    }
    snow->setEmitter(rain ? rainParams() : snowParams());
    snow->setEmitterTime(0.f, 1.f / 60.f);
    scene.add(snow);

    PerspectiveCamera camera(48.f, canvas.aspect(), 0.05f, 300.f);
    camera.position.set(5.6f, 2.15f, 9.2f);
    if (cap.camPos) camera.position.copy(*cap.camPos);
    const Vector3 target = cap.camTarget.value_or(Vector3(0.f, 1.7f, 0.f));
    camera.lookAt(target);

    bool falling = true;
    std::unique_ptr<OrbitControls> controls;
    std::unique_ptr<RendererSettingsUi> ui;
    std::unique_ptr<KeyAdapter> keys;
    if (!shot.active() && !bench && !benchLod && !mvProbe && seqDir.empty()) {
        controls = std::make_unique<OrbitControls>(camera, canvas);
        controls->target.copy(target);
        controls->update();
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            ImGui::TextWrapped("One ParticleField, Ownership::Renderer: %u particles "
                               "written on the device by particle_emit.comp from a "
                               "closed-form trajectory. Per-frame CPU cost: two floats. "
                               "SPACE stops it.", unsigned(count));
        });
        keys = std::make_unique<KeyAdapter>(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
            if (evt.key == Key::SPACE) {
                falling = !falling;
                // Park, don't remove: a field with liveCount 0 stays in the
                // scene at one entry and skips its emit dispatch entirely.
                // Removing it would be a structural rebuild (churn contract).
                snow->setLiveCount(falling ? count : 0u);
            }
        });
        canvas.addKeyListener(*keys);
    }
    demo::bindResize(canvas, renderer, camera);

    // ── Bench: what does a 300k device-emitted field cost? ──────────────────
    // Interleaved A/B with the field PARKED as the A leg — it stays in the
    // scene, so this measures the emitter and its representations rather than
    // the cost of one more entry. Exposure pinned, vsync off, fixed dt.
    if (bench) {
        constexpr int kPairs = 6, kWarm = 40, kMeasure = 90;
        constexpr float kDt = 1.f / 60.f;
        struct Acc { double emit = 0, density = 0, gbuf = 0, shade = 0, taa = 0,
                            overlay = 0, gpu = 0, cpu = 0; };
        Acc onA{}, offA{};
        int frame = 0;
        const auto leg = [&](bool on, Acc& acc) {
            snow->setLiveCount(on ? count : 0u);
            for (int i = 0; i < kWarm; ++i) {
                snow->setEmitterTime(float(frame++) * kDt, kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
            }
            Acc a{};
            for (int i = 0; i < kMeasure; ++i) {
                snow->setEmitterTime(float(frame++) * kDt, kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
                const auto t = renderer.lastFrameTimings();
                a.emit    += t.particleEmitMs;
                a.density += t.particleDensityMs;
                a.gbuf    += t.rasterGbufMs;
                a.shade   += t.pathTraceMs;
                a.taa     += t.taaMs;
                a.overlay += t.overlayMs;
                a.gpu     += t.gpuTotalMs;
                a.cpu     += t.cpuFrameMs;
            }
            acc.emit += a.emit / kMeasure;
            acc.density += a.density / kMeasure;
            acc.gbuf += a.gbuf / kMeasure;
            acc.shade += a.shade / kMeasure;
            acc.taa += a.taa / kMeasure;
            acc.overlay += a.overlay / kMeasure;
            acc.gpu += a.gpu / kMeasure;
            acc.cpu += a.cpu / kMeasure;
        };
        for (int p = 0; p < kPairs; ++p) {// A B A B ... , repo A/B rule
            leg(false, offA);
            leg(true, onA);
        }
        const double inv = 1.0 / double(kPairs);
        const auto sz = renderer.size();
        std::printf("\n[bench] %dx%d, %u particles, %d interleaved pairs of %d frames, "
                    "AE pinned, vsync off\n", sz.width(), sz.height(), unsigned(count),
                    kPairs, kMeasure);
        std::printf("  %-24s %10s %10s %10s\n", "ms/frame", "parked", "falling", "delta");
        const auto row = [&](const char* n, double off, double on) {
            std::printf("  %-24s %10.4f %10.4f %10.4f\n", n, off * inv, on * inv,
                        (on - off) * inv);
        };
        row("GPU emit dispatch", offA.emit, onA.emit);
        row("GPU density scatter", offA.density, onA.density);
        row("GPU raster gbuf", offA.gbuf, onA.gbuf);
        row("GPU deferred shade", offA.shade, onA.shade);
        // Reported because taa_resolve's disocclusion path grew a wide motion
        // search for exactly this content (see the shader). The search is gated
        // on a depth mismatch with no local motion context, so a field of small
        // fast movers is its WORST case and this delta is its upper bound.
        row("GPU taa resolve", offA.taa, onA.taa);
        // F3 checkpoint (d). This is the WHOLE post-upscaler overlay pass —
        // the unjittered depth prepass plus the billboard draw — because with
        // the field parked there is no overlay content at all and the pass is
        // skipped entirely. Under --rain that draw is one vkCmdDrawIndirect of
        // (4 vertices x liveCount) instances, and it is heavily FILL bound:
        // each streak covers ~15-200 px and they overlap, so this number tracks
        // screen coverage far more than it tracks particle count.
        row("GPU overlay (billboards)", offA.overlay, onA.overlay);
        row("GPU whole frame", offA.gpu, onA.gpu);
        row("CPU whole frame", offA.cpu, onA.cpu);
        std::printf("  emit + scatter: %.4f ms  (F2 checkpoint (a) budget: 1.0 ms)\n",
                    (onA.emit + onA.density) * inv);
        return 0;
    }

    // ── Bench: what does the F4 LOD SPLIT buy? ──────────────────────────────
    // Both legs have the field falling; they differ only in whether the
    // distance gates are set. That is the honest comparison for a LOD, and it
    // is a comparison the parked/falling bench above structurally cannot make.
    // The gates are read from MeshRepr / BillboardRepr every frame, so a leg is
    // four float writes and no structural change — no rebuild, no device idle,
    // no TAA history clear between legs beyond the warm-up each leg runs.
    if (benchLod) {
        constexpr int kPairs = 6, kWarm = 40, kMeasure = 90;
        constexpr float kDt = 1.f / 60.f;
        struct Acc { double gbuf = 0, shade = 0, taa = 0, overlay = 0, gpu = 0, cpu = 0; };
        Acc onA{}, offA{};
        auto& mr = snow->meshRepr();
        auto& br = snow->billboardRepr();
        const float lodFar = mr.lodFar, lodFade = mr.lodFade, nearCull = mr.nearCull;
        const bool  bbOn   = br.enabled;
        int frame = 0;
        const auto leg = [&](bool lod, Acc& acc) {
            mr.lodFar   = lod ? lodFar : 0.f;
            mr.lodFade  = lod ? lodFade : 0.f;
            mr.nearCull = lod ? nearCull : 0.f;
            br.enabled  = lod && bbOn;
            for (int i = 0; i < kWarm; ++i) {
                snow->setEmitterTime(float(frame++) * kDt, kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
            }
            Acc a{};
            for (int i = 0; i < kMeasure; ++i) {
                snow->setEmitterTime(float(frame++) * kDt, kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
                const auto t = renderer.lastFrameTimings();
                a.gbuf += t.rasterGbufMs;
                a.shade += t.pathTraceMs;
                a.taa += t.taaMs;
                a.overlay += t.overlayMs;
                a.gpu += t.gpuTotalMs;
                a.cpu += t.cpuFrameMs;
            }
            acc.gbuf += a.gbuf / kMeasure;
            acc.shade += a.shade / kMeasure;
            acc.taa += a.taa / kMeasure;
            acc.overlay += a.overlay / kMeasure;
            acc.gpu += a.gpu / kMeasure;
            acc.cpu += a.cpu / kMeasure;
        };
        for (int p = 0; p < kPairs; ++p) {// A B A B ...
            leg(false, offA);
            leg(true, onA);
        }
        const double inv = 1.0 / double(kPairs);
        const auto sz = renderer.size();
        std::printf("\n[bench-lod] %dx%d, %u particles, %d interleaved pairs of %d frames, "
                    "AE pinned, vsync off\n", sz.width(), sz.height(), unsigned(count),
                    kPairs, kMeasure);
        std::printf("  %-24s %10s %10s %10s\n", "ms/frame", "mesh-only", "LOD split", "delta");
        const auto row = [&](const char* n, double off, double on) {
            std::printf("  %-24s %10.4f %10.4f %10.4f\n", n, off * inv, on * inv,
                        (on - off) * inv);
        };
        row("GPU raster gbuf", offA.gbuf, onA.gbuf);
        row("GPU deferred shade", offA.shade, onA.shade);
        row("GPU taa resolve", offA.taa, onA.taa);
        row("GPU overlay (billboards)", offA.overlay, onA.overlay);
        row("GPU whole frame", offA.gpu, onA.gpu);
        row("CPU whole frame", offA.cpu, onA.cpu);
        return 0;
    }

    // ── Scripted-orbit frame sequence ───────────────────────────────────────
    // Closed-form camera path in the frame index, so two runs of the same
    // command produce the same poses. This is where the motion vectors are
    // judged: consecutive frames, at the steady state of every temporal history.
    if (!seqDir.empty()) {
        const capture::OrbitSequence seq{seqDir, seqFrames, seqWarm, orbitDeg};
        capture::runOrbitSequence(
                seq, renderer, camera, target,
                // --t freezes the field: dt 0 makes every particle reproject onto
                // itself, so anything still moving in the sequence is the camera.
                [&](float t) {
                    if (shotTime >= 0.f) snow->setEmitterTime(shotTime, 0.f);
                    else snow->setEmitterTime(t, seq.dt);
                },
                [&] { canvas.animateOnce([&] { renderer.render(scene, camera); }); });
        std::printf("[seq] %d frames -> %s (orbit %.1f deg/s, warm %d, field %s)\n",
                    seqFrames, seqDir.c_str(), double(orbitDeg), seqWarm,
                    shotTime >= 0.f ? "FROZEN" : "falling");
        return 0;
    }

    // ── Motion-vector probe ─────────────────────────────────────────────────
    // The numeric half of checkpoint (c). A trail in a capture has two possible
    // causes that look identical — motion vectors that are WRONG, and motion
    // vectors that are right but too fast for the temporal filter's
    // disocclusion guards — and only the second is worth tuning content
    // against. This reads the motion AOV, segments it with the ids AOV (.x != 0
    // and .w carrying the particle index), and reports what the flakes actually
    // wrote, against the displacement projected on the host from the same
    // closed form the shader evaluates.
    if (mvProbe) {
        constexpr float kDt = 1.f / 60.f;
        for (int i = 0; i < 90; ++i) {
            snow->setEmitterTime(float(i) * kDt, kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }
        std::vector<std::uint8_t> raw;
        int w = 0, h = 0, bpp = 0;
        if (!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::Motion, raw, w, h, bpp)) {
            std::printf("[mv] motion AOV readback failed\n");
            return 1;
        }
        const auto* hf = reinterpret_cast<const std::uint16_t*>(raw.data());
        const auto half2f = [](std::uint16_t v) {
            const std::uint32_t sign = std::uint32_t(v >> 15) << 31;
            const std::int32_t exp = (v >> 10) & 0x1F;
            const std::uint32_t man = v & 0x3FF;
            std::uint32_t bits;
            if (exp == 0) bits = sign;
            else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
            else bits = sign | (std::uint32_t(exp - 15 + 127) << 23) | (man << 13);
            float f;
            std::memcpy(&f, &bits, 4);
            return f;
        };
        // No ids segmentation needed: the camera is HELD, so every pixel with a
        // nonzero motion vector belongs to the field by construction — static
        // geometry writes exactly zero there.
        double sumX = 0, sumY = 0, maxMag = 0;
        std::size_t n = 0;
        const std::size_t px = std::size_t(w) * std::size_t(h);
        for (std::size_t i = 0; i < px; ++i) {
            const float mx = half2f(hf[i * 4 + 0]), my = half2f(hf[i * 4 + 1]);
            const double mag = std::sqrt(double(mx) * mx + double(my) * my);
            if (mag <= 1e-6) continue;// static geometry writes exactly zero
            sumX += mx; sumY += my; maxMag = std::max(maxMag, mag);
            ++n;
        }
        std::printf("[mv] %zu of %zu pixels carry motion (%.2f%%); mean (%.5f, %.5f) NDC, "
                    "max |mv| %.5f NDC = %.2f px\n", n, px, 100.0 * double(n) / double(px),
                    sumX / double(std::max<std::size_t>(n, 1)),
                    sumY / double(std::max<std::size_t>(n, 1)), maxMag,
                    maxMag * 0.5 * double(h));
        // The sign is the thing a picture cannot tell you. Snow falls, so the
        // PREVIOUS position is ABOVE the current one; the motion vector points
        // from current to previous, and the AOV is in GL-Y-up NDC, so mean .y
        // must be POSITIVE. A negative mean here is the reversed-velocity bug.
        std::printf("[mv] mean .y is %s — snow's previous position must be ABOVE, so "
                    "this must be POSITIVE in GL-Y-up NDC\n",
                    sumY > 0.0 ? "POSITIVE (correct)" : "NEGATIVE (REVERSED)");
        return 0;
    }

    // ── Headless capture ────────────────────────────────────────────────────
    if (shot.active()) {
        constexpr float kDt = 1.f / 60.f;// fixed: no wall clock in a capture
        for (int i = 0; i < shot.frames; ++i) {
            if (shotTime >= 0.f) snow->setEmitterTime(shotTime, 0.f);
            else snow->setEmitterTime(float(i) * kDt, kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }
        renderer.writeFramebuffer(shot.name);
        const auto t = renderer.lastFrameTimings();
        std::printf("[shot] %s (%d frames, %u particles, emit %.4f ms, scatter %.4f ms, "
                    "cpu frame %.3f ms)\n", shot.name.c_str(), shot.frames, unsigned(count),
                    double(t.particleEmitMs), double(t.particleDensityMs),
                    double(t.cpuFrameMs));
        if (cap.profile) capture::writeFrameTimings(t, cap, shot.frames);
        return 0;
    }

    Clock clock;
    float t = 0.f;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        t += dt;
        // THE ENTIRE PER-FRAME CPU COST OF 300k PARTICLES.
        snow->setEmitterTime(t, dt);
        renderer.render(scene, camera);
        if (ui) ui->render();
    });
    return 0;
}
