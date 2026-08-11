// Vulkan deferred — campfire at night (plans/particle-atmosphere.md F1).
//
// The whole demo is one FireEffect and some things for it to light. That is
// the point: the flame is a ParticleField whose density carries F0's blackbody
// emission ramp, the smoke is a second field with its own albedo, and the
// warm moving light on the stones, the logs and the ground is an ordinary
// PointLight — so every renderer feature on screen already existed. Nothing
// here reaches into the backend.
//
// What to look for:
//   • a flame whose SHAPE is particles, not a texture — no billboard cards
//   • flickering warm light with MOVING ray-traced shadows on the props
//   • a smoke column leaning away on the wind, unlit and grey, right above a
//     flame that is emissive and warm — two media, two albedos, one scene
//     (the reason F0 made the medium params per field)
//   • embers rising and burning out — small, uneven in size and brightness,
//     cooling from yellow-white to deep red and slightly streaked along their
//     own velocity. These are a THIRD ParticleField (F3), drawn as vertex-less
//     camera-facing quads and emitted on the device from a closed form, so the
//     whole scene is now reproducible with them switched on
//   • bloom on the core, and the froxel glow of the firelight in the height fog
//   • THE FLAME'S MIRROR IMAGE IN THE POND. A traced reflection ray sees
//     geometry, and a flame is not geometry — it is a density volume with a
//     blackbody emission ramp — so until the water-fire reflection march the
//     pond showed only the flicker PointLight's glint on the wave facets.
//     shadeWater now marches the same medium along its reflected leg: the
//     mirrored column is emission, so it wobbles with the ripples, it is
//     dimmed where the smoke plume crosses in front of it, and it brightens
//     toward grazing angles because it is Fresnel-weighted like any reflection.
//
//   • SNOW, on the S key. A fourth ParticleField, device-emitted, drifting on
//     the SAME wind the smoke and the embers ride — one wind, one world. The
//     flakes near the camera are mesh proxies lit by the firelight and fall
//     into the pond's reflection along with the flame.
//
// Controls:  drag = orbit   scroll = zoom   SPACE = ignite / extinguish
//            D = day / night (day is the mode to LOOK at the smoke: sun-lit
//            with HG forward scatter, while the flame washes out as it should)
//            S = snowfall on / off (the field is always there, parked — the
//            key only moves liveCount, so there is no structural rebuild)
// Headless:  vulkan_fire --shot out.png [--frames N] [--t SECONDS] [--no-embers] [--day]
//            vulkan_fire --snow             start with the snowfall running
//            vulkan_fire --no-pond          the pre-pond campsite, for an A/B
//                                           of the reflection in one binary
//            vulkan_fire --legacy-embers   the pre-F3 ParticleSystem sparks,
//                                          for an A/B in one binary at one seed
//            vulkan_fire --no-glow         embers without F4's billboard bloom chain
//            vulkan_fire --bench            interleaved A/B of the effect's GPU cost
//            vulkan_fire --seq DIR          consecutive frames along a SCRIPTED orbit
//
// --seq is the motion harness. Some defects in a volumetric are invisible in a
// still and only appear while the camera moves (view-anchored quantisation is
// the whole class), so a single --shot cannot see them: the evidence has to be
// a SEQUENCE. Combine with --t to freeze the effect, and then the ONLY thing
// changing between the written frames is the camera — any boiling left in the
// image is view dependence by construction, with nothing to blame on the fire's
// own animation. The path is closed-form in frame index, so two runs of the
// same command produce the same poses.

#include "capture_util.hpp"

#include "threepp/extras/effects/FireEffect.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/IcosahedronGeometry.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kTau = 6.28318530718f;

    // ── The pond ────────────────────────────────────────────────────────────
    // A rectangular hole in the campsite's ground plane, a muddy bed 38 cm
    // below the waterline, and an Ocean tucked over it. Everything else in the
    // scene is authored around this rectangle: the ground is built as four
    // slabs AROUND it, the scattered rocks skip it, and the pot moved out of
    // it. Declared here rather than inside makePond() because three different
    // builders need the same numbers and a pond that disagrees with its own
    // hole is a pond with a floating rim.
    //
    // WHERE it is, and why it is not somewhere prettier: a specular reflection
    // point always lies BETWEEN the viewer and what is reflected, so a mirror
    // of the flame can only appear on water that sits on the CAMERA's side of
    // the fire. With the default pose that puts the flame's mirror image in a
    // ~1.2 m strip running from (0.55, 0.97) toward the camera; the rectangle
    // is sized so an orbit keeps that strip on water for the whole near half of
    // the circle. It starts 0.8 m out so the stone ring stays on dry land.
    struct PondRect { float x0, x1, z0, z1; };
    constexpr PondRect kPond{-1.50f, 3.50f, 0.80f, 4.30f};
    constexpr float kWaterY = -0.07f;// surface; waves ride around it
    constexpr float kBedY   = -0.45f;

    bool inPond(float x, float z, float margin) {
        return x > kPond.x0 - margin && x < kPond.x1 + margin &&
               z > kPond.z0 - margin && z < kPond.z1 + margin;
    }

    auto matteMat(const Color& c, float rough = 0.92f) {
        return MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(c).roughness(rough).metalness(0.f));
    }

    // A deterministic little hash, so the scene dressing is the same picture
    // every run — a capture that has to be LOOKED at and compared is worthless
    // if the rocks move between builds.
    float h01(std::uint32_t i, std::uint32_t s) {
        std::uint32_t x = i * 0x9e3779b9u + s * 0x85ebca6bu;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return float(x >> 8) * (1.f / 16777216.f);
    }

    // ── The fire ring ───────────────────────────────────────────────────────
    // Stones close in are the best light-catchers in the scene: they sit inside
    // the inverse-square falloff, so the flicker reads on them hardest, and
    // they throw the short moving shadows that say "point source, right there".
    std::shared_ptr<Group> makeStoneRing(float radius) {
        auto g = Group::create();
        constexpr int kStones = 11;
        for (int i = 0; i < kStones; ++i) {
            const float a = kTau * (float(i) + 0.35f * h01(std::uint32_t(i), 7u)) / float(kStones);
            const float sc = 0.10f + 0.06f * h01(std::uint32_t(i), 11u);
            auto m = Mesh::create(IcosahedronGeometry::create(sc, 1),
                                  matteMat(Color(0.30f + 0.08f * h01(std::uint32_t(i), 13u),
                                                 0.28f, 0.26f),
                                           0.95f));
            m->position.set(radius * std::cos(a), sc * 0.55f, radius * std::sin(a));
            m->rotation.set(h01(std::uint32_t(i), 17u) * kTau,
                            h01(std::uint32_t(i), 19u) * kTau,
                            h01(std::uint32_t(i), 23u) * kTau);
            m->scale.set(1.f, 0.72f, 1.15f);
            g->add(m);
        }
        return g;
    }

    // Logs leaning into the fire — the fuel, and the geometry the flame is
    // actually standing on, so the emissive volume has something solid at its
    // base instead of hanging in the air.
    std::shared_ptr<Group> makeFuel() {
        auto g = Group::create();
        auto wood = matteMat(Color(0.16f, 0.10f, 0.07f), 0.95f);
        for (int i = 0; i < 5; ++i) {
            const float a = kTau * float(i) / 5.f + 0.3f;
            auto log = Mesh::create(CylinderGeometry::create(0.035f, 0.045f, 0.72f, 10), wood);
            log->position.set(0.16f * std::cos(a), 0.16f, 0.16f * std::sin(a));
            log->rotation.set(0.f, -a, 1.05f);
            g->add(log);
        }
        // A charred bed under them, so the ground right at the fire is not a
        // flat plane the light bounces off unnaturally evenly.
        auto bed = Mesh::create(CylinderGeometry::create(0.34f, 0.40f, 0.05f, 20),
                                matteMat(Color(0.05f, 0.045f, 0.04f), 1.f));
        bed->position.y = 0.025f;
        g->add(bed);
        return g;
    }

    // Things at a few metres: they catch much less light, which is exactly what
    // makes the falloff read. The stack of firewood and the two posts also give
    // the moving shadows something long to sweep across the ground.
    std::shared_ptr<Group> makeCamp() {
        auto g = Group::create();
        auto wood = matteMat(Color(0.19f, 0.13f, 0.09f), 0.9f);

        auto bench = Mesh::create(BoxGeometry::create(1.9f, 0.13f, 0.34f), wood);
        bench->position.set(0.15f, 0.36f, -1.75f);
        bench->rotation.y = 0.12f;
        g->add(bench);
        for (float dx : {-0.75f, 0.95f}) {
            auto leg = Mesh::create(BoxGeometry::create(0.13f, 0.36f, 0.30f), wood);
            leg->position.set(0.15f + dx, 0.18f, -1.75f);
            g->add(leg);
        }

        // Split logs stacked beside the fire.
        for (int i = 0; i < 9; ++i) {
            const int row = i / 3, col = i % 3;
            auto log = Mesh::create(CylinderGeometry::create(0.055f, 0.055f, 0.62f, 10), wood);
            log->position.set(1.55f + 0.02f * float(row),
                              0.06f + 0.115f * float(row),
                              -0.55f + 0.125f * float(col) - 0.03f * float(row));
            log->rotation.set(0.f, 0.18f, math::PI / 2.f);
            g->add(log);
        }

        // Two posts with a slack line — vertical geometry catching a rim of
        // firelight, and the tallest thing near enough to be lit at all.
        // Moved 0.6 m west of where F1 put them: the eastern one stood exactly
        // where the pond's west bank now is, and a post in the water with no
        // footing is worse than a post 60 cm further out.
        for (float dx : {-2.5f, -1.75f}) {
            auto post = Mesh::create(CylinderGeometry::create(0.05f, 0.06f, 1.6f, 10), wood);
            post->position.set(dx, 0.8f, 0.9f);
            g->add(post);
        }

        // A packed bag and a pot: small, close, curved — the shapes whose
        // terminators move most visibly as the light flickers.
        auto bag = Mesh::create(IcosahedronGeometry::create(0.26f, 2),
                                matteMat(Color(0.22f, 0.19f, 0.15f), 0.85f));
        bag->position.set(-0.95f, 0.20f, -0.85f);
        bag->scale.set(1.f, 0.78f, 1.2f);
        g->add(bag);

        auto pot = Mesh::create(CylinderGeometry::create(0.14f, 0.11f, 0.20f, 20),
                                MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                        .color(Color(0.30f, 0.29f, 0.28f))
                                        .roughness(0.42f)
                                        .metalness(0.85f)));
        pot->position.set(0.62f, 0.10f, 0.62f);
        g->add(pot);
        return g;
    }

    // One axis-aligned slab of ground, y = 0, given its own rectangle. The
    // campsite floor is four of these around the pond's hole rather than one
    // big plane: a plane cannot have a hole, and a pond laid ON TOP of an
    // unbroken floor is a puddle, not a pond — the shallow-bottom refraction
    // path needs somewhere for the bed to actually be.
    std::shared_ptr<Mesh> groundSlab(const std::shared_ptr<Material>& mat,
                                     float x0, float x1, float z0, float z1) {
        auto m = Mesh::create(PlaneGeometry::create(x1 - x0, z1 - z0, 1, 1), mat);
        m->rotateX(-math::PI / 2.f);
        m->position.set(0.5f * (x0 + x1), 0.f, 0.5f * (z0 + z1));
        return m;
    }

    std::shared_ptr<Group> makeGround(bool pond) {
        auto g = Group::create();
        auto dirt = matteMat(Color(0.085f, 0.072f, 0.058f), 1.f);
        constexpr float kFar = 30.f;
        if (!pond) {
            g->add(groundSlab(dirt, -kFar, kFar, -kFar, kFar));
        } else {
            g->add(groundSlab(dirt, -kFar, kPond.x0, -kFar, kFar));       // west
            g->add(groundSlab(dirt, kPond.x1, kFar, -kFar, kFar));        // east
            g->add(groundSlab(dirt, kPond.x0, kPond.x1, -kFar, kPond.z0));// south bank
            g->add(groundSlab(dirt, kPond.x0, kPond.x1, kPond.z1, kFar)); // north bank
        }
        // Scattered rocks out to a few metres so the ground is not a featureless
        // plane where the light falls off — the falloff needs something to fall
        // off ON. The ones that would stand in the water are skipped rather than
        // renumbered, so every rock that survives keeps the position it always
        // had and a --no-pond capture is the old scene exactly.
        for (int i = 0; i < 40; ++i) {
            const float a = h01(std::uint32_t(i), 31u) * kTau;
            const float r = 1.6f + 6.5f * h01(std::uint32_t(i), 37u);
            const float sc = 0.05f + 0.16f * h01(std::uint32_t(i), 41u);
            const float x = r * std::cos(a), z = r * std::sin(a);
            if (pond && inPond(x, z, -0.10f)) continue;
            auto m = Mesh::create(IcosahedronGeometry::create(sc, 1),
                                  matteMat(Color(0.13f, 0.115f, 0.10f), 1.f));
            m->position.set(x, sc * 0.4f, z);
            m->rotation.set(h01(std::uint32_t(i), 43u) * kTau, h01(std::uint32_t(i), 47u) * kTau, 0.f);
            m->scale.set(1.f, 0.6f, 1.f);
            g->add(m);
        }
        return g;
    }

    // ── The water ───────────────────────────────────────────────────────────
    // The point of the demo's second half. A campfire's flame is a
    // participating medium, so a traced reflection ray — which sees geometry —
    // used to find nothing there: the water carried the flicker PointLight's
    // glint and no flame. shadeWater now marches the density volumes along its
    // reflected leg, so what appears here is the flame ITSELF, mirrored,
    // perturbed by the wave normals and dimmed where the smoke crosses it.
    //
    // `Ocean` and not a Mesh with a water-ish material: the deferred water path
    // is gated on the FFT-displaced ocean marker, which only a DisplacedMesh
    // carries. Under 100 m it resolves the POND material recipe on its own
    // (green-brown, ~2.5 m visibility, 0.8 m thickness) — nothing here authors
    // water parameters. The mesh overhangs the hole by ~1 m on every side so
    // its own straight edge is hidden under the ground slabs; the surface sits
    // 7 cm down so a wave crest cannot poke through them either.
    std::shared_ptr<Group> makePond() {
        auto g = Group::create();
        const float cx = 0.5f * (kPond.x0 + kPond.x1);
        const float cz = 0.5f * (kPond.z0 + kPond.z1);
        const float sx = kPond.x1 - kPond.x0, sz = kPond.z1 - kPond.z0;

        // Silt bed. Dark and rough: a bright floor competes with the mirror,
        // and this is a campsite pond, not a swimming pool.
        auto bed = Mesh::create(PlaneGeometry::create(sx + 0.6f, sz + 0.6f, 1, 1),
                                matteMat(Color(0.055f, 0.050f, 0.040f), 1.f));
        bed->rotateX(-math::PI / 2.f);
        bed->position.set(cx, kBedY, cz);
        g->add(bed);

        // Banks: four slabs closing the gap between the bed and the ground, so
        // a low camera looking across the pond sees earth at the rim and not
        // out through the side of the world.
        auto bank = matteMat(Color(0.070f, 0.060f, 0.048f), 1.f);
        const float bh = -kBedY;// bed to y = 0
        const auto wall = [&](float x, float z, float w, float d) {
            auto m = Mesh::create(BoxGeometry::create(w, bh, d), bank);
            m->position.set(x, kBedY + 0.5f * bh, z);
            g->add(m);
        };
        wall(kPond.x0 - 0.15f, cz, 0.30f, sz + 0.60f);
        wall(kPond.x1 + 0.15f, cz, 0.30f, sz + 0.60f);
        wall(cx, kPond.z0 - 0.15f, sx, 0.30f);
        wall(cx, kPond.z1 + 0.15f, sx, 0.30f);

        // Stones along the rim — the shore, and the geometry whose own
        // reflection gives the eye a reference for how still the water is.
        for (int i = 0; i < 26; ++i) {
            const float u  = (float(i) + 0.5f) / 26.f;
            const float sc = 0.06f + 0.10f * h01(std::uint32_t(i), 61u);
            // Walk the rectangle's perimeter, jittered outward onto dry land.
            const float per = 2.f * (sx + sz);
            float d = u * per + 0.35f * per * 0.02f * h01(std::uint32_t(i), 67u);
            float x, z, nx, nz;
            if (d < sx)            { x = kPond.x0 + d;        z = kPond.z0;      nx = 0.f;  nz = -1.f; }
            else if (d < sx + sz)  { x = kPond.x1;            z = kPond.z0 + (d - sx);      nx = 1.f; nz = 0.f; }
            else if (d < 2*sx + sz){ x = kPond.x1 - (d - sx - sz); z = kPond.z1; nx = 0.f;  nz = 1.f; }
            else                   { x = kPond.x0;            z = kPond.z1 - (d - 2*sx - sz); nx = -1.f; nz = 0.f; }
            const float off = 0.10f + 0.22f * h01(std::uint32_t(i), 71u);
            auto m = Mesh::create(IcosahedronGeometry::create(sc, 1),
                                  matteMat(Color(0.145f, 0.130f, 0.115f), 0.98f));
            m->position.set(x + nx * off, sc * 0.35f, z + nz * off);
            m->rotation.set(h01(std::uint32_t(i), 73u) * kTau, h01(std::uint32_t(i), 79u) * kTau, 0.f);
            m->scale.set(1.f, 0.55f, 1.1f);
            g->add(m);
        }

        Ocean::Options oo;
        oo.size = std::max(sx, sz) + 2.0f;// overhang the hole, hide the edge
        oo.resolution = 256;
        // Barely a breath of air: a pond's job here is to be a MIRROR, and a
        // chopped-up surface scatters the flame into confetti. Enough ripple to
        // perturb the mirrored column and prove it is water rather than a
        // second fire drawn upside down.
        oo.windSpeed  = 1.7f;
        oo.windTheta  = 0.4f;
        oo.choppiness = 0.30f;
        oo.waveScale  = 0.55f;
        auto water = Ocean::create(oo);
        water->position.set(cx, kWaterY, cz);
        g->add(water);
        return g;
    }

    // ── Snowfall ────────────────────────────────────────────────────────────
    // A device-emitted ParticleField (F2's Ownership::Renderer, snow archetype)
    // over the campsite. Three things about it are contracts rather than taste:
    //
    // ONE WIND. The flakes derive their drift from the SAME vector the fire's
    // smoke and embers ride. Two privately-authored winds in one scene is
    // exactly what made the fjord read as "a set of effects sharing a frame
    // rather than a place" (F4 defect amendment), and the fix there was one
    // wind with a documented surface fraction. Same model here: what the demo
    // authors is the SURFACE wind (FireEffect::Params::wind), and the free
    // stream the snow falls through is that divided by kGroundWindFactor.
    //
    // CREATED ONCE, PARKED. The field is constructed at startup and added to
    // the scene immediately with liveCount = 0; the S key only moves that
    // count. Creating it on the keypress would be a structural rebuild — a
    // vkDeviceWaitIdle and a TAA history clear every time somebody toggles the
    // weather (R-6, and FireEffect's own parked-fields pattern).
    //
    // NO DENSITY VOLUME, deliberately — see the note at the call site.
    constexpr std::uint32_t kSnowCount = 90'000;
    // The surface wind under the trees is a fraction of the free stream. The
    // number is the fjord's, and at this demo's authored 0.24 m/s it puts the
    // snow's drift at 0.68 m/s — a light, visibly leaning fall.
    constexpr float kGroundWindFactor = 0.35f;

    std::shared_ptr<ParticleField> makeSnow(const Vector3& groundWind) {
        ParticleField::Config cfg;
        cfg.capacity      = kSnowCount;
        cfg.ownership     = ParticleField::Ownership::Renderer;
        cfg.wSemantic     = ParticleField::WSemantic::Radius;
        cfg.uniformRadius = 0.011f;
        auto snow = ParticleField::create(cfg);
        snow->name = "snow";

        ParticleField::EmitterParams e;
        // An emission SLAB at the top, not a box the size of the volume: the
        // steady-state cloud is the slab swept down the trajectory, and a
        // full-height spawn box gives a triangular density ramp instead of even
        // snowfall (F2's authoring note, third trap).
        e.spawnCenter.set(0.f, 9.0f, 0.f);
        e.spawnHalfExtent.set(12.f, 0.30f, 12.f);
        e.velocity.set(0.f, -1.15f, 0.f);
        e.speedSpread = 0.10f;
        e.wind.set(groundWind.x / kGroundWindFactor, 0.f, groundWind.z / kGroundWindFactor);
        e.driftAmplitude = 0.34f;
        e.driftFrequency = 0.15f;
        e.driftScale     = 11.f;
        e.driftGrowth    = 0.f;
        // Long enough to cross the 9 m column and leave the frame at the
        // bottom; the duty cycle staggers the respawns so the slab does not
        // pulse.
        e.lifetime  = 9.0f;
        e.dutyCycle = 0.94f;
        e.size       = 0.011f;
        e.sizeJitter = 0.50f;
        e.seed       = 20260812u;
        snow->setEmitter(e);
        snow->setEmitterTime(0.f, 1.f / 60.f);
        snow->setLiveCount(0);// parked until the S key says otherwise

        // A flake near the camera is a SHAPE the eye resolves, so it gets the
        // mesh proxy and the whole G-buffer treatment — shaded by the firelight,
        // shadowed, fogged, TAA-resolved. Past 8 m it is two pixels and the
        // billboard is the same picture for none of the cost (F4's LOD split).
        auto flakeMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}
                        .color(Color(0.96f, 0.97f, 1.00f))
                        .roughness(0.85f)
                        .metalness(0.f));
        snow->setMeshRepr(OctahedronGeometry::create(cfg.uniformRadius, 0), flakeMat);
        auto& mr   = snow->meshRepr();
        mr.lodFar  = 6.5f;
        mr.lodFade = 2.5f;
        // Caps how big the NEAREST flake may get: the camera stands inside the
        // column, and without this a flake 40 cm from the lens paints a 60-px
        // faceted crystal (F4 note 5).
        mr.nearCull = 4.5f;

        snow->setBillboardRepr(Color(0.88f, 0.91f, 0.97f), Color(0.80f, 0.84f, 0.92f),
                               /*intensity*/ 0.38f, /*sizeScale*/ 1.00f);
        auto& br        = snow->billboardRepr();
        br.lodNear      = 4.0f;
        br.lodFade      = 2.5f;
        br.softness     = 0.62f;// a flake is soft-edged at 3 px, not a hard dot
        br.fadePower    = 0.f;  // a flake does not burn out
        br.sizeTaper    = 0.f;
        br.brightJitter = 0.40f;
        return snow;
    }

    // Two times of day, one scene. Night is the WOW shot the effect was built
    // for; DAY exists to LOOK at the smoke. In daylight the flame washes out —
    // physically right, its radiance is competing with a sunlit sky — and the
    // grey column becomes the subject, lit by applyParticleFog's directional
    // sun term (HG phase + centroid shadow ray) plus the sky ambient, instead
    // of by the fire's own light. Every knob below is runtime-settable, so the
    // switch is a keypress, not a restart.
    //
    // Exposure is pinned in BOTH modes. Auto-exposure would swing on the fire
    // (and on every flicker of it), which is the documented capture confound
    // in this repo: it eats the fog and it eats the fire.
    void applyTimeOfDay(bool day, VulkanRenderer& renderer, Scene& scene,
                        DirectionalLight& sun, AmbientLight& ambient) {
        renderer.toneMapping = ToneMapping::ACESFilmic;
        renderer.toneMappingExposure = 1.0f;
        renderer.setAutoExposure(false);
        renderer.setBloomIntensity(0.55f);
        renderer.setBloomThreshold(1.15f);
        renderer.setVolumetricFog(true);
        VulkanRenderer::HeightFogSettings hf;
        hf.baseY = 0.0f;
        if (day) {
            const Color sky(0.60f, 0.73f, 0.92f);
            renderer.setClearColor(sky);
            scene.background = sky;
            renderer.setDeferredStarfield(0.f);
            // The one DirectionalLight is the moon at night and the sun by
            // day (one-sun policy: the scene's DirectionalLight claims the
            // sun slot either way, so re-parameterising it is the honest way
            // to switch — two lights fighting over the slot is not).
            // Placed off to the side so the column is SIDE-lit: forward HG
            // scatter puts the bright rim on the sun side and leaves the
            // shadowed side flat grey, which is exactly the asymmetry a
            // sun-lit smoke column should show.
            sun.color.setRGB(1.00f, 0.97f, 0.90f);
            sun.intensity = 2.6f;
            sun.position.set(5.f, 8.f, 2.f);
            ambient.color.setRGB(0.55f, 0.65f, 0.85f);
            ambient.intensity = 0.35f;
            // Daylight haze, not night mist: thin enough that the fire's
            // froxel glow no longer owns the frame, present enough that the
            // sun's in-scatter has a medium to live in.
            hf.density     = 0.006f;
            hf.falloff     = 3.0f;
            hf.noiseAmount = 0.40f;
        } else {
            renderer.setClearColor(Color(0.010f, 0.013f, 0.022f));
            scene.background = Color(0.010f, 0.013f, 0.022f);
            renderer.setDeferredStarfield(0.85f);
            // A moon: just enough to keep the sky-facing surfaces from being
            // pure black, so the firelight has something to be brighter THAN.
            sun.color.setRGB(0.55f, 0.66f, 0.95f);
            sun.intensity = 0.05f;
            sun.position.set(-6.f, 9.f, -4.f);
            ambient.color.setRGB(0.20f, 0.26f, 0.40f);
            ambient.intensity = 0.05f;
            // Ground mist. This is what turns the fire's PointLight into a
            // visible GLOW rather than a light that only exists where it lands
            // on something — the froxel injector scatters clustered lights
            // through this medium, and a low, shallow layer is where a
            // campfire's glow actually sits.
            hf.density     = 0.032f;
            hf.falloff     = 2.4f;
            hf.noiseAmount = 0.55f;
        }
        renderer.setHeightFog(hf);
    }

}// namespace

int main(int argc, char** argv) {

    std::string shotPath;
    int   shotFrames = 260;
    float shotTime   = -1.f;// >= 0: freeze the effect at this absolute t
    // --no-embers used to be MANDATORY for any capture that had to be compared:
    // the embers were a legacy ParticleSystem with its own RNG and per-frame
    // integration, so no two runs of the same command produced the same image
    // (F1 as-built amendment, point 4). Since F3 they are a third
    // ParticleField, device-emitted from a closed form, and the WHOLE scene is
    // reproducible with them on — so the flag is now a look/perf switch, not a
    // determinism requirement. --legacy-embers brings the old path back for an
    // A/B, and brings its non-determinism back with it.
    bool  noEmbers     = false;
    bool  legacyEmbers = false;
    // F4: the ember field's own bloom pyramid. --no-glow is the A/B leg for its
    // cost, and the leg the 0.5 ms budget is re-checked against both ways.
    bool  noGlow       = false;
    // The pond, and the A/B leg for it in one binary at one seed: --no-pond is
    // the campsite exactly as F1..F4 left it (the ground goes back to a single
    // unbroken plane and every surviving rock keeps its old position), so the
    // reflection march has a same-binary control.
    bool  noPond       = false;
    // Snowfall. Off by default (the campfire is the subject), on with --snow or
    // the S key at runtime. The field exists either way — see makeSnow().
    bool  snowOn       = false;
    bool  bench        = false;
    // TAA is what this ships and is gated on. The other two are here to LOOK
    // at, not to ship: a bright emissive volume is the worst case for the open
    // DLSS/FSR emitter-fog silhouette issue, which plan R-1 files under its own
    // bug rather than fighting here.
    std::string upscaler = "taa";
    // --seq: the motion harness (see the header comment).
    std::string seqDir;
    int   seqFrames = 8;   // consecutive frames written
    int   seqWarm   = 100; // frames run before the first write, so TAA/froxel
                           // history is at its steady state for the MOTION —
                           // not at a cold start, which would confound.
    float orbitDeg  = 22.f;// camera azimuth rate, deg/s; 0 = hold the pose
    bool  day       = false;// --day: start in daylight (D toggles at runtime)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--t" && i + 1 < argc) shotTime = float(std::atof(argv[++i]));
        else if (a == "--no-embers") noEmbers = true;
        else if (a == "--legacy-embers") legacyEmbers = true;
        else if (a == "--no-glow") noGlow = true;
        else if (a == "--no-pond") noPond = true;
        else if (a == "--snow") snowOn = true;
        else if (a == "--bench") bench = true;
        else if (a == "--upscaler" && i + 1 < argc) upscaler = argv[++i];
        else if (a == "--seq" && i + 1 < argc) seqDir = argv[++i];
        else if (a == "--seqframes" && i + 1 < argc) seqFrames = std::atoi(argv[++i]);
        else if (a == "--warm" && i + 1 < argc) seqWarm = std::atoi(argv[++i]);
        else if (a == "--orbit" && i + 1 < argc) orbitDeg = float(std::atof(argv[++i]));
        else if (a == "--day") day = true;
    }
    const capture::Args cap = capture::parseArgs(argc, argv);

    Canvas canvas("Vulkan Deferred - Campfire",
                  {{"size", std::pair<int, int>{1280, 720}},
                   {"vsync", false},
                   {"headless", !shotPath.empty() || bench || !seqDir.empty()}});
    VulkanRenderer renderer(canvas);
    renderer.setDlss(upscaler == "dlss");// F1 ships and gates on TAA; see plan R-1
    renderer.setFsr(upscaler == "fsr");

    Scene scene;

    scene.add(makeGround(!noPond));
    scene.add(makeStoneRing(0.46f));
    scene.add(makeFuel());
    scene.add(makeCamp());
    if (!noPond) scene.add(makePond());

    // ONE DirectionalLight, moon or sun depending on the mode — applyTimeOfDay
    // owns every parameter on it (and on the ambient), so the toggle is a
    // re-parameterisation, never an add/remove.
    auto sun = DirectionalLight::create();
    scene.add(sun);
    auto ambient = AmbientLight::create();
    scene.add(ambient);
    applyTimeOfDay(day, renderer, scene, *sun, *ambient);

    // ── The effect ──────────────────────────────────────────────────────────
    FireEffect::Params fp;
    fp.height = 1.05f;
    fp.radius = 0.25f;
    // ── THIS SCENE'S WIND, stated rather than inherited ─────────────────────
    // One demo, one wind: a light breeze across a clearing at dusk. It is the
    // value FireEffect has always defaulted to, written out here because a
    // world's wind belongs to the world and not to the effect — the fjord's
    // campfire, its chimney, its waves, its meadow and its snowfall now all
    // read one vector, and this demo says its own out loud for the same reason.
    // Unchanged numbers, so the campfire is the campfire F1 shipped.
    fp.wind.set(0.22f, 0.f, 0.09f);
    fp.embers = !noEmbers;
    // --legacy-embers loads the old sprite; the F3 ember field needs no asset
    // at all (the billboard fragment shader draws a procedural spark).
    fp.legacyEmbers = legacyEmbers;
    if (noGlow) fp.emberGlow = 0.f;
    if (fp.embers && legacyEmbers) {
        TextureLoader tl;
        fp.emberTexture = tl.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png",
                                  ColorSpace::sRGB);
        fp.embers = fp.emberTexture != nullptr;
    }
    // ── The weather ─────────────────────────────────────────────────────────
    // Created here, from the SAME wind the fire was just given, and added to
    // the scene parked. NO DensityRepr on it, and that is a decision with a
    // mechanism behind it rather than a look preference: a snow haze over the
    // whole campsite is a 24 x 11 x 24 m volume that ENCLOSES both the water
    // and the fire, and both particle marches take the UNION of the bound
    // boxes as their one interval. The reflected leg's 16 fixed steps would
    // then be spread over ~30 m of mostly-empty air instead of the ~3 m the
    // flame occupies, and the mirror image would degrade to a smear — as would
    // the flame itself under applyParticleFog's 32. The flakes themselves are
    // mesh proxies and billboards, so the snowfall is fully visible without
    // it; what is missing is the haze between the camera and the far props.
    // Recorded as an open issue in plans/particle-atmosphere.md rather than
    // worked around in the march.
    auto snow = makeSnow(fp.wind);

    auto fire = FireEffect::create(fp);
    // Above the log bed, not on the ground plane: the flame's hottest, brightest
    // part is its base, and sinking that into the fuel hides the one thing the
    // emission ramp exists to show.
    fire->position.set(0.f, 0.13f, 0.f);
    scene.add(fire);
    fire->ignite();
    scene.add(snow);
    if (snowOn) snow->setLiveCount(kSnowCount);

    // ONE CLOCK for the whole atmosphere, so every loop below (interactive,
    // --shot, --seq, --bench) advances the fire and the weather together and
    // cannot drift apart. dt = 0 freezes the emitter EXACTLY — both trajectory
    // samples become the same expression, so prevPositions is bit-identical to
    // positions and every flake reprojects onto itself, which is what makes a
    // --t capture converge under TAA.
    const auto tick = [&](float tSec, float dt) {
        fire->update(tSec);
        snow->setEmitterTime(tSec, dt);
    };

    // A SITTING eye height, not a standing one. Fresnel is the whole economy of
    // a reflection: at F1's 1.05 m the water returns ~3% of what it reflects,
    // at 0.58 m it returns ~14%, and the difference between those two numbers
    // is whether the flame's mirror image is a hint or the second subject of
    // the frame. It is also where you actually sit at a campfire. --no-pond
    // keeps the pose, so the A/B is one variable.
    PerspectiveCamera camera(48.f, canvas.aspect(), 0.05f, 200.f);
    camera.position.set(1.62f, 0.58f, 3.10f);
    if (cap.camPos) camera.position.copy(*cap.camPos);
    const Vector3 target = cap.camTarget.value_or(Vector3(0.f, 0.45f, 0.f));
    camera.lookAt(target);

    std::unique_ptr<OrbitControls> controls;
    std::unique_ptr<RendererSettingsUi> ui;
    std::unique_ptr<KeyAdapter> keys;// the canvas keeps a reference, not a copy
    if (shotPath.empty() && !bench && seqDir.empty()) {
        controls = std::make_unique<OrbitControls>(camera, canvas);
        controls->target.copy(target);
        controls->update();
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            ImGui::TextWrapped("One FireEffect: an emissive ParticleField for the flame, "
                               "a second field for the smoke, ember billboards, and one "
                               "PointLight. SPACE toggles it, D toggles day/night.");
        });
        keys = std::make_unique<KeyAdapter>(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
            if (evt.key == Key::SPACE) {
                if (fire->lit()) fire->extinguish();
                else fire->ignite();
            } else if (evt.key == Key::D) {
                day = !day;
                applyTimeOfDay(day, renderer, scene, *sun, *ambient);
            } else if (evt.key == Key::S) {
                // The field is already there — this only moves liveCount, so
                // no structural rebuild and no TAA clear (R-6).
                snowOn = !snowOn;
                snow->setLiveCount(snowOn ? kSnowCount : 0u);
            }
        });
        canvas.addKeyListener(*keys);
    }
    canvas.onWindowResize([&](WindowSize s) {
        camera.aspect = s.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(s);
    });

    // ── Bench: what does the whole effect cost on the GPU? ──────────────────
    // Interleaved A/B with the effect EXTINGUISHED as the A leg (the fields
    // stay in the scene, parked — so this measures the effect's work and not
    // the cost of two more entries), exposure pinned, vsync off, fixed dt.
    if (bench) {
        constexpr int kPairs = 6, kWarm = 40, kMeasure = 90;
        constexpr float kDt = 1.f / 60.f;
        struct Acc { double shade = 0, froxel = 0, density = 0, emit = 0,
                            overlay = 0, gpu = 0; };
        Acc onA{}, offA{};
        int frame = 0;
        const auto leg = [&](bool on, Acc& acc) {
            if (on) fire->ignite(); else fire->extinguish();
            for (int i = 0; i < kWarm; ++i) {
                tick(float(frame++) * kDt, kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
            }
            Acc a{};
            for (int i = 0; i < kMeasure; ++i) {
                tick(float(frame++) * kDt, kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
                const auto t = renderer.lastFrameTimings();
                a.shade   += t.pathTraceMs;
                a.froxel  += t.froxelMs;
                a.density += t.particleDensityMs;
                a.emit    += t.particleEmitMs;
                a.overlay += t.overlayMs;
                a.gpu     += t.gpuTotalMs;
            }
            acc.shade += a.shade / kMeasure;
            acc.froxel += a.froxel / kMeasure;
            acc.density += a.density / kMeasure;
            acc.emit += a.emit / kMeasure;
            acc.overlay += a.overlay / kMeasure;
            acc.gpu += a.gpu / kMeasure;
        };
        for (int p = 0; p < kPairs; ++p) {// A B A B ... , repo A/B rule
            leg(false, offA);
            leg(true, onA);
        }
        const double inv = 1.0 / double(kPairs);
        const auto sz = renderer.size();
        std::printf("\n[bench] %dx%d, %d interleaved pairs of %d frames, AE pinned, vsync off\n",
                    sz.width(), sz.height(), kPairs, kMeasure);
        std::printf("  %-22s %10s %10s %10s\n", "GPU ms/frame", "parked", "lit", "delta");
        const auto row = [&](const char* n, double off, double on) {
            std::printf("  %-22s %10.4f %10.4f %10.4f\n", n, off * inv, on * inv, (on - off) * inv);
        };
        row("deferred shade", offA.shade, onA.shade);
        row("froxel inj+int", offA.froxel, onA.froxel);
        row("density scatter", offA.density, onA.density);
        // F3: the embers moved off the legacy ParticleSystem onto a THIRD
        // ParticleField, so two new lines appear in the effect's bill — the
        // device emitter that writes the sparks' positions, and the overlay
        // pass that draws their quads. Both read zero on the parked leg (a
        // parked field records no dispatch and contributes no overlay content),
        // so the delta IS the cost. The overlay row also carries the unjittered
        // depth prepass, which that pass needs and which nothing else in this
        // scene was asking for — under --legacy-embers the same row instead
        // carries the ParticleSystem's own billboard draw, which is what makes
        // the two runs directly comparable.
        row("particle emit", offA.emit, onA.emit);
        row("overlay (embers)", offA.overlay, onA.overlay);
        row("whole frame", offA.gpu, onA.gpu);
        std::printf("  effect total (shade + froxel + density delta): %.4f ms\n",
                    ((onA.shade - offA.shade) + (onA.froxel - offA.froxel) +
                     (onA.density - offA.density)) * inv);
        std::printf("  effect total INCLUDING the embers (+ emit + overlay): %.4f ms\n",
                    ((onA.shade - offA.shade) + (onA.froxel - offA.froxel) +
                     (onA.density - offA.density) + (onA.emit - offA.emit) +
                     (onA.overlay - offA.overlay)) * inv);
        return 0;
    }

    // ── Scripted-orbit frame sequence ───────────────────────────────────────
    // A closed-form camera path (azimuth linear in the frame index, radius and
    // height held) around the same target the interactive OrbitControls uses,
    // so the written frames are exactly what a user dragging the mouse sees.
    // The frames are consecutive, at the steady state of every temporal history
    // in the renderer, which is the only way to LOOK at a motion-only defect.
    if (!seqDir.empty()) {
        namespace fs = std::filesystem;
        fs::create_directories(seqDir);
        constexpr float kDt = 1.f / 60.f;
        const Vector3 eye0 = camera.position;
        const float r0 = std::hypot(eye0.x - target.x, eye0.z - target.z);
        const float a0 = std::atan2(eye0.z - target.z, eye0.x - target.x);
        const float rate = orbitDeg * (kTau / 360.f);// rad/s
        for (int i = 0; i < seqWarm + seqFrames; ++i) {
            const float t = float(i) * kDt;
            const float a = a0 + rate * t;
            camera.position.set(target.x + r0 * std::cos(a), eye0.y, target.z + r0 * std::sin(a));
            camera.lookAt(target);
            tick(shotTime >= 0.f ? shotTime : t, shotTime >= 0.f ? 0.f : kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
            if (i >= seqWarm) {
                char name[64];
                std::snprintf(name, sizeof(name), "f%02d.png", i - seqWarm);
                renderer.writeFramebuffer((fs::path(seqDir) / name).string());
            }
        }
        std::printf("[seq] %d frames -> %s (orbit %.1f deg/s, warm %d, fx %s)\n",
                    seqFrames, seqDir.c_str(), double(orbitDeg), seqWarm,
                    shotTime >= 0.f ? "FROZEN" : "animated");
        return 0;
    }

    // ── Headless capture ────────────────────────────────────────────────────
    if (!shotPath.empty()) {
        const int frames = cap.frames.value_or(shotFrames);
        constexpr float kDt = 1.f / 60.f;// fixed: no wall clock in a capture
        for (int i = 0; i < frames; ++i) {
            // The emitter is closed-form in t, so a capture can either run the
            // clock or freeze it. --t freezes: identical geometry every frame,
            // which is what lets TAA fully converge and makes two runs of this
            // command comparable byte for byte (bar the legacy embers).
            tick(shotTime >= 0.f ? shotTime : float(i) * kDt, shotTime >= 0.f ? 0.f : kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }
        renderer.writeFramebuffer(shotPath);
        std::printf("[shot] %s (%d frames)\n", shotPath.c_str(), frames);
        if (cap.profile) capture::writeFrameTimings(renderer.lastFrameTimings(), cap, frames);
        return 0;
    }

    Clock clock;
    float t = 0.f;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        t += dt;
        tick(t, dt);
        renderer.render(scene, camera);
        if (ui) ui->render();
    });
    return 0;
}
