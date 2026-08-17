// Ambient bird flock — the drop-in demo.
//
// A hillside with two rooftops, a tilted rail and three trees, and eighteen
// birds that decide for themselves when to fly, where to land, how long to
// stand about and when to leave. Nothing here scripts a bird. The whole
// example is scenery plus a panel; the flock is three lines.
//
// WHAT THIS EXAMPLE IS ACTUALLY DOCUMENTING is the two questions the subsystem
// generates: "my birds don't move" (answered by updateCount/stalledUpdates in
// the panel) and "my birds never land" (answered by perchCount() and the
// perch-marker toggle — you can SEE that the bake found nothing, which no
// amount of staring at the sky will tell you).
//
// SCENE ORDER IS LOAD-BEARING, AND IT IS THE ONE NON-OBVIOUS THING IN HERE.
// The perch table has a hard cap (PerchIndex::Params::maxPerches, 4096) and is
// filled first-come-first-served in scene-traversal order. A 120 m ground plane
// probed on a 1 m grid offers about fourteen thousand candidates, so adding it
// first would fill the table with grass and leave the rooftops, the rail and
// the trees — the perches anyone actually looks at — with none at all. The
// ground therefore goes in LAST. Nothing crashes if you reorder it; the birds
// just stop using the interesting furniture, which is a far worse bug to have
// to notice.
//
// The foliage is excluded from the bake with setPerchFilter: leaf cards are
// double-sided quads with no inside, so a downward probe lands a bird on a leaf
// three metres from the branch that would be holding it up.
//
// Screenshot mode (--shot / --shoot) runs on a FIXED dt rather than the wall
// clock, so the same binary writes byte-reproducible PNGs, and forces the GL
// backend so it never blocks on createRenderer's interactive prompt. It also
// implies --fast-perch: a forty-second capture cannot show a perch cycle whose
// shortest leg is twenty-five seconds.
//
//   flock_demo                       interactive; orbit, and click to scare
//   flock_demo --selftest            headless assertions, PASS/FAIL, exit code
//   flock_demo --shoot <dir>         six PNGs into <dir>, then exits
//   flock_demo --shot <prefix>       the same, into <repo>/aaa_caps/
//   --birds N  --seed N  --frames N  --no-ui  --fast-perch  --gl  --vulkan
//   --cam x,y,z  --look x,y,z        override the capture framing (capture_util)

#include "capture_util.hpp"
#include "renderer_factory.hpp"

#include "threepp/extras/fauna/Flock.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // ── Scenery ──────────────────────────────────────────────────────────

    constexpr float kGroundSize = 120.f;// m, square
    constexpr float kFixedDt = 1.f / 60.f;

    // Two sines, deliberately incommensurate, so the ground is never flat
    // enough to hide a bird's feet floating and never steep enough to reject
    // every ground perch on slope. The Flock reads the height from the BAKED
    // heightfield, not from this function — it exists here only to displace the
    // plane, and the two agree because the bake samples the mesh this produces.
    [[nodiscard]] float groundHeight(float x, float z) {

        return 0.85f * std::sin(x * 0.047f) * std::cos(z * 0.039f) +
               0.40f * std::sin(x * 0.113f + 1.7f) +
               0.25f * std::cos(z * 0.091f - 0.4f);
    }

    [[nodiscard]] std::shared_ptr<Mesh> makeGround() {

        auto geometry = PlaneGeometry::create(kGroundSize, kGroundSize, 40u, 40u);

        // PlaneGeometry lies in local XY with normal +Z; the -90° rotation about
        // X below maps local +Z onto world +Y, so displacing local z IS the
        // world height. Local y maps onto world -z, hence the negation.
        auto* position = geometry->getAttribute<float>("position");
        for (int i = 0, n = position->count(); i < n; ++i) {
            const auto vi = static_cast<std::size_t>(i);
            const float lx = position->getX(vi);
            const float ly = position->getY(vi);
            position->setXYZ(vi, lx, ly, groundHeight(lx, -ly));
        }
        geometry->computeVertexNormals();
        geometry->computeBoundingSphere();

        auto material = MeshPhongMaterial::create(
                MeshPhongMaterial::Params{}
                        .color(Color(0.24f, 0.31f, 0.17f))
                        .shininess(4.f)
                        .specular(Color(0x0a0a0a)));

        auto ground = Mesh::create(geometry, material);
        ground->rotation.x = -math::PI / 2.f;
        ground->receiveShadow = true;
        ground->name = "ground";
        return ground;
    }

    [[nodiscard]] std::shared_ptr<Mesh> makeBlock(const Vector3& size, const Vector3& centre, const Color& colour) {

        auto material = MeshPhongMaterial::create(
                MeshPhongMaterial::Params{}
                        .color(colour)
                        .shininess(8.f)
                        .specular(Color(0x111111)));

        auto block = Mesh::create(BoxGeometry::create(size.x, size.y, size.z), material);
        block->position.copy(centre);
        block->castShadow = true;
        block->receiveShadow = true;
        return block;
    }

    // The thin-perch case, and the reason it is TILTED rather than level.
    //
    // PerchIndex classifies a spot as walkable purely by slope, so a level rail
    // reads as walkable ground and the birds try to stroll along a 10 cm beam.
    // Tipping the section 30° puts the top face between walkableSlope (25°) and
    // maxSlope (35°): still a legal perch, no longer a walkable one. The birds
    // land in a row and stay put, which is what a bird on a wire does.
    [[nodiscard]] std::shared_ptr<Mesh> makeRail() {

        auto rail = makeBlock({14.f, 0.10f, 0.10f}, {-6.f, 3.2f, 12.f}, Color(0.30f, 0.30f, 0.33f));
        rail->rotation.x = math::degToRad(30.f);
        rail->name = "rail";
        return rail;
    }

    struct TreeMeshes {
        std::shared_ptr<Mesh> trunk;
        std::shared_ptr<Mesh> foliage;
    };

    // Real branching geometry for the bake to meet — the flat-roof case is easy
    // and proves nothing. Fixed seeds, so the scene is the same every run.
    [[nodiscard]] TreeMeshes makeTree(unsigned int seed, float height, const Vector3& at) {

        vegetation::TreeParams tp;
        vegetation::applyPreset(0, tp);// Oak
        tp.seed = seed;
        tp.trunkHeight = height;
        tp.trunkRadius = 0.16f;
        tp.crownRadiusX = 2.8f;
        tp.crownRadiusZ = 2.8f;
        tp.crownHeight = 4.2f;
        tp.attractorCount = 380;// modest: the bake builds a BVH per trunk
        tp.leafSize = 0.55f;

        vegetation::TreeGenerator gen(tp.seed);
        gen.buildSkeleton(tp);

        auto bark = vegetation::makeBarkTextures(256, tp.seed, tp.barkColor, tp.barkStyle);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);

        auto barkMat = MeshPhongMaterial::create(
                MeshPhongMaterial::Params{}.color(Color::white).shininess(3.f));
        barkMat->map = bark.first;
        barkMat->normalMap = bark.second;

        auto leafMat = MeshPhongMaterial::create(
                MeshPhongMaterial::Params{}.color(Color::white).shininess(2.f));
        leafMat->map = vegetation::makeLeafClusterTexture(256, tp.seed, tp.leafColor, tp.leafShape);
        leafMat->alphaTest = vegetation::kLeafAlphaTest;
        leafMat->side = Side::Double;
        leafMat->vertexColors = true;// baked canopy occlusion

        TreeMeshes out;
        out.trunk = Mesh::create(gen.makeTrunkGeometry(tp), barkMat);
        out.foliage = Mesh::create(gen.makeLeafGeometry(tp), leafMat);
        out.trunk->name = "branches";
        out.foliage->name = "foliage";// setPerchFilter keys on this
        for (auto* m : {out.trunk.get(), out.foliage.get()}) {
            m->position.set(at.x, groundHeight(at.x, at.z), at.z);
            m->castShadow = true;
            m->receiveShadow = true;
        }
        return out;
    }

    // Everything except the flock. Props first, ground LAST — see the banner.
    void buildScenery(Object3D& scene) {

        scene.add(makeBlock({12.f, 6.f, 9.f}, {14.f, 3.f, -8.f}, Color(0.52f, 0.47f, 0.42f)));
        scene.add(makeBlock({9.f, 11.f, 9.f}, {-13.f, 5.5f, -14.f}, Color(0.46f, 0.44f, 0.46f)));
        scene.add(makeRail());

        struct TreePlan {
            unsigned int seed;
            float height;
            Vector3 at;
        };
        const std::array<TreePlan, 3> plans{
                TreePlan{4021u, 5.4f, {6.f, 0.f, 9.f}},
                TreePlan{9137u, 4.2f, {-19.f, 0.f, 4.f}},
                TreePlan{2255u, 6.1f, {20.f, 0.f, 16.f}}};

        for (const auto& plan : plans) {
            const auto tree = makeTree(plan.seed, plan.height, plan.at);
            scene.add(tree.trunk);
            scene.add(tree.foliage);
        }

        scene.add(makeGround());
    }

    // ── Perch markers ────────────────────────────────────────────────────
    //
    // The answer to "my birds never land". An empty cloud means the bake found
    // nothing, which is a scene problem, not a flock problem — and it takes one
    // glance instead of an afternoon.
    [[nodiscard]] std::shared_ptr<Points> makePerchMarkers(const std::vector<fauna::PerchSpot>& spots) {

        std::vector<float> position;
        std::vector<float> colour;
        position.reserve(spots.size() * 3u);
        colour.reserve(spots.size() * 3u);

        for (const auto& s : spots) {
            // Lifted off the surface so the marker is not z-fighting the very
            // face it was baked from.
            position.push_back(s.position.x + s.normal.x * 0.03f);
            position.push_back(s.position.y + s.normal.y * 0.03f);
            position.push_back(s.position.z + s.normal.z * 0.03f);

            const bool ground = s.ground;
            const bool walk = s.walkable;
            colour.push_back(walk ? (ground ? 0.25f : 0.35f) : 1.00f);
            colour.push_back(walk ? (ground ? 0.85f : 1.00f) : 0.55f);
            colour.push_back(walk ? (ground ? 0.35f : 0.45f) : 0.10f);
        }

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(std::move(position), 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(std::move(colour), 3));
        geometry->computeBoundingSphere();

        auto material = PointsMaterial::create(
                PointsMaterial::Params{}.color(Color::white).size(0.22f).sizeAttenuation(true));
        material->vertexColors = true;

        return Points::create(geometry, material);
    }

    // ── Command line ─────────────────────────────────────────────────────

    struct Options {
        int birds = 18;
        unsigned int seed = 1337u;
        bool noUi = false;
        bool fastPerch = false;
        bool selfTest = false;
        std::string shotPrefix;// --shot PREFIX  → <repo>/aaa_caps/
        std::string shootDir;  // --shoot DIR    → DIR/
        int frames = 2401;
        std::optional<GraphicsAPI> api;// unset → interactive backend prompt
    };

    [[nodiscard]] Options parseOptions(int argc, char** argv) {

        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--birds" && i + 1 < argc) o.birds = std::atoi(argv[++i]);
            else if (a == "--seed" && i + 1 < argc) o.seed = static_cast<unsigned int>(std::max(0, std::atoi(argv[++i])));
            else if (a == "--shot" && i + 1 < argc) o.shotPrefix = argv[++i];
            else if (a == "--shoot" && i + 1 < argc) o.shootDir = argv[++i];
            else if (a == "--frames" && i + 1 < argc) o.frames = std::atoi(argv[++i]);
            else if (a == "--no-ui") o.noUi = true;
            else if (a == "--fast-perch") o.fastPerch = true;
            else if (a == "--selftest") o.selfTest = true;
            else if (a == "--gl") o.api = GraphicsAPI::OpenGL;
            else if (a == "--vulkan") o.api = GraphicsAPI::Vulkan;
        }

        // A capture is forty seconds long and the stock perch cycle is 25–90 s
        // aloft, so an uncompressed capture is forty seconds of cruising and
        // proves nothing about the half of the state machine anyone doubts.
        // --shot / --shoot therefore imply --fast-perch.
        if (!o.shotPrefix.empty() || !o.shootDir.empty()) o.fastPerch = true;

        return o;
    }

    [[nodiscard]] Flock::Params makeFlockParams(const Options& o) {

        Flock::Params p;
        p.seed = o.seed;
        p.birdCount = std::clamp(o.birds, 0, 256);
        // Slightly tighter and lower than the stock territory so the flock stays
        // over the scenery instead of loitering off the edge of the frame.
        //
        // DO NOT SHRINK roamRadius MUCH FURTHER. A startled flock overshoots the
        // territory by a roughly FIXED distance in metres (evade runs at 1.5x
        // target speed for up to a second), not by a fixed fraction of the
        // radius, so the "inside 1.5x roamRadius" guarantee that --selftest
        // checks gets tighter as the radius gets smaller. At 30 m it is already
        // down to half a metre of slack.
        p.home.set(0.f, 14.f, 0.f);
        p.roamRadius = 38.f;
        p.cruiseAltitude = 12.f;// just over the tall roof — they use the furniture

        // A jackdaw rather than the default starling: at the 30–50 m this scene
        // is watched from, a 0.42 m starling is six pixels of dark smudge and
        // every cue this subsystem spends its budget on — the wingtip wash, the
        // spanwise twist, the tail fan — is below the resolution of the image.
        // massKg drives the beat allometrically, so the bigger bird also gets
        // the slower, heavier stroke that goes with it (~5.2 Hz, not 8.5).
        p.shape.bodyLength = 0.32f;
        p.shape.bodyRadius = 0.042f;
        p.shape.wingSpan = 0.66f;
        p.massKg = 0.35f;

        p.birdsCastShadow = true;// the deformation is in the verts, so it is correct

        // A 1 m probe grid finds the branch tops; 1.2 m thinning keeps the
        // ground from reading as a regular lattice of landing pads.
        p.perch.probeSpacing = 1.0f;
        p.perch.perchMinSeparation = 1.2f;

        if (o.fastPerch) {
            p.perchIntervalMin = 4.f;
            p.perchIntervalMax = 10.f;
            p.restIntervalMin = 3.f;
            p.restIntervalMax = 8.f;
        }
        return p;
    }

    // Leaf cards have no inside; a probe that lands on one perches a bird three
    // metres from the branch that should be holding it up.
    [[nodiscard]] std::function<bool(const Mesh&)> perchFilter() {

        return [](const Mesh& m) { return m.name != "foliage"; };
    }

    [[nodiscard]] const char* stateName(Flock::BirdState s) {

        switch (s) {
            case Flock::BirdState::Cruise: return "Cruise";
            case Flock::BirdState::Approach: return "Approach";
            case Flock::BirdState::Flare: return "Flare";
            case Flock::BirdState::Perched: return "Perched";
            case Flock::BirdState::Launch: return "Launch";
            case Flock::BirdState::Evade: return "Evade";
        }
        return "?";
    }

    [[nodiscard]] std::array<int, 6> stateHistogram(const Flock& flock) {

        std::array<int, 6> counts{};
        for (int i = 0; i < flock.birdCount(); ++i) {
            counts[static_cast<std::size_t>(flock.stateOf(i))]++;
        }
        return counts;
    }

    // ── --selftest ───────────────────────────────────────────────────────
    //
    // No window, no renderer, no wall clock. Builds the same scenery, bakes it
    // blocking, and runs the assertions that a positions-only replay cannot
    // see — chiefly that a landing ever fires at all. A state machine that
    // compiles, runs, produces no NaN and never once perches is exactly the bug
    // this catches.
    [[nodiscard]] int runSelfTest(const Options& o) {

        bool pass = true;
        const auto check = [&pass](const char* what, bool ok) {
            std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
            if (!ok) pass = false;
        };

        Scene scene;
        buildScenery(scene);

        auto flock = Flock::create(makeFlockParams(o));
        flock->setPerchFilter(perchFilter());
        scene.add(flock);
        flock->bakePerchesBlocking(scene);

        const auto& params = flock->params();
        const float homeY = params.home.y;
        const float bound = 1.5f * params.roamRadius;
        const int cap = static_cast<int>(std::ceil(params.maxPerchedFraction *
                                                   static_cast<float>(flock->birdCount())));

        const auto finite = [&](const Flock& f) {
            for (int i = 0; i < f.birdCount(); ++i) {
                const Vector3& p = f.birdPosition(i);
                const Vector3& v = f.birdVelocity(i);
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;
                if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) return false;
            }
            return true;
        };

        // Pass A — settled soak. Items 6 and 10.
        check("perchCount() > 0 after a blocking bake", flock->perchCount() > 0);
        int perchedMin = flock->birdCount() + 1;
        int perchedMax = 0;
        bool allFinite = true;
        for (int step = 0; step < 3600; ++step) {
            flock->update(kFixedDt);
            allFinite = allFinite && finite(*flock);
            perchedMin = std::min(perchedMin, flock->perchedCount());
            perchedMax = std::max(perchedMax, flock->perchedCount());
        }
        const bool populated = flock->birdCount() > 0;
        check("soak: every position and velocity finite", allFinite);
        check("soak: at least one bird perched at some point", !populated || perchedMax > 0);
        check("soak: the sky never emptied", !populated || perchedMin < flock->birdCount());
        check("soak: perched never exceeded maxPerchedFraction", perchedMax <= cap);

        // Pass B — repeated startles. Item 8: the territory has to hold.
        float worstRadius = 0.f;
        allFinite = true;
        for (int step = 0; step < 3600; ++step) {
            if (step % 100 == 0) flock->startle(params.home, 1e9f);
            flock->update(kFixedDt);
            allFinite = allFinite && finite(*flock);
            for (int i = 0; i < flock->birdCount(); ++i) {
                const Vector3& p = flock->birdPosition(i);
                const float dx = p.x - params.home.x;
                const float dz = p.z - params.home.z;
                worstRadius = std::max(worstRadius, std::sqrt(dx * dx + dz * dz));
            }
        }
        check("startled: every position and velocity finite", allFinite);
        {
            std::ostringstream m;
            m << "startled: flock stayed inside 1.5x roamRadius (" << worstRadius << " m of " << bound << ")";
            check(m.str().c_str(), worstRadius <= bound);
        }

        // Pass C — no perches at all. Item 5: birds fly, nothing NaNs, nothing
        // lands, nothing falls out of the world, nothing logs.
        {
            auto lonely = Flock::create(makeFlockParams(o));
            bool everPerched = false;
            bool aboveFloor = true;
            allFinite = true;
            for (int step = 0; step < 600; ++step) {
                lonely->update(kFixedDt);
                allFinite = allFinite && finite(*lonely);
                for (int i = 0; i < lonely->birdCount(); ++i) {
                    everPerched = everPerched || lonely->stateOf(i) == Flock::BirdState::Perched;
                    aboveFloor = aboveFloor && lonely->birdPosition(i).y > homeY - 200.f;
                }
            }
            check("no perches: perchCount() == 0", lonely->perchCount() == 0);
            check("no perches: every position and velocity finite", allFinite);
            check("no perches: no bird ever perched", !everPerched);
            check("no perches: no bird fell out of the world", aboveFloor);
        }

        std::cout << (pass ? "PASS" : "FAIL") << " flock_demo --selftest" << std::endl;
        return pass ? 0 : 1;
    }

}// namespace


int main(int argc, char** argv) {

    // ── The literal drop-in, verbatim and before any tuning ──────────────
    //
    //     auto birds = Flock::create();
    //     scene->add(birds);
    //     canvas.animate([&] { birds->update(clock.getDelta()); renderer->render(*scene, *camera); });
    //
    // plus `birds->bakePerches(*scene);` if you want them to land on anything.
    // That is the entire contract. Everything below is this demo showing its
    // working: flags, a panel, click-to-scare and a reproducible capture path.

    const Options options = parseOptions(argc, argv);
    const capture::Args shotArgs = capture::parseArgs(argc, argv);

    if (options.selfTest) return runSelfTest(options);

    const bool capturing = !options.shotPrefix.empty() || !options.shootDir.empty();
    const std::string prefix = options.shotPrefix.empty() ? std::string("flock") : options.shotPrefix;
    const int lastFrame = shotArgs.frames.value_or(options.frames);

    // A capture must never sit on createRenderer's interactive prompt, where
    // only the literal "2" selects Vulkan and nothing at all selects a default.
    std::optional<GraphicsAPI> api = options.api;
    if (capturing && !api) api = GraphicsAPI::OpenGL;

    Canvas canvas("Ambient Flock", {{"vsync", !capturing}, {"aa", 4}});
    auto renderer = createRenderer(canvas, api);
    renderer->setClearColor(Color(0.62f, 0.72f, 0.84f));
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.05f;

    // The whole rendering design of this subsystem exists so that it survives
    // both backends, so say which one is on screen.
    const bool isGl = dynamic_cast<GLRenderer*>(renderer.get()) != nullptr;
    std::cout << "[flock] backend: " << (isGl ? "OpenGL" : "Vulkan") << std::endl;

    // DECLARED BEFORE THE SCENE ON PURPOSE. setObserver() takes a raw
    // non-owning pointer that is dereferenced once per bird per frame, and the
    // scene owns the flock — so the camera has to outlive the scene, which in a
    // block-scoped main means it has to be declared first.
    // High and well back: the loiter volume is a 30 m sphere and the birds
    // drift inside it, so a camera framed on the scenery alone loses the flock
    // within a minute.
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 400.f);
    camera.position.set(26.f, 12.f, 26.f);

    Scene scene;
    scene.background = Color(0.62f, 0.72f, 0.84f);
    // Fog is what stops distant birds being distractingly crisp; it is the main
    // reason they sit IN the scene instead of on top of it.
    scene.fog = Fog(Color(0.66f, 0.75f, 0.86f), 25.f, 140.f);

    scene.add(AmbientLight::create(Color(0.62f, 0.70f, 0.82f), 0.55f));

    // A low sun on purpose: a grazing key is what makes the spanwise wing twist
    // scintillate across the flock, and that scintillation is the strongest
    // single argument for baking these vertices on the CPU at all.
    auto sun = DirectionalLight::create(Color(1.0f, 0.94f, 0.82f), 2.2f);
    sun->position.set(-38.f, 22.f, 26.f);
    sun->castShadow = true;
    {
        auto* shadowCam = sun->shadow->camera->as<OrthographicCamera>();
        shadowCam->left = shadowCam->bottom = -45.f;
        shadowCam->right = shadowCam->top = 45.f;
        shadowCam->nearPlane = 1.f;
        shadowCam->farPlane = 160.f;
        sun->shadow->mapSize.set(2048, 2048);
        sun->shadow->bias = -0.0005f;
    }
    scene.add(sun);

    buildScenery(scene);

    Flock::Params params = makeFlockParams(options);
    std::shared_ptr<Flock> flock;
    std::shared_ptr<Points> markers;
    bool showMarkers = false;
    bool markersDirty = true;

    const auto refreshMarkers = [&] {
        if (markers) {
            scene.remove(*markers);
            markers.reset();
        }
        // A bake that found nothing leaves the cloud NULL rather than adding an
        // empty one — which is the honest rendering of "there is nowhere to
        // land", and it is the whole point of the toggle.
        const auto& spots = flock->perchIndex().spots();
        if (spots.empty()) return;

        markers = makePerchMarkers(spots);
        markers->visible = showMarkers;
        markers->name = "perchMarkers";
        scene.add(markers);
    };

    // Params are fixed at construction — there is no setCount and no live
    // reseed, deliberately (topology is immutable for the object's lifetime),
    // so every knob that is not wind rebuilds the flock and re-bakes.
    const auto rebuild = [&] {
        if (flock) {
            flock->setObserver(nullptr);
            flock->setDisturbanceSource(nullptr);
            scene.remove(*flock);
        }
        flock = Flock::create(params);
        flock->setPerchFilter(perchFilter());
        scene.add(flock);
        flock->setObserver(capturing ? nullptr : &camera);
        if (capturing) flock->bakePerchesBlocking(scene);
        else flock->bakePerches(scene);
        markersDirty = true;
    };

    rebuild();

    // --cam / --look are applied BEFORE OrbitControls is constructed: the
    // controls derive their spherical state from the camera they are handed, so
    // moving the camera afterwards would be undone by the first update().
    Vector3 target{0.f, 10.f, 0.f};
    if (shotArgs.camPos) camera.position.copy(*shotArgs.camPos);
    if (shotArgs.camTarget) target.copy(*shotArgs.camTarget);
    camera.lookAt(target);

    OrbitControls controls{camera, canvas};
    controls.target.copy(target);
    controls.enabled = !capturing;
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    // ── Click to scare ───────────────────────────────────────────────────
    //
    // One Raycaster, one startle(). recursive MUST be true: Raycaster's default
    // is false and the trees are nested, so a non-recursive test quietly misses
    // half the scene (cf. examples/misc/raycast.cpp).
    Raycaster raycaster;
    MouseDownListener clickToScare([&](int button, const Vector2& pos) {
        if (button != 0 || !flock) return;
        const auto size = canvas.size();
        const Vector2 ndc{(pos.x / static_cast<float>(size.width())) * 2.f - 1.f,
                          -(pos.y / static_cast<float>(size.height())) * 2.f + 1.f};
        raycaster.setFromCamera(ndc, camera);
        const auto hits = raycaster.intersectObjects(scene.children, true);
        if (hits.empty()) return;
        flock->startle(hits.front().point, 40.f);
    });
    if (!capturing) canvas.addMouseListener(clickToScare);

    // ── Panel ────────────────────────────────────────────────────────────
    bool rebuildRequested = false;
    const auto touched = [&](bool changed) { rebuildRequested = rebuildRequested || changed; };

    std::unique_ptr<RendererSettingsUi> ui;
    if (!capturing && !options.noUi) {
        ui = std::make_unique<RendererSettingsUi>(canvas, *renderer, [&] {
            const auto counts = stateHistogram(*flock);

            ImGui::Text("birds %d   perched %d   flying %d",
                        flock->birdCount(), flock->perchedCount(), flock->flyingCount());
            ImGui::Text("perches %llu", static_cast<unsigned long long>(flock->perchCount()));
            if (!flock->bakeComplete()) {
                ImGui::ProgressBar(flock->bakeProgress(), ImVec2(-1, 0), "baking");
            }
            ImGui::Text("updates %llu   stalled %llu",
                        static_cast<unsigned long long>(flock->updateCount()),
                        static_cast<unsigned long long>(flock->stalledUpdates()));

            ImGui::SeparatorText("States");
            for (std::size_t s = 0; s < counts.size(); ++s) {
                const float frac = flock->birdCount() > 0
                                           ? static_cast<float>(counts[s]) / static_cast<float>(flock->birdCount())
                                           : 0.f;
                char label[32];
                std::snprintf(label, sizeof(label), "%d", counts[s]);
                ImGui::ProgressBar(frac, ImVec2(-120, 0), label);
                ImGui::SameLine();
                ImGui::TextUnformatted(stateName(static_cast<Flock::BirdState>(s)));
            }

            ImGui::SeparatorText("Population (rebuilds)");
            touched(ImGui::SliderInt("Birds", &params.birdCount, 0, 256));
            {
                int seed = static_cast<int>(params.seed);
                if (ImGui::InputInt("Seed", &seed)) {
                    params.seed = static_cast<unsigned int>(std::max(seed, 0));
                    rebuildRequested = true;
                }
            }
            touched(ImGui::SliderFloat("Cruise speed", &params.cruiseSpeed, 2.f, 16.f, "%.1f m/s"));
            touched(ImGui::SliderFloat("Wingbeat", &params.wingbeatHz, 0.f, 20.f, "%.1f Hz (0 = allometric)"));
            touched(ImGui::SliderFloat("Perch interval min", &params.perchIntervalMin, 1.f, 120.f, "%.0f s"));
            touched(ImGui::SliderFloat("Perch interval max", &params.perchIntervalMax, 1.f, 180.f, "%.0f s"));
            touched(ImGui::SliderFloat("Rest interval min", &params.restIntervalMin, 1.f, 120.f, "%.0f s"));
            touched(ImGui::SliderFloat("Rest interval max", &params.restIntervalMax, 1.f, 180.f, "%.0f s"));

            ImGui::SeparatorText("Disturb");
            if (ImGui::Button("Scare", ImVec2(-1, 0))) {
                flock->startle(params.home, 1e9f);
            }
            if (ImGui::Checkbox("Perch markers", &showMarkers) && markers) {
                markers->visible = showMarkers;
            }
        },
                                                   "Ambient Flock");
    }

    if (!capturing) {
        std::cout << "[flock] orbit: drag to rotate, wheel to zoom, left-click the scene to scare them."
                  << std::endl;
    }

    // ── Capture script ───────────────────────────────────────────────────
    //
    // Fixed dt, fixed camera, blocking bake before frame 1, so the whole run is
    // a pure function of the binary and the seed.
    struct Shot {
        int frame;
        const char* suffix;
        bool close;
    };
    constexpr std::array<Shot, 6> kShots{{
            {90, "f0090_formed", false},
            {900, "f0900_approach", false},
            {1500, "f1500_close", true},
            {1800, "f1800_mixed", false},
            {2100, "f2100_ripple", false},
            {2400, "f2400_reformed", false},
    }};
    constexpr int kStartleFrame = 2070;

    const auto outputPath = [&](const char* suffix) {
        const std::string file = prefix + "_" + suffix + ".png";
        if (!options.shootDir.empty()) {
            std::filesystem::path p = std::filesystem::path(options.shootDir) / file;
            std::filesystem::create_directories(p.parent_path());
            return p;
        }
        return capture::shotOutputPath(file);
    };

    // A second camera for the close-up so the wide framing is never disturbed:
    // the simulation does not know a picture is being taken.
    PerspectiveCamera closeCam(34.f, canvas.aspect(), 0.05f, 300.f);

    Clock clock;
    int frame = 0;

    canvas.animate([&] {
        if (capturing) {

            if (frame == kStartleFrame) flock->startle({6.f, 4.f, -3.f}, 45.f);

            flock->update(kFixedDt);
            renderer->render(scene, camera);

            for (const auto& shot : kShots) {
                if (shot.frame != frame) continue;

                if (shot.close) {
                    // Frame a perched bird if there is one — folded wings, legs
                    // down and a planted foot is the pose worth inspecting.
                    int subject = 0;
                    for (int i = 0; i < flock->birdCount(); ++i) {
                        if (flock->stateOf(i) == Flock::BirdState::Perched) {
                            subject = i;
                            break;
                        }
                    }
                    const Vector3 p = flock->birdPosition(subject);
                    closeCam.aspect = camera.aspect;
                    closeCam.updateProjectionMatrix();
                    closeCam.position.set(p.x + 0.62f, p.y + 0.36f, p.z + 0.62f);
                    closeCam.lookAt(p);
                    renderer->render(scene, closeCam);
                }

                const auto path = outputPath(shot.suffix);
                renderer->writeFramebuffer(path);
                std::cout << "wrote " << std::filesystem::absolute(path).string() << std::endl;
            }

            if (++frame >= lastFrame) {
                std::cout << "birds=" << flock->birdCount()
                          << " perched=" << flock->perchedCount()
                          << " flying=" << flock->flyingCount()
                          << " perches=" << flock->perchCount()
                          << " stalled=" << flock->stalledUpdates()
                          << std::endl;
                std::exit(0);
            }
            return;
        }

        controls.update();

        if (rebuildRequested && !ImGui::IsAnyItemActive()) {
            rebuild();
            rebuildRequested = false;
        }
        if (markersDirty && flock->bakeComplete()) {
            refreshMarkers();
            markersDirty = false;
        }

        flock->update(clock.getDelta());
        renderer->render(scene, camera);
        if (ui) ui->render();
    });
}
