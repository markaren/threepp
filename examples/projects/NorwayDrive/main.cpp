// NORWAY DRIVE — a physics car on real-world geodata roads.
//
// The lightweight `norway_terrain` viewer renders a Kartverket DTM + NVDB road
// region pack. This demo makes it DRIVABLE: it reuses the exact same visual
// stack (GeoTerrainPack + RoadNetwork + makeGeoProvider + TileTerrain) and adds
// a PhysX PxVehicle2 engine-drive car (the Drive demo's CarRig, procedural — no
// GLB dependency) on top.
//
// Physics ground representation (the crux — the world is 8 km, far too big for
// one fine collider):
//   • Each ROAD RIBBON (RoadNetwork::buildMeshes geometry) is cooked as a static
//     PxTriangleMesh — the exact banked driving surface, cheap because ribbons
//     are thin. This is what the car actually drives on.
//   • The TERRAIN is a static-trimesh WINDOW (a few hundred metres, ~2 m cells)
//     sampled from the SAME provider.height, re-cooked and recentred when the
//     car nears the window edge. It catches the car off-road / on the verge and
//     keeps physics matching the visuals (trench included — harmless, because
//     the ribbon collider sits above the trench where the car drives).
//
// Default pack: trollstigen (the Fv63 hairpins are the point). Set another with
// a path arg or THREEPP_REGION_PACK.
//
// Controls:
//   W / Up     throttle          S / Down   brake / reverse
//   A / Left   steer left        D / Right  steer right
//   SPACE      handbrake         R          reset onto the road
//   Mouse drag orbits the chase camera, wheel zooms.
//
//   norway_drive [<pack-dir>]
//   norway_drive --shot out.png [--frames N]   headless auto-drive + capture

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxVehicleEngineDrive.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"

#include "../Drive/MustangRig.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    constexpr float kDeg2Rad = 3.14159265358979323846f / 180.f;

    // A square terrain collider patch built from the provider height, re-cooked
    // and recentred as the car roams so PhysX never has to hold the whole 8 km
    // world at fine resolution. Only ~res² triangles are live at a time.
    //
    // The COOK (PxCreateTriangleMesh — the ~30 ms BVH build measured here) runs on
    // a WORKER thread; only the cheap shape/actor swap touches the scene on the
    // main thread. A synchronous main-thread cook per recentre was a periodic
    // frame hitch while driving (dt spike → PhysX spiral-guard → rubber-band).
    class TerrainWindow {
    public:
        TerrainWindow(PhysxWorld& world, const terrain::TerrainProvider& prov,
                      float size, int res, float recenterDist)
            : world_(world), prov_(prov), size_(size), res_(res), recenter_(recenterDist),
              cookParams_(world.physics().getTolerancesScale()) {}

        // Synchronous cook + install — used ONCE at spawn so frame 0 has ground.
        void rebuild(float cx, float cz) {
            std::vector<float> pos;
            std::vector<PxU32> idx;
            buildArrays(cx, cz, pos, idx);
            installActor(cookMesh(pos, idx));
            center_.set(cx, 0.f, cz);
        }

        // Poll a finished async cook (swap it in), and launch a new cook when the
        // car nears the current window's edge. Never blocks the main thread on
        // the BVH build.
        void update(const Vector3& carPos) {
            if (cooking_.valid() &&
                cooking_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                installActor(cooking_.get());
                center_ = pendingCenter_;
            }
            const float dx = carPos.x - center_.x, dz = carPos.z - center_.z;
            if (!cooking_.valid() && dx * dx + dz * dz > recenter_ * recenter_) {
                pendingCenter_.set(carPos.x, 0.f, carPos.z);
                const float cx = carPos.x, cz = carPos.z;
                cooking_ = std::async(std::launch::async, [this, cx, cz] {
                    std::vector<float> pos;
                    std::vector<PxU32> idx;
                    buildArrays(cx, cz, pos, idx);
                    return cookMesh(pos, idx);// PxCreateTriangleMesh: no scene touch → thread-safe
                });
            }
        }

        [[nodiscard]] int rebuilds() const { return rebuilds_; }

    private:
        void buildArrays(float cx, float cz, std::vector<float>& pos, std::vector<PxU32>& idx) const {
            const int vdim = res_ + 1;
            const float step = size_ / static_cast<float>(res_);
            const float x0 = cx - size_ * 0.5f, z0 = cz - size_ * 0.5f;
            pos.clear();
            pos.reserve(static_cast<size_t>(vdim) * vdim * 3);
            for (int j = 0; j < vdim; ++j) {
                const float z = z0 + static_cast<float>(j) * step;
                for (int i = 0; i < vdim; ++i) {
                    const float x = x0 + static_cast<float>(i) * step;
                    pos.push_back(x);
                    pos.push_back(prov_.height(x, z));
                    pos.push_back(z);
                }
            }
            idx.clear();
            idx.reserve(static_cast<size_t>(res_) * res_ * 6);
            for (int j = 0; j < res_; ++j)
                for (int i = 0; i < res_; ++i) {
                    const auto a = static_cast<PxU32>(j * vdim + i);
                    const auto b = a + 1;
                    const auto c = a + static_cast<PxU32>(vdim);
                    const auto d = c + 1;
                    idx.insert(idx.end(), {a, c, b, b, c, d});
                }
        }

        // Worker-safe: builds a standalone PxTriangleMesh (no PxScene interaction).
        PxTriangleMesh* cookMesh(const std::vector<float>& pos, const std::vector<PxU32>& idx) const {
            PxTriangleMeshDesc desc;
            desc.points.count = static_cast<PxU32>(pos.size() / 3);
            desc.points.stride = sizeof(float) * 3;
            desc.points.data = pos.data();
            desc.triangles.count = static_cast<PxU32>(idx.size() / 3);
            desc.triangles.stride = sizeof(PxU32) * 3;
            desc.triangles.data = idx.data();
            return PxCreateTriangleMesh(cookParams_, desc);
        }

        // Main-thread only: wrap a cooked mesh in a static actor, add it, drop the
        // previous window. Cook-before-remove: the car is never over a hole.
        void installActor(PxTriangleMesh* mesh) {
            if (!mesh) return;
            auto& physics = world_.physics();
            PxShape* shape = physics.createShape(PxTriangleMeshGeometry(mesh),
                                                 world_.defaultMaterial(), true);
            PxRigidStatic* body = physics.createRigidStatic(PxTransform(PxIdentity));
            body->attachShape(*shape);
            shape->release();
            mesh->release();// the shape holds a reference
            world_.scene().addActor(*body);
            if (actor_) world_.removeActor(actor_);
            actor_ = body;
            ++rebuilds_;
        }

        PhysxWorld& world_;
        const terrain::TerrainProvider& prov_;
        float size_;
        int res_;
        float recenter_;
        PxCookingParams cookParams_;
        Vector3 center_{0.f, -1e9f, 0.f};
        Vector3 pendingCenter_;
        std::future<PxTriangleMesh*> cooking_;
        PxRigidStatic* actor_ = nullptr;
        int rebuilds_ = 0;
    };

}// namespace

int main(int argc, char** argv) {

    // ── args ──────────────────────────────────────────────────────────────────
    std::string packArg;
    std::string shotPath;
    std::string modelPath = std::string(DATA_FOLDER) +
                            "/models/gltf/ford_mustang_1967/1967_ford_mustang_shelby_cobra_gt500.glb";
    int shotFrames = 900;// ~15 s at 1/60
    bool realDt = false;// step the capture with the REAL variable frame time (jitter test)
    bool noClamp = false;// DIAGNOSTIC: disable the sim-dt clamp (repro the rubber-band)
    std::string tracePath;// --trace <csv>: real vsync window, auto-drive, log per-frame smoothness, exit
    int traceFrames = 720;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (a == "--realdt") realDt = true;
        else if (a == "--noclamp") noClamp = true;
        else if (a == "--trace" && i + 1 < argc) tracePath = argv[++i];
        else if (a == "--trace-frames" && i + 1 < argc) traceFrames = std::atoi(argv[++i]);
        else if (!a.empty() && a[0] != '-') packArg = a;
    }
    if (packArg.empty())
        if (const char* env = std::getenv("THREEPP_REGION_PACK")) packArg = env;
    if (packArg.empty())
        packArg = (std::filesystem::path(PROJECT_FOLDER) / "geodata" / "trollstigen").string();
    const bool capturing = !shotPath.empty();
    const bool tracing = !tracePath.empty();
    const bool autoDrive = capturing || tracing;
    // Smoothness A/B regression toggles (env; the DEFAULTS are the smooth config,
    // established by measurement — see PhysxWorld::Settings::smoothTimestep and the
    // camera block below). Set these to reproduce the OLD juddering behaviour:
    //   THREEPP_DTSMOOTH=1     re-enable the EMA dt low-pass (raised jerk ~11x)
    //   THREEPP_CAM_LINEAR=1   revert the chase-cam to frame-rate-dependent min(1,dt·k)
    const bool dtSmooth = std::getenv("THREEPP_DTSMOOTH") != nullptr;
    const bool camLinear = std::getenv("THREEPP_CAM_LINEAR") != nullptr;

    // ── load pack ───────────────────────────────────────────────────────────────
    terrain::GeoTerrainPack pack;
    try {
        pack = terrain::GeoTerrainPack::load(packArg);
    } catch (const std::exception& e) {
        std::cerr << "[drive] failed to load region pack '" << packArg << "': " << e.what() << std::endl;
        return 1;
    }
    const terrain::GeoRegion& reg = pack.region;
    std::cout << "[drive] loaded pack '" << reg.name << "' worldSize " << reg.worldSize
              << " m, height " << reg.heightMin << ".." << reg.heightMax << " m, roads "
              << pack.roads.size() << "\n";

    // ── road network + unified ground height ─────────────────────────────────────
    std::vector<road::RoadSpec> specs;
    specs.reserve(pack.roads.size());
    for (const auto& gr : pack.roads) {
        road::RoadSpec s;
        s.id = gr.id;
        s.category = gr.category;
        s.width = gr.width;
        s.points = gr.points;
        specs.push_back(std::move(s));
    }
    road::RoadNetwork network(std::move(specs));
    // Conform to the RAW grid first (roads must meet real ground), then bake the
    // road cut into the DEM (terrain::carveRoads): the provider below is a pure
    // bicubic of the carved grid — C1 everywhere, no runtime corridor warp — and
    // the physics terrain window inherits the same smooth field automatically.
    network.conformTo([&pack](float x, float z) { return pack.grid.sampleBicubic(x, z); });
    terrain::carveRoads(pack.grid, network);

    terrain::GeoTerrainOptions gopt;
    gopt.snowHeightMin = std::max(reg.heightMax - 350.f, 900.f);
    gopt.grassHeightMax = std::clamp(reg.heightMin + 0.45f * (reg.heightMax - reg.heightMin), 200.f, 900.f);
    const terrain::TerrainProvider prov = terrain::makeGeoProvider(pack, network, gopt);

    // ── tile terrain (visuals) ───────────────────────────────────────────────────
    terrain::TileTerrainOptions tileOpts;
    tileOpts.worldSize = reg.worldSize;
    tileOpts.rootGrid = 4;
    tileOpts.maxDepth = 5;
    tileOpts.tileRes = 96;
    tileOpts.splitFactor = 1.2f;
    tileOpts.mergeFactor = 1.7f;
    tileOpts.splatTexelsPerQuad = 2;
    tileOpts.asyncBake = true;
    // One tile swap per frame (default 2): while DRIVING the camera moves fast so
    // tiles churn constantly, and each swap add/removes a scene entry — on Vulkan
    // that is a rebuild+waitIdle drain. Spreading swaps keeps the per-frame cost
    // (and thus frame-time jitter that feeds back into the physics dt) low.
    tileOpts.maxSwapsPerFrame = 1;
    tileOpts.refineBias = [&network](float cx, float cz, float half) {
        return network.corridorIntersects(cx, cz, half) ? 2.2f : 1.0f;
    };
    {
        const terrain::DetailMaps dm = terrain::makeDetailMaps({});
        tileOpts.detailMap = dm.albedo;
        tileOpts.detailNormalMap = dm.normalRough;
        tileOpts.detailRepeat = 0.6f;
        tileOpts.detailStrength = 0.7f;
        tileOpts.detailNormalScale = 1.0f;
        tileOpts.detailRoughStrength = 0.5f;
    }

    // ── renderer ──────────────────────────────────────────────────────────────
    Canvas canvas("threepp - NORWAY DRIVE", {{"vsync", true}, {"aa", 4}});
    auto renderer = capturing ? createRenderer(canvas, GraphicsAPI::Vulkan)
                              : createRenderer(canvas);
#ifdef THREEPP_WITH_VULKAN
    if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        // 2× G-buffer MSAA, no upscaler, native scale. With MSAA the raster runs
        // UNJITTERED — the jittered TAA/DLSS paths tremble the whole image with
        // the 8-phase Halton pattern (roads/edges visibly shake; measured ±0.9 px
        // global shifts at a static camera), MSAA2-unjittered is rock-solid at
        // the same fps. See norway_terrain.cpp for the measurements.
        vk->setGbufferMsaa(2);
        vk->setDlss(false);
        vk->setFsr(false);
        vk->setRenderScale(1.0f);
        // Draw the Mustang's glass (tagged to this layer in MustangRig) as a
        // post-shade tint over the full-quality image instead of re-tracing the
        // scene behind it on the deferred path (headless capture forces Vulkan).
        vk->setOverlayLayer(static_cast<int>(drive::MustangRig::kOverlayLayer));
    }
#endif
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    Scene scene;
    RGBELoader rgbe;
    if (auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene.background = env;
        scene.environment = env;
    } else {
        scene.background = Color(0.55f, 0.70f, 0.92f);
    }

    auto tiles = terrain::TileTerrain::create(prov, tileOpts);
    tiles->name = "norway_terrain";
    scene.add(tiles);

    // Road ribbons (visuals) — reused as the driving-surface colliders below.
    auto roads = network.buildMeshes();
    scene.add(roads);

    if (reg.heightMin < 1.0f) {
        auto seaMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                           .color(Color(0.045f, 0.09f, 0.13f))
                                                           .roughness(0.15f)
                                                           .metalness(0.f));
        auto sea = Mesh::create(PlaneGeometry::create(reg.worldSize * 1.2f, reg.worldSize * 1.2f), seaMat);
        sea->rotation.x = -math::PI / 2.f;
        sea->position.y = reg.seaLevel + 0.15f;
        scene.add(sea);
    }

    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.88f), 2.8f);
    {
        const float az = 215.f * kDeg2Rad, el = 34.f * kDeg2Rad;
        sun->position.set(std::cos(el) * std::sin(az), std::sin(el), std::cos(el) * std::cos(az));
        sun->position.multiplyScalar(std::max(reg.worldSize, 2000.f));
    }
    sun->castShadow = true;
    {
        auto* cam = sun->shadow->camera->as<OrthographicCamera>();
        cam->left = cam->bottom = -140.f;
        cam->right = cam->top = 140.f;
        cam->nearPlane = 1.f;
        cam->farPlane = std::max(reg.worldSize * 3.f, 6000.f);
        sun->shadow->mapSize.set(4096, 4096);
        sun->shadow->bias = -0.0005f;
    }
    Object3D sunTarget;
    sunTarget.position.set(0.f, reg.heightMin, 0.f);
    sun->setTarget(sunTarget);
    scene.add(sun);

    // ── physics ─────────────────────────────────────────────────────────────────
    PhysxWorld::Settings worldSettings;
    worldSettings.smoothTimestep = dtSmooth;// default false (measured smoother); env re-enables for A/B
    PhysxWorld world(worldSettings);

    // Road ribbons as exact static-trimesh driving surfaces.
    int ribbonColliders = 0;
    roads->traverseType<Mesh>([&](Mesh& m) {
        if (world.addStaticTrimesh(m)) ++ribbonColliders;
    });

    // Collision-only APRON along both ribbon edges: the carved terrain sits
    // clearance (~0.4 m) below the ribbon surface, so the shoulder-outer edge is
    // otherwise a curb-height cliff — a wheel straying off the pavement takes a
    // 60 m/s² jolt dropping onto the terrain window (measured pre-fix). The
    // apron ramps from the shoulder edge (centerline height) down to just below
    // the carved bench over 2.5 m, so leaving/rejoining the road is smooth.
    // Both windings are emitted so the wheel raycasts can never back-face miss.
    {
        std::vector<float> pos;
        std::vector<unsigned int> idx;
        const float apronW = 2.5f;
        const float drop = 0.45f;// just below the carve ceiling (clearance 0.40)
        network.forEachSegment([&](float ax, float az, float ha, float bx, float bz, float hb,
                                   float /*pavedHalf*/, float corridorHalf) {
            const float dx = bx - ax, dz = bz - az;
            const float len = std::sqrt(dx * dx + dz * dz);
            if (len < 1e-6f) return;
            const float px = -dz / len, pz = dx / len;// unit perpendicular (XZ)
            for (float side : {-1.f, 1.f}) {
                const float inO = side * corridorHalf, outO = side * (corridorHalf + apronW);
                const auto base = static_cast<unsigned int>(pos.size() / 3);
                const float v[4][3] = {
                        {ax + px * inO, ha, az + pz * inO},          // a inner (shoulder edge)
                        {bx + px * inO, hb, bz + pz * inO},          // b inner
                        {ax + px * outO, ha - drop, az + pz * outO}, // a outer (bench level)
                        {bx + px * outO, hb - drop, bz + pz * outO}};// b outer
                for (const auto& p : v) pos.insert(pos.end(), {p[0], p[1], p[2]});
                idx.insert(idx.end(), {base, base + 1, base + 2, base + 1, base + 3, base + 2});
                idx.insert(idx.end(), {base, base + 2, base + 1, base + 1, base + 2, base + 3});
            }
        });
        auto apron = BufferGeometry::create();
        apron->setIndex(idx);
        apron->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        world.addStaticTrimesh(*apron);// collision only — never added to the scene
    }

    // Terrain window collider (recentres with the car).
    TerrainWindow terrainWin(world, prov, /*size*/ 300.f, /*res*/ 150, /*recenter*/ 80.f);

    // ── spawn on the longest road midpoint, aligned to its heading ───────────────
    Vector3 roadMid, roadDir;
    if (!network.longestRoad(roadMid, roadDir)) {
        roadMid.set(0.f, reg.heightMin, 0.f);
        roadDir.set(0.f, 0.f, 1.f);
    }
    const float spawnSurf = [&] {
        const float s = network.surfaceHeight(roadMid.x, roadMid.z, roadMid.y);
        return std::isnan(s) ? roadMid.y : s;
    }();
    terrainWin.rebuild(roadMid.x, roadMid.z);// ground under the spawn before the first step

    // Longest-road centerline — spawn alignment + capture-mode auto-steer
    // (pure-pursuit) so the car FOLLOWS the hairpins instead of ploughing
    // straight off the first bend.
    const std::vector<Vector3> centerline = network.longestRoadCenterline();
    // Seed the pure-pursuit progress index to the GLOBAL-nearest centerline
    // sample at the spawn point (the per-frame advance is only a local descent,
    // so it must start near the car or it sticks at a spurious local minimum).
    size_t ppIdx = 0;
    {
        float best = std::numeric_limits<float>::max();
        for (size_t i = 0; i < centerline.size(); ++i) {
            const float dx = roadMid.x - centerline[i].x;
            const float dz = roadMid.z - centerline[i].z;
            const float d2 = dx * dx + dz * dz;
            if (d2 < best) { best = d2; ppIdx = i; }
        }
    }
    // Replace the global end-to-start heading with the LOCAL tangent at the
    // spawn point (a winding road's global heading can be way off; a misaligned
    // spawn sends the car across the shoulder while the auto-steer acquires the
    // line — measured 40+ m/s^2 jolts).
    if (centerline.size() >= 2) {
        const size_t i0 = (ppIdx > 0) ? ppIdx - 1 : ppIdx;
        const size_t i1 = std::min(ppIdx + 1, centerline.size() - 1);
        Vector3 tan(centerline[i1].x - centerline[i0].x, 0.f, centerline[i1].z - centerline[i0].z);
        if (tan.length() > 1e-4f) roadDir = tan.normalize();
    }

    // ── car model (Mustang GLB): load + measure BEFORE configuring the vehicle —
    // the physics needs the real wheel radius / track / wheelbase. Mirrors the
    // Drive demo (measure the model, don't hardcode).
    GLTFLoader gltf;
    auto carModelResult = gltf.load(modelPath);
    if (!carModelResult || !carModelResult->scene) {
        std::cerr << "[drive] failed to load car model: " << modelPath << "\n";
        return 1;
    }
    auto carModel = carModelResult->scene;
    {
        // The GT500 GLB exports ~100x too small; normalise to a realistic length
        // so the measured vehicle (and everything tuned against it) stays in metres.
        carModel->updateMatrixWorld(true);
        Box3 mb;
        mb.setFromObject(*carModel);
        Vector3 sz;
        mb.getSize(sz);
        constexpr float targetLength = 4.6f;// '67 Shelby GT500 fastback
        const float carLength = std::max({sz.x, sz.z, 1e-6f});
        carModel->scale.multiplyScalar(targetLength / carLength);
        carModel->updateMatrixWorld(true);
    }
    const auto carMeas = drive::MustangRig::measure(*carModel);
    std::cout << "[drive] car " << (carMeas.valid ? "measured" : "FALLBACK")
              << " WxHxL=" << carMeas.chassisWidth << "x" << carMeas.chassisHeight << "x"
              << carMeas.chassisLength << "  wheelR=" << carMeas.wheelRadius
              << "  track=" << carMeas.trackWidth << "  wheelbase=" << carMeas.wheelbase << "\n";

    PhysxVehicleEngineDrive::Settings vs;
    // Collider a touch inside the visible shell so it doesn't snag on scenery;
    // measured numbers keep the defaults' ~4 m turning radius, tight enough for
    // the hairpins.
    vs.chassisWidth = carMeas.chassisWidth * 0.92f;
    vs.chassisHeight = carMeas.chassisHeight * 0.85f;
    vs.chassisLength = carMeas.chassisLength * 0.96f;
    vs.wheelRadius = carMeas.wheelRadius;
    vs.wheelHalfWidth = carMeas.wheelHalfWidth;
    vs.trackWidth = carMeas.trackWidth;
    vs.wheelbase = carMeas.wheelbase;
    vs.suspensionTravelDist = 0.16f;// firmer — a sports coupe
    vs.suspensionAttachmentY = carMeas.wheelCenterYRel + vs.suspensionTravelDist * 0.5f;
    vs.spawnPosition = {roadMid.x, spawnSurf + 1.2f, roadMid.z};
    {
        // Align to the LOCAL road tangent at the spawn midpoint — the global
        // end-to-start heading from longestRoad() can be way off on a winding
        // road, and a misaligned spawn sends the car across the shoulder while
        // the auto-steer acquires the line (measured 40+ m/s^2 jolts).
        Quaternion q;
        q.setFromUnitVectors(Vector3(0.f, 0.f, 1.f), roadDir);
        vs.spawnRotation = q;
    }
    PhysxVehicleEngineDrive vehicle(world, vs);

    auto carRig = std::make_unique<drive::MustangRig>(carModel, carMeas);
    scene.add(carRig->root());
    world.bind(*carRig->root(), *vehicle.chassisActor());

    std::cout << "[drive] " << ribbonColliders << " road-ribbon colliders + recentring terrain window\n"
              << "[drive] spawn on road '" << (pack.roads.empty() ? "?" : pack.roads.front().id)
              << "' at (" << roadMid.x << ", " << spawnSurf << ", " << roadMid.z << ")\n"
              << "[drive] controls: W/Up throttle  S/Down brake  A/D steer  SPACE handbrake  R reset\n"
              << std::flush;

    // ── input ─────────────────────────────────────────────────────────────────
    bool throttleDown = false, brakeDown = false, handbrakeDown = false;
    bool steerLeftDown = false, steerRightDown = false, respawn = false;
    auto keyToggle = [&](Key key, bool down) {
        switch (key) {
            case Key::W: case Key::UP: throttleDown = down; break;
            case Key::S: case Key::DOWN: brakeDown = down; break;
            case Key::A: case Key::LEFT: steerLeftDown = down; break;
            case Key::D: case Key::RIGHT: steerRightDown = down; break;
            case Key::SPACE: handbrakeDown = down; break;
            default: break;
        }
    };
    KeyAdapter pressAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent e) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (e.key == Key::R) respawn = true;
        keyToggle(e.key, true);
    });
    KeyAdapter releaseAdapter(KeyAdapter::KEY_RELEASED, [&](KeyEvent e) { keyToggle(e.key, false); });
    canvas.addKeyListener(pressAdapter);
    canvas.addKeyListener(releaseAdapter);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    // ── chase camera (orbit around the car; mirrors the Drive demo) ──────────────
    auto camera = PerspectiveCamera::create(60.f, canvas.aspect(), 0.2f, std::max(reg.worldSize * 3.f, 12000.f));
    Vector3 camPos{0, 6, -12}, camTarget{0, 1, 0};
    float orbitYaw = 0.f, orbitPitch = 0.36f, orbitDist = 11.77f;
    bool orbiting = false;
    Vector2 lastMouse;
    struct OrbitMouse : MouseListener {
        float &yaw, &pitch, &dist;
        bool& dragging;
        Vector2& lastPos;
        OrbitMouse(float& y, float& p, float& d, bool& dr, Vector2& lp)
            : yaw(y), pitch(p), dist(d), dragging(dr), lastPos(lp) {}
        void onMouseDown(int b, const Vector2& p) override {
            if (ImGui::GetIO().WantCaptureMouse) return;
            if (b == 0) { dragging = true; lastPos = p; }
        }
        void onMouseUp(int b, const Vector2&) override { if (b == 0) dragging = false; }
        void onMouseMove(const Vector2& p) override {
            if (!dragging) return;
            const Vector2 d = p - lastPos;
            lastPos = p;
            yaw -= d.x * 0.01f;
            pitch = std::clamp(pitch + d.y * 0.01f, 0.05f, 1.4f);
        }
        void onMouseWheel(const Vector2& delta) override {
            if (ImGui::GetIO().WantCaptureMouse) return;
            dist = std::clamp(dist - delta.y, 4.f, 45.f);
        }
    } orbitMouse(orbitYaw, orbitPitch, orbitDist, orbiting, lastMouse);
    canvas.addMouseListener(orbitMouse);

    canvas.onWindowResize([&](WindowSize s) {
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(s);
    });

    // ── HUD ─────────────────────────────────────────────────────────────────────
    float steerCmd = 0.f, throttleCmd = 0.f, brakeCmd = 0.f, fps = 0.f;
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float w = 260 * ui.dpiScale();
        ImGui::SetNextWindowPos({static_cast<float>(canvas.size().width()) - w, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({w, 0}, 0);
        ImGui::Begin("Norway Drive");
        ImGui::Text("W/S throttle/brake  A/D steer");
        ImGui::Text("SPACE handbrake  R reset");
        ImGui::Separator();
        ImGui::Text("FPS   : %.0f", fps);
        ImGui::Text("Speed : %.0f km/h", std::abs(vehicle.forwardSpeed()) * 3.6f);
        ImGui::Text("Gear  : %s  %.0f rpm", vehicle.gearLabel().c_str(), vehicle.engineRpm());
        ImGui::End();
    });

    auto doRespawn = [&] {
        auto* actor = vehicle.chassisActor();
        actor->setGlobalPose(toPxTransform(vs.spawnPosition, vs.spawnRotation));
        actor->setLinearVelocity(PxVec3(0));
        actor->setAngularVelocity(PxVec3(0));
        actor->wakeUp();
        vehicle.setThrottle(0.f);
        vehicle.setBrake(0.f);
        vehicle.setSteer(0.f);
        vehicle.setDirection(PhysxVehicleEngineDrive::Direction::Drive);
        steerCmd = 0.f;
    };

    // ── loop ─────────────────────────────────────────────────────────────────────
    Clock clock;
    int frame = 0;
    float fpsAccum = 0.f;
    int fpsFrames = 0;
    // Headless verification accumulators: chassis height vs the road surface,
    // and vertical-acceleration spikes (felt "humps") correlated with terrain-
    // window re-cooks.
    float minErr = std::numeric_limits<float>::max(), maxErr = -std::numeric_limits<float>::max();
    float maxSpeed = 0.f;
    float prevY = 0.f, prevVy = 0.f, maxAbsAccel = 0.f;
    int accelSpikes = 0, prevRebuilds = 0;
    bool havePrevY = false, havePrevVy = false;
    // Issue-1 (rubber-band) instrumentation. dt distribution + per-stage timing +
    // RENDER-pose smoothness (the interpolated car position the camera tracks).
    std::vector<float> dtMsLog;
    dtMsLog.reserve(static_cast<size_t>(std::max(shotFrames, 1)));
    float maxRecookMs = 0.f, maxTileMs = 0.f, maxStepMs = 0.f;
    int lostTimeFrames = 0;// frames whose dt exceeded 4 substeps (physics time lost)
    Vector3 prevRenderPos;
    bool haveRenderPos = false, havePrevPhysSpeed = false;
    float prevPhysSpeed = 0.f, maxPhysSpeedJump = 0.f, maxRawDtMs = 0.f;
    int physSpeedGlitches = 0, renderBackward = 0, dtSpikeFrame = -1, bigDt = 0;
    // --trace: per-frame smoothness log on the REAL vsync path. rawDt = wall frame
    // period; renderPos = interpolated car pos the camera tracks; carScreen = the
    // car's on-screen NDC (what the eye judges). rawPos = un-interpolated actor pos.
    std::string traceCsv = "frame,rawDtMs,renderX,renderY,renderZ,rawX,rawY,rawZ,screenX,screenY,physSpeed\n";

    canvas.animate([&] {
        // Capture normally steps at a fixed 1/60 (deterministic verification);
        // --realdt uses the REAL variable frame time so the exact
        // variable-dt→fixed-accumulator path the interactive loop uses is
        // exercised headlessly (the rubber-band repro).
        const float rawDt = (capturing && !realDt) ? (1.f / 60.f) : clock.getDelta();

        // ── Rubber-band fix ──────────────────────────────────────────────────
        // PhysxWorld::step already fixes the timestep (EMA + 1/60 accumulator +
        // interpolated bindings), so ordinary variable dt is smooth. The problem
        // is HITCHES: a tile-bake / Vulkan entry-churn frame can take 100-150 ms
        // (measured), and dt above maxSubSteps·fixedDt (= 4/60 ≈ 66 ms) trips the
        // accumulator's spiral-of-death guard, which ZEROES it and LOSES sim time
        // — the render then jumps to recover = rubber-band. Clamp the dt fed to
        // physics AND the frame-rate-relative smoothing so a hitch becomes a brief
        // slow-motion frame that stays in sync instead of a snap. (fps / dt stats
        // still use the raw value.)
        const float dt = noClamp ? rawDt : std::min(rawDt, 1.f / 30.f);
        if (capturing) {
            dtMsLog.push_back(rawDt * 1000.f);
            if (rawDt > 4.f / 60.f) ++lostTimeFrames;// hitch frames that WOULD lose time unclamped
            if (rawDt * 1000.f > maxRawDtMs) { maxRawDtMs = rawDt * 1000.f; dtSpikeFrame = frame; }
            if (frame > 20 && rawDt > 1.f / 30.f) ++bigDt;// hitches after warmup
        }
        fpsAccum += rawDt;
        if (++fpsFrames, fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        const PxTransform pose = vehicle.chassisActor()->getGlobalPose();
        const Vector3 carPos(pose.p.x, pose.p.y, pose.p.z);
        const float speedKmh = std::abs(vehicle.forwardSpeed()) * 3.6f;

        float steerInput = (steerLeftDown ? 1.f : 0.f) - (steerRightDown ? 1.f : 0.f);
        float autoThrottle = throttleDown ? 1.f : 0.f;

        // Capture mode: auto-drive the road via pure-pursuit along the longest
        // road's centerline, holding a modest cruise so the car actually FOLLOWS
        // the hairpins rather than launching straight off the first bend.
        if (autoDrive && centerline.size() >= 2) {
            // Advance the progress index to the local nearest point ahead (never
            // backward — avoids snapping to a stacked switchback overhead).
            auto d2 = [&](size_t i) {
                const float dx = carPos.x - centerline[i].x, dz = carPos.z - centerline[i].z;
                return dx * dx + dz * dz;
            };
            while (ppIdx + 1 < centerline.size() && d2(ppIdx + 1) <= d2(ppIdx)) ++ppIdx;
            // SPEED-ADAPTIVE lookahead: short at low speed so the pure-pursuit
            // hugs the centerline through the tight hairpins (a fixed long
            // lookahead cuts the apex and noses the car into the inside cut
            // wall); longer at speed to damp straight-line weave.
            const float lookahead = std::clamp(7.f + std::abs(vehicle.forwardSpeed()) * 0.55f, 8.f, 20.f);
            size_t ti = ppIdx;
            float acc = 0.f;
            while (ti + 1 < centerline.size() && acc < lookahead) {
                acc += centerline[ti].distanceTo(centerline[ti + 1]);
                ++ti;
            }
            Vector3 fwd = Vector3(0.f, 0.f, 1.f).applyQuaternion(fromPxQuat(pose.q));
            fwd.y = 0.f;
            Vector3 toT(centerline[ti].x - carPos.x, 0.f, centerline[ti].z - carPos.z);
            float steerMag = 0.f;
            if (fwd.length() > 1e-4f && toT.length() > 1e-4f) {
                Vector3 f = fwd.clone().normalize(), t = toT.clone().normalize();
                const float cross = f.x * t.z - f.z * t.x;// signed heading error
                const float dot = std::clamp(f.dot(t), -1.f, 1.f);
                const float ang = std::atan2(cross, dot);
                steerInput = std::clamp(-ang * 1.6f, -1.f, 1.f);
                steerMag = std::abs(steerInput);
            }
            // Corner slower: target a low cruise, and cut throttle hard while
            // steering hard so the car crawls the hairpins under control.
            const float cruise = 32.f;// km/h
            autoThrottle = (speedKmh < cruise) ? 1.f : 0.1f;
            autoThrottle *= (1.f - 0.75f * steerMag);
            autoThrottle = std::max(autoThrottle, 0.25f);// always enough to keep crawling
        }

        const float steerScale = 1.f / (1.f + speedKmh * 0.015f);
        const float slew = std::min(1.f, dt * 3.f);
        steerCmd += (steerInput * steerScale - steerCmd) * slew;
        throttleCmd = autoThrottle;
        brakeCmd = brakeDown ? 1.f : 0.f;
        vehicle.setThrottle(throttleCmd);
        vehicle.setBrake(brakeCmd);
        vehicle.setHandbrake(handbrakeDown ? 1.f : 0.f);
        vehicle.setSteer(steerCmd);

        if (respawn) { respawn = false; doRespawn(); }

        using clk = std::chrono::steady_clock;
        auto tw0 = clk::now();
        terrainWin.update(carPos);
        auto tw1 = clk::now();
        world.step(dt);
        auto tw2 = clk::now();
        carRig->update(vehicle, dt, brakeCmd, 0);
        if (capturing) {
            const float recookMs = std::chrono::duration<float, std::milli>(tw1 - tw0).count();
            const float stepMs = std::chrono::duration<float, std::milli>(tw2 - tw1).count();
            maxRecookMs = std::max(maxRecookMs, recookMs);
            maxStepMs = std::max(maxStepMs, stepMs);
        }

        // Chase camera in chassis-local space, transformed to world.
        carRig->root()->updateMatrixWorld();
        const Matrix4& chassisMat = *carRig->root()->matrixWorld;
        const float cosP = std::cos(orbitPitch);
        Vector3 desiredCam(orbitDist * std::sin(orbitYaw) * cosP,
                           orbitDist * std::sin(orbitPitch),
                           -orbitDist * std::cos(orbitYaw) * cosP);
        desiredCam.applyMatrix4(chassisMat);
        Vector3 desiredTarget(0.f, 1.1f, 1.5f);
        desiredTarget.applyMatrix4(chassisMat);
        // Frame-rate-independent smoothing (default): min(1,dt·k) is linear in dt,
        // so its effective time-constant drifts with the frame rate — under variable
        // dt the camera lag jitters and the whole world appears to shimmy even when
        // the car itself is smooth. 1-exp(-k·dt) has a fixed time-constant (1/k s).
        const float lerp = camLinear ? std::min(1.f, dt * 5.f) : (1.f - std::exp(-5.f * dt));
        camPos.lerp(desiredCam, lerp);
        camTarget.lerp(desiredTarget, lerp);
        camera->position.copy(camPos);
        camera->lookAt(camTarget);

        auto tl0 = std::chrono::steady_clock::now();
        tiles->update(camera->position);
        if (capturing)
            maxTileMs = std::max(maxTileMs,
                                 std::chrono::duration<float, std::milli>(
                                         std::chrono::steady_clock::now() - tl0)
                                         .count());
        renderer->render(scene, *camera);
        ui.render();

        if (tracing) {
            const Vector3 rpos = carRig->root()->position;
            const PxTransform rawp = vehicle.chassisActor()->getGlobalPose();
            Vector3 ndc = rpos.clone().project(*camera);// world → NDC via the live camera
            char line[256];
            std::snprintf(line, sizeof(line),
                          "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f\n",
                          frame, rawDt * 1000.f, rpos.x, rpos.y, rpos.z,
                          rawp.p.x, rawp.p.y, rawp.p.z, ndc.x, ndc.y, vehicle.forwardSpeed());
            traceCsv += line;
            if (++frame >= traceFrames) {
                std::ofstream(tracePath) << traceCsv;
                std::cout << "[drive] wrote trace " << tracePath << " (" << frame << " frames, "
                          << "dtSmooth=" << (dtSmooth ? "on" : "off") << ", cam="
                          << (camLinear ? "linear" : "exp") << ")\n"
                          << std::flush;
                std::exit(0);
            }
        }

        // Smoothness — dt-INDEPENDENT so headless dt jitter (0..350 ms) can't
        // pollute it (dividing a displacement by a noisy dt does).
        //   • physics velocity glitch: |Δ forwardSpeed| per frame — a real jolt
        //     (collision, launch, time-loss recovery) spikes this;
        //   • render backward: the interpolated render pos (what the chase cam
        //     tracks) moving OPPOSITE the heading while driving = a rubber-band
        //     spring. Both should stay ~0 on a smooth drive.
        if (capturing && frame > 30) {
            const float ps = vehicle.forwardSpeed();
            if (havePrevPhysSpeed) {
                const float dvs = std::abs(ps - prevPhysSpeed);
                maxPhysSpeedJump = std::max(maxPhysSpeedJump, dvs);
                if (dvs > 1.0f) ++physSpeedGlitches;// >1 m/s in one frame = a jolt
            }
            prevPhysSpeed = ps;
            havePrevPhysSpeed = true;

            const Vector3 rp = carRig->root()->position;
            if (haveRenderPos && std::abs(ps) > 2.f) {
                Vector3 fwd = Vector3(0.f, 0.f, 1.f).applyQuaternion(
                        fromPxQuat(vehicle.chassisActor()->getGlobalPose().q));
                const float along = (rp.x - prevRenderPos.x) * fwd.x + (rp.z - prevRenderPos.z) * fwd.z;
                if (along < -0.002f) ++renderBackward;// sprang backward = rubber-band
            }
            prevRenderPos = rp;
            haveRenderPos = true;
        }

        if (capturing) {
            const float spd = std::abs(vehicle.forwardSpeed()) * 3.6f;
            maxSpeed = std::max(maxSpeed, spd);
            // Road surface at the car = the ribbon surface (conformed centerline +
            // surfaceRaise) at the FOLLOWED progress point — not surfaceHeight(x,z),
            // whose XZ-nearest lookup snaps to the wrong stacked switchback at a
            // hairpin. This is unambiguously the road the car is actually on.
            const float roadSurf = centerline.empty()
                                           ? std::numeric_limits<float>::quiet_NaN()
                                           : centerline[ppIdx].y + road::RoadNetwork::kSurfaceRaise;
            const float err = carPos.y - roadSurf;// chassis origin above road surface
            if (frame > 20 && !std::isnan(roadSurf)) {
                minErr = std::min(minErr, err);
                maxErr = std::max(maxErr, err);
                if (err < -0.5f || err > 3.f)
                    std::cout << "[drive] WARN frame " << frame << " chassisY-roadSurf=" << err
                              << " m OUT OF BAND (car x=" << carPos.x << " z=" << carPos.z << ")\n";
            }
            // Vertical acceleration of the chassis (finite difference at the
            // fixed timestep). Spikes = felt humps; log alongside the terrain-
            // window re-cook count so a collider swap that jolts the car is
            // immediately visible in the log.
            {
                const float y = carPos.y;
                if (havePrevY) {
                    const float vy = (y - prevY) / dt;
                    if (havePrevVy && frame > 20) {
                        const float acc = (vy - prevVy) / dt;
                        maxAbsAccel = std::max(maxAbsAccel, std::abs(acc));
                        if (std::abs(acc) > 25.f) {
                            ++accelSpikes;
                            std::cout << "[drive] ACCEL SPIKE frame " << frame << ": " << acc
                                      << " m/s^2 (recooks so far " << terrainWin.rebuilds() << ")\n";
                        }
                    }
                    prevVy = vy;
                    havePrevVy = true;
                }
                prevY = y;
                havePrevY = true;
            }
            if (terrainWin.rebuilds() != prevRebuilds) {
                prevRebuilds = terrainWin.rebuilds();
                std::cout << "[drive] terrain window re-cook #" << prevRebuilds << " at frame "
                          << frame << "\n";
            }
            if (frame % 120 == 0)
                std::cout << "[t=" << frame << "] speed=" << spd << " km/h  gear=" << vehicle.gearLabel()
                          << "  chassisY-roadSurf=" << (std::isnan(roadSurf) ? 0.f : err)
                          << " m  ppIdx=" << ppIdx << "/" << centerline.size() << "\n"
                          << std::flush;
            if (++frame >= shotFrames) {
                const auto outPath = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
                std::filesystem::create_directories(outPath.parent_path());
                renderer->writeFramebuffer(outPath);
                // dt distribution (percentiles are robust to the frame-0 warmup).
                std::vector<float> sorted = dtMsLog;
                std::sort(sorted.begin(), sorted.end());
                auto pct = [&](float q) {
                    return sorted.empty() ? 0.f : sorted[std::min(sorted.size() - 1,
                                                                  static_cast<size_t>(q * sorted.size()))];
                };
                const float dtMax = sorted.empty() ? 0.f : sorted.back();
                float dtSum = 0.f;
                for (float v : dtMsLog) dtSum += v;
                const float dtMean = dtMsLog.empty() ? 0.f : dtSum / static_cast<float>(dtMsLog.size());
                std::cout << "[drive] wrote " << outPath.string() << "\n"
                          << "[drive] auto-drive height error vs road surface: min " << minErr
                          << " m, max " << maxErr << " m; max speed " << maxSpeed << " km/h; "
                          << fps << " fps\n"
                          << "[drive] vertical accel: max |a| " << maxAbsAccel << " m/s^2, spikes(>25) "
                          << accelSpikes << ", terrain re-cooks " << terrainWin.rebuilds() << "\n"
                          << "[drive] dt(ms): mean " << dtMean << " p50 " << pct(0.5f) << " p95 "
                          << pct(0.95f) << " p99 " << pct(0.99f) << " max " << dtMax << " @frame "
                          << dtSpikeFrame << "  (mode " << (realDt ? "REAL-VARIABLE" : "fixed 1/60") << ")\n"
                          << "[drive] stage max ms: recook " << maxRecookMs << ", world.step " << maxStepMs
                          << ", tiles.update " << maxTileMs << "; hitches>33ms after warmup " << bigDt << "\n"
                          << "[drive] SMOOTHNESS: physics |Δv| max " << maxPhysSpeedJump << " m/s, jolts(>1m/s) "
                          << physSpeedGlitches << "; render-backward(rubber-band) frames " << renderBackward << "\n"
                          << std::flush;
                std::exit(0);
            }
        }
    });

    return 0;
}
