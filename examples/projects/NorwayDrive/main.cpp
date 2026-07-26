// NORWAY DRIVE — a physics car on real-world geodata roads.
//
// The `norway_terrain` viewer renders a Kartverket DTM + NVDB road region pack
// with the BAKED-ROAD pipeline (roads carved + painted INTO the terrain, only
// bridge decks + near-detail ground-ribbon chunks kept as geometry). This demo
// makes that DRIVABLE: it reuses the exact same visual stack (GeoTerrainPack +
// RoadNetwork profile/carve/paint + makeGeoProvider + TileTerrain) and adds a
// PhysX PxVehicle2 engine-drive Mustang on top.
//
// Physics ground representation (the crux — the world is 8 km, far too big for
// one fine collider):
//   • The TERRAIN is a static-trimesh WINDOW (a few hundred metres, ~2 m cells)
//     sampled from provider.height, re-cooked and recentred as the car roams.
//     Because carveRoads(bakeSurface) SETS the road bed into the DEM (flat across,
//     cut AND fill), this window already IS the road surface at grade — it catches
//     the car off-road AND carries distant roads with no separate collider.
//   • The near-detail ground-ribbon CHUNKS + BRIDGE decks (RoadNetwork::
//     buildGroundChunkMeshes / buildBridgeMeshes) are cooked as persistent static
//     PxTriangleMeshes — the exact banked surface the car rides, sitting
//     kSurfaceRaise above the baked bed (matching what's drawn). Distance-culling
//     hides far chunks visually but keeps their colliders.
//   • Roads classified as TUNNELS or FERRIES are excluded (a car never drives
//     through a mountain or on the open sea); BRIDGE spans get deck colliders.
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

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/physx/PhysxVehicleEngineDrive.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoBuildings.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"
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
    // ND_TOPDOWN: an above-and-behind camera looking down at the car + the road
    // behind it — the view that makes the moving-car reflection/GI ghost obvious
    // (the road-behind is a big flat area where the temporal history smears).
    const bool topDown = std::getenv("ND_TOPDOWN") != nullptr;

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
    // BAKED-ROAD pipeline (same as norway_terrain). Conform to the RAW grid first
    // (roads must meet real ground) with the elevation PROFILE enabled — the
    // pack's NVDB point heights classify every span: a data height well above
    // ground (or ground = water) is a BRIDGE deck that spans the vertical U
    // instead of draping into it; well below ground is a TUNNEL, excluded; a long
    // water run that never clears the deck minimum is a ferry leg, excluded (a
    // road is never submerged). THEN bake the roadbed into the DEM: the paved band
    // is SET to the exact road surface (cut AND fill, dead flat across), so the
    // provider is a pure bicubic of the carved grid — C1 everywhere, no runtime
    // corridor warp — and the recentring physics terrain window inherits that flat
    // road bed automatically. Bridge/excluded spans never carve (deck spans the
    // ground; tunnels/ferries don't exist on the surface).
    road::RoadProfileOptions rpo;
    rpo.enabled = true;
    rpo.seaLevel = reg.seaLevel;
    // SMOOTH-ROAD alignment (ARC-LENGTH grade smoothing, density-independent — the
    // per-sample 1-2-1 pass can't do this, its reach scales with the sub-metre
    // spacing). A ~40 m window gives a flat, engineered vertical alignment instead
    // of a drape that follows the DEM's 5-30 m undulation and bounces the car at
    // speed. It is the grade of the SMOOTH ROAD RIBBON the car drives on (cooked as
    // one continuous C1 collider per run below) and of the baked bed under it, so
    // the whole road — geometry, physics, paint — is the one smooth surface.
    // ND_ROAD_SMOOTH=<metres> overrides (0 = raw DEM drape).
    rpo.gradeSmoothing = 40.f;
    if (const char* e = std::getenv("ND_ROAD_SMOOTH"); e && e[0] != '\0') rpo.gradeSmoothing = std::strtof(e, nullptr);
    network.conformTo([&pack](float x, float z) { return pack.grid.sampleBicubic(x, z); }, 14, rpo);
    terrain::RoadCarveOptions rco;
    rco.bakeSurface = true;
    // Defaults: full 6 m cut band (the tile-quad anti-poke guarantee) + narrow
    // asymmetric FILL, so the road hugs steep sidehills on a tight embankment.
    terrain::carveRoads(pack.grid, network, rco);

    terrain::GeoTerrainOptions gopt;
    gopt.snowHeightMin = std::max(reg.heightMax - 350.f, 900.f);
    gopt.grassHeightMax = std::clamp(reg.heightMin + 0.45f * (reg.heightMax - reg.heightMin), 200.f, 900.f);
    // Paint asphalt into the terrain albedo over the paved band — the baked bed IS
    // the road, so distant tiles (beyond the near-ribbon cull) carry a mip-filtered
    // painted road that cannot coverage-shimmer the way sub-pixel ribbon geometry
    // does. Near-tile splat texels are ~0.6-1.3 m; a feather below the texel size
    // reads as a staircase, so 1.2 m.
    gopt.paintRoads = true;
    gopt.roadEdgeFeather = 1.2f;
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
    // Capture forces the Vulkan deferred renderer (detail-map layer is Vulkan-only);
    // ND_CAPTURE_GL captures the forward GL path instead (to A/B renderer-specific
    // artifacts headlessly — the interactive default is GL). Interactive keeps the
    // renderer-select menu.
    auto renderer = !capturing ? createRenderer(canvas)
                    : createRenderer(canvas, std::getenv("ND_CAPTURE_GL") ? GraphicsAPI::OpenGL
                                                                          : GraphicsAPI::Vulkan);
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
        // ND_DENOISE=0 disables the deferred SVGF/temporal denoiser (A/B the
        // reflection/GI temporal ghost of the moving car — denoiser off = noisy but
        // ghost-free, the user's on/off knob for the artifact).
        if (const char* e = std::getenv("ND_DENOISE"); e && e[0] == '0') vk->setDenoise(false);
        // ND_NO_UPSCALE=1 drops DLSS/FSR to native TAA — stage bisect for
        // image-space ghosting (afterimages of the moving car).
        if (std::getenv("ND_NO_UPSCALE")) { vk->setDlss(false); vk->setFsr(false); }
    }
#endif
    // Tone mapping per backend: the forward GL path uses Neutral — its
    // ACESFilmic is the three.js implementation whose deliberate 1/0.6
    // viewing-environment gain pushes this bright scene (2.8 sun + full-sky
    // IBL, no GI/AO occlusion on the forward path) into the shoulder: the
    // asphalt turns pale gray and the grass flattens ("washed out" — the same
    // reason the plain drive demo runs Neutral). The Vulkan deferred path
    // keeps ACESFilmic: its physically-calibrated pipeline meters the same
    // scene correctly, and that is the validated look.
    renderer->toneMapping = ToneMapping::Neutral;
#ifdef THREEPP_WITH_VULKAN
    if (dynamic_cast<VulkanRenderer*>(renderer.get()))
        renderer->toneMapping = ToneMapping::ACESFilmic;
#endif
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

    // Roads are BAKED into the terrain (carve + paint above); only bridge decks
    // stay ribbon geometry. NEAR-DETAIL HYBRID (same as norway_terrain): on-ground
    // ribbon CHUNKS give stand-on detail (crisp edges, lane markings — the ~1 m
    // painted splat texels can't hold that) where the car is; each chunk is
    // distance-culled per frame below, and beyond the cull the painted+baked bed
    // alone is the road. The chunks + bridge decks double as the driving-surface
    // colliders (cooked once world exists, below) — the car rides the ribbon the
    // eye sees. ND_RIBBON_ROADS=1 restores the legacy full-ribbon look for A/B.
    const bool ribbonMode = std::getenv("ND_RIBBON_ROADS") != nullptr;
    std::shared_ptr<Group> roadRibbons;// full ribbons (ribbon mode) OR ground chunks (hybrid)
    std::shared_ptr<Group> bridges;    // bridge decks (hybrid mode only)
    std::vector<std::pair<Object3D*, Vector3>> roadChunkCenters;
    std::vector<float> roadChunkRadii;
    if (ribbonMode) {
        roadRibbons = network.buildMeshes();
    } else {
        bridges = network.buildBridgeMeshes();
        scene.add(bridges);
        roadRibbons = network.buildGroundChunkMeshes();
        for (auto* child : roadRibbons->children) {
            auto* mesh = child->as<Mesh>();
            if (!mesh) continue;
            auto geo = mesh->geometry();
            geo->computeBoundingSphere();
            roadChunkCenters.emplace_back(child, geo->boundingSphere->center);
            roadChunkRadii.push_back(geo->boundingSphere->radius);
        }
    }
    scene.add(roadRibbons);
    // ND_GLOSSY_ROAD: wet-look road (low roughness) so the moving car's REFLECTION
    // ghost is reproducible on the near ribbon (the default matte 0.95 asphalt
    // barely reflects, hiding the temporal artifact offline).
    if (std::getenv("ND_GLOSSY_ROAD"))
        roadRibbons->traverseType<Mesh>([](Mesh& m) {
            for (const auto& mat : m.materials())
                if (auto* sm = dynamic_cast<MeshStandardMaterial*>(mat.get())) {
                    sm->roughness = 0.12f;
                    sm->needsUpdate();
                }
        });
    const float ribbonDist = [] {
        const char* e = std::getenv("ND_ROAD_RIBBON_DIST");
        return e ? std::strtof(e, nullptr) : 600.f;// 6 m road ≈ 5 px here
    }();

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

    // Driving surface = ONE continuous SMOOTH RIBBON per ground run (buildGround-
    // ChunkMeshes with an effectively infinite chunk length → a single C1 piece
    // per run, NO chunk-boundary slope breaks) + the bridge decks. This is the
    // "smooth second geometry" the car rides: its grade is the arc-length-smoothed
    // alignment, so the ride is flat at speed regardless of the DEM underneath. The
    // distance-culled RENDER chunks (roadRibbons) are a SEPARATE build — visuals
    // only. The baked terrain sits kSurfaceRaise (0.12 m) below this ribbon (so it
    // never pokes through) and is the off-road catch + distant painted road.
    // In ribbon mode the full ribbon set is both collider and visual.
    std::shared_ptr<Group> collisionRibbons =
            ribbonMode ? roadRibbons : network.buildGroundChunkMeshes(1e9f);
    int roadColliders = 0, bridgeColliders = 0;
    collisionRibbons->traverseType<Mesh>([&](Mesh& m) {
        if (world.addStaticTrimesh(m)) ++roadColliders;
    });
    if (bridges)
        bridges->traverseType<Mesh>([&](Mesh& m) {
            if (world.addStaticTrimesh(m)) ++bridgeColliders;
        });

    // Buildings (packs fetched with --buildings): extruded OSM footprints with
    // nDSM-measured heights, batched into 500 m chunk meshes with per-building
    // vertex colours — same build as the norway_terrain viewer, plus a static
    // trimesh collider per chunk so the car hits walls instead of ghosting
    // through the village. ND_NO_BUILDINGS=1 hides them for A/B.
    if (!pack.buildings.empty() && !std::getenv("ND_NO_BUILDINGS")) {
        auto buildings = terrain::buildGeoBuildingMeshes(pack);
        int buildingColliders = 0;
        buildings->traverseType<Mesh>([&](Mesh& m) {
            if (world.addStaticTrimesh(m)) ++buildingColliders;
        });
        std::cout << "[drive] buildings: " << pack.buildings.size() << " footprints in "
                  << buildings->children.size() << " chunk meshes ("
                  << buildingColliders << " colliders)\n" << std::flush;
        scene.add(buildings);
    }

    // Terrain window collider (recentres with the car).
    TerrainWindow terrainWin(world, prov, /*size*/ 300.f, /*res*/ 150, /*recenter*/ 80.f);

    // ── spawn at the TOP of the longest DRIVABLE road stretch ────────────────────
    // The longest CONTIGUOUS non-excluded centerline run (not the longest road):
    // on coastal packs the longest road is often the ferry crossing, which
    // profile classification EXCLUDES — spawning on it drops the car in the fjord
    // (no collider there). This run is guaranteed to be real pavement end to end,
    // so both the spawn and the pure-pursuit auto-steer stay on the road.
    // The run is ordered TOP→BOTTOM (reversed if needed) and the car starts ~50 m
    // down from the crest, heading down-route — on Trollstigen that is the top of
    // the pass with the hairpin descent ahead.
    std::vector<Vector3> centerline = network.longestDrivableRun();
    if (centerline.size() >= 2 && centerline.front().y < centerline.back().y)
        std::reverse(centerline.begin(), centerline.end());
    Vector3 roadMid, roadDir;
    if (centerline.size() >= 2) {
        size_t si = 0;
        float acc = 0.f;
        while (si + 1 < centerline.size() && acc < 50.f) {
            acc += centerline[si].distanceTo(centerline[si + 1]);
            ++si;
        }
        roadMid = centerline[si];
        const Vector3 a = centerline.front(), b = centerline.back();
        roadDir.set(b.x - a.x, 0.f, b.z - a.z);
        if (roadDir.length() < 1e-4f) roadDir.set(0.f, 0.f, 1.f);
        else roadDir.normalize();
    } else {
        roadMid.set(0.f, reg.heightMin, 0.f);
        roadDir.set(0.f, 0.f, 1.f);
    }
    const float spawnSurf = [&] {
        const float s = network.surfaceHeight(roadMid.x, roadMid.z, roadMid.y);
        return std::isnan(s) ? roadMid.y : s;
    }();
    terrainWin.rebuild(roadMid.x, roadMid.z);// ground under the spawn before the first step

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
    // Tame the mirror-glossy GLB paint to a SATIN clearcoat. A near-mirror painted
    // panel is the worst case for the deferred reflection channel: screen-static
    // but world-moving under the chase cam, its 1-spp GGX reflection gets a short
    // history and boils / F0-tints (see deferred_shade's "MOTION GUARD"), and it
    // reads as "too glossy". Flooring roughness widens the BRDF lobe so the blur
    // absorbs the noise and the reflection is stable. Glass/lights are left alone
    // (MustangRig tags them); we only lift very-smooth painted metal.
    carModel->traverseType<Mesh>([](Mesh& mesh) {
        for (const auto& m : mesh.materials()) {
            if (auto* sm = dynamic_cast<MeshStandardMaterial*>(m.get());
                sm && sm->metalness > 0.4f && sm->roughness < 0.30f) {
                sm->roughness = 0.34f;
                sm->needsUpdate();
            }
        }
    });
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
    // Suspension TRAVEL. The old 0.16 (half the 0.32 default) was a firm sports-
    // coupe tune, but on a terrain-following mountain road at speed the short travel
    // BOTTOMS OUT over the road's mid-scale undulation and the car bounces. The
    // spring/damper themselves are well-tuned (~1.6 Hz, 0.66 damping ratio); it's
    // only the travel that was starving. More travel absorbs the undulation without
    // touching the road geometry (no launch artifacts). ND_SUSP_TRAVEL overrides.
    vs.suspensionTravelDist = 0.30f;
    if (const char* e = std::getenv("ND_SUSP_TRAVEL"); e && e[0] != '\0') vs.suspensionTravelDist = std::strtof(e, nullptr);
    vs.suspensionAttachmentY = carMeas.wheelCenterYRel + vs.suspensionTravelDist * 0.5f;
    // ARCADE handling (racing-game feel, not a sim). Lower the max front-wheel
    // angle (34° -> 24°) so the car turns in progressively instead of darting, and
    // raise tyre grip so it stays planted rather than sliding into understeer. The
    // per-frame steer slew + speed-sensitive scaling in the loop finish the job.
    vs.maxSteerAngleRad = 0.42f;
    if (const char* e = std::getenv("ND_STEER_MAX"); e && e[0] != '\0') vs.maxSteerAngleRad = std::strtof(e, nullptr);
    vs.tireFriction = 2.6f;
    vs.lateralStiffness = 130'000.f;// stickier lateral bite — less slide, more grip
    vs.spawnPosition = {roadMid.x, spawnSurf + 1.2f, roadMid.z};
    {
        // Align to the LOCAL road tangent at the spawn midpoint — the run's
        // end-to-start heading can be way off on a winding road, and a misaligned
        // spawn sends the car across the shoulder while the auto-steer acquires
        // the line (measured 40+ m/s^2 jolts). roadDir was overwritten with that
        // local tangent when ppIdx was seeded.
        Quaternion q;
        q.setFromUnitVectors(Vector3(0.f, 0.f, 1.f), roadDir);
        vs.spawnRotation = q;
    }
    PhysxVehicleEngineDrive vehicle(world, vs);

    auto carRig = std::make_unique<drive::MustangRig>(carModel, carMeas);
    scene.add(carRig->root());
    // DIAGNOSTIC (ND_BRIGHT_LIGHTS): force the tail/emissive materials bright so the
    // reflection/GI ghost of the MOVING car is visible in a daytime headless capture
    // (normally the 0.6-intensity tail-lights wash out and the artifact can't be
    // reproduced offline). Repro target for the moving-reflected-content history fix.
    if (std::getenv("ND_BRIGHT_LIGHTS")) {
        carRig->root()->traverseType<Mesh>([](Mesh& mesh) {
            for (const auto& m : mesh.materials())
                if (auto* sm = dynamic_cast<MeshStandardMaterial*>(m.get());
                    sm && sm->emissiveIntensity > 0.f) {
                    sm->emissiveIntensity = 20.f;
                    sm->needsUpdate();
                }
        });
    }
    world.bind(*carRig->root(), *vehicle.chassisActor());

    std::cout << "[drive] " << roadColliders << (ribbonMode ? " full-ribbon" : " ground-chunk")
              << " colliders + " << bridgeColliders << " bridge decks + recentring terrain window\n"
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
    RendererSettings settings(*renderer);
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
        ImGui::Separator();
        settings.drawCollapsed();
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
    std::string traceCsv = "frame,rawDtMs,renderX,renderY,renderZ,rawX,rawY,rawZ,screenX,screenY,physSpeed,camPitch\n";

    // The world is fully built: narrow the static vertex attributes. Terrain
    // ribbon, roads and the car shell qualify; ocean (DisplacedMesh), grass and
    // any skinned meshes are skipped automatically. Tiles the streamer creates
    // later arrive float — startup geometry is where the bulk of the bytes are.
    {
        const size_t saved = compressSceneAttributes(scene);
        std::cout << "[drive] compressed vertex attributes: "
                  << saved / (1024.0 * 1024.0) << " MiB reclaimed\n";
    }

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
            // ND_CRUISE overrides the target speed (to probe ride bounce at speed).
            static const float cruise = [] {
                const char* e = std::getenv("ND_CRUISE");
                return (e && e[0] != '\0') ? std::strtof(e, nullptr) : 32.f;
            }();// km/h
            autoThrottle = (speedKmh < cruise) ? 1.f : 0.1f;
            autoThrottle *= (1.f - 0.75f * steerMag);
            autoThrottle = std::max(autoThrottle, 0.25f);// always enough to keep crawling
        }

        // Arcade steering feel: fall the steer authority off faster with speed (so
        // fast corners need a deliberate input, not a twitch) and slew the command
        // in more gently (smooth turn-in, no darting). Low-speed hairpins keep near
        // full lock.
        const float steerScale = 1.f / (1.f + speedKmh * 0.03f);
        const float slew = std::min(1.f, dt * 2.2f);
        steerCmd += (steerInput * steerScale - steerCmd) * slew;

        // Forward / REVERSE. Auto-drive is always forward. In manual, S brakes while
        // still rolling forward, then engages REVERSE once nearly stopped — and in
        // reverse S becomes the (backward) throttle and W the brake; pressing W once
        // stopped re-selects Drive. Mirrors how an automatic gearbox feels.
        using Dir = PhysxVehicleEngineDrive::Direction;
        if (autoDrive) {
            throttleCmd = autoThrottle;
            brakeCmd = brakeDown ? 1.f : 0.f;
            // DIAGNOSTIC (ND_STOP_AT=<frame>): hard-stop the auto-drive at that
            // frame and hold the brake. A capture some frames later separates
            // MOTION-HISTORY residue (brightness displaced behind the now-parked
            // car — pure temporal artifact) from steady-state effects (bloom,
            // bounce — they sit ON/AROUND the car regardless of motion).
            static const long stopAt = [] {
                const char* e = std::getenv("ND_STOP_AT");
                return (e && e[0] != '\0') ? std::strtol(e, nullptr, 10) : -1L;
            }();
            if (stopAt >= 0 && frame >= stopAt) { throttleCmd = 0.f; brakeCmd = 1.f; }
        } else {
            const float fwd = vehicle.forwardSpeed();
            if (vehicle.direction() == Dir::Drive) {
                if (brakeDown && fwd < 0.5f) vehicle.setDirection(Dir::Reverse);
            } else if (throttleDown && fwd > -0.5f) {
                vehicle.setDirection(Dir::Drive);
            }
            const bool reversing = vehicle.direction() == Dir::Reverse;
            throttleCmd = reversing ? (brakeDown ? 1.f : 0.f) : (throttleDown ? 1.f : 0.f);
            brakeCmd = reversing ? (throttleDown ? 1.f : 0.f) : (brakeDown ? 1.f : 0.f);
        }
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

        // Chase camera in chassis-local space, transformed to world. Following the
        // car's full orientation (incl. pitch) is what lets you SEE down the road on
        // climbs and descents — essential on the hairpins; a world-level cam leaves
        // you driving blind over crests.
        carRig->root()->updateMatrixWorld();
        const Matrix4& chassisMat = *carRig->root()->matrixWorld;
        const float cosP = std::cos(orbitPitch);
        Vector3 desiredCam(orbitDist * std::sin(orbitYaw) * cosP,
                           orbitDist * std::sin(orbitPitch),
                           -orbitDist * std::cos(orbitYaw) * cosP);
        desiredCam.applyMatrix4(chassisMat);
        Vector3 desiredTarget(0.f, 1.1f, 1.5f);
        desiredTarget.applyMatrix4(chassisMat);
        if (topDown) {
            // Nearly overhead, slightly ahead, looking down-and-back: the car sits
            // upper-center with the road it JUST LEFT filling the lower frame —
            // the framing where trailing ghosts (stale history / afterimages of
            // the car at its previous positions) are visible. Uses the car's
            // heading so it tracks through turns.
            Vector3 fwd = Vector3(0.f, 0.f, 1.f).applyQuaternion(fromPxQuat(pose.q));
            fwd.y = 0.f;
            if (fwd.length() > 1e-4f) fwd.normalize();
            desiredCam = carPos + fwd * 4.f + Vector3(0.f, 14.f, 0.f);
            desiredTarget = carPos - fwd * 4.f + Vector3(0.f, 0.3f, 0.f);
        }
        // DIAGNOSTIC (ND_STATIC_CAM): pin the camera in WORLD space (elevated,
        // looking at a fixed road point set on the first capture frame) and let
        // the car auto-drive THROUGH the frame. Isolates the canonical hard case
        // for RT-temporal shadows — static camera + static ground + MOVING
        // occluder — from any chase-cam / raw-vs-interpolated-pose motion. If the
        // shadow trails HERE it is purely the renderer's temporal reprojection
        // (the moving shadow has no motion vector on the static ground); if it is
        // clean here but trails in chase, the cause is the pose/camera path.
        static bool  staticCam = std::getenv("ND_STATIC_CAM") != nullptr;
        static bool  staticSet = false;
        static Vector3 staticEye, staticTgt;
        if (staticCam) {
            if (!staticSet) {
                Vector3 fwd = Vector3(0.f, 0.f, 1.f).applyQuaternion(fromPxQuat(pose.q));
                fwd.y = 0.f; if (fwd.length() > 1e-4f) fwd.normalize();
                // Eye beside/above the road; look at the road ~25 m ahead so the
                // car drives from far → near → past under a fixed view.
                staticTgt = carPos + fwd * 25.f;
                staticEye = carPos + fwd * 10.f + Vector3(0.f, 9.f, 0.f)
                          + Vector3(fwd.z, 0.f, -fwd.x) * 6.f;// 6 m to the side
                staticSet = true;
            }
            camera->position.copy(staticEye);
            camera->lookAt(staticTgt);
        } else {
        // Frame-rate-independent smoothing (default): min(1,dt·k) is linear in dt,
        // so its effective time-constant drifts with the frame rate — under variable
        // dt the camera lag jitters and the whole world appears to shimmy even when
        // the car itself is smooth. 1-exp(-k·dt) has a fixed time-constant (1/k s).
        const float lerp = camLinear ? std::min(1.f, dt * 5.f) : (1.f - std::exp(-5.f * dt));
        camPos.lerp(desiredCam, lerp);
        camTarget.lerp(desiredTarget, lerp);
        camera->position.copy(camPos);
        camera->lookAt(camTarget);
        }

        // Near-detail ribbon chunks: visible within ribbonDist of the camera (10%
        // hysteresis so a boundary chunk doesn't flip every frame). Beyond it the
        // baked+painted roadbed alone carries the road. Rendering only — the chunk
        // COLLIDERS stay live regardless, so the car keeps its driving surface.
        for (size_t ci = 0; ci < roadChunkCenters.size(); ++ci) {
            auto* obj = roadChunkCenters[ci].first;
            const float d = camera->position.distanceTo(roadChunkCenters[ci].second) -
                            roadChunkRadii[ci];
            if (obj->visible) {
                if (d > ribbonDist * 1.1f) obj->visible = false;
            } else if (d < ribbonDist) {
                obj->visible = true;
            }
        }

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
            // Camera look-PITCH (rad): how far up/down the view is aimed. Its
            // frame-to-frame swing IS the "view rock" the rigid cam adds over
            // undulations; the level cam holds it near-constant.
            Vector3 look = camTarget.clone().sub(camPos);
            const float lookLen = std::max(look.length(), 1e-4f);
            const float camPitch = std::asin(std::clamp(look.y / lookLen, -1.f, 1.f));
            char line[288];
            std::snprintf(line, sizeof(line),
                          "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.5f\n",
                          frame, rawDt * 1000.f, rpos.x, rpos.y, rpos.z,
                          rawp.p.x, rawp.p.y, rawp.p.z, ndc.x, ndc.y, vehicle.forwardSpeed(), camPitch);
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
            // DIAGNOSTIC (ND_DUMP_FROM=<frame>): dump EVERY frame from there to the
            // end into aaa_caps/seq/ — a temporal artifact (history smear trailing a
            // moving object) is invisible in a single still; a consecutive-frame
            // sequence lets offline analysis image the residue directly.
            {
                static const long dumpFrom = [] {
                    const char* e = std::getenv("ND_DUMP_FROM");
                    return (e && e[0] != '\0') ? std::strtol(e, nullptr, 10) : -1L;
                }();
                if (dumpFrom >= 0 && frame >= dumpFrom) {
                    const auto seqDir = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / "seq";
                    std::filesystem::create_directories(seqDir);
                    renderer->writeFramebuffer(seqDir / (std::string(std::getenv("ND_DUMP_TAG") ? std::getenv("ND_DUMP_TAG") : "f")
                                                          + "_" + std::to_string(frame) + ".png"));
                }
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
