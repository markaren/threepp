// Procedural driving demo.
//
// A car designed from the ground up (CarRig.hpp — primitives, visible coil-over
// shock absorbers, working lights) on the full PxVehicle2 engine-drive stack
// (PhysxVehicleEngineDrive — engine + clutch + gearbox + autobox + differential),
// driving a procedurally generated world: rolling terrain (TerrainGenerator) cut
// through by a point-to-point road (RoadGenerator), lined with an instanced
// forest, grass and wildflowers. Backend-flexible via createRenderer; looks best
// on the Vulkan path tracer (aerial haze + real headlight cones at night).
//
// Controls:
//   W / S      throttle / brake        A / D   steer
//   SPACE      handbrake               R       toggle Drive / Reverse
//   N          neutral                 T       auto / manual gearbox
//   Q / E      shift down / up (manual)
//   Z / C      left / right indicator  L       headlights
//   H          horn                    F       day / night
//   V          driver view             Backspace  respawn

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxDebugRenderer.hpp"
#include "threepp/extras/physx/PhysxVehicleEngineDrive.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/road/RoadGenerator.hpp"
#include "threepp/extras/terrain/TerrainGenerator.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/textures/DataTexture.hpp"

#include "CarRig.hpp"
#include "DriveSounds.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    // ── Vegetation prototypes (same recipe as the forest demo) ────────────────
    struct TreeVariant {
        std::shared_ptr<BufferGeometry> trunkGeo, leafGeo;
        std::shared_ptr<MeshStandardMaterial> barkMat, leafMat;
    };

    TreeVariant makeVariant(int preset, unsigned int seed) {
        vegetation::TreeParams tp;
        vegetation::applyPreset(preset, tp);
        tp.seed = seed;
        if (preset == 2) {
            tp.barkColor = {0.70f, 0.69f, 0.66f};
            tp.leafDensity = 0.97f;
            tp.leafClumping = 0.35f;
        }
        vegetation::TreeGenerator gen(seed);
        gen.buildSkeleton(tp);
        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);
        auto bark = vegetation::makeBarkTextures(256, seed, tp.barkColor);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;
        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        v.leafMat->map = vegetation::makeLeafClusterTexture(256, seed, tp.leafColor);
        v.leafMat->alphaTest = 0.5f;
        v.leafMat->side = Side::Double;
        v.leafMat->vertexColors = true;
        return v;
    }

    std::shared_ptr<BufferGeometry> makeGrassBlade() {
        constexpr int seg = 4;
        constexpr float wBase = 0.05f;
        const Vector3 bottom{0.06f, 0.13f, 0.04f};
        const Vector3 top{0.20f, 0.34f, 0.11f};
        std::vector<float> pos, nrm, uv, col;
        std::vector<unsigned int> idx;
        for (int i = 0; i <= seg; ++i) {
            const float t = static_cast<float>(i) / seg;
            const float y = t, w = wBase * (1.f - t);
            const float r = bottom.x + (top.x - bottom.x) * t;
            const float g = bottom.y + (top.y - bottom.y) * t;
            const float b = bottom.z + (top.z - bottom.z) * t;
            for (int s = 0; s < 2; ++s) {
                pos.push_back(s == 0 ? -w : w);
                pos.push_back(y);
                pos.push_back(0.f);
                nrm.push_back(0.f); nrm.push_back(0.85f); nrm.push_back(0.53f);
                uv.push_back(s == 0 ? 0.f : 1.f); uv.push_back(t);
                col.push_back(r); col.push_back(g); col.push_back(b);
            }
        }
        for (int i = 0; i < seg; ++i) {
            const auto a = static_cast<unsigned int>(i * 2);
            const unsigned int b = a + 1, c = a + 2, d = a + 3;
            idx.push_back(a); idx.push_back(b); idx.push_back(c);
            idx.push_back(b); idx.push_back(d); idx.push_back(c);
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        return geo;
    }

    std::shared_ptr<BufferGeometry> makeFlowerCard() {
        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        const float hw = 0.5f;
        const Vector3 up{0.f, 1.f, 0.f};
        auto addQuad = [&](const Vector3& right, const Vector3& face) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            Vector3 n;
            n.copy(face).multiplyScalar(0.4f).add(up).normalize();
            Vector3 c[4];
            c[0].set(0.f, 0.f, 0.f).addScaledVector(right, -hw);
            c[1].set(0.f, 0.f, 0.f).addScaledVector(right, hw);
            c[2].set(0.f, 1.f, 0.f).addScaledVector(right, hw);
            c[3].set(0.f, 1.f, 0.f).addScaledVector(right, -hw);
            const float us[4] = {0.f, 1.f, 1.f, 0.f};
            const float vs[4] = {0.f, 0.f, 1.f, 1.f};
            for (int i = 0; i < 4; ++i) {
                pos.push_back(c[i].x); pos.push_back(c[i].y); pos.push_back(c[i].z);
                nrm.push_back(n.x); nrm.push_back(n.y); nrm.push_back(n.z);
                uv.push_back(us[i]); uv.push_back(vs[i]);
            }
            idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
            idx.push_back(base); idx.push_back(base + 2); idx.push_back(base + 3);
        };
        addQuad(Vector3(1.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f));
        addQuad(Vector3(0.f, 0.f, 1.f), Vector3(1.f, 0.f, 0.f));
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        return geo;
    }

    std::shared_ptr<BufferGeometry> makeRock(unsigned int seed) {
        constexpr int latSegs = 5, lonSegs = 7;
        constexpr float PI = 3.14159265358979f;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-PI, PI);
        const float p1 = u(rng), p2 = u(rng), p3 = u(rng);
        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        for (int lat = 0; lat <= latSegs; ++lat) {
            const float theta = static_cast<float>(lat) / latSegs * PI;
            const float sinT = std::sin(theta), cosT = std::cos(theta);
            for (int lon = 0; lon <= lonSegs; ++lon) {
                const float phi = static_cast<float>(lon) / lonSegs * 2.f * PI;
                const float nx = sinT * std::cos(phi), ny = cosT, nz = sinT * std::sin(phi);
                float disp = 1.f + 0.30f * std::sin(2.f * phi + p1) * sinT +
                             0.24f * std::cos(3.f * phi + p2) +
                             0.22f * std::sin(3.f * theta + p3);
                disp = std::clamp(disp, 0.6f, 1.5f);
                pos.push_back(nx * disp); pos.push_back(ny * disp); pos.push_back(nz * disp);
                nrm.push_back(nx); nrm.push_back(ny); nrm.push_back(nz);
                uv.push_back(static_cast<float>(lon) / lonSegs);
                uv.push_back(static_cast<float>(lat) / latSegs);
            }
        }
        const int rowVerts = lonSegs + 1;
        for (int lat = 0; lat < latSegs; ++lat)
            for (int lon = 0; lon < lonSegs; ++lon) {
                const auto a = static_cast<unsigned int>(lat * rowVerts + lon);
                const auto b = static_cast<unsigned int>(a + rowVerts);
                idx.push_back(a); idx.push_back(a + 1); idx.push_back(b);
                idx.push_back(a + 1); idx.push_back(b + 1); idx.push_back(b);
            }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        return geo;
    }

    float smooth01(float e0, float e1, float x) {
        const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }

}// namespace

int main(int argc, char** argv) {

    // Headless capture: `drive --shot out.png [--frames N] [--night]` auto-drives
    // for N frames, writes one PNG, and exits. Used for verification screenshots.
    std::string shotPath;
    int shotFrames = 200;
    bool shotNight = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) shotPath = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--night") == 0) shotNight = true;
    }
    const bool capturing = !shotPath.empty();
    int shotFrame = 0;

    Canvas canvas("Procedural Drive", {{"vsync", true}, {"aa", 4}});
    auto renderer = createRenderer(canvas);

    bool vulkanBackend = false;
#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
    if (vk) {
        vulkanBackend = true;
        vk->setRenderScale(0.8f);
    }
#endif
    const bool isGL = !vulkanBackend && dynamic_cast<GLRenderer*>(renderer.get()) != nullptr;

    renderer->setClearColor(Color(0.62f, 0.72f, 0.84f));
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 1.0f;
    renderer->shadowMap().enabled = true;
    renderer->shadowMap().type = ShadowMap::PFCSoft;

    auto scene = Scene::create();

    // ── Sky + sun ─────────────────────────────────────────────────────────
    RGBELoader hdrLoader;
    std::shared_ptr<Texture> hdr =
            hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr");
    if (hdr) {
        scene->background = hdr;
        scene->environment = hdr;
    }

    auto sun = DirectionalLight::create(Color(1.0f, 0.97f, 0.90f), 2.8f);
    sun->position.set(180.f, 220.f, 120.f);
    sun->castShadow = true;
    {
        auto* cam = sun->shadow->camera->as<OrthographicCamera>();
        cam->left = cam->bottom = -120.f;
        cam->right = cam->top = 120.f;
        cam->nearPlane = 1.f;
        cam->farPlane = 700.f;
        sun->shadow->mapSize.set(4096, 4096);
        sun->shadow->bias = -0.0005f;
    }
    scene->add(sun);

    auto ambient = AmbientLight::create(Color::white, 0.25f);
    if (!vulkanBackend) scene->add(ambient);

    // ── Physics ───────────────────────────────────────────────────────────
    PhysxWorld world;

    // ── Terrain ───────────────────────────────────────────────────────────
    terrain::TerrainParams terr;
    terr.seed = 4242u;
    terr.worldSize = 620.f;
    terr.resolution = 240;
    terr.noiseType = terrain::NoiseType::fBm;// no erosion => heightAt matches mesh
    terr.featureScale = 260.f;
    terr.octaves = 6;
    terr.amplitude = 26.f;
    terr.warp = 0.3f;
    terr.heightExponent = 1.0f;
    terr.erosion = terrain::ErosionType::None;
    terr.grassColor = {0.26f, 0.33f, 0.16f};
    terr.screeColor = {0.45f, 0.42f, 0.34f};
    terr.rockColor = {0.40f, 0.37f, 0.33f};
    terr.snowLine = 3.0f;// effectively no snow
    terr.slopeGrassMax = 0.5f;

    terrain::TerrainGenerator terrainGen(terr.seed);
    terrainGen.buildField(terr);// populate field for heightAt + splat bake

    // ── Road: a point-to-point route wandering across the terrain ──────────
    road::RoadParams roadParams;
    roadParams.laneWidth = 3.4f;
    roadParams.laneCount = 2;
    roadParams.shoulderWidth = 2.6f;
    roadParams.surfaceRaise = 0.12f;
    roadParams.maxBanking = 0.03f;// keep < raise/pavedHalf so curves don't poke terrain
    roadParams.samplesPerSegment = 26;

    std::vector<Vector3> roadPoints;
    {
        const float half = terr.worldSize * 0.5f * 0.82f;
        const Vector3 a(-half, 0.f, -half * 0.85f);
        const Vector3 b(half, 0.f, half * 0.75f);
        const int N = 7;
        // Perpendicular to the a→b direction, in XZ.
        Vector3 dir = b.clone().sub(a);
        dir.y = 0.f;
        dir.normalize();
        const Vector3 perp(-dir.z, 0.f, dir.x);
        std::mt19937 rng(7u);
        std::uniform_real_distribution<float> j(-1.f, 1.f);
        for (int i = 0; i < N; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(N - 1);
            Vector3 base = a.clone().lerp(b, t);
            const float wander = std::sin(t * math::PI * 1.7f) * (terr.worldSize * 0.14f) +
                                 j(rng) * (terr.worldSize * 0.04f);
            base.add(perp.clone().multiplyScalar(wander));
            roadPoints.push_back(base);
        }
    }

    road::RoadGenerator roadGen(roadPoints, roadParams);
    roadGen.conformTo([&](float x, float z) { return terrainGen.heightAt(x, z, terr); }, 16);

    const float corridorHalf = roadGen.corridorHalfWidth();
    const float flattenMargin = 8.f;
    // Unified ground height: flatten the full road corridor to the road
    // elevation, then grade back to natural terrain over `flattenMargin`.
    auto groundHeight = [&](float x, float z, float th) {
        const float d = roadGen.distanceToCenter(x, z);
        if (d >= corridorHalf + flattenMargin) return th;
        const float ch = roadGen.centerHeightAt(x, z);
        const float w = 1.f - smooth01(corridorHalf, corridorHalf + flattenMargin, d);
        return th + (ch - th) * w;
    };
    // Placement height = ground (terrain blended into the road corridor).
    auto gh = [&](float x, float z) { return groundHeight(x, z, terrainGen.heightAt(x, z, terr)); };
    // A degenerate (zero-scale) matrix to hide unused instances off-screen.
    Matrix4 hiddenMatrix;
    hiddenMatrix.compose(Vector3(0.f, -9999.f, 0.f), Quaternion(), Vector3(0.f, 0.f, 0.f));

    // Terrain mesh, displaced by the unified ground height.
    auto terrainMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color::white).roughness(0.97f).metalness(0.f));
    std::shared_ptr<BufferGeometry> terrainGeo =
            PlaneGeometry::create(terr.worldSize, terr.worldSize,
                                  static_cast<unsigned int>(terr.resolution),
                                  static_cast<unsigned int>(terr.resolution));
    terrainGeo->rotateX(-math::PI * 0.5f);
    {
        auto* pos = terrainGeo->getAttribute<float>("position");
        auto& a = pos->array();
        for (int i = 0; i < pos->count(); ++i) {
            const float x = a[i * 3 + 0], z = a[i * 3 + 2];
            a[i * 3 + 1] = groundHeight(x, z, terrainGen.heightAt(x, z, terr));
        }
        pos->needsUpdate();
        terrainGeo->computeVertexNormals();
        terrainGeo->computeBoundingBox();
        terrainGeo->computeBoundingSphere();
    }
    {
        auto tex = DataTexture::create(ImageData{terrainGen.bakeSplatColors(terr)},
                                       static_cast<unsigned int>(terrainGen.dim()),
                                       static_cast<unsigned int>(terrainGen.dim()));
        tex->colorSpace = ColorSpace::sRGB;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        terrainMat->map = tex;
    }
    auto terrainMesh = Mesh::create(terrainGeo, terrainMat);
    terrainMesh->receiveShadow = true;
    scene->add(terrainMesh);
    world.addStaticTrimesh(*terrainGeo);// drivable collider

    // ── Road surface ───────────────────────────────────────────────────────
    auto roadGeo = roadGen.buildSurface();
    auto roadMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
    {
        const int rw = 128, rh = 256;
        auto tex = DataTexture::create(ImageData{roadGen.bakeSurfaceTexture(rw, rh)},
                                       static_cast<unsigned int>(rw), static_cast<unsigned int>(rh));
        tex->colorSpace = ColorSpace::sRGB;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::Linear;
        tex->wrapS = TextureWrapping::ClampToEdge;
        tex->wrapT = TextureWrapping::Repeat;// road tiles along its length
        roadMat->map = tex;
    }
    auto roadMesh = Mesh::create(roadGeo, roadMat);
    roadMesh->receiveShadow = true;
    scene->add(roadMesh);
    world.addStaticTrimesh(*roadGeo);

    // ── Vehicle (engine drive) ──────────────────────────────────────────────
    PhysxVehicleEngineDrive::Settings vs;
    vs.chassisWidth = 1.9f;
    vs.chassisHeight = 1.3f;
    vs.chassisLength = 4.4f;
    vs.wheelRadius = 0.36f;
    vs.wheelHalfWidth = 0.16f;
    vs.trackWidth = 1.66f;
    vs.wheelbase = 2.7f;
    vs.suspensionAttachmentY = -0.35f;
    vs.suspensionTravelDist = 0.32f;
    {
        const Vector3 startP = roadGen.startPoint();
        Vector3 fwd = roadGen.startForward();
        vs.spawnPosition = {startP.x, startP.y + 1.4f, startP.z};
        Quaternion q;
        q.setFromUnitVectors(Vector3(0.f, 0.f, 1.f), fwd);
        vs.spawnRotation = q;
    }
    PhysxVehicleEngineDrive vehicle(world, vs);

    drive::CarRig::Config carCfg;
    carCfg.bodyWidth = vs.chassisWidth;
    carCfg.bodyHeight = vs.chassisHeight;
    carCfg.bodyLength = vs.chassisLength;
    carCfg.wheelRadius = vs.wheelRadius;
    carCfg.wheelHalfWidth = vs.wheelHalfWidth;
    carCfg.trackWidth = vs.trackWidth;
    carCfg.wheelbase = vs.wheelbase;
    carCfg.suspensionAttachmentY = vs.suspensionAttachmentY;
    carCfg.suspensionTravelDist = vs.suspensionTravelDist;
    auto carRig = std::make_unique<drive::CarRig>(carCfg);
    scene->add(carRig->root());
    world.bind(*carRig->root(), *vehicle.chassisActor());

    // Driver POV camera, parented to the car body.
    auto povCamera = PerspectiveCamera::create(72.f, canvas.aspect(), 0.05f, 2000.f);
    povCamera->position.set(0.34f, 0.55f, -0.15f);
    povCamera->rotation.y = math::PI;// car forward is +Z; flip to look ahead
    carRig->root()->add(povCamera);

    auto physxDebug = std::make_shared<PhysxDebugRenderer>(world);
    physxDebug->enableDefaults();
    physxDebug->visible = false;
    scene->add(physxDebug);

    // ── Scatter: forest over the terrain, minus the road corridor ───────────
    auto slopeNy = [&](float x, float z) {
        const float e = 1.5f;
        const float hx = terrainGen.heightAt(x + e, z, terr) - terrainGen.heightAt(x - e, z, terr);
        const float hz = terrainGen.heightAt(x, z + e, terr) - terrainGen.heightAt(x, z - e, terr);
        return (2.f * e) / std::sqrt(hx * hx + hz * hz + (2.f * e) * (2.f * e));
    };
    const float treeClear = corridorHalf + 5.f;// keep trunks off the verge

    std::vector<TreeVariant> variants{
            makeVariant(0, 101u), makeVariant(0, 202u),
            makeVariant(1, 303u), makeVariant(1, 505u), makeVariant(2, 404u)};
    {
        const float half = terr.worldSize * 0.5f * 0.94f;
        const float spacing = 13.f;
        const int cells = std::max(1, static_cast<int>((2.f * half) / spacing));
        std::vector<std::vector<Matrix4>> xf(variants.size());
        std::mt19937 rng(1u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        Quaternion q;
        const Vector3 up{0.f, 1.f, 0.f};
        for (int cz = 0; cz < cells; ++cz)
            for (int cx = 0; cx < cells; ++cx) {
                if (u01(rng) > 0.72f) continue;
                const float jx = (u01(rng) - 0.5f) * spacing * 0.8f;
                const float jz = (u01(rng) - 0.5f) * spacing * 0.8f;
                const float x = -half + (cx + 0.5f) * (2.f * half / cells) + jx;
                const float z = -half + (cz + 0.5f) * (2.f * half / cells) + jz;
                if (roadGen.distanceToCenter(x, z) < treeClear) continue;
                if (slopeNy(x, z) < 0.86f) continue;
                const size_t vi = static_cast<size_t>(u01(rng) * variants.size()) % variants.size();
                const float s = 1.1f + u01(rng) * 1.1f;
                q.setFromAxisAngle(up, u01(rng) * 6.2831853f);
                const float gy = gh(x, z);
                Matrix4 m;
                m.compose(Vector3(x, gy - 0.2f, z), q, Vector3(s, s, s));
                xf[vi].push_back(m);
                // Trunk collider (tall thin box rising from the ground) so the
                // car bumps into trees instead of driving through them.
                world.addStatic(PxBoxGeometry(0.22f * s, 1.5f * s, 0.22f * s),
                                toPxTransform(Vector3(x, gy + 1.5f * s - 0.2f, z)));
            }
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            if (xf[vi].empty()) continue;
            auto trunks = InstancedMesh::create(variants[vi].trunkGeo, variants[vi].barkMat, xf[vi].size());
            auto leaves = InstancedMesh::create(variants[vi].leafGeo, variants[vi].leafMat, xf[vi].size());
            trunks->castShadow = trunks->receiveShadow = true;
            leaves->castShadow = true;
            for (size_t i = 0; i < xf[vi].size(); ++i) {
                trunks->setMatrixAt(i, xf[vi][i]);
                leaves->setMatrixAt(i, xf[vi][i]);
            }
            trunks->instanceMatrix()->needsUpdate();
            leaves->instanceMatrix()->needsUpdate();
            scene->add(trunks);
            scene->add(leaves);
        }
    }

    // ── Grass + wildflowers, banded along the road (cheap, lines the route) ──
    const auto& centerline = roadGen.centerlineSamples();
    auto scatterAlongRoad = [&](int count, unsigned seed, float bandInner, float bandOuter,
                                const std::function<void(float, float, float)>& place) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        if (centerline.size() < 2) return;
        for (int i = 0; i < count; ++i) {
            const size_t si = static_cast<size_t>(u01(rng) * (centerline.size() - 1));
            const Vector3& c = centerline[si];
            const float side = (u01(rng) < 0.5f) ? -1.f : 1.f;
            const float dist = bandInner + u01(rng) * (bandOuter - bandInner);
            // Local perpendicular from the polyline segment.
            const Vector3& c2 = centerline[std::min(si + 1, centerline.size() - 1)];
            Vector3 t = c2.clone().sub(c);
            t.y = 0.f;
            if (t.length() < 1e-4f) t = Vector3(0.f, 0.f, 1.f);
            t.normalize();
            const float px = c.x - t.z * side * dist;
            const float pz = c.z + t.x * side * dist;
            place(px, pz, gh(px, pz));
        }
    };

    {
        auto grassMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.97f).metalness(0.f));
        grassMat->vertexColors = true;
        grassMat->side = Side::Double;
        grassMat->envMapIntensity = 0.45f;
        const int bladeCount = vulkanBackend ? 14000 : 42000;
        auto grass = InstancedMesh::create(makeGrassBlade(), grassMat, static_cast<size_t>(bladeCount));
        std::mt19937 rng(7u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const Vector3 up{0.f, 1.f, 0.f};
        Quaternion q;
        Matrix4 m;
        int idx = 0;
        scatterAlongRoad(bladeCount, 7u, corridorHalf + 0.3f, corridorHalf + 22.f,
                         [&](float x, float z, float h) {
                             if (idx >= bladeCount) return;
                             const float s = 0.5f + u01(rng) * 0.5f;
                             const float hgt = 0.3f + u01(rng) * 0.4f;
                             q.setFromAxisAngle(up, u01(rng) * 6.2831853f);
                             m.compose(Vector3(x, h - 0.05f, z), q, Vector3(s, hgt, s));
                             grass->setMatrixAt(static_cast<size_t>(idx++), m);
                         });
        for (; idx < bladeCount; ++idx) grass->setMatrixAt(static_cast<size_t>(idx), hiddenMatrix);
        grass->instanceMatrix()->needsUpdate();
        scene->add(grass);
    }

    {
        const int perVariant = vulkanBackend ? 500 : 1100;
        const Vector3 up{0.f, 1.f, 0.f};
        for (int fv = 0; fv < 3; ++fv) {
            auto mat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color::white).roughness(0.9f).metalness(0.f));
            mat->map = vegetation::makeFlowerTexture(128, static_cast<unsigned int>(1 + fv));
            mat->alphaTest = 0.5f;
            mat->side = Side::Double;
            mat->envMapIntensity = 0.6f;
            auto fm = InstancedMesh::create(makeFlowerCard(), mat, static_cast<size_t>(perVariant));
            std::mt19937 rng(static_cast<unsigned>(200 + fv));
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            Quaternion q;
            Matrix4 m;
            int idx = 0;
            scatterAlongRoad(perVariant, static_cast<unsigned>(200 + fv), corridorHalf + 0.5f, corridorHalf + 18.f,
                             [&](float x, float z, float h) {
                                 if (idx >= perVariant) return;
                                 const float s = 0.25f + u01(rng) * 0.25f;
                                 q.setFromAxisAngle(up, u01(rng) * 6.2831853f);
                                 m.compose(Vector3(x, h - 0.03f, z), q, Vector3(s, s * (1.f + u01(rng) * 0.6f), s));
                                 fm->setMatrixAt(static_cast<size_t>(idx++), m);
                             });
            for (; idx < perVariant; ++idx) fm->setMatrixAt(static_cast<size_t>(idx), hiddenMatrix);
            fm->instanceMatrix()->needsUpdate();
            scene->add(fm);
        }
    }

    // ── Rocks ───────────────────────────────────────────────────────────────
    {
        auto rockMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(0.30f, 0.28f, 0.25f)).roughness(1.f).metalness(0.f));
        rockMat->flatShading = true;
        rockMat->envMapIntensity = 0.3f;
        std::vector<std::shared_ptr<BufferGeometry>> rgeos{makeRock(1u), makeRock(2u), makeRock(3u)};
        std::vector<std::vector<Matrix4>> xf(rgeos.size());
        std::mt19937 rng(99u);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        const float half = terr.worldSize * 0.5f * 0.9f;
        Quaternion q;
        Vector3 axis;
        Matrix4 m;
        for (int i = 0; i < 120; ++i) {
            const float x = (u01(rng) - 0.5f) * 2.f * half;
            const float z = (u01(rng) - 0.5f) * 2.f * half;
            if (roadGen.distanceToCenter(x, z) < corridorHalf + 1.5f) continue;
            const float sc = 0.5f + u01(rng) * 1.6f;
            const float gy = gh(x, z);
            axis.set(u01(rng) - 0.5f, u01(rng) - 0.5f, u01(rng) - 0.5f).normalize();
            q.setFromAxisAngle(axis, u01(rng) * 6.2831853f);
            m.compose(Vector3(x, gy - sc * 0.25f, z), q,
                      Vector3(sc, sc * (0.75f + u01(rng) * 0.35f), sc));
            xf[static_cast<size_t>(u01(rng) * rgeos.size()) % rgeos.size()].push_back(m);
            // Boulder collider — a sphere the car hits (big rocks block, small ride over).
            world.addStatic(PxSphereGeometry(sc * 0.72f), toPxTransform(Vector3(x, gy - sc * 0.05f, z)));
        }
        for (size_t gi = 0; gi < rgeos.size(); ++gi) {
            if (xf[gi].empty()) continue;
            auto rocks = InstancedMesh::create(rgeos[gi], rockMat, xf[gi].size());
            rocks->castShadow = rocks->receiveShadow = true;
            for (size_t i = 0; i < xf[gi].size(); ++i) rocks->setMatrixAt(i, xf[gi][i]);
            rocks->instanceMatrix()->needsUpdate();
            scene->add(rocks);
        }
    }

    // ── Fog ─────────────────────────────────────────────────────────────────
    Color dayFog(0.66f, 0.75f, 0.86f);
    Color nightFog(0.04f, 0.05f, 0.09f);
    scene->fog = Fog(dayFog, terr.worldSize * 0.3f, terr.worldSize * 0.95f);

    // ── Cameras ─────────────────────────────────────────────────────────────
    auto camera = PerspectiveCamera::create(60.f, canvas.aspect(), 0.1f, 2000.f);
    camera->position.set(0, 6, -12);

    // ── Audio ───────────────────────────────────────────────────────────────
    drive::DriveSounds sounds;
    sounds.init();
    bool audioOn = true;
    float audioVol = 0.7f;

    // ── Input state ─────────────────────────────────────────────────────────
    bool throttleDown = false, brakeDown = false, handbrakeDown = false;
    bool steerLeftDown = false, steerRightDown = false;
    bool hornDown = false, driverView = false, respawn = false, night = shotNight;
    int turnSignal = 0;// -1 left, 0 none, +1 right

    auto keyToggle = [&](Key key, bool down) {
        switch (key) {
            case Key::W: case Key::UP: throttleDown = down; break;
            case Key::S: case Key::DOWN: brakeDown = down; break;
            case Key::A: case Key::LEFT: steerLeftDown = down; break;
            case Key::D: case Key::RIGHT: steerRightDown = down; break;
            case Key::SPACE: handbrakeDown = down; break;
            case Key::H: hornDown = down; break;
            default: break;
        }
    };

    KeyAdapter pressAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        switch (evt.key) {
            case Key::R:
                vehicle.setDirection(vehicle.direction() == PhysxVehicleEngineDrive::Direction::Reverse
                                             ? PhysxVehicleEngineDrive::Direction::Drive
                                             : PhysxVehicleEngineDrive::Direction::Reverse);
                break;
            case Key::N: vehicle.setDirection(PhysxVehicleEngineDrive::Direction::Neutral); break;
            case Key::T:
                vehicle.setTransmissionMode(
                        vehicle.transmissionMode() == PhysxVehicleEngineDrive::TransmissionMode::Automatic
                                ? PhysxVehicleEngineDrive::TransmissionMode::Manual
                                : PhysxVehicleEngineDrive::TransmissionMode::Automatic);
                break;
            case Key::E: vehicle.shiftUp(); break;
            case Key::Q: vehicle.shiftDown(); break;
            case Key::L: carRig->setHeadlights(!carRig->headlightsOn()); break;
            case Key::Z: turnSignal = (turnSignal == -1) ? 0 : -1; break;
            case Key::C: turnSignal = (turnSignal == 1) ? 0 : 1; break;
            case Key::V: driverView = !driverView; break;
            case Key::F: night = !night; break;
            case Key::BACKSPACE: respawn = true; break;
            default: break;
        }
        keyToggle(evt.key, true);
    });
    KeyAdapter releaseAdapter(KeyAdapter::KEY_RELEASED, [&](KeyEvent evt) { keyToggle(evt.key, false); });
    canvas.addKeyListener(pressAdapter);
    canvas.addKeyListener(releaseAdapter);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    // ── HUD ─────────────────────────────────────────────────────────────────
    float steerCmd = 0.f, throttleCmd = 0.f, brakeCmd = 0.f;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float w = 290 * ui.dpiScale();
        ImGui::SetNextWindowPos({static_cast<float>(canvas.size().width()) - w, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({w, 0}, 0);
        ImGui::Begin("Procedural Drive");
        ImGui::Text("W/S throttle/brake  A/D steer");
        ImGui::Text("R Drive/Rev  N neutral  SPACE handbrake");
        ImGui::Text("T auto/manual  Q/E shift  L lights");
        ImGui::Text("Z/C blinkers  H horn  F day/night  V POV");
        ImGui::Separator();
        ImGui::Text("FPS   : %.0f", fps);
        ImGui::Text("Speed : %.0f km/h", std::abs(vehicle.forwardSpeed()) * 3.6f);
        ImGui::Text("Gear  : %s   Engine: %.0f rpm", vehicle.gearLabel().c_str(), vehicle.engineRpm());
        const char* dirTxt = vehicle.direction() == PhysxVehicleEngineDrive::Direction::Drive ? "DRIVE"
                             : vehicle.direction() == PhysxVehicleEngineDrive::Direction::Reverse ? "REVERSE"
                                                                                                  : "NEUTRAL";
        ImGui::Text("Mode  : %s  %s", dirTxt,
                    vehicle.transmissionMode() == PhysxVehicleEngineDrive::TransmissionMode::Automatic ? "(auto)" : "(manual)");
        ImGui::ProgressBar(vehicle.engineRpmFraction(), {-1, 0}, "RPM");
        ImGui::ProgressBar(throttleCmd, {-1, 0}, "Throttle");
        ImGui::ProgressBar(brakeCmd, {-1, 0}, "Brake");
        ImGui::Separator();
        ImGui::Checkbox("PhysX debug", &physxDebug->visible);
        if (sounds.ok) {
            ImGui::Checkbox("Audio", &audioOn);
            ImGui::SliderFloat("Volume", &audioVol, 0.f, 1.f, "%.2f");
        }
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        povCamera->aspect = size.aspect();
        povCamera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    // ── Chase camera state ──────────────────────────────────────────────────
    Vector3 camPos{0, 6, -12}, camTarget{0, 1, 0};

    bool prevNight = false;
    auto applyDayNight = [&] {
        if (night) {
            sun->intensity = 0.18f;
            sun->color = Color(0.5f, 0.6f, 0.85f);
            if (ambient) ambient->intensity = 0.05f;
            scene->background = nightFog;
            if (scene->fog) std::get<Fog>(*scene->fog) = Fog(nightFog, terr.worldSize * 0.18f, terr.worldSize * 0.7f);
            carRig->setHeadlights(true);
        } else {
            sun->intensity = 2.8f;
            sun->color = Color(1.0f, 0.97f, 0.90f);
            if (ambient) ambient->intensity = 0.25f;
            if (hdr) scene->background = hdr; else scene->background = dayFog;
            if (scene->fog) std::get<Fog>(*scene->fog) = Fog(dayFog, terr.worldSize * 0.3f, terr.worldSize * 0.95f);
        }
    };

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        fpsAccum += dt;
        if (++fpsFrames >= 30 || fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / std::max(fpsAccum, 1e-4f);
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (night != prevNight) {
            applyDayNight();
            prevNight = night;
        }

        // Headless capture auto-drives forward so the shot shows the car rolling.
        if (capturing) {
            throttleDown = true;
            if (shotNight) carRig->setHeadlights(true);
        }

        // Build commands.
        const float steerInput = (steerLeftDown ? 1.f : 0.f) - (steerRightDown ? 1.f : 0.f);
        const float speedKmhAbs = std::abs(vehicle.forwardSpeed()) * 3.6f;
        const float steerScale = 1.f / (1.f + speedKmhAbs * 0.015f);
        const float steerSlew = std::min(1.f, dt * 2.f);
        steerCmd += (steerInput * steerScale - steerCmd) * steerSlew;
        throttleCmd = throttleDown ? 1.f : 0.f;
        brakeCmd = brakeDown ? 1.f : 0.f;

        vehicle.setThrottle(throttleCmd);
        vehicle.setBrake(brakeCmd);
        vehicle.setHandbrake(handbrakeDown ? 1.f : 0.f);
        vehicle.setSteer(steerCmd);

        if (respawn) {
            respawn = false;
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
        }

        world.step(dt);
        physxDebug->update();
        carRig->update(vehicle, dt, brakeCmd, turnSignal);

        // Chase / POV camera.
        carRig->root()->updateMatrixWorld();
        const Matrix4& chassisMat = *carRig->root()->matrixWorld;
        Vector3 desiredCam(0.f, 4.2f, -11.f);
        desiredCam.applyMatrix4(chassisMat);
        Vector3 desiredTarget(0.f, 1.2f, 3.f);
        desiredTarget.applyMatrix4(chassisMat);
        const float lerp = std::min(1.f, dt * 5.f);
        camPos.lerp(desiredCam, lerp);
        camTarget.lerp(desiredTarget, lerp);
        camera->position.copy(camPos);
        camera->lookAt(camTarget);

        Camera& activeCamera = driverView ? static_cast<Camera&>(*povCamera) : static_cast<Camera&>(*camera);

        sounds.update(dt, vehicle, throttleCmd, hornDown, activeCamera, audioOn ? audioVol : 0.f);

        renderer->render(*scene, activeCamera);

        if (capturing) {
            if (shotFrame % 60 == 0 || shotFrame + 1 >= shotFrames)
                std::cout << "[t=" << shotFrame << "] speed=" << std::abs(vehicle.forwardSpeed()) * 3.6f
                          << " km/h  rpm=" << vehicle.engineRpm()
                          << "  gear=" << vehicle.gearLabel() << "\n";
            if (++shotFrame >= shotFrames) {
                renderer->writeFramebuffer(shotPath);
                std::cout << "wrote " << shotPath << "\n";
                std::exit(0);
            }
        }

        ui.render();
    });

    return 0;
}
