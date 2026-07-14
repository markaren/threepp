// ============================================================================
//  threepp First-Person Shooter — an engine-blockout ("UE prototype grid")
//  arena demo built around the animated AK-12 first-person rig.
// ============================================================================
//
//  What it shows, all at once:
//    * A stylised prototype-grid level: one procedural, tileable grid texture
//      (DataTexture) with WORLD-SIZED UVs so the checker stays 1 m everywhere,
//      orange accent geometry, emissive trim strips — the classic engine
//      blockout look. Everything is a real PhysX collider.
//    * The Sketchfab AK-12 rig (arms + gun + Idle/Shoot/Reload/Equip clips,
//      CC-BY by SlipS) as a camera-mounted viewmodel with procedural sway/bob,
//      additive Shoot/Reload overlays, muzzle flash (additive sprite + point
//      light) and spent brass ejected from the rig's own Eject bone.
//    * SWAT enemies (the tps_shooter hero, swat.glb) that pursue through a
//      BFS flow field, stop with line of sight and return fire in bursts —
//      and on death hand their SKELETON to physics: capsule bodies + cone
//      joints are built over the major bones and the animated character
//      itself ragdolls. Their rifle drops as a separate physics prop.
//    * Bullet-impact decals (DecalGeometry parented to the hit mesh), dust /
//      blood sprite bursts, tracers, and procedurally synthesised audio.
//
//  Controls:  WASD move   mouse look   LMB fire (full auto)   R reload
//             SHIFT sprint   SPACE jump   ENTER restart (when dead)   ESC quit
// ============================================================================

#include "threepp/threepp.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/audio/Audio.hpp"
#include "threepp/audio/WavFile.hpp"
#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/SpriteInteractor.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/DecalGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace threepp;
using namespace ::physx;
namespace fs = std::filesystem;

// Minimal GLFW FFI for pointer-lock mouse-look (GLFW is compiled into the
// threepp library, so these resolve at link time — same trick as tps_shooter).
extern "C" {
void glfwSetInputMode(void* window, int mode, int value);
int glfwRawMouseMotionSupported(void);
}
namespace glfwc {
    constexpr int CURSOR = 0x00033001;
    constexpr int RAW_MOUSE_MOTION = 0x00033005;
    constexpr int CURSOR_NORMAL = 0x00034001;
    constexpr int CURSOR_DISABLED = 0x00034003;
    constexpr int TRUE_ = 1;
}// namespace glfwc

namespace {

// clang-format off
#include "fps_constants.hpp"
#include "fps_audio.hpp"
#include "fps_hud.hpp"
#include "fps_entities.hpp"
#include "fps_level.hpp"
#include "fps_ragdoll.hpp"
// clang-format on

// A live enemy = pooled SWAT visual + a locked capsule body; a dead one keeps
// the visual and swaps the capsule for a skinned ragdoll.
struct Enemy {
    EnemySlot* slot = nullptr;
    std::shared_ptr<Mesh> proxy;// invisible physics capsule
    PxRigidDynamic* body = nullptr;
    int hp = kEnemyHp;
    bool alive = true;
    float deadTtl = 0.f;
    float hitFlinch = 0.f;
    float fireCd = 0.f;  // between bursts
    int burstLeft = 0;   // shots remaining in the current burst
    float burstCd = 0.f; // between shots inside a burst
    float yaw = 0.f;     // smoothed facing
    SkinnedRagdoll ragdoll;
    PxRigidDynamic* rifleBody = nullptr;// dropped weapon (spawned at death)
};

constexpr float kEnemyRadius = 0.35f;
constexpr float kEnemyLen = 1.05f;
constexpr float kEnemyHalf = kEnemyLen * 0.5f + kEnemyRadius;

}// namespace

int main(int argc, char** argv) {

    // Headless capture (dev): fps_demo --shot <name.png> [--frames N] [--run]
    // [--spin] [--api gl|vulkan] [--timing]. Renders N fixed-dt frames and
    // saves via writeFramebuffer (Vulkan backend), then exits. --timing
    // prints per-frame wall ms (stutter hunting; works windowed too).
    std::string shotPath;
    int shotFrames = 180, shotFrame = 0;
    bool shotAutoRun = false;
    bool shotSpin = false;
    bool shotFire = false;// hold LMB from frame 30 (muzzle flash / casings / decals repro)
    bool shotKill = false;// kill the nearest enemy at frame 220 + track the corpse (ragdoll repro)
    bool timing = false; // print per-frame wall ms (stutter hunting)
    std::optional<GraphicsAPI> apiOverride;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--run") shotAutoRun = true;
        else if (a == "--spin") shotSpin = true;
        else if (a == "--fire") shotFire = true;
        else if (a == "--killtest") shotKill = true;
        else if (a == "--timing") timing = true;
        else if (a == "--api" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "gl") apiOverride = GraphicsAPI::OpenGL;
            else if (v == "vulkan" || v == "vk") apiOverride = GraphicsAPI::Vulkan;
        }
    }

#ifndef __APPLE__
    uiScale = monitor::contentScale().first;
#endif
    const auto screen = monitor::monitorSize();
    const int winW = std::min(static_cast<int>(1280 * uiScale), screen.width() * 9 / 10);
    const int winH = std::min(static_cast<int>(800 * uiScale), screen.height() * 9 / 10);
    Canvas canvas(Canvas::Parameters().title("threepp - FPS (AK-12)").size(winW, winH).antialiasing(4));
    auto renderer = createRenderer(canvas, apiOverride);
    renderer->shadowMap().enabled = true;
    renderer->autoClear = false;
    renderer->toneMapping = ToneMapping::ACESFilmic;
    renderer->toneMappingExposure = 0.9f;

    if (auto vk = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        vk->setMotionBlur(0.35f);
    }

    // Pointer-lock mouse-look; released on game-over for the restart button.
    Vector2 lastMouse{-1, -1};
    bool haveMouse = false;
    auto setCursorLocked = [&](bool locked) {
        if (auto* w = canvas.windowPtr()) {
            glfwSetInputMode(w, glfwc::CURSOR, locked ? glfwc::CURSOR_DISABLED : glfwc::CURSOR_NORMAL);
            if (locked && glfwRawMouseMotionSupported())
                glfwSetInputMode(w, glfwc::RAW_MOUSE_MOTION, glfwc::TRUE_);
        }
        haveMouse = false;// skip the first (jumpy) delta after a mode switch
    };

    // ===== world =============================================================
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(68, canvas.aspect(), 0.02f, 400.f);
    camera->rotation.setOrder(Euler::YXZ);// FPS: yaw, then pitch
    scene->add(camera);                   // the viewmodel hangs under the camera

    // HDR sky = background + IBL on every backend (same asset the tps demo uses)
    Vector3 sunDir;
    sunDir.setFromSphericalCoords(1.f, math::degToRad(90.f - 40.f), math::degToRad(35.f));
    RGBELoader hdrLoader;
    if (auto hdr = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene->background = hdr;
        scene->environment = hdr;
    }
    auto hemi = HemisphereLight::create(0xbdd4ff, 0x8b8d92, 0.3f);
    hemi->position.set(0, 50, 0);
    scene->add(hemi);
    auto sun = DirectionalLight::create(0xfff1d8, 1.7f);
    sun->position.copy(sunDir * 80.f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(4096, 4096);
    sun->shadow->bias = -0.0004f;
    sun->shadow->camera->as<OrthographicCamera>()->left = -kArena * 1.3f;
    sun->shadow->camera->as<OrthographicCamera>()->right = kArena * 1.3f;
    sun->shadow->camera->as<OrthographicCamera>()->top = kArena * 1.3f;
    sun->shadow->camera->as<OrthographicCamera>()->bottom = -kArena * 1.3f;
    sun->shadow->camera->nearPlane = 1.f;
    sun->shadow->camera->farPlane = 240.f;
    sun->shadow->camera->updateProjectionMatrix();
    scene->add(sun);
    // ONE-SUN: the Vulkan deferred resolves the HDR env's sun itself; the
    // raster stand-in would override the measured one (see tps_shooter).
    if (dynamic_cast<VulkanRendererCore*>(renderer.get())) sun->visible = false;

    // ===== audio =============================================================
    SoundBank sfx;
    sfx.init(*camera);

    // ===== physics + level ===================================================
    // Arcade gravity (see fps_constants.hpp kGravity) — real-world 9.81 made
    // the jump feel floaty/moon-like at a usable jump speed.
    PhysxWorld::Settings physSettings;
    physSettings.gravity = Vector3(0.f, -kGravity, 0.f);
    PhysxWorld world(physSettings);
    Level lvl = buildLevel(*scene, world);
    auto& actorToMesh = lvl.actorToMesh;

    // ===== enemy navigation grid (BFS flow field, from tps_shooter) ==========
    const int gridN = static_cast<int>(std::lround(kArena * 2.f / kNavCell));
    auto navIdx = [&](int r, int c) { return static_cast<size_t>(r) * gridN + c; };
    auto colOf = [&](float w) { return std::clamp(static_cast<int>((w + kArena) / kNavCell), 0, gridN - 1); };
    auto cellCenter = [&](int c) { return -kArena + (static_cast<float>(c) + 0.5f) * kNavCell; };
    std::vector<uint8_t> navBlocked(static_cast<size_t>(gridN) * gridN, 0);
    {
        PxOverlapBuffer ob;
        const PxBoxGeometry probe(kNavCell * 0.45f, 0.7f, kNavCell * 0.45f);
        PxQueryFilterData fd;
        fd.flags = PxQueryFlags(PxQueryFlag::eSTATIC);
        for (int gz = 0; gz < gridN; ++gz)
            for (int gx = 0; gx < gridN; ++gx) {
                const PxTransform pose(PxVec3(cellCenter(gx), 0.9f, cellCenter(gz)));
                if (world.scene().overlap(probe, pose, ob, fd)) navBlocked[navIdx(gz, gx)] = 1;
            }
        for (auto& dyn : lvl.dynamics) navBlocked[navIdx(colOf(dyn.home.z), colOf(dyn.home.x))] = 1;
    }
    std::vector<int> navDist(static_cast<size_t>(gridN) * gridN, -1);
    std::vector<int> navFrontier;
    int navPlayerRow = -1, navPlayerCol = -1;
    auto rebuildFlow = [&](int prow, int pcol) {
        std::fill(navDist.begin(), navDist.end(), -1);
        navFrontier.clear();
        navDist[navIdx(prow, pcol)] = 0;
        navFrontier.push_back(prow * gridN + pcol);
        size_t head = 0;
        while (head < navFrontier.size()) {
            const int cur = navFrontier[head++];
            const int r = cur / gridN, c = cur % gridN;
            const int nd = navDist[navIdx(r, c)] + 1;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    if (!dr && !dc) continue;
                    const int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= gridN || nc < 0 || nc >= gridN) continue;
                    if (navBlocked[navIdx(nr, nc)] || navDist[navIdx(nr, nc)] != -1) continue;
                    if (dr && dc && (navBlocked[navIdx(r, nc)] || navBlocked[navIdx(nr, c)])) continue;
                    navDist[navIdx(nr, nc)] = nd;
                    navFrontier.push_back(nr * gridN + nc);
                }
        }
    };

    // ===== player ============================================================
    auto playerProxy = Mesh::create(CapsuleGeometry::create(kPlayerRadius, kPlayerLen), MeshBasicMaterial::create());
    playerProxy->visible = false;
    playerProxy->position.set(0, kPlayerHalf, 10.f);
    scene->add(playerProxy);
    auto* playerBody = world.add(*playerProxy, 80.f);
    playerBody->setRigidDynamicLockFlags(
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
    playerBody->setMaxLinearVelocity(40.f);
    {
        // our own aim raycasts must never hit the player capsule
        PxShape* sh = nullptr;
        playerBody->getShapes(&sh, 1);
        if (sh) sh->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, false);
    }

    float camYaw = 0.f, camPitch = 0.f;
    float recoilKick = 0.f, recoilYaw = 0.f;
    Vector3 playerPos{0, kPlayerHalf, 10.f};
    Vector3 aimDir{0, 0, -1};
    bool wasGrounded = true;
    float bobPhase = 0.f, bobAmp = 0.f;

    // ===== AK-12 viewmodel ===================================================
    // The whole Sketchfab rig (arms + gun + bones) hangs under this group,
    // which hangs under the camera. The group takes the procedural sway/bob.
    auto vmRig = Group::create();
    camera->add(vmRig);
    std::shared_ptr<Object3D> vmModel;
    std::unique_ptr<AnimationMixer> vmMixer;
    AnimationAction *vmIdle = nullptr, *vmShoot = nullptr, *vmReload = nullptr, *vmEquip = nullptr;
    AnimationAction* vmCurrent = nullptr;
    Object3D* vmMuzzle = nullptr;// Silencer_metarig — muzzle flash / tracer origin
    Object3D* vmEject = nullptr; // Eject_metarig — casing origin
    // Placement under the camera is AUTO-FIT at load: the Sketchfab rig is
    // ~2.4x oversized and floats ~2.5 m up, so constants would be brittle.
    // Scale is normalised from the bind-pose armspan, then the model is
    // positioned so the Eject bone (the breech — between the hands) lands on
    // this camera-space anchor. Tuned via headless captures.
    const float kVmArmSpan = 1.55f;              // target bind-pose armspan (m)
    const Vector3 kVmAnchor{0.13f, -0.22f, -0.40f};// breech in camera space
    const float kVmYaw = math::PI;
    {
        GLTFLoader loader;
        const std::string akPath = std::string(DATA_FOLDER) + "/models/gltf/ak-12animated/ak-12animated.glb";
        if (auto res = loader.load(akPath)) {
            vmModel = res->scene;
            vmModel->traverseType<Mesh>([](Mesh& m) {
                m.castShadow = false;// floating first-person arms shadow looks wrong
                m.frustumCulled = false;
            });
            vmModel->rotation.y = kVmYaw;
            vmRig->add(vmModel);

            vmMixer = std::make_unique<AnimationMixer>(*vmModel);
            // additive overlays must be converted BEFORE clipAction
            for (auto& c : res->animations) {
                std::string n = c->name();
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (n == "shoot" || n == "reload") c->makeAdditive();
            }
            auto pick = [&](std::string want) -> AnimationAction* {
                std::transform(want.begin(), want.end(), want.begin(), ::tolower);
                for (auto& c : res->animations) {
                    std::string n = c->name();
                    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                    if (n == want) return vmMixer->clipAction(c);
                }
                return nullptr;
            };
            vmIdle = pick("idle");
            vmShoot = pick("shoot");
            vmReload = pick("reload");
            vmEquip = pick("equip");
            for (auto* a : {vmIdle, vmShoot, vmReload, vmEquip})
                if (a) a->setLoop(Loop::Repeat);
            if (vmShoot) vmShoot->setDuration(0.16f);       // full-auto recoil pop
            if (vmReload) vmReload->setDuration(kReloadTime);// squeeze the 4 s clip
            // base layer: Equip plays first, crossfades to Idle (timer below)
            vmCurrent = vmEquip ? vmEquip : vmIdle;
            if (vmCurrent) vmCurrent->play();
            if (vmIdle && vmCurrent != vmIdle) { /* started by the equip timer */ }
            // overlays: active but silent until weighted in
            if (vmShoot) {
                vmShoot->play();
                vmShoot->setEffectiveWeight(0.f);
            }
            if (vmReload) {
                vmReload->play();
                vmReload->setEffectiveWeight(0.f);
            }

            vmModel->traverse([&](Object3D& o) {
                if (o.name.empty()) return;
                std::string n = o.name;
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (!vmMuzzle && n.find("silencer") != std::string::npos) vmMuzzle = &o;
                if (!vmEject && n.find("eject") != std::string::npos) vmEject = &o;
            });
            std::cout << "Loaded AK-12 viewmodel: " << res->animations.size() << " clip(s); bones muzzle="
                      << (vmMuzzle ? "ok" : "MISSING") << " eject=" << (vmEject ? "ok" : "MISSING") << std::endl;
            {
                // auto-fit: normalise scale by armspan, anchor the breech
                vmModel->updateMatrixWorld(true);
                Box3 bb;
                bb.setFromObject(*vmModel);
                const float span = bb.max().x - bb.min().x;
                const float s = kVmArmSpan / (span > 1e-4f ? span : 1.f);
                vmModel->scale.set(s, s, s);
                vmModel->updateMatrixWorld(true);
                Vector3 ejectAt;
                if (vmEject) ejectAt.setFromMatrixPosition(*vmEject->matrixWorld);
                vmModel->position.copy(kVmAnchor - ejectAt);
                vmModel->updateMatrixWorld(true);
                Vector3 mz;
                if (vmMuzzle) mz.setFromMatrixPosition(*vmMuzzle->matrixWorld);
                std::cout << "  vm auto-fit: scale " << s << ", muzzle at "
                          << mz.x << "," << mz.y << "," << mz.z << std::endl;
            }
        } else {
            std::cerr << "Failed to load " << akPath << "\n";
        }
    }

    // ===== muzzle flash (player): additive sprite + point light, pooled =======
    TextureLoader texLoader;
    auto flashTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/star.png", ColorSpace::sRGB);
    auto vmFlashMat = SpriteMaterial::create();
    vmFlashMat->map = flashTex;
    vmFlashMat->color = Color(0xffd9a0);
    vmFlashMat->transparent = true;
    vmFlashMat->depthWrite = false;
    vmFlashMat->blending = Blending::Additive;
    auto vmFlash = Sprite::create(vmFlashMat);
    vmFlash->visible = false;
    vmRig->add(vmFlash);
    float vmFlashT = 0.f;
    auto vmLight = PointLight::create(0xffc36b, 0.f, 10.f);
    vmRig->add(vmLight);
    // one shared flash light for whichever enemy fired last (a persistent
    // light avoids the shader permutation churn of adding/removing lights)
    auto enemyLight = PointLight::create(0xffc36b, 0.f, 9.f);
    enemyLight->position.set(0, -50, 0);
    scene->add(enemyLight);
    float enemyLightT = 0.f;

    // ===== transient fx (tracers / flashes), casings, decals, particles ======
    std::vector<Ephemeral> fx;
    auto tracerMat = LineBasicMaterial::create();
    tracerMat->color = Color(0xfff2a0);
    tracerMat->transparent = true;
    tracerMat->depthWrite = false;

    auto casingGeo = CylinderGeometry::create(0.0055f, 0.0055f, 0.03f, 8);
    auto casingMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(0xd8b04e).roughness(0.35f).metalness(0.85f).envMapIntensity(0.7f));
    std::vector<Casing> casings;
    const size_t kMaxCasings = 30;
    // Casing meshes are pooled: created once, kept in the scene, and recycled
    // by moving them. Adding/removing a mesh changes the deferred renderer's
    // entry list → full structural rebuild (device drain), and a mid-list
    // removal also resets the TAA/GI accumulation — per-shot and per-expiry
    // that reads as constant stutter. Parked casings sit far below the arena.
    constexpr float kCasingParkY = -60.f;
    std::vector<std::shared_ptr<Mesh>> casingFree;
    for (size_t i = 0; i < kMaxCasings; ++i) {
        auto cm = Mesh::create(casingGeo, casingMat);
        cm->castShadow = true;
        cm->position.set(0.f, kCasingParkY, 0.f);
        scene->add(cm);
        casingFree.push_back(cm);
    }
    auto parkCasing = [&](const std::shared_ptr<Mesh>& m) {
        m->position.set(0.f, kCasingParkY, 0.f);
        casingFree.push_back(m);
    };

    // MeshStandardMaterial, not Phong: the Vulkan deferred renders Standard
    // materials only — a Phong decal simply never shows up there (GL is fine
    // with either).
    auto decalMat = MeshStandardMaterial::create();
    decalMat->map = texLoader.load(std::string(DATA_FOLDER) + "/textures/decal/decal-diffuse.png", ColorSpace::sRGB);
    decalMat->normalMap = texLoader.load(std::string(DATA_FOLDER) + "/textures/decal/decal-normal.jpg", ColorSpace::NoColorSpace);
    decalMat->normalScale.set(1.f, 1.f);
    decalMat->color = Color(0x1a1a1a);
    decalMat->roughness = 0.85f;
    decalMat->metalness = 0.f;
    decalMat->transparent = true;
    decalMat->depthTest = true;
    decalMat->depthWrite = false;
    decalMat->polygonOffset = true;
    decalMat->polygonOffsetFactor = -4.f;
    // Decals are pooled like the casings (entry-list changes = deferred
    // structural rebuild): kMaxDecals meshes with fixed-capacity geometries
    // live in the scene from the start, and a stamp rewrites one geometry in
    // place (degenerate-padded to the cap so the vertex count never changes →
    // the renderer takes the BLAS-refit path, not the full rebuild). The
    // verts are baked in WORLD space; riding a moving crate is done by
    // keeping matrix = targetWorldNow * inverse(targetWorldAtStamp), updated
    // per frame below (identity while the target holds still).
    constexpr int kDecalVertCap = 600;// whole-triangle capacity per pooled decal
    struct DecalSlot {
        std::shared_ptr<Mesh> mesh;
        Object3D* target = nullptr;
        Matrix4 invStamp;
    };
    std::vector<DecalSlot> decals(kMaxDecals);
    for (auto& d : decals) {
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(kDecalVertCap * 3, 0.f), 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(std::vector<float>(kDecalVertCap * 3, 0.f), 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(std::vector<float>(kDecalVertCap * 2, 0.f), 2));
        d.mesh = Mesh::create(geo, decalMat);
        d.mesh->matrixAutoUpdate = false;
        d.mesh->frustumCulled = false;// bounds include the degenerate padding
        scene->add(d.mesh);
    }
    int decalNext = 0;
    auto stampDecal = [&](Mesh* target, const Vector3& point, const Vector3& worldNormal) {
        if (!target) return;
        target->updateMatrixWorld();
        Matrix4 helper;
        helper.setPosition(point);
        helper.lookAt(point, point + worldNormal, Vector3::Z());
        Euler orientation;
        orientation.setFromRotationMatrix(helper);
        orientation.z = frand(0.f, math::PI * 2.f);
        const float s = frand(0.22f, 0.34f);
        auto geo = DecalGeometry::create(*target, point, orientation, Vector3(s, s, s));
        auto* srcPos = geo->getAttribute<float>("position");
        auto* srcNrm = geo->getAttribute<float>("normal");
        auto* srcUv = geo->getAttribute<float>("uv");
        if (!srcPos || !srcNrm || !srcUv || srcPos->count() < 3) return;
        // whole triangles only, truncated to the pool capacity
        const int n = std::min(srcPos->count(), kDecalVertCap) / 3 * 3;

        DecalSlot& d = decals[decalNext];
        decalNext = (decalNext + 1) % kMaxDecals;
        auto* dstPos = d.mesh->geometry()->getAttribute<float>("position");
        auto* dstNrm = d.mesh->geometry()->getAttribute<float>("normal");
        auto* dstUv = d.mesh->geometry()->getAttribute<float>("uv");
        auto fill = [](const std::vector<float>& src, std::vector<float>& dst, int used, int stride) {
            std::copy_n(src.begin(), used * stride, dst.begin());
            // pad with the last real vertex → degenerate (never rasterized/hit)
            for (size_t i = used * stride; i < dst.size(); ++i)
                dst[i] = dst[i - stride];
        };
        fill(srcPos->array(), dstPos->array(), n, 3);
        fill(srcNrm->array(), dstNrm->array(), n, 3);
        fill(srcUv->array(), dstUv->array(), n, 2);
        dstPos->needsUpdate();
        dstNrm->needsUpdate();
        dstUv->needsUpdate();
        d.mesh->geometry()->boundingBox.reset();
        d.mesh->geometry()->boundingSphere.reset();

        d.target = target;
        d.invStamp.copy(*target->matrixWorld).invert();
        d.mesh->matrix->identity();// = targetWorld * invStamp at stamp time
        d.mesh->matrixWorldNeedsUpdate = true;
    };

    // Both dust and blood use the same soft, feathered sprite (a hard-edged
    // disc reads as flying "paintball" blobs, not a mist) — blood is
    // distinguished by tint, smaller/more-numerous sprites, a slower/more
    // diffuse spray, and a capped max opacity so it hazes rather than
    // sitting fully opaque.
    auto dustTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png", ColorSpace::sRGB);
    std::vector<ParticleBurst> bursts;
    auto spawnParticles = [&](const Vector3& point, const Vector3& dir, const Vector3& normal, bool blood) {
        const int count = blood ? 20 : 11;
        ParticleBurst b;
        b.pos.resize(count);
        b.vel.resize(count);
        Vector3 base = blood ? (normal * 0.5f - dir * 0.9f) : normal;
        if (base.length() < 1e-4f) base = normal;
        base.normalize();
        b.mat = SpriteMaterial::create();
        b.mat->map = dustTex;
        b.mat->color = blood ? Color(0x7a1414) : Color(0xb9bfc7);// dust matches the grid grey
        b.mat->transparent = true;
        b.mat->depthWrite = false;
        b.group = Group::create();
        for (int i = 0; i < count; ++i) {
            b.pos[i] = point;
            // wide random spread (vs. a tight directional cone) diffuses the
            // burst into a cloud instead of a spray of discrete streaks
            Vector3 v = base + Vector3(frand(-1.f, 1.f), frand(-1.f, 1.f), frand(-1.f, 1.f)) * (blood ? 1.1f : 0.75f);
            if (v.length() < 1e-4f) v = base;
            v.normalize();
            b.vel[i] = v * (blood ? frand(0.9f, 2.4f) : frand(1.2f, 3.2f));
            auto s = Sprite::create(b.mat);
            s->position.copy(point);
            const float size = blood ? frand(0.05f, 0.11f) : 0.26f;// varied droplet size reads as fine mist
            s->scale.set(size, size, 1.f);
            b.group->add(s);
            b.sprites.push_back(s);
        }
        scene->add(b.group);
        b.life = b.ttl = blood ? 0.5f : 0.70f;
        b.gravity = blood ? 7.f : 3.5f;
        b.drag = blood ? 2.2f : 3.0f;
        b.maxOpacity = blood ? 0.55f : 1.f;
        bursts.push_back(std::move(b));
    };

    // ===== enemy pool (SWAT + rifle, loaded once per slot) ====================
    constexpr int kEnemyPool = kMaxEnemies + 3;// alive + a few lingering corpses
    std::vector<std::unique_ptr<EnemySlot>> pool;
    {
        GLTFLoader loader;
        const std::string swatPath = std::string(DATA_FOLDER) + "/models/gltf/swat.glb";
        const std::string riflePath = std::string(DATA_FOLDER) + "/models/gltf/rifle.glb";
        std::cout << "Loading enemy pool (" << kEnemyPool << "x swat.glb)..." << std::endl;
        for (int i = 0; i < kEnemyPool; ++i) {
            auto res = loader.load(swatPath);
            if (!res) {
                std::cerr << "Failed to load " << swatPath << "\n";
                break;
            }
            auto slot = std::make_unique<EnemySlot>();
            slot->rig = Group::create();
            // Pooled rigs stay VISIBLE and are parked below the arena when
            // unused: toggling visibility adds/removes their skinned meshes
            // from the deferred renderer's entry list, and each spawn then
            // paid a structural rebuild + first-visibility texture/BLAS
            // upload (measured ~240 ms) while each despawn also reset the
            // TAA/GI accumulation. Parked this way, spawn/despawn is a pure
            // transform change. Idle rigs don't tick their mixer, so the
            // parked skeletons cost no skinning/BLAS work per frame.
            slot->rig->position.set(0.f, kEnemyParkY, 0.f);
            scene->add(slot->rig);
            auto& model = res->scene;
            model->traverseType<Mesh>([](Mesh& m) {
                m.castShadow = true;
                m.frustumCulled = false;// skinned bounds are bind-pose; never cull
            });
            // normalise the skeleton's world span to the capsule height
            model->updateMatrixWorld(true);
            float minY = 1e9f, maxY = -1e9f;
            model->traverse([&](Object3D& o) {
                Vector3 wp;
                wp.setFromMatrixPosition(*o.matrixWorld);
                minY = std::min(minY, wp.y);
                maxY = std::max(maxY, wp.y);
            });
            const float skelH = maxY - minY;
            model->scale *= kEnemyCharHeight / (skelH > 1e-4f ? skelH : 1.f);
            slot->model = model;
            slot->rig->add(model);

            slot->mixer = std::make_unique<AnimationMixer>(*model);
            for (auto& c : res->animations) {
                std::string n = c->name();
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (n == "firing rifle") c->makeAdditive();
            }
            auto pick = [&](std::string want) -> AnimationAction* {
                std::transform(want.begin(), want.end(), want.begin(), ::tolower);
                for (auto& c : res->animations) {
                    std::string n = c->name();
                    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                    if (n == want) return slot->mixer->clipAction(c);
                }
                return nullptr;
            };
            slot->anims.idle = pick("rifle aiming idle");
            slot->anims.run = pick("rifle run");
            if (!slot->anims.run) slot->anims.run = pick("walking");
            slot->anims.fire = pick("firing rifle");
            slot->anims.hit = pick("hit reaction");
            for (auto* a : {slot->anims.idle, slot->anims.run, slot->anims.fire, slot->anims.hit})
                if (a) a->setLoop(Loop::Repeat);
            if (!slot->anims.idle && !res->animations.empty()) slot->anims.idle = slot->mixer->clipAction(res->animations.front());
            slot->current = slot->anims.idle;
            if (slot->current) slot->current->play();
            if (slot->anims.fire) {
                slot->anims.fire->play();
                slot->anims.fire->setEffectiveWeight(0.f);
            }

            model->traverse([&](Object3D& o) {
                if (o.name.empty()) return;
                std::string n = o.name;
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (!slot->handBone && n.find("righthand") != std::string::npos) slot->handBone = &o;
                if (!slot->leftHandBone && n.find("lefthand") != std::string::npos) slot->leftHandBone = &o;
                if (!slot->hipsBone && n.find("hips") != std::string::npos) {
                    slot->hipsBone = &o;
                    slot->hipsBind = o.position;
                }
            });

            // rifle instance, pinned into the right hand each frame
            slot->rifle = Group::create();
            if (auto r = loader.load(riflePath)) {
                auto& gun = r->scene;
                gun->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
                gun->updateMatrixWorld(true);
                Box3 gb;
                gb.setFromObject(*gun);
                const Vector3 gsz = gb.getSize();
                const float gmax = std::max({gsz.x, gsz.y, gsz.z});
                if (gmax > 1e-4f) gun->scale *= 0.72f / gmax;
                slot->rifle->add(gun);
                slot->rifle->updateMatrixWorld(true);
                Box3 lb;
                lb.setFromObject(*gun);
                const Vector3 lc = lb.getCenter();
                slot->muzzleLocal.set(lc.x, lc.y, lb.max().z);
            }
            slot->rifle->position.set(0.f, kEnemyParkY, 0.f);// parked, like the rig
            scene->add(slot->rifle);

            // per-slot muzzle flash sprite (world-space, at the rifle tip)
            slot->flashMat = SpriteMaterial::create();
            slot->flashMat->map = flashTex;
            slot->flashMat->color = Color(0xffd9a0);
            slot->flashMat->transparent = true;
            slot->flashMat->depthWrite = false;
            slot->flashMat->blending = Blending::Additive;
            slot->flash = Sprite::create(slot->flashMat);
            slot->flash->visible = false;
            slot->flash->scale.set(0.5f, 0.5f, 1.f);
            scene->add(slot->flash);
            pool.push_back(std::move(slot));
        }
        std::cout << "Enemy pool ready (" << pool.size() << " slots)" << std::endl;
    }

    // ===== enemies ===========================================================
    std::vector<std::unique_ptr<Enemy>> enemies;
    std::unordered_set<const PxRigidActor*> ragdollActors;// corpse hits -> blood
    auto enemyProxyGeo = CapsuleGeometry::create(kEnemyRadius, kEnemyLen);
    float enemySpawnTimer = 1.f;

    auto removeEnemy = [&](Enemy* e) {
        if (e->ragdoll.valid()) {
            for (auto& p : e->ragdoll.parts) ragdollActors.erase(p.body);
            e->ragdoll.destroy(world);
        }
        if (e->body) {
            world.unbind(*e->proxy);
            world.scene().removeActor(*e->body);
            e->body->release();
            e->body = nullptr;
        }
        if (e->rifleBody) {
            world.removeActor(e->rifleBody);// also drops the rifle binding
            e->rifleBody = nullptr;
        }
        if (e->proxy) scene->remove(*e->proxy);
        if (e->slot) {
            // park, don't hide — see the pool-creation comment
            e->slot->rig->position.set(0.f, kEnemyParkY, 0.f);
            e->slot->rifle->position.set(0.f, kEnemyParkY, 0.f);
            e->slot->flash->visible = false;
            e->slot->inUse = false;
            e->slot = nullptr;
        }
    };

    auto spawnEnemy = [&]() {
        EnemySlot* slot = nullptr;
        for (auto& s : pool)
            if (!s->inUse) {
                slot = s.get();
                break;
            }
        if (!slot) {
            // pool exhausted by lingering corpses — reclaim the oldest one
            Enemy* oldest = nullptr;
            for (auto& e : enemies)
                if (!e->alive && (!oldest || e->deadTtl < oldest->deadTtl)) oldest = e.get();
            if (!oldest) return;
            removeEnemy(oldest);
            enemies.erase(std::remove_if(enemies.begin(), enemies.end(),
                                         [&](const std::unique_ptr<Enemy>& p) { return p.get() == oldest; }),
                          enemies.end());
            for (auto& s : pool)
                if (!s->inUse) {
                    slot = s.get();
                    break;
                }
            if (!slot) return;
        }

        const float ang = frand(0.f, 2.f * math::PI);
        const float r = frand(kArena * 0.55f, kArena - 3.f);
        const Vector3 at(std::cos(ang) * r, kEnemyHalf, std::sin(ang) * r);

        auto e = std::make_unique<Enemy>();
        e->slot = slot;
        slot->inUse = true;
        slot->rig->position.copy(at);// unparked; physics re-drives it each frame
        e->proxy = Mesh::create(enemyProxyGeo, MeshBasicMaterial::create());
        e->proxy->visible = false;
        e->proxy->position.copy(at);
        scene->add(e->proxy);
        e->body = world.add(*e->proxy, 60.f);
        e->body->setRigidDynamicLockFlags(
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
        e->body->userData = e.get();
        e->fireCd = frand(0.6f, 1.4f);
        e->yaw = frand(-math::PI, math::PI);
        // reset the recycled slot's animation state
        if (slot->current) {
            slot->current->stop();
        }
        slot->current = slot->anims.idle;
        if (slot->current) {
            slot->current->reset();
            slot->current->play();
        }
        if (slot->anims.fire) {
            slot->anims.fire->reset();
            slot->anims.fire->play();
            slot->anims.fire->setEffectiveWeight(0.f);
        }
        slot->fireW = 0.f;
        enemies.push_back(std::move(e));
    };

    auto killEnemy = [&](Enemy* e, const Vector3& impulseDir) {
        e->alive = false;
        e->deadTtl = kRagdollTtl;
        Vector3 vel(0, 0, 0);
        if (e->body) {
            const PxVec3 v = e->body->getLinearVelocity();
            vel.set(v.x, v.y, v.z);
            world.unbind(*e->proxy);
            world.scene().removeActor(*e->body);
            e->body->release();
            e->body = nullptr;
        }
        // hand the skeleton to physics
        e->ragdoll.build(world, *e->slot->model, vel, (impulseDir + Vector3(0, 0.35f, 0)) * 42.f);
        for (auto& p : e->ragdoll.parts) ragdollActors.insert(p.body);
        // the rifle drops out of the hand as its own prop
        if (!e->rifleBody) {
            e->slot->rifle->updateMatrixWorld(true);
            Vector3 rp, rs;
            Quaternion rq;
            e->slot->rifle->matrixWorld->decompose(rp, rq, rs);
            e->rifleBody = world.addDynamic(PxBoxGeometry(0.05f, 0.09f, 0.36f),
                                            PxTransform(toPxVec3(rp), toPxQuat(rq)), 400.f);
            e->rifleBody->setLinearVelocity(toPxVec3(vel + impulseDir * 1.5f));
            e->rifleBody->setAngularVelocity(PxVec3(frand(-4.f, 4.f), frand(-4.f, 4.f), frand(-4.f, 4.f)));
            world.bind(*e->slot->rifle, *e->rifleBody);
        }
    };

    // ===== game state ========================================================
    int health = 100;
    float healthF = 100.f;   // float shadow for smooth regen
    float sinceHurt = 100.f;// seconds since last damage (drives regen)
    int ammo = kMagSize;
    int score = 0;
    bool reloading = false;
    float reloadTimer = 0.f;
    float fireTimer = 0.f;
    float equipTimer = kEquipTime;// blocks firing while the draw anim plays
    bool firing = false;
    bool firedEmpty = false;
    bool gameOver = false;
    float hitMarkerT = 0.f;
    bool hitWasKill = false;
    float scorePopT = 0.f;
    float chipHealth = 100.f;
    float chSpread = 6.f;
    float hudT = 0.f;
    float stepTimer = 0.f;
    bool stepLeft = false;
    float shootW = 0.f, reloadW = 0.f;
    bool wasReloading = false;
    struct DmgArc {
        std::shared_ptr<Group> g;
        std::vector<std::shared_ptr<MeshBasicMaterial>> mats;
        float t = 0.f;
        float bearing = 0.f;
    };
    std::array<DmgArc, 3> dmgArcs;
    int dmgArcNext = 0;

    // ===== player fire =======================================================
    auto muzzleWorld = [&](Object3D* bone, Vector3& out) {
        if (!bone) return false;
        bone->updateWorldMatrix(true, false);
        out.setFromMatrixPosition(*bone->matrixWorld);
        return true;
    };

    auto fire = [&]() {
        sfx.shot.play(1.f, frand(0.97f, 1.03f));
        ammo--;
        fireTimer = kFireInterval;
        recoilKick = std::min(recoilKick + kRecoilPerShot, kRecoilMax);
        recoilYaw += frand(-kRecoilYawKick, kRecoilYawKick);
        if (vmShoot) vmShoot->reset();

        const Vector3 origin = camera->position;
        Vector3 dir(0, 0, -1);
        dir.applyQuaternion(camera->quaternion);
        // slight full-auto spread (~0.4 deg), grows a touch while moving
        {
            const PxVec3 pv = playerBody->getLinearVelocity();
            const float pSpeed = std::sqrt(pv.x * pv.x + pv.z * pv.z);
            const float spread = 0.007f + pSpeed * 0.0012f;
            Vector3 rightV(1, 0, 0), upV(0, 1, 0);
            rightV.applyQuaternion(camera->quaternion);
            upV.applyQuaternion(camera->quaternion);
            dir += rightV * frand(-spread, spread) + upV * frand(-spread, spread);
            dir.normalize();
        }

        PxRaycastBuffer hitBuf;
        const bool hasHit = world.scene().raycast(toPxVec3(origin), toPxVec3(dir), 250.f, hitBuf) && hitBuf.hasBlock;
        const Vector3 end = hasHit ? fromPxVec3(hitBuf.block.position) : origin + dir * 250.f;

        // muzzle flash + tracer + casing, all anchored on the rig's own bones
        Vector3 mw;
        if (muzzleWorld(vmMuzzle, mw)) {
            // tracer from the true barrel tip (ray still starts at the camera
            // so the crosshair stays pixel-accurate)
            auto tg = BufferGeometry::create();
            tg->setFromPoints(std::vector<Vector3>{mw, end});
            auto tracer = Line::create(tg, tracerMat);
            scene->add(tracer);
            fx.push_back({tracer, 0.05f});

            // flash: rig-local so it rides the camera for its 2-frame life
            vmRig->updateWorldMatrix(true, false);
            Matrix4 inv;
            inv.copy(*vmRig->matrixWorld).invert();
            Vector3 local = mw;
            local.applyMatrix4(inv);
            vmFlash->position.copy(local + Vector3(0.f, 0.f, -0.10f));
            const float fs = frand(0.10f, 0.16f);
            vmFlash->scale.set(fs, fs, 1.f);
            vmFlashMat->rotation = frand(0.f, math::PI * 2.f);
            vmFlash->visible = true;
            vmFlashT = 0.045f;
            vmLight->position.copy(local);
            vmLight->intensity = 6.f;
        }
        Vector3 ew;
        if (muzzleWorld(vmEject, ew)) {
            Vector3 right(1, 0, 0), up(0, 1, 0), fwd(0, 0, -1);
            right.applyQuaternion(camera->quaternion);
            up.applyQuaternion(camera->quaternion);
            fwd.applyQuaternion(camera->quaternion);
            if (casingFree.empty()) {// recycle the oldest airborne casing
                parkCasing(casings.front().mesh);
                casings.erase(casings.begin());
            }
            auto cm = casingFree.back();
            casingFree.pop_back();
            cm->position.copy(ew);
            Casing c;
            c.mesh = cm;
            c.vel = right * frand(1.1f, 1.8f) + up * frand(1.2f, 1.7f) + fwd * frand(-0.4f, -0.1f);
            c.spinAxis.set(frand(-1.f, 1.f), frand(-1.f, 1.f), frand(-1.f, 1.f));
            if (c.spinAxis.length() < 1e-3f) c.spinAxis.set(0.f, 0.f, 1.f);
            c.spinAxis.normalize();
            c.spinRate = frand(16.f, 30.f);
            c.ttl = frand(2.2f, 3.0f);
            // brass may land on a platform, not the floor — resolve at spawn
            PxRaycastBuffer gb;
            PxQueryFilterData fd;
            fd.flags = PxQueryFlags(PxQueryFlag::eSTATIC);
            if (world.scene().raycast(toPxVec3(ew), PxVec3(0, -1, 0), 8.f, gb,
                                      PxHitFlags(PxHitFlag::eDEFAULT), fd) &&
                gb.hasBlock) {
                c.groundY = gb.block.position.y + 0.015f;
            }
            casings.push_back(std::move(c));
        }

        if (!hasHit) return;
        PxRigidActor* actor = hitBuf.block.actor;
        const Vector3 point = fromPxVec3(hitBuf.block.position);
        const Vector3 hitNormal = fromPxVec3(hitBuf.block.normal);

        // live enemy?
        if (actor && actor->userData) {
            auto* e = static_cast<Enemy*>(actor->userData);
            if (e->alive) {
                spawnParticles(point, dir, hitNormal, /*blood*/ true);
                // headshot: enemies are a single capsule collider, so approximate
                // the head as the top slice of it — any hit there is a one-shot kill.
                const float bodyTopY = fromPxVec3(actor->getGlobalPose().p).y + kEnemyHalf;
                const bool headshot = point.y >= bodyTopY - kHeadshotZone;
                if (headshot) e->hp = 0;
                else e->hp--;
                e->hitFlinch = 0.45f;
                hitMarkerT = 0.12f;
                hitWasKill = false;
                if (e->hp <= 0) {
                    killEnemy(e, dir);
                    score += 100;
                    hitMarkerT = 0.25f;
                    hitWasKill = true;
                    scorePopT = 0.8f;
                    sfx.kill.play(1.f, frand(0.95f, 1.05f));
                } else {
                    sfx.hit.play(frand(0.85f, 1.f), frand(0.92f, 1.08f));
                }
                return;
            }
        }
        // corpse? blood + shove, no decal
        if (ragdollActors.count(actor)) {
            spawnParticles(point, dir, hitNormal, /*blood*/ true);
            if (auto* rd = actor->is<PxRigidDynamic>())
                PxRigidBodyExt::addForceAtPos(*rd, toPxVec3(dir * 90.f), hitBuf.block.position, PxForceMode::eIMPULSE);
            return;
        }

        // environment: impact + dust + scorch decal (+ knock dynamics around)
        sfx.metal.play(frand(0.7f, 0.95f), frand(0.9f, 1.1f));
        spawnParticles(point, dir, hitNormal, /*blood*/ false);
        if (auto it = actorToMesh.find(actor); it != actorToMesh.end()) {
            stampDecal(it->second, point, hitNormal);
        }
        if (auto* rd = actor ? actor->is<PxRigidDynamic>() : nullptr) {
            PxRigidBodyExt::addForceAtPos(*rd, toPxVec3(dir * 220.f), hitBuf.block.position, PxForceMode::eIMPULSE);
        }
    };

    // ===== enemy fire ========================================================
    auto enemyShoot = [&](Enemy* e) {
        EnemySlot* s = e->slot;
        s->rifle->updateMatrixWorld(true);
        Vector3 mw = s->muzzleLocal;
        mw.applyMatrix4(*s->rifle->matrixWorld);
        const Vector3 chest = playerPos + Vector3(0, 0.35f, 0);
        Vector3 toP = chest - mw;
        const float dist = toP.length();
        toP.normalize();

        sfx.enemyShot.playAt(mw, frand(0.92f, 1.02f));
        s->flash->position.copy(mw);
        const float fs = frand(0.35f, 0.5f);
        s->flash->scale.set(fs, fs, 1.f);
        s->flashMat->rotation = frand(0.f, math::PI * 2.f);
        s->flash->visible = true;
        s->flashT = 0.045f;
        enemyLight->position.copy(mw);
        enemyLight->intensity = 7.f;
        enemyLightT = 0.05f;

        // hit chance falls with range and player speed; misses land visibly
        const PxVec3 pv = playerBody->getLinearVelocity();
        const float pSpeed = std::sqrt(pv.x * pv.x + pv.z * pv.z);
        const float pHit = std::clamp(0.48f - dist * 0.014f - pSpeed * 0.05f, 0.06f, 0.48f);
        Vector3 endPt;
        if (!gameOver && frand(0.f, 1.f) < pHit) {
            endPt = chest + Vector3(frand(-0.1f, 0.1f), frand(-0.2f, 0.2f), frand(-0.1f, 0.1f));
            health -= kEnemyDamage;
            healthF = static_cast<float>(health);
            sinceHurt = 0.f;
            sfx.hurt.play(1.f, frand(0.95f, 1.05f));
            {
                auto& slotArc = dmgArcs[dmgArcNext];
                dmgArcNext = static_cast<int>((dmgArcNext + 1) % dmgArcs.size());
                slotArc.t = 1.f;
                slotArc.bearing = std::atan2(mw.x - playerPos.x, mw.z - playerPos.z);
            }
        } else {
            // deliberate near-miss: aim past the player, land it on the level
            Vector3 side(-toP.z, 0.f, toP.x);
            Vector3 missDir = toP + side * frand(-0.09f, 0.09f) + Vector3(0, frand(-0.03f, 0.06f), 0);
            missDir.normalize();
            PxRaycastBuffer hb;
            endPt = mw + missDir * 60.f;
            if (world.scene().raycast(toPxVec3(mw + missDir * 0.6f), toPxVec3(missDir), 60.f, hb) && hb.hasBlock) {
                endPt = fromPxVec3(hb.block.position);
                const Vector3 n = fromPxVec3(hb.block.normal);
                spawnParticles(endPt, missDir, n, /*blood*/ false);
                if (auto it = actorToMesh.find(hb.block.actor); it != actorToMesh.end())
                    stampDecal(it->second, endPt, n);
            }
        }
        auto tg = BufferGeometry::create();
        tg->setFromPoints(std::vector<Vector3>{mw, endPt});
        auto tracer = Line::create(tg, tracerMat);
        scene->add(tracer);
        fx.push_back({tracer, 0.06f});
    };

    // ===== HUD (SVG overlay) =================================================
    auto ui = Scene::create();
    auto sz = canvas.size();
    auto uiCam = OrthographicCamera::create(0, sz.width(), sz.height(), 0, 0.1f, 100);
    uiCam->position.z = 10;
    Layout layout;
    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    auto hudScale = [&](const std::shared_ptr<Object3D>& g) {
        g->scale.set(g->scale.x < 0.f ? -uiScale : uiScale,
                     g->scale.y < 0.f ? -uiScale : uiScale, 1.f);
    };

    // crosshair: 4 ticks + dot, spread eased by recoil + movement
    auto crosshair = Group::create();
    std::vector<std::shared_ptr<MeshBasicMaterial>> chMats;
    std::array<std::shared_ptr<Group>, 4> chTicks;
    {
        const float len = 9, th = 2;
        auto mkTick = [&](int i, float w, float h) {
            auto r = rect(w, h, 0xffffff, 0.9f);
            r.mesh->position.set(-w / 2, -h / 2, 0);
            chTicks[i] = Group::create();
            chTicks[i]->add(r.mesh);
            crosshair->add(chTicks[i]);
            chMats.push_back(r.material);
        };
        mkTick(0, th, len);
        mkTick(1, th, len);
        mkTick(2, len, th);
        mkTick(3, len, th);
        auto dot = rect(3, 3, 0xffffff, 0.9f);
        dot.mesh->position.set(-1.5f, -1.5f, 0);
        crosshair->add(dot.mesh);
        chMats.push_back(dot.material);
    }
    ui->add(crosshair);
    layout.add(crosshair, 0.5f, 0.5f, 0, 0, 0.5f);

    // health bar with damage-lag chip (bottom-left)
    {
        auto fg = Group::create();
        fg->add(panel(224, 26, 6, kPanel, 0.75f, kPanelEdge, 1.5f));
        fg->scale.y = -1.f;
        hudScale(fg);
        ui->add(fg);
        layout.add(fg, 0.f, 0.f, 22, 56, 0.1f);
    }
    auto chipFill = rect(216, 18, 0xffaa55, 0.55f);
    {
        auto cg = Group::create();
        cg->add(chipFill.mesh);
        cg->scale.y = -1.f;
        hudScale(cg);
        ui->add(cg);
        layout.add(cg, 0.f, 0.f, 26, 52, 0.15f);
    }
    auto healthFill = rect(216, 18, kHudGood, 0.95f);
    {
        auto hg = Group::create();
        hg->add(healthFill.mesh);
        hg->scale.y = -1.f;
        hudScale(hg);
        ui->add(hg);
        layout.add(hg, 0.f, 0.f, 26, 52, 0.2f);
    }
    auto healthTxt = makeText(font, "100", 0xffffff, 15, 0.f, 0.f, 134, 43,
                              TextSprite::HorizontalAlignment::Center);
    ui->add(healthTxt);
    ui->add(makeText(font, "HP", kHudCyan, 12, 0.f, 0.f, 30, 70));

    // ammo (bottom-right) + magazine pips
    Readout ammoTxt{makeText(font, "", 0xffffff, 34, 1.f, 0.f, -40, 60,
                             TextSprite::HorizontalAlignment::Right)};
    ui->add(ammoTxt.sprite);
    ui->add(makeText(font, "AK-12", kHudCyan, 13, 1.f, 0.f, -40, 92,
                     TextSprite::HorizontalAlignment::Right));
    Readout reloadTxt{makeText(font, "", kHudWarn, 16, 1.f, 0.f, -40, 28,
                               TextSprite::HorizontalAlignment::Right)};
    ui->add(reloadTxt.sprite);
    auto magGroup = Group::create();
    std::vector<std::shared_ptr<MeshBasicMaterial>> pipMats;
    for (int i = 0; i < kMagSize; ++i) {
        auto p = rect(4, 16, 0xffffff, 0.95f);
        p.mesh->position.set(-static_cast<float>(i) * 7.f - 4.f, 0.f, 0.f);
        magGroup->add(p.mesh);
        pipMats.push_back(p.material);
    }
    hudScale(magGroup);
    ui->add(magGroup);
    layout.add(magGroup, 1.f, 0.f, -40, 112, 0.2f);

    // score + hostiles (top)
    Readout scoreTxt{makeText(font, "SCORE 0", kHudCyan, 22, 0.f, 1.f, 24, -34)};
    ui->add(scoreTxt.sprite);
    Readout aliveTxt{makeText(font, "", 0xffffff, 16, 1.f, 1.f, -24, -34,
                              TextSprite::HorizontalAlignment::Right)};
    ui->add(aliveTxt.sprite);

    // hit marker + kill pop
    auto hitMarker = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><g stroke=")") + hex(kHudWarn) +
                                   R"(" stroke-width="3"><line x1="-11" y1="-11" x2="-4" y2="-4"/><line x1="11" y1="-11" x2="4" y2="-4"/>)" +
                                   R"(<line x1="-11" y1="11" x2="-4" y2="4"/><line x1="11" y1="11" x2="4" y2="4"/></g></svg>)");
    hitMarker->visible = false;
    ui->add(hitMarker);
    layout.add(hitMarker, 0.5f, 0.5f, 0, 0, 0.6f);
    auto hitMats = svgMats(hitMarker);
    auto scorePop = makeText(font, "+100", kHudGood, 16, 0.5f, 0.5f, 30, 14,
                             TextSprite::HorizontalAlignment::Left);
    scorePop->visible = false;
    ui->add(scorePop);

    // damage-direction arcs
    for (auto& a : dmgArcs) {
        a.g = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><path d=")") +
                            wedgePath(56.f, 68.f, 26.f) + R"(" fill=)" + '"' + hex(kHudWarn) + R"("/></svg>)");
        a.mats = svgMats(a.g);
        a.g->visible = false;
        hudScale(a.g);
        ui->add(a.g);
        layout.add(a.g, 0.5f, 0.5f, 0, 0, 0.55f);
    }

    // low-health vignette
    auto vignette = rect(1, 1, kHudWarn, 0.f);
    ui->add(vignette.mesh);
    layout.addRaw([m = vignette.mesh](float W, float H) { m->scale.set(W, H, 1); m->position.set(0, 0, 0.05f); });

    ui->add(makeText(font, "WASD move    MOUSE look    LMB fire    R reload    SHIFT sprint    SPACE jump",
                     0x9fb6c8, 12, 0.5f, 0.f, 0, 20,
                     TextSprite::HorizontalAlignment::Center));

    // game over panel + restart
    auto overDim = rect(1, 1, 0x000000, 0.f);
    ui->add(overDim.mesh);
    layout.addRaw([m = overDim.mesh](float W, float H) { m->scale.set(W, H, 1); m->position.set(0, 0, 0.65f); });
    auto over = Group::create();
    over->visible = false;
    ui->add(over);
    {
        auto pg = Group::create();
        pg->add(panel(420, 210, 10, kPanel, 0.92f, kPanelEdge, 2.f));
        pg->scale.y = -1.f;
        hudScale(pg);
        over->add(pg);
        layout.add(pg, 0.5f, 0.5f, -210, 105, 0.7f);
    }
    over->add(makeText(font, "GAME OVER", kHudWarn, 40, 0.5f, 0.5f, 0, 50,
                       TextSprite::HorizontalAlignment::Center));
    Readout overScore{makeText(font, "", 0xffffff, 22, 0.5f, 0.5f, 0, 0,
                               TextSprite::HorizontalAlignment::Center)};
    over->add(overScore.sprite);
    over->add(makeText(font, "PRESS ENTER OR CLICK RESTART", kHudCyan, 15, 0.5f, 0.5f, 0, -40,
                       TextSprite::HorizontalAlignment::Center));
    auto restartLabel = makeText(font, "[ RESTART ]", kHudGood, 20, 0.5f, 0.5f, 0, -75,
                                 TextSprite::HorizontalAlignment::Center);
    over->add(restartLabel);
    auto restartHitMat = SpriteMaterial::create();
    restartHitMat->visible = false;
    auto restartHit = Sprite::create(restartHitMat);
    restartHit->screenSpace = true;
    restartHit->screenAnchor.set(0.5f, 0.5f);
    restartHit->scale.set(180 * uiScale, 44 * uiScale, 1);
    restartHit->center.set(0.5f, 0.5f);
    over->add(restartHit);
    SpriteInteractor interactor(canvas, *ui);

    // ===== restart ===========================================================
    std::function<void()> restart = [&]() {
        health = 100;
        healthF = 100.f;
        sinceHurt = 100.f;
        ammo = kMagSize;
        score = 0;
        reloading = false;
        gameOver = false;
        over->visible = false;
        equipTimer = kEquipTime;
        if (vmEquip && vmCurrent != vmEquip) {
            vmEquip->reset();
            vmEquip->play();
            if (vmCurrent) vmCurrent->crossFadeTo(vmEquip, 0.1f);
            vmCurrent = vmEquip;
        }
        for (auto& e : enemies) removeEnemy(e.get());
        enemies.clear();
        for (auto& c : casings) parkCasing(c.mesh);
        casings.clear();
        recoilKick = 0.f;
        recoilYaw = 0.f;
        chipHealth = 100.f;
        scorePopT = 0.f;
        hitWasKill = false;
        enemySpawnTimer = 1.f;
        for (auto& a : dmgArcs) a.t = 0.f;
        playerBody->setGlobalPose(toPxTransform(Vector3(0, kPlayerHalf, 10.f)));
        playerBody->setLinearVelocity(PxVec3(0));
        for (auto& d : lvl.dynamics) {
            d.body->setGlobalPose(toPxTransform(d.home));
            d.body->setLinearVelocity(PxVec3(0));
            d.body->setAngularVelocity(PxVec3(0));
            d.body->wakeUp();
        }
        setCursorLocked(true);
    };
    restartHit->onMouseUp = [&](int) {
        if (gameOver) restart();
    };

    // ===== input =============================================================
    MouseMoveListener look([&](const Vector2& p) {
        if (gameOver) {
            lastMouse = p;
            haveMouse = true;
            return;
        }
        if (haveMouse) {
            const float dx = p.x - lastMouse.x;
            const float dy = p.y - lastMouse.y;
            camYaw -= dx * kMouseSens;
            camPitch -= dy * kMouseSens;
            camPitch = std::clamp(camPitch, -1.45f, 1.45f);
        }
        lastMouse = p;
        haveMouse = true;
    });
    canvas.addMouseListener(look);
    MouseDownListener down([&](int button, const Vector2&) {
        if (button == 0) firing = true;
    });
    MouseUpListener up([&](int button, const Vector2&) {
        if (button == 0) {
            firing = false;
            firedEmpty = false;
        }
    });
    canvas.addMouseListener(down);
    canvas.addMouseListener(up);

    bool jumpQueued = false;
    canvas.onKeyPressed([&](KeyEvent e) {
        if (e.key == Key::R && !reloading && ammo < kMagSize && equipTimer <= 0.f) {
            reloading = true;
            reloadTimer = kReloadTime;
            sfx.reload.play();
        } else if (e.key == Key::SPACE && !gameOver) {
            jumpQueued = true;
        } else if (e.key == Key::ENTER && gameOver) {
            restart();
        }
    });

    // ===== resize ============================================================
    auto relayout = [&](WindowSize s) {
        const float W = static_cast<float>(s.width());
        const float H = static_cast<float>(s.height());
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(s);
        uiCam->right = W;
        uiCam->top = H;
        uiCam->updateProjectionMatrix();
        layout.apply(W, H);
    };
    canvas.onWindowResize([&](WindowSize s) { relayout(s); });
    relayout(sz);
    setCursorLocked(true);

    // ===== main loop =========================================================
    Clock clock;
    canvas.animate([&] {
        float dt = clock.getDelta();
        if (timing) {
            // real wall time between animate() entries — spikes are stutters
            std::cout << "frame " << shotFrame << " " << dt * 1000.f << " ms"
                      << " ammo=" << ammo << (reloading ? " RELOAD" : "")
                      << " casings=" << casings.size()
                      << " decals=" << std::count_if(decals.begin(), decals.end(), [](const auto& d) { return d.target != nullptr; })
                      << std::endl;
        }
        if (dt > 0.05f) dt = 0.05f;
        if (!shotPath.empty()) dt = 1.f / 60.f;// deterministic captures
        if (shotSpin) camYaw += 1.2f * dt;
        if (shotFire) {
            firing = shotFrame > 30;
            camYaw = math::PI;// face away from the ramp...
            camPitch = -0.5f; // ...and down at the open floor ~3 m out (decal inspection)
        }
        // --killtest: mid-run, execute the nearest enemy and keep the camera on
        // the corpse so the capture shows the skinned ragdoll in motion.
        static Enemy* killTarget = nullptr;
        if (shotKill && shotFrame == 200) {
            // grab the nearest live enemy and plant it 5 m in front of the camera
            float best = 1e9f;
            for (auto& e : enemies) {
                if (!e->alive) continue;
                const PxVec3 ep = e->body->getGlobalPose().p;
                const float d = Vector3(ep.x - playerPos.x, 0, ep.z - playerPos.z).length();
                if (d < best) {
                    best = d;
                    killTarget = e.get();
                }
            }
            if (killTarget) {
                // open floor behind the spawn point (the area ahead is the ramp)
                const Vector3 at = playerPos + Vector3(1.5f, 0.f, 5.f);
                killTarget->body->setGlobalPose(PxTransform(PxVec3(at.x, kEnemyHalf, at.z)));
            }
        }
        if (shotKill && shotFrame == 230 && killTarget && killTarget->alive) {
            killEnemy(killTarget, Vector3(-std::sin(camYaw), 0.f, -std::cos(camYaw)));
            score += 100;
        }
        if (shotKill && killTarget && killTarget->slot) {
            const Vector3 at = killTarget->slot->rig->position;
            camYaw = std::atan2(playerPos.x - at.x, playerPos.z - at.z);
            const float dist = Vector3(at.x - playerPos.x, 0, at.z - playerPos.z).length();
            camPitch = std::atan2(0.4f - kEyeOffset - kPlayerHalf, std::max(1.f, dist));
        }

        // --- player body / grounding ---
        const PxTransform pt = playerBody->getGlobalPose();
        playerPos.set(pt.p.x, pt.p.y, pt.p.z);
        PxRaycastBuffer gb;
        const bool grounded = world.scene().raycast(pt.p, PxVec3(0, -1, 0), kPlayerHalf + 0.18f, gb) && gb.hasBlock;

        // --- movement (camera-relative, ground only) ---
        float moveSpeed = 0.f;
        if (!gameOver) {
            PxVec3 vel = playerBody->getLinearVelocity();
            if (grounded) {
                const Vector3 F(-std::sin(camYaw), 0, -std::cos(camYaw));
                const Vector3 R(std::cos(camYaw), 0, -std::sin(camYaw));
                float fwd = ((canvas.isKeyDown(Key::W) || shotAutoRun) ? 1.f : 0.f) - (canvas.isKeyDown(Key::S) ? 1.f : 0.f);
                float str = (canvas.isKeyDown(Key::D) ? 1.f : 0.f) - (canvas.isKeyDown(Key::A) ? 1.f : 0.f);
                Vector3 move = F * fwd + R * str;
                const bool sprint = canvas.isKeyDown(Key::LEFT_SHIFT);
                const float spd = sprint ? kRunSpeed : kWalkSpeed;
                if (move.length() > 0.01f) {
                    move.normalize();
                    vel.x = move.x * spd;
                    vel.z = move.z * spd;
                    moveSpeed = spd;
                } else {
                    vel.x = 0;
                    vel.z = 0;
                }
                if (jumpQueued) vel.y = kJumpSpeed;
            }
            playerBody->setLinearVelocity(vel);
        }
        jumpQueued = false;

        // --- camera (recoil folds into the view) ---
        {
            const float r = std::min(1.f, dt * kRecoilRecover);
            recoilKick -= recoilKick * r;
            recoilYaw -= recoilYaw * r;
        }
        // head bob: amplitude eases with ground speed
        {
            const float target = (grounded && moveSpeed > 0.1f) ? 1.f : 0.f;
            bobAmp += (target - bobAmp) * std::min(1.f, dt * 8.f);
            bobPhase += dt * (moveSpeed > kWalkSpeed + 0.5f ? 11.f : 7.5f) * bobAmp;
        }
        const float bobY = std::sin(bobPhase * 2.f) * 0.015f * bobAmp;
        camera->position.set(playerPos.x, playerPos.y + kEyeOffset + bobY, playerPos.z);
        camera->rotation.set(camPitch + recoilKick, camYaw + recoilYaw, 0.f);
        aimDir.set(0, 0, -1);
        aimDir.applyQuaternion(camera->quaternion);

        // --- weapon state ---
        equipTimer = std::max(0.f, equipTimer - dt);
        if (vmEquip && vmIdle && vmCurrent == vmEquip && equipTimer <= 0.25f) {
            vmIdle->reset();
            vmIdle->play();
            vmCurrent->crossFadeTo(vmIdle, 0.25f);
            vmCurrent = vmIdle;
        }
        fireTimer -= dt;
        if (reloading) {
            reloadTimer -= dt;
            if (reloadTimer <= 0.f) {
                reloading = false;
                ammo = kMagSize;
            }
        }
        if (firing && !gameOver && equipTimer <= 0.f) {
            if (ammo > 0 && !reloading && fireTimer <= 0.f) {
                fire();
            } else if (ammo == 0 && !reloading && !firedEmpty) {
                sfx.empty.play();
                firedEmpty = true;
                reloading = true;// auto-reload
                reloadTimer = kReloadTime;
                sfx.reload.play();
            }
        }

        // viewmodel overlays + sway/bob
        if (vmMixer) {
            if (reloading && !wasReloading && vmReload) vmReload->reset();
            wasReloading = reloading;
            const float shootTarget = (firing && !reloading && ammo > 0 && equipTimer <= 0.f && !gameOver) ? 1.f : 0.f;
            const float reloadTarget = reloading ? 1.f : 0.f;
            shootW += (shootTarget - shootW) * std::min(1.f, dt * 25.f);
            reloadW += (reloadTarget - reloadW) * std::min(1.f, dt * 12.f);
            if (vmShoot) vmShoot->setEffectiveWeight(shootW);
            if (vmReload) vmReload->setEffectiveWeight(reloadW);
            vmMixer->update(dt);
        }
        {
            // sway: the rig lags the look slightly; bob: figure-8 while moving
            static float swayX = 0.f, swayY = 0.f;
            static float lastYaw = 0.f, lastPitch = 0.f;
            const float dYaw = wrapPi(camYaw - lastYaw);
            const float dPitch = camPitch - lastPitch;
            lastYaw = camYaw;
            lastPitch = camPitch;
            swayX += (std::clamp(dYaw * 5.f, -0.05f, 0.05f) - swayX) * std::min(1.f, dt * 10.f);
            swayY += (std::clamp(dPitch * 5.f, -0.05f, 0.05f) - swayY) * std::min(1.f, dt * 10.f);
            vmRig->position.set(std::sin(bobPhase) * 0.010f * bobAmp + swayX * 0.5f,
                                bobY * 0.6f - std::abs(std::sin(bobPhase)) * 0.006f * bobAmp - swayY * 0.4f,
                                0.f);
            vmRig->rotation.set(-swayY * 0.6f, swayX * 0.8f, swayX * 0.3f);
        }
        vmFlashT -= dt;
        if (vmFlashT <= 0.f) vmFlash->visible = false;
        vmLight->intensity = std::max(0.f, vmLight->intensity - dt * 220.f);
        enemyLightT -= dt;
        if (enemyLightT <= 0.f) enemyLight->intensity = std::max(0.f, enemyLight->intensity - dt * 200.f);

        // --- flow field refresh ---
        {
            const int pr = colOf(playerPos.z), pc = colOf(playerPos.x);
            if (pr != navPlayerRow || pc != navPlayerCol) {
                navPlayerRow = pr;
                navPlayerCol = pc;
                rebuildFlow(pr, pc);
            }
        }

        // --- enemy spawn ---
        if (!gameOver) {
            enemySpawnTimer -= dt;
            int aliveCount = 0;
            for (auto& e : enemies)
                if (e->alive) aliveCount++;
            if (aliveCount < kMaxEnemies && enemySpawnTimer <= 0.f) {
                spawnEnemy();
                enemySpawnTimer = 1.6f;
            }
        }

        // --- enemy AI ---
        for (auto& e : enemies) {
            if (!e->alive) {
                e->deadTtl -= dt;
                if (e->slot && e->slot->flash->visible) e->slot->flash->visible = false;
                continue;
            }
            EnemySlot* s = e->slot;
            const PxVec3 ep = e->body->getGlobalPose().p;
            Vector3 toPlayer(playerPos.x - ep.x, 0, playerPos.z - ep.z);
            const float d = toPlayer.length();

            // line of sight: static geometry only (eye height -> player chest)
            bool los = false;
            {
                const Vector3 eye(ep.x, ep.y + 0.45f, ep.z);
                Vector3 dir = (playerPos + Vector3(0, 0.35f, 0)) - eye;
                const float dl = dir.length();
                if (dl > 1e-3f) {
                    dir.multiplyScalar(1.f / dl);
                    PxRaycastBuffer hb;
                    PxQueryFilterData fd;
                    fd.flags = PxQueryFlags(PxQueryFlag::eSTATIC);
                    los = !(world.scene().raycast(toPxVec3(eye), toPxVec3(dir), dl, hb,
                                                  PxHitFlags(PxHitFlag::eDEFAULT), fd) &&
                            hb.hasBlock);
                }
            }

            PxVec3 v = e->body->getLinearVelocity();
            const bool engaging = d < kEnemyFireRange && los;
            if (!engaging) {
                // steer down the flow field (routes around cover)
                Vector3 desired = toPlayer;
                desired.normalize();
                const int er = colOf(ep.z), ec = colOf(ep.x);
                auto rawDist = [&](int r, int c) -> float {
                    if (r < 0 || r >= gridN || c < 0 || c >= gridN) return 1e6f;
                    const int dv = navDist[navIdx(r, c)];
                    return dv < 0 ? 1e6f : static_cast<float>(dv);
                };
                Vector3 dir = desired;
                const float here = rawDist(er, ec);
                if (here < 1e6f) {
                    auto g = [&](int r, int c) { return std::min(rawDist(r, c), here + 3.f); };
                    Vector3 flow(g(er, ec - 1) - g(er, ec + 1), 0.f,
                                 g(er - 1, ec) - g(er + 1, ec));
                    if (flow.length() > 1e-3f) {
                        flow.normalize();
                        dir = flow;
                    }
                }
                for (auto& o : enemies) {
                    if (o.get() == e.get() || !o->alive) continue;
                    const PxVec3 op = o->body->getGlobalPose().p;
                    Vector3 away(ep.x - op.x, 0.f, ep.z - op.z);
                    const float sd = away.length();
                    if (sd > 1e-3f && sd < kSeparation)
                        dir += away * ((kSeparation - sd) / sd * 0.6f);
                }
                if (dir.length() < 1e-3f) dir = desired;
                dir.normalize();
                v.x = dir.x * kEnemySpeed;
                v.z = dir.z * kEnemySpeed;
            } else {
                v.x = 0;
                v.z = 0;
                // burst fire
                e->fireCd -= dt;
                if (e->burstLeft > 0) {
                    e->burstCd -= dt;
                    if (e->burstCd <= 0.f) {
                        enemyShoot(e.get());
                        e->burstLeft--;
                        e->burstCd = kEnemyBurstGap;
                    }
                } else if (e->fireCd <= 0.f && !gameOver) {
                    e->burstLeft = static_cast<int>(kEnemyBurst);
                    e->burstCd = 0.f;
                    e->fireCd = kEnemyFireInterval + frand(0.f, 0.5f);
                }
            }
            e->body->setLinearVelocity(v);
            e->hitFlinch = std::max(0.f, e->hitFlinch - dt);

            // facing: velocity when moving, the player when engaging
            {
                float wantYaw = e->yaw;
                if (engaging) wantYaw = std::atan2(toPlayer.x, toPlayer.z);
                else if (std::abs(v.x) + std::abs(v.z) > 0.2f) wantYaw = std::atan2(v.x, v.z);
                e->yaw += wrapPi(wantYaw - e->yaw) * std::min(1.f, dt * 8.f);
            }

            // base anim: run / idle / flinch; fire is an additive overlay
            {
                AnimationAction* want = s->anims.idle;
                if (std::abs(v.x) + std::abs(v.z) > 0.3f && s->anims.run) want = s->anims.run;
                if (e->hitFlinch > 0.f && s->anims.hit) want = s->anims.hit;
                if (want && want != s->current) {
                    want->reset();
                    want->play();
                    if (s->current) s->current->crossFadeTo(want, 0.18f);
                    s->current = want;
                }
                const float fireTarget = (e->burstLeft > 0 && s->anims.fire) ? 1.f : 0.f;
                s->fireW += (fireTarget - s->fireW) * std::min(1.f, dt * 15.f);
                if (s->anims.fire) s->anims.fire->setEffectiveWeight(s->fireW);
                s->mixer->update(dt);
                // pin clip root motion — physics owns the position (see tps_shooter)
                if (s->hipsBone) {
                    s->hipsBone->position.x = s->hipsBind.x;
                    s->hipsBone->position.y = s->hipsBind.y;
                }
            }

            // place the rig + pin the rifle into the right hand
            s->rig->position.set(ep.x, ep.y - kEnemyHalf, ep.z);
            s->rig->rotation.y = e->yaw;
            if (s->handBone) {
                s->rig->updateMatrixWorld(true);
                Vector3 hp, hs;
                Quaternion hq;
                s->handBone->matrixWorld->decompose(hp, hq, hs);
                Vector3 off(0.f, 0.15f, 0.f);
                off.applyQuaternion(hq);
                s->rifle->position.copy(hp + off);
                if (s->leftHandBone) {
                    Vector3 lp, ls;
                    Quaternion lq;
                    s->leftHandBone->matrixWorld->decompose(lp, lq, ls);
                    s->rifle->lookAt(lp);
                } else {
                    Vector3 fwd(std::sin(e->yaw), 0.f, std::cos(e->yaw));
                    s->rifle->lookAt(s->rifle->position + fwd);
                }
                s->rifle->quaternion.multiply(
                        Quaternion().setFromAxisAngle(Vector3(0.f, 1.f, 0.f), math::degToRad(-25.f)));
            }
        }
        // expired corpses
        for (auto it = enemies.begin(); it != enemies.end();) {
            if (!(*it)->alive && (*it)->deadTtl <= 0.f) {
                removeEnemy(it->get());
                it = enemies.erase(it);
            } else {
                ++it;
            }
        }
        if (health <= 0 && !gameOver) {
            health = 0;
            gameOver = true;
            over->visible = true;
            overScore.set("FINAL SCORE   " + std::to_string(score));
            setCursorLocked(false);
        }

        // --- physics ---
        world.step(dt);

        // corpses: write the simulated bodies back into the skeletons
        for (auto& e : enemies)
            if (!e->alive && e->ragdoll.valid()) e->ragdoll.drive();

        // --- footsteps ---
        const bool justLanded = grounded && !wasGrounded;
        wasGrounded = grounded;
        if (!gameOver && justLanded) {
            sfx.step.play(1.2f, frand(0.78f, 0.85f));
            stepTimer = 0.25f;
        }
        if (!gameOver && moveSpeed > 0.1f && grounded) {
            stepTimer -= dt;
            if (stepTimer <= 0.f) {
                const bool sprint = moveSpeed > kWalkSpeed + 0.5f;
                stepLeft = !stepLeft;
                sfx.step.play(sprint ? 1.f : 0.65f,
                              (stepLeft ? 0.96f : 1.04f) * frand(0.94f, 1.06f));
                stepTimer = sprint ? 0.32f : 0.45f;
            }
        } else {
            stepTimer = 0.12f;
        }

        // --- transient fx ---
        for (auto it = fx.begin(); it != fx.end();) {
            it->ttl -= dt;
            if (it->ttl <= 0.f) {
                scene->remove(*it->obj);
                it = fx.erase(it);
            } else {
                ++it;
            }
        }
        // enemy muzzle flashes: independent per-slot timer (NOT burstCd, which
        // freezes once a burst ends — see fps_entities.hpp EnemySlot::flashT)
        for (auto& s : pool) {
            if (!s->flash->visible) continue;
            s->flashT -= dt;
            if (s->flashT <= 0.f) s->flash->visible = false;
        }

        // --- casings (ballistic + tumble + damped bounce) ---
        for (auto it = casings.begin(); it != casings.end();) {
            auto& c = *it;
            c.ttl -= dt;
            if (c.ttl <= 0.f) {
                parkCasing(c.mesh);
                it = casings.erase(it);
                continue;
            }
            c.vel.y -= 9.81f * dt;
            c.mesh->position.add(c.vel * dt);
            if (c.mesh->position.y < c.groundY && c.vel.y < 0.f) {
                c.mesh->position.y = c.groundY;
                c.vel.y = -c.vel.y * 0.35f;
                c.vel.x *= 0.6f;
                c.vel.z *= 0.6f;
                c.spinRate *= 0.5f;
                if (!c.tinked) {
                    c.tinked = true;
                    sfx.tink.play(frand(0.5f, 0.8f), frand(0.9f, 1.15f));
                }
            }
            Quaternion dq;
            dq.setFromAxisAngle(c.spinAxis, c.spinRate * dt);
            c.mesh->quaternion.premultiply(dq);
            ++it;
        }

        // --- decals ride the mesh they were stamped on ---
        for (auto& d : decals) {
            if (!d.target) continue;
            d.mesh->matrix->copy(*d.target->matrixWorld).multiply(d.invStamp);
            d.mesh->matrixWorldNeedsUpdate = true;
        }

        // --- impact particles ---
        for (auto it = bursts.begin(); it != bursts.end();) {
            auto& b = *it;
            b.ttl -= dt;
            if (b.ttl <= 0.f) {
                scene->remove(*b.group);
                it = bursts.erase(it);
                continue;
            }
            for (size_t i = 0; i < b.pos.size(); ++i) {
                b.vel[i].y -= b.gravity * dt;
                b.vel[i] *= std::max(0.f, 1.f - b.drag * dt);
                b.pos[i] += b.vel[i] * dt;
                b.sprites[i]->position.copy(b.pos[i]);
            }
            b.mat->opacity = std::clamp(b.ttl / b.life, 0.f, 1.f) * b.maxOpacity;
            ++it;
        }

        // --- health regen (out-of-combat) ---
        sinceHurt += dt;
        if (!gameOver && health > 0 && health < 100 && sinceHurt > kRegenDelay) {
            healthF = std::min(100.f, healthF + kRegenRate * dt);
            health = static_cast<int>(healthF);
        }

        // --- HUD ---
        hudT += dt;
        healthFill.mesh->scale.x = std::max(0.001f, static_cast<float>(health) / 100.f);
        healthFill.material->color = Color(health > 50 ? kHudGood : (health > 25 ? 0xffaa33 : kHudWarn));
        healthFill.material->opacity = health <= 25 ? 0.7f + 0.25f * std::sin(hudT * 8.f) : 0.95f;
        if (static_cast<float>(health) > chipHealth) chipHealth = static_cast<float>(health);
        chipHealth = std::max(static_cast<float>(health), chipHealth - dt * 35.f);
        chipFill.mesh->scale.x = std::max(0.001f, chipHealth / 100.f);
        healthTxt->setText(std::to_string(health));
        {
            std::ostringstream os;
            os << ammo << " / " << kMagSize;
            ammoTxt.set(os.str());
        }
        reloadTxt.set(reloading ? "RELOADING" : (ammo == 0 ? "EMPTY" : ""));
        {
            const int lit = reloading
                                    ? static_cast<int>((1.f - reloadTimer / kReloadTime) * kMagSize + 1e-3f)
                                    : ammo;
            const bool dry = !reloading && ammo == 0;
            for (int i = 0; i < kMagSize; ++i) {
                const bool on = i < lit;
                pipMats[i]->color = Color(dry ? kHudWarn : (on ? 0xffffff : kPanelEdge));
                pipMats[i]->opacity = dry ? 0.25f + 0.4f * (0.5f + 0.5f * std::sin(hudT * 10.f))
                                          : (on ? 0.95f : 0.35f);
            }
        }
        scoreTxt.set("SCORE " + std::to_string(score));
        {
            int a = 0;
            for (auto& e : enemies)
                if (e->alive) a++;
            aliveTxt.set("HOSTILES " + std::to_string(a));
        }
        {
            const PxVec3 pv = playerBody->getLinearVelocity();
            const float speed = std::sqrt(pv.x * pv.x + pv.z * pv.z);
            const float target = std::clamp(6.f + recoilKick * 60.f + speed * 0.9f, 2.f, 26.f);
            chSpread += (target - chSpread) * std::min(1.f, dt * 14.f);
            const float sPix = chSpread + 4.5f;
            chTicks[0]->position.set(0, sPix, 0);
            chTicks[1]->position.set(0, -sPix, 0);
            chTicks[2]->position.set(-sPix, 0, 0);
            chTicks[3]->position.set(sPix, 0, 0);
        }
        crosshair->scale.set(uiScale * (1.f + recoilKick * 4.f), uiScale * (1.f + recoilKick * 4.f), 1.f);
        hitMarkerT -= dt;
        hitMarker->visible = hitMarkerT > 0.f;
        if (hitMarker->visible) {
            const float pop = hitWasKill ? 1.f + 0.6f * std::min(1.f, hitMarkerT / 0.25f) : 1.f;
            hitMarker->scale.set(uiScale * pop, uiScale * pop, 1);
            for (auto& m : hitMats) m->color = Color(hitWasKill ? kHudWarn : 0xffffff);
        }
        scorePopT -= dt;
        scorePop->visible = scorePopT > 0.f;
        if (scorePop->visible) scorePop->position.set(30.f * uiScale, (14.f + (0.8f - scorePopT) * 55.f) * uiScale, 0.f);
        for (auto& a : dmgArcs) {
            a.t -= dt;
            a.g->visible = a.t > 0.f;
            if (!a.g->visible) continue;
            a.g->rotation.z = wrapPi(a.bearing - camYaw);
            const float o = std::clamp(a.t, 0.f, 1.f) * 0.8f;
            for (auto& m : a.mats) m->opacity = o;
        }
        vignette.material->opacity = std::clamp((60.f - health) / 60.f, 0.f, 0.55f);
        overDim.material->opacity += ((gameOver ? 0.55f : 0.f) - overDim.material->opacity) * std::min(1.f, dt * 6.f);
        for (auto& m : chMats) m->color = Color(hitMarkerT > 0.f ? kHudWarn : 0xffffff);

        // ===== render: world, then SVG overlay =====
        renderer->autoClear = true;
        renderer->render(*scene, *camera);
        renderer->autoClear = false;
        renderer->clearDepth();
        renderer->render(*ui, *uiCam);

        if (!shotPath.empty() && shotKill && shotFrame == shotFrames - 1) {
            std::cout << "player " << playerPos.x << "," << playerPos.z << " camYaw " << camYaw << std::endl;
            for (auto& e : enemies) {
                Vector3 hips;
                if (e->slot && e->slot->hipsBone) hips.setFromMatrixPosition(*e->slot->hipsBone->matrixWorld);
                std::cout << (e->alive ? "alive" : "DEAD ") << " rig "
                          << e->slot->rig->position.x << "," << e->slot->rig->position.y << "," << e->slot->rig->position.z
                          << "  hipsW " << hips.x << "," << hips.y << "," << hips.z << std::endl;
            }
            int slotIdx = 0;
            for (auto& s : pool) {
                Vector3 p;
                p.setFromMatrixPosition(*s->rig->matrixWorld);
                std::cout << "slot" << slotIdx++ << (s->inUse ? " USED " : " free ") << p.x << "," << p.y << "," << p.z
                          << " vis=" << s->rig->visible << std::endl;
            }
        }
        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            renderer->writeFramebuffer(path);
            std::cout << "wrote " << path.string() << std::endl;
            std::exit(0);
        }
    });
}
