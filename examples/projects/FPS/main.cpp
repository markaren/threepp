// ============================================================================
//  threepp Shooting Range — an engine-blockout ("UE prototype grid") range
//  built around the animated AK-12 first-person rig and PhysX rigid bodies.
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
//    * A shooting range whose targets are REAL RIGID BODIES, not scripted
//      knock-over animations: steel plates on posts, crate stacks and barrels,
//      each a PhysX dynamic with a mesh collider. Every hit applies an impulse
//      AT THE CONTACT POINT, so where you hit decides what happens — a plate
//      caught on the edge spins off its post, one hit square tips straight
//      back, and a crate stack collapses the way its own masses say it should.
//      Plates score when they actually go down and stand themselves back up a
//      few seconds later, so the range keeps feeding you targets.
//    * Bullet-impact decals (DecalGeometry parented to the hit mesh — so they
//      ride a crate you knock across the floor), dust bursts, tracers, and
//      procedurally synthesised audio.
//
//  Controls:  WASD move   mouse look   LMB fire (full auto)   R reload
//             SHIFT sprint   SPACE jump   ENTER reset the range   ESC quit
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

#ifdef FPS_DEMO_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

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
// clang-format on

}// namespace

int main(int argc, char** argv) {

    // Headless capture (dev): fps_demo --shot <name.png> [--frames N] [--run]
    // [--spin] [--look yawDeg pitchDeg] [--api gl|vulkan] [--timing].
    // Renders N fixed-dt frames and
    // saves via writeFramebuffer (Vulkan backend), then exits. --timing
    // prints per-frame wall ms (stutter hunting; works windowed too).
    std::string shotPath;
    int shotFrames = 180, shotFrame = 0;
    bool shotAutoRun = false;
    bool shotSpin = false;
    bool shotFire = false;// hold LMB from frame 30 (muzzle flash / casings / decals repro)
    bool timing = false; // print per-frame wall ms (stutter hunting)
    std::optional<float> shotYaw, shotPitch;// --look <yawDeg> <pitchDeg>
    std::optional<GraphicsAPI> apiOverride;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--run") shotAutoRun = true;
        else if (a == "--spin") shotSpin = true;
        else if (a == "--fire") shotFire = true;
        else if (a == "--look" && i + 2 < argc) {
            shotYaw = std::strtof(argv[++i], nullptr);
            shotPitch = std::strtof(argv[++i], nullptr);
        }
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

#ifdef FPS_DEMO_WITH_VULKAN
    if (auto vk = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        vk->setMotionBlur(0.35f);
    }
#endif

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
#ifdef FPS_DEMO_WITH_VULKAN
    if (dynamic_cast<VulkanRenderer*>(renderer.get())) sun->visible = false;
#endif

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
    // Actor -> target, wired HERE and not inside buildLevel: the props are
    // pushed into a vector, and every push can move the ones already in it, so
    // any pointer taken during the build would dangle. Done once the vector is
    // final, it stays valid for the run.
    for (auto& d : lvl.dynamics) d.body->userData = &d;

    // ===== player ============================================================
    auto playerProxy = Mesh::create(CapsuleGeometry::create(kPlayerRadius, kPlayerLen), MeshBasicMaterial::create());
    playerProxy->visible = false;
    playerProxy->position.set(0, kPlayerHalf, kFiringLineZ - 2.8f);
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

    // Facing DOWN the range (+Z): the camera looks along its own -Z, so a
    // half turn is what puts the lanes in front of the player at spawn.
    float camYaw = math::PI, camPitch = 0.f;
    float recoilKick = 0.f, recoilYaw = 0.f;
    Vector3 playerPos{0, kPlayerHalf, kFiringLineZ - 2.8f};
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
    // Length of the Equip clip, read off the loaded clip. The draw must run to
    // its natural end (see kEquipTime) — that is where it meets Idle's pose.
    float vmEquipLen = kEquipTime;
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
            for (auto& c : res->animations) {
                std::string n = c->name();
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (n == "equip") vmEquipLen = c->getDuration();
            }
            for (auto* a : {vmIdle, vmShoot, vmReload, vmEquip})
                if (a) a->setLoop(Loop::Repeat);
            if (vmShoot) vmShoot->setDuration(0.16f);// full-auto recoil pop
            if (vmEquip) {
                // Draw: play once at the authored rate and hold the last frame
                // (which IS Idle's first pose) while the crossfade runs. It used
                // to be a Repeat action cut off 0.57 s early — mid-pose, 0.67 rad
                // from Idle — which lurched on every spawn and restart.
                vmEquip->setLoop(Loop::Once, 1);
                vmEquip->setClampWhenFinished(true);
            }
            if (vmReload) {
                // Reload is an ADDITIVE overlay squeezed into kReloadTime. As a
                // Repeat action it wrapped to frame 0 exactly as the reload
                // ended — and since the weight only STARTED fading at that
                // moment, the hands visibly snapped back to the start of the
                // reload before going quiet. Loop::Once + clampWhenFinished
                // holds the last frame instead (paused, still fed), so the
                // fade-out happens from the pose the clip ended on. Finishing
                // kReloadHold early leaves that held frame on screen for the
                // whole fade.
                vmReload->setLoop(Loop::Once, 1);
                vmReload->setClampWhenFinished(true);
                vmReload->setDuration(kReloadTime - kReloadHold);
            }
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
            // Clip durations matter: a base-layer clip held on screen for
            // LONGER than its own length silently wraps and replays (that was
            // the reload snap, and the equip draw before setDuration below).
            for (auto& c : res->animations)
                std::cout << "  clip '" << c->name() << "' " << c->getDuration() << " s" << std::endl;
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
    // Decals are a fixed-count InstancedMesh of oriented quads. Earlier
    // iterations added/removed decal meshes (deferred structural rebuild per
    // stamp) or rewrote a pooled geometry in place (BLAS-refit device drain
    // on the stamp frame + prevVertex-resync drain the frame after — a pair
    // of ~33 ms hitches per shot on Vulkan). Instance-matrix edits are
    // transform-only: the renderer re-reads them every frame with no drain.
    // A quad doesn't conform to corners like DecalGeometry, but bullet
    // scorch marks land on flat floor/wall/crate faces — lifted ~4 mm along
    // the hit normal to avoid z-fighting in both raster and RT.
    auto decalGeo = PlaneGeometry::create(1.f, 1.f);
    auto decals = InstancedMesh::create(decalGeo, decalMat, kMaxDecals);
    decals->frustumCulled = false;// instances are scattered; cull per-frame is meaningless
    struct DecalSlot {
        Object3D* target = nullptr;// stamped-on mesh (decals ride moving crates)
        Matrix4 local;             // instance matrix in target-local space
    };
    std::vector<DecalSlot> decalSlots(kMaxDecals);
    {
        Matrix4 park;
        park.setPosition(0.f, -70.f, 0.f);// below the arena until stamped
        for (int i = 0; i < kMaxDecals; ++i) decals->setMatrixAt(i, park);
        decals->instanceMatrix()->needsUpdate();
    }
    scene->add(decals);
    int decalNext = 0;
    auto stampDecal = [&](Mesh* target, const Vector3& point, const Vector3& worldNormal) {
        if (!target) return;
        target->updateMatrixWorld();
        Quaternion q;
        q.setFromUnitVectors(Vector3(0, 0, 1), worldNormal);// quad faces +Z
        Quaternion roll;
        roll.setFromAxisAngle(Vector3(0, 0, 1), frand(0.f, math::PI * 2.f));
        q.multiply(roll);
        const float s = frand(0.22f, 0.34f);
        Matrix4 m;
        m.compose(point + worldNormal * 0.004f, q, Vector3(s, s, 1.f));

        DecalSlot& d = decalSlots[decalNext];
        d.target = target;
        d.local.copy(*target->matrixWorld).invert().multiply(m);
        decals->setMatrixAt(decalNext, m);
        decals->instanceMatrix()->needsUpdate();
        decalNext = (decalNext + 1) % kMaxDecals;
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

    // ===== game state ========================================================
    int health = 100;
    float healthF = 100.f;   // float shadow for smooth regen
    float sinceHurt = 100.f;// seconds since last damage (drives regen)
    int ammo = kMagSize;
    int score = 0;
    bool reloading = false;
    float reloadTimer = 0.f;
    float fireTimer = 0.f;
    // blocks firing while the draw anim plays: the clip's own length plus the
    // crossfade tail, so the draw is never cut mid-pose
    float equipTimer = vmEquipLen + kEquipFade;
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

        // A steel plate? Score it, but only on the swing that actually puts it
        // down — `down` latches, so a plate you keep hitting while it lies
        // there does not farm points, and one that merely rocks scores nothing.
        // The knockdown itself is left entirely to the impulse below: nothing
        // here scripts the fall.
        if (actor && actor->userData) {
            auto* d = static_cast<Dynamic*>(actor->userData);
            if (d->kind == Dynamic::Kind::Plate && !d->down) {
                hitMarkerT = 0.12f;
                hitWasKill = false;
                sfx.hit.play(frand(0.85f, 1.f), frand(0.92f, 1.08f));
            }
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

    // score + plates standing (top). No health bar: nothing on a range shoots
    // back, and a permanently full HP bar is dead pixels.
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
        equipTimer = vmEquipLen + kEquipFade;
        if (vmEquip && vmCurrent != vmEquip) {
            vmEquip->reset();
            vmEquip->play();
            if (vmCurrent) vmCurrent->crossFadeTo(vmEquip, 0.1f);
            vmCurrent = vmEquip;
        }
        chipHealth = 100.f;
        scorePopT = 0.f;
        hitWasKill = false;
        playerBody->setGlobalPose(toPxTransform(Vector3(0, kPlayerHalf, kFiringLineZ - 2.8f)));
        playerBody->setLinearVelocity(PxVec3(0));
        for (auto& d : lvl.dynamics) {
            d.down = false;
            d.resetIn = 0.f;
            d.body->setGlobalPose(PxTransform(toPxVec3(d.home), toPxQuat(d.homeRot)));
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
                      << " decals=" << std::count_if(decalSlots.begin(), decalSlots.end(), [](const auto& d) { return d.target != nullptr; });
#ifdef FPS_DEMO_WITH_VULKAN
            if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get()))
                std::cout << " overlayMs=" << vk->lastFrameTimings().overlayMs;
#endif
            std::cout << std::defaultfloat << std::endl;
        }
        if (dt > 0.05f) dt = 0.05f;
        if (!shotPath.empty()) dt = 1.f / 60.f;// deterministic captures
        if (shotSpin) camYaw += 1.2f * dt;
        if (shotFire) {
            firing = shotFrame > 30;
            camYaw = math::PI;  // down the range...
            camPitch = -0.045f; // ...and just under level, onto the near plates
        }
        // --look <yawDeg> <pitchDeg>: pin the camera for a capture. Needed to
        // inspect the ground AT the player's feet (viewmodel shadow checks) —
        // at pitch 0 the nearest visible floor is already ~2.4 m out.
        if (shotYaw) camYaw = math::degToRad(*shotYaw);
        if (shotPitch) camPitch = math::degToRad(*shotPitch);
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
        // Crossfade to Idle only once the draw has actually finished and is
        // holding its final frame (clampWhenFinished), so the blend is between
        // two poses that already match.
        if (vmEquip && vmIdle && vmCurrent == vmEquip && equipTimer <= kEquipFade) {
            vmIdle->reset();
            vmIdle->play();
            vmCurrent->crossFadeTo(vmIdle, kEquipFade);
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
            // reset() clears the clampWhenFinished pause from the previous
            // reload, so the Once action runs again from frame 0.
            if (reloading && !wasReloading && vmReload) {
                vmReload->reset();
                vmReload->play();
            }
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

        // --- physics ---
        world.step(dt);

        // --- targets: score the plates that went down, stand them back up ---
        // "Down" is measured off the body's own up axis rather than off a hit
        // count, so what scores is the plate actually falling — clip one hard
        // enough to spin it off its post and that counts; graze one and it
        // rocks back upright and does not.
        for (auto& d : lvl.dynamics) {
            const PxTransform pose = d.body->getGlobalPose();
            if (d.kind == Dynamic::Kind::Plate) {
                const PxVec3 up = pose.q.rotate(PxVec3(0, 1, 0));
                if (!d.down && up.y < kPlateDownCos) {
                    d.down = true;
                    d.resetIn = kPlateResetDelay;
                    score += 100;
                    scorePopT = 0.8f;
                    hitMarkerT = 0.25f;
                    hitWasKill = true;
                    sfx.kill.play(1.f, frand(0.95f, 1.05f));
                } else if (d.down) {
                    d.resetIn -= dt;
                    if (d.resetIn <= 0.f) {
                        d.down = false;
                        d.body->setGlobalPose(PxTransform(toPxVec3(d.home), toPxQuat(d.homeRot)));
                        d.body->setLinearVelocity(PxVec3(0));
                        d.body->setAngularVelocity(PxVec3(0));
                        d.body->wakeUp();
                    }
                }
            }
            // Anything shot clean off the range comes home rather than falling
            // through the world forever.
            if (pose.p.y < kPropRecycleY) {
                d.body->setGlobalPose(PxTransform(toPxVec3(d.home), toPxQuat(d.homeRot)));
                d.body->setLinearVelocity(PxVec3(0));
                d.body->setAngularVelocity(PxVec3(0));
                d.down = false;
                d.resetIn = 0.f;
            }
        }

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
        {
            bool anyDecal = false;
            for (int i = 0; i < kMaxDecals; ++i) {
                const auto& d = decalSlots[i];
                if (!d.target) continue;
                Matrix4 w;
                w.multiplyMatrices(*d.target->matrixWorld, d.local);
                decals->setMatrixAt(i, w);
                anyDecal = true;
            }
            if (anyDecal) decals->instanceMatrix()->needsUpdate();
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

        // --- HUD ---
        hudT += dt;
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
            int up = 0, plates = 0;
            for (const auto& d : lvl.dynamics) {
                if (d.kind != Dynamic::Kind::Plate) continue;
                ++plates;
                if (!d.down) ++up;
            }
            aliveTxt.set("PLATES " + std::to_string(up) + "/" + std::to_string(plates));
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
        overDim.material->opacity += ((gameOver ? 0.55f : 0.f) - overDim.material->opacity) * std::min(1.f, dt * 6.f);
        for (auto& m : chMats) m->color = Color(hitMarkerT > 0.f ? kHudWarn : 0xffffff);

        // ===== render: world, then SVG overlay =====
        renderer->autoClear = true;
        renderer->render(*scene, *camera);
        renderer->autoClear = false;
        renderer->clearDepth();
        renderer->render(*ui, *uiCam);

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            renderer->writeFramebuffer(path);
            std::cout << "wrote " << path.string() << std::endl;
            setCursorLocked(false);// std::exit runs no destructors — Canvas's
                                   // pointer-release never happens, so do it here
            std::exit(0);
        }
    });
}
