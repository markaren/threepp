// ============================================================================
//  threepp Third-Person Shooter — a showcase prototype
// ============================================================================
//
//  Pulls together a lot of threepp at once:
//    * GLTF + skeletal animation (Soldier.glb) with idle/walk/run crossfades
//    * PhysX integration (extras/physx/PhysxWorld): a velocity-driven capsule
//      character, static level colliders, and dynamic crates/barrels you can
//      shoot and knock around. Shooting is a PhysX scene raycast.
//    * A pure-SVG heads-up display (crosshair, health, ammo, score, hit marker,
//      low-health vignette, game-over panel) drawn in an ortho overlay pass —
//      the same technique as examples/loaders/svg_ui.cpp.
//    * Procedurally synthesised placeholder sound effects (no asset files) via
//      the miniaudio-backed Audio API.
//
//  Controls:  WASD move   mouse look   LMB fire   SHIFT sprint
//             SPACE jump   R reload     ENTER restart (when dead)   ESC quit
//
//  Everything here is a stand-in / prototype: the soldier is the placeholder
//  character, enemies are simple capsule bots, sounds are synthesised.
// ============================================================================

#include "threepp/threepp.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/audio/Audio.hpp"
#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/SpriteInteractor.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/DecalGeometry.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/objects/TextSprite.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace ::physx;
namespace fs = std::filesystem;

// Minimal GLFW FFI for pointer-lock mouse-look. GLFW is compiled into the
// threepp library (as an OBJECT library), so these resolve at link time with
// no GLFW include / include-path needed. canvas.windowPtr() returns the
// GLFWwindow* (void*-compatible). Constants copied from GLFW/glfw3.h.
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

    // ---- tuning constants --------------------------------------------------
    constexpr float kArena = 28.f;// half-extent of the play area
    constexpr float kPlayerRadius = 0.35f;
    constexpr float kPlayerLen = 1.1f;                              // capsule cylinder segment
    constexpr float kPlayerHalf = kPlayerLen * 0.5f + kPlayerRadius;// centre->foot
    constexpr float kWalkSpeed = 3.2f;
    constexpr float kRunSpeed = 6.4f;
    constexpr float kJumpSpeed = 5.2f;
    constexpr float kMouseSens = 0.0026f;
    constexpr float kFireInterval = 0.11f;
    constexpr float kReloadTime = 1.25f;
    constexpr int kMagSize = 12;
    constexpr int kMaxDecals = 48;// bullet-impact decals before the oldest recycles
    constexpr float kEnemySpeed = 2.3f;
    constexpr int kEnemyHp = 3;
    constexpr int kMaxEnemies = 6;
    constexpr float kEnemyAttackRange = 1.7f;

    // ---- procedural jump pose (no jump clip ships with Soldier.glb) ---------
    // Layered on top of the locomotion clip while airborne. Axis signs are
    // guesses for the Mixamo rig — flip a sign if a limb bends the wrong way.
    constexpr float kJumpTuckThigh = -1.0f;// raise thighs (knees toward chest)
    constexpr float kJumpTuckKnee = 1.3f;  // bend knees in the air
    constexpr float kJumpLean = 0.18f;     // slight torso lean
    constexpr float kJumpLandKnee = 0.7f;  // extra knee bend on landing (squash)

    // ---- palette -----------------------------------------------------------
    constexpr int kHudCyan = 0x35c2ff;
    constexpr int kHudGood = 0x47e07a;
    constexpr int kHudWarn = 0xff4d4d;
    constexpr int kPanel = 0x0e1b2a;
    constexpr int kPanelEdge = 0x1d3b57;

    std::mt19937 rng{1337};
    float frand(float a, float b) {
        return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(rng);
    }

    // ========================================================================
    //  Procedural placeholder sound effects
    // ========================================================================
    // 16-bit mono PCM WAV writer + a few quick synths. The Audio API loads
    // from a file path, so we render these to a temp dir at startup.

    void writeWav(const fs::path& path, const std::vector<float>& samples, int sr = 44100) {
        std::ofstream f(path, std::ios::binary);
        auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
        auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); };
        const uint32_t dataBytes = static_cast<uint32_t>(samples.size()) * 2u;
        f.write("RIFF", 4);
        u32(36 + dataBytes);
        f.write("WAVE", 4);
        f.write("fmt ", 4);
        u32(16);
        u16(1);// PCM
        u16(1);// mono
        u32(sr);
        u32(sr * 2);
        u16(2);
        u16(16);
        f.write("data", 4);
        u32(dataBytes);
        for (float x : samples) {
            const auto q = static_cast<int16_t>(std::lround(std::clamp(x, -1.f, 1.f) * 32767.f));
            f.write(reinterpret_cast<const char*>(&q), 2);
        }
    }

    std::vector<float> synthShot(int sr = 44100) {
        const int n = sr * 18 / 100;// 0.18s
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float env = std::exp(-t * 22.f);
            const float noise = frand(-1.f, 1.f);
            const float thump = std::sin(2.f * math::PI * 70.f * t) * std::exp(-t * 12.f);
            s[i] = std::clamp((noise * 0.8f + thump * 0.6f) * env, -1.f, 1.f);
        }
        return s;
    }

    std::vector<float> synthClick(int sr = 44100) {
        const int n = sr * 3 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            s[i] = frand(-1.f, 1.f) * std::exp(-t * 120.f) * 0.5f;
        }
        return s;
    }

    std::vector<float> synthReload(int sr = 44100) {
        const int n = sr * 35 / 100;
        std::vector<float> s(n, 0.f);
        auto clickAt = [&](float at, float amp) {
            const int start = static_cast<int>(at * sr);
            for (int i = 0; i < sr * 3 / 100 && start + i < n; ++i) {
                const float t = static_cast<float>(i) / sr;
                s[start + i] += frand(-1.f, 1.f) * std::exp(-t * 90.f) * amp;
            }
        };
        clickAt(0.f, 0.5f);
        clickAt(0.16f, 0.4f);
        clickAt(0.30f, 0.6f);
        return s;
    }

    std::vector<float> synthHit(int sr = 44100) {
        const int n = sr * 8 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            s[i] = std::sin(2.f * math::PI * 1400.f * t) * std::exp(-t * 30.f) * 0.5f;
        }
        return s;
    }

    std::vector<float> synthThud(int sr = 44100) {
        const int n = sr * 14 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float tone = std::sin(2.f * math::PI * 180.f * t);
            s[i] = (tone * 0.6f + frand(-1.f, 1.f) * 0.4f) * std::exp(-t * 24.f);
        }
        return s;
    }

    std::vector<float> synthHurt(int sr = 44100) {
        const int n = sr * 28 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float freq = 420.f - 260.f * (t / 0.28f);
            s[i] = std::sin(2.f * math::PI * freq * t) * std::exp(-t * 6.f) * 0.6f;
        }
        return s;
    }

    std::vector<float> synthStep(int sr = 44100) {
        const int n = sr * 7 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            s[i] = frand(-1.f, 1.f) * std::exp(-t * 45.f) * 0.35f;
        }
        return s;
    }

    // A pooled, retriggerable sound: round-robins a few voices so rapid fire
    // overlaps instead of cutting itself off. Degrades to a no-op if the audio
    // device or file failed to initialise.
    struct Sound {
        std::vector<std::unique_ptr<Audio>> voices;
        size_t next = 0;
        void play() {
            if (voices.empty()) return;
            voices[next]->stop();
            voices[next]->seekToStart();// rewind so re-fire restarts from frame 0
            voices[next]->play();
            next = (next + 1) % voices.size();
        }
    };

    struct SoundBank {
        std::unique_ptr<AudioListener> listener;
        Sound shot, empty, reload, hit, thud, hurt, step, metal;
        bool ok = false;

        void init(Object3D& attachTo) {
            try {
                const fs::path dir = fs::temp_directory_path() / "threepp_tps_sounds";
                fs::create_directories(dir);
                struct Spec {
                    const char* name;       // temp WAV name (synth fallback)
                    std::vector<float> data;// synth samples
                    Sound* dst;
                    int voices;
                    std::string file;// external audio file; used instead of synth when set
                };

                // Real submachine-gun sample for the gun; the rest stay
                // procedural. Falls back to the synth shot if the file is absent.
                const std::string assets = std::string(PROJECT_FOLDER) + "/examples/projects/Shooter/assets/";
                const std::string gunFile = assets + "submachine_gun.mp3";
                const std::string reloadFile = assets + "reload_1911.mp3";
                const std::string metalFile = assets + "metal_impact.mp3";
                std::vector<Spec> specs{
                        {"shot.wav", synthShot(), &shot, 6, fs::exists(gunFile) ? gunFile : std::string{}},
                        {"empty.wav", synthClick(), &empty, 2, {}},
                        {"reload.wav", synthReload(), &reload, 2, fs::exists(reloadFile) ? reloadFile : std::string{}},
                        {"hit.wav", synthHit(), &hit, 4, {}},
                        {"metal.wav", synthThud(), &metal, 4, fs::exists(metalFile) ? metalFile : std::string{}},
                        {"thud.wav", synthThud(), &thud, 4, {}},
                        {"hurt.wav", synthHurt(), &hurt, 2, {}},
                        {"step.wav", synthStep(), &step, 4, {}}};

                listener = std::make_unique<AudioListener>();
                attachTo.addRef(*listener);
                for (auto& sp : specs) {
                    std::string path;
                    if (!sp.file.empty()) {
                        path = sp.file;// external sample (e.g. the gun MP3)
                    } else {
                        path = (dir / sp.name).string();
                        writeWav(path, sp.data);// render the synth fallback
                    }
                    for (int i = 0; i < sp.voices; ++i) {
                        auto a = std::make_unique<Audio>(*listener, path);
                        a->setVolume(0.6f);
                        sp.dst->voices.push_back(std::move(a));
                    }
                }
                ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[audio] disabled: " << e.what() << "\n";
            }
        }
    };

    // ========================================================================
    //  SVG HUD toolkit (condensed from examples/loaders/svg_ui.cpp)
    // ========================================================================

    std::shared_ptr<Group> buildSvg(const std::vector<SVGLoader::SVGData>& svgData) {
        auto group = Group::create();
        for (const auto& data : svgData) {
            const auto& fill = data.style.fill;
            if (fill && *fill != "none") {
                auto m = MeshBasicMaterial::create();
                m->color.copy(data.path.color);
                m->opacity = data.style.fillOpacity;
                m->transparent = true;
                m->depthTest = false;
                m->depthWrite = false;
                m->side = Side::Double;
                auto mesh = Mesh::create(ShapeGeometry::create(SVGLoader::createShapes(data)), m);
                mesh->name = data.style.id;
                group->add(mesh);
            }
            const auto& stroke = data.style.stroke;
            if (stroke && *stroke != "none") {
                auto sMat = MeshBasicMaterial::create();
                sMat->color.setStyle(*stroke);
                sMat->opacity = data.style.strokeOpacity;
                sMat->transparent = true;
                sMat->depthTest = false;
                sMat->depthWrite = false;
                sMat->side = Side::Double;
                for (const auto& subPath : data.path.subPaths) {
                    auto sg = SVGLoader::pointsToStroke(subPath->getPoints(), data.style);
                    if (sg) group->add(Mesh::create(sg, sMat));
                }
            }
        }
        return group;
    }

    std::shared_ptr<Group> svgFromString(const std::string& svg) {
        SVGLoader loader;
        return buildSvg(loader.parse(svg));
    }

    std::string hex(int rgb) {
        std::ostringstream os;
        os << '#' << std::hex << std::setfill('0') << std::setw(6) << (rgb & 0xffffff);
        return os.str();
    }

    // Owned-material rect (so we can recolour / rescale it). Built from <rect>.
    struct RectMesh {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<MeshBasicMaterial> material;
    };
    RectMesh rect(float w, float h, int color, float opacity = 1.f) {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width=")" << w
            << R"(" height=")" << h << R"(" fill="#ffffff"/></svg>)";
        SVGLoader loader;
        auto data = loader.parse(svg.str());
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(color);
        mat->opacity = opacity;
        mat->transparent = true;
        mat->depthTest = false;
        mat->depthWrite = false;
        mat->side = Side::Double;
        auto geo = ShapeGeometry::create(SVGLoader::createShapes(data.front()));
        return {Mesh::create(geo, mat), mat};
    }

    std::shared_ptr<TextSprite> makeText(const Font& font, const std::string& text, int color,
                                         float px, float ax, float ay, float ox, float oy,
                                         TextSprite::HorizontalAlignment h = TextSprite::HorizontalAlignment::Left,
                                         TextSprite::VerticalAlignment v = TextSprite::VerticalAlignment::Center) {
        auto t = TextSprite::create(font, px);
        t->setColor(Color(color));
        t->setText(text);
        t->setHorizontalAlignment(h);
        t->setVerticalAlignment(v);
        t->screenSpace = true;
        t->screenAnchor.set(ax, ay);
        t->position.set(ox, oy, 0.f);
        return t;
    }

    // A text readout that only re-rasterises when its content changes.
    struct Readout {
        std::shared_ptr<TextSprite> sprite;
        std::string last;
        void set(const std::string& s) {
            if (s == last) return;
            last = s;
            sprite->setText(s);
        }
    };

    // Responsive anchoring: position = anchor*viewport + pixel offset, re-applied
    // on resize. Mirrors the svg_ui layout helper.
    struct Layout {
        std::vector<std::function<void(float, float)>> fns;
        void add(const std::shared_ptr<Object3D>& g, float ax, float ay, float ox, float oy, float z) {
            fns.emplace_back([=](float W, float H) { g->position.set(ax * W + ox, ay * H + oy, z); });
        }
        void addRaw(std::function<void(float, float)> fn) { fns.emplace_back(std::move(fn)); }
        void apply(float W, float H) {
            for (auto& f : fns) f(W, H);
        }
    };

    // ========================================================================
    //  Game entities
    // ========================================================================

    struct Enemy {
        std::shared_ptr<Mesh> visual;// capsule body (added to scene)
        std::shared_ptr<MeshStandardMaterial> mat;
        PxRigidDynamic* body = nullptr;
        int hp = kEnemyHp;
        bool alive = true;
        float deadTtl = 0.f;
        float attackCd = 0.f;
    };

    struct Ephemeral {// short-lived visual (tracer / flash / spark)
        std::shared_ptr<Object3D> obj;
        float ttl;
    };

}// namespace

int main() {

    Canvas canvas(Canvas::Parameters().title("threepp - Third Person Shooter").size(1280, 800).antialiasing(4));
    // Force the GL backend so the demo launches straight into the game instead
    // of prompting for a renderer on stdin (createRenderer's default behaviour).
    auto renderer = createRenderer(canvas, GraphicsAPI::WebGPU);
    renderer->shadowMap().enabled = true;
    renderer->autoClear = false;

    const float dpi = monitor::contentScale().first;

    // Pointer-lock mouse-look: cursor hidden + grabbed while playing (raw,
    // unbounded deltas), released on game-over so the restart button is
    // clickable, re-grabbed on restart.
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

    // ===== world ============================================================
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(70, canvas.aspect(), 0.1f, 1000.f);
    camera->position.set(0, 3, 8);

    // One sun direction (low golden-hour sun) drives both the sky shader and the
    // directional light, so the lighting matches the sky.
    Vector3 sunDir;
    sunDir.setFromSphericalCoords(1.f, math::degToRad(90.f - 15.f), math::degToRad(150.f));

    // Procedural Preetham sky dome (the shader pins itself to the far plane, so
    // its scale is independent of the camera's far distance).
    auto sky = Sky::create();
    sky->scale.setScalar(1500.f);
    {
        auto& u = sky->materialAs<ShaderMaterial>()->uniforms;
        u.at("turbidity").value<float>() = 6.f;
        u.at("rayleigh").value<float>() = 2.2f;
        u.at("mieCoefficient").value<float>() = 0.005f;
        u.at("mieDirectionalG").value<float>() = 0.82f;
        u.at("sunPosition").value<Vector3>().copy(sunDir);
    }
    scene->add(sky);

    const Color fogColor(0xd8c7ac);// warm horizon haze, blends geometry into the sky
    scene->background = fogColor;
    scene->fog = Fog(fogColor, 48.f, 150.f);

    // lighting: warm low sun + cool sky fill + a touch of ambient
    auto hemi = HemisphereLight::create(0xaeccff, 0x4a4031, 0.55f);
    hemi->position.set(0, 50, 0);
    scene->add(hemi);
    scene->add(AmbientLight::create(0xffffff, 0.12f));
    auto sun = DirectionalLight::create(0xffe4bd, 2.9f);
    sun->position.copy(sunDir * 80.f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    sun->shadow->bias = -0.0004f;
    sun->shadow->camera->as<OrthographicCamera>()->left = -kArena;
    sun->shadow->camera->as<OrthographicCamera>()->right = kArena;
    sun->shadow->camera->as<OrthographicCamera>()->top = kArena;
    sun->shadow->camera->as<OrthographicCamera>()->bottom = -kArena;
    sun->shadow->camera->nearPlane = 1.f;
    sun->shadow->camera->farPlane = 240.f;
    sun->shadow->camera->updateProjectionMatrix();
    scene->add(sun);

    // ===== audio ============================================================
    SoundBank sfx;
    sfx.init(*camera);

    // ===== physics ==========================================================
    PhysxWorld world;

    // Maps each PhysX collider back to its visual Mesh so a shot can stamp a
    // decal on whatever it hit (and parent it there so it rides moving crates).
    std::unordered_map<const PxRigidActor*, Mesh*> actorToMesh;

    // ---- materials --------------------------------------------------------
    // Solid PBR colours (no image maps). The GL backend crashes uploading the
    // large non-power-of-two assets (sand.jpg 2070x1381, crate.gif), so the
    // world stays texture-free; the sky + lighting still carry the look.
    auto groundMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xbeae89).roughness(1.f).metalness(0.f));
    auto concreteMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x9a958c).roughness(0.85f).metalness(0.04f));
    auto sandstoneMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xb8a17e).roughness(0.9f).metalness(0.f));
    auto crateMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xb5793a).roughness(0.7f).metalness(0.05f).map(TextureLoader().load(std::string(DATA_FOLDER) + "/textures/crate.gif")));
    auto barrelMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x3f7d4f).roughness(0.45f).metalness(0.4f));

    // ---- ground ------------------------------------------------------------
    auto ground = Mesh::create(BoxGeometry::create(kArena * 2, 1.f, kArena * 2), groundMat);
    ground->position.y = -0.5f;
    ground->receiveShadow = true;
    scene->add(ground);
    actorToMesh[world.addStatic(*ground)] = ground.get();

    // ---- static-geometry helpers (each collider registers for decals/dust) -
    // Axis-aligned or rotated box. Shape is inferred from the geometry (no
    // scaling), so ramps tilt via `pitch` rather than a scaled mesh.
    auto addBox = [&](const Vector3& pos, const Vector3& size, const std::shared_ptr<Material>& mat, float yaw = 0.f, float pitch = 0.f) {
        auto m = Mesh::create(BoxGeometry::create(size.x, size.y, size.z), mat);
        m->position.copy(pos);
        m->rotation.set(pitch, yaw, 0.f);
        m->castShadow = true;
        m->receiveShadow = true;
        scene->add(m);
        actorToMesh[world.addStatic(*m)] = m.get();
        return m;
    };
    auto addPillar = [&](float x, float z, float radius, float height, const std::shared_ptr<Material>& mat) {
        auto m = Mesh::create(CylinderGeometry::create(radius, radius, height, 20), mat);
        m->position.set(x, height * 0.5f, z);
        m->castShadow = true;
        m->receiveShadow = true;
        scene->add(m);
        if (auto* b = world.addStaticTrimesh(*m)) actorToMesh[b] = m.get();
        return m;
    };

    // ---- perimeter walls + corner towers ----------------------------------
    const float wallH = 5.f;
    addBox({0, wallH * 0.5f, -kArena}, {kArena * 2, wallH, 1.f}, concreteMat);
    addBox({0, wallH * 0.5f, kArena}, {kArena * 2, wallH, 1.f}, concreteMat);
    addBox({-kArena, wallH * 0.5f, 0}, {1.f, wallH, kArena * 2}, concreteMat);
    addBox({kArena, wallH * 0.5f, 0}, {1.f, wallH, kArena * 2}, concreteMat);
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            addBox({sx * kArena, 3.5f, sz * kArena}, {4.5f, 7.f, 4.5f}, sandstoneMat);

    // ---- central pillar plaza ---------------------------------------------
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            addPillar(sx * 8.f, sz * 8.f, 0.55f, 6.f, sandstoneMat);
    (void) addPillar;

    // ---- raised bunker with a ramp up (rewards the jump) ------------------
    {
        const float pz = -16.f, pw = 12.f, pd = 7.f, ph = 1.4f;
        addBox({0, ph * 0.5f, pz}, {pw, ph, pd}, concreteMat);                      // platform (top at ph; low enough to jump onto the edge)
        addBox({0, ph + 0.5f, pz - pd * 0.5f + 0.5f}, {pw, 1.f, 1.f}, sandstoneMat);// rear parapet (cover up top)
        // ramp on the +Z (centre-facing) edge. Flip the pitch sign if it tilts
        // the wrong way.
        const float run = 7.f;
        addBox({0, ph * 0.5f - 0.1f, pz + pd * 0.5f + run * 0.5f - 1.f}, {5.f, 0.5f, run}, concreteMat, 0.f, std::atan2(ph, run));
    }

    // ---- corner L-cover ----------------------------------------------------
    auto addCornerCover = [&](float cx, float cz) {
        const float ix = cx > 0 ? -1.f : 1.f, iz = cz > 0 ? -1.f : 1.f;
        addBox({cx + ix * 2.f, 0.8f, cz}, {5.f, 1.6f, 1.f}, sandstoneMat);
        addBox({cx, 0.8f, cz + iz * 2.f}, {1.f, 1.6f, 5.f}, sandstoneMat);
    };
    addCornerCover(-18, -18);
    addCornerCover(18, -18);
    addCornerCover(-18, 18);
    addCornerCover(18, 18);

    // ---- mid-lane sandbag cover -------------------------------------------
    addBox({17, 0.5f, 0}, {1.2f, 1.f, 6.f}, sandstoneMat);
    addBox({-17, 0.5f, 0}, {1.2f, 1.f, 6.f}, sandstoneMat);
    addBox({0, 0.5f, 17}, {6.f, 1.f, 1.2f}, sandstoneMat);

    // ---- dynamic crates (shootable / pushable) — home pose kept for restart -
    struct Dynamic {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* body;
        Vector3 home;
    };
    std::vector<Dynamic> dynamics;
    auto crateGeo = BoxGeometry::create(1.f, 1.f, 1.f);
    auto spawnCrateStack = [&](float cx, float cz, int height) {
        for (int y = 0; y < height; ++y) {
            auto m = Mesh::create(crateGeo, crateMat);
            m->position.set(cx + frand(-0.05f, 0.05f), 0.5f + y * 1.001f, cz + frand(-0.05f, 0.05f));
            m->castShadow = true;
            m->receiveShadow = true;
            scene->add(m);
            auto* body = world.add(*m, 60.f);
            actorToMesh[body] = m.get();
            dynamics.push_back({m, body, m->position.clone()});
        }
    };
    spawnCrateStack(13, 11, 3);
    spawnCrateStack(-13, 12, 2);
    spawnCrateStack(12, -12, 3);
    spawnCrateStack(-12, -13, 2);
    spawnCrateStack(6, -14, 2);

    // ---- dynamic barrels (convex collider cooked from cylinder verts) ------
    auto barrelGeo = CylinderGeometry::create(0.45f, 0.45f, 1.2f, 18);
    auto spawnBarrels = [&](float cx, float cz, int n) {
        for (int i = 0; i < n; ++i) {
            auto m = Mesh::create(barrelGeo, barrelMat);
            m->position.set(cx + frand(-1.2f, 1.2f), 0.6f, cz + frand(-1.2f, 1.2f));
            m->castShadow = true;
            m->receiveShadow = true;
            scene->add(m);
            if (auto* body = world.addDynamicConvex(*m, 35.f)) {
                actorToMesh[body] = m.get();
                dynamics.push_back({m, body, m->position.clone()});
            }
        }
    };
    spawnBarrels(16, -4, 3);
    spawnBarrels(-16, 4, 3);
    spawnBarrels(4, 16, 2);

    // ===== player ===========================================================
    // Collision proxy: a capsule rigid body with locked rotation, driven by
    // setting horizontal velocity each frame (classic kinematic-velocity
    // character) while PhysX handles gravity, walls and slopes.
    auto playerProxy = Mesh::create(CapsuleGeometry::create(kPlayerRadius, kPlayerLen), MeshBasicMaterial::create());
    playerProxy->visible = false;
    playerProxy->position.set(0, kPlayerHalf, 0);
    auto* playerBody = world.add(*playerProxy, 80.f);
    playerBody->setRigidDynamicLockFlags(
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
    playerBody->setMaxLinearVelocity(40.f);
    {
        // Disable scene queries on the player's own shape so our aim/ground
        // raycasts never hit the player (the camera sits behind them).
        PxShape* sh = nullptr;
        playerBody->getShapes(&sh, 1);
        if (sh) sh->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, false);
    }

    // Visual: animated Soldier.glb (idle / walk / run crossfades)
    auto playerRig = Group::create();
    scene->add(playerRig);
    std::unique_ptr<AnimationMixer> mixer;
    AnimationAction* idleA = nullptr;
    AnimationAction* walkA = nullptr;
    AnimationAction* runA = nullptr;
    AnimationAction* currentA = nullptr;
    // Leg/spine bones for the procedural jump (found by Mixamo-name substring).
    Object3D* boneLUpLeg = nullptr;
    Object3D* boneRUpLeg = nullptr;
    Object3D* boneLLeg = nullptr;
    Object3D* boneRLeg = nullptr;
    Object3D* boneSpine = nullptr;
    {
        GLTFLoader loader;
        auto res = loader.load(std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb");
        if (res) {
            auto& model = res->scene;
            model->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
            playerRig->add(model);
            mixer = std::make_unique<AnimationMixer>(*model);
            auto pick = [&](const std::string& want) -> AnimationAction* {
                for (auto& c : res->animations) {
                    std::string n = c->name();
                    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                    if (n.find(want) != std::string::npos) return mixer->clipAction(c);
                }
                return nullptr;
            };
            idleA = pick("idle");
            walkA = pick("walk");
            runA = pick("run");
            if (!idleA && !res->animations.empty()) idleA = mixer->clipAction(res->animations.front());
            currentA = idleA;
            if (currentA) currentA->play();

            // Find leg/spine bones by case-insensitive Mixamo-name substring
            // ("mixamorigLeftUpLeg", etc.). Note "leftupleg" does not contain
            // "leftleg", so the thigh and shin lookups stay distinct.
            auto findBone = [&](const std::string& sub) -> Object3D* {
                Object3D* found = nullptr;
                model->traverse([&](Object3D& o) {
                    if (found || o.name.empty()) return;
                    std::string n = o.name;
                    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                    if (n.find(sub) != std::string::npos) found = &o;
                });
                return found;
            };
            boneLUpLeg = findBone("leftupleg");
            boneRUpLeg = findBone("rightupleg");
            boneLLeg = findBone("leftleg");
            boneRLeg = findBone("rightleg");
            boneSpine = findBone("spine");

            std::cout << "Loaded soldier with " << res->animations.size() << " clip(s)\n";
        } else {
            std::cerr << "Failed to load Soldier.glb\n";
        }
    }
    // Soldier model faces +Z by default; the camera looks along the player's
    // forward, so we yaw the rig to camera yaw (+offset). Flip by PI if the
    // model ends up back-to-front.
    constexpr float kModelYaw = math::PI;

    // ===== camera rig (third person, mouse look) ============================
    float camYaw = 0.f, camPitch = 0.45f, camDist = 6.f;
    Vector3 playerPos{0, kPlayerHalf, 0};

    // jump-pose state (procedural; see kJump* constants)
    float airTuck = 0.f;   // 0 grounded .. 1 fully tucked (eased)
    float landSquash = 0.f;// brief landing crouch impulse, decays
    bool wasGrounded = true;

    // ===== enemies ==========================================================
    std::vector<std::unique_ptr<Enemy>> enemies;
    auto enemyGeo = CapsuleGeometry::create(0.35f, 1.0f);
    auto headGeo = SphereGeometry::create(0.22f, 16, 12);
    float enemySpawnTimer = 0.f;

    auto spawnEnemy = [&]() {
        const float ang = frand(0.f, 2.f * math::PI);
        const float r = frand(kArena * 0.5f, kArena - 3.f);
        auto e = std::make_unique<Enemy>();
        e->mat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(0xc0392b).roughness(0.6f).metalness(0.1f));
        e->visual = Mesh::create(enemyGeo, e->mat);
        e->visual->position.set(std::cos(ang) * r, 0.9f, std::sin(ang) * r);
        e->visual->castShadow = true;
        auto head = Mesh::create(headGeo, MeshStandardMaterial::create(
                                                  MeshStandardMaterial::Params{}.color(0x2c2c2c).roughness(0.5f)));
        head->position.set(0, 0.62f, 0.05f);
        e->visual->add(head);
        scene->add(e->visual);
        e->body = world.add(*e->visual, 45.f);
        e->body->setRigidDynamicLockFlags(
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
        Enemy* raw = e.get();
        e->body->userData = raw;
        enemies.push_back(std::move(e));
    };

    auto killEnemy = [&](Enemy* e, const Vector3& impulseDir) {
        e->alive = false;
        e->deadTtl = 1.6f;
        e->mat->color = Color(0x3a1512);
        e->body->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(0));// let it tumble
        PxRigidBodyExt::addForceAtPos(*e->body,
                                      toPxVec3((impulseDir + Vector3(0, 0.6f, 0)) * 380.f),
                                      e->body->getGlobalPose().p + PxVec3(0, 0.5f, 0),
                                      PxForceMode::eIMPULSE);
    };

    auto removeEnemy = [&](Enemy* e) {
        world.unbind(*e->visual);
        world.scene().removeActor(*e->body);
        e->body->release();
        scene->remove(*e->visual);
    };

    // ===== transient effects (tracers / flashes / sparks) ===================
    std::vector<Ephemeral> fx;
    auto tracerMat = LineBasicMaterial::create();
    tracerMat->color = Color(0xfff2a0);
    tracerMat->transparent = true;
    tracerMat->depthWrite = false;
    auto flashGeo = SphereGeometry::create(0.16f, 8, 6);

    // ===== game state =======================================================
    int health = 100;
    int ammo = kMagSize;
    int score = 0;
    bool reloading = false;
    float reloadTimer = 0.f;
    float fireTimer = 0.f;
    bool firing = false;    // LMB held
    bool firedEmpty = false;// debounce empty click
    bool gameOver = false;
    float hitMarkerT = 0.f;// >0 while the hit marker flashes
    float stepTimer = 0.f;

    // ===== bullet-impact decals =============================================
    // Reuses threepp's DecalGeometry: each shot projects a scorch decal onto the
    // hit surface (the textured-splat decal asset, tinted dark), parented to the
    // hit mesh so it rides knocked-around crates/barrels.
    TextureLoader texLoader;
    auto decalMat = MeshPhongMaterial::create();
    decalMat->map = texLoader.load(std::string(DATA_FOLDER) + "/textures/decal/decal-diffuse.png", ColorSpace::sRGB);
    decalMat->normalMap = texLoader.load(std::string(DATA_FOLDER) + "/textures/decal/decal-normal.jpg", ColorSpace::NoColorSpace);
    decalMat->normalScale.set(1.f, 1.f);
    decalMat->color = Color(0x161616);// scorch tint
    decalMat->specular = Color(0x222222);
    decalMat->shininess = 18.f;
    decalMat->transparent = true;
    decalMat->depthTest = true;
    decalMat->depthWrite = false;
    decalMat->polygonOffset = true;// lift off the surface to beat z-fighting
    decalMat->polygonOffsetFactor = -4.f;

    std::vector<std::shared_ptr<Mesh>> decals;
    auto stampDecal = [&](Mesh* target, const Vector3& point, const Vector3& worldNormal) {
        if (!target) return;
        target->updateMatrixWorld();

        // Orientation: look from the hit point along the surface normal (same
        // recipe as examples/objects/decal.cpp), plus a random roll for variety.
        Matrix4 helper;
        helper.setPosition(point);
        helper.lookAt(point, point + worldNormal, Vector3::Z());
        Euler orientation;
        orientation.setFromRotationMatrix(helper);
        orientation.z = frand(0.f, math::PI * 2.f);

        const float s = frand(0.26f, 0.40f);
        auto geo = DecalGeometry::create(*target, point, orientation, Vector3(s, s, s));
        // Coplanar with the surface; the decal material's polygonOffset keeps it
        // from z-fighting (honored by both the GL and WGPU backends).
        auto decal = Mesh::create(geo, decalMat);

        // DecalGeometry bakes vertices in WORLD space. Parent to the hit mesh
        // with a local matrix of inverse(target world) so the decal stays glued
        // when the target moves (identity for static surfaces).
        Matrix4 inv;
        inv.copy(*target->matrixWorld).invert();
        decal->matrixAutoUpdate = false;
        decal->matrix->copy(inv);
        decal->matrixWorldNeedsUpdate = true;
        target->add(decal);

        decals.push_back(decal);
        if (static_cast<int>(decals.size()) > kMaxDecals) {
            decals.front()->removeFromParent();
            decals.erase(decals.begin());
        }
    };

    // ===== impact particles (dust on surfaces, blood on enemies) ============
    // A small Points cloud per hit, integrated on the CPU (velocity + gravity +
    // drag) and faded out via material opacity. One draw call per burst; the
    // soft round look comes from the smoke / disc sprite textures.
    auto dustTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png", ColorSpace::sRGB);
    auto bloodTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/sprites/disc.png", ColorSpace::sRGB);

    struct ParticleBurst {
        std::shared_ptr<Points> points;
        std::shared_ptr<PointsMaterial> mat;
        std::shared_ptr<FloatBufferAttribute> posAttr;
        std::vector<Vector3> pos, vel;
        float ttl, life, gravity, drag;
    };
    std::vector<ParticleBurst> bursts;

    auto spawnParticles = [&](const Vector3& point, const Vector3& dir, const Vector3& normal, bool blood) {
        const int count = blood ? 20 : 14;
        ParticleBurst b;
        b.pos.resize(count);
        b.vel.resize(count);
        std::vector<float> flat(count * 3, 0.f);

        // Blood sprays back toward the shooter and out of the surface; dust
        // puffs outward along the surface normal.
        Vector3 base = blood ? (normal * 0.5f - dir * 0.9f) : normal;
        if (base.length() < 1e-4f) base = normal;
        base.normalize();

        for (int i = 0; i < count; ++i) {
            b.pos[i] = point;
            Vector3 v = base + Vector3(frand(-1.f, 1.f), frand(-1.f, 1.f), frand(-1.f, 1.f)) * (blood ? 0.45f : 0.75f);
            if (v.length() < 1e-4f) v = base;
            v.normalize();
            b.vel[i] = v * (blood ? frand(2.5f, 6.5f) : frand(1.2f, 3.2f));
            flat[i * 3 + 0] = point.x;
            flat[i * 3 + 1] = point.y;
            flat[i * 3 + 2] = point.z;
        }

        b.posAttr = FloatBufferAttribute::create(flat, 3);
        b.posAttr->setUsage(DrawUsage::Dynamic);
        auto geo = BufferGeometry::create();
        geo->setAttribute("position", b.posAttr);
        b.mat = PointsMaterial::create(PointsMaterial::Params{}
                                               .color(blood ? Color(0x9a0d0d) : Color(0x9c8f76))
                                               .size(blood ? 0.18f : 0.30f)
                                               .sizeAttenuation(true)
                                               .map(blood ? bloodTex : dustTex));
        b.mat->transparent = true;
        b.mat->depthWrite = false;
        b.points = Points::create(geo, b.mat);
        b.life = b.ttl = blood ? 0.55f : 0.70f;
        b.gravity = blood ? 16.f : 3.5f;
        b.drag = blood ? 1.5f : 3.0f;
        scene->add(b.points);
        bursts.push_back(std::move(b));
    };

    // ===== fire (PhysX raycast) =============================================
    auto fire = [&]() {
        sfx.shot.play();
        ammo--;
        fireTimer = kFireInterval;

        Vector3 pivot = playerPos + Vector3(0, kPlayerLen, 0);
        Vector3 origin = camera->position;
        Vector3 dir = (pivot - origin);
        dir.normalize();

        PxRaycastBuffer hitBuf;
        const bool hasHit = world.scene().raycast(toPxVec3(origin), toPxVec3(dir), 200.f, hitBuf) && hitBuf.hasBlock;

        Vector3 muzzle = playerPos + Vector3(0, kPlayerLen * 0.9f, 0) + dir * 0.6f;
        Vector3 end = hasHit ? fromPxVec3(hitBuf.block.position) : origin + dir * 200.f;

        // tracer
        auto tg = BufferGeometry::create();
        tg->setFromPoints(std::vector<Vector3>{muzzle, end});
        auto tracer = Line::create(tg, tracerMat);
        scene->add(tracer);
        fx.push_back({tracer, 0.05f});

        // muzzle flash
        auto flashMat = MeshBasicMaterial::create();
        flashMat->color = Color(0xffe08a);
        flashMat->transparent = true;
        auto flash = Mesh::create(flashGeo, flashMat);
        flash->position.copy(muzzle);
        scene->add(flash);
        fx.push_back({flash, 0.05f});

        if (!hasHit) return;

        PxRigidActor* actor = hitBuf.block.actor;
        Vector3 point = fromPxVec3(hitBuf.block.position);
        Vector3 hitNormal = fromPxVec3(hitBuf.block.normal);

        // enemy hit? -> blood spray
        if (actor && actor->userData) {
            auto* e = static_cast<Enemy*>(actor->userData);
            if (e->alive) {
                spawnParticles(point, dir, hitNormal, /*blood*/ true);
                e->hp--;
                hitMarkerT = 0.12f;
                if (e->hp <= 0) {
                    killEnemy(e, dir);
                    score += 100;
                    sfx.thud.play();
                } else {
                    sfx.hit.play();
                }
                return;
            }
        }

        // environment hit -> hard metal impact + dust + scorch decal
        sfx.metal.play();
        spawnParticles(point, dir, hitNormal, /*blood*/ false);
        if (auto it = actorToMesh.find(actor); it != actorToMesh.end()) {
            stampDecal(it->second, point, hitNormal);
        }

        // knock the hit body around if it's dynamic
        if (auto* rd = actor ? actor->is<PxRigidDynamic>() : nullptr) {
            PxRigidBodyExt::addForceAtPos(*rd, toPxVec3(dir * 220.f),
                                          hitBuf.block.position, PxForceMode::eIMPULSE);
        }
    };

    // ===== HUD (SVG overlay) ================================================
    auto ui = Scene::create();
    auto sz = canvas.size();
    auto uiCam = OrthographicCamera::create(0, sz.width(), sz.height(), 0, 0.1f, 100);
    uiCam->position.z = 10;
    Layout layout;
    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    // crosshair (4 ticks + dot), recolours on hit
    auto crosshair = Group::create();
    std::vector<std::shared_ptr<MeshBasicMaterial>> chMats;
    {
        auto mk = [&](float w, float h, float x, float y) {
            auto r = rect(w, h, 0xffffff, 0.9f);
            r.mesh->position.set(x - w / 2, y - h / 2, 0);
            crosshair->add(r.mesh);
            chMats.push_back(r.material);
        };
        const float g = 6, len = 8, th = 2;
        mk(th, len, 0, g + th);// up
        mk(th, len, 0, -g);     // down
        mk(len, th, -g - th, th);    // left
        mk(len, th, g + th, th);     // right
        mk(3, 3, 0, 1.5f);      // dot
    }
    ui->add(crosshair);
    layout.add(crosshair, 0.5f, 0.5f, 0, 0, 0.5f);

    // health bar (bottom-left): frame + fill
    {
        auto frame = rect(224, 26, kPanel, 0.7f);
        auto fg = Group::create();
        fg->add(frame.mesh);
        fg->scale.y = -1.f;
        ui->add(fg);
        layout.add(fg, 0.f, 0.f, 22, 56, 0.1f);
    }
    auto healthFill = rect(216, 18, kHudGood, 0.95f);
    {
        auto hg = Group::create();
        hg->add(healthFill.mesh);
        hg->scale.y = -1.f;
        ui->add(hg);
        layout.add(hg, 0.f, 0.f, 26, 52, 0.2f);
    }
    auto healthTxt = makeText(font, "100", 0xffffff, 15 * dpi, 0.f, 0.f, 134, 43,
                              TextSprite::HorizontalAlignment::Center);
    ui->add(healthTxt);
    ui->add(makeText(font, "HP", kHudCyan, 12 * dpi, 0.f, 0.f, 30, 70));

    // ammo (bottom-right)
    Readout ammoTxt{makeText(font, "", 0xffffff, 34 * dpi, 1.f, 0.f, -40, 60,
                             TextSprite::HorizontalAlignment::Right)};
    ui->add(ammoTxt.sprite);
    ui->add(makeText(font, "AMMO", kHudCyan, 13 * dpi, 1.f, 0.f, -40, 92,
                     TextSprite::HorizontalAlignment::Right));
    Readout reloadTxt{makeText(font, "", kHudWarn, 16 * dpi, 1.f, 0.f, -40, 28,
                               TextSprite::HorizontalAlignment::Right)};
    ui->add(reloadTxt.sprite);

    // score + enemies (top)
    Readout scoreTxt{makeText(font, "SCORE 0", kHudCyan, 22 * dpi, 0.f, 1.f, 24, -34)};
    ui->add(scoreTxt.sprite);
    Readout aliveTxt{makeText(font, "", 0xffffff, 16 * dpi, 1.f, 1.f, -24, -34,
                              TextSprite::HorizontalAlignment::Right)};
    ui->add(aliveTxt.sprite);

    // hit marker (centre, flashes)
    auto hitMarker = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><g stroke=")") + hex(kHudWarn) +
                                   R"(" stroke-width="3"><line x1="-11" y1="-11" x2="-4" y2="-4"/><line x1="11" y1="-11" x2="4" y2="-4"/>)" +
                                   R"(<line x1="-11" y1="11" x2="-4" y2="4"/><line x1="11" y1="11" x2="4" y2="4"/></g></svg>)");
    hitMarker->visible = false;
    ui->add(hitMarker);
    layout.add(hitMarker, 0.5f, 0.5f, 0, 0, 0.6f);

    // low-health vignette (full-screen red, opacity tracks damage)
    auto vignette = rect(1, 1, kHudWarn, 0.f);
    vignette.mesh->position.set(0, 0, 0);
    ui->add(vignette.mesh);
    layout.addRaw([m = vignette.mesh](float W, float H) { m->scale.set(W, H, 1); m->position.set(0, 0, 0.05f); });

    // controls hint (bottom-centre)
    ui->add(makeText(font, "WASD move    MOUSE look    LMB fire    SHIFT sprint    SPACE jump    R reload",
                     0x9fb6c8, 12 * dpi, 0.5f, 0.f, 0, 20,
                     TextSprite::HorizontalAlignment::Center));

    // game-over panel (hidden until death)
    auto over = Group::create();
    over->visible = false;
    ui->add(over);
    {
        auto panel = rect(420, 200, kPanel, 0.92f);
        auto pg = Group::create();
        pg->add(panel.mesh);
        pg->scale.y = -1.f;
        over->add(pg);
        layout.add(pg, 0.5f, 0.5f, -210, 100, 0.7f);
    }
    auto overTitle = makeText(font, "GAME OVER", kHudWarn, 40 * dpi, 0.5f, 0.5f, 0, 50,
                              TextSprite::HorizontalAlignment::Center);
    over->add(overTitle);
    Readout overScore{makeText(font, "", 0xffffff, 22 * dpi, 0.5f, 0.5f, 0, 0,
                               TextSprite::HorizontalAlignment::Center)};
    over->add(overScore.sprite);
    over->add(makeText(font, "PRESS ENTER OR CLICK RESTART", kHudCyan, 15 * dpi, 0.5f, 0.5f, 0, -40,
                       TextSprite::HorizontalAlignment::Center));
    // restart hit-target (SpriteInteractor)
    auto restartLabel = makeText(font, "[ RESTART ]", kHudGood, 20 * dpi, 0.5f, 0.5f, 0, -75,
                                 TextSprite::HorizontalAlignment::Center);
    over->add(restartLabel);
    auto restartHitMat = SpriteMaterial::create();
    restartHitMat->visible = false;
    auto restartHit = Sprite::create(restartHitMat);
    restartHit->screenSpace = true;
    restartHit->screenAnchor.set(0.5f, 0.5f);
    restartHit->scale.set(180, 44, 1);
    restartHit->center.set(0.5f, 0.5f);
    over->add(restartHit);

    SpriteInteractor interactor(canvas, *ui);

    // ===== restart ==========================================================
    std::function<void()> restart = [&]() {
        health = 100;
        ammo = kMagSize;
        score = 0;
        reloading = false;
        gameOver = false;
        over->visible = false;
        // clear enemies
        for (auto& e : enemies) removeEnemy(e.get());
        enemies.clear();
        // reset player
        playerBody->setGlobalPose(toPxTransform(Vector3(0, kPlayerHalf, 0)));
        playerBody->setLinearVelocity(PxVec3(0));
        // reset dynamic props
        for (auto& d : dynamics) {
            d.body->setGlobalPose(toPxTransform(d.home));
            d.body->setLinearVelocity(PxVec3(0));
            d.body->setAngularVelocity(PxVec3(0));
            d.body->wakeUp();
        }
        setCursorLocked(true);// re-grab the mouse
    };

    restartHit->onMouseUp = [&](int) {
        if (gameOver) restart();
    };

    // ===== input ============================================================
    // mouse look via frame-to-frame delta (pointer-locked: raw virtual deltas)
    MouseMoveListener look([&](const Vector2& p) {
        if (gameOver) {// don't spin the camera while aiming for the restart button
            lastMouse = p;
            haveMouse = true;
            return;
        }
        if (haveMouse) {
            const float dx = p.x - lastMouse.x;
            const float dy = p.y - lastMouse.y;
            camYaw -= dx * kMouseSens;
            camPitch += dy * kMouseSens;
            camPitch = std::clamp(camPitch, -0.2f, 1.2f);
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

    struct WheelListener: MouseListener {
        std::function<void(float)> f;
        void onMouseWheel(const Vector2& d) override { f(d.y); }
    } wheel;
    wheel.f = [&](float dy) { camDist = std::clamp(camDist - dy * 0.6f, 3.f, 11.f); };
    canvas.addMouseListener(wheel);

    bool jumpQueued = false;
    canvas.onKeyPressed([&](KeyEvent e) {
        if (e.key == Key::R && !reloading && ammo < kMagSize) {
            reloading = true;
            reloadTimer = kReloadTime;
            sfx.reload.play();
        } else if (e.key == Key::SPACE && !gameOver) {
            jumpQueued = true;
        } else if (e.key == Key::ENTER && gameOver) {
            restart();
        }
    });

    // ===== resize ===========================================================
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

    setCursorLocked(true);// grab the mouse for FPS-style look

    // ===== main loop ========================================================
    Clock clock;
    canvas.animate([&] {
        float dt = clock.getDelta();
        if (dt > 0.05f) dt = 0.05f;// clamp big hitches

        // --- read player body pose ---
        const PxTransform pt = playerBody->getGlobalPose();
        playerPos.set(pt.p.x, pt.p.y, pt.p.z);

        // ground check (shared by jump + jump animation)
        PxRaycastBuffer gb;
        const bool grounded = world.scene().raycast(pt.p, PxVec3(0, -1, 0), kPlayerHalf + 0.18f, gb) && gb.hasBlock;

        // --- movement input (camera-relative; ground only — no air control) ---
        float moveSpeed = 0.f;
        if (!gameOver) {
            PxVec3 vel = playerBody->getLinearVelocity();
            if (grounded) {
                const Vector3 F(std::sin(camYaw), 0, std::cos(camYaw));
                const Vector3 R(-std::cos(camYaw), 0, std::sin(camYaw));// screen-right = cross(F, up)
                float fwd = (canvas.isKeyDown(Key::W) ? 1.f : 0.f) - (canvas.isKeyDown(Key::S) ? 1.f : 0.f);
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
            // Airborne: leave horizontal velocity alone so takeoff momentum
            // carries, but the player can't steer or accelerate mid-jump.
            playerBody->setLinearVelocity(vel);
        }
        jumpQueued = false;

        // --- firing ---
        fireTimer -= dt;
        if (reloading) {
            reloadTimer -= dt;
            if (reloadTimer <= 0.f) {
                reloading = false;
                ammo = kMagSize;
            }
        }
        if (firing && !gameOver) {
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

        // --- step physics ---
        world.step(dt);

        // re-read after step for visuals
        const PxTransform pt2 = playerBody->getGlobalPose();
        playerPos.set(pt2.p.x, pt2.p.y, pt2.p.z);

        // --- place + animate soldier ---
        playerRig->position.set(playerPos.x, playerPos.y - kPlayerHalf, playerPos.z);
        playerRig->rotation.y = camYaw + kModelYaw;
        if (mixer) {
            AnimationAction* want = idleA;
            if (moveSpeed > kWalkSpeed + 0.5f) want = runA ? runA : walkA;
            else if (moveSpeed > 0.1f)
                want = walkA ? walkA : idleA;
            if (want && want != currentA) {
                want->reset();
                want->play();
                if (currentA) currentA->crossFadeTo(want, 0.2f);
                currentA = want;
            }
            mixer->update(dt);

            // Procedural jump pose layered on top of the locomotion clip: tuck
            // thighs + bend knees while airborne, deep knee bend on landing.
            // mixer->update just rewrote each bone's quaternion this frame, so
            // multiplying an offset here does not compound across frames.
            if (grounded && !wasGrounded) landSquash = 1.f;// just landed
            airTuck += ((grounded ? 0.f : 1.f) - airTuck) * std::min(1.f, dt * 12.f);
            landSquash = std::max(0.f, landSquash - dt * 4.f);
            wasGrounded = grounded;

            auto poseX = [](Object3D* b, float ang) {
                if (!b) return;
                Quaternion q;
                q.setFromAxisAngle(Vector3::X(), ang);
                b->quaternion.multiply(q);
            };
            poseX(boneLUpLeg, kJumpTuckThigh * airTuck);
            poseX(boneRUpLeg, kJumpTuckThigh * airTuck);
            poseX(boneLLeg, kJumpTuckKnee * airTuck + kJumpLandKnee * landSquash);
            poseX(boneRLeg, kJumpTuckKnee * airTuck + kJumpLandKnee * landSquash);
            poseX(boneSpine, kJumpLean * airTuck);
        }

        // footstep sfx
        if (!gameOver && moveSpeed > 0.1f) {
            stepTimer -= dt;
            if (stepTimer <= 0.f) {
                sfx.step.play();
                stepTimer = moveSpeed > kWalkSpeed + 0.5f ? 0.30f : 0.45f;
            }
        }

        // --- camera rig ---
        const Vector3 pivot = playerPos + Vector3(0, kPlayerLen, 0);
        Vector3 camOff(
                -std::sin(camYaw) * std::cos(camPitch) * camDist,
                std::sin(camPitch) * camDist + 0.4f,
                -std::cos(camYaw) * std::cos(camPitch) * camDist);
        camera->position.copy(pivot + camOff);
        camera->lookAt(pivot);

        // --- enemies ---
        if (!gameOver) {
            enemySpawnTimer -= dt;
            int aliveCount = 0;
            for (auto& e : enemies)
                if (e->alive) aliveCount++;
            if (aliveCount < kMaxEnemies && enemySpawnTimer <= 0.f) {
                spawnEnemy();
                enemySpawnTimer = 1.4f;
            }
        }
        for (auto& e : enemies) {
            const PxVec3 ep = e->body->getGlobalPose().p;
            if (e->alive) {
                Vector3 toPlayer(playerPos.x - ep.x, 0, playerPos.z - ep.z);
                const float d = toPlayer.length();
                PxVec3 v = e->body->getLinearVelocity();
                if (d > kEnemyAttackRange) {
                    toPlayer.normalize();
                    v.x = toPlayer.x * kEnemySpeed;
                    v.z = toPlayer.z * kEnemySpeed;
                } else {
                    v.x = 0;
                    v.z = 0;
                    e->attackCd -= dt;
                    if (e->attackCd <= 0.f && !gameOver) {
                        health -= 9;
                        e->attackCd = 0.8f;
                        sfx.hurt.play();
                        if (health <= 0) {
                            health = 0;
                            gameOver = true;
                            over->visible = true;
                            overScore.set("FINAL SCORE   " + std::to_string(score));
                            setCursorLocked(false);// release the mouse for the restart UI
                        }
                    }
                }
                e->body->setLinearVelocity(v);
            } else {
                e->deadTtl -= dt;
            }
        }
        // remove expired dead enemies
        for (auto it = enemies.begin(); it != enemies.end();) {
            if (!(*it)->alive && (*it)->deadTtl <= 0.f) {
                removeEnemy(it->get());
                it = enemies.erase(it);
            } else {
                ++it;
            }
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

        // --- impact particles (dust / blood) ---
        for (auto it = bursts.begin(); it != bursts.end();) {
            auto& b = *it;
            b.ttl -= dt;
            if (b.ttl <= 0.f) {
                scene->remove(*b.points);
                it = bursts.erase(it);
                continue;
            }
            for (size_t i = 0; i < b.pos.size(); ++i) {
                b.vel[i].y -= b.gravity * dt;
                b.vel[i] *= std::max(0.f, 1.f - b.drag * dt);
                b.pos[i] += b.vel[i] * dt;
                b.posAttr->setXYZ(i, b.pos[i].x, b.pos[i].y, b.pos[i].z);
            }
            b.posAttr->needsUpdate();
            b.mat->opacity = std::clamp(b.ttl / b.life, 0.f, 1.f);
            ++it;
        }

        // --- HUD update ---
        healthFill.mesh->scale.x = std::max(0.001f, static_cast<float>(health) / 100.f);
        healthFill.material->color = Color(health > 50 ? kHudGood : (health > 25 ? 0xffaa33 : kHudWarn));
        healthTxt->setText(std::to_string(health));
        {
            std::ostringstream os;
            os << ammo << " / " << kMagSize;
            ammoTxt.set(os.str());
        }
        reloadTxt.set(reloading ? "RELOADING" : (ammo == 0 ? "EMPTY" : ""));
        scoreTxt.set("SCORE " + std::to_string(score));
        {
            int a = 0;
            for (auto& e : enemies)
                if (e->alive) a++;
            aliveTxt.set("HOSTILES " + std::to_string(a));
        }
        hitMarkerT -= dt;
        hitMarker->visible = hitMarkerT > 0.f;
        vignette.material->opacity = std::clamp((60.f - health) / 60.f, 0.f, 0.55f);
        for (auto& m : chMats) m->color = Color(hitMarkerT > 0.f ? kHudWarn : 0xffffff);

        // ===== render: world, then SVG overlay =====
        renderer->autoClear = true;
        renderer->render(*scene, *camera);
        renderer->autoClear = false;
        renderer->clearDepth();
        renderer->render(*ui, *uiCam);
    });
}
