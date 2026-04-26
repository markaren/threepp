// WgpuPathTracer Showcase — "The Jewel Room"
// ─────────────────────────────────────────────────────────────────────────────
// Demonstrates every major renderer feature:
//   • ReSTIR DI         — 16 ceiling lights + 4 animated orbital emissives
//   • Beer-Lambert glass — amber & emerald spheres cast colored light
//   • Dispersion        — crystal ball shows prismatic colour separation
//   • Clearcoat         — piano-black lacquer sphere
//   • Sheen             — red-velvet torus-knot
//   • Metals            — gold mirror, rough copper, brushed chrome
//   • Many-light scene  — pillars create complex occlusion patterns
//   • Wall art          — gold grille (back), torus mirrors (left), chrome octahedra (right)
//
// Keyboard shortcuts:
//   T  — cycle render mode (PathTracer → Raytracer → Raster)
//   R  — toggle ReSTIR DI
//   D  — toggle denoiser
//   A  — toggle light animation
//   F  — toggle foveated rendering

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

#include <array>
#include <cmath>

using namespace threepp;

namespace {

    // ── Tiny emissive sphere ──────────────────────────────────────────────────────
    std::shared_ptr<Mesh> emissiveBall(const Color& c, float intensity, float r = 0.20f) {
        auto mat = MeshStandardMaterial::create({{"color", c}, {"emissive", c}, {"emissiveIntensity", intensity}, {"roughness", 1.f}});
        return Mesh::create(SphereGeometry::create(r, 16, 16), mat);
    }

    // ── Slim dark-metal pedestal ──────────────────────────────────────────────────
    std::shared_ptr<Mesh> makePedestal() {
        auto mat = MeshStandardMaterial::create({{"color", Color(0.10f, 0.10f, 0.12f)},
                                                 {"roughness", 0.12f},
                                                 {"metalness", 0.85f}});
        return Mesh::create(CylinderGeometry::create(0.36f, 0.48f, 1.5f, 32), mat);
    }

    // ── Room walls / floor / ceiling ──────────────────────────────────────────────
    void buildRoom(Scene& scene) {
        constexpr float W = 26.f, H = 13.f;

        struct Wall {
            float pw, ph;// plane size
            Color col;
            float rough, metal;
            float rx, ry;    // euler rotation
            float px, py, pz;// position
            bool backSide = false;
        };

        const std::array<Wall, 6> walls = {{
                // floor — polished obsidian
                {W, W, Color(0.04f, 0.04f, 0.05f), 0.04f, 0.7f, -math::PI / 2, 0, 0, 0, 0},
                // ceiling — matte charcoal
                {W, W, Color(0.03f, 0.03f, 0.03f), 1.f, 0.f, math::PI / 2, 0, 0, H, 0},
                // back wall — neutral dark
                {W, H, Color(0.11f, 0.11f, 0.13f), 0.88f, 0.f, 0, 0, 0, H / 2, -W / 2},
                // front wall — neutral dark
                {W, H, Color(0.11f, 0.11f, 0.13f), 0.88f, 0.f, 0, 0, 0, H / 2, W / 2, true},
                // left wall — deep crimson (colour bleeding)
                {W, H, Color(0.30f, 0.03f, 0.03f), 0.85f, 0.f, 0, math::PI / 2, -W / 2, H / 2, 0},
                // right wall — deep cobalt
                {W, H, Color(0.03f, 0.05f, 0.32f), 0.85f, 0.f, 0, -math::PI / 2, W / 2, H / 2, 0},
        }};

        for (auto& w : walls) {
            auto mat = MeshStandardMaterial::create({{"color", w.col},
                                                     {"roughness", w.rough},
                                                     {"metalness", w.metal}});
            mat->side = w.backSide ? Side::Back : Side::Front;
            auto m = Mesh::create(PlaneGeometry::create(w.pw, w.ph), mat);
            m->rotation.x = w.rx;
            m->rotation.y = w.ry;
            m->position.set(w.px, w.py, w.pz);
            scene.add(m);
        }
    }

    // ── 4 tall dark pillars for shadow complexity ─────────────────────────────────
    void buildPillars(Scene& scene) {
        constexpr float H = 13.f;
        auto mat = MeshStandardMaterial::create({{"color", Color(0.16f, 0.16f, 0.18f)},
                                                 {"roughness", 0.30f},
                                                 {"metalness", 0.35f}});
        const float xs[] = {-5.5f, 5.5f, -5.5f, 5.5f};
        const float zs[] = {-5.0f, -5.0f, 2.5f, 2.5f};
        for (int i = 0; i < 4; ++i) {
            auto m = Mesh::create(BoxGeometry::create(1.1f, H, 1.1f), mat);
            m->position.set(xs[i], H / 2.f, zs[i]);
            scene.add(m);
        }
    }

    // ── 4 LED strip panels on ceiling ─────────────────────────────────────────────
    // Long thin emissive rectangles — coloured light bars running wall-to-wall.
    // Placed just below the ceiling so they're visible from below and cast clean
    // directional light pools on the floor, unlike point-like spheres.
    void buildCeilingLights(Scene& scene) {
        constexpr float Y = 12.96f;      // just under ceiling (13.0) to avoid z-fight
        constexpr float STRIP_LEN = 22.f;// runs wall-to-wall along X
        constexpr float STRIP_W = 0.55f; // narrow so it reads as a strip

        struct Strip {
            float z;
            Color col;
            float intensity;
        };
        const std::array<Strip, 4> strips = {{
                {-8.0f, Color(1.f, 0.92f, 0.70f), 30.f}, // warm white
                {-1.5f, Color(0.25f, 0.55f, 1.f), 30.f}, // cool blue
                {4.0f, Color(1.f, 0.52f, 0.08f), 30.f},  // amber
                {9.5f, Color(0.22f, 0.90f, 0.38f), 30.f},// green
        }};

        for (auto& s : strips) {
            auto mat = MeshStandardMaterial::create({{"color", s.col},
                                                     {"emissive", s.col},
                                                     {"emissiveIntensity", s.intensity},
                                                     {"roughness", 1.f}});
            auto m = Mesh::create(PlaneGeometry::create(STRIP_LEN, STRIP_W), mat);
            m->position.set(0.f, Y, s.z);
            m->rotation.x = math::PI / 2;// flip normal to face downward into the room
            scene.add(m);
        }
    }

    // ── Central sapphire glass sphere ─────────────────────────────────────────────
    void buildCentreSphere(Scene& scene) {
        auto mat = MeshPhysicalMaterial::create(
                {{"color", Color::white}, {"transmission", 1.f}, {"roughness", 0.f}, {"metalness", 0.f}});
        mat->setIor(1.65f);
        mat->attenuationColor = Color(0.08f, 0.20f, 1.f);
        mat->attenuationDistance = 2.5f;
        auto sphere = Mesh::create(SphereGeometry::create(1.25f, 64, 64), mat);
        sphere->position.set(0.f, 1.25f, -0.5f);
        scene.add(sphere);
    }

    // ── Ring of 8 showcase objects on pedestals ───────────────────────────────────
    void buildShowcaseRing(Scene& scene) {
        constexpr float RING_R = 5.6f;
        constexpr float OBJ_Y = 2.05f;// sphere centre = pedestal top (1.5) + radius (0.55)

        for (int i = 0; i < 8; ++i) {
            const float angle = float(i) / 8.f * 2.f * math::PI;
            const float x = RING_R * std::cos(angle);
            const float z = RING_R * std::sin(angle);

            // Pedestal (top sits at y = 1.5)
            auto ped = makePedestal();
            ped->position.set(x, 0.75f, z);
            scene.add(ped);

            std::shared_ptr<Mesh> obj;

            switch (i) {
                case 0: {
                    // Mirror gold
                    auto m = MeshStandardMaterial::create(
                            {{"color", Color(1.f, 0.84f, 0.f)}, {"roughness", 0.01f}, {"metalness", 1.f}});
                    obj = Mesh::create(SphereGeometry::create(0.55f, 64, 64), m);
                    break;
                }
                case 1: {
                    // Rough copper
                    auto m = MeshStandardMaterial::create(
                            {{"color", Color(0.95f, 0.54f, 0.31f)}, {"roughness", 0.38f}, {"metalness", 1.f}});
                    obj = Mesh::create(SphereGeometry::create(0.55f, 48, 48), m);
                    break;
                }
                case 2: {
                    // Crystal ball — clear glass + strong dispersion (prism effect)
                    auto m = MeshPhysicalMaterial::create({{"color", Color::white},
                                                           {"transmission", 1.f},
                                                           {"roughness", 0.f},
                                                           {"metalness", 0.f}});
                    m->setIor(1.75f);
                    m->dispersion = 0.45f;
                    obj = Mesh::create(SphereGeometry::create(0.55f, 64, 64), m);
                    break;
                }
                case 3: {
                    // Amber glass — Beer-Lambert warm tint
                    auto m = MeshPhysicalMaterial::create({{"color", Color::white},
                                                           {"transmission", 1.f},
                                                           {"roughness", 0.f},
                                                           {"metalness", 0.f}});
                    m->setIor(1.50f);
                    m->attenuationColor = Color(1.f, 0.55f, 0.08f);
                    m->attenuationDistance = 0.9f;
                    obj = Mesh::create(SphereGeometry::create(0.55f, 64, 64), m);
                    break;
                }
                case 4: {
                    // Piano-black clearcoat lacquer
                    auto m = MeshPhysicalMaterial::create({{"color", Color(0.02f, 0.02f, 0.025f)},
                                                           {"roughness", 0.55f},
                                                           {"metalness", 0.f}});
                    m->clearcoat = 1.f;
                    m->clearcoatRoughness = 0.03f;
                    obj = Mesh::create(SphereGeometry::create(0.55f, 48, 48), m);
                    break;
                }
                case 5: {
                    // Emerald glass — Beer-Lambert green tint
                    auto m = MeshPhysicalMaterial::create({{"color", Color::white},
                                                           {"transmission", 1.f},
                                                           {"roughness", 0.f},
                                                           {"metalness", 0.f}});
                    m->setIor(1.58f);
                    m->attenuationColor = Color(0.08f, 1.f, 0.28f);
                    m->attenuationDistance = 0.8f;
                    obj = Mesh::create(SphereGeometry::create(0.55f, 64, 64), m);
                    break;
                }
                case 6: {
                    // Brushed chrome
                    auto m = MeshStandardMaterial::create({{"color", Color(0.84f, 0.84f, 0.87f)},
                                                           {"roughness", 0.26f},
                                                           {"metalness", 1.f}});
                    obj = Mesh::create(SphereGeometry::create(0.55f, 48, 48), m);
                    break;
                }
                case 7: {
                    // Red velvet torus-knot — sheen material
                    auto m = MeshPhysicalMaterial::create({{"color", Color(0.48f, 0.02f, 0.02f)},
                                                           {"roughness", 0.88f},
                                                           {"metalness", 0.f}});
                    m->sheenColor = Color(1.f, 0.12f, 0.12f);
                    m->sheenRoughness = 0.32f;
                    obj = Mesh::create(TorusKnotGeometry::create(0.36f, 0.13f, 128, 24, 2, 3), m);
                    break;
                }
            }

            if (obj) {
                obj->position.set(x, OBJ_Y, z);
                scene.add(obj);
            }
        }
    }

    // ── Wall decorations ──────────────────────────────────────────────────────────
    // Back wall : art-deco gold grille (fins + bars + accent spheres)
    // Left wall : 3 circular gold torus mirrors
    // Right wall: 3 chrome floating octahedra
    void buildWallDecor(Scene& scene) {
        constexpr float W = 26.f;
        constexpr float HALF = W / 2.f;

        // ── Back wall (z = -HALF): art-deco gold grille ───────────────────────────
        {
            auto goldMat = MeshStandardMaterial::create({{"color", Color(0.72f, 0.58f, 0.38f)},
                                                         {"roughness", 0.18f},
                                                         {"metalness", 0.95f}});
            auto accentMat = MeshStandardMaterial::create({{"color", Color(1.0f, 0.88f, 0.55f)},
                                                           {"roughness", 0.05f},
                                                           {"metalness", 1.0f}});
            // 5 vertical fins running floor-to-ceiling
            const float finX[] = {-9.f, -4.5f, 0.f, 4.5f, 9.f};
            for (float x : finX) {
                auto fin = Mesh::create(BoxGeometry::create(0.18f, 10.f, 0.16f), goldMat);
                fin->position.set(x, 5.f, -HALF + 0.1f);
                scene.add(fin);
            }
            // 3 full-width horizontal bars
            const float barY[] = {3.f, 6.f, 9.f};
            for (float y : barY) {
                auto bar = Mesh::create(BoxGeometry::create(22.f, 0.14f, 0.14f), goldMat);
                bar->position.set(0.f, y, -HALF + 0.1f);
                scene.add(bar);
            }
            // Small polished accent spheres at every fin × bar crossing
            for (float x : finX) {
                for (float y : barY) {
                    auto dot = Mesh::create(SphereGeometry::create(0.13f, 16, 16), accentMat);
                    dot->position.set(x, y, -HALF + 0.18f);
                    scene.add(dot);
                }
            }
        }

        // ── Left wall (x = -HALF, crimson): 3 gold torus mirrors ─────────────────
        {
            auto ringMat = MeshStandardMaterial::create({{"color", Color(1.f, 0.84f, 0.f)},
                                                         {"roughness", 0.05f},
                                                         {"metalness", 1.f}});
            auto mirrorMat = MeshStandardMaterial::create({{"color", Color(0.95f, 0.96f, 1.f)},
                                                           {"roughness", 0.01f},
                                                           {"metalness", 1.f}});
            const float zPos[] = {-7.f, 0.f, 7.f};
            for (float z : zPos) {
                // Gold ring frame
                auto ring = Mesh::create(TorusGeometry::create(1.1f, 0.09f, 16, 64), ringMat);
                ring->position.set(-HALF + 0.14f, 7.f, z);
                ring->rotation.y = math::PI / 2.f;
                scene.add(ring);
                // Polished mirror disc inside the ring (cylinder axis along X)
                auto disc = Mesh::create(CylinderGeometry::create(1.0f, 1.0f, 0.04f, 48), mirrorMat);
                disc->position.set(-HALF + 0.1f, 7.f, z);
                disc->rotation.z = math::PI / 2.f;
                scene.add(disc);
            }
        }

        // ── Right wall (x = +HALF, cobalt): 3 chrome octahedra ───────────────────
        {
            auto chromeMat = MeshStandardMaterial::create({{"color", Color(0.84f, 0.84f, 0.87f)},
                                                           {"roughness", 0.06f},
                                                           {"metalness", 1.f}});
            const float zPos[] = {-7.f, 0.f, 7.f};
            for (int i = 0; i < 3; ++i) {
                auto oct = Mesh::create(OctahedronGeometry::create(0.75f), chromeMat);
                oct->position.set(HALF - 1.0f, 7.f, zPos[i]);
                oct->rotation.z = math::PI / 8.f;// slight tilt for dynamism
                oct->rotation.y = float(i) * math::PI / 6.f;
                scene.add(oct);
            }
        }
    }

    // ── 4 orbiting coloured emissive lights ──────────────────────────────────────
    struct Orbiter {
        std::shared_ptr<Mesh> mesh;
        float phase, height, radius;
    };

    std::vector<Orbiter> buildOrbiters(Scene& scene) {
        struct Def {
            Color col;
            float intensity, phase, height, radius;
        };
        const std::array<Def, 4> defs = {{
                {Color(1.f, 0.15f, 0.05f), 14.f, 0.f, 3.8f, 4.2f},
                {Color(0.1f, 0.5f, 1.f), 14.f, math::PI / 2, 5.2f, 3.6f},
                {Color(0.2f, 1.f, 0.30f), 12.f, math::PI, 4.5f, 4.8f},
                {Color(1.f, 0.8f, 0.1f), 12.f, 3 * math::PI / 2, 3.2f, 5.0f},
        }};

        std::vector<Orbiter> orbiters;
        for (auto& d : defs) {
            auto m = emissiveBall(d.col, d.intensity, 0.22f);
            scene.add(m);
            orbiters.push_back({m, d.phase, d.height, d.radius});
        }
        return orbiters;
    }

}// namespace

int main() {

    Canvas canvas("WgpuPathTracer — The Jewel Room",
                  { {"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setMaxBounces(4);
    pathTracer.setDenoiserEnabled(true);
    pathTracer.setReSTIREnabled(true);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setExposure(1.0f);

    // ── Scene ──────────────────────────────────────────────────────────────────
    Scene scene;
    scene.background = Color::aliceblue;

    buildRoom(scene);
    buildPillars(scene);
    buildCeilingLights(scene);
    buildCentreSphere(scene);
    buildShowcaseRing(scene);
    // Orbiters MUST be added before wall decor so their mesh indices stay < 64.
    // The path tracer tracks moved meshes in a 64-bit bitmask; meshes at index ≥ 64
    // are silently ignored, so animated emissives beyond that limit never get their
    // accumulation buffer cleared — they smear and dim after frame 0.
    auto orbiters = buildOrbiters(scene);
    buildWallDecor(scene);

    // Spot lights — used only in raytracer / raster mode for comparison
    auto spot1 = SpotLight::create(Color(1.f, 0.92f, 0.70f), 2.f, 30.f,
                                   math::PI / 7.f, 0.15f, 2.f);
    spot1->position.set(-4.f, 12.f, 2.f);
    spot1->visible = true;
    scene.add(spot1);

    auto spot2 = SpotLight::create(Color(0.40f, 0.60f, 1.f), 1.5f, 30.f,
                                   math::PI / 8.f, 0.20f, 2.f);
    spot2->position.set(4.f, 11.f, -2.f);
    spot2->visible = true;
    scene.add(spot2);

    // ── Camera ─────────────────────────────────────────────────────────────────
    PerspectiveCamera camera(48.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 5.5f, 16.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 3.5f, 0.f);
    controls.update();

    // ── State ──────────────────────────────────────────────────────────────────
    bool restirOn = pathTracer.restirEnabled();
    bool restirGiOn = pathTracer.restirGiEnabled();
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool animating = true;
    bool foveated = pathTracer.foveatedRendering();
    int maxBounces = pathTracer.maxBounces();
    float exposure = 1.0f;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;
    float orbitTime = 0.f;

    KeyAdapter keys(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {

        if (ev.key == Key::R) {
            restirOn = !restirOn;
            pathTracer.setReSTIREnabled(restirOn);
        }
        if (ev.key == Key::G) {
            restirGiOn = !restirGiOn;
            pathTracer.setReSTIRGIEnabled(restirGiOn);
        }
        if (ev.key == Key::D) {
            denoiserOn = !denoiserOn;
            pathTracer.setDenoiserEnabled(denoiserOn);
        }
        if (ev.key == Key::A) { animating = !animating; }
        if (ev.key == Key::F) {
            foveated = !foveated;
            pathTracer.setFoveatedRendering(foveated);
        }
    });
    canvas.addKeyListener(keys);

    // ── ImGui overlay ──────────────────────────────────────────────────────────
    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10});
        ImGui::SetNextWindowSize({290, 0});
        ImGui::Begin("The Jewel Room");

        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frames accumulated: %d", pathTracer.frameCount());
        ImGui::Separator();

        ImGui::TextDisabled("[R] ReSTIR DI [G] ReSTIR GI");
        ImGui::TextDisabled("[A] animate  [F] foveated");
        ImGui::TextDisabled("[D] denoise");
        ImGui::Separator();


        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 4.f))
            pathTracer.setExposure(exposure);

        if (ImGui::Checkbox("ReSTIR DI (R)", &restirOn))
            pathTracer.setReSTIREnabled(restirOn);
        if (ImGui::Checkbox("ReSTIR GI (R)", &restirGiOn))
            pathTracer.setReSTIRGIEnabled(restirGiOn);
        if (ImGui::Checkbox("Denoiser (D)", &denoiserOn))
            pathTracer.setDenoiserEnabled(denoiserOn);
        if (ImGui::Checkbox("Foveated (F)", &foveated))
            pathTracer.setFoveatedRendering(foveated);
        if (ImGui::Checkbox("Animate (A)", &animating)) {}
        if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 12))
            pathTracer.setMaxBounces(maxBounces);


        ImGui::Separator();
        ImGui::TextDisabled("Ring (L to R):");
        ImGui::TextDisabled("  Gold mirror / Rough copper");
        ImGui::TextDisabled("  Crystal (dispersion) / Amber glass");
        ImGui::TextDisabled("  Piano black (clearcoat) / Emerald glass");
        ImGui::TextDisabled("  Chrome / Red velvet (sheen)");
        ImGui::TextDisabled("Centre: Sapphire glass");

        ImGui::End();
    });

    IOCapture io;
    io.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&io);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        pathTracer.setSize({ns.width(), ns.height()});
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    // ── Render loop ────────────────────────────────────────────────────────────
    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = static_cast<float>(fpsFrames) / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        // Animate orbiting emissive lights
        if (animating) {
            orbitTime += dt * 0.35f;
            for (auto& o : orbiters) {
                const float a = orbitTime + o.phase;
                o.mesh->position.set(o.radius * std::cos(a), o.height,
                                     o.radius * std::sin(a));
            }
        }

        controls.update();

        pathTracer.render(scene, camera);

        ui.render();
    });
}
