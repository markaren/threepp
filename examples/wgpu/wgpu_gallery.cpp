// Material gallery — demanding path tracer stress test.
// Mix of geometries, multiple emissive light sources, and varied materials.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    // --- Materials ---

    auto matChrome() {
        return MeshStandardMaterial::create({
                {"color", Color(0.95f, 0.93f, 0.88f)},
                {"roughness", 0.02f},
                {"metalness", 1.0f},
        });
    }

    auto matGold() {
        return MeshStandardMaterial::create({
                {"color", Color(1.0f, 0.76f, 0.33f)},
                {"roughness", 0.1f},
                {"metalness", 1.0f},
        });
    }

    auto matCopper() {
        return MeshStandardMaterial::create({
                {"color", Color(0.72f, 0.45f, 0.20f)},
                {"roughness", 0.15f},
                {"metalness", 1.0f},
        });
    }

    auto matGlass(Color tint = Color::white) {
        return MeshPhysicalMaterial::create({
                {"color", tint},
                {"transmission", 0.95f},
                {"ior", 1.5f},
                {"roughness", 0.0f},
                {"metalness", 0.0f},
                {"attenuationDistance", 2.0f},
        });
    }

    auto matRoughDiffuse(Color c, float roughness = 0.9f, Side side = Side::Front) {
        return MeshStandardMaterial::create({
                {"color", c},
                {"roughness", roughness},
                {"side", side},
        });
    }

    auto matEmissive(Color c, float intensity) {
        return MeshStandardMaterial::create({
                {"color", c},
                {"emissive", c},
                {"emissiveIntensity", intensity},
                {"roughness", 1.0f},
        });
    }

    // --- Room (open-front box, 20x10x20) ---

    struct RoomResult {
        std::shared_ptr<Group> room;
        std::shared_ptr<Group> windowGroup;
        std::shared_ptr<Mesh> solidWall;
    };

    auto makeRoom() {
        auto group = Group::create();
        constexpr float W = 20.f;// width (x)
        constexpr float H = 10.f;// height (y)
        constexpr float D = 20.f;// depth (z)

        // Floor
        auto floor = Mesh::create(PlaneGeometry::create(W, D), matRoughDiffuse(Color(0.6f, 0.6f, 0.6f), 0.9f, Side::Front));
        floor->rotation.x = -math::PI / 2.f;
        floor->receiveShadow = true;
        group->add(floor);

        // Ceiling
        auto ceiling = Mesh::create(PlaneGeometry::create(W, D), matRoughDiffuse(Color(0.85f, 0.85f, 0.85f), 0.9f, Side::Front));
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = H;
        ceiling->receiveShadow = true;
        group->add(ceiling);

        // Back wall
        auto back = Mesh::create(PlaneGeometry::create(W, H), matRoughDiffuse(Color(0.75f, 0.75f, 0.75f), 0.9f, Side::Front));
        back->position.set(0.f, H / 2.f, -D / 2.f);
        back->receiveShadow = true;
        group->add(back);

        // Front wall
        auto front = Mesh::create(PlaneGeometry::create(W, H), matRoughDiffuse(Color(0.75f, 0.75f, 0.75f), 0.9f, Side::Back));
        front->position.set(0.f, H / 2.f, D / 2.f);
        front->receiveShadow = true;
        group->add(front);

        // Solid left wall (shown when window is off)
        auto wallMat = matRoughDiffuse(Color(0.7f, 0.35f, 0.2f), 0.9f, Side::Front);
        auto leftSolid = Mesh::create(PlaneGeometry::create(D, H), wallMat);
        leftSolid->rotation.y = math::PI / 2.f;
        leftSolid->position.set(-W / 2.f, H / 2.f, 0.f);
        leftSolid->visible = false;
        leftSolid->receiveShadow = true;
        group->add(leftSolid);

        // Left wall with window opening (warm terracotta)
        // Window: 4 wide x 3 tall, centred at y=5, z=2
        constexpr float winW = 4.f, winH = 3.f;
        constexpr float winCY = 5.f, winCZ = 2.f;

        auto windowGroup = Group::create();

        // Below window
        float belowH = winCY - winH / 2.f;
        auto leftBelow = Mesh::create(PlaneGeometry::create(D, belowH), wallMat);
        leftBelow->rotation.y = math::PI / 2.f;
        leftBelow->position.set(-W / 2.f, belowH / 2.f, 0.f);
        windowGroup->add(leftBelow);

        // Above window
        float aboveH = H - (winCY + winH / 2.f);
        auto leftAbove = Mesh::create(PlaneGeometry::create(D, aboveH), wallMat);
        leftAbove->rotation.y = math::PI / 2.f;
        leftAbove->position.set(-W / 2.f, H - aboveH / 2.f, 0.f);
        windowGroup->add(leftAbove);

        // Left of window
        float leftOfW = (D / 2.f) + (winCZ - winW / 2.f);
        auto leftLeft = Mesh::create(PlaneGeometry::create(leftOfW, winH), wallMat);
        leftLeft->rotation.y = math::PI / 2.f;
        leftLeft->position.set(-W / 2.f, winCY, -D / 2.f + leftOfW / 2.f);
        windowGroup->add(leftLeft);

        // Right of window
        float rightOfW = (D / 2.f) - (winCZ + winW / 2.f);
        auto leftRight = Mesh::create(PlaneGeometry::create(rightOfW, winH), wallMat);
        leftRight->rotation.y = math::PI / 2.f;
        leftRight->position.set(-W / 2.f, winCY, D / 2.f - rightOfW / 2.f);
        windowGroup->add(leftRight);

        // Bright emissive panel filling the window opening (flush with wall)
        auto sunPanel = Mesh::create(PlaneGeometry::create(winW + 2, winH + 2),
                                     matEmissive(Color(1.0f, 0.95f, 0.8f), 50.0f));
        sunPanel->rotation.y = math::PI / 2.f;
        sunPanel->position.set(-W / 2.f - 0.1f, winCY, winCZ);
        windowGroup->add(sunPanel);

        // Window cross frames
        constexpr float frameT = 0.08f;
        auto frameMat = matRoughDiffuse(Color(0.15f, 0.12f, 0.1f), 0.8f);

        // Horizontal bar
        auto hBar = Mesh::create(BoxGeometry::create(frameT, winH, frameT), frameMat);
        hBar->rotation.y = math::PI / 2.f;
        hBar->position.set(-W / 2.f + 0.01f, winCY, winCZ);
        windowGroup->add(hBar);

        // Vertical bar
        auto vBar = Mesh::create(BoxGeometry::create(frameT, frameT, winW), frameMat);
        vBar->position.set(-W / 2.f + 0.01f, winCY, winCZ);
        windowGroup->add(vBar);

        group->add(windowGroup);

        // Right wall (cool blue-grey)
        auto right = Mesh::create(PlaneGeometry::create(D, H),
                                  matRoughDiffuse(Color(0.3f, 0.4f, 0.55f), 0.9f, Side::Front));
        right->rotation.y = -math::PI / 2.f;
        right->position.set(W / 2.f, H / 2.f, 0.f);
        right->receiveShadow = true;
        group->add(right);

        return RoomResult{group, windowGroup, leftSolid};
    }

    // --- Object pedestals (simple cylinders) ---

    auto makePedestal(float x, float z, float radius = 0.8f, float height = 0.4f) {
        auto mesh = Mesh::create(CylinderGeometry::create(radius, radius, height, 32),
                                 matRoughDiffuse(Color(0.25f, 0.25f, 0.28f), 0.7f));
        mesh->position.set(x, height / 2.f, z);
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        return mesh;
    }

    // --- Grid of small spheres (geometry stress) ---

    auto makeSphereGrid() {
        auto group = Group::create();
        // 5x5 grid of spheres with varying roughness/metalness
        for (int ix = 0; ix < 5; ix++) {
            for (int iz = 0; iz < 5; iz++) {
                float roughness = static_cast<float>(ix) / 4.f;
                float metalness = static_cast<float>(iz) / 4.f;
                float hue = static_cast<float>(ix * 5 + iz) / 25.f;

                // Simple hue-to-RGB
                Color c;
                c.setHSL(hue, 0.7f, 0.5f);

                auto mat = MeshStandardMaterial::create({
                        {"color", c},
                        {"roughness", roughness},
                        {"metalness", metalness},
                });

                auto sphere = Mesh::create(SphereGeometry::create(0.3f, 24, 24), mat);
                float x = -6.f + ix * 1.0f;
                float z = -6.f + iz * 1.0f;
                sphere->position.set(x, 0.35f, z);
                sphere->castShadow = true;
                sphere->receiveShadow = true;
                group->add(sphere);
            }
        }
        return group;
    }

}// namespace

int main() {

    Canvas canvas("Material Gallery", {{"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.outputEncoding = Encoding::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(0.0f);
    pathTracer.setExposure(1.0f);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setReSTIREnabled(false);
    pathTracer.setMaxBounces(5);

    // ---- Scene ----
    Scene scene;
    scene.background = Color::black;

    auto [room, windowGroup, solidWall] = makeRoom();
    scene.add(room);

    // --- Hero objects on pedestals ---

    // Chrome torus knot (centre-back)
    auto ped1 = makePedestal(0.f, -4.f);
    scene.add(ped1);
    auto torusKnot = Mesh::create(TorusKnotGeometry::create(0.7f, 0.22f, 128, 32), matChrome());
    torusKnot->position.set(0.f, 1.4f, -4.f);
    torusKnot->castShadow = true;
    torusKnot->receiveShadow = true;
    scene.add(torusKnot);

    // Gold torus (near stormtrooper)
    auto ped2 = makePedestal(3.5f, -5.f);
    scene.add(ped2);
    auto torus = Mesh::create(TorusGeometry::create(0.65f, 0.25f, 32, 48), matGold());
    torus->position.set(3.5f, 1.2f, -5.f);
    torus->rotation.x = 0.4f;
    torus->castShadow = true;
    torus->receiveShadow = true;
    scene.add(torus);

    // Copper icosahedron (right)
    auto ped3 = makePedestal(5.f, -1.f);
    scene.add(ped3);
    auto ico = Mesh::create(IcosahedronGeometry::create(0.7f, 4), matCopper());
    ico->position.set(5.f, 1.3f, -1.f);
    ico->castShadow = true;
    ico->receiveShadow = true;
    scene.add(ico);

    // Glass sphere (front-left)
    auto ped4 = makePedestal(-3.f, 4.f);
    scene.add(ped4);
    auto glassBall = Mesh::create(SphereGeometry::create(0.6f, 48, 48), matGlass());
    glassBall->position.set(-3.f, 1.1f, 4.f);
    glassBall->castShadow = true;
    glassBall->receiveShadow = true;
    scene.add(glassBall);

    // Tinted glass sphere (front-right)
    auto ped5 = makePedestal(3.f, 4.f);
    scene.add(ped5);
    auto tintedGlass = Mesh::create(SphereGeometry::create(0.6f, 48, 48), matGlass(Color(0.3f, 0.8f, 0.4f)));
    tintedGlass->position.set(3.f, 1.1f, 4.f);
    tintedGlass->castShadow = true;
    tintedGlass->receiveShadow = true;
    scene.add(tintedGlass);

    // Stormtrooper model (back-right)
    ModelLoader loader;
    auto trooper = loader.load(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");
    trooper->traverseType<Mesh>([&](Mesh& m) {
        m.castShadow = true;
        m.receiveShadow = true;
    });
    trooper->position.set(6.f, 0.f, -6.f);
    trooper->rotation.y = -0.5f;
    scene.add(trooper);

    // Emissive orb (floating, back-left)
    auto emOrb = Mesh::create(BoxGeometry::create(), matEmissive(Color(0.2f, 1.0f, 0.6f), 5.f));
    emOrb->rotateY(math::degToRad(45));
    emOrb->position.set(-9.f, 0.5f, -9.f);
    scene.add(emOrb);

    auto emPoint = PointLight::create(Color(0.2f, 1.0f, 0.6f), 1.0f);
    emPoint->position.copy(emOrb->position);
    emPoint->castShadow = true;
    emPoint->visible = false;
    scene.add(emPoint);

    // Cone (mid-right)
    auto cone = Mesh::create(ConeGeometry::create(0.6f, 1.5f, 32),
                             matRoughDiffuse(Color(0.9f, 0.85f, 0.3f), 0.4f));
    cone->position.set(6.f, 0.75f, 2.f);
    cone->castShadow = true;
    cone->receiveShadow = true;
    scene.add(cone);

    // Capsule (mid-left)
    auto capsule = Mesh::create(CapsuleGeometry::create(0.35f, 1.0f, 16, 24),
                                matRoughDiffuse(Color(0.4f, 0.2f, 0.6f), 0.6f));
    capsule->position.set(-6.f, 0.85f, 2.f);
    capsule->castShadow = true;
    capsule->receiveShadow = true;
    scene.add(capsule);

    // Sphere grid (roughness/metalness matrix)
    scene.add(makeSphereGrid());


    // ---- Camera ----
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 5.f, 16.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 3.f, -1.f);
    controls.update();

    // ---- UI ----
    bool raster = false;
    bool showWindow = true;
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool foveatOn = pathTracer.foveatedRendering();
    bool restirOn = pathTracer.restirEnabled();
    bool restirGiOn = pathTracer.restirGiEnabled();
    int maxBounces = pathTracer.maxBounces();
    float exposure = pathTracer.exposure();
    float pixelScale = pathTracer.pixelScale();
    int aovMode = pathTracer.aovMode();
    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            raster = !raster;
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});
        ImGui::Begin(raster ? "Raster" : "Path Tracer");
        ImGui::Text("FPS: %.1f", fps);
        if (!raster) ImGui::Text("Frames: %d", pathTracer.frameCount());
        ImGui::Separator();

        if (ImGui::Checkbox("Window", &showWindow)) {
            windowGroup->visible = showWindow;
            solidWall->visible = !showWindow;
        }

        if (!raster && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 2.0f))
                pathTracer.setExposure(exposure);
            if (ImGui::SliderFloat("Pixel Scale", &pixelScale, 0.25f, 1.2f, "%.2f"))
                pathTracer.setPixelScale(pixelScale);

            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::Checkbox("Foveat", &foveatOn))
                pathTracer.setFoveatedRendering(foveatOn);
            if (ImGui::Checkbox("ReSTIR DI", &restirOn))
                pathTracer.setReSTIREnabled(restirOn);
            if (ImGui::Checkbox("ReSTIR GI", &restirGiOn))
                pathTracer.setReSTIRGIEnabled(restirGiOn);
            if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 6))
                pathTracer.setMaxBounces(maxBounces);

            ImGui::Separator();
            ImGui::TextUnformatted("AOV debug");
            // Labels align with WgpuPathTracerShaders_Rt.cpp aovMode handling.
            static const char* kAovLabels[] = {
                "Off",                   //  0
                "Depth",                 //  1
                "Normals",               //  2
                "Albedo",                //  3
                "Instance ID",           //  4
                "Roughness",             //  5
                "Adaptive bounce",       //  6
                "(reserved 7)",          //  7
                "(reserved 8)",          //  8
                "(reserved 9)",          //  9
                "diffRadFinal (w3.xyz)", // 10
                "specRadFinal (w4.xyz)", // 11
                "touchedMoved bit",      // 12
                "flagBits (RGB)",        // 13
                "b0Point fract",         // 14
                "primaryDepth norm",     // 15
                "primaryMeshIdx",        // 16
                "primaryMatIdx",         // 17
                "b0Alpha (rough²)",      // 18
            };
            constexpr int kNumAov = std::size(kAovLabels);
            if (aovMode < 0) aovMode = 0;
            if (aovMode >= kNumAov) aovMode = 0;
            if (ImGui::Combo("AOV", &aovMode, kAovLabels, kNumAov)) {
                pathTracer.setAOVMode(aovMode);
            }
        }

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = []() -> bool { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        pathTracer.setSize({ns.width(), ns.height()});
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();

        emPoint->visible = raster;
        emOrb->visible = !emPoint->visible;

        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        controls.update();

        if (!raster) {
            pathTracer.render(scene, camera);
        } else {
            renderer.render(scene, camera);
        }

        ui.render();
    });
}
