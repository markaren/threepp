
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Vulkan animation", {{"vsync", false}});

    VulkanRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;

    // ---- Scene objects ----
    TextureLoader tl;
    auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/uv_grid_opengl.jpg", ColorSpace::sRGB);

    auto boxMesh = Mesh::create(
            BoxGeometry::create(1.5f, 1.5f, 1.5f),
            MeshStandardMaterial::create({{"map", tex}, {"roughness", 0.9f}}));
    boxMesh->position.set(-2.8f, 1.f, 3.f);

    auto enclosingBox = Mesh::create(
            BoxGeometry::create(),
            MeshBasicMaterial::create({{"color", Color::black}, {"side", Side::Back}}));
    enclosingBox->scale *= 200;

    auto sphere1 = Mesh::create(
            SphereGeometry::create(0.85f, 32, 32),
            MeshStandardMaterial::create({{"color", Color::orangered},
                                          {"roughness", 0.85f},
                                          {"emissive", Color::orangered},
                                          {"emissiveIntensity", 5.8f}}));
    sphere1->position.set(2.8f, 2.f, -6.f);

    auto sphere2 = Mesh::create(
            SphereGeometry::create(0.85f, 32, 32),
            MeshStandardMaterial::create({{"color", Color::steelblue},
                                          {"roughness", 0.01f},
                                          {"metalness", 0.9f}}));
    sphere2->position.set(2.8f, 1.f, 0.f);

    auto glassSphere = Mesh::create(
            SphereGeometry::create(0.45f, 32, 32),
            MeshPhysicalMaterial::create({{"color", Color::pink},
                                          {"transmission", 0.8f},
                                          {"ior", 1.5f},
                                          {"roughness", 0.1f},
                                          {"metalness", 0.f}}));
    glassSphere->position.set(0.f, 0.2f, 3.f);

    float floorSize = 32.f;
    auto floor = Mesh::create(
            PlaneGeometry::create(floorSize, floorSize),
            MeshStandardMaterial::create({{"color", Color::darkgrey},
                                          {"roughness", 0.99f},
                                          {"side", Side::Double}}));
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -1.f;

    RGBELoader imgLoader;
    auto envMap = imgLoader.load(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr");

    Scene scene;
    scene.background = envMap;
    scene.environment = envMap;

    scene.add(boxMesh);
    scene.add(sphere1);
    scene.add(sphere2);
    scene.add(glassSphere);
    scene.add(floor);
    scene.add(enclosingBox);

    auto grid = GridHelper::create(floorSize);
    grid->position.y = -0.99f;
    scene.add(grid);

    ModelLoader loader;
    auto stormTrooper = loader.load(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");
    stormTrooper->rotation.z = -math::PI;
    stormTrooper->position.z = -4.f;
    stormTrooper->position.y = -1.f;
    stormTrooper->scale *= 0.8;
    scene.add(stormTrooper);

    std::unique_ptr<AnimationMixer> mixer;
    if (!stormTrooper->animations.empty()) {
        std::cout << "Loaded " << stormTrooper->animations.size() << " animation clip(s)." << std::endl;
        mixer = std::make_unique<AnimationMixer>(*stormTrooper);
        mixer->clipAction(stormTrooper->animations.front())->play();
    }

    // ---- Lights ----
    auto pointLight = PointLight::create(Color::white, 0.9f);
    pointLight->castShadow = true;
    pointLight->shadow->bias = -0.005f;
    pointLight->shadow->mapSize.set(1024, 1024);
    pointLight->position.set(5.f, 6.f, -2.f);
    scene.add(pointLight);


    // ---- Camera + controls ----
    PerspectiveCamera rtCam(60.f, canvas.aspect(), 0.1f, 300.f);
    rtCam.position.set(0.f, 3.f, 8.f);
    OrbitControls controls{rtCam, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    controls.update();

    // ---- UI ----
    bool denoiserOn = renderer.denoise();
    bool restirOn = renderer.restirDIEnabled();
    bool animate = true;
    bool showEnclosingBox = true;
    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

    std::cout << "Press T to switch render mode: PathTracer -> Raster" << std::endl;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            // pathTracerOn = !pathTracerOn;
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});
        ImGui::Begin("Vulkan");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();

        ImGui::Checkbox("AnimateBox", &animate);

        if (ImGui::Checkbox("EnclosingBox", &showEnclosingBox)) {
            renderer.resetAccumulation();
        }

        ImGui::SliderFloat("Exposure", &renderer.toneMappingExposure, 0.1f, 2.0f);

        if (ImGui::Checkbox("Denoiser", &denoiserOn)) {
            renderer.setDenoise(denoiserOn);
        }
        if (ImGui::Checkbox("ReSTIR DI", &restirOn)) {
            renderer.setRestirDIEnabled(restirOn);
        }

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        rtCam.aspect = canvas.aspect();
        rtCam.updateProjectionMatrix();
    });

    Clock clock;
    float elapsed = 0.f;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        elapsed += dt;

        enclosingBox->visible = showEnclosingBox;

        // FPS
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (animate) {
            if (mixer) mixer->update(dt);

            boxMesh->rotation.y += dt * 0.6f;
            boxMesh->rotation.x += dt * 0.3f;
        }

        controls.update();

        // renderer.usePathTracer = pathTracerOn;
        renderer.render(scene, rtCam);

        ui.render();
    });
}
