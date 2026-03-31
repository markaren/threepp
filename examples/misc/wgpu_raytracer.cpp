// GPU path tracer example using WgpuPathTracer.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Wgpu Path Tracer",
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", true}});

    WgpuRenderer renderer(canvas);
    renderer.shadowMap().enabled = true;

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(0.5f);
    pathTracer.setSamplesPerPixel(2);
    pathTracer.setDenoiserEnabled(false);

    // ---- Scene objects ----
    TextureLoader tl;
    auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/uv_grid_opengl.jpg");

    auto boxMesh = Mesh::create(
            BoxGeometry::create(1.5f, 1.5f, 1.5f),
            MeshStandardMaterial::create({{"map", tex}, {"roughness", 0.9f}}));
    boxMesh->position.set(-2.8f, 1.f, 3.f);

    auto enclosingBox = Mesh::create(
            BoxGeometry::create(),
            MeshStandardMaterial::create({{"color", Color::black}, {"roughness", 0.9f}, {"side", Side::Back}}));
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

    auto floor = Mesh::create(
            PlaneGeometry::create(16.f, 16.f, 4, 4),
            MeshStandardMaterial::create({{"color", Color::darkgrey},
                                          {"roughness", 0.99f}}));
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -1.f;

    boxMesh->castShadow = true;
    sphere1->castShadow = true;
    sphere2->castShadow = true;
    floor->receiveShadow = true;

    ImageLoader imgLoader;
    auto img = imgLoader.loadHDR(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr", 4, false);
    auto env = Texture::create(*img);

    Scene scene;
    scene.environment = env;

    scene.add(boxMesh);
    scene.add(sphere1);
    scene.add(sphere2);
    scene.add(glassSphere);
    scene.add(floor);
    scene.add(enclosingBox);

    auto grid = GridHelper::create(16);
    grid->position.y = -0.99f;
    scene.add(grid);

    ModelLoader loader;
    auto obj = loader.load(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");
    obj->traverseType<Mesh>([&](Mesh& m) {
        m.castShadow = true;
    });
    obj->position.z = -4.f;
    obj->position.y = -1.f;
    scene.add(obj);

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
    bool pathTracerOn = false;
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool animateBox = true;
    bool showEnclosingBox = true;
    bool raster = false;
    int maxBounces = pathTracer.maxBounces();
    float exposure = pathTracer.exposure();
    float ambientFactor = pathTracer.ambientFactor();

    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

    std::cout << "Press T to switch render mode: Raster -> Raytracer -> PathTracer" << std::endl;
    int renderMode = 0;// 0=raytracer, 1=pathtracer, 2=raster,
    std::vector<std::string> renderModeNames = {"Raytracer", "PathTracer", "Raster"};
    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            renderMode = (renderMode + 1) % 3;
            raster = (renderMode == 2);
            pathTracerOn = (renderMode == 1);
            if (!raster) {
                pathTracer.setMode(pathTracerOn ? WgpuPathTracer::Mode::PathTracer
                                                : WgpuPathTracer::Mode::Raytracer);
            }
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});
        ImGui::Begin(renderModeNames[renderMode].c_str());
        ImGui::Text("FPS: %.1f", fps);
        if (!raster) ImGui::Text("Frames: %d", pathTracer.frameCount());
        ImGui::Separator();

        ImGui::Checkbox("AnimateBox", &animateBox);
        ImGui::Checkbox("EnclosingBox", &showEnclosingBox);

        if (renderMode == 0 || renderMode == 1) {
            if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
                pathTracer.setExposure(exposure);
        }

        if (renderMode == 1 && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 16))
                pathTracer.setMaxBounces(maxBounces);
            if (ImGui::SliderFloat("Ambient", &ambientFactor, 0.0f, 0.2f))
                pathTracer.setAmbientFactor(ambientFactor);
        }

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        pathTracer.setSize({ns.width(), ns.height()});
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

        if (animateBox) {
            boxMesh->rotation.y += dt * 0.6f;
            boxMesh->rotation.x += dt * 0.3f;
        }

        controls.update();

        if (!raster) {
            pathTracer.render(scene, rtCam);
        } else {
            renderer.render(scene, rtCam);
        }

        ui.render();
    });
}
