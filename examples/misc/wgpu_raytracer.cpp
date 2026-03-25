// GPU path tracer example using WgpuPathTracer.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Wgpu Path Tracer",
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.shadowMap().enabled = true;
    renderer.setClearColor(Color(0x000000));

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setSamplesPerPixel(2);

    // ---- Scene objects ----
    TextureLoader tl;
    auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/uv_grid_opengl.jpg");

    auto boxMesh = Mesh::create(
            BoxGeometry::create(1.5f, 1.5f, 1.5f),
            MeshStandardMaterial::create({{"map", tex}, {"roughness", 0.9f}}));
    boxMesh->position.set(0.f, 1.f, -1.f);

    auto enclosingBox = Mesh::create(
            BoxGeometry::create(),
            MeshStandardMaterial::create({{"color", Color::black}, {"roughness", 0.9f}}));
    enclosingBox->scale *= 1000;

    auto sphere1 = Mesh::create(
            SphereGeometry::create(0.85f, 32, 32),
            MeshStandardMaterial::create({{"color", Color::orangered},
                                          {"roughness", 0.85f},
                                          {"emissive", Color::orangered},
                                          {"emissiveIntensity", 0.9f}}));
    sphere1->position.set(-2.8f, 1.f, 0.f);

    auto sphere2 = Mesh::create(
            SphereGeometry::create(0.85f, 32, 32),
            MeshStandardMaterial::create({{"color", Color::steelblue},
                                          {"roughness", 0.1f},
                                          {"metalness", 0.9f}}));
    sphere2->position.set(2.8f, 1.f, 0.f);

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

    Scene scene;
    scene.add(boxMesh);
    scene.add(sphere1);
    scene.add(sphere2);
    scene.add(floor);
    scene.add(enclosingBox);

    ModelLoader loader;
    auto obj = loader.load(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");
    obj->traverseType<Mesh>([&](Mesh& m) {
        m.castShadow = true;
        auto texMat = m.material()->as<MaterialWithMap>();
        m.setMaterial(MeshStandardMaterial::create({{"map", texMat ? texMat->map : nullptr}, {"roughness", 0.9f}}));
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

    auto pointLight2 = PointLight::create(Color::white, 0.4f);
    pointLight2->castShadow = true;
    pointLight2->shadow->bias = -0.005f;
    pointLight2->shadow->mapSize.set(1024, 1024);
    pointLight2->position.set(-5.f, 6.f, 4.f);
    scene.add(pointLight2);

    // ---- Camera + controls ----
    PerspectiveCamera rtCam(60.f, canvas.aspect(), 0.1f, 200.f);
    rtCam.position.set(0.f, 3.f, 8.f);
    OrbitControls controls{rtCam, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    controls.update();

    // ---- UI ----
    bool pathTracerOn = false;
    bool lightMoving = false;
    bool animateBox = true;
    bool showEnclosingBox = true;
    bool raster = false;

    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            pathTracerOn = !pathTracerOn;
            pathTracer.setMode(pathTracerOn ? WgpuPathTracer::Mode::PathTracer
                                            : WgpuPathTracer::Mode::Raytracer);
        } else if (ev.key == Key::R) {
            raster = !raster;
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({220, 0}, ImGuiCond_Once);
        ImGui::Begin("Raytracer");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frames: %d", pathTracer.frameCount());
        ImGui::Separator();
        if (ImGui::Checkbox("Path tracer (T)", &pathTracerOn))
            pathTracer.setMode(pathTracerOn ? WgpuPathTracer::Mode::PathTracer
                                            : WgpuPathTracer::Mode::Raytracer);
        ImGui::Checkbox("Moving light", &lightMoving);
        ImGui::Checkbox("AnimateBox", &animateBox);
        ImGui::Checkbox("EnclosingBox", &showEnclosingBox);
        ImGui::Checkbox("Raster (R)", &raster);
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
        if (lightMoving) {
            pointLight->position.set(
                    5.f * std::cos(elapsed * 0.6f),
                    6.f + std::sin(elapsed * 0.3f),
                    -2.f + 4.f * std::sin(elapsed * 0.6f));
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
