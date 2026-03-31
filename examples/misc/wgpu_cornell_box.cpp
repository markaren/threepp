// Cornell Box — classic path tracer test scene.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto whiteMat(float roughness = 0.95f) {
        return MeshStandardMaterial::create({{"color", Color(0.73f, 0.73f, 0.73f)}, {"roughness", roughness}});
    }

    // Thin box used as the ceiling light panel (emissive)
    auto makeLightPanel() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color::white},
                {"emissive", Color::white},
                {"emissiveIntensity", 5.0f},
                {"roughness", 1.0f},
        });
        auto mesh = Mesh::create(PlaneGeometry::create(2.6f, 2.6f), mat);
        mesh->rotation.x = math::PI / 2.f;
        mesh->position.set(0.f, 9.99f, 0.f);
        return mesh;
    }

    // Floor, ceiling, back wall (white)
    auto makeRoom() {
        auto group = Group::create();
        constexpr float S = 10.f;

        // Floor
        auto floor = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        floor->rotation.x = -math::PI / 2.f;
        group->add(floor);

        // Ceiling
        auto ceiling = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = S;
        group->add(ceiling);

        // Back wall
        auto back = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        back->position.set(0.f, S / 2.f, -S / 2.f);
        group->add(back);

        // Left wall (red)
        auto leftMat = MeshStandardMaterial::create({{"color", Color(0.65f, 0.05f, 0.05f)}, {"roughness", 0.95f}});
        auto left = Mesh::create(PlaneGeometry::create(S, S), leftMat);
        left->rotation.y = math::PI / 2.f;
        left->position.set(-S / 2.f, S / 2.f, 0.f);
        group->add(left);

        // Right wall (green)
        auto rightMat = MeshStandardMaterial::create({{"color", Color(0.12f, 0.45f, 0.15f)}, {"roughness", 0.95f}});
        auto right = Mesh::create(PlaneGeometry::create(S, S), rightMat);
        right->rotation.y = -math::PI / 2.f;
        right->position.set(S / 2.f, S / 2.f, 0.f);
        group->add(right);

        return group;
    }

    // Tall box (white, slightly rotated)
    auto makeTallBox() {
        auto mesh = Mesh::create(BoxGeometry::create(1.5f, 3.f, 1.5f), whiteMat());
        mesh->position.set(-1.5f, 1.5f, -1.5f);
        mesh->rotation.y = 0.3f;
        return mesh;
    }

    // Short box (white, slightly rotated)
    auto makeShortBox() {
        auto mesh = Mesh::create(BoxGeometry::create(1.5f, 1.5f, 1.5f), whiteMat());
        mesh->position.set(1.5f, 0.75f, 1.0f);
        mesh->rotation.y = -0.3f;
        return mesh;
    }

    // Metal sphere on the short box
    auto makeMetalSphere() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.95f, 0.93f, 0.88f)},
                {"roughness", 0.02f},
                {"metalness", 1.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.6f, 48, 48), mat);
        mesh->position.set(1.5f, 2.1f, 1.0f); // sits on short box (1.5 height + 0.6 radius)
        return mesh;
    }

    // Glass sphere on the floor
    auto makeGlassSphere() {
        auto mat = MeshPhysicalMaterial::create({
                {"color", Color::white},
                {"transmission", 0.95f},
                {"ior", 1.5f},
                {"roughness", 0.0f},
                {"metalness", 0.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.7f, 48, 48), mat);
        mesh->position.set(-1.5f, 0.7f, 1.8f);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("Cornell Box",
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", false}});

    WgpuRenderer renderer(canvas);

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(0.0f);
    pathTracer.setExposure(0.8f);
    pathTracer.setSamplesPerPixel(2);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setMaxBounces(6);
    pathTracer.setMode(WgpuPathTracer::Mode::Raytracer);

    // ---- Scene ----
    Scene scene;
    scene.background = Color::black;

    auto room = makeRoom();
    scene.add(room);
    scene.add(makeLightPanel());
    scene.add(makeTallBox());
    scene.add(makeShortBox());
    scene.add(makeMetalSphere());
    scene.add(makeGlassSphere());

    // Point light for raytracer mode (emissive NEE handles path tracer mode)
    auto light = PointLight::create(Color::white, 1.5f);
    light->position.set(0.f, 9.5f, 0.f);
    scene.add(light);

    // ---- Camera ----
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 5.f, 14.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 4.f, 0.f);
    controls.update();

    // ---- UI ----
    int renderMode = 0;
    std::vector<std::string> renderModeNames = {"Raytracer", "PathTracer", "Raster"};
    bool raster = false;
    bool pathTracerOn = false;
    bool denoiserOn = pathTracer.denoiserEnabled();
    int maxBounces = pathTracer.maxBounces();
    float exposure = pathTracer.exposure();
    float ambientFactor = pathTracer.ambientFactor();
    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

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
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();

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
