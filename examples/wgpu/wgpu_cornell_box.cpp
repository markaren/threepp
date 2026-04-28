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
                {"emissiveIntensity", 16.0f},
                {"roughness", 1.0f},
        });
        auto mesh = Mesh::create(PlaneGeometry::create(2.6f, 2.6f), mat);
        mesh->rotation.x = math::PI / 2.f;
        mesh->position.set(0.f, 9.99f, 0.f);
        return mesh;
    }

    // Floor, ceiling, back wall (white).  Returns the back wall separately so
    // the animate loop can wobble its vertices to exercise the path tracer's
    // per-frame geometry-change fast path.
    struct Room {
        std::shared_ptr<Group> group;
        std::shared_ptr<Mesh> backWall;
    };
    Room makeRoom() {
        auto group = Group::create();
        constexpr float S = 10.f;

        // Floor
        auto floor = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        floor->rotation.x = -math::PI / 2.f;
        floor->receiveShadow = true;
        group->add(floor);

        // Ceiling
        auto ceiling = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = S;
        group->add(ceiling);

        // Back wall — subdivided so per-vertex wobble is visible.
        auto back = Mesh::create(PlaneGeometry::create(S, S, 40, 40), whiteMat());
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

        return {group, back};
    }

    // Tall box (white, slightly rotated)
    auto makeTallBox() {
        auto mesh = Mesh::create(BoxGeometry::create(1.5f, 3.f, 1.5f), whiteMat());
        mesh->position.set(-1.5f, 1.5f, -1.5f);
        mesh->rotation.y = 0.3f;
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        return mesh;
    }

    // Short box (white, slightly rotated)
    auto makeShortBox() {
        auto mesh = Mesh::create(BoxGeometry::create(1.5f, 1.5f, 1.5f), whiteMat());
        mesh->position.set(1.5f, 0.75f, 1.0f);
        mesh->rotation.y = -0.3f;
        mesh->castShadow = true;
        mesh->receiveShadow = true;
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
        mesh->castShadow = true;
        mesh->receiveShadow = true;
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
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("Cornell Box",
                  { {"vsync", false}});

    WgpuRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.shadowMap().enabled = true;

    auto& pathTracer = renderer.pathTracer();
    pathTracer.setExposure(0.8f);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setMaxBounces(6);
    pathTracer.setReSTIREnabled(true);
    pathTracer.setReSTIRGIEnabled(false);
    pathTracer.setFoveatedRendering(false);
    pathTracer.setFireflyClamp(0);

    // ---- Scene ----
    Scene scene;
    scene.background = Color::black;

    auto room = makeRoom();
    scene.add(room.group);
    scene.add(makeLightPanel());
    scene.add(makeTallBox());
    scene.add(makeShortBox());
    scene.add(makeMetalSphere());
    scene.add(makeGlassSphere());

    // Point light for raytracer mode (emissive NEE handles path tracer mode)
    auto light = PointLight::create(Color::white, 0.5f);
    light->position.set(0.f, 9.5f, 0.f);
    light->castShadow = true;
    scene.add(light);

    // ---- Camera ----
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 5.f, 14.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 4.f, 0.f);
    controls.update();

    // ---- UI ----
    bool raster = false;
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool restdirOn = pathTracer.restirEnabled();
    bool restirGIOn = pathTracer.restirGiEnabled();
    bool foveatOn = pathTracer.foveatedRendering();
    bool tlasOn = pathTracer.tlasEnabled();
    bool wobbleOn = false;
    int maxBounces = pathTracer.maxBounces();
    float exposure = pathTracer.exposure();
    bool fogOn = false;
    float fogDensity = 0.08f;
    float fogColor[3] = {0.5f, 0.5f, 0.6f};
    float fogG = 0.0f;  // HG anisotropy: >0 forward ("god rays"), <0 back
    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

    // Snapshot the back wall's original z positions so per-frame wobble stays
    // bounded and can be toggled off to restore rest pose.
    auto backPos = room.backWall->geometry()->getAttribute<float>("position");
    std::vector<float> backWallZ0(backPos->count());
    for (auto i = 0; i < backPos->count(); ++i) backWallZ0[i] = backPos->getZ(i);
    float wobbleT = 0.f;

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

        if (!raster && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
                pathTracer.setExposure(exposure);
            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::Checkbox("ReSTIR DI", &restdirOn))
                pathTracer.setReSTIREnabled(restdirOn);
            if (ImGui::Checkbox("ReSTIR GI", &restirGIOn))
                pathTracer.setReSTIRGIEnabled(restirGIOn);
            if (ImGui::Checkbox("Foveated", &foveatOn))
                pathTracer.setFoveatedRendering(foveatOn);
            if (ImGui::Checkbox("TLAS", &tlasOn))
                pathTracer.setTlasEnabled(tlasOn);
            ImGui::Checkbox("Wobble back wall", &wobbleOn);
            if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 8))
                pathTracer.setMaxBounces(maxBounces);
            ImGui::Separator();
            ImGui::Checkbox("Fog", &fogOn);
            if (fogOn) {
                ImGui::SliderFloat("Fog density", &fogDensity, 0.001f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::ColorEdit3("Fog color", fogColor);
                ImGui::SliderFloat("Fog anisotropy g", &fogG, -0.9f, 0.9f, "%.2f");
            }
        }

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();

        light->visible = raster;

        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        // Animate back wall vertices — exercises the path tracer's per-frame
        // geometry fast path (CPU repack + partial upload + BVH/BLAS refit).
        if (wobbleOn) {
            wobbleT += dt;
            const float wave = std::sin(math::TWO_PI * 0.5f * wobbleT);
            for (auto i = 0; i < backPos->count(); ++i) {
                backPos->setZ(i, backWallZ0[i] + 0.3f * wave * std::sin(i * 0.3f));
            }
            backPos->needsUpdate();
        }

        controls.update();

        if (fogOn) {
            scene.fog = FogExp2(Color(fogColor[0], fogColor[1], fogColor[2]), fogDensity);
            pathTracer.setFogAnisotropy(fogG);
        } else {
            scene.fog.reset();
        }

        renderer.usePathTracer = !raster;
        renderer.render(scene, camera);

        ui.render();
    });
}
