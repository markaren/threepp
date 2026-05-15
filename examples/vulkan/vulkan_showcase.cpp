// Vulkan PT showcase — Cornell-style stage demonstrating the renderer's
// path-traced features: PBR lobes (diffuse / metal / glass / clearcoat),
// emissive area lights via NEE, alpha-test cutouts via any-hit, color bleed
// from coloured walls, and per-pixel temporal accumulation under camera and
// mesh motion. An animated torus knot exercises the per-pixel motion gate
// landed on this branch — only its pixels reset accumulation; the static
// walls keep converging.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <iostream>
#include <string>

using namespace threepp;

namespace {

    auto whiteWallMat() {
        return MeshStandardMaterial::create({{"color", Color(0.73f, 0.73f, 0.73f)}, {"roughness", 0.95f}});
    }

    struct Room {
        std::shared_ptr<Group> group;
    };

    Room makeRoom() {
        auto group = Group::create();
        constexpr float S = 10.f;

        auto floor = Mesh::create(PlaneGeometry::create(S, S), whiteWallMat());
        floor->rotation.x = -math::PI / 2.f;
        group->add(floor);

        auto ceiling = Mesh::create(PlaneGeometry::create(S, S), whiteWallMat());
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = S;
        group->add(ceiling);

        auto back = Mesh::create(PlaneGeometry::create(S, S), whiteWallMat());
        back->position.set(0.f, S / 2.f, -S / 2.f);
        group->add(back);

        auto leftMat = MeshStandardMaterial::create({{"color", Color(0.65f, 0.05f, 0.05f)}, {"roughness", 0.95f}});
        auto left = Mesh::create(PlaneGeometry::create(S, S), leftMat);
        left->rotation.y = math::PI / 2.f;
        left->position.set(-S / 2.f, S / 2.f, 0.f);
        group->add(left);

        auto rightMat = MeshStandardMaterial::create({{"color", Color(0.12f, 0.45f, 0.15f)}, {"roughness", 0.95f}});
        auto right = Mesh::create(PlaneGeometry::create(S, S), rightMat);
        right->rotation.y = -math::PI / 2.f;
        right->position.set(S / 2.f, S / 2.f, 0.f);
        group->add(right);

        return {group};
    }

    // Bright emissive ceiling panel — primary area light for NEE.
    auto makeLightPanel() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color::white},
                {"emissive", Color::white},
                {"emissiveIntensity", 18.0f},
                {"roughness", 1.0f},
        });
        auto mesh = Mesh::create(PlaneGeometry::create(3.0f, 3.0f), mat);
        mesh->rotation.x = math::PI / 2.f;
        mesh->position.set(0.f, 9.99f, 0.f);
        return mesh;
    }

    // Materials gallery: 6 spheres in a row showing the BSDF lobes.
    std::vector<std::shared_ptr<Mesh>> makeMaterialGallery() {
        std::vector<std::shared_ptr<Mesh>> out;
        auto sphereGeom = SphereGeometry::create(0.55f, 64, 48);
        const float y = 0.55f;
        const float dx = 1.4f;
        const float x0 = -dx * 2.5f;

        // 1. Rough white diffuse
        {
            auto mat = MeshStandardMaterial::create({
                    {"color", Color(0.85f, 0.85f, 0.85f)},
                    {"roughness", 0.95f},
                    {"metalness", 0.0f},
            });
            auto m = Mesh::create(sphereGeom, mat);
            m->position.set(x0 + 0 * dx, y, 2.5f);
            out.push_back(m);
        }
        // 2. Rough gold metal
        {
            auto mat = MeshStandardMaterial::create({
                    {"color", Color(1.0f, 0.78f, 0.32f)},
                    {"roughness", 0.25f},
                    {"metalness", 1.0f},
            });
            auto m = Mesh::create(sphereGeom, mat);
            m->position.set(x0 + 1 * dx, y, 2.5f);
            out.push_back(m);
        }
        // 3. Smooth chrome mirror
        {
            auto mat = MeshStandardMaterial::create({
                    {"color", Color(0.97f, 0.97f, 0.97f)},
                    {"roughness", 0.02f},
                    {"metalness", 1.0f},
            });
            auto m = Mesh::create(sphereGeom, mat);
            m->position.set(x0 + 2 * dx, y, 2.5f);
            out.push_back(m);
        }
        // 4. Glass (transmission + IOR)
        {
            auto mat = MeshPhysicalMaterial::create({
                    {"color", Color::white},
                    {"transmission", 1.0f},
                    {"roughness", 0.0f},
                    {"metalness", 0.0f},
            });
            mat->setIor(1.5f);
            auto m = Mesh::create(sphereGeom, mat);
            m->position.set(x0 + 3 * dx, y, 2.5f);
            out.push_back(m);
        }
        // 5. Red clearcoat (car-paint)
        {
            auto mat = MeshPhysicalMaterial::create({
                    {"color", Color(0.65f, 0.05f, 0.05f)},
                    {"roughness", 0.4f},
                    {"metalness", 0.0f},
                    {"clearcoat", 1.0f},
                    {"clearcoatRoughness", 0.05f},
            });
            auto m = Mesh::create(sphereGeom, mat);
            m->position.set(x0 + 4 * dx, y, 2.5f);
            out.push_back(m);
        }
        // 6. Cyan emissive
        {
            auto mat = MeshStandardMaterial::create({
                    {"color", Color::black},
                    {"emissive", Color(0.1f, 0.85f, 1.0f)},
                    {"emissiveIntensity", 4.0f},
                    {"roughness", 1.0f},
            });
            auto m = Mesh::create(sphereGeom, mat);
            m->position.set(x0 + 5 * dx, y, 2.5f);
            out.push_back(m);
        }
        return out;
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT — Showcase",
                  {{"vsync", false}, {"size", WindowSize{1600, 1000}}});

    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    scene.background = Color::black;

    auto room = makeRoom();
    scene.add(room.group);
    scene.add(makeLightPanel());

    auto gallery = makeMaterialGallery();
    for (auto& m : gallery) scene.add(m);

    // Animated rotating torus knot — exercises the per-pixel motion gate.
    // Static walls behind it should keep accumulating cleanly while only
    // the moving pixels reset their FC.
    auto knotMat = MeshPhysicalMaterial::create({
            {"color", Color(0.95f, 0.93f, 0.88f)},
            {"roughness", 0.15f},
            {"metalness", 1.0f},
    });
    auto knot = Mesh::create(TorusKnotGeometry::create(0.55f, 0.18f, 128, 24), knotMat);
    knot->position.set(0.f, 4.5f, -1.0f);
    scene.add(knot);

    // Suzanne — textured PBR (baseColor + metallicRoughness maps) exercises
    // the bindless texture path that the abstract sphere materials skip.
    // Tilted onto her left ear so she's lying on her side on a small pedestal.
    GLTFLoader gltfLoader;
    auto suzanneScene = gltfLoader.load(
            std::string(DATA_FOLDER) + "/models/gltf/Suzanne/glTF/Suzanne.gltf");
    if (suzanneScene && suzanneScene->scene) {
        auto pedestal = Mesh::create(BoxGeometry::create(2.6f, 0.2f, 1.6f),
                                     whiteWallMat());
        pedestal->position.set(0.f, 0.1f, -2.0f);
        scene.add(pedestal);

        auto suz = suzanneScene->scene;
        // +π/4 around Z bisects the (left-ear, chin) corner — head tilted
        // 45° so both the side of the jaw and the ear contact the surface.
        suz->rotation.z = math::PI / 4.f;
        suz->position.set(0.f, 0.f, -2.f);
        scene.add(suz);

        // Snap the rolled bbox bottom onto the pedestal top so the ear
        // touches the surface regardless of the asset's exact bbox.
        Box3 bbox;
        bbox.setFromObject(*suz);
        const float pedestalTop = 0.2f;
        suz->position.y += pedestalTop - bbox.min().y /2;
    } else {
        std::cerr << "Failed to load Suzanne model" << std::endl;
    }

    // Alpha-test cutout — exercises the any-hit shader. The three.js logo
    // becomes a hard-edged silhouette with a clean ground shadow, since the
    // shadow-ray any-hit also fires on cutout textures.
    TextureLoader texLoader;
    auto cutoutTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/three.png", SRGBColorSpace);
    auto cutoutMat = MeshStandardMaterial::create({
            {"color", Color::white},
            {"roughness", 0.6f},
            {"metalness", 0.0f},
    });
    cutoutMat->map = cutoutTex;
    cutoutMat->alphaTest = 0.5f;
    cutoutMat->side = Side::Double;
    auto cutoutPlane = Mesh::create(PlaneGeometry::create(2.0f, 2.0f), cutoutMat);
    cutoutPlane->position.set(-3.2f, 1.0f, 0.0f);
    cutoutPlane->rotation.y = 0.4f;
    scene.add(cutoutPlane);

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 4.5f, 11.5f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 3.5f, 0.f);
    controls.update();

    bool spin = true;
    float exposure = renderer.toneMappingExposure;
    int toneMode = static_cast<int>(renderer.toneMapping);
    int spp = renderer.samplesPerPixel();
    float renderScale = renderer.renderScale();
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({320, 0});
        ImGui::Begin("Vulkan PT - Showcase");
        ImGui::Text("FPS: %.1f", fps);

        ImGui::Separator();
        ImGui::TextWrapped("Cornell-style stage: red/green walls bleed "
                           "colour onto neighbouring surfaces, ceiling panel "
                           "drives NEE, glass refracts via transmission, "
                           "the torus knot animates to demo per-pixel motion "
                           "gating, and the three.js cutout exercises the "
                           "any-hit alpha test.");
        ImGui::Separator();

        ImGui::Checkbox("Animate torus knot", &spin);

        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
            renderer.toneMappingExposure = exposure;

        const char* toneItems[] = {"None", "Linear", "Reinhard", "Cineon", "ACESFilmic"};
        if (ImGui::Combo("Tone mapping", &toneMode, toneItems, IM_ARRAYSIZE(toneItems)))
            renderer.toneMapping = static_cast<ToneMapping>(toneMode);

        bool restirDI = renderer.restirDIEnabled();
        if (ImGui::Checkbox("ReSTIR DI", &restirDI))
            renderer.setRestirDIEnabled(restirDI);
        bool restirGI = renderer.restirGIEnabled();
        if (ImGui::Checkbox("ReSTIR GI", &restirGI))
            renderer.setRestirGIEnabled(restirGI);

        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 16))
            renderer.setSamplesPerPixel(spp);

        // Path-trace render scale: < 1 traces fewer pixels, then upscales.
        if (ImGui::SliderFloat("Render scale", &renderScale, 0.25f, 1.0f, "%.2f"))
            renderer.setRenderScale(renderScale);

        ImGui::Separator();
        ImGui::TextDisabled("Drag = orbit, scroll = zoom");
        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = []() -> bool { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;
    float t = 0.f;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (spin) {
            t += dt;
            knot->rotation.y = t * 0.6f;
            knot->rotation.x = t * 0.3f;
        }

        controls.update();
        renderer.render(scene, camera);
        ui.render();
    });

    return 0;
}
