
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

int main() {

    Canvas canvas("Transmission", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 0, 8);

    scene->add(AmbientLight::create(0xffffff, 0.3f));
    auto dirLight = DirectionalLight::create(0xffffff, 1.5f);
    dirLight->position.set(5, 5, 5);
    scene->add(dirLight);

    RGBELoader hdrLoader;
    if (auto hdrTex = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr")) {
        scene->background = hdrTex;
        scene->environment = hdrTex;
    }

    // Colorful opaque background objects
    std::array<Color, 6> colors = {
            Color(0xe74c3c), Color(0x3498db), Color(0x2ecc71),
            Color(0xf39c12), Color(0x9b59b6), Color(0x1abc9c)};

    std::vector<std::shared_ptr<Mesh>> boxes;
    auto boxGeo = BoxGeometry::create(1.2f, 1.2f, 1.2f);
    for (int i = 0; i < 6; ++i) {
        auto mat = MeshStandardMaterial::create();
        mat->color = colors[i];
        mat->roughness = 0.5f;
        auto box = Mesh::create(boxGeo, mat);
        float angle = static_cast<float>(i) / 6.f * math::TWO_PI;
        box->position.set(std::cos(angle) * 3.0f, std::sin(angle) * 3.0f, -2.0f);
        boxes.push_back(box);
        scene->add(box);
    }

    // Glass sphere
    auto sphereGeo = SphereGeometry::create(1.5f, 64, 32);
    auto glassMat = MeshPhysicalMaterial::create();
    glassMat->transmission = 1.0f;
    glassMat->roughness = 0.0f;
    glassMat->metalness = 0.0f;
    glassMat->setIor(1.5f);
    glassMat->thickness = 2.0f;
    auto sphere = Mesh::create(sphereGeo, glassMat);
    scene->add(sphere);

    // ImGui params (local copies for sliders)
    float transmission = glassMat->transmission;
    float roughness = glassMat->roughness;
    float metalness = glassMat->metalness;
    float ior = glassMat->ior;
    float thickness = glassMat->thickness;
    float attenuationDistance = glassMat->attenuationDistance;
    std::array<float, 3> attenuationColor = {1.f, 1.f, 1.f};
    float exposure = renderer.toneMappingExposure;

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({250 * ui.dpiScale(), 0}, 0);
        ImGui::Begin("Transmission");

        if (ImGui::SliderFloat("Transmission", &transmission, 0.f, 1.f)) {
            glassMat->transmission = transmission;
            glassMat->needsUpdate();
        }
        if (ImGui::SliderFloat("Roughness", &roughness, 0.f, 1.f)) {
            glassMat->roughness = roughness;
            glassMat->needsUpdate();
        }
        if (ImGui::SliderFloat("Metalness", &metalness, 0.f, 1.f)) {
            glassMat->metalness = metalness;
            glassMat->needsUpdate();
        }
        if (ImGui::SliderFloat("IOR", &ior, 1.f, 2.333f)) {
            glassMat->setIor(ior);
            glassMat->needsUpdate();
        }
        if (ImGui::SliderFloat("Thickness", &thickness, 0.f, 5.f)) {
            glassMat->thickness = thickness;
            glassMat->needsUpdate();
        }
        if (ImGui::SliderFloat("Attenuation Dist.", &attenuationDistance, 0.f, 10.f)) {
            glassMat->attenuationDistance = attenuationDistance;
            glassMat->needsUpdate();
        }
        if (ImGui::ColorEdit3("Attenuation Color", attenuationColor.data())) {
            glassMat->attenuationColor.setRGB(attenuationColor[0], attenuationColor[1], attenuationColor[2]);
            glassMat->needsUpdate();
        }
        ImGui::Separator();
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 3.f)) {
            renderer.toneMappingExposure = exposure;
        }

        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    OrbitControls controls(*camera, canvas);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        auto t = clock.getElapsedTime();
        for (size_t i = 0; i < boxes.size(); ++i) {
            float angle = static_cast<float>(i) / 6.f * math::TWO_PI + t * 0.3f;
            boxes[i]->position.set(std::cos(angle) * 3.0f, std::sin(angle) * 3.0f, -2.0f);
        }
        renderer.render(*scene, *camera);
        ui.render();
    });
}
