
#include "threepp/objects/Water.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

#ifdef HAS_IMGUI
#include "threepp/extras/imgui/ImguiContext.hpp"
#endif

using namespace threepp;

int main() {

    Canvas canvas("Water", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;
    renderer.toneMapping = ToneMapping::ACESFilmic;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(55, canvas.aspect(), 1, 2000);
    camera->position.set(-50, 120, 500);

    OrbitControls controls{*camera, canvas};
    controls.maxPolarAngle = math::PI * 0.495f;
    controls.target.set(0, 10, 0);
    controls.minDistance = 40;
    controls.maxDistance = 400;
    controls.update();

    auto light = DirectionalLight::create(0xffffff);
    scene->add(light);

    const auto sphereGeometry = SphereGeometry::create(30);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x0000ff);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    scene->add(sphere);

    TextureLoader textureLoader{};
    auto texture = textureLoader.load("data/textures/waternormals.jpg");
    texture->wrapS = TextureWrapping::Repeat;
    texture->wrapT = TextureWrapping::Repeat;

    Water::Options opt;
    opt.textureHeight = 512;
    opt.textureWidth = 512;
    opt.alpha = 0.8f;
    opt.waterNormals = texture;
    opt.distortionScale = 3.7f;
    opt.sunDirection = light->position.clone().normalize();
    opt.sunColor = light->color;
    opt.waterColor = 0x001e0f;
    opt.fog = scene->fog.has_value();

    auto waterGeometry = PlaneGeometry::create(10000, 10000);

    auto water = Water::create(waterGeometry, opt);
    water->rotateX(math::degToRad(-90));
    scene->add(water);

    auto sky = Sky::create();
    sky->scale.setScalar(10000);
    auto& shaderUniforms = sky->material()->as<ShaderMaterial>()->uniforms;
    shaderUniforms.at("turbidity").value<float>() = 10;
    shaderUniforms.at("rayleigh").value<float>() = 1;
    shaderUniforms.at("mieCoefficient").value<float>() = 0.005;
    shaderUniforms.at("mieDirectionalG").value<float>() = 0.8;
    scene->add(sky);

    Vector3 sun;
    float elevation = 2;
    float azimuth = 180;

    auto computeSunPosition = [&] {
        float phi = math::degToRad(90 - elevation);
        float theta = math::degToRad(azimuth);

        light->position.setFromSphericalCoords(1, phi, theta);
        shaderUniforms.at("sunPosition").value<Vector3>().copy(light->position);
    };
    computeSunPosition();

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

#ifdef HAS_IMGUI

    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Controls");
        ImGui::SliderFloat("turbidity", &shaderUniforms.at("turbidity").value<float>(), 0, 20);
        ImGui::SliderFloat("rayleigh", &shaderUniforms.at("rayleigh").value<float>(), 0, 4);
        ImGui::SliderFloat("mieCoefficient", &shaderUniforms.at("mieCoefficient").value<float>(), 0, 0.1);
        ImGui::SliderFloat("mieDirectionalG", &shaderUniforms.at("mieDirectionalG").value<float>(), 0, 1);
        if (ImGui::SliderFloat("elevation", &elevation, 0, 90)) {
            computeSunPosition();
        }
        if (ImGui::SliderFloat("azimuth", &azimuth, -180, 180)) {
            computeSunPosition();
        }
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);
#endif

    Clock clock;
    auto& timeUniform = water->material()->as<ShaderMaterial>()->uniforms.at("time");
    canvas.animate([&]() {
        float t = clock.getElapsedTime();

        sphere->position.y = std::sin(t) * 20 + 5;
        sphere->rotation.x = t * 0.05f;
        sphere->rotation.z = t * 0.051f;

        timeUniform.setValue(t);

        renderer.render(*scene, *camera);

#ifdef HAS_IMGUI
        ui.render();
#endif
    });
}
