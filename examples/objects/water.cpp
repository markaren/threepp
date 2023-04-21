
#include "threepp/objects/Water.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(55, canvas.getAspect(), 1, 2000);
    camera->position.set(-300, 120, -150);

    OrbitControls controls{camera, canvas};
    controls.maxPolarAngle = math::PI * 0.495f;
    controls.target.set(0, 10, 0);
    controls.minDistance = 40;
    controls.maxDistance = 400;
    controls.update();

    auto light = DirectionalLight::create(0xffffff);
    light->position.set(100, 10, 100);
    scene->add(light);

    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.toneMapping = ACESFilmicToneMapping;

    const auto sphereGeometry = SphereGeometry::create(30);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x0000ff);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    scene->add(sphere);

    TextureLoader textureLoader{};
    auto texture = textureLoader.load("data/textures/waternormals.jpg");
    texture->wrapS = RepeatWrapping;
    texture->wrapT = RepeatWrapping;

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
    auto shaderUniforms = sky->material()->as<ShaderMaterial>()->uniforms;
    shaderUniforms->at("turbidity").value<float>() = 10;
    shaderUniforms->at("rayleigh").value<float>() = 1;
    shaderUniforms->at("mieCoefficient").value<float>() = 0.005;
    shaderUniforms->at("mieDirectionalG").value<float>() = 0.8;
    shaderUniforms->at("sunPosition").value<Vector3>().copy(light->position);
    scene->add(sky);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
        sphere->position.y = std::sin(t) * 20 + 5;
        sphere->rotation.x = t * 0.05f;
        sphere->rotation.z = t * 0.051f;

        water->material()->as<ShaderMaterial>()->uniforms->at("time").setValue(t);

        renderer.render(scene, camera);
    });
}
