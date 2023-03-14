
#include "threepp/threepp.hpp"
#include "threepp/objects/Sky.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);
    renderer.shadowMap().enabled = true;
    renderer.toneMapping = ACESFilmicToneMapping;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(-5, 2, -5);

    auto light = DirectionalLight::create();
    light->position.set(150, 50, 150);
    light->castShadow = true;
    scene->add(light);

    auto sky = Sky::create();
    sky->scale.setScalar(1000);
    sky->material()->as<ShaderMaterial>()->uniforms->at("turbidity").value<float>() = 10;
    sky->material()->as<ShaderMaterial>()->uniforms->at("rayleigh").value<float>() = 1;
    sky->material()->as<ShaderMaterial>()->uniforms->at("mieCoefficient").value<float>() = 0.005f;
    sky->material()->as<ShaderMaterial>()->uniforms->at("mieDirectionalG").value<float>() = 0.8f;
    sky->material()->as<ShaderMaterial>()->uniforms->at("sunPosition").value<Vector3>().copy(light->position);
    scene->add(sky);

    OrbitControls controls{camera, canvas};

    auto helper = DirectionalLightHelper::create(light);
    scene->add(helper);

    const auto geometry = TorusKnotGeometry::create(0.75f, 0.2f, 128, 64);
    const auto material = MeshStandardMaterial::create();
    material->roughness = 0.1;
    material->metalness = 0.1;
    material->color = 0xff0000;
    material->emissive = 0x000000;
    auto mesh = Mesh::create(geometry, material);
    mesh->castShadow = true;
    mesh->position.y = 2;
    scene->add(mesh);

    const auto planeGeometry = PlaneGeometry::create(100, 100);
    const auto planeMaterial = MeshLambertMaterial::create();
    planeMaterial->color = Color::gray;
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->rotateX(math::degToRad(90));
    plane->receiveShadow = true;
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
        mesh->rotation.y -= 0.5f * dt;

        light->position.x = 100 * std::sin(t);
        light->position.z = 100 * std::cos(t);

        sky->material()->as<ShaderMaterial>()->uniforms->at("sunPosition").value<Vector3>().copy(light->position);

        light->updateMatrixWorld();
        helper->update();

        renderer.render(scene, camera);
    });

}
