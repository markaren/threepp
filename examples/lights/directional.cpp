
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/helpers/DirectionalLightHelper.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    auto createSky(const Vector3& lightPosition) {

        auto sky = Sky::create();
        sky->scale.setScalar(1000);
        auto& skyUniforms = sky->material()->as<ShaderMaterial>()->uniforms;
        skyUniforms.at("turbidity").value<float>() = 10;
        skyUniforms.at("rayleigh").value<float>() = 1;
        skyUniforms.at("mieCoefficient").value<float>() = 0.005f;
        skyUniforms.at("mieDirectionalG").value<float>() = 0.8f;
        skyUniforms.at("sunPosition").value<Vector3>().copy(lightPosition);

        return sky;
    }

    auto createPlane() {

        const auto planeGeometry = PlaneGeometry::create(100, 100);
        const auto planeMaterial = MeshLambertMaterial::create();
        planeMaterial->color = Color::gray;
        planeMaterial->side = Side::Double;
        auto plane = Mesh::create(planeGeometry, planeMaterial);
        plane->rotateX(math::degToRad(90));
        plane->receiveShadow = true;

        return plane;
    }

    auto createTorusKnot() {

        const auto geometry = TorusKnotGeometry::create(0.75f, 0.2f, 128, 64);
        const auto material = MeshStandardMaterial::create();
        material->roughness = 0.1f;
        material->metalness = 0.1f;
        material->color = 0xff0000;
        material->emissive = 0x000000;
        auto mesh = Mesh::create(geometry, material);
        mesh->castShadow = true;
        mesh->position.y = 2;

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("DirectionalLight", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFCSoft;
    renderer.toneMapping = ToneMapping::ACESFilmic;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.set(-5, 2, -5);

    auto light = DirectionalLight::create();
    light->position.set(150, 50, 150);
    light->castShadow = true;
    scene->add(light);

    auto sky = createSky(light->position);
    auto shaderMaterial = sky->material()->as<ShaderMaterial>();
    auto& sunPositionUniform = shaderMaterial->uniforms.at("sunPosition").value<Vector3>();
    scene->add(sky);

    OrbitControls controls{*camera, canvas};

    auto helper = DirectionalLightHelper::create(*light);
    scene->add(helper);

    auto torusKnot = createTorusKnot();
    scene->add(torusKnot);

    auto plane = createPlane();
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        torusKnot->rotation.y -= 0.5f * dt;

        light->position.x = 100 * std::sin(clock.elapsedTime);
        light->position.z = 100 * std::cos(clock.elapsedTime);

        sunPositionUniform.copy(light->position);

        light->updateMatrixWorld();
        helper->update();

        renderer.render(*scene, *camera);
    });
}
