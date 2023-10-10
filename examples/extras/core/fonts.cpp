#include "threepp/geometries/EdgesGeometry.hpp"
#include "threepp/geometries/TextGeometry.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Fonts", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFCSoft;

    auto scene = Scene::create();
    scene->background = Color::black;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 10000);
    camera->position.set(0, 5, 40);

    auto light = DirectionalLight::create();
    light->position.set(10, 5, 10);
    light->lookAt(Vector3::ZEROS());
    light->castShadow = true;
    auto shadowCamera = light->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = shadowCamera->bottom = -20;
    shadowCamera->right = shadowCamera->top = 20;
    scene->add(light);

    auto pointLight = PointLight::create();
    pointLight->intensity = 0.2f;
    pointLight->position.set(0, 2, 10);
    scene->add(pointLight);

    OrbitControls controls{*camera, canvas};

    FontLoader loader;
    auto font = loader.load("data/fonts/optimer_bold.typeface.json");

    if (font) {
        TextGeometry::Options opts(*font, 10);
        opts.height = 1;
        auto geometry = TextGeometry::create("threepp!", opts);
        geometry->center();

        auto material = MeshPhongMaterial::create();
        material->color = Color::orange;

        auto mesh = Mesh::create(geometry, material);
        mesh->castShadow = true;
        scene->add(mesh);

        auto lineMaterial = LineBasicMaterial::create({{"color", Color::black}});
        lineMaterial->color = Color::black;
        lineMaterial->linewidth = 20000;
        auto edges = LineSegments::create(EdgesGeometry::create(*geometry, 10), lineMaterial);
        mesh->add(edges);
    }

    auto planeMaterial = MeshPhongMaterial::create();
    planeMaterial->color = Color::gray;
    auto plane = Mesh::create(PlaneGeometry::create(1000, 1000), planeMaterial);
    plane->position.y = -8;
    plane->rotateX(math::degToRad(-90));
    plane->receiveShadow = true;
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(*scene, *camera);
    });
}
