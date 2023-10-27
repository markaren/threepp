#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Simple skinning", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    PerspectiveCamera camera(45, canvas.aspect(), 0.1, 10000);
    camera.position.set(18, 6, -18);

    Scene scene;
    scene.background = Color(0xa0a0a0);
    scene.fog = Fog(0xa0a0a0, 70, 100);

    // ground

    auto geometry = PlaneGeometry::create(500, 500);
    auto material = MeshPhongMaterial::create({{"color", 0x999999},
                                               {"depthWrite", false}});

    auto ground = Mesh::create(geometry, material);
    ground->rotation.x = -math::PI / 2;
    ground->receiveShadow = true;
    scene.add(ground);

    auto grid = GridHelper::create(500, 100, 0x000000, 0x000000);
    grid->material()->opacity = 0.2;
    grid->material()->transparent = true;
    scene.add(grid);

    // lights

    auto hemiLight = HemisphereLight::create(0xffffff, 0x444444, 0.6f);
    hemiLight->position.set(0, 200, 0);
    scene.add(hemiLight);

    auto dirLight = DirectionalLight::create(0xffffff, 0.8f);
    dirLight->position.set(0, 20, 10);
    dirLight->castShadow = true;
    dirLight->shadow->camera->as<OrthographicCamera>()->top = 18;
    dirLight->shadow->camera->as<OrthographicCamera>()->bottom = -10;
    dirLight->shadow->camera->as<OrthographicCamera>()->left = -12;
    dirLight->shadow->camera->as<OrthographicCamera>()->right = 12;
    scene.add(dirLight);

    //

    AssimpLoader loader;
    //    auto gltf = loader.load("data/models/gltf/SimpleSkinning.gltf");
    auto gltf = loader.load("data/models/gltf/Soldier.glb");
    gltf->traverseType<Mesh>([](Mesh& m) {
        m.receiveShadow = true;
        m.castShadow = true;
        m.material()->opacity = 0.6;
        m.material()->transparent = true;
    });
//    gltf->position.y = -10;
    gltf->scale *= 5;
    scene.add(gltf);

    auto skeletonHelper = SkeletonHelper::create(*gltf);
    skeletonHelper->material()->as<LineBasicMaterial>()->linewidth = 2;
    scene.add(skeletonHelper);

    //

    OrbitControls controls{camera, canvas};
    //    controls.enablePan = false;
    //    controls.minDistance = 5;
    //    controls.maxDistance = 50;

    //

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto& bones = skeletonHelper->getBones();

    Clock clock;
    canvas.animate([&] {
        renderer.render(scene, camera);

        auto time = clock.getElapsedTime();
        for (auto i = 0; i < bones.size(); i++) {
            bones[i]->rotation.x = std::sin(time) * 5 / float(bones.size());
        }

    });
}