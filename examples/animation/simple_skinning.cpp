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
    renderer.shadowMap().type = threepp::ShadowMap::PFCSoft;

    PerspectiveCamera camera(45, canvas.aspect(), 0.1, 10000);
    camera.position.set(0, 6, -10);

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
    scene.add(dirLight);

    //

    AssimpLoader loader;

    auto soldier = loader.load("data/models/gltf/Soldier.glb");
    soldier->traverseType<Mesh>([](Mesh& m) {
        m.receiveShadow = true;
        m.castShadow = true;
    });
    scene.add(soldier);
    soldier->scale *= 2;
    soldier->position.x = -2;

    auto skeletonHelperSoldier = SkeletonHelper::create(*soldier);
    skeletonHelperSoldier->material()->as<LineBasicMaterial>()->linewidth = 2;
    scene.add(skeletonHelperSoldier);

    //

    auto stormTrooper = loader.load("data/models/collada/stormtrooper/stormtrooper.dae");
    stormTrooper->traverseType<Mesh>([](Mesh& m) {
        m.receiveShadow = true;
        m.castShadow = true;

        if (auto mat = m.material()->as<MaterialWithMap>()) {
            mat->map->wrapS = TextureWrapping::Repeat;
            mat->map->wrapT = TextureWrapping::Repeat;
        }
    });
    scene.add(stormTrooper);
    stormTrooper->scale *= 0.6;
    stormTrooper->position.x = 2;

    auto skeletonHelperTrooper = SkeletonHelper::create(*stormTrooper);
    skeletonHelperTrooper->material()->as<LineBasicMaterial>()->linewidth = 2;
    scene.add(skeletonHelperTrooper);

    //

    OrbitControls controls{camera, canvas};
    controls.minDistance = 2;
    controls.maxDistance = 20;

    //

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto& soldierBones = skeletonHelperSoldier->getBones();
    auto& trooperBones = skeletonHelperTrooper->getBones();

    Clock clock;
    canvas.animate([&] {
        renderer.render(scene, camera);

        auto time = clock.getElapsedTime();
        for (auto i = 0; i < soldierBones.size(); i++) {
            soldierBones[i]->rotation.y = std::sin(time) * 5 / float(soldierBones.size());
        }
        for (auto i = 0; i < trooperBones.size(); i++) {
            trooperBones[i]->rotation.y = std::sin(time) * 5 / float(trooperBones.size());
        }
    });
}
