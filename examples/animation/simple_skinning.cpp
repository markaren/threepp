#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Simple skinning", {{"aa", 8}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFCSoft;

    PerspectiveCamera camera(45, canvas.aspect(), 0.1, 10000);
    camera.position.set(0, 6, -15);

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

    auto model = loader.load(std::string(DATA_FOLDER) + "/models/gltf/SimpleSkinning.gltf");
    model->traverseType<SkinnedMesh>([](auto& m) {

        m.receiveShadow = true;
        m.castShadow = true;

    });
    scene.add(model);

    auto mixer = AnimationMixer(*model);
    mixer.clipAction(AnimationClip::findByName(model->animations, "Take 01"))->play();

    auto skeletonHelper = SkeletonHelper::create(*model);
    skeletonHelper->material()->as<LineBasicMaterial>()->linewidth = 2;
    scene.add(skeletonHelper);

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


    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        mixer.update(dt);

        renderer.render(scene, camera);
    });
}
