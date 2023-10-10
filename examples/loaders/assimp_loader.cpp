
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

namespace {

    auto loadGlb(AssimpLoader& loader) {

        auto model = loader.load("data/models/gltf/zedm.glb");
        model->scale *= 50;
        return model;
    }

    auto loadObj(AssimpLoader& loader) {

        auto model = loader.load("data/models/obj/female02/female02.obj");
        return model;
    }

    auto loadStl(AssimpLoader& loader) {

        auto model = loader.load("data/models/stl/pr2_head_pan.stl");
        model->scale *= 100;
        return model;
    }

    auto addLights(Scene& scene) {
        auto light1 = PointLight::create(0xffffff, 0.5f);
        light1->position.set(45, 115, 25);
        scene.add(light1);

        auto light2 = PointLight::create(0xffffff, 0.5f);
        light2->position.set(-45, 115, 125);
        scene.add(light2);

        auto light3 = PointLight::create(0xffffff, 0.5f);
        light3->position.set(0, 25, -30);
        scene.add(light3);
    }

}// namespace

int main() {

    Canvas canvas{"Assimp loader", {{"aa", 4}}};
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.set(0, 100, 175);

    float sep = 40;
    AssimpLoader loader;
    auto glb = loadGlb(loader);
    auto obj = loadObj(loader);
    auto stl = loadStl(loader);

    Box3 bb;
    bb.setFromObject(*obj);

    obj->position.x = -sep;

    bb.getCenter(glb->position);
    glb->position.x = sep;

    bb.getCenter(stl->position);
    stl->position.x = 0;

    scene->add(glb);
    scene->add(obj);
    scene->add(stl);

    addLights(*scene);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        for (auto& child : scene->children) {

            child->rotation.y += 1 * dt;
        }

        renderer.render(*scene, *camera);
    });
}
