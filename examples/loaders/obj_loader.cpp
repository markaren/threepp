
#include <threepp/threepp.hpp>

using namespace threepp;

void createAndAddLights(Scene& scene) {
    auto light1 = PointLight::create(0xffffff, 1.f);
    light1->position.set(45, 115, 25);
    scene.add(light1);

    auto light2 = PointLight::create(0xffffff, 1.f);
    light2->position.set(-45, 115, 125);
    scene.add(light2);

    auto light3 = PointLight::create(0xffffff, 1.f);
    light3->position.set(0, 25, -30);
    scene.add(light3);
}

int main() {

    Canvas canvas{"OBJ loader", {{"aa", 8}}};
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.set(0, 100, 150);

    OBJLoader loader;
    auto obj1 = loader.load("data/models/obj/female02/female02.obj");
    obj1->position.x = -30;
    scene->add(obj1);

    TextureLoader tl;
    auto tex = tl.load("data/textures/uv_grid_opengl.jpg");
    auto obj2 = loader.load("data/models/obj/female02/female02.obj", false);
    obj2->position.x = 30;
    obj2->traverseType<Mesh>([tex](Mesh& child) {
        auto m = MeshPhongMaterial::create();
        m->map = tex;
        child.setMaterial(m);
    });
    scene->add(obj2);

    createAndAddLights(*scene);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        obj1->rotation.y += 1 * dt;
        obj2->rotation.y += 1 * dt;

        renderer.render(*scene, *camera);
    });
}
