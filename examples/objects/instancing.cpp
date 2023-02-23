
#include "threepp/objects/LOD.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    int amount = 10;
    int count = std::pow(amount, 3);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(amount, amount, amount);

    OrbitControls controls{camera, canvas};

    auto light = DirectionalLight::create(0xffffff);
    light->position.set(1,1,1);
    scene->add(light);

    auto material = MeshPhongMaterial::create();
    auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
    auto mesh = InstancedMesh::create(geometry, material, count);

    Matrix4 matrix;
    Color color{Color::whitesmoke};
    size_t index = 0;
    float offset = static_cast<float>(amount - 1) / 2;
    for (int x = 0; x < amount; x++) {
        for (int y = 0; y < amount; y++) {
            for (int z = 0; z < amount; z++) {
                matrix.setPosition(offset - x, offset - y, offset - z);
                mesh->setMatrixAt(index, matrix);
                mesh->setColorAt(index, color);
                ++index;
            }
        }
    }

    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Vector2 mouse{-10, -10};
    MouseMoveListener l([&](auto pos){
        auto size = canvas.getSize();
        mouse.x = (pos.x / static_cast<float>(size.width)) * 2 - 1;
        mouse.y = -(pos.y / static_cast<float>(size.height)) * 2 + 1;
    });
    canvas.addMouseListener(&l);

    Raycaster racaster;
    std::unordered_map<int, bool> map;
    canvas.animate([&](float dt) {

        racaster.setFromCamera(mouse, camera);
        auto intersects = racaster.intersectObject(mesh.get());

        if (!intersects.empty()) {
            auto instanceId = intersects.front().instanceId;
            if (instanceId && !map[*instanceId]) {
                mesh->setColorAt(*instanceId, color.randomize());
                mesh->instanceColor->needsUpdate();
                map[*instanceId] = true;
            }
        }

        renderer.render(scene, camera);
    });
}
