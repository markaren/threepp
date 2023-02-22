
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 4;

    OrbitControls controls{camera, canvas};

    auto group = Group::create();

    const auto material = MeshBasicMaterial::create();
    material->color.setHex(0xff0000);
    material->wireframe = true;

    {
        const auto geometry = CylinderGeometry::create(0.5f, 0.5f);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = -1;
        group->add(mesh);
    }

    {
        const auto geometry = ConeGeometry::create(0.5f);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = 1;
        group->add(mesh);
    }

    scene->add(group);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {

        group->rotation.y += 1.f * dt;

        renderer.render(scene, camera);
    });
}
