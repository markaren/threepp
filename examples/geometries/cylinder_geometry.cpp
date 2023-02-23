
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

    {
        const auto geometry = CylinderGeometry::create(0.5f, 0.5f);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = -1;
        group->add(mesh);

        auto line = LineSegments::create(WireframeGeometry::create(*geometry));
        line->material()->as<LineBasicMaterial>()->depthTest = false;
        line->material()->as<LineBasicMaterial>()->opacity = 0.5;
        line->material()->as<LineBasicMaterial>()->transparent = true;
        mesh->add(line);
    }

    {
        const auto geometry = ConeGeometry::create(0.5f);
        auto mesh = Mesh::create(geometry, material);
        mesh->position.x = 1;
        group->add(mesh);

        auto line = LineSegments::create(WireframeGeometry::create(*geometry));
        line->material()->as<LineBasicMaterial>()->depthTest = false;
        line->material()->as<LineBasicMaterial>()->opacity = 0.5;
        line->material()->as<LineBasicMaterial>()->transparent = true;
        mesh->add(line);
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
