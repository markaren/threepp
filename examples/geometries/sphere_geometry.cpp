
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 3;

    OrbitControls controls{camera, canvas};

    const auto geometry = SphereGeometry::create(0.5f, 16 ,16);
    const auto material = MeshBasicMaterial::create();
    material->color.setHex(0xff0000);
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto line = LineSegments::create(WireframeGeometry::create(*geometry));
    line->material()->as<LineBasicMaterial>()->depthTest = false;
    line->material()->as<LineBasicMaterial>()->opacity = 0.5;
    line->material()->as<LineBasicMaterial>()->transparent = true;
    mesh->add(line);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(scene, camera);
    });
}
