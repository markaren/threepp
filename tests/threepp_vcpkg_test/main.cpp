
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create();
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto scene = Scene::create();

    auto geometry = BoxGeometry::create();
    auto material = MeshBasicMaterial::create();
    material->color = Color::green;
    auto mesh = Mesh::create(geometry, material);

    scene->add(mesh);

    canvas.animate([&]{

        renderer.render(scene, camera);
    });

}