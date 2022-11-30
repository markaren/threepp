
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 1;

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);
    renderer.setSize(canvas.getSize());

    OrbitControls controls{camera, canvas};

    STLLoader loader;
    auto geometry = loader.load("data/models/stl/pr2_head_pan.stl");
    auto material = MeshBasicMaterial::create();
    material->color = Color::brown;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.animate([&](float dt) {

        renderer.render(scene, camera);
    });
}
