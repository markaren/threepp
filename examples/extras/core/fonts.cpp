#include "threepp/extras/core/Font.hpp"
#include "threepp/geometries/EdgesGeometry.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 0, 40);

    OrbitControls controls{camera, canvas};

    FontLoader loader;
    auto data = loader.load("data/fonts/optimer_bold.typeface.json");
    Font font(*data);
    auto shapes = font.generateShapes("threepp!", 10);

    auto material = MeshBasicMaterial::create();
    material->color = 0x00ff00;
    material->side = DoubleSide;
    auto extrude = ExtrudeGeometry::create(shapes);
    auto mesh = Mesh::create(extrude, material);
    scene->add(mesh);

    auto lineMaterial = LineBasicMaterial::create();
    lineMaterial->color = Color::black;
    auto edges = LineSegments::create(EdgesGeometry::create(*extrude, 10), lineMaterial);
    mesh->add(edges);


    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        renderer.render(scene, camera);
    });
}
