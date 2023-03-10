#include "threepp/extras/core/Font.hpp"
#include "threepp/geometries/EdgesGeometry.hpp"
#include "threepp/loaders/FontLoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 100);
    camera->position.set(0, 0, 40);

    OrbitControls controls{camera, canvas};

    FontLoader loader;
    auto data = loader.load("data/fonts/optimer_bold.typeface.json");
    Font font(*data);
    auto shapes = font.generateShapes("threepp!", 10);

    auto geometry = ShapeGeometry::create(shapes);
    geometry->center();
    auto material = MeshBasicMaterial::create();
    material->color = 0x00ff00;
    material->side = DoubleSide;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto lineMaterial = LineBasicMaterial::create();
    lineMaterial->color = Color::black;
    auto edges = LineSegments::create(EdgesGeometry::create(*geometry, 0.1f), lineMaterial);
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
