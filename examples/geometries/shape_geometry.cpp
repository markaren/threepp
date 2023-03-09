#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 5, 30);

    OrbitControls controls{camera, canvas};

    float x = 0, y = 0;

    auto heartShape = Shape();

    heartShape.moveTo(x + 5, y + 5);
    heartShape.bezierCurveTo(x + 5, y + 5, x + 4, y, x, y);
    heartShape.bezierCurveTo(x - 6, y, x - 6, y + 7, x - 6, y + 7);
    heartShape.bezierCurveTo(x - 6, y + 11, x - 3, y + 15.4f, x + 5, y + 19);
    heartShape.bezierCurveTo(x + 12, y + 15.4f, x + 16, y + 11, x + 16, y + 7);
    heartShape.bezierCurveTo(x + 16, y + 7, x + 16, y, x + 10, y);
    heartShape.bezierCurveTo(x + 7, y, x + 5, y + 5, x + 5, y + 5);

    auto geometry = ShapeGeometry::create(heartShape);
    geometry->center();
    auto material = MeshBasicMaterial::create();
    material->color = 0x00ff00;
    material->side = DoubleSide;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    mesh->add(LineSegments::create(WireframeGeometry::create(*geometry)));

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        mesh->rotation.y += 0.8f * dt;
        mesh->rotation.x += 0.5f * dt;

        renderer.render(scene, camera);
    });
}