#include "threepp/geometries/ConvexGeometry.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    std::vector<float> flatten(const std::vector<Vector3>& positions) {
        std::vector<float> res;
        for (const auto& p : positions) {
            res.emplace_back(p.x);
            res.emplace_back(p.y);
            res.emplace_back(p.z);
        }
        return res;
    }

}// namespace

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 5, 30);

    OrbitControls controls{camera, canvas};

    std::vector<Vector3> pts;
    pts.reserve(10);
    for (auto i = 0; i < 10; i++) {
        pts.emplace_back(math::randomInRange(-10.f, 10.f), math::randomInRange(-10.f, 10.f), math::randomInRange(-10.f, 10.f));
    }

    auto pointsGeometry = BufferGeometry::create();
    pointsGeometry->setAttribute("position", FloatBufferAttribute::create(flatten(pts), 3));

    auto convexGeometry = ConvexGeometry::create(pts);
    auto convexMaterial = MeshBasicMaterial::create();
    convexMaterial->color = Color::gray;
    convexMaterial->transparent = true;
    convexMaterial->side = FrontSide;
    convexMaterial->opacity = 0.8f;
    scene->add(Mesh::create(convexGeometry, convexMaterial));


    auto points = Points::create(pointsGeometry);
    points->material()->as<PointsMaterial>()->color = Color::red;
    scene->add(points);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        renderer.render(scene, camera);
    });
}