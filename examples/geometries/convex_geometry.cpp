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

    std::vector<float> createColorBuffer(size_t numPoints, const Color& c) {
        std::vector<float> res;
        for (unsigned i = 0; i < numPoints; i++) {
            res.emplace_back(c.r);
            res.emplace_back(c.g);
            res.emplace_back(c.b);
        }
        return res;
    }

}// namespace

int main() {

    Canvas canvas("ConvexGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);
    camera->position.set(0, 5, 30);

    OrbitControls controls{*camera, canvas};

    int numPoints = 100;
    std::vector<Vector3> pts;
    pts.reserve(numPoints);
    for (unsigned i = 0; i < numPoints; i++) {
        pts.emplace_back(math::randFloatSpread(20), math::randFloatSpread(20), math::randFloatSpread(20));
    }

    auto pointsGeometry = BufferGeometry::create();
    pointsGeometry->setAttribute("color", FloatBufferAttribute::create(createColorBuffer(numPoints, Color::red), 3));
    pointsGeometry->setAttribute("position", FloatBufferAttribute::create(flatten(pts), 3));
    pointsGeometry->computeBoundingBox();

    auto convexGeometry = ConvexGeometry::create(pts);
    auto convexMaterial = MeshBasicMaterial::create();
    convexMaterial->color = Color::gray;
    convexMaterial->transparent = true;
    convexMaterial->opacity = 0.8f;
    auto convex = Mesh::create(convexGeometry, convexMaterial);
    scene->add(convex);

    for (unsigned i = 0; i < numPoints; i++) {
        const auto& p = pts[i];
        if (convexGeometry->containsPoint(p, -0.1)) {
            pointsGeometry->getAttribute<float>("color")->setXYZ(i, 0, 0, 0);
        }
    }

    auto points = Points::create(pointsGeometry, PointsMaterial::create({{"vertexColors", true}}));
    convex->add(points);

    auto lineMaterial = LineBasicMaterial::create({{"color", Color::black}});
    auto edges = LineSegments::create(WireframeGeometry::create(*convexGeometry), lineMaterial);
    convex->add(edges);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        convex->rotation.y += 0.2f * dt;

        renderer.render(*scene, *camera);
    });
}
