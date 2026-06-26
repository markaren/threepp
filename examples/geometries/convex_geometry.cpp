#include "threepp/geometries/ConvexGeometry.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("ConvexGeometry", {{"aa", 4}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::blue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);
    camera->position.set(0, 5, 30);

    OrbitControls controls{*camera, canvas};

    constexpr int numPoints = 100;
    std::vector<Vector3> pts;
    std::shared_ptr<ConvexGeometry> convexGeometry;

    // Random point sets are occasionally degenerate or trigger precision failures
    // in quickhull. Retry with a fresh sample until it succeeds.
    for (int attempt = 0; attempt < 10; ++attempt) {
        pts.clear();
        pts.reserve(numPoints);
        for (auto i = 0; i < numPoints; i++) {
            pts.emplace_back(math::randFloatSpread(20), math::randFloatSpread(20), math::randFloatSpread(20));
        }
        try {
            convexGeometry = ConvexGeometry::create(pts);
            break;
        } catch (const std::runtime_error&) {
            if (attempt == 9) throw;
        }
    }

    auto convexMaterial = MeshBasicMaterial::create();
    convexMaterial->color = Color::gray;
    convexMaterial->transparent = true;
    convexMaterial->opacity = 0.8f;
    auto convex = Mesh::create(convexGeometry, convexMaterial);
    scene->add(convex);

    // Collect unique hull vertex positions from the geometry's position buffer.
    // The quickhull algo copies exact float values from pts, so exact equality works.
    auto* posAttr = convexGeometry->getAttribute<float>("position");
    std::vector<Vector3> hullVerts;
    for (size_t i = 0; i < posAttr->count(); i++) {
        Vector3 v(posAttr->getX(i), posAttr->getY(i), posAttr->getZ(i));
        bool dup = false;
        for (const auto& hv : hullVerts) {
            if (hv.x == v.x && hv.y == v.y && hv.z == v.z) { dup = true; break; }
        }
        if (!dup) hullVerts.push_back(v);
    }

    constexpr float sphereRadius = 0.3f;
    auto spheres = InstancedMesh::create(SphereGeometry::create(sphereRadius, 8, 6),
                                         MeshBasicMaterial::create(),
                                         numPoints);
    Matrix4 matrix;
    for (auto i = 0; i < numPoints; i++) {
        const auto& p = pts[i];
        matrix.setPosition(p);
        spheres->setMatrixAt(i, matrix);
        spheres->setColorAt(i, convexGeometry->containsPoint(p, -sphereRadius) ? Color::black : Color::red);
    }
    convex->add(spheres);

    auto lineMaterial = LineBasicMaterial::create(LineBasicMaterial::Params{}.color(Color::black));
    auto edges = LineSegments::create(WireframeGeometry::create(*convexGeometry), lineMaterial);
    convex->add(edges);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        convex->rotation.y += 0.2f * dt;

        renderer->render(*scene, *camera);
    });
}
