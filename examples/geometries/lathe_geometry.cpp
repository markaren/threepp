#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    std::vector<Vector2> generateLathePoints() {
        std::vector<Vector2> points;
        for (unsigned i = 0; i < 10; i++) {
            points.emplace_back(std::sin(static_cast<float>(i) * 0.2f) * 10 + 5, (static_cast<float>(i) - 5.f) * 2.f);
        }
        return points;
    }

    std::shared_ptr<Mesh> createLathe() {
        auto geometry = LatheGeometry::create(generateLathePoints());
        auto material = MeshNormalMaterial::create({{"side", DoubleSide}});
        auto mesh = Mesh::create(geometry, material);

        auto line = LineSegments::create(WireframeGeometry::create(*geometry));
        mesh->add(line);

        return mesh;
    }

    std::shared_ptr<Mesh> createCapsule() {

        auto geometry = CapsuleGeometry::create();
        auto material = MeshNormalMaterial::create();
        auto mesh = Mesh::create(geometry, material);

        auto line = LineSegments::create(WireframeGeometry::create(*geometry));
        mesh->add(line);

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("LatheGeometry", {{"antialiasing", 4}});
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::gray;
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto lathe = createLathe();
    lathe->scale *= 0.1;
    lathe->position.x = -2;
    auto capsule = createCapsule();
    capsule->position.x = 2;

    scene->add(lathe);
    scene->add(capsule);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        lathe->rotation.y += 0.8f * dt;
        lathe->rotation.x += 0.5f * dt;

        capsule->rotation.y += 0.8f * dt;
        capsule->rotation.x += 0.5f * dt;

        renderer.render(scene, camera);
    });
}
