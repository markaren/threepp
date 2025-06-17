
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    struct CustomSineCurve final: Curve3 {

        explicit CustomSineCurve(float scale): scale(scale) {}

        void getPoint(float t, Vector3& target) const override {
            const float tx = t * 3 - 1.5f;
            const float ty = std::sin(math::PI * 2 * t);
            const float tz = 0;

            target.set(tx, ty, tz).multiplyScalar(scale);
        }

    private:
        float scale;
    };

    auto makeTubeMesh() {
        const auto curve = std::make_shared<CustomSineCurve>(10.f);

        const auto geometry = TubeGeometry::create(curve);
        const auto material = MeshBasicMaterial::create(
                {{"color", 0xff0000},
                 {"side", Side::Double}});
        auto mesh = Mesh::create(geometry, material);

        const auto lineMaterial = LineBasicMaterial::create(
                {{"depthTest", false},
                 {"opacity", 0.5f},
                 {"transparent", true}});
        const auto line = LineSegments::create(WireframeGeometry::create(*geometry), lineMaterial);
        mesh->add(line);

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("TubeGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 20;

    OrbitControls controls{*camera, canvas};

    auto curve = makeTubeMesh();
    const auto geometry = curve->geometry();
    scene->add(curve);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    const auto count = static_cast<float>(geometry->getIndex()->count());
    canvas.animate([&]() {
        curve->rotation.y += 1 * clock.getDelta();

        const auto map = math::mapLinear(std::sin(clock.elapsedTime), -1, 1, 0, count);
        geometry->setDrawRange(0, static_cast<int>(map));

        renderer.render(*scene, *camera);
    });
}
