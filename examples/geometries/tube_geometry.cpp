
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    struct CustomSineCurve: Curve3 {

        explicit CustomSineCurve(float scale): scale(scale) {}

        void getPoint(float t, Vector3& target) const override {
            float tx = t * 3 - 1.5f;
            float ty = std::sin(math::PI * 2 * t);
            float tz = 0;

            target.set(tx, ty, tz).multiplyScalar(scale);
        }

    private:
        float scale;
    };

}// namespace

int main() {

    Canvas canvas("TubeGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 20;

    OrbitControls controls{*camera, canvas};

    auto curve = std::make_shared<CustomSineCurve>(10.f);

    const auto geometry = TubeGeometry::create(curve);
    const auto material = MeshBasicMaterial::create({{"color", 0xff0000},
                                                     {"side", Side::Double}});
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    auto lineMaterial = LineBasicMaterial::create({{"depthTest", false},
                                                   {"opacity", 0.5f},
                                                   {"transparent", true}});
    auto line = LineSegments::create(WireframeGeometry::create(*geometry), lineMaterial);
    mesh->add(line);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });


    Clock clock;
    const auto count = geometry->getIndex()->count();
    canvas.animate([&]() {
        mesh->rotation.y += 1 * clock.getDelta();

        auto map = math::mapLinear(std::sin(clock.elapsedTime), -1, 1, 0, static_cast<float>(count));
        geometry->setDrawRange(0, static_cast<int>(map));

        renderer.render(*scene, *camera);
    });
}
