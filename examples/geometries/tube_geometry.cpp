
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    class CustomSineCurve : public Curve3 {

    public:
        float scale;

        explicit CustomSineCurve(float scale) : scale(scale) {}

        void getPoint(float t, Vector3 &target) override {
            float tx = t * 3 - 1.5f;
            float ty = std::sin(math::PI * 2 * t);
            float tz = 0;

            target.set(tx, ty, tz).multiplyScalar(scale);
        }
    };

}// namespace

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 20;

    OrbitControls controls{camera, canvas};

    auto curve = std::make_shared<CustomSineCurve>(10.f);

    const auto geometry = TubeGeometry::create(curve);
    const auto material = MeshBasicMaterial::create();
    material->color.setHex(0xff0000);
    material->wireframe = true;
    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&]() {
        renderer.render(scene, camera);
    });
}
