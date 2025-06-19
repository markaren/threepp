
#include "threepp/threepp.hpp"

#include "threepp/objects/BVH.hpp"

#include <cmath>
#include <iostream>

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
        auto curve = std::make_unique<CustomSineCurve>(10.f);

        const auto geometry = TubeGeometry::create(std::move(curve));
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

    auto tube = makeTubeMesh();
    const auto geometry = tube->geometry();
    scene->add(tube);

    BVH bvh;
    bvh.build(*tube->geometry());

    std::vector<Box3> boxes;
    bvh.collectBoxes(boxes);

    Group boxesGroup;
    boxesGroup.visible = false; // Start with boxes hidden
    for (const auto& box : boxes) {
        auto helper = Box3Helper::create(box);
        box.getCenter(helper->position);
        boxesGroup.add(helper);
    }
    tube->add(boxesGroup);

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](const KeyEvent& evt) {
        static bool show = boxesGroup.visible;
        if (evt.key == Key::B) {
            show = !show;
            boxesGroup.visible = show;
        }
    });
    std::cout << "Press 'B' to toggle BVH visibility." << std::endl;

    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    const auto count = static_cast<float>(geometry->getIndex()->count());
    canvas.animate([&]() {
        tube->rotation.y += 1 * clock.getDelta();

        const auto map = math::mapLinear(std::sin(clock.elapsedTime), -1, 1, 0, count);
        geometry->setDrawRange(0, static_cast<int>(map));

        renderer.render(*scene, *camera);
    });
}
