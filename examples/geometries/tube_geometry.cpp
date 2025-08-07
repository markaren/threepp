
#include "threepp/threepp.hpp"

#include "bvh/BVH.hpp"

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

    auto makeBoxes(const std::vector<BVHBox3>& boxes) {

        auto boxesGroup = Group::create();
        boxesGroup->visible = false;// Start with boxes hidden
        for (const auto& box : boxes) {
            auto helper = Box3Helper::create(box, box.isLeaf() ? Color::green : Color::red);
            box.getCenter(helper->position);
            boxesGroup->add(helper);
        }

        return boxesGroup;
    }

}// namespace

int main() {

    Canvas canvas("TubeGeometry", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 20;

    OrbitControls controls{*camera, canvas};

    auto tube1 = makeTubeMesh();
    const auto geometry1 = tube1->geometry();
    scene->add(tube1);

    BVH bvh1;
    bvh1.build(*geometry1);

    std::vector<BVHBox3> boxes;
    bvh1.collectBoxes(boxes);

    auto boxesGroup = makeBoxes(boxes);
    tube1->add(boxesGroup);

    auto tube2 = makeTubeMesh();
    tube2->position.x = 10;
    scene->add(tube2);

    BVH bvh2;
    bvh2.build(*tube2->geometry());

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](const KeyEvent& evt) {
        static bool show = boxesGroup->visible;
        if (evt.key == Key::B) {
            show = !show;
            boxesGroup->visible = show;
        }
    });
    std::cout << "Press 'B' to toggle BVH visibility." << std::endl;

    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto collisions = InstancedMesh::create(
            SphereGeometry::create(0.5f, 16, 8),
            MeshBasicMaterial::create({{"color", Color::yellow}}), 10000);
    collisions->frustumCulled = false;// Disable frustum culling for collisions
    scene->add(collisions);


    Clock clock;
    const auto count = static_cast<float>(geometry1->getIndex()->count());
    canvas.animate([&]() {
        tube1->rotation.y += 1 * clock.getDelta();

        const auto map = math::mapLinear(std::sin(clock.elapsedTime), -1, 1, 0, count);
        geometry1->setDrawRange(0, static_cast<int>(map));

        if (BVH::intersects(bvh1, bvh2, *tube1->matrixWorld, *tube2->matrixWorld)) {
            static Matrix4 m;
            const auto res = BVH::intersect(bvh1, *tube1->matrixWorld, bvh2, *tube2->matrixWorld);
            collisions->setCount(res.size());
            for (int i = 0; i < res.size(); ++i) {
                const auto& intersection = res[i];
                collisions->setMatrixAt(i, m.makeTranslation(intersection.position));
            }
            collisions->instanceMatrix()->needsUpdate();
        } else {
            collisions->setCount(0);
        }

        renderer.render(*scene, *camera);
    });
}
