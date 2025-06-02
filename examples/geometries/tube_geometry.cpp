
#include "threepp/threepp.hpp"
#include "threepp/objects/BVH.hpp"

#include <cmath>
#include <iostream>

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

    std::shared_ptr<Object3D> createTubeObject() {
        auto curve = std::make_shared<CustomSineCurve>(10.f);

        const auto geometry = TubeGeometry::create(curve);
        const auto material = MeshBasicMaterial::create({{"color", 0xff0000},
                                                         {"side", Side::Double}});
        auto mesh = Mesh::create(geometry, material);

        auto lineMaterial = LineBasicMaterial::create({{"depthTest", false},
                                                       {"opacity", 0.5f},
                                                       {"transparent", true}});
        auto line = LineSegments::create(WireframeGeometry::create(*geometry), lineMaterial);
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

    const auto tubeObject1 = createTubeObject();
    const auto tubeObject2 = createTubeObject();

    tubeObject1->position.x = -5;
    tubeObject2->position.x = 5;

    scene->add(tubeObject1);
    scene->add(tubeObject2);

    BVH bvh1;
    bvh1.build(*tubeObject1->geometry());

    BVH bvh2;
    bvh2.build(*tubeObject2->geometry());

    auto result = bvh1.intersect(bvh2);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });


    Clock clock;
    float rotationSpeed = 1.0f;
    const auto count = tubeObject1->geometry()->getIndex()->count();
    canvas.animate([&]() {
        tubeObject1->rotation.y += rotationSpeed * clock.getDelta();
        tubeObject2->rotation.y -= rotationSpeed * clock.getDelta();

        const auto map = math::mapLinear(std::sin(clock.elapsedTime), -1, 1, 0, static_cast<float>(count));
        tubeObject1->geometry()->setDrawRange(0, static_cast<int>(map));

        if (BVH::intersects(bvh1, *tubeObject1->matrixWorld, bvh2, *tubeObject2->matrixWorld)) {
            tubeObject2->material()->as<MaterialWithColor>()->color = Color(0x00ff00);
        } else {
            tubeObject2->material()->as<MaterialWithColor>()->color = Color(0xff0000);
        }

        renderer.render(*scene, *camera);
    });
}
