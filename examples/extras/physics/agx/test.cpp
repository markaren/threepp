

#include "threepp/threepp.hpp"

#include "threepp/extras/physics/AgxPhysics.hpp"

using namespace threepp;

namespace {

    inline Vector3 randomPosition() {
        return {math::randomInRange(-5.f, 5.f), math::randomInRange(5.f, 15.f), math::randomInRange(-5.f, 5.f)};
    }

    void initPositions(InstancedMesh& m) {
        Matrix4 matrix;
        for (unsigned i = 0; i < m.count; i++) {

            matrix.setPosition(randomPosition());
            m.setMatrixAt(i, matrix);
            m.setColorAt(i, Color().randomize());
        }
    }

    auto createSphere() {

        const auto geometry = SphereGeometry::create(0.5f);
        const auto material = MeshLambertMaterial::create();
        material->color.setHex(0xff0000);
        auto mesh = Mesh::create(geometry, material);
        return mesh;
    }

    auto createPlane() {

        const auto geometry = BoxGeometry::create(10, 1, 10);
        const auto material = MeshLambertMaterial::create();
        material->color = Color::gray;
        auto mesh = Mesh::create(geometry, material);
        return mesh;
    }

    auto createBoxes(unsigned int count) {
        const auto geometry = BoxGeometry::create(0.5f, 0.5f, 0.5f);
        const auto material = MeshPhongMaterial::create();
        return InstancedMesh::create(geometry, material, count);
    }

}// namespace

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 5, 5);

    OrbitControls controls{camera, canvas};

    auto light = HemisphereLight::create();
    scene->add(light);

    auto sphere = createSphere();
    sphere->position.y = 10;
    auto plane = createPlane();

    auto boxes = createBoxes(20);
    boxes->instanceMatrix->setUsage(DynamicDrawUsage);
    initPositions(*boxes);

    scene->add(sphere);
    scene->add(plane);
    scene->add(boxes);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    AgxPhysics agx;
    agx.addMesh(*sphere, 1);
    agx.addMesh(*plane, 0);
    agx.addMesh(*boxes, 10);

    agx.saveScene("test");

    canvas.animate([&](float dt) {
        agx.step(dt);

        renderer.render(scene, camera);
    });
}
