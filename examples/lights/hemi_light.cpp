
#include "threepp/helpers/HemisphereLightHelper.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto createBox(const Vector3& pos, const Color& color) {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create({{"color", color}});
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.copy(pos);

        return box;
    }

    auto createPlane() {
        const auto planeGeometry = PlaneGeometry::create(5, 5);
        const auto planeMaterial = MeshPhongMaterial::create({{"color", Color::gray}, {"side", DoubleSide}});
        auto plane = Mesh::create(planeGeometry, planeMaterial);
        plane->position.y = -1;
        plane->rotateX(math::degToRad(90));

        return plane;
    }

}// namespace

int main() {

    Canvas canvas("HemisphereLight", {{"antialiasing", 4}});
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 2, 5);

    OrbitControls controls{camera, canvas};

    auto light = HemisphereLight::create(0xffffbb, 0x082820);
    light->position.y = 2;
    scene->add(light);

    auto helper = HemisphereLightHelper::create(*light, 0.5f);
    scene->add(helper);

    auto group = Group::create();
    group->add(createBox({-1, 0, 0}, Color::red));
    group->add(createBox({1, 0, 0}, Color::green));
    scene->add(group);

    auto plane = createPlane();
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        group->rotation.y += 0.5f * dt;

        renderer.render(scene, camera);
    });
}
