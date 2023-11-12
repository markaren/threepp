
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {


    auto createPlane() {

        auto geometry = PlaneGeometry::create(200, 200, 50, 50);
        geometry->applyMatrix4(Matrix4().makeRotationX(math::degToRad(90)));
        auto material = MeshBasicMaterial::create();
        material->side = Side::Double;
        material->color = Color::navy;

        auto mesh = Mesh::create(geometry, material);
        auto wireframe = Mesh::create(geometry, MeshBasicMaterial::create({{"wireframe", true},
                                                                           {"color", Color::darkgray}}));
        mesh->add(wireframe);

        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("PlaneGeometry - dynamic", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(70, canvas.aspect(), 0.1f, 1000);
    camera->position.set(-100, 20, -100);

    OrbitControls controls{*camera, canvas};

    auto plane = createPlane();
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    auto position = plane->geometry()->getAttribute<float>("position");
    position->setUsage(threepp::DrawUsage::Dynamic);
    canvas.animate([&]() {
        float time = clock.getElapsedTime();

        renderer.render(*scene, *camera);

        for (unsigned i = 0; i < position->count(); i++) {

            float y = 2 * std::sin(static_cast<float>(i) / 5.f + (time * 20 + static_cast<float>(i)) / 7.f);
            position->setY(i, y);

            position->needsUpdate();
        }
    });
}
