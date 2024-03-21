#include "threepp/math/Lut.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>

using namespace threepp;

namespace {

    void rosenbrock(PlaneGeometry& geometry, float a = 1, float b = 100) {

        auto pos = geometry.getAttribute<float>("position");
        auto& vertices = pos->array();

        std::vector<float> yValues;
        yValues.reserve(vertices.size() / 3);
        for (auto i = 0; i < vertices.size(); i += 3) {

            float x = vertices[i];
            float z = vertices[i + 2];
            float y = ((a - x) * (a - x)) + b * ((z - (x * x)) * (z - (x * x)));

            vertices[i + 1] = y;
            yValues.emplace_back(y);
        }

        Lut::addColorMap("rainbow", {{0.f, 0x0000ff}, {0.001f, 0x00ffff}, {0.01f, 0xffff00}, {0.1f, 0xff0000}, {1.f, Color::darkred}});
        Lut lut("rainbow", 256 * 256);
        auto colors = std::vector<float>(vertices.size());
        auto minMax = std::minmax_element(yValues.begin(), yValues.end());
        for (auto i = 0; i < vertices.size(); i += 3) {

            float y = vertices[i + 1];

            Color c = lut.getColor(math::mapLinear(y, *minMax.first, *minMax.second, 0, 1));
            c.toArray(colors, i);
        }
        geometry.setAttribute("color", FloatBufferAttribute::create(colors, 3));
        geometry.scale(1, 0.001, 1);

        pos->needsUpdate();
    }

}// namespace

int main() {

    Canvas canvas("Lut", {{"aa", 6}});
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.set(5, 4, -5);

    OrbitControls controls(camera, canvas);

    auto planeGeometry = PlaneGeometry::create(4, 4, 100, 100);
    auto planeGeometry2 = PlaneGeometry::create(4, 4, 40, 40);
    planeGeometry->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2).setPosition(0, 0, 1));
    planeGeometry2->applyMatrix4(Matrix4().makeRotationX(-math::PI / 2).setPosition(0, 0, 1));
    rosenbrock(*planeGeometry);
    rosenbrock(*planeGeometry2);
    auto plane = Mesh::create(planeGeometry, MeshBasicMaterial::create({{"vertexColors", true}}));
    auto wireframe = Mesh::create(planeGeometry2, MeshBasicMaterial::create({{"wireframe", true}}));
    wireframe->material()->depthTest = false;
    wireframe->material()->opacity = 0.25;
    wireframe->material()->transparent = true;

    plane->add(wireframe);
    scene.add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();

        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(scene, camera);
    });
}
