
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    auto createGeometry() {
        auto geometry = BoxGeometry::create(2, 2, 2, 32, 32, 32);
        auto positionAttribute = geometry->getAttribute<float>("position");

        std::vector<float> spherePositions;
        spherePositions.reserve(positionAttribute->count() * 3);
        std::vector<float> twistPositions(positionAttribute->count() * 3);
        Vector3 direction(1, 0, 0);
        Vector3 vertex;

        for (unsigned i = 0, j = 0; i < positionAttribute->count(); i++, j += 3) {
            auto x = positionAttribute->getX(i);
            auto y = positionAttribute->getY(i);
            auto z = positionAttribute->getZ(i);

            spherePositions.insert(spherePositions.begin(),
                                   {x * std::sqrt(1 - (y * y / 2) - (z * z / 2) + (y * y * z * z / 3)),
                                    y * std::sqrt(1 - (z * z / 2) - (x * x / 2) + (z * z * x * x / 3)),
                                    z * std::sqrt(1 - (x * x / 2) - (y * y / 2) + (x * x * y * y / 3))});

            vertex.set(x * 2, y, z);

            vertex.applyAxisAngle(direction, math::PI * x / 2).toArray(twistPositions, j);
        }

        auto& morphPositions = geometry->getMorphAttribute<float>("position");
        morphPositions.resize(2);
        morphPositions[0] = FloatBufferAttribute::create(spherePositions, 3);
        morphPositions[1] = FloatBufferAttribute::create(twistPositions, 3);

        return geometry;
    }

}// namespace

int main() {

    Canvas canvas("Morphtargets");
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color(0x8FBCD4);

    auto camera = PerspectiveCamera::create();
    camera->position.z = 10;
    scene->add(camera);

    scene->add(AmbientLight::create(0x8FBCD4, 0.4f));

    auto pointLight = PointLight::create(0xffffff, 1.f);
    camera->add(pointLight);

    auto geometry = createGeometry();

    auto material = MeshPhongMaterial::create();
    material->color = 0xff0000;
    material->flatShading = true;
    material->morphTargets = true;

    auto mesh = Mesh::create(geometry, material);
    scene->add(mesh);

    OrbitControls controls{*camera, canvas};
    controls.enableZoom = true;

    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });
}