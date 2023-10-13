
#include "threepp/extras/imgui/ImguiContext.hpp"
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

            spherePositions.insert(spherePositions.end(),
                                   {x * std::sqrt(1 - (y * y / 2.f) - (z * z / 2.f) + (y * y * z * z / 3.f)),
                                    y * std::sqrt(1 - (z * z / 2.f) - (x * x / 2.f) + (z * z * x * x / 3.f)),
                                    z * std::sqrt(1 - (x * x / 2.f) - (y * y / 2.f) + (x * x * y * y / 3.f))});

            vertex.set(x * 2, y, z);

            vertex.applyAxisAngle(direction, math::PI * x / 2.f).toArray(twistPositions, j);
        }

        auto morphPositions = geometry->getMorphAttribute("position");
        morphPositions->emplace_back(FloatBufferAttribute::create(spherePositions, 3));
        morphPositions->emplace_back(FloatBufferAttribute::create(twistPositions, 3));

        return geometry;
    }

}// namespace

int main() {

    Canvas canvas("Morphtargets");
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;

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
    mesh->morphTargetInfluences().emplace_back();
    mesh->morphTargetInfluences().emplace_back();
    scene->add(mesh);

    OrbitControls controls{*camera, canvas};

    auto ui = ImguiFunctionalContext(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);

        ImGui::Begin("Morphing");
        ImGui::SliderFloat("sphere", &mesh->morphTargetInfluences().at(0), 0, 1);
        ImGui::SliderFloat("twist", &mesh->morphTargetInfluences().at(1), 0, 1);
        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&] {
        renderer.render(*scene, *camera);

        ui.render();
    });
}
