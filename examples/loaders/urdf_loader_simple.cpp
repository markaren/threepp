
#include <iostream>
#include <threepp/extras/imgui/ImguiContext.hpp>
#include <threepp/loaders/OBJLoader.hpp>
#include <threepp/loaders/URDFLoader.hpp>
#include <threepp/threepp.hpp>

#include <cmath>

using namespace threepp;

int main() {

    std::filesystem::path urdfPath = std::string(DATA_FOLDER) + "/urdf/model.urdf";
    if (!exists(urdfPath)) {
        std::cerr << "File not found: " << urdfPath << std::endl;
        return 1;
    }

    Canvas canvas{"URDF loader", {{"aa", 4}}};
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.z = 1;

    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create(Color::aliceblue, Color::grey);
    scene->add(light);

    URDFLoader loader;
    auto robot = loader.load(urdfPath);
    robot->rotation.x = -math::PI / 2;
    robot->rotation.z = math::PI / 2;
    robot->showColliders(false);
    scene->add(robot);

    Box3 bb;
    bb.setFromObject(*robot);

    Vector3 size = bb.getSize();
    camera->position.set(0, size.y * 2.f, 0);
    controls.update();

    bool animate{false};
    bool showColliders{false};

    float angle = 0;
    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({0, 0}, 0);

        ImGui::Begin("Settings");

        ImGui::Checkbox("Animate", &animate);
        if (ImGui::Checkbox("Show Colliders", &showColliders)) {
            robot->showColliders(showColliders);
        }

        if (ImGui::SliderAngle("angle", &angle)) {
            robot->setJointValue(0, angle);
            animate = false;
        }

        ImGui::End();
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](const WindowSize& sz) {
        camera->aspect = sz.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(sz);
    });

    Clock clock;
    canvas.animate([&]() {
        if (animate) {
            angle = robot->getJointValue(0, false);
            robot->setJointValue(0, robot->getJointRange(0).mid() + std::sin(clock.getElapsedTime()) * 0.5f);
        }

        renderer.render(*scene, *camera);
        ui.render();
    });
}
