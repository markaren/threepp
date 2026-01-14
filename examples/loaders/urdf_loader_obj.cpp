
#include <iostream>
#include <threepp/extras/imgui/ImguiContext.hpp>
#include <threepp/loaders/OBJLoader.hpp>
#include <threepp/loaders/URDFLoader.hpp>
#include <threepp/threepp.hpp>

#include <cmath>

using namespace threepp;

int main() {

    std::filesystem::path urdfPath = std::string(DATA_FOLDER) + "/urdf/crane3r.urdf";
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
    loader.setGeometryLoader(std::make_shared<OBJLoader>());
    auto robot = loader.load(urdfPath);
    robot->rotation.x = -math::PI / 2;
    robot->showColliders(false);
    auto ranges = robot->getJointRanges();
    std::vector<float> midValues(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        if (std::isinf(ranges[i].min) && std::isinf(ranges[i].max)) {
            midValues[i] = 0.f;
        } else {
            midValues[i] = (ranges[i].min + ranges[i].max) / 2.f;
        }
    }
    robot->setJointValues(midValues);
    scene->add(robot);

    Box3 bb;
    bb.setFromObject(*robot);

    Vector3 size = bb.getSize();
    camera->position.set(0, size.y * 1.5f, size.x);
    controls.update();

    bool animate{false};
    bool showColliders{false};
    const auto info = robot->getArticulatedJointInfo();
    std::vector<float> jointValues = robot->jointValues(true);


    auto axis = AxesHelper::create(size.length() * 0.1f);
    scene->add(axis);

    std::vector<std::string> labels;
    for (auto i = 0; i < robot->numDOF(); i++) {
        labels.emplace_back("j" + std::to_string(i + 1));
    }

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({0, 0}, 0);

        ImGui::Begin("Settings");

        ImGui::Checkbox("Animate", &animate);
        if (ImGui::Checkbox("Show Colliders", &showColliders)) {
            robot->showColliders(showColliders);
        }

        for (auto i = 0; i < robot->numDOF(); i++) {
            const auto type = info[i].type;
            const auto minmax = robot->getJointRange(i, true);
            const bool isRevolute = type == Robot::JointType::Revolute;
            const float min = minmax.min > (isRevolute ? -360.f : -1.f) ? minmax.min : (isRevolute ? -360.f : -1.f);
            const float max = minmax.max < (isRevolute ? 360.f : 1.f) ? minmax.max : (isRevolute ? 360.f : 1.f);
            if (ImGui::SliderFloat(labels[i].c_str(), &jointValues[i], min, max)) {
                robot->setJointValue(i, jointValues[i], isRevolute);
                animate = false;
            }
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
            for (auto i = 0; i < robot->numDOF(); ++i) {
                jointValues[i] = robot->getJointValue(i, true);
                robot->setJointValue(i, robot->getJointRange(i).mid() + std::sin(clock.getElapsedTime()) * 0.5f);
            }
        }

        const auto m = robot->getEndEffectorTransform();
        axis->position.setFromMatrixPosition(m);
        axis->quaternion.setFromRotationMatrix(m);

        renderer.render(*scene, *camera);
        ui.render();
    });
}
