
#include <threepp/extras/imgui/ImguiContext.hpp>
#include <threepp/loaders/URDFLoader.hpp>
#include <threepp/threepp.hpp>

#include <cmath>

using namespace threepp;

int main(int argc, char** argv) {

    std::filesystem::path urdfPath = std::filesystem::path(DATA_FOLDER) / "urdf" / "lbr_iiwa_14_r820.urdf";
    if (argc > 1) {
        urdfPath = argv[1];
        if (!exists(urdfPath)) {
            std::cerr << "File not found: " << urdfPath << std::endl;
            return 1;
        }
    }

    Canvas canvas{"URDF loader", {{"aa", 4}}};
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.z = 1;

    OrbitControls controls{*camera, canvas};

    auto light = HemisphereLight::create(Color::aliceblue, Color::grey);
    light->intensity = 1.5f;
    scene->add(light);

    auto ambientLight = AmbientLight::create(0xffffff, 0.3f);
    scene->add(ambientLight);

    URDFLoader loader;
    auto robot = loader.load(urdfPath);
    robot->rotation.x = -math::PI / 2;
    robot->showColliders(false);
    scene->add(robot);

    Box3 bb;
    bb.setFromObject(*robot);

    Vector3 size = bb.getSize();
    camera->position.set(0, size.y * 1.5f, size.z * 3.f);
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

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
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

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {

        if (animate) {
            for (auto i = 0; i < robot->numDOF(); ++i) {
                jointValues[i] = robot->getJointValue(i, true);
                robot->setJointValue(i, std::sin(clock.getElapsedTime()) * 0.5f);
            }
        }

        const auto m = robot->computeEndEffectorTransform(jointValues, true);
        axis->position.setFromMatrixPosition(m);
        axis->quaternion.setFromRotationMatrix(m);

        renderer->render(*scene, *camera);
        ui.render();
    });
}
