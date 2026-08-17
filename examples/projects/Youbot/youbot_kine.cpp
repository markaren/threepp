// The KUKA youbot, driven by the library's inverse kinematics.
//
// The Collada model is geometry only — it carries no joint metadata — so the
// arm is described here as a Robot: a kinematic tree of five revolute joints
// with the gripper on a fixed joint at the end. That Robot is what IkSolver
// solves against, and the joint values it produces are then written to the
// model's own joint nodes. Nothing but the numbers is shared between the two,
// which is exactly the split a real cell has between its controller's model of
// the arm and the arm itself.
//
// Everything here is in radians and in the units the model is drawn in (it is
// scaled x10 on load, so one unit is a decimetre). The IK options below are
// scaled to match — a solver tuned for a metre-scale arm takes steps ten times
// too small in this world.
//
// Sliders command either the joint angles directly or a tool position for the
// solver to reach; WASD drives the base.

#include "renderer_factory.hpp"

#include "threepp/threepp.hpp"

#include "Youbot.hpp"

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/kinematics/InverseKinematics.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/TextSprite.hpp"

#include <string>
#include <vector>

using namespace threepp;

namespace {

    // The arm, root to tip. Each joint sits at `origin` in the frame of the
    // joint before it and turns about `axis`. The axes alternate between -Y and
    // -Z rather than being the +Z a URDF would use, because these are the
    // model's own joint frames.
    struct ArmJoint {
        std::string name;
        Vector3 origin;
        Vector3 axis;
        float limit;// symmetric, radians
    };

    const std::vector<ArmJoint> armJoints{
            {"j1", {1.67f, 1.3f, 0.f}, {0.f, -1.f, 0.f}, math::PI},
            {"j2", {0.33f, 1.15f, 0.f}, {0.f, 0.f, -1.f}, math::PI / 2},
            {"j3", {0.f, 1.6f, 0.f}, {0.f, 0.f, -1.f}, math::PI / 2},
            {"j4", {0.f, 1.3f, 0.f}, {0.f, 0.f, -1.f}, math::PI / 2},
            {"j5", {0.f, 0.85f, 0.f}, {0.f, -1.f, 0.f}, math::PI},
    };

    // Last joint frame -> gripper tip. The gripper does not articulate here, so
    // it is a fixed joint on the robot rather than an IkOptions::toolOffset:
    // either would place the same tool frame, but as part of the robot it also
    // shows up in the robot's own end-effector transform.
    const Vector3 toolOffset{0.f, 1.225f, 0.f};

    std::shared_ptr<Object3D> namedObject(const std::string& name) {
        auto object = std::make_shared<Object3D>();
        object->name = name;
        return object;
    }

    // The kinematic model of the arm, assembled the way URDFLoader assembles one
    // from a file: links, then the joints that connect them, then finalize().
    std::shared_ptr<Robot> createArmModel() {

        auto robot = std::make_shared<Robot>();

        auto linkName = [](size_t index) {
            return index == 0 ? std::string("base") : "link" + std::to_string(index);
        };

        for (size_t i = 0; i <= armJoints.size(); ++i) {
            robot->addLink(namedObject(linkName(i)));
        }
        robot->addLink(namedObject("tool"));

        for (size_t i = 0; i < armJoints.size(); ++i) {
            const auto& joint = armJoints[i];

            auto jointObject = namedObject(joint.name);
            jointObject->position.copy(joint.origin);

            robot->addJoint(jointObject, {.axis = joint.axis,
                                          .type = Robot::JointType::Revolute,
                                          .name = joint.name,
                                          .range = Robot::JointRange{-joint.limit, joint.limit},
                                          .parent = linkName(i),
                                          .child = linkName(i + 1)});
        }

        auto toolJoint = namedObject("tool_mount");
        toolJoint->position.copy(toolOffset);
        robot->addJoint(toolJoint, {.type = Robot::JointType::Fixed,
                                    .name = "tool_mount",
                                    .parent = linkName(armJoints.size()),
                                    .child = "tool"});

        robot->finalize();
        robot->updateMatrix();

        return robot;
    }

    struct YoubotUI: ImguiContext {

        bool jointMode = true;
        bool posMode = false;

        Vector3 pos;
        std::vector<Robot::JointRange> limits;
        std::vector<float> values;
        RendererSettings settings;

        YoubotUI(const Canvas& canvas, Renderer& renderer, const Robot& robot)
            : ImguiContext(canvas, renderer),
              limits(robot.getJointRanges()),
              values(robot.numDOF(), 0.f),
              settings(renderer) {

            pos.setFromMatrixPosition(robot.computeEndEffectorTransform(values));
        }

    protected:
        void onRender() override {

            ImGui::SetNextWindowPos({}, 0, {});
            ImGui::SetNextWindowSize({}, 0);
            ImGui::Begin("Youbot");

            // The values are radians; SliderAngle shows them as degrees.
            ImGui::Text("Target angles");
            for (size_t i = 0; i < values.size(); ++i) {
                const auto label = "j" + std::to_string(i + 1);
                ImGui::SliderAngle(label.c_str(), &values[i],
                                   math::radToDeg(limits[i].min), math::radToDeg(limits[i].max));
                jointMode = jointMode || ImGui::IsItemEdited();
            }

            posMode = !jointMode;

            ImGui::Text("Target pos");
            ImGui::SliderFloat("px", &pos.x, -10, 10);
            posMode = posMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("py", &pos.y, -10, 10);
            posMode = posMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("pz", &pos.z, -10, 10);
            posMode = posMode || ImGui::IsItemEdited();

            jointMode = !posMode;

            settings.drawCollapsed();

            ImGui::End();
        }
    };

    void setupLights(Scene& scene) {
        auto light1 = DirectionalLight::create(0xffffff, 1.f);
        light1->position.set(1, 1, 1);
        scene.add(light1);

        auto light2 = AmbientLight::create(0xffffff, 1.f);
        scene.add(light2);
    }
}// namespace

int main() {

    Canvas canvas{Canvas::Parameters().title("Youbot-kine").size({1280, 720}).antialiasing(8)};
    auto renderer = createRenderer(canvas);
    renderer->toneMapping = ToneMapping::ACESFilmic;


    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01, 100);
    camera->position.set(-15, 8, 15);

    OrbitControls controls(*camera, canvas);

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    setupLights(*scene);

    auto endEffectorHelper = AxesHelper::create(1);
    endEffectorHelper->visible = false;

    auto targetHelper = AxesHelper::create(2);
    targetHelper->visible = false;

    FontLoader fontLoader;
    const auto font = *fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/helvetiker_regular.typeface.json");

    auto textHandle = TextSprite::create(font);
    textHandle->setColor(Color::black);
    textHandle->setText("Loading model..");
    textHandle->setVerticalAlignment(TextSprite::VerticalAlignment::Center);
    textHandle->setHorizontalAlignment(TextSprite::HorizontalAlignment::Center);
    textHandle->setWorldScale(20*monitor::contentScale().first);
    textHandle->screenSpace = true;
    textHandle->screenAnchor.set(0.5f, 0.5f);          // viewport centre
    textHandle->position.set(-5.f, -5.f, 0.f);         // matches old HUD margin sign at pos.x/y >= 0.5
    scene->add(textHandle);

    Youbot* youbot = nullptr;
    auto youbotGroup = loadAsync([path = std::string(DATA_FOLDER) + "/models/collada/youbot.dae"]() -> std::shared_ptr<Group> {
        auto y = Youbot::create(path);
        auto wrapper = Group::create();
        wrapper->add(std::shared_ptr<Youbot>(y.release()));
        return wrapper;
    });

    youbotGroup->onLoaded([&](AsyncGroup& g) {
        g.traverse([&](Object3D& obj) {
            if (!youbot) {
                if (auto* y = dynamic_cast<Youbot*>(&obj)) {
                    youbot = y;
                }
            }
        });
        if (youbot) {
            youbot->add(targetHelper);
            youbot->add(endEffectorHelper);
            endEffectorHelper->visible = true;
            textHandle->setText("Use WASD keys to steer robot");
            textHandle->setVerticalAlignment(TextSprite::VerticalAlignment::Above);
            textHandle->setHorizontalAlignment(TextSprite::HorizontalAlignment::Left);
            textHandle->screenAnchor.set(0.f, 0.f);    // bottom-left
            textHandle->position.set(5.f, 5.f, 0.f);   // 5 px margin
        } else {
            textHandle->setText("Error loading model");
        }
    });

    scene->add(youbotGroup);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    const auto arm = createArmModel();

    IkOptions ikOptions;
    ikOptions.task = IkTask::Position;// point the tool at the target; any wrist angle will do
    // Distances are decimetres here, so the metre-scale defaults are scaled by
    // ten: the tolerance, the per-iteration trust region, and the damping alike
    // (damping is compared against the Jacobian's singular values, which grow
    // with the arm).
    ikOptions.positionTolerance = 1e-3f;
    ikOptions.maxPositionStep = 0.5f;
    ikOptions.damping = 0.8f;
    // The solve runs once per frame, so it is a tracker rather than a one-shot:
    // a handful of iterations per frame is plenty, and the speed cap turns what
    // is left of the error into smooth motion instead of a snap.
    ikOptions.maxIterations = 10;
    ikOptions.maxJointSpeed = 3.f;

    const IkSolver ikSolver(*arm, ikOptions);

    YoubotUI ui(canvas, *renderer, *arm);

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        renderer->render(*scene, *camera);

        if (youbot) {

            ui.render();

            // The target is a point in the WORLD, while the arm model — and so
            // the solver — works in the robot's own frame. Converting between
            // the two here is what makes the arm hold a fixed point as the base
            // drives out from under it.
            if (ui.posMode) {
                targetHelper->position.copy(ui.pos)
                        .applyMatrix4(Matrix4().copy(*youbot->matrixWorld).invert());
                targetHelper->visible = true;

                ikSolver.solve(ui.values, targetHelper->position, dt);
            } else {
                // Follow the tool with the sliders, so switching to position
                // mode picks up where the arm already is.
                ui.pos.setFromMatrixPosition(arm->computeEndEffectorTransform(ui.values));
                ui.pos.applyMatrix4(*youbot->matrixWorld);
                targetHelper->visible = false;
            }

            const auto toolTransform = ikSolver.toolTransform(ui.values);
            endEffectorHelper->position.setFromMatrixPosition(toolTransform);
            endEffectorHelper->quaternion.setFromRotationMatrix(toolTransform);

            youbot->setJointValues(ui.values);

            // WASD steering — poll held keys directly (no listener class, no manual state).
            if (canvas.isKeyDown(Key::W)) youbot->driveForwards(dt);
            if (canvas.isKeyDown(Key::S)) youbot->driveBackwards(dt);
            if (canvas.isKeyDown(Key::D)) youbot->driveRight(dt);
            if (canvas.isKeyDown(Key::A)) youbot->driveLeft(dt);
        }
    });
}
