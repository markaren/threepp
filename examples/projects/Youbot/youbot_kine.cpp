
#include "threepp/threepp.hpp"

#include "Youbot.hpp"
#include "kine/Kine.hpp"
#include "kine/ik/CCDSolver.hpp"

#include "KeyController.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/TextSprite.hpp"

#include <future>

using namespace threepp;
using namespace kine;

namespace {
    struct YoubotUI: ImguiContext {

        bool jointMode = true;
        bool posMode = false;

        Vector3 pos;
        std::vector<KineLimit> limits;
        std::vector<float> values;

        explicit YoubotUI(const Canvas& canvas, Kine& kine)
            : ImguiContext(canvas),
              limits(kine.limits()),
              values(kine.meanAngles()) {

            pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(values));
        }

        void onRender() override {

            ImGui::SetNextWindowPos({}, 0, {});
            ImGui::SetNextWindowSize({}, 0);
            ImGui::Begin("Youbot");

            ImGui::Text("Target angles");
            ImGui::SliderFloat("j1", &values[0], *limits[0].min(), *limits[0].max());
            jointMode = jointMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("j2", &values[1], *limits[1].min(), *limits[1].max());
            jointMode = jointMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("j3", &values[2], *limits[2].min(), *limits[2].max());
            jointMode = jointMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("j4", &values[3], *limits[3].min(), *limits[3].max());
            jointMode = jointMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("j5", &values[4], *limits[4].min(), *limits[4].max());
            jointMode = jointMode || ImGui::IsItemEdited();

            posMode = !jointMode;

            ImGui::Text("Target pos");
            ImGui::SliderFloat("px", &pos.x, -10, 10);
            posMode = posMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("py", &pos.y, -10, 10);
            posMode = posMode || ImGui::IsItemEdited();
            ImGui::SliderFloat("pz", &pos.z, -10, 10);
            posMode = posMode || ImGui::IsItemEdited();

            jointMode = !posMode;

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
    GLRenderer renderer{canvas.size()};
    renderer.autoClear = false;
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();

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

    TextSprite textHandle(font);
    textHandle.setColor(Color::black);
    textHandle.setText("Loading model..");
    textHandle.setVerticalAlignment(TextSprite::VerticalAlignment::Center);
    textHandle.setHorizontalAlignment(TextSprite::HorizontalAlignment::Center);
    textHandle.setWorldScale(20*monitor::contentScale().first);

    HUD hud(renderer);
    hud.add(textHandle).setNormalizedPosition({0.5, 0.5});

    TaskManager tm;

    std::shared_ptr<Youbot> youbot;
    std::unique_ptr<KeyController> keyController;
    auto loadFuture = std::async(std::launch::async, [&] {
        try {
            youbot = Youbot::create(std::string(DATA_FOLDER) + "/models/collada/youbot.dae");
            youbot->add(targetHelper);
            youbot->add(endEffectorHelper);
            endEffectorHelper->visible = true;
            keyController = std::make_unique<KeyController>(*youbot);

            tm.invokeLater([&] {
                canvas.addKeyListener(*keyController);
                scene->add(youbot);
                textHandle.setText("Use WASD keys to steer robot");
                textHandle.setHorizontalAlignment(TextSprite::HorizontalAlignment::Left);
                textHandle.setVerticalAlignment(TextSprite::VerticalAlignment::Above);
                hud.getStoredOptions(textHandle)->setNormalizedPosition(0, 0);
            });
        } catch (const std::exception& e) {
            tm.invokeLater([&, msg = std::string(e.what())] {
                textHandle.setText("Error: " + msg);
            });
        }
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto ikSolver = CCDSolver(1, 0.001f, 0.00001f);
    auto kine = KineBuilder()
                        .addLink(Vector3(1.67, 1.3, 0))
                        .addRevoluteJoint(Vector3::Y() * -1, {-180.f, 180.f})
                        .addLink(Vector3(0.33, 1.15, 0))
                        .addRevoluteJoint(Vector3::Z() * -1, {-90.f, 90.f})
                        .addLink(Vector3(0, 1.6, 0))
                        .addRevoluteJoint(Vector3::Z() * -1, {-90.f, 90.f})
                        .addLink(Vector3(0, 1.3, 0))
                        .addRevoluteJoint(Vector3::Z() * -1, {-90.f, 90.f})
                        .addLink(Vector3(0, 0.85, 0))
                        .addRevoluteJoint(Vector3::Y() * -1, {-180.f, 180.f})
                        .addLink(Vector3(0, 1.225, 0))
                        .build();

    YoubotUI ui(canvas, kine);

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        tm.handleTasks();

        renderer.clear();
        renderer.render(*scene, *camera);

        if (youbot) {

            ui.render();

            const auto endEffectorTransformation = kine.calculateEndEffectorTransformation(ui.values);
            endEffectorHelper->position.setFromMatrixPosition(endEffectorTransformation);
            endEffectorHelper->quaternion.setFromRotationMatrix(endEffectorTransformation);

            if (ui.jointMode) {
                ui.pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(youbot->getJointValues()));
                targetHelper->visible = false;
            }
            if (ui.posMode) {

                auto target = Matrix4().setPosition(ui.pos);
                target.premultiply(Matrix4().copy(*youbot->matrixWorld).invert());
                targetHelper->position.setFromMatrixPosition(target);
                targetHelper->quaternion.setFromRotationMatrix(target);
                ui.values = ikSolver.solveIK(kine, targetHelper->position, youbot->getJointValues());
                targetHelper->visible = true;
            }

            youbot->setJointValues(ui.values);
            keyController->update(dt);
        }

        hud.render();
    });

    loadFuture.get();

}
