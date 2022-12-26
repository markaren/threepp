
#include "threepp/threepp.hpp"

#include "Kine.hpp"
#include "Youbot.hpp"
#include "ik/DLSSolver.hpp"
#include "ik/CCDSolver.hpp"

#include <thread>

#include "../../imgui_helper.hpp"

using namespace threepp;
using namespace kine;

struct MyUI : public imggui_helper {

    bool mouseHover = false;
    bool jointMode = true;
    bool posMode = false;

    Vector3 pos;
    std::vector<KineLimit> limits;
    std::vector<float> values;

    explicit MyUI(const Canvas &canvas, Kine &kine)
        : imggui_helper(canvas),
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

        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};

int main() {

    Canvas canvas{Canvas::Parameters().size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.01, 100);
    camera->position.set(-15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto light = DirectionalLight::create(0xffffff, 0.1f);
    scene->position.set(1, 1, 1);
    scene->add(light);


    auto endEffectorHelper = AxesHelper::create(1);
    endEffectorHelper->visible = false;
    scene->add(endEffectorHelper);


    std::unique_ptr<Youbot> youbot;
    std::thread t([&] {
        AssimpLoader loader;
        youbot = Youbot::create("data/models/collada/youbot.dae");
        youbot->setup(canvas);

        canvas.invokeLater([&] {
            scene->add(youbot->base);
            endEffectorHelper->visible = true;
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto ikSolver = CCDSolver();
    auto kine = KineBuilder()
                        .addLink(Vector3(1.67, 1.3, 0))
                        .addRevoluteJoint(Vector3::Y*-1, {-180.f, 180.f})
                        .addLink(Vector3(0.33, 1.15, 0))
                        .addRevoluteJoint(Vector3::Z*-1, {-90.f, 90.f})
                        .addLink(Vector3(0, 1.6, 0))
                        .addRevoluteJoint(Vector3::Z*-1, {-90.f, 90.f})
                        .addLink(Vector3(0, 1.3, 0))
                        .addRevoluteJoint(Vector3::Z*-1, {-90.f, 90.f})
                        .addLink(Vector3(0, 0.85, 0))
                        .addRevoluteJoint(Vector3::Y*-1, {-180.f, 180.f})
                        .addLink(Vector3(0, 1.2, 0))
                        .build();

    MyUI ui(canvas, kine);

    auto targetHelper = AxesHelper::create(2);
    targetHelper->visible = false;
    scene->add(targetHelper);

    canvas.animate([&](float dt) {
        renderer.render(scene, camera);

        if (youbot) {

            ui.render();
            controls.enabled = !ui.mouseHover;

            auto endEffectorPosition = kine.calculateEndEffectorTransformation(ui.values);
            endEffectorHelper->position.setFromMatrixPosition(endEffectorPosition);

            targetHelper->position.copy(ui.pos);

            if (ui.jointMode) {
                ui.pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(youbot->getJointValues()));
                targetHelper->visible = false;
            }
            if (ui.posMode) {
                ui.values = ikSolver.solveIK(kine, ui.pos, youbot->getJointValues());
                targetHelper->visible = true;
            }

            youbot->setJointValues(ui.values);
            youbot->update(dt);
        }
    });

    t.join();
}
