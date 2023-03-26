
#include "Crane3R.hpp"

#include "threepp/threepp.hpp"
#include "threepp/utils/ThreadPool.hpp"

using namespace threepp;

#ifdef HAS_IMGUI
#include "threepp/extras/imgui/imgui_context.hpp"

#include "kine/Kine.hpp"
#include "kine/ik/CCDSolver.hpp"

using namespace kine;

struct MyUI: imgui_context {

    bool mouseHover = false;
    bool jointMode = true;
    bool posMode = false;
    bool enableController = false;

    Vector3 pos;
    std::vector<KineLimit> limits;
    std::vector<float> values;

    explicit MyUI(const Canvas& canvas, Kine& kine)
        : imgui_context(canvas.window_ptr()),
          limits(kine.limits()),
          values(kine.meanAngles()) {

        pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(values));
    }

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Crane3R");

        ImGui::Text("Target angles");
        ImGui::SliderFloat("j1", &values[0], *limits[0].min(), *limits[0].max());
        jointMode = jointMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("j2", &values[1], *limits[1].min(), *limits[1].max());
        jointMode = jointMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("j3", &values[2], *limits[2].min(), *limits[2].max());
        jointMode = jointMode || ImGui::IsItemEdited();

        posMode = !jointMode;

        ImGui::Text("Target pos");
        ImGui::SliderFloat("px", &pos.x, -10, 10);
        posMode = posMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("py", &pos.y, 0, 10);
        posMode = posMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("pz", &pos.z, 0, 10);
        posMode = posMode || ImGui::IsItemEdited();

        jointMode = !posMode;

        ImGui::Checkbox("controller", &enableController);

        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};
#endif

int main() {

    Canvas canvas{"Crane3R", {{"size", WindowSize{1280, 720}}, {"antialiasing", 8}}};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create(60, canvas.getAspect(), 0.01, 100);
    camera->position.set(-15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto scene = Scene::create();

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto endEffectorHelper = AxesHelper::create(1);
    endEffectorHelper->visible = false;
    scene->add(endEffectorHelper);

    auto light = AmbientLight::create(Color::white);
    scene->add(light);

    utils::ThreadPool pool;
    std::shared_ptr<Crane3R> crane;
    pool.submit([&] {
        crane = Crane3R::create();
        canvas.invokeLater([&, crane] {
            scene->add(crane);
            endEffectorHelper->visible = true;
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });


#ifdef HAS_IMGUI

    auto ikSolver = std::make_unique<CCDSolver>();
    Kine kine = KineBuilder()
                        .addRevoluteJoint(Vector3::Y(), {-90.f, 90.f})
                        .addLink(Vector3::Y() * 4.2)
                        .addRevoluteJoint(Vector3::X(), {-80.f, 0.f})
                        .addLink(Vector3::Z() * 7)
                        .addRevoluteJoint(Vector3::X(), {40.f, 140.f})
                        .addLink(Vector3::Z() * 5.2)
                        .build();

    MyUI ui(canvas, kine);

    auto targetHelper = AxesHelper::create(2);
    targetHelper->visible = false;
    scene->add(targetHelper);

#endif
    canvas.animate([&](float dt) {
        renderer.render(scene, camera);

        if (crane) {

#ifdef HAS_IMGUI
            ui.render();
            controls.enabled = !ui.mouseHover;

            auto endEffectorPosition = kine.calculateEndEffectorTransformation(inDegrees(crane->getValues()));
            endEffectorHelper->position.setFromMatrixPosition(endEffectorPosition);

            targetHelper->position.copy(ui.pos);

            if (ui.jointMode) {
                ui.pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(ui.values));
                targetHelper->visible = false;
            }
            if (ui.posMode) {
                ui.values = ikSolver->solveIK(kine, ui.pos, inDegrees(crane->getValues()));
                targetHelper->visible = true;
            }

            crane->controllerEnabled = ui.enableController;
            crane->setTargetValues(asAngles(ui.values, Angle::Repr::DEG));
#endif

            crane->update(dt);
        }
    });

}
