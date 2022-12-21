
#include "Crane3R.hpp"
#include "threepp/threepp.hpp"

#include <thread>

using namespace threepp;

#ifdef HAS_IMGUI
#include "../../imgui_helper.hpp"

#include "Kine.hpp"
#include "ik/DLSSolver.hpp"

using namespace kine;

template<size_t nKineDof>
struct MyUI : public imggui_helper {

    bool mouseHover = false;
    bool jointMode = false;
    bool posMode = false;

    Vector3 pos;
    std::array<KineLimit, nKineDof> limits;
    std::array<float, nKineDof> angles;

    explicit MyUI(const Canvas &canvas, Kine<nKineDof> &kine)
        : imggui_helper(canvas),
          limits(kine.limits()),
          angles(kine.meanAngles()) {

        pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(angles));
    }

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Crane3R");

        ImGui::Text("Target angles");
        ImGui::SliderFloat("j1", &angles[0], *limits[0].min(), *limits[0].max());
        jointMode = ImGui::IsItemEdited();
        ImGui::SliderFloat("j2", &angles[1], *limits[1].min(), *limits[1].max());
        jointMode = jointMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("j3", &angles[2], *limits[2].min(), *limits[2].max());
        jointMode = jointMode || ImGui::IsItemEdited();

        ImGui::Text("Target pos");
        ImGui::SliderFloat("px", &pos.x, -10, 10);
        posMode = ImGui::IsItemEdited();
        ImGui::SliderFloat("py", &pos.y, 0, 10);
        posMode = posMode || ImGui::IsItemEdited();
        ImGui::SliderFloat("pz", &pos.z, 0, 10);
        posMode = posMode || ImGui::IsItemEdited();

        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};
#endif

int main() {

    Canvas canvas{Canvas::Parameters().size({1280, 720}).antialiasing(8)};
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

    std::shared_ptr<Crane3R> crane;
    std::thread t([&] {
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

    DLSSolver<3> ikSolver;
    Kine<3> kine = KineBuilder()
                           .addRevoluteJoint(Vector3::Y, {-90.f, 90.f})
                           .addLink(Vector3::Y * 4.2)
                           .addRevoluteJoint(Vector3::X, {-80.f, 0.f})
                           .addLink(Vector3::Z * 7)
                           .addRevoluteJoint(Vector3::X, {40.f, 140.f})
                           .addLink(Vector3::Z * 5.2)
                           .build<3>();

    MyUI<3> ui(canvas, kine);

    auto targetHelper = AxesHelper::create(2);
    targetHelper->visible = false;
    scene->add(targetHelper);

#endif
    canvas.animate([&] {
        renderer.render(scene, camera);

        if (crane) {

#ifdef HAS_IMGUI
            ui.render();
            controls.enabled = !ui.mouseHover;

            auto endEffectorPosition = kine.calculateEndEffectorTransformation(crane->getValues());
            endEffectorHelper->position.setFromMatrixPosition(endEffectorPosition);

            targetHelper->position.copy(ui.pos);

            if (ui.jointMode) {
                ui.pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(ui.angles));
                targetHelper->visible = false;
            }
            if (ui.posMode) {
                ui.angles = ikSolver.solveIK(kine, ui.pos, crane->getValues());
                targetHelper->visible = true;
            }

            crane->setTargetValues(ui.angles);
#endif

            crane->update();
        }
    });

    t.join();
}
