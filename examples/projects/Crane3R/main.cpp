
#include "Crane3R.hpp"

#include "kine/Kine.hpp"
#include "threepp/controls/TransformControls.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"
#include "utility/Angle.hpp"

using namespace threepp;
using namespace kine;

#include "kine/ik/CCDSolver.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

struct CraneUI: ImguiContext {

    bool jointMode = true;
    bool posMode = false;
    bool enableController = false;

    Vector3 pos;
    std::vector<KineLimit> limits;
    std::vector<float> values;

    explicit CraneUI(const Canvas& canvas, Renderer& renderer, Kine& kine)
        : ImguiContext(canvas,renderer),
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

        ImGui::End();
    }
};

auto createGrid() {

    unsigned int size = 30;
    auto material = ShadowMaterial::create();
    auto plane = Mesh::create(PlaneGeometry::create(size, size), material);
    plane->rotation.x = -math::PI / 2;
    plane->receiveShadow = true;

    auto grid = GridHelper::create(size, size, Color::yellowgreen);
    grid->rotation.x = math::PI / 2;
    plane->add(grid);

    return plane;
}

int main() {

    Canvas canvas{"Crane3R", {{"size", WindowSize{1280, 720}}, {"antialiasing", 8}}};
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01, 100);
    camera->position.set(-15, 8, 15);

    OrbitControls controls(*camera, canvas);

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto grid = createGrid();
    scene->add(grid);

    auto endEffectorHelper = AxesHelper::create(1);
    endEffectorHelper->visible = false;
    scene->add(endEffectorHelper);

    auto light1 = AmbientLight::create(Color::white);
    auto light2 = DirectionalLight::create(Color::white);
    light2->shadow->camera->as<OrthographicCamera>()->top = 15;
    light2->shadow->camera->as<OrthographicCamera>()->bottom = -15;
    light2->shadow->camera->as<OrthographicCamera>()->left = 15;
    light2->shadow->camera->as<OrthographicCamera>()->right = -15;
    light2->position.set(-100, 100, 50);
    light2->castShadow = true;
    scene->add(light1);
    scene->add(light2);

    FontLoader fontLoader;
    const auto font = *fontLoader.load(std::string(DATA_FOLDER) + "/fonts/typeface/helvetiker_regular.typeface.json");

    auto handle = TextSprite::create(font, 20.f * monitor::contentScale().first);
    handle->setText("Loading model..");
    handle->setColor(Color::black);
    handle->setVerticalAlignment(TextSprite::VerticalAlignment::Center);
    handle->setHorizontalAlignment(TextSprite::HorizontalAlignment::Center);
    handle->screenSpace = true;
    handle->screenAnchor.set(0.5f, 0.5f);
    handle->position.set(0.f, 0.f, 0.f);
    scene->add(handle);

    Kine kine = KineBuilder()
                        .addRevoluteJoint(Vector3::Y(), {-90.f, 90.f})
                        .addLink(Vector3::Y() * 4.2)
                        .addRevoluteJoint(Vector3::X(), {-80.f, 0.f})
                        .addLink(Vector3::Z() * 7)
                        .addRevoluteJoint(Vector3::X(), {40.f, 140.f})
                        .addLink(Vector3::Z() * 5.2)
                        .build();

    Crane3R* crane = nullptr;
    auto craneGroup = loadAsync([&kine]() -> std::shared_ptr<Group> {
        auto c = Crane3R::create();
        c->setTargetValues(asAngles(kine.meanAngles(), Angle::Repr::DEG));
        c->traverseType<Mesh>([](Mesh& m) {
            m.castShadow = true;
        });
        return c;
    });

    craneGroup->onLoaded([&](AsyncGroup& g) {
        for (auto* child : g.children) {
            if (auto* c = dynamic_cast<Crane3R*>(child)) {
                crane = c;
                break;
            }
        }
        scene->remove(*handle);
        endEffectorHelper->visible = true;
    });

    scene->add(craneGroup);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    IOCapture capture{};
    capture.preventMouseEvent = [] {
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&capture);

    auto ikSolver = std::make_unique<CCDSolver>();

    CraneUI ui(canvas, *renderer, kine);

    auto targetHelper = AxesHelper::create(2);
    targetHelper->visible = false;
    scene->add(targetHelper);

    TransformControls transformControls(*camera, canvas);
    transformControls.attach(*targetHelper);
    scene->add(transformControls);

    LambdaEventListener changeListener([&](Event& event) {
        controls.enabled = !std::any_cast<bool>(event.target);
    });

    transformControls.addEventListener("dragging-changed", changeListener);

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();

        transformControls.visible = targetHelper->visible;

        renderer->render(*scene, *camera);

        if (crane) {

            ui.render();

            auto endEffectorPosition = kine.calculateEndEffectorTransformation(inDegrees(crane->getValues()));
            endEffectorHelper->position.setFromMatrixPosition(endEffectorPosition);

            if (controls.enabled) {
                targetHelper->position.copy(ui.pos);
            } else {
                ui.pos.copy(targetHelper->position);
            }

            if (ui.jointMode) {
                ui.pos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(ui.values));
                targetHelper->visible = false;
            }
            if (ui.posMode) {
                ui.values = ikSolver->solveIK(kine, targetHelper->position, inDegrees(crane->getValues()));
                targetHelper->visible = true;
            }

            crane->controllerEnabled = ui.enableController;
            crane->setTargetValues(asAngles(ui.values, Angle::Repr::DEG));

            crane->update(dt);
        }
    });
}
