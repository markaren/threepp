
#include "threepp/threepp.hpp"

#include "BulletEngine.hpp"

#include "PID.hpp"
#include "../imgui_helper.hpp"

using namespace threepp;

class MyGroup : public Group {

public:
    std::shared_ptr<Mesh> box;
    std::shared_ptr<Mesh> cylinder;

    MyGroup() {
        auto cylinderGeometry = CylinderGeometry::create(0.5, 0.5, 0.1);
        cylinderGeometry->rotateX(math::DEG2RAD * 90);

        auto boxGeometry = BoxGeometry::create(0.1, 1, 0.1);
        boxGeometry->translate(0, boxGeometry->height / 2, 0);

        auto material = MeshBasicMaterial::create();
        material->color = 0x000000;

        cylinder = Mesh::create(cylinderGeometry, material);
        box = Mesh::create(boxGeometry, material);

        add(cylinder);
        add(box);
    }
};


struct MyUI : public imggui_helper {

    bool mouseHover = false;

    PID& pid;
    float targetAngle{};

    explicit MyUI(const Canvas &canvas, PID& pid) : imggui_helper(canvas),pid(pid) {}

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("PID control");

        ImGui::Text("Target angle");
        ImGui::SliderFloat("theta", &targetAngle, -179, 179);

        ImGui::Text("PID params");
        ImGui::SliderFloat("kp", &pid.kp, 0.1, 10);
        ImGui::SliderFloat("ti", &pid.ti, 0, 1);
        ImGui::SliderFloat("td", &pid.td, 0, 1);


        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};



int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.set(0, 0, -5);

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto group = std::make_shared<MyGroup>();
    scene->add(group);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });


    BulletEngine engine;

    auto body = engine.registerGroup(group, 10);
    btHingeConstraint c(*body, btVector3(), btVector3(0, 0, 1));
    c.enableAngularMotor(true, 0, 1.f);
    engine.addConstraint(&c);

    PID pid(1, 0.001, 0.1);
    pid.setWindupGuard(0.1f);

    MyUI ui(canvas, pid);

    const float maxMotorVelocity = 5;
    canvas.animate([&](float dt) {

        controls.enabled = !ui.mouseHover;

        engine.step(dt);

        float out = pid.regulate(ui.targetAngle * math::DEG2RAD, c.getHingeAngle(), dt);
        c.setMotorTargetVelocity(out * maxMotorVelocity);


        renderer.render(scene, camera);
        ui.render();
    });
}
