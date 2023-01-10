
#include "threepp/threepp.hpp"

#include "BulletEngine.hpp"

#include "PID.hpp"
#include "threepp/extras/imgui/imgui_context.hpp"

using namespace threepp;

std::shared_ptr<Object3D> createTarget() {
    auto geom = BoxGeometry::create(0.1, 1, 0.1);
    geom->translate(0, geom->height / 2 + 1, 0);
    auto material = MeshBasicMaterial::create();
    material->color = Color::green;
    material->transparent = true;
    material->opacity = 0.9f;
    auto target = Mesh::create(geom, material);

    return target;
}

std::shared_ptr<Group> createObject() {
    auto group = Group::create();

    auto cylinderGeometry = CylinderGeometry::create(0.5, 0.5, 0.1);
    cylinderGeometry->rotateX(math::DEG2RAD * 90);

    auto boxGeometry = BoxGeometry::create(0.1, 1, 0.1);
    boxGeometry->translate(0, boxGeometry->height / 2, 0);

    auto material = MeshBasicMaterial::create();
    material->color = 0x000000;

    auto cylinder = Mesh::create(cylinderGeometry, material);
    auto box = Mesh::create(boxGeometry, material);

    group->add(cylinder);
    group->add(box);

    return group;
}

struct ControllableOptions {
    float targetAngle;
    float maxMotorVelocity;

    PID &pid;

    explicit ControllableOptions(PID &pid, float targetAngle = 0, float maxMotorVelocity = 5) : targetAngle(targetAngle), maxMotorVelocity(maxMotorVelocity), pid(pid) {}
};


struct MyUI : imgui_context {


    explicit MyUI(const Canvas &canvas, ControllableOptions &opt)
        : imgui_context(canvas.window_ptr()), opt(opt) {}

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("PID control");

        ImGui::Text("Motor settings");
        ImGui::SliderFloat("theta", &opt.targetAngle, -179, 179);
        ImGui::SliderFloat("maxSpeed", &opt.maxMotorVelocity, 0.1, 20);

        ImGui::Text("PID params");
        ImGui::SliderFloat("kp", &opt.pid.kp, 0.1, 10);
        ImGui::SliderFloat("ti", &opt.pid.ti, 0, 1);
        ImGui::SliderFloat("td", &opt.pid.td, 0, 1);

        ImGui::PlotLines("Error", errors.data(), static_cast<int>(errors.size()));

        ImGui::End();

        errors.emplace_back(opt.pid.error());
        if (errors.size() > 100) {
            errors.erase(errors.begin(), errors.begin() + 1);
        }
    }

private:
    ControllableOptions &opt;
    std::vector<float> errors;
};


int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(-4, 4, -3, 3);
    camera->position.set(0, 0, 5);

    auto controllable = createObject();
    scene->add(controllable);

    auto target = createTarget();
    scene->add(target);

    canvas.onWindowResize([&](WindowSize size) {
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });


    BulletEngine engine;

    auto body = engine.registerGroup(controllable, 10);
    btHingeConstraint c(*body, btVector3(), btVector3(0, 0, 1));
    c.enableAngularMotor(true, 0, 1.f);
    engine.addConstraint(&c);

    PID pid(1, 0.001, 0.1);
    pid.setWindupGuard(0.1f);

    ControllableOptions opt(pid);

    MyUI ui(canvas, opt);

    canvas.animate([&](float dt) {
        engine.step(dt);

        float out = pid.regulate(opt.targetAngle * math::DEG2RAD, c.getHingeAngle(), dt);
        c.setMotorTargetVelocity(out * opt.maxMotorVelocity);

        target->rotation.z = opt.targetAngle * math::DEG2RAD;

        renderer.render(scene, camera);
        ui.render();
    });
}
