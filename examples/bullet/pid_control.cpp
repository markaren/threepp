
#include "threepp/threepp.hpp"

#include "BulletEngine.hpp"

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

    float targetAngle{};

    explicit MyUI(const Canvas &canvas) : imggui_helper(canvas) {}

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("PID control");

        ImGui::Text("Target angle");
        ImGui::SliderFloat("px", &targetAngle, -179, 179);


        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};

class PID {

public:
    PID(float kp, float ti, float td) : kp(kp), ti(ti), td(td) {}

    float regulate(float setPoint, float measuredValue, float dt) {
        if (dt == 0) dt = std::numeric_limits<float>::min();

        float curr_error = (setPoint - measuredValue);

        // integration with windup guarding
        integral += (curr_error * dt);
        if (windup_guard) {
            if (integral < -(*windup_guard)) {
                integral = -(*windup_guard);
            } else if (integral > *windup_guard) {
                integral = *windup_guard;
            }
        }

        // differentiation
        float diff = ((curr_error - prev_error) / dt);

        // save current error as previous error for next iteration
        prev_error = curr_error;

        // scaling
        float P = (kp * curr_error);
        float I = (ti * integral);
        float D = (td * diff);

        // summation of terms
        return math::clamp(P + I + D, -1.f, 1.f);
    }

    void setWindupGuard(const std::optional<float> &windupGuard) {
        windup_guard = windupGuard;
    }

private:
    float kp;
    float ti;
    float td;

    float integral;
    float prev_error{};
    std::optional<float> windup_guard;
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

    auto body = engine.registerGroup(group, 1);
    btHingeConstraint c(*body, btVector3(), btVector3(0, 0, 1));
    c.enableAngularMotor(true, 0, std::numeric_limits<float>::max());
    engine.addConstraint(&c);
    //    c.setLimit(math::DEG2RAD * -90, math::DEG2RAD * 90, , 0.01f, 0.f);
    MyUI ui(canvas);

    PID pid(1, 0.001, 0.1);
    pid.setWindupGuard(0.1f);

    float maxMotorVelocity = 5;
    float t = 0;
    canvas.animate([&](float dt) {
        engine.step(dt);
        float out = pid.regulate(ui.targetAngle * math::DEG2RAD, c.getHingeAngle(), dt);
        c.setMotorTargetVelocity(out * maxMotorVelocity);

        t = 0;

        controls.enabled = !ui.mouseHover;

        renderer.render(scene, camera);
        ui.render();
    });
}
