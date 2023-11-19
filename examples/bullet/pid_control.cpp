
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physics/BulletPhysics.hpp"
#include "threepp/threepp.hpp"

#include "utility/PID.hpp"

using namespace threepp;

namespace {

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

    std::shared_ptr<Mesh> createObject() {

        auto cylinderGeometry = CylinderGeometry::create(0.5, 0.5, 0.1);
        cylinderGeometry->rotateX(math::DEG2RAD * 90);

        auto boxGeometry = BoxGeometry::create(0.1, 1, 0.1);
        boxGeometry->translate(0, boxGeometry->height / 2, 0);

        auto material = MeshBasicMaterial::create({{"color", 0x000000}});

        auto cylinder = Mesh::create(cylinderGeometry, material);
        auto box = Mesh::create(boxGeometry, material);

        cylinder->add(box);

        return cylinder;
    }

    struct ControllableOptions {

        float targetAngle;
        float maxMotorVelocity;

        ControllableOptions(float targetAngle, float maxMotorVelocity)
            : targetAngle(targetAngle), maxMotorVelocity(maxMotorVelocity) {}
    };


    struct MyUI: ImguiContext {

        explicit MyUI(const Canvas& canvas, PID& pid, ControllableOptions& opt)
            : ImguiContext(canvas.windowPtr()), pid_(pid), opt_(opt) {}

        void onRender() override {

            ImGui::SetNextWindowPos({}, 0, {});
            ImGui::SetNextWindowSize({}, 0);
            ImGui::Begin("PID control");

            ImGui::Text("Motor settings");
            ImGui::SliderFloat("theta", &opt_.targetAngle, -179, 179);
            ImGui::SliderFloat("maxSpeed", &opt_.maxMotorVelocity, 0.1f, 20);

            ImGui::Text("PID params");
            ImGui::SliderFloat("kp", &pid_.params().kp, 0.1f, 10);
            ImGui::SliderFloat("ti", &pid_.params().ti, 0, 1);
            ImGui::SliderFloat("td", &pid_.params().td, 0, 1);

            ImGui::PlotLines("Error", errors.data(), static_cast<int>(errors.size()));

            ImGui::End();

            errors.emplace_back(pid_.error());
            if (errors.size() > 100) {
                errors.erase(errors.begin(), errors.begin() + 1);
            }
        }

    private:
        PID& pid_;
        ControllableOptions& opt_;
        std::vector<float> errors;
    };


}// namespace

int main() {

    Canvas canvas("PID control");
    GLRenderer renderer(canvas.size());
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

    BulletPhysics bullet;
    bullet.addMesh(*controllable, 100, true);
    auto rb = bullet.get(*controllable);
    btHingeConstraint c(*rb->body, {}, {0, 0, 1});
    c.enableAngularMotor(true, 0, 1.f);
    bullet.addConstraint(&c);

    PID pid(1, 0.001f, 0.1f);
    pid.setWindupGuard(0.1f);

    ControllableOptions opt(0, 5);
    MyUI ui(canvas, pid, opt);

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();
        bullet.step(dt);

        float out = pid.regulate(opt.targetAngle * math::DEG2RAD, c.getHingeAngle(), dt);
        c.setMotorTargetVelocity(out * opt.maxMotorVelocity);

        target->rotation.z = opt.targetAngle * math::DEG2RAD;

        renderer.render(*scene, *camera);
        ui.render();
    });
}
