
#include "DCMotor.hpp"
#include "utility/Regulator.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    class VisualisationObject: public Object3D {

    public:
        VisualisationObject() {

            target = createTarget();
            add(target);

            motor = createMotor();
            add(motor);

            auto ring = createRing();
            add(ring);
        }

        void setTargetPos(float pos) {
            this->target->rotation.z = pos;
        }

        void setMotorPos(float pos) {
            this->motor->rotation.z = pos;
        }

    private:
        std::shared_ptr<Object3D> target;
        std::shared_ptr<Object3D> motor;

        static std::shared_ptr<Object3D> createTarget() {

            auto geom = BoxGeometry::create(0.1, 0.25, 0.1);
            geom->translate(0, 0.625, 0);

            auto material = MeshBasicMaterial::create();
            material->color = Color::green;
            material->transparent = true;
            material->opacity = 0.9f;

            return Mesh::create(geom, material);
        }

        static std::shared_ptr<Mesh> createMotor() {

            auto cylinderGeometry = CylinderGeometry::create(0.2, 0.2, 0.1);
            cylinderGeometry->rotateX(math::DEG2RAD * 90);

            auto boxGeometry = BoxGeometry::create(0.1, 0.5, 0.1);
            boxGeometry->translate(0, boxGeometry->height / 2, 0);

            auto material = MeshBasicMaterial::create({{"color", 0x000000}});

            auto cylinder = Mesh::create(cylinderGeometry, material);
            auto box = Mesh::create(boxGeometry, material);

            cylinder->add(box);

            return cylinder;
        }

        static std::shared_ptr<Mesh> createRing() {
            auto ringGeometry = RingGeometry::create(0.5f, 0.75f, 32, 8, math::PI / 2, math::PI);
            auto mat = MeshBasicMaterial::create({{"color", Color::red}});
            mat->opacity = 0.1f;
            mat->transparent = true;

            return Mesh::create(ringGeometry, mat);
        }
    };

}// namespace

int main() {

    Canvas canvas("MotorController", {{"aa", 6}});
    const auto size = canvas.size();
    GLRenderer renderer(size);
    renderer.autoClear = false;

    Scene scene;
    scene.background = Color::white;

    const float frustumSize = 2;
    OrthographicCamera camera(-frustumSize * size.aspect() / 2, frustumSize * size.aspect() / 2, frustumSize / 2, -frustumSize / 2);
    camera.position.z = 1;

    // motor parameters
    double resistance = 2.0;   // Ohms
    double inertia = 0.1;      // kg * m^2
    double damping = 0.05;     // N * m / rad / s
    float targetPosition = 120;// degrees

    // Create a DC motor object
    DCMotor motor(resistance, inertia, damping);

    // Create a PID controller object
    PIDRegulator controller(1.0, 0.1, 0.2);// Adjust PID gains as needed

    auto motorVisuals = VisualisationObject();
    scene.add(motorVisuals);

    HUD hud(canvas.size());

    FontLoader fontLoader;
    auto font = fontLoader.defaultFont();
    TextGeometry::Options opts(font, 20);

    auto targetText = Text2D::create(opts, "Target position: " + std::to_string(targetPosition));
    targetText->setColor(Color::black);
    hud.add(targetText, HUD::Options().setNormalizedPosition({0.f, 0.05f}));

    auto measuredText = Text2D::create(opts, "Measured position: " + std::to_string(math::radToDeg(motor.getPosition())));
    measuredText->setColor(Color::black);
    hud.add(measuredText, HUD::Options());

    canvas.onWindowResize([&](WindowSize size) {
        camera.left = -frustumSize * size.aspect() / 2;
        camera.right = frustumSize * size.aspect() / 2;
        camera.top = frustumSize / 2;
        camera.bottom = -frustumSize / 2;
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto& params = controller.params();
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Motor Controller");

        ImGui::Text("Target position");
        if(ImGui::SliderFloat("deg", &targetPosition, 0, 180)) {
            targetText->setText("Target position: " + std::to_string(targetPosition), opts);
        }
        ImGui::Text("PID gains");
        ImGui::SliderFloat("kp", &params.kp, 0.01f, 10.f);
        ImGui::SliderFloat("ti", &params.ti, 0.001f, 2.f);
        ImGui::SliderFloat("td", &params.td, 0.001f, 2.f);

        ImGui::End();
    });


    Clock clock;
    long long it{};
    canvas.animate([&]() {
        float dt = clock.getDelta();
        double measuredPosition = motor.getPosition();

        if (it++ % 10 == 2) {
            measuredText->setText("Measured position: " + std::to_string(math::radToDeg(measuredPosition)), opts);
        }

        double error = math::degToRad(targetPosition) - measuredPosition;
        double gain = controller.regulate(error, dt);

        motor.update(gain, dt);

        renderer.clear();
        renderer.render(scene, camera);
        hud.apply(renderer);
        ui.render();

        motorVisuals.setTargetPos(math::degToRad(targetPosition));
        motorVisuals.setMotorPos(static_cast<float>(motor.getPosition()));
    });
}
