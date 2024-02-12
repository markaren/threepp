
#include "DCMotor.hpp"
#include "utility/PID.hpp"

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

using namespace threepp;

namespace {

    std::shared_ptr<Object3D> createTarget() {

        auto geom = BoxGeometry::create(0.1, 0.25, 0.1);
        geom->translate(0, 0.625, 0);

        auto material = MeshBasicMaterial::create();
        material->color = Color::green;
        material->transparent = true;
        material->opacity = 0.9f;

        auto target = Mesh::create(geom, material);
        return target;
    }

    std::shared_ptr<Mesh> createObject() {

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

}// namespace

int main() {

    Canvas canvas("MotorController");
    GLRenderer renderer(canvas.size());
    renderer.autoClear = false;

    Scene scene;
    scene.background = Color::white;

    PerspectiveCamera camera(50, canvas.size().aspect(), 0.1, 100);
    camera.position.z = 2;

    // motor parameters
    double resistance = 2.0;   // Ohms
    double inertia = 0.1;      // kg * m^2
    double damping = 0.05;     // N * m / rad / s
    float targetPosition = 120;// degrees

    // Create a DC motor object
    DCMotor motor(resistance, inertia, damping);

    // Create a PID controller object
    PID controller(1.0, 0.1, 0.2);// Adjust PID gains as needed

    FontLoader fontLoader;
    auto font = fontLoader.defaultFont();

    auto object = createObject();
    scene.add(object);

    auto target = createTarget();
    scene.add(target);

    HUD hud(canvas.size());
    TextGeometry::Options opts(font, 20);
    auto targetText = Text2D::create(opts, "Target position: " + std::to_string(targetPosition));
    targetText->setColor(Color::black);
    hud.add(targetText, HUD::Options().setNormalizedPosition({0.f, 0.05f}));
    auto measuredText = Text2D::create(opts, "Measured position: " + std::to_string(math::radToDeg(motor.getPosition())));
    measuredText->setColor(Color::black);
    hud.add(measuredText, HUD::Options());

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    auto& params = controller.params();
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Motor Controller");

        ImGui::Text("Target position");
        ImGui::SliderFloat("deg", &targetPosition, 0, 180);
        ImGui::Text("PID gains");
        ImGui::SliderFloat("kp", &params.kp, 0.01f, 10.f);
        ImGui::SliderFloat("ti", &params.ti, 0.001f, 1.f);
        ImGui::SliderFloat("td", &params.td, 0.001f, 1.f);

        ImGui::End();
    });


    Clock clock;
    long long it{};
    canvas.animate([&]() {
        float dt = clock.getDelta();
        double measuredPosition = motor.getPosition();

        if (it++ % 10 == 2) {
            targetText->setText("Target position: " + std::to_string(targetPosition), opts);
            measuredText->setText("Measured position: " + std::to_string(math::radToDeg(measuredPosition)), opts);
        }

        double gain = controller.regulate(math::degToRad(targetPosition), measuredPosition, dt);

        motor.update(gain, dt);

        renderer.clear();
        renderer.render(scene, camera);
        hud.apply(renderer);
        ui.render();

        target->rotation.z = math::degToRad(targetPosition);
        object->rotation.z = static_cast<float>(motor.getPosition());
    });
}
