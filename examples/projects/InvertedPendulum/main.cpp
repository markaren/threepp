
#include "threepp/threepp.hpp"
#include "utility/Regulator.hpp"

#include "InvertedPendulum.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Inverted pendulum");
    GLRenderer renderer(canvas.size());

    Scene scene;

    float frustumSize = 5;
    OrthographicCamera camera(-(frustumSize * canvas.aspect()) / 2, (frustumSize * canvas.aspect()) / 2, frustumSize / 2, -frustumSize / 2, 0.1, 100);
    camera.position.z = 1;

    InvertedPendulum ip;
    float externalForce = 0;
    bool applyControl = true;

    PIDRegulator regulator(50, 2, 10);
    regulator.setWindupGuard(2.f);

    auto cartGeometry = BoxGeometry::create(1, 0.2, 0.1);
    auto cart = Mesh::create(cartGeometry, MeshBasicMaterial::create({{"color", 0x00ff00}}));
    scene.add(cart);

    auto pendulum = Mesh::create(CylinderGeometry::create(0.05, 0.05, ip.getPendulumLength()), MeshBasicMaterial::create({{"color", 0xff0000}}));
    pendulum->geometry()->translate(0, ip.getPendulumLength() / 2 + cartGeometry->height / 2, 0);
    cart->add(pendulum);

    canvas.onWindowResize([&](WindowSize size) {
        camera.left = -frustumSize * size.aspect() / 2;
        camera.right = frustumSize * size.aspect() / 2;
        camera.top = frustumSize / 2;
        camera.bottom = -frustumSize / 2;
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    ImguiFunctionalContext ui(canvas, [&] {
        externalForce = 0;

        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("Inverted Pendulum");
        ImGui::Checkbox("Apply control", &applyControl);
        if (ImGui::Button("Push left")) {
            externalForce = -500;
        } else if (ImGui::Button("Push right")) {
            externalForce = 500;
        }
        ImGui::End();
    });

    Clock clock;
    canvas.animate([&] {
        cart->position.x = static_cast<float>(ip.getCartPosition());
        pendulum->rotation.z = static_cast<float>(ip.getPendulumAngle());

        ip.simulate(clock.getDelta(), externalForce, applyControl ? &regulator : nullptr);

        renderer.render(scene, camera);

        ui.render();
    });
}
