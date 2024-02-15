
#include "threepp/threepp.hpp"

#include "InvertedPendulum.hpp"

#if HAS_IMGUI
#include "threepp/extras/imgui/ImguiContext.hpp"
#endif

using namespace threepp;

int main() {

    Canvas canvas("Inverted pendulum");
    GLRenderer renderer(canvas.size());

    Scene scene;

    float frustumSize = 5;
    OrthographicCamera camera(-frustumSize * canvas.aspect() / 2, frustumSize * canvas.aspect() / 2, frustumSize / 2, -frustumSize / 2, 0.1, 100);
    camera.position.z = 1;

    InvertedPendulum ip(50, 2);

    auto cart = Mesh::create(BoxGeometry::create(1, 0.2, 0.1), MeshBasicMaterial::create({{"color", 0x00ff00}}));
//    cart->position.y = -frustumSize / 2 + 0.25f;
    scene.add(cart);

    auto pendulum = Mesh::create(CylinderGeometry::create(0.05, 0.05, ip.getPendulumLength()), MeshBasicMaterial::create({{"color", 0xff0000}}));
    pendulum->geometry()->translate(0, ip.getPendulumLength() / 2 + 0.1f, 0);
    cart->add(pendulum);

    canvas.onWindowResize([&](WindowSize size) {
        camera.left = -frustumSize * size.aspect() / 2;
        camera.right = frustumSize * size.aspect() / 2;
        camera.top = frustumSize / 2;
        camera.bottom = -frustumSize / 2;
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    float externalForce = 0;
#if HAS_IMGUI
    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::Begin("Inverted Pendulum");
        ImGui::SliderFloat("ExternalForce", &externalForce, -10, 10);
        ImGui::End();
    });
#endif

    Clock clock;
    canvas.animate([&] {
        cart->position.x = static_cast<float>(ip.getCartPosition());
        pendulum->rotation.z = static_cast<float>(ip.getPendulumAngle());

        ip.simulate(clock.getDelta(), externalForce);

        renderer.render(scene, camera);

#if HAS_IMGUI
        ui.render();
#endif
    });
}
