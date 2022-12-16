
#include "threepp/threepp.hpp"
#include "imgui_helper.hpp"

using namespace threepp;


struct MyGui: public imggui_helper {

    explicit MyGui(Canvas &canvas);

    void onRender() override;

    float planePosX{};
    float planePosY{};
    float planePosZ{};
    float planeRotX{};
    float planeRotY{};
    float planeRotZ{};

};

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    boxMaterial->color.setRGB(1,0,0);
    boxMaterial->transparent = true;
    boxMaterial->opacity = 0.1f;
    auto box = Mesh::create(boxGeometry, boxMaterial);
    scene->add(box);

    auto wiredBox = Mesh::create(boxGeometry, boxMaterial->clone());
    wiredBox->material<MeshBasicMaterial>()->wireframe = true;
    wiredBox->material<MeshBasicMaterial>()->color = Color::white;
    box->add(wiredBox);

    const auto sphereGeometry = SphereGeometry::create(0.5f);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x00ff00);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(-1);
    box->add(sphere);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setHex(Color::yellow);
    planeMaterial->transparent = true;
    planeMaterial->opacity = 0.5f;
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setZ(-2);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size){
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    MyGui ui(canvas);
    canvas.animate([&](float dt) {
        box->rotation.y +=  0.5f * dt;

        plane->position.set(ui.planePosX, ui.planePosY, ui.planePosZ);
        plane->rotation.set(math::DEG2RAD * ui.planeRotX, math::DEG2RAD * ui.planeRotY, math::DEG2RAD * ui.planeRotZ);

        renderer.render(scene, camera);
        ui.render();

    });
}

MyGui::MyGui(Canvas &canvas) : imggui_helper(canvas) {}

void MyGui::onRender() {

    ImGui::SetNextWindowPos({0, 0}, 0, {0,0});
    ImGui::SetNextWindowSize({200, 100}, 0);
    ImGui::Begin("Plane position settings");
    ImGui::SliderFloat("x", &planePosX, -5.0f, 5.0f);
    ImGui::SliderFloat("y", &planePosY, -5.0f, 5.0f);
    ImGui::SliderFloat("z", &planePosZ, -5.0f, 5.0f);
    ImGui::End();

    ImGui::SetNextWindowPos({200, 0}, 0, {0,0});
    ImGui::SetNextWindowSize({200, 0}, 0);
    ImGui::Begin("Plane rotation settings");
    ImGui::SliderFloat("x", &planeRotX, 0.0f, 360.0f);
    ImGui::SliderFloat("y", &planeRotY, 0.0f, 360.0f);
    ImGui::SliderFloat("z", &planeRotZ, 0.0f, 360.0f);
    ImGui::End();

}
