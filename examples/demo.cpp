
#include "threepp/threepp.hpp"
#include "imgui_helper.hpp"

using namespace threepp;


struct MyGui: public imggui_helper {

    explicit MyGui(Canvas &canvas);

    void onRender() override;

    float planePosX{};
    float planePosY{};
    float planePosZ{};
    float planeRotY{};

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
        plane->rotation.y = math::DEG2RAD * ui.planeRotY;

        renderer.render(scene, camera);
        ui.render();


    });
}

MyGui::MyGui(Canvas &canvas) : imggui_helper(canvas) {}

void MyGui::onRender() {

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver, ImVec2(0,0));
    ImGui::Begin("Plane settings");
    ImGui::SliderFloat("pos x", &planePosX, 0.0f, 1.0f);
    ImGui::SliderFloat("pos y", &planePosY, 0.0f, 1.0f);
    ImGui::SliderFloat("pos z", &planePosZ, 0.0f, 1.0f);
    ImGui::SliderFloat("rotation y", &planeRotY, 0.0f, 360.0f);
    ImGui::End();

}


