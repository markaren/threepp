
#include "threepp/threepp.hpp"

#include <glad/glad.h>

using namespace threepp;

#ifdef HAS_IMGUI
#include "threepp/extras/imgui/imgui_context.hpp"

struct MyGui: public imgui_context {

    bool colorChanged = false;

    explicit MyGui(const Canvas &canvas) : imgui_context(canvas.window_ptr()) {}

    void onRender() override {

        ImGui::SetNextWindowPos({0, 0}, 0, {0,0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Plane transform");
        ImGui::SliderFloat3("position", posBuf_.data(), -5.f, 5.f);
        ImGui::SliderFloat3("rotation", eulerBuf_.data(), -180.f, 180.f);
        ImGui::ColorEdit4("Color", colorBuf_.data());
        colorChanged = ImGui::IsItemEdited();

        ImGui::End();

    }

    const Vector3& position() {
        pos_.fromArray(posBuf_);
        return pos_;
    }

    const Euler& rotation() {
        euler_.set(math::DEG2RAD * eulerBuf_[0], math::DEG2RAD * eulerBuf_[1], math::DEG2RAD * eulerBuf_[2]);
        return euler_;
    }

    const std::array<float, 4>& color() {
        return colorBuf_;
    }

private:

    Vector3 pos_;
    Euler euler_;

    std::array<float, 3> posBuf_{};
    std::array<float, 3> eulerBuf_{};
    std::array<float, 4> colorBuf_{0,0,0,1};

};
#endif

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

    renderer.enableTextRendering();
    auto& handle = renderer.textHandle();
    handle.setPosition(canvas.getSize().width-130, 0);
    handle.color = Color::red;

#ifdef HAS_IMGUI
    MyGui ui(canvas);
#endif
    canvas.animate([&](float dt) {
        box->rotation.y +=  0.5f * dt;
        handle.setText("Delta=" + std::to_string(dt));

        renderer.render(scene, camera);

#ifdef HAS_IMGUI
        ui.render();

        plane->position.copy(ui.position());
        plane->rotation.copy(ui.rotation());

        if (ui.colorChanged) {
            auto& c = ui.color();
            planeMaterial->color.fromArray(c);
            planeMaterial->opacity = c[3];
            planeMaterial->transparent = c[3] != 1;
        }

#endif

    });

}
