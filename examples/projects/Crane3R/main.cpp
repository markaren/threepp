
#include "threepp/threepp.hpp"

#include <thread>

using namespace threepp;

#ifdef HAS_IMGUI
#include "../../imgui_helper.hpp"

struct MyUI : public imggui_helper {

    bool mouseHover = false;
    std::array<float, 3> angles;

    explicit MyUI(const threepp::Canvas &canvas) : imggui_helper(canvas) {}

    void onRender() override {

        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({}, 0);
        ImGui::Begin("Crane3R");

        ImGui::SliderFloat("j1", &angles[0], -90, 90);
        ImGui::SliderFloat("j2", &angles[1], -90, 0);
        ImGui::SliderFloat("j3", &angles[2], 0, 120);

        mouseHover = ImGui::IsWindowHovered();
        ImGui::End();
    }
};
#endif

class Crane3R : public Group {

public:

    void setAngles(float j1, float j2, float j3) {
        p1_->rotation.z = math::DEG2RAD * j1;
        p2_->rotation.y = math::DEG2RAD * j2;
        p3_->rotation.y = math::DEG2RAD * j3;
    }

    static std::shared_ptr<Crane3R> create() {

        OBJLoader loader;
        auto part1 = loader.load("data/models/obj/Crane3R/4200/4200.obj");
        auto part2 = loader.load("data/models/obj/Crane3R/7000/7000.obj");
        part2->position.set(0, 0, 4.2);
        part1->add(part2);
        auto part3 = loader.load("data/models/obj/Crane3R/5200/5200.obj");
        part3->position.set(7, 0, 0);
        part2->add(part3);

        part1->rotateX(-math::PI / 2);

        part1->name = "part1";
        part2->name = "part2";
        part3->name = "part3";

        return std::shared_ptr<Crane3R>(new Crane3R(part1));
    }

private:
    Object3D* p1_;
    Object3D* p2_;
    Object3D* p3_;

    explicit Crane3R(const std::shared_ptr<Group> &obj) {

        p1_ = obj->getObjectByName("part1");
        p2_ = obj->getObjectByName("part2");
        p3_ = obj->getObjectByName("part3");

        add(obj);
    }
};

int main() {

    Canvas canvas{Canvas::Parameters().size({1280, 720}).antialiasing(8)};
    GLRenderer renderer{canvas};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create(60, canvas.getAspect());
    camera->position.set(15, 8, 15);

    OrbitControls controls(camera, canvas);

    auto scene = Scene::create();

    auto grid = GridHelper::create(20, 10, Color::yellowgreen);
    scene->add(grid);

    auto light = AmbientLight::create(Color::white);
    scene->add(light);

    std::shared_ptr<Crane3R> crane;
    std::thread t([&] {
        crane = Crane3R::create();
        canvas.invokeLater([&, crane] {
            scene->add(crane);
        });
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

#ifdef HAS_IMGUI
    MyUI ui(canvas);
#endif
    canvas.animate([&] {
        renderer.render(scene, camera);

#ifdef HAS_IMGUI

        if (crane) {
            ui.render();

            auto &angles = ui.angles;
            crane->setAngles(angles[0], angles[1], angles[2]);
        }

        controls.enabled = !ui.mouseHover;

#endif
    });

    t.join();
}
