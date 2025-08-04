#include "threepp/threepp.hpp"

#include <cmath>
#include <algorithm>

using namespace threepp;

struct MyMouseListener: MouseListener {

    MyMouseListener(bool& d, int& slider, Canvas& c, OrbitControls& oc)
        : dragging(d), sliderPos(slider), canvas(c), controls(oc) {}

    void onMouseDown(int button, const Vector2& pos) override {
        if (button == 0) {

            // Check to see if we're within a short tolerance to the
            // slider. Let's say 10 pixels for now.
            constexpr float kSliderTol = 10.f;
            if (std::abs(pos.x - static_cast<float>(sliderPos)) < kSliderTol) {
                dragging = true;
                controls.enabled = false;
            }
        }
    }

    void onMouseUp(int button, const Vector2& pos) override {
        if (button == 0) {
            dragging = false;
            controls.enabled = true;
        }
    }

    void onMouseMove(const Vector2& pos) override {
        if (dragging) {
            sliderPos = std::clamp(static_cast<int>(pos.x), 0, canvas.size().width());
        }
    }

private:
    bool& dragging;
    int& sliderPos;
    Canvas& canvas;
    OrbitControls& controls;
};

int main() {

    Canvas canvas("Multiple Scenes", {{"aa", 4}});

    Scene sceneLeft;
    sceneLeft.background = Color(0xBCD48F);

    Scene sceneRight;
    sceneRight.background = Color(0x8FBCD4);

    auto geometry = IcosahedronGeometry::create(1, 3);

    auto materialLeft = MeshStandardMaterial::create({{"color", Color::lightgrey}});
    auto materialRight = MeshStandardMaterial::create({{"color", Color::gray}});
    materialRight->wireframe = true;

    auto meshLeft = Mesh::create(geometry, materialLeft);
    sceneLeft.add(meshLeft);

    auto meshRight = Mesh::create(geometry, materialRight);
    sceneRight.add(meshRight);

    auto camera = PerspectiveCamera::create(35, canvas.aspect(), 0.1f, 100);
    camera->position.z = 6;

    auto light = HemisphereLight::create(0xffffff, 0x444444);
    light->position.set(-2, 2, 2);

    sceneLeft.add(light);
    sceneRight.add(light->clone());// need to clone second light for some reason

    OrbitControls controls(*camera, canvas);

    GLRenderer renderer(canvas.size());
    renderer.setScissorTest(true);
    renderer.setClearAlpha(0);
    renderer.setClearColor(0);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    int sliderPos = canvas.size().width() / 2;
    bool dragging = false;

    MyMouseListener listener{dragging, sliderPos, canvas, controls};
    canvas.addMouseListener(listener);

    canvas.animate([&]() {
        const auto size = canvas.size();

        renderer.setScissor(0, 0, sliderPos, size.height());
        renderer.render(sceneLeft, *camera);

        renderer.setScissor(sliderPos, 0, size.width() - sliderPos, size.height());
        renderer.render(sceneRight, *camera);
    });
}
