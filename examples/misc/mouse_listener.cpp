
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    struct MyListener: MouseListener {

        float &t;

        explicit MyListener(float &t) : t(t) {}

        void onMouseDown(int button, Vector2 pos) override {
            std::cout << "onMouseDown, button= " << button << ", pos=" << pos << " at t=" << t << std::endl;
        }

        void onMouseUp(int button, Vector2 pos) override {
            std::cout << "onMouseUp, button= " << button << ", pos=" << pos << " at t=" << t << std::endl;
        }

        void onMouseMove(Vector2 pos) override {
            std::cout << "onMouseMove, " << "pos=" << pos << " at t=" << t << std::endl;
        }

        void onMouseWheel(Vector2 delta) override {
            std::cout << "onMouseWheel, " << "delta=" << delta << " at t=" << t << std::endl;
        }
    };

}

int main() {

    Canvas canvas;

    float t = 0;
    auto l = std::make_shared<MyListener>(t);
    canvas.addMouseListener(l);

    canvas.animate([&](float dt) {
        t += dt;
    });
}
