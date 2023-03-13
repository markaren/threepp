
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

namespace {

    struct MyListener: MouseListener {

        float& t;

        explicit MyListener(float& t): t(t) {}

        void onMouseDown(int button, const Vector2& pos) override {
            std::cout << "onMouseDown, button= " << button << ", pos=" << pos << " at t=" << t << std::endl;
        }

        void onMouseUp(int button, const Vector2& pos) override {
            std::cout << "onMouseUp, button= " << button << ", pos=" << pos << " at t=" << t << std::endl;
        }

        void onMouseMove(const Vector2& pos) override {
            std::cout << "onMouseMove, "
                      << "pos=" << pos << " at t=" << t << std::endl;
        }

        void onMouseWheel(const Vector2& delta) override {
            std::cout << "onMouseWheel, "
                      << "delta=" << delta << " at t=" << t << std::endl;
        }
    };

}// namespace

int main() {

    Canvas canvas;

    float t = 0;
    MyListener l{t};
    canvas.addMouseListener(&l);

    canvas.animate([&](float dt) {
        t += dt;
    });
}
