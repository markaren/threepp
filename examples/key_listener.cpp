
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    struct MyListener: KeyListener {

        float &t;

        explicit MyListener(float &t) : t(t) {}

        void onKeyPressed(KeyEvent evt) override {
            std::cout << "onKeyPressed at t=" << t << std::endl;
        }

        void onKeyReleased(KeyEvent evt) override {
            std::cout << "onKeyReleased at t=" << t << std::endl;
        }

        void onKeyRepeat(KeyEvent evt) override {
            std::cout << "onKeyRepeat at t=" << t << std::endl;
        }
    };

}

int main() {

    Canvas canvas;

    float t = 0;
    auto l = std::make_shared<MyListener>(t);
    canvas.addKeyListener(l);

    bool finish = false;
    canvas.animate([&](float dt) {
        t += dt;

        if (t > 2 && t < 4) {
            if (canvas.removeKeyListener(l)) {
                std::cout << "removed listener" << std::endl;
            }
        } else if (!finish && t > 5) {
            std::cout << "re-added listener" << std::endl;
            canvas.addKeyListener(l);
            finish = true;
        }

    });
}
