
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

namespace {

    struct MyListener: KeyListener {

        float& t;

        explicit MyListener(float& t): t(t) {}

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

}// namespace

int main() {

    Canvas canvas;
    Clock clock;

    MyListener l{clock.elapsedTime};
    canvas.addKeyListener(&l);

    bool finish = false;
    canvas.animate([&]() {

        clock.getElapsedTime();

        if (clock.elapsedTime > 2 && clock.elapsedTime < 4) {
            if (canvas.removeKeyListener(&l)) {
                std::cout << "removed listener" << std::endl;
            }
        } else if (!finish && clock.elapsedTime > 5) {
            std::cout << "re-added listener" << std::endl;
            canvas.addKeyListener(&l);
            finish = true;
        }
    });
}
