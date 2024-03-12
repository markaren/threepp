
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

namespace {

    struct MyMouseListener: MouseListener {

        float& t;

        explicit MyMouseListener(float& t): t(t) {}

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

    struct MyKeyListener: KeyListener {

        float& t;

        explicit MyKeyListener(float& t): t(t) {}

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

    Canvas canvas("Mouse and Key Listeners Demo");
    Clock clock;

    MyMouseListener ml{clock.elapsedTime};
    MyKeyListener kl{clock.elapsedTime};
    canvas.addMouseListener(ml);
    canvas.addKeyListener(kl);

    bool finish = false;
    canvas.animate([&]() {
        clock.getElapsedTime();

        if (clock.elapsedTime > 2 && clock.elapsedTime < 4) {
            if (canvas.removeKeyListener(kl)) {
                std::cout << "removed key listener" << std::endl;
            }
        } else if (!finish && clock.elapsedTime > 5) {
            std::cout << "re-added key listener" << std::endl;
            canvas.addKeyListener(kl);
            finish = true;
        }
    });
}
