
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

namespace {

	void onMouseDown(MouseButtonEvent & e, double t) {
		std::cout << "onMouseDown, button= " << e.button << ", pos=" << e.pos << " at t=" << t << std::endl;
	}

	void onMouseUp(MouseButtonEvent & e, double t) {
		std::cout << "onMouseUp, button= " << e.button << ", pos=" << e.pos << " at t=" << t << std::endl;
	}

	void onMouseMove(MouseMoveEvent &e, double t) {
		std::cout << "onMouseMove, "
				  << "pos=" << e.pos << " at t=" << t << std::endl;
	}

	void onMouseWheel(MouseWheelEvent &e, double t) {
		std::cout << "onMouseWheel, "
				  << "delta=" << e.offset << " at t=" << t << std::endl;
	}

	void onKeyPressed(KeyEvent evt, double t) {
		std::cout << "onKeyPressed at t=" << t << std::endl;
	}

	void onKeyReleased(KeyEvent evt, double t) {
		std::cout << "onKeyReleased at t=" << t << std::endl;
	}

	void onKeyRepeat(KeyEvent evt, double t) {
		std::cout << "onKeyRepeat at t=" << t << std::endl;
	}

}// namespace

int main() {

    Canvas canvas("Mouse and Key Listeners Demo");
    Clock clock;

	Subscriptions subs_;
    subs_ << canvas.mouse.OnMouseDown.subscribe([&](auto& e) { onMouseDown(e, clock.elapsedTime); });
    subs_ << canvas.mouse.OnMouseUp.subscribe([&](auto& e) { onMouseUp(e, clock.elapsedTime); });
    subs_ << canvas.mouse.OnMouseMove.subscribe([&](auto& e) { onMouseMove(e, clock.elapsedTime); });
    subs_ << canvas.mouse.OnMouseWheel.subscribe([&](auto& e) { onMouseWheel(e, clock.elapsedTime); });

	Subscriptions key_subs_;
	auto subscribe_keys = [&]() {
		key_subs_ << canvas.keys.OnKeyPressed.subscribe([&](auto& e) {onKeyPressed(e, clock.elapsedTime); });
		key_subs_ << canvas.keys.OnKeyReleased.subscribe([&](auto& e) {onKeyReleased(e, clock.elapsedTime); });
		key_subs_ << canvas.keys.OnKeyRepeat.subscribe([&](auto& e) {onKeyRepeat(e, clock.elapsedTime); });
	};

    bool finish = false;
    canvas.animate([&]() {
        clock.getElapsedTime();

        if (clock.elapsedTime > 2 && clock.elapsedTime < 4) {
			key_subs_.clear();
			std::cout << "removed key listener" << std::endl;
        } else if (!finish && clock.elapsedTime > 5) {
			subscribe_keys();
            std::cout << "re-added key listener" << std::endl;
            finish = true;
        }
    });
}
