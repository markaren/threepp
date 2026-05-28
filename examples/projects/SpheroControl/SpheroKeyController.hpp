
#ifndef SPHEROSIM_KEYCONTROLLER_HPP
#define SPHEROSIM_KEYCONTROLLER_HPP

#include "Sphero.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <cmath>

// Drive/steer keys are polled each frame via isKeyDown (no manual press/release
// state). The one edge action — R toggles raw mode — is an owning onKeyPressed
// callback, so this is no longer a KeyListener subclass.
class SpheroKeyController {

public:
    SpheroKeyController(Sphero& sphero, threepp::PeripheralsEventSource& input)
        : sphero_(&sphero), input_(&input) {
        input.onKeyPressed([this](threepp::KeyEvent evt) {
            if (evt.key == threepp::Key::R) raw_ = !raw_;
        });
    }

    void update() {
        using threepp::Key;

        if (raw_) {
            const uint8_t left = input_->isKeyDown(Key::W) ? 0x01 : input_->isKeyDown(Key::S) ? 0x02
                                                                                              : 0x00;
            const uint8_t right = input_->isKeyDown(Key::UP) ? 0x01 : input_->isKeyDown(Key::DOWN) ? 0x02
                                                                                                  : 0x00;
            heading_ = static_cast<uint16_t>(std::round(normalizeRotation(sphero_->rotation.y * threepp::math::RAD2DEG)));
            sphero_->driveRaw(left, 255, right, 255);
            return;
        }

        if (input_->isKeyDown(Key::A)) heading_ = (heading_ == 360) ? 1 : heading_ + 1;
        if (input_->isKeyDown(Key::D)) heading_ = (heading_ == 0) ? 359 : heading_ - 1;

        Sphero::Flags flags;
        uint8_t speed = 0;
        if (input_->isKeyDown(Key::W)) {
            speed = 255;
            flags.clearFlag(Sphero::Flags::DriveReverse);
        } else if (input_->isKeyDown(Key::S)) {
            speed = 255;
            flags.setFlag(Sphero::Flags::DriveReverse);
        }
        if (input_->isKeyDown(Key::SPACE)) flags.setFlag(Sphero::Flags::Boost);

        sphero_->driveWithHeading(speed, heading_, flags);
    }

private:
    Sphero* sphero_;
    threepp::PeripheralsEventSource* input_;
    bool raw_{false};
    uint16_t heading_{0};

    static float normalizeRotation(float rotation) {
        while (rotation < 0) {
            rotation += 360;
        }
        while (rotation >= 360) {
            rotation -= 360;
        }
        return rotation;
    }
};

#endif//SPHEROSIM_KEYCONTROLLER_HPP
