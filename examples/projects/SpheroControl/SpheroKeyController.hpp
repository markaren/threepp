
#ifndef SPHEROSIM_KEYCONTROLLER_HPP
#define SPHEROSIM_KEYCONTROLLER_HPP

#include "Sphero.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <cmath>

// Drives the robot from the keyboard, in one of two ways:
//   heading  W/S drive forwards and backwards, A/D turn the heading the robot
//            holds, Space boosts.
//   raw      a tank: W/S run the left track, Up/Down the right one.
//
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

    [[nodiscard]] bool raw() const {
        return raw_;
    }

    // Takes the wheel again, holding whatever heading the robot has now.
    void reset() {
        heading_ = sphero_->heading();
    }

    void update(float dt) {
        using threepp::Key;

        if (raw_) {
            const uint8_t left = input_->isKeyDown(Key::W) ? 0x01 : input_->isKeyDown(Key::S) ? 0x02
                                                                                              : 0x00;
            const uint8_t right = input_->isKeyDown(Key::UP) ? 0x01 : input_->isKeyDown(Key::DOWN) ? 0x02
                                                                                                  : 0x00;
            reset();
            sphero_->driveRaw(left, rawSpeed, right, rawSpeed);
            return;
        }

        if (input_->isKeyDown(Key::D)) heading_ += turnRate * dt;
        if (input_->isKeyDown(Key::A)) heading_ -= turnRate * dt;
        heading_ = std::fmod(heading_ + 360.f, 360.f);

        Sphero::Flags flags;
        uint8_t speed = 0;
        if (input_->isKeyDown(Key::W)) {
            speed = driveSpeed;
        } else if (input_->isKeyDown(Key::S)) {
            speed = driveSpeed;
            flags.setFlag(Sphero::Flags::DriveReverse);
        }
        if (input_->isKeyDown(Key::SPACE)) flags.setFlag(Sphero::Flags::Boost);

        sphero_->driveWithHeading(speed, static_cast<uint16_t>(heading_) % 360, flags);
    }

private:
    static constexpr uint8_t driveSpeed = 100;
    static constexpr uint8_t rawSpeed = 90;
    static constexpr float turnRate = 150.f;// [deg/s]

    Sphero* sphero_;
    threepp::PeripheralsEventSource* input_;
    bool raw_{false};
    float heading_{0};
};

#endif//SPHEROSIM_KEYCONTROLLER_HPP
