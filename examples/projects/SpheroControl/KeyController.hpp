
#ifndef SPHEROSIM_KEYCONTROLLER_HPP
#define SPHEROSIM_KEYCONTROLLER_HPP

#include "Sphero.hpp"
#include "threepp/input/KeyListener.hpp"

#include <cmath>

class KeyController : public threepp::KeyListener {

public:
    explicit KeyController(Sphero &sphero)
        : sphero_(&sphero) {}

    void update() {

        if (raw_) {

            keyState_.heading = static_cast<uint16_t>(std::round(normalizeRotation(sphero_->rotation.y * threepp::math::RAD2DEG)));
            sphero_->driveRaw(rawKeyState_.left, 255, rawKeyState_.right, 255);

        } else {

            if (rotating.first) {
                if (keyState_.heading == 360) {
                    keyState_.heading = 1;
                } else {
                    keyState_.heading += 1;
                }
            }

            if (rotating.second) {
                if (keyState_.heading == 0) {
                    keyState_.heading = 359;
                } else {
                    keyState_.heading -= 1;
                }
            }

            sphero_->driveWithHeading(keyState_.speed, keyState_.heading, keyState_.flags);
        }
    }

    void onKeyPressed(threepp::KeyEvent evt) override {

        if (evt.key == threepp::Key::R) {
            raw_ = !raw_;
        }

        if (raw_) {
            if (evt.key == threepp::Key::W) {
                rawKeyState_.left = 0x01;
            } else if (evt.key == threepp::Key::S) {
                rawKeyState_.left = 0x02;
            } else if (evt.key == threepp::Key::UP) {
                rawKeyState_.right = 0x1;
            } else if (evt.key == threepp::Key::DOWN) {
                rawKeyState_.right = 0x02;
            }
        } else {
            if (evt.key == threepp::Key::W) {
                keyState_.speed = 255;
                keyState_.flags.clearFlag(Sphero::Flags::DriveReverse);
            } else if (evt.key == threepp::Key::S) {
                keyState_.speed = 255;
                keyState_.flags.setFlag(Sphero::Flags::DriveReverse);
            } else if (evt.key == threepp::Key::A) {
                rotating.first = true;
            } else if (evt.key == threepp::Key::D) {
                rotating.second = true;
            } else if (evt.key == threepp::Key::SPACE) {
                keyState_.flags.setFlag(Sphero::Flags::Boost);
            }
        }
    }

    void onKeyReleased(threepp::KeyEvent evt) override {
        if (raw_) {
            if (evt.key == threepp::Key::W) {
                rawKeyState_.left = 0x0;
            } else if (evt.key == threepp::Key::S) {
                rawKeyState_.left = 0x0;
            } else if (evt.key == threepp::Key::UP) {
                rawKeyState_.right = 0x0;
            } else if (evt.key == threepp::Key::DOWN) {
                rawKeyState_.right = 0x0;
            }
        } else {
            if (evt.key == threepp::Key::W) {
                keyState_.speed = 0;
            } else if (evt.key == threepp::Key::S) {
                keyState_.speed = 0;
            } else if (evt.key == threepp::Key::A) {
                rotating.first = false;
            } else if (evt.key == threepp::Key::D) {
                rotating.second = false;
            } else if (evt.key == threepp::Key::SPACE) {
                keyState_.flags.clearFlag(Sphero::Flags::Boost);
            }
        }
    }

private:
    struct RawKeyState {
        uint8_t left = 0x0;
        uint8_t right = 0x0;
    } rawKeyState_;

    struct KeyState {
        uint16_t heading{0};
        uint8_t speed{0};
        Sphero::Flags flags;
    } keyState_;


    Sphero *sphero_;
    bool raw_{false};
    std::pair<bool, bool> rotating{false, false};

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
