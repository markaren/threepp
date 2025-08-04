
#ifndef THREEPP_KEYCONTROLLER_HPP
#define THREEPP_KEYCONTROLLER_HPP

#include "threepp/input/KeyListener.hpp"

#include "Youbot.hpp"

class KeyController: public threepp::KeyListener {

public:
    KeyController(Youbot& youbot)
        : youbot_(&youbot) {}

    void onKeyPressed(KeyEvent evt) override {
        if (evt.key == Key::W) {
            keyState_.up = true;
        } else if (evt.key == Key::S) {
            keyState_.down = true;
        } else if (evt.key == Key::D) {
            keyState_.right = true;
        } else if (evt.key == Key::A) {
            keyState_.left = true;
        }
    }

    void onKeyReleased(KeyEvent evt) override {
        if (evt.key == Key::W) {
            keyState_.up = false;
        } else if (evt.key == Key::S) {
            keyState_.down = false;
        } else if (evt.key == Key::D) {
            keyState_.right = false;
        } else if (evt.key == Key::A) {
            keyState_.left = false;
        }
    }

    void update(float dt) {

        if (keyState_.up) {
            youbot_->driveForwards(dt);
        }
        if (keyState_.down) {
            youbot_->driveBackwards(dt);
        }
        if (keyState_.right) {
            youbot_->driveRight(dt);
        }
        if (keyState_.left) {
            youbot_->driveLeft(dt);
        }
    }

private:
    Youbot* youbot_;

    struct KeyState {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } keyState_;
};

#endif//THREEPP_KEYCONTROLLER_HPP
