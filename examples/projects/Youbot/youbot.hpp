
#ifndef THREEPP_YOUBOUT_HPP
#define THREEPP_YOUBOUT_HPP

#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/objects/Group.hpp"

using namespace threepp;

struct wasd {
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
};

struct Youbot {

    std::shared_ptr<Group> base;

    void setup(Canvas &canvas) {

        canvas.addKeyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
            if (evt.key == 87) {
                wasd_.up = true;
            } else if (evt.key == 83) {
                wasd_.down = true;
            } else if (evt.key == 68) {
                wasd_.right = true;
            } else if (evt.key == 65) {
                wasd_.left = true;
            }
        });

        canvas.addKeyAdapter(KeyAdapter::Mode::KEY_RELEASED, [&](KeyEvent evt) {
            if (evt.key == 87) {
                wasd_.up = false;
            } else if (evt.key == 83) {
                wasd_.down = false;
            } else if (evt.key == 68) {
                wasd_.right = false;
            } else if (evt.key == 65) {
                wasd_.left = false;
            }
        });
    }

    void driveForwards(float dt) {
        base->translateX(translationSpeed * dt);
        back_left_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
        back_right_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
        front_left_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
        front_right_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
    }

    void driveBackwards(float dt) {
        base->translateX(-translationSpeed * dt);
        back_left_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
        back_right_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
        front_left_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
        front_right_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
    }

    void driveRight(float dt) {
        base->rotateY(-rotationSpeed * dt);
        back_left_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
        back_right_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
        front_left_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
        front_right_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
    }

    void driveLeft(float dt) {
        base->rotateY(rotationSpeed * dt);
        back_left_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
        back_right_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
        front_left_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
        front_right_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
    }

    void update(float dt) {

        if (wasd_.up) {
            driveForwards(dt);
        }
        if (wasd_.down) {
            driveBackwards(dt);
        }
        if (wasd_.right) {
            driveRight(dt);
        }
        if (wasd_.left) {
            driveLeft(dt);
        }
    }

    static std::unique_ptr<Youbot> create(const std::filesystem::path &path) {
        AssimpLoader loader;
        auto model = loader.load(path);
        model->scale.multiplyScalar(10);

        return std::unique_ptr<Youbot>(new Youbot(model));
    }

private:
    wasd wasd_;
    float rotationSpeed = 2;
    float translationSpeed = 5;

    Object3D *front_left_wheel;
    Object3D *front_right_wheel;
    Object3D *back_left_wheel;
    Object3D *back_right_wheel;

    explicit Youbot(std::shared_ptr<Group> model) : base(std::move(model)) {

        back_left_wheel = base->getObjectByName("back-left_wheel");
        back_right_wheel = base->getObjectByName("back-right_wheel");
        front_left_wheel = base->getObjectByName("front-left_wheel_join");
        front_right_wheel = base->getObjectByName("front-right_wheel");
    }
};

#endif//THREEPP_YOUBOUT_HPP
