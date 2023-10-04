
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

struct Youbot: Object3D, KeyListener {

    void onKeyPressed(KeyEvent evt) override {
        if (evt.key == Key::W) {
            wasd_.up = true;
        } else if (evt.key == Key::S) {
            wasd_.down = true;
        } else if (evt.key == Key::D) {
            wasd_.right = true;
        } else if (evt.key == Key::A) {
            wasd_.left = true;
        }
    }

    void onKeyReleased(KeyEvent evt) override {
        if (evt.key == Key::W) {
            wasd_.up = false;
        } else if (evt.key == Key::S) {
            wasd_.down = false;
        } else if (evt.key == Key::D) {
            wasd_.right = false;
        } else if (evt.key == Key::A) {
            wasd_.left = false;
        }
    }

    void driveForwards(float dt) {
        translateX(translationSpeed * dt);
        back_left_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
        back_right_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
        front_left_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
        front_right_wheel->rotateY(math::DEG2RAD * translationSpeed * 100 * dt);
    }

    void driveBackwards(float dt) {
        translateX(-translationSpeed * dt);
        back_left_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
        back_right_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
        front_left_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
        front_right_wheel->rotateY(-math::DEG2RAD * translationSpeed * 100 * dt);
    }

    void driveRight(float dt) {
        rotateY(-rotationSpeed * dt);
        back_left_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
        back_right_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
        front_left_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
        front_right_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
    }

    void driveLeft(float dt) {
        rotateY(rotationSpeed * dt);
        back_left_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
        back_right_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
        front_left_wheel->rotateY(-math::DEG2RAD * rotationSpeed * 200 * dt);
        front_right_wheel->rotateY(math::DEG2RAD * rotationSpeed * 200 * dt);
    }

    void setJointValues(const std::vector<float>& vals) {
        arm_joint1->rotation.z = math::DEG2RAD * vals[0];
        arm_joint2->rotation.z = math::DEG2RAD * (vals[1] - 90);
        arm_joint3->rotation.z = math::DEG2RAD * (vals[2] - 90);
        arm_joint4->rotation.z = math::DEG2RAD * vals[3];
        arm_joint5->rotation.z = math::DEG2RAD * vals[4];
    }

    std::vector<float> getJointValues() {
        return {
                arm_joint1->rotation.z() * math::RAD2DEG,
                arm_joint2->rotation.z() * math::RAD2DEG + 90,
                arm_joint3->rotation.z() * math::RAD2DEG + 90,
                arm_joint4->rotation.z() * math::RAD2DEG,
                arm_joint5->rotation.z() * math::RAD2DEG,
        };
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

    static std::unique_ptr<Youbot> create(const std::filesystem::path& path) {
        AssimpLoader loader;
        auto model = loader.load(path);
        model->scale.multiplyScalar(10);

        return std::unique_ptr<Youbot>(new Youbot(model));
    }

private:
    wasd wasd_;
    float rotationSpeed = 2;
    float translationSpeed = 5;

    Object3D* front_left_wheel;
    Object3D* front_right_wheel;
    Object3D* back_left_wheel;
    Object3D* back_right_wheel;

    Object3D* arm_joint1;
    Object3D* arm_joint2;
    Object3D* arm_joint3;
    Object3D* arm_joint4;
    Object3D* arm_joint5;

    explicit Youbot(const std::shared_ptr<Group>& model) {

        add(model);

        back_left_wheel = getObjectByName("back-left_wheel");
        back_right_wheel = getObjectByName("back-right_wheel");
        front_left_wheel = getObjectByName("front-left_wheel_join");
        front_right_wheel = getObjectByName("front-right_wheel");

        arm_joint1 = getObjectByName("arm_joint_1");
        arm_joint2 = getObjectByName("arm_joint_2");
        arm_joint3 = getObjectByName("arm_joint_3");
        arm_joint4 = getObjectByName("arm_joint_4");
        arm_joint5 = getObjectByName("arm_joint_5");
    }
};

#endif//THREEPP_YOUBOUT_HPP
