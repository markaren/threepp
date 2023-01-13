
#ifndef THREEPP_ACTUATOR_HPP
#define THREEPP_ACTUATOR_HPP

#include "PID.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"

class Actuator {

public:
    virtual void setGain(float gain) = 0;

    virtual float getProcessOutput() = 0;

    virtual void update() {}

    virtual ~Actuator() = default;
};

class Object3DActuator : public Actuator {

public:
    enum Axis {
        X,
        Y,
        Z
    };

    Object3DActuator(threepp::Object3D *obj, Axis axis, float maxSpeed)
        : obj_(obj),
          axis_(axis),
          gain_(0),
          maxSpeed_(maxSpeed) {}

    void setGain(float gain) override {
        gain_ = gain;
    }

    void update() override {
        switch (axis_) {
            case X:
                obj_->rotateOnAxis(threepp::Vector3::X, gain_ * maxSpeed_);
                break;
            case Y:
                obj_->rotateOnAxis(threepp::Vector3::Y, gain_ * maxSpeed_);
                break;
            case Z:
                obj_->rotateOnAxis(threepp::Vector3::Z, gain_ * maxSpeed_);
                break;
        }
    }

    float getProcessOutput() override {
        switch (axis_) {
            case X:
                return obj_->rotation.x();
            case Y:
                return obj_->rotation.y();
            case Z:
                return obj_->rotation.z();
        }
    }

private:
    float gain_;
    float maxSpeed_;

    Axis axis_;
    threepp::Object3D *obj_;
};


#endif//THREEPP_ACTUATOR_HPP
