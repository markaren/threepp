
#ifndef THREEPP_ACTUATOR_HPP
#define THREEPP_ACTUATOR_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"

class Actuator {

public:
    virtual void setGain(float gain) = 0;

    virtual float getProcessOutput() = 0;

    virtual void update() {}

    virtual ~Actuator() = default;
};

class Object3DActuator: public Actuator {

public:
    enum Axis {
        X,
        Y,
        Z
    };

    Object3DActuator(threepp::Object3D* obj, Axis axis, float maxSpeed, std::pair<float, float> limit)
        : obj_(obj),
          axis_(axis),
          gain_(0),
          maxSpeed_(maxSpeed),
          limit_(limit) {}

    void setGain(float gain) override {
        gain_ = gain;
    }

    void update() override {
        switch (axis_) {
            case X:
                obj_->rotation.x += gain_ * maxSpeed_;
                if (limit_) {
                    obj_->rotation.x.clamp(limit_->first, limit_->second);
                }
                break;
            case Y:
                obj_->rotation.y += gain_ * maxSpeed_;
                if (limit_) {
                    obj_->rotation.y.clamp(limit_->first, limit_->second);
                }
                break;
            case Z:
                obj_->rotation.z += gain_ * maxSpeed_;
                if (limit_) {
                    obj_->rotation.z.clamp(limit_->first, limit_->second);
                }
                break;
        }
    }

    float getProcessOutput() override {
        switch (axis_) {
            case X:
                return obj_->rotation.x;
            case Y:
                return obj_->rotation.y;
            case Z:
                return obj_->rotation.z;
            default:
                throw std::runtime_error("");
        }
    }

private:
    float gain_;
    float maxSpeed_;
    std::optional<std::pair<float, float>> limit_;

    Axis axis_;
    threepp::Object3D* obj_;
};


#endif//THREEPP_ACTUATOR_HPP
