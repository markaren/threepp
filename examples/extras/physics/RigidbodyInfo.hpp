

#ifndef THREEPP_RIGIDBODYINFO_HPP
#define THREEPP_RIGIDBODYINFO_HPP

#include <any>
#include <optional>
#include <string>

#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Object3D;

    struct PhysicsInfo {
        float gravity = -9.81f;
    };

    struct JointInfo {

        enum class Type {
            LOCK, HINGE, PRISMATIC, BALL
        };

        threepp::Vector3 anchor;
        threepp::Vector3 axis{0, 1, 0};
        threepp::Object3D* connectedBody = nullptr;
        std::optional<Vector2> limits;

        Type type = Type::LOCK;

        JointInfo& setConnectedBody(threepp::Object3D& body) {
            connectedBody = &body;
            return *this;
        }

        JointInfo& setAxis(Vector3 axis) {
            this->axis = axis;
            return *this;
        }

        JointInfo& setLimits(Vector2 limits) {
            this->limits = limits;
            return *this;
        }

        JointInfo& setAnchor(Vector3 anchor) {
            this->anchor = anchor;
            return *this;
        }

        JointInfo& setType(Type type) {
            this->type = type;
            return *this;
        }

    };

    struct RigidBodyInfo {

        enum class Type {
            STATIC,
            DYNAMIC
        };

        std::optional<float> mass;
        Type type = Type::DYNAMIC;

        std::optional<JointInfo> joint;

        JointInfo& addJoint() {
            joint = JointInfo();
            return *joint;
        }
    };

}// namespace threepp

#endif//THREEPP_RIGIDBODYINFO_HPP
