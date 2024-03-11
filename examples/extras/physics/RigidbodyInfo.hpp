

#ifndef THREEPP_RIGIDBODYINFO_HPP
#define THREEPP_RIGIDBODYINFO_HPP

#include <optional>
#include <string>

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

        Type type = Type::LOCK;

        virtual ~JointInfo() = default;
    };

    struct RigidBodyInfo {

        enum class Type {
            STATIC,
            DYNAMIC
        };

        std::optional<float> mass;
        Type type = Type::DYNAMIC;

        void* body = nullptr;

        std::optional<JointInfo> joint;
    };

}// namespace threepp

#endif//THREEPP_RIGIDBODYINFO_HPP
