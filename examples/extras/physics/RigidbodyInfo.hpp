

#ifndef THREEPP_RIGIDBODYINFO_HPP
#define THREEPP_RIGIDBODYINFO_HPP

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Object3D;

    struct JointInfo {

        enum class Type {
            LOCK,
            HINGE,
            PRISMATIC,
            BALL
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

    struct CapsuleCollider {
        float halfHeight;
        float radius;

        CapsuleCollider(float radius, float halfHeight)
            : radius(radius), halfHeight(halfHeight) {}
    };

    struct SphereCollider {
        float radius;

        explicit SphereCollider(float radius)
            : radius(radius) {}
    };

    struct BoxCollider {
        float halfWidth = 0.5;
        float halfHeight = 0.5;
        float halfDepth = 0.5;

        BoxCollider(float halfWidth, float halfHeight, float halfDepth)
            : halfWidth(halfWidth), halfHeight(halfHeight), halfDepth(halfDepth) {}
    };

    struct MaterialInfo {
        float friction;
        float restitution ;

        MaterialInfo(float friction, float restitution)
            : friction(friction), restitution(restitution) {}
    };

    using Collider = std::variant<SphereCollider, BoxCollider, CapsuleCollider>;

    struct RigidBodyInfo {

        enum class Type {
            STATIC,
            DYNAMIC
        };

        Type _type;
        std::optional<float> _mass;
        std::optional<JointInfo> _joint;
        std::vector<std::pair<Collider, Matrix4>> _colliders;
        std::optional<MaterialInfo> _material;
        bool _useVisualGeometryAsCollider{true};

        explicit RigidBodyInfo(Type type = Type::DYNAMIC): _type(type) {}

        RigidBodyInfo& setMass(float mass) {
            this->_mass = mass;
            return *this;
        }

        RigidBodyInfo& addCollider(Collider collider, Matrix4 offset = Matrix4()) {
            this->_colliders.emplace_back(collider, offset);
            return *this;
        }

        RigidBodyInfo& useVisualGeometryAsCollider(bool flag) {
            this->_useVisualGeometryAsCollider = flag;
            return *this;
        }

        RigidBodyInfo& setMaterialProperties(float friction, float restitution) {
            this->_material = MaterialInfo(friction, restitution);
            return *this;
        }

        JointInfo& addJoint() {
            this->_joint = JointInfo();
            return *_joint;
        }
    };

}// namespace threepp

#endif//THREEPP_RIGIDBODYINFO_HPP
