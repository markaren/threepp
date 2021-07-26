// https://github.com/mrdoob/three.js/blob/r129/src/core/Object3D.js

#ifndef THREEPP_OBJECT3D_HPP
#define THREEPP_OBJECT3D_HPP

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include "threepp/materials/Material.hpp"

#include "threepp/core/EventDispatcher.hpp"
#include "threepp/core/Layers.hpp"

#include <functional>
#include <memory>
#include <optional>

namespace threepp {

    class Raycaster;
    struct Intersection;

    class BufferGeometry;

    class Object3D : public EventDispatcher {

    public:
        inline static Vector3 defaultUp{0,1,0};

        const unsigned int id{_object3Did++};

        const std::string uuid{utils::generateUUID()};

        std::string name;

        Object3D *parent = nullptr;
        std::vector<std::shared_ptr<Object3D>> children;

        Vector3 up{defaultUp};

        Vector3 position;
        Euler rotation;
        Quaternion quaternion;
        Vector3 scale {1, 1, 1};

        Matrix4 modelViewMatrix;
        Matrix3 normalMatrix;

        Matrix4 matrix;
        Matrix4 matrixWorld;

        bool matrixAutoUpdate = true;
        bool matrixWorldNeedsUpdate = false;

        Layers layers;
        bool visible = true;

        bool castShadow = false;
        bool receiveShadow = false;

        bool frustumCulled = true;
        int renderOrder = 0;

        void applyMatrix4(const Matrix4 &matrix);

        Object3D &applyQuaternion(const Quaternion &q);

        void setRotationFromAxisAngle(const Vector3 &axis, float angle);

        void setRotationFromEuler(const Euler &euler);

        void setRotationFromMatrix(const Matrix4 &m);

        void setRotationFromQuaternion(const Quaternion &q);

        Object3D &rotateOnAxis(const Vector3 &axis, float angle);

        Object3D &rotateOnWorldAxis(const Vector3 &axis, float angle);

        Object3D &rotateX(float angle);

        Object3D &rotateY(float angle);

        Object3D &rotateZ(float angle);

        Object3D &translateOnAxis(const Vector3 &axis, float distance);

        Object3D &translateX(float distance);

        Object3D &translateY(float distance);

        Object3D &translateZ(float distance);

        void localToWorld(Vector3 &vector) const;

        void worldToLocal(Vector3 &vector) const;

        void lookAt(const Vector3 &vector);

        void lookAt(float x, float y, float z);

        Object3D &add(const std::shared_ptr<Object3D> &object);

        Object3D &remove(const std::shared_ptr<Object3D> &object);

        Object3D &remove(Object3D *object);

        Object3D &removeFromParent();

        Object3D &clear();

        Object3D *getObjectByName(const std::string &name);

        void getWorldPosition(Vector3 &target);

        void getWorldQuaternion(Quaternion &target);

        void getWorldScale(Vector3 &target);

        virtual void getWorldDirection(Vector3 &target);

        virtual void raycast(Raycaster &raycaster, std::vector<Intersection> &intersects) {}

        void traverse(const std::function<void(Object3D &)> &callback);

        void traverseVisible(const std::function<void(Object3D &)> &callback);

        void traverseAncestors(const std::function<void(Object3D &)> &callback) const;

        void updateMatrix();

        virtual void updateMatrixWorld(bool force = false);

        virtual void updateWorldMatrix(bool updateParents, bool updateChildren);

        static std::shared_ptr<Object3D> create() {

            return std::shared_ptr<Object3D>(new Object3D());
        }

        virtual BufferGeometry *geometry() {

            return nullptr;
        }

        virtual Material *material() {

            return nullptr;
        }

        virtual ~Object3D() = default;

    protected:
        Object3D();

    private:

        inline static unsigned int _object3Did{0};

        friend class Box3;
        friend class Frustum;
    };

}// namespace threepp

#endif// THREEPP_OBJECT3D_HPP
