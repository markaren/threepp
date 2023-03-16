// https://github.com/mrdoob/three.js/blob/r129/src/core/Object3D.js

#ifndef THREEPP_OBJECT3D_HPP
#define THREEPP_OBJECT3D_HPP

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include "threepp/core/EventDispatcher.hpp"
#include "threepp/core/Layers.hpp"

#include "misc.hpp"

#include <functional>
#include <memory>
#include <optional>

namespace threepp {

    class Material;
    class Raycaster;
    struct Intersection;

    class Scene;
    class BufferGeometry;

    typedef std::function<void(void*, Scene*, Camera*, BufferGeometry*, Material*, std::optional<GeometryGroup>)> RenderCallback;

    class Object3D: public EventDispatcher {

    public:
        inline static Vector3 defaultUp{0, 1, 0};
        inline static bool defaultMatrixAutoUpdate{true};

        unsigned int id{_object3Did++};

        const std::string uuid;

        std::string name;

        Object3D* parent = nullptr;
        std::vector<std::shared_ptr<Object3D>> children;

        Vector3 up{defaultUp};

        Vector3 position;
        Euler rotation;
        Quaternion quaternion;
        Vector3 scale{1, 1, 1};

        Matrix4 modelViewMatrix;
        Matrix3 normalMatrix;

        std::shared_ptr<Matrix4> matrix;
        std::shared_ptr<Matrix4> matrixWorld;

        bool matrixAutoUpdate = defaultMatrixAutoUpdate;
        bool matrixWorldNeedsUpdate = false;

        Layers layers;
        bool visible = true;

        bool castShadow = false;
        bool receiveShadow = false;

        bool frustumCulled = true;
        unsigned int renderOrder = 0;

        std::optional<RenderCallback> onBeforeRender;
        std::optional<RenderCallback> onAfterRender;

        Object3D();

        [[nodiscard]] virtual std::string type() const {

            return "Object3D";
        }

        void applyMatrix4(const Matrix4& matrix);

        Object3D& applyQuaternion(const Quaternion& q);

        void setRotationFromAxisAngle(const Vector3& axis, float angle);

        void setRotationFromEuler(const Euler& euler);

        void setRotationFromMatrix(const Matrix4& m);

        void setRotationFromQuaternion(const Quaternion& q);

        Object3D& rotateOnAxis(const Vector3& axis, float angle);

        Object3D& rotateOnWorldAxis(const Vector3& axis, float angle);

        Object3D& rotateX(float angle);

        Object3D& rotateY(float angle);

        Object3D& rotateZ(float angle);

        Object3D& translateOnAxis(const Vector3& axis, float distance);

        Object3D& translateX(float distance);

        Object3D& translateY(float distance);

        Object3D& translateZ(float distance);

        void localToWorld(Vector3& vector) const;

        void worldToLocal(Vector3& vector) const;

        void lookAt(const Vector3& vector);

        void lookAt(float x, float y, float z);

        Object3D& add(const std::shared_ptr<Object3D>& object);

        Object3D& remove(const std::shared_ptr<Object3D>& object);

        Object3D& remove(Object3D* object);

        Object3D& removeFromParent();

        Object3D& clear();

        Object3D* getObjectByName(const std::string& name);

        Vector3& getWorldPosition(Vector3& target);

        Quaternion& getWorldQuaternion(Quaternion& target);

        Vector3& getWorldScale(Vector3& target);

        virtual void getWorldDirection(Vector3& target);

        virtual void raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) {}

        void traverse(const std::function<void(Object3D&)>& callback);

        void traverseVisible(const std::function<void(Object3D&)>& callback);

        void traverseAncestors(const std::function<void(Object3D&)>& callback);

        template<class T>
        void traverseType(const std::function<void(T&)>& callback) {
            traverse([&](Object3D& o) {
                auto dyn = dynamic_cast<T*>(&o);
                if (dyn) {
                    callback(*dyn);
                }
            });
        }


        void updateMatrix();

        virtual void updateMatrixWorld(bool force = false);

        virtual void updateWorldMatrix(std::optional<bool> updateParents = std::nullopt, std::optional<bool> updateChildren = std::nullopt);

        static std::shared_ptr<Object3D> create() {

            return std::shared_ptr<Object3D>(new Object3D());
        }

        virtual BufferGeometry* geometry() {

            return nullptr;
        }

        virtual Material* material() {

            return nullptr;
        }

        virtual std::vector<Material*> materials() {

            return {};
        }

        [[nodiscard]] virtual const Material* material() const {

            return nullptr;
        }

        template<class T>
        T* as() {

            return dynamic_cast<T*>(this);
        }

        template<class T>
        bool is() {

            return dynamic_cast<T*>(this) != nullptr;
        }

        void copy(const Object3D& source, bool recursive = true);

        virtual std::shared_ptr<Object3D> clone(bool recursive = true);

        ~Object3D() override;

    private:
        inline static unsigned int _object3Did{0};
    };

}// namespace threepp

#endif// THREEPP_OBJECT3D_HPP
