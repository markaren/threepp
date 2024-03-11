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

#include <any>
#include <functional>
#include <memory>
#include <optional>

namespace threepp {

    class Material;
    class Raycaster;
    struct Intersection;
    class Object3D;
    class BufferGeometry;

    typedef std::function<void(void*, Object3D*, Camera*, BufferGeometry*, Material*, std::optional<GeometryGroup>)> RenderCallback;

    // This is the base class for most objects in three.js and provides a set of properties and methods for manipulating objects in 3D space.
    //Note that this can be used for grouping objects via the .add( object ) method which adds the object as a child, however it is better to use Group for this.
    class Object3D: public EventDispatcher {

    public:
        inline static Vector3 defaultUp{0, 1, 0};
        inline static bool defaultMatrixAutoUpdate{true};

        // Unique number for this object instance.
        unsigned int id{_object3Did++};

        // UUID of this object instance. This gets automatically assigned, so this shouldn't be edited.
        const std::string uuid;

        // Optional name of the object (doesn't need to be unique). Default is an empty string.
        std::string name;

        // Non-owning pointer to Object's parent in the scene graph. An object can have at most one parent.
        Object3D* parent = nullptr;
        // Vector with object's children. See Group for info on manually grouping objects.
        std::vector<Object3D*> children;

        // This is used by the lookAt method, for example, to determine the orientation of the result.
        //Default is Object3D::defaultUp - that is, ( 0, 1, 0 ).
        Vector3 up{defaultUp};

        // A Vector3 representing the object's local position. Default is `(0, 0, 0)`.
        Vector3 position;
        // Object's local rotation (see Euler angles), in radians.
        Euler rotation;
        // Object's local rotation as a Quaternion.
        Quaternion quaternion;
        // The object's local scale. Default is Vector3( 1, 1, 1 ).
        Vector3 scale{1, 1, 1};

        // This is passed to the shader and used to calculate the position of the object.
        Matrix4 modelViewMatrix;
        // This is passed to the shader and used to calculate lighting for the object. It is the transpose of the inverse of the upper left 3x3 sub-matrix of this object's modelViewMatrix.
        //The reason for this special matrix is that simply using the modelViewMatrix could result in a non-unit length of normals (on scaling) or in a non-perpendicular direction (on non-uniform scaling)
        //On the other hand the translation part of the modelViewMatrix is not relevant for the calculation of normals. Thus a Matrix3 is sufficient.
        Matrix3 normalMatrix;

        // The local transform matrix.
        std::shared_ptr<Matrix4> matrix;
        // The global transform of the object. If the Object3D has no parent, then it's identical to the local transform .matrix.
        std::shared_ptr<Matrix4> matrixWorld;

        // When this is set, it calculates the matrix of position, (rotation or quaternion) and scale every frame and also recalculates the matrixWorld property.
        // Default is Object3D::defaultMatrixAutoUpdate (true).
        bool matrixAutoUpdate = defaultMatrixAutoUpdate;
        // When this is set, it calculates the matrixWorld in that frame and resets this property to false. Default is false.
        bool matrixWorldNeedsUpdate = false;

        // The layer membership of the object.
        // The object is only visible if it has at least one layer in common with the Camera in use.
        // This property can also be used to filter out unwanted objects in ray-intersection tests when using Raycaster.
        Layers layers;
        // Object gets rendered if true. Default is true.
        bool visible = true;

        // Whether the object gets rendered into shadow map. Default is false.
        bool castShadow = false;
        bool receiveShadow = false;

        // When this is set, it checks every frame if the object is in the frustum of the camera before rendering the object.
        // If set to false the object gets rendered every frame even if it is not in the frustum of the camera. Default is true.
        bool frustumCulled = true;
        // This value allows the default rendering order of scene graph objects to be overridden although opaque and transparent objects remain sorted independently.
        // When this property is set for an instance of Group, all descendants objects will be sorted and rendered together. Sorting is from lowest to highest renderOrder. Default value is 0.
        unsigned int renderOrder = 0;

        std::unordered_map<std::string, std::any> userData;

        std::optional<RenderCallback> onBeforeRender;
        std::optional<RenderCallback> onAfterRender;

        Object3D();

        Object3D(Object3D&& source) noexcept;
        Object3D& operator=(Object3D&&) = delete;
        Object3D(const Object3D&) = delete;
        Object3D& operator=(const Object3D&) = delete;

        [[nodiscard]] virtual std::string type() const;

        // Applies the matrix transform to the object and updates the object's position, rotation and scale.
        void applyMatrix4(const Matrix4& matrix);

        // Applies the rotation represented by the quaternion to the object.
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

        void localToWorld(Vector3& vector);

        // Converts the vector from world space to this object's local space.
        void worldToLocal(Vector3& vector);

        // Rotates the object to face a point in world space.
        void lookAt(const Vector3& vector);

        // Rotates the object to face a point in world space.
        void lookAt(float x, float y, float z);

        // Adds object as child of this object. An arbitrary number of objects may be added.
        // Any current parent on an object passed in here will be removed, since an object can have at most one parent.
        // This version of add takes ownership of the passed in object
        void add(const std::shared_ptr<Object3D>& object);

        // Adds object as child of this object. An arbitrary number of objects may be added.
        // Any current parent on an object passed in here will be removed, since an object can have at most one parent.
        // This version of add does NOT take ownership of the passed in object
        virtual void add(Object3D& object);

        // Removes object as child of this object.
        virtual void remove(Object3D& object);

        // Removes this object from its current parent.
        void removeFromParent();

        // Removes all child objects.
        void clear();

        // Searches through an object and its children, starting with the object itself, and returns the first with a matching name.
        // Note that for most objects the name is an empty string by default. You will have to set it manually to make use of this method.
        Object3D* getObjectByName(const std::string& name);

        // Returns a vector representing the position of the object in world space.
        void getWorldPosition(Vector3& target);

        // Returns a quaternion representing the rotation of the object in world space.
        void getWorldQuaternion(Quaternion& target);

        // Returns a vector of the scaling factors applied to the object for each axis in world space.
        void getWorldScale(Vector3& target);

        // Returns a vector representing the direction of object's positive z-axis in world space.
        virtual void getWorldDirection(Vector3& target);

        virtual void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) {}

        void traverse(const std::function<void(Object3D&)>& callback);

        void traverseVisible(const std::function<void(Object3D&)>& callback);

        void traverseAncestors(const std::function<void(Object3D&)>& callback);

        template<class T>
        void traverseType(const std::function<void(T&)>& callback) {
            traverse([&](Object3D& o) {
                if (auto dyn = dynamic_cast<T*>(&o)) {
                    callback(*dyn);
                }
            });
        }

        // Updates the local transform.
        void updateMatrix();

        virtual void updateMatrixWorld(bool force = false);

        virtual void updateWorldMatrix(std::optional<bool> updateParents = std::nullopt, std::optional<bool> updateChildren = std::nullopt);

        static std::shared_ptr<Object3D> create() {

            return std::make_shared<Object3D>();
        }

        virtual BufferGeometry* geometry() {

            return nullptr;
        }

        virtual Material* material() {

            return nullptr;
        }

        template<class T>
        T* as() {

            static_assert(std::is_base_of<Object3D, typename std::remove_cv<typename std::remove_pointer<T>::type>::type>::value,
                          "T must be a base class of Object3D");

            return dynamic_cast<T*>(this);
        }

        template<class T>
        [[nodiscard]] bool is() const {

            return dynamic_cast<const T*>(this) != nullptr;
        }

        void copy(const Object3D& source, bool recursive = true);

        virtual std::shared_ptr<Object3D> clone(bool recursive = true);

        ~Object3D() override;

    private:
        inline static unsigned int _object3Did{0};

        std::vector<std::shared_ptr<Object3D>> children_;
    };

}// namespace threepp

#endif// THREEPP_OBJECT3D_HPP
