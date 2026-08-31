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
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>

namespace threepp {

    class Material;
    class Raycaster;
    struct Intersection;
    class Object3D;
    class BufferGeometry;
    class AnimationClip;

    typedef std::function<void(void*, Object3D*, Camera*, BufferGeometry*, Material*, std::optional<GeometryGroup>)> RenderCallback;

    // This is the base class for most objects in three.js and provides a set of properties and methods for manipulating objects in 3D space.
    //Note that this can be used for grouping objects via the .add( object ) method which adds the object as a child, however it is better to use Group for this.
    //
    // enable_shared_from_this is for the Python bindings: pybind11 uses it to
    // adopt the existing control block whenever a raw Object3D* crosses into
    // Python (children/parent/traverse/...), so the wrapper shares ownership.
    // Without it those wrappers are non-owning, and one that outlives its
    // object segfaults on destruction: Mesh/Points/Line inherit Object3D
    // virtually, so pybind11's instance deregistration must read the (dead)
    // object's vtable to locate the base. Objects on the stack are fine as
    // long as shared_from_this() is never called on them.
    class Object3D: public EventDispatcher, public std::enable_shared_from_this<Object3D> {

    public:
        inline static Vector3 defaultUp{0, 1, 0};
        inline static bool defaultMatrixAutoUpdate{true};

        // Unique number for this object instance.
        // Atomic: loaders run on a worker thread (loadAsync detaches one), so
        // objects are constructed concurrently with the main thread's. A torn
        // read-modify-write here hands two objects the SAME id, and ids are used
        // as identity downstream — GLRenderer skips re-uploading a material's
        // uniforms when the id matches the last bound one, so a collision renders
        // one material with another's parameters.
        unsigned int id{_object3Did.fetch_add(1, std::memory_order_relaxed)};

        // UUID of this object instance. Automatically assigned; only serialization
        // round-trips (ObjectLoader) have a reason to overwrite it.
        std::string uuid;

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
        //
        // The Matrix4 lives by VALUE inside this object (matrixLocal_ below);
        // this handle is a non-owning alias to it — no heap allocation, no
        // refcount. The shared_ptr type is kept so the three.js-style aliasing
        // stays source-compatible: helpers re-point it at another object's
        // matrixWorld (`helper.matrix = light.matrixWorld`). An alias does NOT
        // keep its target alive — the target must outlive the aliasing object
        // (every in-tree helper already holds its target by reference/pointer
        // and assumed exactly that).
        std::shared_ptr<Matrix4> matrix{std::shared_ptr<Matrix4>{}, &matrixLocal_};
        // The global transform of the object. If the Object3D has no parent, then it's identical to the local transform .matrix.
        // Stored by value like `matrix` (see matrixWorldLocal_); same aliasing caveat.
        std::shared_ptr<Matrix4> matrixWorld{std::shared_ptr<Matrix4>{}, &matrixWorldLocal_};

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
        // Opt-out for the renderer's automatic mesh LOD (Vulkan setAutoLod).
        // Set false on meshes that manage their own level of detail (e.g.
        // TileTerrain quadtree tiles) — stacking automatic simplification on
        // top of system-managed LOD flattens shading against neighbours at
        // other levels for little performance return. Default is true
        // (checked on the mesh itself, not inherited).
        bool autoLod = true;
        // This value allows the default rendering order of scene graph objects to be overridden although opaque and transparent objects remain sorted independently.
        // When this property is set for an instance of Group, all descendants objects will be sorted and rendered together. Sorting is from lowest to highest renderOrder. Default value is 0.
        // Signed, as in three.js: a negative value pushes an object behind the
        // default-ordered ones (the usual way to pin a skybox or backdrop).
        int renderOrder = 0;

        std::unordered_map<std::string, std::any> userData;

        std::vector<std::shared_ptr<AnimationClip>> animations;

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

        // Which way lookAt() faces this object: false (the default) turns
        // local +Z toward the target; true turns local -Z toward it, the way
        // a camera looks. Cameras and Lights are detected by type inside
        // lookAt(); a node that is neither but still images along its local
        // -Z (the pinhole sensors) overrides this so lookAt() aims its beams
        // rather than its back.
        [[nodiscard]] virtual bool usesCameraLookAtConvention() const { return false; }

        // Adds object as child of this object. An arbitrary number of objects may be added.
        // Any current parent on an object passed in here will be removed, since an object can have at most one parent.
        // This version of add takes ownership of the passed in object
        void add(const std::shared_ptr<Object3D>& object);

        // Adds object as child of this object. An arbitrary number of objects may be added.
        // Any current parent on an object passed in here will be removed, since an object can have at most one parent.
        // This version of add does NOT take ownership of the passed in object
        virtual void addRef(Object3D& object);

        // Removes object as child of this object.
        virtual void remove(Object3D& object);

        // Removes this object from its current parent.
        //
        // Returns the owning reference if the parent held one, so a caller can
        // keep the object alive across the call:
        //     auto kept = obj->removeFromParent();   // obj stays valid
        // Discarding the result on an object the parent owned destroys it — the
        // return value is the only thing keeping it alive. Returns nullptr when
        // the object was attached with addRef() (parent never owned it) or had
        // no parent.
        std::shared_ptr<Object3D> removeFromParent();

        // Removes all child objects.
        void clear();

        // Searches through an object and its children, starting with the object itself, and returns the first with a matching name.
        // Note that for most objects the name is an empty string by default. You will have to set it manually to make use of this method.
        //
        // When T is given, the search is for the first node matching BOTH the name
        // and the type. It used to resolve the first node matching the name only
        // and then cast it, so getObjectByName<Mesh>("wheel") returned nullptr
        // whenever any non-Mesh node named "wheel" (a Group, a Bone) came first in
        // traversal order — a silent miss on a name that really was present.
        template<class T = Object3D>
        T* getObjectByName(const std::string& name) {

            if (this->name == name) {
                if (auto* self = dynamic_cast<T*>(this)) return self;
            }

            for (const auto& child : this->children) {

                if (auto* found = child->getObjectByName<T>(name)) return found;
            }

            return nullptr;
        }

        // Returns a vector representing the position of the object in world space.
        void getWorldPosition(Vector3& target);

        // Returns a quaternion representing the rotation of the object in world space.
        void getWorldQuaternion(Quaternion& target);

        // Returns a vector of the scaling factors applied to the object for each axis in world space.
        void getWorldScale(Vector3& target);

        // Returns a vector representing the direction of object's positive z-axis in world space.
        virtual void getWorldDirection(Vector3& target);

        virtual void raycast(const Raycaster&, std::vector<Intersection>&) {}

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

        /**
         * @brief Updates the transformation matrix in world space of this 3D object and its descendants.
         *
         * Also recomputes the local transformation matrix. The computation can be controlled with
         * the matrixAutoUpdate and matrixWorldAutoUpdate flags.
         *
         * @param force When true, recomputation is forced even when matrixWorldNeedsUpdate is false. Default is false.
         */
        virtual void updateMatrixWorld(bool force = false);

        /**
         * @brief An alternative to updateMatrixWorld with more control over ancestor and descendant updates.
         *
         * @param updateParents Whether ancestor nodes should be updated. Default is false.
         * @param updateChildren Whether descendant nodes should be updated. Default is false.
         */
        virtual void updateWorldMatrix(bool updateParents = false, bool updateChildren = false);

        static std::shared_ptr<Object3D> create() {

            return std::make_shared<Object3D>();
        }

        [[nodiscard]] virtual std::shared_ptr<BufferGeometry> geometry() const {

            return nullptr;
        }

        [[nodiscard]] virtual std::shared_ptr<Material> material() const {

            return nullptr;
        }

        template<class T>
            requires std::is_base_of<Object3D, T>::value
        T* as() {

            return dynamic_cast<T*>(this);
        }

        template<class T>
            requires std::is_base_of<Object3D, T>::value
        const T* as() const {

            return dynamic_cast<const T*>(this);
        }

        template<class T>
            requires std::is_base_of<Object3D, T>::value
        [[nodiscard]] bool is() const {

            return dynamic_cast<const T*>(this) != nullptr;
        }

        /**
         * @brief Convenience for the common material()->as<T>() downcast.
         *
         * Returns this object's material viewed as T*, or nullptr if there is no material or it
         * is not a T. T may be a concrete material (e.g. MeshStandardMaterial) or a capability
         * mixin (e.g. MaterialWithColor). Null-safe (never dereferences a missing material) and
         * returns nullptr rather than throwing on a type mismatch, so it doubles as a test:
         *   if (auto* m = mesh->materialAs<MeshStandardMaterial>()) m->roughness = 0.4f;
         */
        template<class T>
            requires std::is_base_of<Material, T>::value
        [[nodiscard]] T* materialAs() const {

            const auto m = material();
            return m ? dynamic_cast<T*>(m.get()) : nullptr;
        }

        virtual void copy(const Object3D& source, bool recursive = true);

        template<class T = Object3D>
        std::shared_ptr<T> clone(bool recursive = true) {

            auto clone = createDefault();
            clone->copy(*this, recursive);

            return std::dynamic_pointer_cast<T>(clone);
        }

        ~Object3D() override;

    protected:
        virtual std::shared_ptr<Object3D> createDefault() {

            return std::make_shared<Object3D>();
        }

    private:
        inline static std::atomic<unsigned int> _object3Did{0};

        // Unlink `object` from this node: drop it from `children`, clear its
        // parent, fire "remove", and hand back the owning reference if this node
        // held one. The caller decides whether that reference dies (destroying
        // the object) or is kept. Shared by remove() and removeFromParent() so
        // detach order is defined in exactly one place.
        std::shared_ptr<Object3D> detachChild(Object3D& object);

        std::vector<std::shared_ptr<Object3D>> children_;

        // updateMatrix() change-detection cache: the position/quaternion/scale
        // values at the last compose, plus the matrix bytes that compose
        // produced (so a direct user write to `matrix` is still clobbered on
        // the next updateMatrix(), exactly like three.js). When neither moved,
        // updateMatrix() is a 26-float compare instead of a compose — and by
        // not raising matrixWorldNeedsUpdate it lets updateMatrixWorld() skip
        // the world multiply for the whole unchanged subtree. Pure polling;
        // no user-side notification ever required.
        // When matrixAutoUpdate is false, updateMatrixWorld() reuses
        // composedMatrix_ as a snapshot of the externally-driven `matrix`
        // (helpers alias another object's matrixWorld) so external writes
        // still raise matrixWorldNeedsUpdate; composedPqs_ is NaN-poisoned in
        // that mode so re-enabling matrixAutoUpdate always recomposes.
        std::array<float, 10> composedPqs_{};
        std::array<float, 16> composedMatrix_{};
        bool composedValid_ = false;

        // Value storage for the public `matrix`/`matrixWorld` handles above.
        // Keeping the payload in the object's own cache lines (instead of two
        // per-node heap allocations) removes an indirection from every
        // updateMatrix()/updateMatrixWorld() compare and every scene-prep
        // matrixWorld read, and drops 2 allocs + ~2 control blocks per node.
        // It also makes a moved-from object keep VALID matrices — the move
        // constructor used to null the source's shared_ptrs while the parent's
        // children list could still reference it (a guaranteed nullptr deref
        // on the next traversal).
        Matrix4 matrixLocal_;
        Matrix4 matrixWorldLocal_;
    };

}// namespace threepp

#endif// THREEPP_OBJECT3D_HPP
