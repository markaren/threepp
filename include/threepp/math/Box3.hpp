// https://github.com/mrdoob/three.js/blob/r129/src/math/Box3.js

#ifndef THREEPP_BOX3_HPP
#define THREEPP_BOX3_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/math/infinity.hpp"

#include <vector>

namespace threepp {

    class Triangle;
    class Sphere;
    class Plane;
    class Object3D;

    /**
     * Represents an axis-aligned bounding box (AABB) in 3D space.
     */
    class Box3 {

    public:
        Box3();

        Box3(const Vector3& min, const Vector3& max);

        [[nodiscard]] const Vector3& min() const;

        [[nodiscard]] const Vector3& max() const;

        Box3& set(const Vector3& min, const Vector3& max);

        Box3& set(float minX, float minY, float minZ, float maxX, float maxY, float maxZ);

        template<class ArrayLike>
        Box3& setFromArray(const ArrayLike& array) {

            auto minX = +Infinity<float>;
            auto minY = +Infinity<float>;
            auto minZ = +Infinity<float>;

            auto maxX = -Infinity<float>;
            auto maxY = -Infinity<float>;
            auto maxZ = -Infinity<float>;

            for (int i = 0, l = array.size(); i < l; i += 3) {

                const auto x = array[i];
                const auto y = array[i + 1];
                const auto z = array[i + 2];

                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (z < minZ) minZ = z;

                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                if (z > maxZ) maxZ = z;
            }

            this->min_.set(minX, minY, minZ);
            this->max_.set(maxX, maxY, maxZ);

            return *this;
        }

        template<class ArrayLike>
        Box3& setFromPoints(const ArrayLike& points) {

            this->makeEmpty();

            for (const auto& point : points) {

                this->expandByPoint(point);
            }

            return *this;
        }

        // Centers this box on center and sets this box's width, height and depth to the values specified
        // in size
        Box3& setFromCenterAndSize(const Vector3& center, const Vector3& size);

        // Computes the world-axis-aligned bounding box of an Object3D (including its children),
        // accounting for the object's, and children's, world transforms.
        // The function may result in a larger box than strictly necessary.
        Box3& setFromObject(Object3D& object, bool precise = false);

        [[nodiscard]] Box3 clone() const;

        Box3& copy(const Box3& box);

        // Makes this box empty.
        Box3& makeEmpty();

        // Returns true if this box includes zero points within its bounds.
        // Note that a box with equal lower and upper bounds still includes one point, the one both bounds share.
        [[nodiscard]] bool isEmpty() const;

        // Get the center point of the box as a Vector3. Returns by reference
        void getCenter(Vector3& target) const;

        // Get the center point of the box as a Vector3. Returns by reference
        [[nodiscard]] Vector3 getCenter() const;

        // Get the width, height and depth of this box.
        void getSize(Vector3& target) const;

        // Get the width, height and depth of this box.
        [[nodiscard]] Vector3 getSize() const;

        Box3& expandByPoint(const Vector3& point);

        Box3& expandByVector(const Vector3& vector);

        Box3& expandByScalar(float scalar);

        Box3& expandByObject(Object3D& object, bool presice = false);

        [[nodiscard]] bool containsPoint(const Vector3& point) const;

        // Returns true if this box includes the entirety of box. If this and box are identical,
        // this function also returns true.
        [[nodiscard]] bool containsBox(const Box3& box) const;

        void getParameter(const Vector3& point, Vector3& target) const;

        [[nodiscard]] bool intersectsBox(const Box3& box) const;

        [[nodiscard]] bool intersectsSphere(const Sphere& sphere) const;

        [[nodiscard]] bool intersectsPlane(const Plane& plane) const;

        [[nodiscard]] bool intersectsTriangle(const Triangle& triangle) const;

        Vector3& clampPoint(const Vector3& point, Vector3& target) const;

        // Returns the distance from any edge of this box to the specified point.
        // If the point lies inside of this box, the distance will be 0.
        [[nodiscard]] float distanceToPoint(const Vector3& point) const;

        void getBoundingSphere(Sphere& target) const;

        Box3& intersect(const Box3& box);

        Box3& union_(const Box3& box);

        // Transforms this Box3 with the supplied matrix.
        Box3& applyMatrix4(const Matrix4& matrix);

        Box3& translate(const Vector3& offset);

        [[nodiscard]] bool equals(const Box3& box) const;

        bool operator==(const Box3& other) const;

        friend std::ostream& operator<<(std::ostream& os, const Box3& v) {
            os << "Box3(min=" << v.min_ << ", max=" << v.max_ << ")";
            return os;
        }

    private:
        Vector3 min_;
        Vector3 max_;
    };

}// namespace threepp

#endif//THREEPP_BOX3_HPP
