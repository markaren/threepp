// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferGeometry.js

#ifndef THREEPP_BUFFERGEOMETRY_HPP
#define THREEPP_BUFFERGEOMETRY_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/misc.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Sphere.hpp"

#include <any>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp {

    class BufferGeometry {

    public:
        const unsigned int id = _id++;

        std::string uuid = generateUUID();

        std::vector<GeometryGroup> groups;

        std::optional<Box3> boundingBox;
        std::optional<Sphere> boundingSphere;

        DrawRange drawRange = DrawRange{0, Infinity<int>};

        BufferGeometry() = default;

        BufferGeometry(const BufferGeometry &) = delete;

        std::vector<int> &getIndex() {

            return this->index;
        }

        BufferGeometry &setIndex(const std::vector<int> &index) {

            this->index = index;

            return *this;
        }

        template<class T>
        BufferAttribute<T> &getAttribute(const std::string &name) {

            if (!hasAttribute(name)) throw std::runtime_error("No attribute named: " + name);

            return std::any_cast<BufferAttribute<T> &>(attributes_[name]);
        }

        template<class T>
        void setAttribute(const std::string &name, BufferAttribute<T> attribute) {

            attributes_[name] = std::move(attribute);
        }

        bool hasAttribute(const std::string &name) {

            return attributes_.count(name) != 0;
        }

        void addGroup(int start, int count, int materialIndex = 0) {

            groups.emplace_back(GeometryGroup{start, count, materialIndex});
        }

        void clearGroups() {

            groups.clear();
        }

        BufferGeometry &setDrawRange(int start, int count) {

            this->drawRange.start = start;
            this->drawRange.count = count;

            return *this;
        }

        BufferGeometry &applyMatrix4(const Matrix4 &matrix) {

            if (this->attributes_.count("position")) {

                auto position = std::any_cast<BufferAttribute<float>>(this->attributes_["position"]);

                position.applyMatrix4(matrix);

                position.needsUpdate();
            }


            if (this->attributes_.count("normal")) {

                auto normal = std::any_cast<BufferAttribute<float>>(this->attributes_["normal"]);

                auto normalMatrix = Matrix3().getNormalMatrix(matrix);

                normal.applyNormalMatrix(normalMatrix);

                normal.needsUpdate();
            }


            if (this->attributes_.count("tangent")) {

                auto tangent = std::any_cast<BufferAttribute<float>>(this->attributes_["tangent"]);

                tangent.transformDirection(matrix);

                tangent.needsUpdate();
            }

            if (!this->boundingBox) {

                this->computeBoundingBox();
            }

            if (!this->boundingSphere) {

                this->computeBoundingSphere();
            }

            return *this;
        }

        BufferGeometry &applyQuaternion(const Quaternion &q) {

            _m1.makeRotationFromQuaternion(q);

            this->applyMatrix4(_m1);

            return *this;
        }

        BufferGeometry &rotateX(float angle) {

            // rotate geometry around world x-axis

            _m1.makeRotationX(angle);

            this->applyMatrix4(_m1);

            return *this;
        }

        BufferGeometry &rotateY(float angle) {

            // rotate geometry around world y-axis

            _m1.makeRotationY(angle);

            this->applyMatrix4(_m1);

            return *this;
        }

        BufferGeometry &rotateZ(float angle) {

            // rotate geometry around world z-axis

            _m1.makeRotationZ(angle);

            this->applyMatrix4(_m1);

            return *this;
        }

        BufferGeometry &translate(float x, float y, float z) {

            // translate geometry

            _m1.makeTranslation(x, y, z);

            this->applyMatrix4(_m1);

            return *this;
        }

        BufferGeometry &scale(float x, float y, float z) {

            // scale geometry

            _m1.makeScale(x, y, z);

            this->applyMatrix4(_m1);

            return *this;
        }

        BufferGeometry &center() {

            this->computeBoundingBox();

            this->boundingBox->getCenter( _offset );
            _offset.negate();

            this->translate( _offset.x, _offset.y, _offset.z );

            return *this;

        }

        void computeBoundingBox() {

            if ( !this->boundingBox ) {

                this->boundingBox = Box3();

            }

            if ( this->attributes_.count("position") != 0 ) {

                const auto& position = std::any_cast<BufferAttribute<float>&>(this->attributes_["position"]);

                this->boundingBox->setFromBufferAttribute( position );


            } else {

                this->boundingBox->makeEmpty();

            }

            if ( std::isnan( this->boundingBox->min().x ) || std::isnan( this->boundingBox->min().y ) || std::isnan( this->boundingBox->min().z ) ) {

                // TODO
                //console.error( 'THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.', this );

            }

        }

        void computeBoundingSphere() {

            if ( !this->boundingSphere ) {

                this->boundingSphere = Sphere();

            }

            if ( this->attributes_.count("position") != 0 ) {

                const auto& position = std::any_cast<BufferAttribute<float>&>(this->attributes_["position"]);

                // first, find the center of the bounding sphere

                auto center = this->boundingSphere->center;

                _box.setFromBufferAttribute( position );

                // process morph attributes if present

                _box.getCenter( center );

                // second, try to find a boundingSphere with a radius smaller than the
                // boundingSphere of the boundingBox: sqrt(3) smaller in the best case

                float maxRadiusSq = 0;

                for ( auto i = 0, il = position.count(); i < il; i ++ ) {

                    _vector.fromBufferAttribute( position, i );

                    maxRadiusSq = std::max( maxRadiusSq, center.distanceToSquared( _vector ) );

                }

                this->boundingSphere->radius = std::sqrt( maxRadiusSq );

                if ( std::isnan( this->boundingSphere->radius ) ) {

                    // TODO
                    //console.error( 'THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.', this );

                }

            }

        }

    private:
        std::vector<int> index;
        std::unordered_map<std::string, std::any> attributes_;

        static Matrix4 _m1;
        static Vector3 _offset;
        static Box3 _box;
        static Vector3 _vector;

        static unsigned int _id;
    };


}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRY_HPP
