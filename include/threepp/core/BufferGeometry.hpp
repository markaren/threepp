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

#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp {

    class BufferGeometry : public EventDispatcher {

    public:
        const unsigned int id = ++_id;

        const std::string uuid = utils::generateUUID();

        std::vector<GeometryGroup> groups;

        std::optional<Box3> boundingBox;
        std::optional<Sphere> boundingSphere;

        DrawRange drawRange = DrawRange{0, std::numeric_limits<int>::max() / 2};

        BufferGeometry() = default;

        [[nodiscard]] bool hasIndex() const;

        IntBufferAttribute *getIndex();

        const IntBufferAttribute *getIndex() const;

        BufferGeometry &setIndex(std::vector<int> index);

        BufferGeometry &setIndex(std::unique_ptr<IntBufferAttribute> index);

        template<class T>
        TypedBufferAttribute<T> *getAttribute(const std::string &name) {

            if (!hasAttribute(name)) return nullptr;

            return dynamic_cast<TypedBufferAttribute<T> *>(attributes_.at(name).get());
        }

        template<class T>
        const TypedBufferAttribute<T> *getAttribute(const std::string &name) const {

            if (!hasAttribute(name)) return nullptr;

            return dynamic_cast<TypedBufferAttribute<T> *>(attributes_.at(name).get());
        }

        [[nodiscard]] const std::unordered_map<std::string, std::unique_ptr<BufferAttribute>> &getAttributes() const;

        void setAttribute(const std::string &name, std::unique_ptr<BufferAttribute> attribute);

        bool hasAttribute(const std::string &name) const;

        void addGroup(int start, int count, int materialIndex = 0);

        void clearGroups();

        BufferGeometry &setDrawRange(int start, int count);

        BufferGeometry &applyMatrix4(const Matrix4 &matrix);

        BufferGeometry &applyQuaternion(const Quaternion &q);

        BufferGeometry &rotateX(float angle);

        BufferGeometry &rotateY(float angle);

        BufferGeometry &rotateZ(float angle);

        BufferGeometry &translate(float x, float y, float z);

        BufferGeometry &scale(float x, float y, float z);

        BufferGeometry &center();

        void computeBoundingBox();

        void computeBoundingSphere();

        void normalizeNormals();

        void dispose();

        void copy(const BufferGeometry &source);

        [[nodiscard]] std::shared_ptr<BufferGeometry> clone() const {
            auto g = std::make_shared<BufferGeometry>();
            g->copy(*this);
            return g;
        }

        static std::shared_ptr<BufferGeometry> create() {

            return std::make_shared<BufferGeometry>();
        }

        virtual ~BufferGeometry() = default;

    private:
        std::unique_ptr<IntBufferAttribute> index_;
        std::unordered_map<std::string, std::unique_ptr<BufferAttribute>> attributes_;

        inline static unsigned int _id{0};
    };

}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRY_HPP
