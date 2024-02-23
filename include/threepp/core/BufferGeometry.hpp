// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferGeometry.js

#ifndef THREEPP_BUFFERGEOMETRY_HPP
#define THREEPP_BUFFERGEOMETRY_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/math/Sphere.hpp"

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/core/BufferAttribute.hpp"

#include <optional>
#include <unordered_map>

namespace threepp {

    class BufferGeometry: public EventDispatcher {

    public:
        const unsigned int id{++_id};

        const std::string uuid;

        std::string name;

        bool morphTargetsRelative{false};

        std::vector<GeometryGroup> groups;

        std::optional<Box3> boundingBox;
        std::optional<Sphere> boundingSphere;

        DrawRange drawRange{0, std::numeric_limits<int>::max() / 2};

        BufferGeometry();

        [[nodiscard]] virtual std::string type() const {

            return "BufferGeometry";
        }

        [[nodiscard]] bool hasIndex() const;

        IntBufferAttribute* getIndex();

        [[nodiscard]] const IntBufferAttribute* getIndex() const;

        template<class ArrayLike>
        BufferGeometry& setIndex(const ArrayLike& index) {

            this->index_ = IntBufferAttribute::create(index, 1);

            return *this;
        }

        BufferAttribute* getAttribute(const std::string& name);

        template<class T>
        TypedBufferAttribute<T>* getAttribute(const std::string& name) {

            if (!hasAttribute(name)) return nullptr;

            return dynamic_cast<TypedBufferAttribute<T>*>(attributes_.at(name).get());
        }

        template<class T>
        const TypedBufferAttribute<T>* getAttribute(const std::string& name) const {

            if (!hasAttribute(name)) return nullptr;

            return dynamic_cast<TypedBufferAttribute<T>*>(attributes_.at(name).get());
        }

        std::vector<std::shared_ptr<BufferAttribute>>* getMorphAttribute(const std::string& name);

        std::vector<std::shared_ptr<BufferAttribute>>* getOrCreateMorphAttribute(const std::string& name);

        [[nodiscard]] const std::unordered_map<std::string, std::shared_ptr<BufferAttribute>>& getAttributes() const;

        void setAttribute(const std::string& name, std::shared_ptr<BufferAttribute> attribute);

        void deleteAttribute(const std::string& name);

        [[nodiscard]] bool hasAttribute(const std::string& name) const;

        [[nodiscard]] const std::unordered_map<std::string, std::vector<std::shared_ptr<BufferAttribute>>>& getMorphAttributes() const;

        void addGroup(int start, int count, unsigned int materialIndex = 0);

        void clearGroups();

        BufferGeometry& setDrawRange(int start, int count);

        BufferGeometry& applyMatrix4(const Matrix4& matrix);

        BufferGeometry& applyQuaternion(const Quaternion& q);

        BufferGeometry& rotateX(float angle);

        BufferGeometry& rotateY(float angle);

        BufferGeometry& rotateZ(float angle);

        BufferGeometry& translate(float x, float y, float z);

        BufferGeometry& scale(float x, float y, float z);

        BufferGeometry& center();

        BufferGeometry& setFromPoints(const std::vector<Vector2>& points);

        BufferGeometry& setFromPoints(const std::vector<Vector3>& points);

        void computeBoundingBox();

        void computeBoundingSphere();

        void normalizeNormals();

        [[nodiscard]] std::shared_ptr<BufferGeometry> toNonIndexed() const;

        void computeVertexNormals();

        void dispose();

        void copy(const BufferGeometry& source);

        [[nodiscard]] std::shared_ptr<BufferGeometry> clone() const;

        ~BufferGeometry() override;

        static std::shared_ptr<BufferGeometry> create();

    private:
        bool disposed_ = false;
        std::unique_ptr<IntBufferAttribute> index_;
        std::unordered_map<std::string, std::shared_ptr<BufferAttribute>> attributes_;
        std::unordered_map<std::string, std::vector<std::shared_ptr<BufferAttribute>>> morphAttributes_;

        inline static unsigned int _id{0};
    };

}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRY_HPP
