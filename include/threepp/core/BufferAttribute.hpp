// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferAttribute.js

#ifndef THREEPP_BUFFER_ATTRIBUTE_HPP
#define THREEPP_BUFFER_ATTRIBUTE_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/constants.hpp"
#include "threepp/core/misc.hpp"

#include <iostream>
#include <vector>

namespace threepp {

    template<class T>
    class TypedBufferAttribute;

    class BufferAttribute {

    public:

        UpdateRange updateRange{0, -1};

        unsigned int version = 0;

        [[nodiscard]] virtual int count() const = 0;

        [[nodiscard]] int itemSize() const {

            return itemSize_;
        }

        [[nodiscard]] bool normalized() const {

            return normalized_;
        }

        [[nodiscard]] int getUsage() const {

            return usage_;
        }

        void needsUpdate() {

            ++version;
        }

        void setUsage(int value) {

            this->usage_ = value;
        }

        template<class T>
        TypedBufferAttribute<T>* typed() {

            return dynamic_cast<TypedBufferAttribute<T>*>(this);
        }

        virtual ~BufferAttribute() = default;

    protected:
        int itemSize_{};
        bool normalized_{};

        int usage_{StaticDrawUsage};

        BufferAttribute() = default;

        BufferAttribute(int itemSize, bool normalized)
            : itemSize_(itemSize), normalized_(normalized) {}

        void copy(const BufferAttribute& source) {

//            this->name = source.name;
            this->itemSize_ = source.itemSize_;
            this->normalized_ = source.normalized_;

            this->usage_ = source.usage_;
        }

        inline static Vector3 _vector{};
        inline static Vector2 _vector2{};
    };

    template<class T>
    class TypedBufferAttribute : public BufferAttribute {

    public:

        [[nodiscard]] int count() const override {

            return count_;
        }

        virtual const std::vector<T> &array() const {

            return array_;
        }

        TypedBufferAttribute<T> &copyAt(unsigned int index1, const TypedBufferAttribute<T> &attribute, unsigned int index2) {

            index1 *= this->itemSize_;
            index2 *= attribute.itemSize_;

            for (auto i = 0, l = this->itemSize_; i < l; i++) {

                this->array_[index1 + i] = attribute.array_[index2 + i];
            }

            return &this;
        }

        TypedBufferAttribute<T> &copyArray(const std::vector<T> &array) {

            this->array_ = array;

            return *this;
        }

        TypedBufferAttribute<T> &copyColorsArray(const std::vector<Color> &colors) {

            int offset = 0;

            for (const auto &color : colors) {

                array_[offset++] = color.r;
                array_[offset++] = color.g;
                array_[offset++] = color.b;
            }

            return *this;
        }

        TypedBufferAttribute<T> &copyVector2sArray(const std::vector<Vector2> &vectors) {

            int offset = 0;

            for (const auto &vector : vectors) {

                array_[offset++] = vector.x;
                array_[offset++] = vector.y;
            }

            return *this;
        }

        TypedBufferAttribute<T> &copyVector3sArray(const std::vector<Vector3> &vectors) {

            int offset = 0;

            for (const auto &vector : vectors) {

                array_[offset++] = vector.x;
                array_[offset++] = vector.y;
                array_[offset++] = vector.z;
            }

            return *this;
        }

        TypedBufferAttribute<T> &copyVector4sArray(std::vector<Vector4> &vectors) {

            int offset = 0;

            for (const auto &vector : vectors) {

                array_[offset++] = vector.x;
                array_[offset++] = vector.y;
                array_[offset++] = vector.z;
                array_[offset++] = vector.w;
            }

            return *this;
        }

        TypedBufferAttribute<T> &applyMatrix3(const Matrix3 &m) {

            if (this->itemSize_ == 2) {

                for (size_t i = 0, l = this->count_; i < l; i++) {

                    setFromBufferAttribute(_vector2, i);
                    _vector2.applyMatrix3(m);

                    this->setXY(i, _vector2.x, _vector2.y);
                }

            } else if (this->itemSize_ == 3) {

                for (size_t i = 0, l = this->count_; i < l; i++) {

                    setFromBufferAttribute(_vector, i);
                    _vector.applyMatrix3(m);

                    this->setXYZ(i, _vector.x, _vector.y, _vector.z);
                }
            }

            return *this;
        }

        TypedBufferAttribute<T> &applyMatrix4(const Matrix4 &m) {

            for (int i = 0, l = this->count_; i < l; i++) {

                _vector.x = this->getX(i);
                _vector.y = this->getY(i);
                _vector.z = this->getZ(i);

                _vector.applyMatrix4(m);

                this->setXYZ(i, _vector.x, _vector.y, _vector.z);
            }

            return *this;
        }

        TypedBufferAttribute<T> &applyNormalMatrix(const Matrix3 &m) {

            for (int i = 0, l = this->count_; i < l; i++) {

                _vector.x = this->getX(i);
                _vector.y = this->getY(i);
                _vector.z = this->getZ(i);

                _vector.applyNormalMatrix(m);

                this->setXYZ(i, _vector.x, _vector.y, _vector.z);
            }

            return *this;
        }

        TypedBufferAttribute<T> &transformDirection(const Matrix4 &m) {

            for (int i = 0, l = this->count_; i < l; i++) {

                _vector.x = this->getX(i);
                _vector.y = this->getY(i);
                _vector.z = this->getZ(i);

                _vector.transformDirection(m);

                this->setXYZ(i, _vector.x, _vector.y, _vector.z);
            }

            return *this;
        }

        [[nodiscard]] T getX(int index) const {

            return this->array_[index * this->itemSize_];
        }

        TypedBufferAttribute<T> &setX(int index, T x) {

            this->array_[index * this->itemSize_] = x;

            return *this;
        }

        [[nodiscard]] T getY(int index) const {

            return this->array_[index * this->itemSize_ + 1];
        }

        TypedBufferAttribute<T> &setY(int index, T y) {

            this->array_[index * this->itemSize_ + 1] = y;

            return *this;
        }

        [[nodiscard]] T getZ(int index) const {

            return this->array_[index * this->itemSize_ + 2];
        }

        TypedBufferAttribute<T> &setZ(int index, T z) {

            this->array_[index * this->itemSize_ + 2] = z;

            return *this;
        }

        [[nodiscard]] T getW(int index) const {

            return this->array_[index * this->itemSize_ + 3];
        }

        TypedBufferAttribute<T> &setW(int index, T w) {

            this->array_[index * this->itemSize_ + 3] = w;

            return *this;
        }

        TypedBufferAttribute<T> &setXY(int index, T x, T y) {

            index *= this->itemSize_;

            this->array_[index + 0] = x;
            this->array_[index + 1] = y;

            return *this;
        }

        TypedBufferAttribute<T> &setXYZ(int index, T x, T y, T z) {

            index *= this->itemSize_;

            this->array_[index + 0] = x;
            this->array_[index + 1] = y;
            this->array_[index + 2] = z;

            return *this;
        }

        TypedBufferAttribute<T> &setXYZW(int index, T x, T y, T z, T w) {

            index *= this->itemSize_;

            this->array_[index + 0] = x;
            this->array_[index + 1] = y;
            this->array_[index + 2] = z;
            this->array_[index + 3] = w;

            return *this;
        }

        void setFromBufferAttribute(Vector2 &target, int index) const {

            target.x = getX(index);
            target.y = getY(index);
        }

        void setFromBufferAttribute(Vector3 &target, int index) const {

            target.x = getX(index);
            target.y = getY(index);
            target.z = getZ(index);
        }

        void setFromBufferAttribute(Vector4 &target, int index) const {

            target.x = getX(index);
            target.y = getY(index);
            target.z = getZ(index);
            target.w = getW(index);
        }

        void setFromBufferAttribute(Box3 &target) const {

            auto minX = +Infinity<float>;
            auto minY = +Infinity<float>;
            auto minZ = +Infinity<float>;

            auto maxX = -Infinity<float>;
            auto maxY = -Infinity<float>;
            auto maxZ = -Infinity<float>;

            for (int i = 0, l = count(); i < l; i++) {

                const auto x = getX(i);
                const auto y = getY(i);
                const auto z = getZ(i);

                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (z < minZ) minZ = z;

                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                if (z > maxZ) maxZ = z;
            }

            target.set(minX, minY, minZ, maxX, maxY, maxZ);
        }

        void copy(const TypedBufferAttribute<T> &source) {
            BufferAttribute::copy(source);

            this->count_ = source.count_;
            this->array_ = array_;
        }

        [[nodiscard]] std::unique_ptr<TypedBufferAttribute<T>> clone() const {
            auto clone = std::unique_ptr<TypedBufferAttribute<T>>(new TypedBufferAttribute<T>());
            clone->copy(*this);

            return clone;
        }

        static std::unique_ptr<TypedBufferAttribute<T>> create(std::vector<T> array, int itemSize, bool normalized = false) {

            return std::unique_ptr<TypedBufferAttribute<T>>(new TypedBufferAttribute<T>(array, itemSize, normalized));
        }

    protected:
        TypedBufferAttribute() = default;

        TypedBufferAttribute(const std::vector<T>& array, int count): array_(array), count_(count) {}

        TypedBufferAttribute(const std::vector<T>& array, int itemSize, bool normalized)
            : BufferAttribute(itemSize, normalized), array_(array), count_(array_.size() / itemSize) {}

    private:
        std::vector<T> array_;
        int count_{};

    };

    typedef TypedBufferAttribute<int> IntBufferAttribute;
    typedef TypedBufferAttribute<float> FloatBufferAttribute;


}// namespace threepp

#endif//THREEPP_BUFFER_ATTRIBUTE_HPP
