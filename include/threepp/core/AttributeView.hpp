#ifndef THREEPP_ATTRIBUTEVIEW_HPP
#define THREEPP_ATTRIBUTEVIEW_HPP

#include "threepp/core/BufferAttribute.hpp"

#include <utility>
#include <vector>

namespace threepp {

    // Read-only float view over a BufferAttribute of any scalar type.
    //
    // Zero-copy when the attribute is already float — data() points straight at
    // the attribute's own storage and nothing is allocated, so code converted to
    // use this view behaves exactly as it did when it called ->array().data().
    // For a narrow integer attribute the values are widened once into owned
    // storage, applying the UNORM/SNORM mapping when the attribute is
    // `normalized`.
    //
    // Construct from a null attribute to get an empty view; test with operator
    // bool. An InterleavedBufferAttribute is de-strided into owned storage, so
    // data()/operator[] are always tightly packed regardless of the source
    // layout — unlike ->array(), which exposes the whole interleaved buffer.
    class FloatAttributeView {

    public:
        FloatAttributeView() = default;

        explicit FloatAttributeView(const BufferAttribute* attribute);

        // Copying would leave data_ dangling into the source's owned_ storage,
        // so the view is move-only and the move re-points at the moved-to vector.
        FloatAttributeView(const FloatAttributeView&) = delete;
        FloatAttributeView& operator=(const FloatAttributeView&) = delete;

        FloatAttributeView(FloatAttributeView&& other) noexcept
            : size_(other.size_), itemSize_(other.itemSize_), count_(other.count_),
              owned_(std::move(other.owned_)) {

            data_ = owned_.empty() ? other.data_ : owned_.data();
            other.data_ = nullptr;
            other.size_ = 0;
        }

        FloatAttributeView& operator=(FloatAttributeView&& other) noexcept {

            if (this != &other) {
                size_ = other.size_;
                itemSize_ = other.itemSize_;
                count_ = other.count_;
                owned_ = std::move(other.owned_);
                data_ = owned_.empty() ? other.data_ : owned_.data();
                other.data_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        explicit operator bool() const {

            return data_ != nullptr;
        }

        [[nodiscard]] const float* data() const {

            return data_;
        }

        [[nodiscard]] size_t size() const {

            return size_;
        }

        [[nodiscard]] int itemSize() const {

            return itemSize_;
        }

        [[nodiscard]] int count() const {

            return count_;
        }

        // True when the values had to be widened, i.e. data() points at storage
        // owned by this view rather than at the attribute.
        [[nodiscard]] bool widened() const {

            return !owned_.empty();
        }

        float operator[](size_t i) const {

            THREEPP_ASSERT_MSG(i < size_, "FloatAttributeView: index out of range");
            return data_[i];
        }

    private:
        const float* data_{};
        size_t size_{};
        int itemSize_{};
        int count_{};
        std::vector<float> owned_;
    };

}// namespace threepp

#endif//THREEPP_ATTRIBUTEVIEW_HPP
