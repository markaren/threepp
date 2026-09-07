#include "threepp/core/AttributeView.hpp"

#include "threepp/core/InterleavedBufferAttribute.hpp"

#include <cstdint>

using namespace threepp;

namespace {

    template<class T>
    void widenInto(const BufferAttribute& attribute, std::vector<float>& out) {

        const auto* src = static_cast<const T*>(attribute.data());
        const size_t n = attribute.byteLength() / sizeof(T);
        const auto type = attribute.type();
        const bool normalized = attribute.normalized();

        out.resize(n);

        if (normalized) {
            for (size_t i = 0; i < n; ++i) {
                out[i] = denormalize(static_cast<float>(src[i]), type);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(src[i]);
            }
        }
    }

}// namespace

FloatAttributeView::FloatAttributeView(const BufferAttribute* attribute) {

    if (!attribute) return;

    itemSize_ = attribute->itemSize();
    count_ = attribute->count();

    // An interleaved attribute's storage is the whole strided buffer, so a
    // direct pointer indexed as element*itemSize would read the wrong
    // components. De-stride into owned storage so operator[] and data() are
    // always tightly packed, matching what the stride-aware virtual accessors
    // used to return.
    if (const auto* ib = dynamic_cast<const InterleavedBufferAttribute*>(attribute)) {

        const auto& packed = ib->data->array();
        const auto stride = static_cast<size_t>(ib->data->stride());

        owned_.resize(static_cast<size_t>(count_) * itemSize_);
        for (size_t v = 0; v < static_cast<size_t>(count_); ++v) {
            for (int c = 0; c < itemSize_; ++c) {
                owned_[v * itemSize_ + c] = packed[v * stride + ib->offset + c];
            }
        }

        data_ = owned_.data();
        size_ = owned_.size();
        return;
    }

    if (attribute->type() == AttributeType::Float) {

        // Already float: hand back the attribute's own storage untouched.
        data_ = static_cast<const float*>(attribute->data());
        size_ = attribute->byteLength() / sizeof(float);
        return;
    }

    switch (attribute->type()) {
        case AttributeType::UInt32: widenInto<std::uint32_t>(*attribute, owned_); break;
        case AttributeType::UInt16: widenInto<std::uint16_t>(*attribute, owned_); break;
        case AttributeType::Int16: widenInto<std::int16_t>(*attribute, owned_); break;
        case AttributeType::UInt8: widenInto<std::uint8_t>(*attribute, owned_); break;
        case AttributeType::Int8: widenInto<std::int8_t>(*attribute, owned_); break;
        case AttributeType::Float: break;// handled above
    }

    data_ = owned_.data();
    size_ = owned_.size();
}
