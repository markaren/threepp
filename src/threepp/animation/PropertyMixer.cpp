
#include "threepp/animation/PropertyMixer.hpp"

#include <functional>

using namespace threepp;

namespace {

    using MixFunction = std::function<void(PropertyMixer*, std::vector<float>&, int, int, float, int)>;
    using MixFunctionAdditive = MixFunction;
    using SetIdentity = std::function<void(PropertyMixer*)>;


}// namespace

struct PropertyMixer::Impl {

    MixFunction _mixBufferRegion;
    MixFunctionAdditive _mixBufferRegionAdditive;
    SetIdentity _setIdentity;

    PropertyMixer& scope;

    Impl(PropertyMixer& scope, const std::string& typeName)
        : scope(scope) {


        // buffer layout: [ incoming | accu0 | accu1 | orig | addAccu | (optional work) ]
        //
        // interpolators can use .buffer as their .result
        // the data then goes to 'incoming'
        //
        // 'accu0' and 'accu1' are used frame-interleaved for
        // the cumulative result and are compared to detect
        // changes
        //
        // 'orig' stores the original state of the property
        //
        // 'add' is used for additive cumulative results
        //
        // 'work' is optional and is only present for quaternion types. It is used
        // to store intermediate quaternion multiplication results

        MixFunction mixFunction;
        MixFunctionAdditive mixFunctionAdditive;
        SetIdentity setIdentity;

        if (typeName == "quaternion") {
            mixFunction = _slerp;
            mixFunctionAdditive = _slerpAdditive;
            setIdentity = _setAdditiveIdentityQuaternion;

            scope.buffer = std::vector<float>(scope.valueSize * 6);
            scope._workIndex = 5;

        } else if (typeName == "string" || typeName == "bool") {

            mixFunction = _select;

            // Use the regular mix function and for additive on these types,
            // additive is not relevant for non-numeric types
            mixFunctionAdditive = _select;

            setIdentity = _setAdditiveIdentityOther;

            this->scope.buffer = std::vector<float>(scope.valueSize * 5);
        } else {
            mixFunction = _lerp;
            mixFunctionAdditive = _lerpAdditive;
            setIdentity = _setAdditiveIdentityNumeric;

            scope.buffer = std::vector<float>(scope.valueSize * 5);
        }

        this->_mixBufferRegion = mixFunction;
        this->_mixBufferRegionAdditive = mixFunctionAdditive;
        this->_setIdentity = setIdentity;
    }

    void accumulate(int accuIndex, float weight) const {

        // note: happily accumulating nothing when weight = 0, the caller knows
        // the weight and shouldn't have made the call in the first place

        auto& buffer = scope.buffer;
        const auto stride = scope.valueSize;
        const auto offset = accuIndex * stride + stride;

        auto currentWeight = scope.cumulativeWeight;

        if (currentWeight == 0) {

            // accuN := incoming * weight

            for (unsigned i = 0; i != stride; ++i) {

                buffer[offset + i] = buffer[i];
            }

            currentWeight = weight;

        } else {

            // accuN := accuN + incoming * weight

            currentWeight += weight;
            const auto mix = weight / currentWeight;
            _mixBufferRegion(&scope, buffer, offset, 0, mix, stride);
        }

        scope.cumulativeWeight = currentWeight;
    }

    void accumulateAdditive(float weight) const {
        auto& buffer = scope.buffer;
        const auto stride = scope.valueSize;
        const auto offset = stride * scope._addIndex;

        if (scope.cumulativeWeightAdditive == 0) {

            // add = identity

            _setIdentity(&scope);
        }

        // add := add + incoming * weight

        this->_mixBufferRegionAdditive(&scope, buffer, offset, 0, weight, stride);
        scope.cumulativeWeightAdditive += weight;
    }

    void apply(int accuIndex) const {

        const auto stride = scope.valueSize;
        auto& buffer = this->scope.buffer;
        const auto offset = accuIndex * stride + stride;

        const auto weight = scope.cumulativeWeight;
        const auto weightAdditive = scope.cumulativeWeightAdditive;

        auto& binding = this->scope.binding;

        scope.cumulativeWeight = 0;
        scope.cumulativeWeightAdditive = 0;

        if (weight < 1) {

            // accuN := accuN + original * ( 1 - cumulativeWeight )

            const auto originalValueOffset = stride * scope._origIndex;

            this->_mixBufferRegion(&scope,
                                   buffer, offset, originalValueOffset, 1 - weight, stride);
        }

        if (weightAdditive > 0) {

            // accuN := accuN + additive accuN

            this->_mixBufferRegionAdditive(&scope, buffer, offset, scope._addIndex * stride, 1, stride);
        }

        for (unsigned i = stride, e = stride + stride; i != e; ++i) {

            if (buffer[i] != buffer[i + stride]) {

                // value has changed -> update scene graph

                binding->setValue(buffer, offset);
                break;
            }
        }
    }

    void saveOriginalState() const {
        auto& binding = this->scope.binding;

        auto& buffer = this->scope.buffer;
        const auto stride = scope.valueSize;

        const auto originalValueOffset = stride * scope._origIndex;

        binding->getValue(buffer, originalValueOffset);

        // accu[0..1] := orig -- initially detect changes against the original
        for (unsigned i = stride, e = originalValueOffset; i != e; ++i) {

            buffer[i] = buffer[originalValueOffset + (i % stride)];
        }

        // Add to identity for additive
        _setIdentity(&scope);

        scope.cumulativeWeight = 0;
        scope.cumulativeWeightAdditive = 0;
    }

    void restoreOriginalState() const {
        const auto originalValueOffset = scope.valueSize * 3;
        scope.binding->setValue(scope.buffer, originalValueOffset);
    }
};


PropertyMixer::PropertyMixer(const std::shared_ptr<PropertyBinding>& binding, const std::string& typeName, size_t valueSize)
    : binding(binding), valueSize(valueSize), pimpl_(std::make_unique<Impl>(*this, typeName)) {}


void PropertyMixer::accumulate(int accuIndex, float weight) const {

    pimpl_->accumulate(accuIndex, weight);
}

void PropertyMixer::accumulateAdditive(float weight) const {

    pimpl_->accumulateAdditive(weight);
}

void PropertyMixer::apply(int accuIndex) const {

    pimpl_->apply(accuIndex);
}

void PropertyMixer::restoreOriginalState() const {

    pimpl_->restoreOriginalState();
}

void PropertyMixer::saveOriginalState() const {

    pimpl_->saveOriginalState();
}


PropertyMixer::~PropertyMixer() = default;

void PropertyMixer::_select(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

    if (t >= 0.5) {

        for (auto i = 0; i != stride; ++i) {

            buffer[dstOffset + i] = buffer[srcOffset + i];
        }
    }
}
void PropertyMixer::_slerp(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

    Quaternion::slerpFlat(buffer, dstOffset, buffer, dstOffset, buffer, srcOffset, t);
}
void PropertyMixer::_lerp(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

    const auto s = 1 - t;

    for (auto i = 0; i != stride; ++i) {

        const auto j = dstOffset + i;

        buffer[j] = buffer[j] * s + buffer[srcOffset + i] * t;
    }
}

void PropertyMixer::_lerpAdditive(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

    for (auto i = 0; i != stride; ++i) {

        const auto j = dstOffset + i;

        buffer[j] = buffer[j] + buffer[srcOffset + i] * t;
    }
}

void PropertyMixer::_slerpAdditive(PropertyMixer* that, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

    const auto workOffset = that->_workIndex * stride;

    // Store result in intermediate buffer offset
    Quaternion::multiplyQuaternionsFlat(buffer, workOffset, buffer, dstOffset, buffer, srcOffset);

    // Slerp to the intermediate result
    Quaternion::slerpFlat(buffer, dstOffset, buffer, dstOffset, buffer, workOffset, t);
}

void PropertyMixer::_setAdditiveIdentityNumeric(PropertyMixer* that) {

    const auto startIndex = that->_addIndex * that->valueSize;
    const auto endIndex = startIndex + that->valueSize;

    for (auto i = startIndex; i < endIndex; i++) {

        that->buffer[i] = 0;
    }
}

void PropertyMixer::_setAdditiveIdentityQuaternion(PropertyMixer* that) {

    _setAdditiveIdentityNumeric(that);
    that->buffer[that->_addIndex * that->valueSize + 3] = 1;
}

void PropertyMixer::_setAdditiveIdentityOther(PropertyMixer* that) {
    const size_t startIndex = that->_origIndex * that->valueSize;
    const size_t targetIndex = that->_addIndex * that->valueSize;

    for (size_t i = 0; i < that->valueSize; ++i) {
        that->buffer[targetIndex + i] = that->buffer[startIndex + i];
    }
}
