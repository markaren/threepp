
#include "threepp/animation/PropertyMixer.hpp"

#include <functional>

using namespace threepp;

namespace {

    using MixFunction = std::function<void(PropertyMixer*, std::vector<float>&, int, int, float, int)>;
    using MixFunctionAdditive = MixFunction;
    using SetIdentity = std::function<void()>;


}// namespace

struct PropertyMixer::Impl {

    size_t valueSize;

    int _origIndex = 3;
    int _addIndex = 4;
    int _workIndex = -1;

    float cumulativeWeight = 0;
    float cumulativeWeightAdditive = 0;

    MixFunction mixFunction;
    MixFunctionAdditive mixFunctionAdditive;
    SetIdentity setIdentity;

    PropertyMixer& scope;

    Impl(PropertyMixer& scope, const std::string& typeName, size_t valueSize)
        : scope(scope), valueSize(valueSize) {


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

        if (typeName == "quaternion") {
            mixFunction = _slerp;
            mixFunctionAdditive = _slerpAdditive;
            this->scope.buffer = std::vector<float>(valueSize * 6);
            this->_workIndex = 5;

        } else if (typeName == "string" || typeName == "bool") {

            mixFunction = _select;

            // Use the regular mix function and for additive on these types,
            // additive is not relevant for non-numeric types
            mixFunctionAdditive = _select;

            this->scope.buffer = std::vector<float>(valueSize * 5);
        } else {
            mixFunction = _lerp;
            mixFunctionAdditive = _lerpAdditive;

            scope.buffer = std::vector<float>(valueSize * 5);
        }
    }

    void accumulate(int accuIndex, float weight) {

        // note: happily accumulating nothing when weight = 0, the caller knows
        // the weight and shouldn't have made the call in the first place

        auto& buffer = this->scope.buffer;
        const auto stride = this->valueSize;
        const auto offset = accuIndex * stride + stride;

        auto currentWeight = this->cumulativeWeight;

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
            //            this._mixBufferRegion( buffer, offset, 0, mix, stride );
        }

        this->cumulativeWeight = currentWeight;
    }

    void accumulateAdditive(float weight) {
    }

    void apply(int accuIndex) {

        const auto stride = this->valueSize;
        const auto& buffer = this->scope.buffer;
        const auto offset = accuIndex * stride + stride;

        const auto weight = this->cumulativeWeight;
        const auto weightAdditive = this->cumulativeWeightAdditive;

        auto& binding = this->scope.binding;

        this->cumulativeWeight = 0;
        this->cumulativeWeightAdditive = 0;

        if (weight < 1) {

            // accuN := accuN + original * ( 1 - cumulativeWeight )

            const auto originalValueOffset = stride * this->_origIndex;

            //                                this->_mixBufferRegion(
            //                                        buffer, offset, originalValueOffset, 1 - weight, stride);
        }

        if (weightAdditive > 0) {

            // accuN := accuN + additive accuN

            //                    this->_mixBufferRegionAdditive(buffer, offset, this->_addIndex * stride, 1, stride);
        }

        for (unsigned i = stride, e = stride + stride; i != e; ++i) {

            if (buffer[i] != buffer[i + stride]) {

                // value has changed -> update scene graph

                binding->setValue(buffer, offset);
                break;
            }
        }
    }

    void restoreOriginalState() {
    }

    void saveOriginalState() {
        auto& binding = this->scope.binding;

        auto& buffer = this->scope.buffer;
        const auto stride = this->valueSize;

        const auto originalValueOffset = stride * this->_origIndex;

        binding->getValue(buffer, originalValueOffset);

        // accu[0..1] := orig -- initially detect changes against the original
        for (unsigned i = stride, e = originalValueOffset; i != e; ++i) {

            buffer[i] = buffer[originalValueOffset + (i % stride)];
        }

        // Add to identity for additive
        this->setIdentity();

        this->cumulativeWeight = 0;
        this->cumulativeWeightAdditive = 0;
    }
};


PropertyMixer::PropertyMixer(const std::shared_ptr<PropertyBinding>& binding, const std::string& typeName, size_t valueSize)
    : binding(binding), pimpl_(std::make_unique<Impl>(*this, typeName, valueSize)) {}


void PropertyMixer::accumulate(int accuIndex, float weight) {

    pimpl_->accumulate(accuIndex, weight);
}

void PropertyMixer::accumulateAdditive(float weight) {

    pimpl_->accumulateAdditive(weight);
}

void PropertyMixer::apply(int accuIndex) {

    pimpl_->apply(accuIndex);
}

void PropertyMixer::restoreOriginalState() {

    pimpl_->restoreOriginalState();
}

void PropertyMixer::saveOriginalState() {

    pimpl_->saveOriginalState();
}


PropertyMixer::~PropertyMixer() = default;
