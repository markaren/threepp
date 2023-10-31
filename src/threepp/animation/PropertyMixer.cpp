
#include "threepp/animation/PropertyMixer.hpp"

#include "threepp/math/Quaternion.hpp"

#include <functional>

using namespace threepp;

namespace {

    using MixFunction = std::function<void(std::vector<float>&, int, int, float, int)>;

    // mix functions

    void _select(std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

        if (t >= 0.5) {

            for (auto i = 0; i != stride; ++i) {

                buffer[dstOffset + i] = buffer[srcOffset + i];
            }
        }
    }

    void _slerp(std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

        Quaternion::slerpFlat(buffer, dstOffset, buffer, dstOffset, buffer, srcOffset, t);
    }


    void _lerp(std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

        const auto s = 1 - t;

        for (auto i = 0; i != stride; ++i) {

            const auto j = dstOffset + i;

            buffer[j] = buffer[j] * s + buffer[srcOffset + i] * t;
        }
    }

    void _lerpAdditive(std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

        for (auto i = 0; i != stride; ++i) {

            const auto j = dstOffset + i;

            buffer[j] = buffer[j] + buffer[srcOffset + i] * t;
        }
    }

}// namespace

struct PropertyMixer::Impl {

    PropertyBinding binding;
    int valueSize;

    std::vector<float> buffer;

    Impl(const PropertyBinding& binding, const std::string& typeName, int valueSize)
        : binding(binding), valueSize(valueSize) {

        MixFunction mixFunction;
        MixFunction mixFunctionAdditive;
        MixFunction setIdentity;

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
            //            mixFunctionAdditive = _slerpAdditive;
            this->buffer = std::vector<float>(valueSize * 6);
            this->_workIndex = 5;

        } else if (typeName == "string" || typeName == "bool") {

            mixFunction = _select;

            // Use the regular mix function and for additive on these types,
            // additive is not relevant for non-numeric types
            mixFunctionAdditive = _select;

            this->buffer = std::vector<float>(valueSize * 5);
        } else {
            mixFunction = _lerp;
            mixFunctionAdditive = _lerpAdditive;

            buffer = std::vector<float>(valueSize * 5);
        }
    }

    void accumulate(int accuIndex, float weight) {

    }

    void accumulateAdditive(float weight) {

    }

    void apply(float accuIndex) {

        const auto stride = this->valueSize;
        const auto& buffer = this->buffer;
        const auto offset = accuIndex * stride + stride;

        const auto weight = this->cumulativeWeight;
        const auto weightAdditive = this->cumulativeWeightAdditive;

        const auto& binding = this->binding;

        this->cumulativeWeight = 0;
        this->cumulativeWeightAdditive = 0;

        if (weight < 1) {

            // accuN := accuN + original * ( 1 - cumulativeWeight )

            const auto originalValueOffset = stride * this->_origIndex;

            //                    this->_mixBufferRegion(
            //                            buffer, offset, originalValueOffset, 1 - weight, stride);
        }

        if (weightAdditive > 0) {

            // accuN := accuN + additive accuN

            //                    this->_mixBufferRegionAdditive(buffer, offset, this->_addIndex * stride, 1, stride);
        }

        for (unsigned i = stride, e = stride + stride; i != e; ++i) {

            if (buffer[i] != buffer[i + stride]) {

                // value has changed -> update scene graph

                //                        binding.setValue(buffer, offset);
                break;
            }
        }
    }

private:
    int _origIndex = 3;
    int _addIndex = 4;
    int _workIndex = -1;

    float cumulativeWeight = 0;
    float cumulativeWeightAdditive = 0;

    int useCount = 0;
    int referenceCount = 0;

    void _slerpAdditive(std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

        const auto workOffset = this->_workIndex * stride;

        // Store result in intermediate buffer offset
        Quaternion::multiplyQuaternionsFlat(buffer, workOffset, buffer, dstOffset, buffer, srcOffset);

        // Slerp to the intermediate result
        Quaternion::slerpFlat(buffer, dstOffset, buffer, dstOffset, buffer, workOffset, t);
    }
};

PropertyMixer::PropertyMixer(const PropertyBinding& binding, const std::string& typeName, int valueSize)
    : pimpl_(std::make_unique<Impl>(binding, typeName, valueSize)) {}

void PropertyMixer::accumulate(int accuIndex, float weight) {

    pimpl_->accumulate(accuIndex, weight);
}

void PropertyMixer::accumulateAdditive(float weight) {

    pimpl_->accumulateAdditive(weight);
}

void PropertyMixer::apply(int accuIndex) {

    pimpl_-> apply(accuIndex);
}


PropertyMixer::~PropertyMixer() = default;
