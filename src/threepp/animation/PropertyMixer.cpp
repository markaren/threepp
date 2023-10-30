
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

    MixFunction mixFunction;

    Impl(const PropertyBinding& binding, const std::string& typeName, int valueSize)
        : binding(binding), valueSize(valueSize) {}

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

PropertyMixer::~PropertyMixer() = default;
