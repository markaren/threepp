
#ifndef THREEPP_PROPERTYMIXER_HPP
#define THREEPP_PROPERTYMIXER_HPP

#include "threepp/animation/PropertyBinding.hpp"

#include <memory>
#include <string>

namespace threepp {

    class PropertyMixer {

    public:
        PropertyMixer(const std::shared_ptr<PropertyBinding>& binding, const std::string& typeName, size_t valueSize);

        void accumulate(int accuIndex, float weight) const;

        void accumulateAdditive(float weight);

        void apply(int accuIndex);

        void restoreOriginalState();

        void saveOriginalState();

        ~PropertyMixer();

    private:
        friend class AnimationMixer;

        int _workIndex{-1};

        size_t useCount{0};
        size_t referenceCount{0};
        size_t _cacheIndex{0};
        std::vector<float> buffer;

        size_t valueSize;

        int _origIndex = 3;
        int _addIndex = 4;

        float cumulativeWeight = 0;
        float cumulativeWeightAdditive = 0;

        std::shared_ptr<PropertyBinding> binding;

        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        // mix functions

        static void _select(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

            if (t >= 0.5) {

                for (auto i = 0; i != stride; ++i) {

                    buffer[dstOffset + i] = buffer[srcOffset + i];
                }
            }
        }

        static void _slerp(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

            Quaternion::slerpFlat(buffer, dstOffset, buffer, dstOffset, buffer, srcOffset, t);
        }


        static void _lerp(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

            const auto s = 1 - t;

            for (auto i = 0; i != stride; ++i) {

                const auto j = dstOffset + i;

                buffer[j] = buffer[j] * s + buffer[srcOffset + i] * t;
            }
        }

        static void _lerpAdditive(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

            for (auto i = 0; i != stride; ++i) {

                const auto j = dstOffset + i;

                buffer[j] = buffer[j] + buffer[srcOffset + i] * t;
            }
        }

        static void _slerpAdditive(PropertyMixer* that, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride) {

            const auto workOffset = that->_workIndex * stride;

            // Store result in intermediate buffer offset
            Quaternion::multiplyQuaternionsFlat(buffer, workOffset, buffer, dstOffset, buffer, srcOffset);

            // Slerp to the intermediate result
            Quaternion::slerpFlat(buffer, dstOffset, buffer, dstOffset, buffer, workOffset, t);
        }

        static void _setAdditiveIdentityNumeric(PropertyMixer* that) {

            const auto startIndex = that->_addIndex * that->valueSize;
            const auto endIndex = startIndex + that->valueSize;

            for (auto i = startIndex; i < endIndex; i++) {

                that->buffer[i] = 0;
            }
        }


        static void _setAdditiveIdentityQuaternion(PropertyMixer* that) {

            that->_setAdditiveIdentityNumeric(that);
            that->buffer[that->_addIndex * that->valueSize + 3] = 1;
        }
    };

}// namespace threepp

#endif//THREEPP_PROPERTYMIXER_HPP
