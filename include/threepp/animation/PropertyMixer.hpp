
#ifndef THREEPP_PROPERTYMIXER_HPP
#define THREEPP_PROPERTYMIXER_HPP

#include "threepp/animation/PropertyBinding.hpp"

#include <memory>
#include <optional>
#include <string>

namespace threepp {

    class PropertyMixer {

    public:
        PropertyMixer(const std::shared_ptr<PropertyBinding>& binding, const std::string& typeName, size_t valueSize);

        void accumulate(int accuIndex, float weight);

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
    };

}// namespace threepp

#endif//THREEPP_PROPERTYMIXER_HPP
