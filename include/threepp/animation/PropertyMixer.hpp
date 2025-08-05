
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

        void accumulateAdditive(float weight) const;

        void apply(int accuIndex) const;

        void restoreOriginalState() const;

        void saveOriginalState() const;

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

        static void _select(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride);

        static void _slerp(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride);

        static void _lerp(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride);

        static void _lerpAdditive(PropertyMixer*, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride);

        static void _slerpAdditive(PropertyMixer* that, std::vector<float>& buffer, int dstOffset, int srcOffset, float t, int stride);

        static void _setAdditiveIdentityNumeric(PropertyMixer* that);

        static void _setAdditiveIdentityQuaternion(PropertyMixer* that);

        static void _setAdditiveIdentityOther(PropertyMixer* that);
    };

}// namespace threepp

#endif//THREEPP_PROPERTYMIXER_HPP
