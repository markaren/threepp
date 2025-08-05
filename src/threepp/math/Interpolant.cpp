
#include "threepp/math/Interpolant.hpp"

using namespace threepp;

Interpolant::Interpolant(Sample parameterPositions, Sample sampleValues, int sampleSize, Sample* resultBuffer)
    : parameterPositions(std::move(parameterPositions)),
      sampleValues(std::move(sampleValues)),
      valueSize(sampleSize) {

    if (resultBuffer) {
        this->resultBuffer = resultBuffer;
    } else {
        this->_resultBuffer.resize(sampleSize);
        this->resultBuffer = &_resultBuffer;
    }
}

// C++
Sample Interpolant::evaluate(float t) {

    const auto& pp = this->parameterPositions;
    size_t i1 = this->_cachedIndex;
    float t1 = (i1 < pp.size()) ? pp[i1] : std::numeric_limits<float>::quiet_NaN();
    float t0 = (i1 > 0 && i1 - 1 < pp.size()) ? pp[i1 - 1] : std::numeric_limits<float>::quiet_NaN();

    // Forward scan
    if (!(t < t1)) {
        for (size_t giveUpAt = i1 + 2;;) {
            if (std::isnan(t1)) {
                if (t < t0) break;
                // after end
                i1 = pp.size();
                this->_cachedIndex = i1;
                return this->copySampleValue_(i1 - 1);
            }
            if (i1 == giveUpAt) break;
            t0 = t1;
            t1 = (++i1 < pp.size()) ? pp[i1] : std::numeric_limits<float>::quiet_NaN();
            if (t < t1) {
                this->_cachedIndex = i1;
                this->intervalChanged_(i1, t0, t1);
                return this->interpolate_(i1, t0, t, t1);
            }
        }
        // Binary search on the right
        size_t right = pp.size();
        while (i1 < right) {
            size_t mid = (i1 + right) >> 1;
            if (t < pp[mid]) {
                right = mid;
            } else {
                i1 = mid + 1;
            }
        }
        t1 = (i1 < pp.size()) ? pp[i1] : std::numeric_limits<float>::quiet_NaN();
        t0 = (i1 > 0 && i1 - 1 < pp.size()) ? pp[i1 - 1] : std::numeric_limits<float>::quiet_NaN();
        if (std::isnan(t0)) {
            this->_cachedIndex = 0;
            return this->copySampleValue_(0);
        }
        if (std::isnan(t1)) {
            i1 = pp.size();
            this->_cachedIndex = i1;
            return this->copySampleValue_(i1 - 1);
        }
        this->_cachedIndex = i1;
        this->intervalChanged_(i1, t0, t1);
        return this->interpolate_(i1, t0, t, t1);
    }

    // Reverse scan
    if (!(t >= t0)) {
        const float t1global = (pp.size() > 1) ? pp[1] : std::numeric_limits<float>::quiet_NaN();
        if (t < t1global) {
            i1 = 2;
            t0 = t1global;
        }
        for (int giveUpAt = static_cast<int>(i1) - 2;;) {
            if (std::isnan(t0)) {
                this->_cachedIndex = 0;
                return this->copySampleValue_(0);
            }
            if (static_cast<int>(i1) == giveUpAt) break;
            t1 = t0;
            int idx = static_cast<int>(--i1) - 1;
            t0 = (idx >= 0 && static_cast<size_t>(idx) < pp.size()) ? pp[idx] : std::numeric_limits<float>::quiet_NaN();
            if (t >= t0) {
                this->_cachedIndex = i1;
                this->intervalChanged_(i1, t0, t1);
                return this->interpolate_(i1, t0, t, t1);
            }
        }
        // Binary search on the left
        size_t right = i1;
        i1 = 0;
        while (i1 < right) {
            size_t mid = (i1 + right) >> 1;
            if (t < pp[mid]) {
                right = mid;
            } else {
                i1 = mid + 1;
            }
        }
        t1 = (i1 < pp.size()) ? pp[i1] : std::numeric_limits<float>::quiet_NaN();
        t0 = (i1 > 0 && i1 - 1 < pp.size()) ? pp[i1 - 1] : std::numeric_limits<float>::quiet_NaN();
        if (std::isnan(t0)) {
            this->_cachedIndex = 0;
            return this->copySampleValue_(0);
        }
        if (std::isnan(t1)) {
            i1 = pp.size();
            this->_cachedIndex = i1;
            return this->copySampleValue_(i1 - 1);
        }
        this->_cachedIndex = i1;
        this->intervalChanged_(i1, t0, t1);
        return this->interpolate_(i1, t0, t, t1);
    }

    // The interval is valid
    this->_cachedIndex = i1;
    this->intervalChanged_(i1, t0, t1);
    return this->interpolate_(i1, t0, t, t1);
}


Sample Interpolant::copySampleValue_(size_t index) const {

    // copies a sample value to the result buffer

    auto result = this->resultBuffer;
    const auto& values = this->sampleValues;
    auto stride = this->valueSize;
    auto offset = index * stride;

    for (auto i = 0; i != stride; ++i) {

        result->at(i) = values[offset + i];
    }

    return *result;
}

std::optional<InterpolantSettings> Interpolant::getSettings_() const {

    if (settings) return *settings;
    if (DefaultSettings_) return *DefaultSettings_;

    return std::nullopt;
}
