
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

Sample Interpolant::evaluate(float t) {

    auto& pp = this->parameterPositions;
    auto i1 = this->_cachedIndex;
    auto t1 = pp[i1];
    auto t0 = pp[i1 - 1];


    validate_interval: {

    seek: {

    size_t right;

    linear_scan: {

    forward_scan:
    if (!(t < t1)) {

        for (auto giveUpAt = i1 + 2;;) {

            if (std::isnan(t1)) {

                if (t < t0) goto forward_scan;

                // after end

                i1 = pp.size();
                this->_cachedIndex = i1;
                return this->copySampleValue_(i1 - 1 /*, t, t0*/);
            }

            if (i1 == giveUpAt) break;// this loop

            t0 = t1;
            t1 = pp[++i1];

            if (t < t1) {

                // we have arrived at the sought interval
                goto seek;
            }
        }

        // prepare binary search on the right side of the index
        right = pp.size();
        goto linear_scan;
    }// forward scan
}    //linear scan

    // binary search

    while (i1 < right) {

        auto mid = (i1 + right) >> 1;

        if (t < pp[mid]) {

            right = mid;

        } else {

            i1 = mid + 1;
        }
    }

    t1 = pp[i1];
    t0 = pp[i1 - 1];

    // check boundary cases, again

    if (std::isnan(t0)) {

        this->_cachedIndex = 0;
        return this->copySampleValue_(0 /*, t, t1*/);
    }

    if (std::isnan(t1)) {

        i1 = pp.size();
        this->_cachedIndex = i1;
        return this->copySampleValue_(i1 - 1 /*, t0, t*/);
    }

}// seek

    this->_cachedIndex = i1;

    this->intervalChanged_(i1, t0, t1);

}// validate_interval

    return this->interpolate_(i1, t0, t, t1);
}

Sample Interpolant::copySampleValue_(size_t index) {

    // copies a sample value to the result buffer

    auto result = this->resultBuffer;
    auto values = this->sampleValues;
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
