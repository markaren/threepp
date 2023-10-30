
#include "threepp/animation/KeyframeTrack.hpp"

#include "threepp/math/interpolants/CubicInterpolant.hpp"
#include "threepp/math/interpolants/DiscreteInterpolant.hpp"
#include "threepp/math/interpolants/LinearInterpolant.hpp"

using namespace threepp;

Interpolation KeyframeTrack::defaultInterpolation = Interpolation::Linear;

KeyframeTrack::KeyframeTrack(std::string name, const std::vector<float>& times, const std::vector<float>& values, std::optional<Interpolation> interpolation)
    : name_(std::move(name)),
      times_(times),
      values_(values),
      interpolation_(interpolation.value_or(defaultInterpolation)) {}

KeyframeTrack& KeyframeTrack::shift(float timeOffset) {

    if (timeOffset != 0) {

        for (float& time : times_) {

            time += timeOffset;
        }
    }

    return *this;
}

KeyframeTrack& KeyframeTrack::scale(float timeScale) {

    if (timeScale != 1) {

        for (float& time : times_) {

            time *= timeScale;
        }
    }

    return *this;
}

KeyframeTrack& KeyframeTrack::trim(float startTime, float endTime) {

    auto nKeys = times_.size();

    int from = 0;
    int to = static_cast<int>(nKeys) - 1;

    while (from != nKeys && times_[from] < startTime) {

        ++from;
    }

    while (to != -1 && times_[to] > endTime) {

        --to;
    }

    ++to;// inclusive -> exclusive bound

    if (from != 0 || to != nKeys) {

        // empty tracks are forbidden, so keep at least one keyframe
        if (from >= to) {

            to = std::max(to, 1);
            from = to - 1;
        }

        auto stride = static_cast<int>(this->getValueSize());
        this->times_ = AnimationUtils::arraySlice(times_, from, to);
        this->values_ = AnimationUtils::arraySlice(this->values_, from * stride, to * stride);
    }

    return *this;
}

KeyframeTrack& KeyframeTrack::optimize() {

    // times or values may be shared with other tracks, so overwriting is unsafe
    auto times = AnimationUtils::arraySlice(this->times_);
    auto values = AnimationUtils::arraySlice(this->values_);
    const auto stride = this->getValueSize();

    const auto smoothInterpolation = this->getInterpolation() == Interpolation::Smooth;

    const auto lastIndex = times.size() - 1;

    auto writeIndex = 1;

    for (unsigned i = 1; i < lastIndex; ++i) {

        bool keep = false;

        const auto& time = times[i];
        const auto& timeNext = times[i + 1];

        // remove adjacent keyframes scheduled at the same time

        if (time != timeNext && (i != 1 || time != times[0])) {

            if (!smoothInterpolation) {

                // remove unnecessary keyframes same as their neighbors

                auto offset = i * stride,
                     offsetP = offset - stride,
                     offsetN = offset + stride;

                for (unsigned j = 0; j != stride; ++j) {

                    const auto value = values[offset + j];

                    if (value != values[offsetP + j] ||
                        value != values[offsetN + j]) {

                        keep = true;
                        break;
                    }
                }

            } else {

                keep = true;
            }
        }

        // in-place compaction

        if (keep) {

            if (i != writeIndex) {

                times[writeIndex] = times[i];

                const auto readOffset = i * stride,
                           writeOffset = writeIndex * stride;

                for (unsigned j = 0; j != stride; ++j) {

                    values[writeOffset + j] = values[readOffset + j];
                }
            }

            ++writeIndex;
        }
    }

    // flush last keyframe (compaction looks ahead)

    if (lastIndex > 0) {

        times[writeIndex] = times[lastIndex];

        for (unsigned readOffset = lastIndex * stride, writeOffset = writeIndex * stride, j = 0; j != stride; ++j) {

            values[writeOffset + j] = values[readOffset + j];
        }

        ++writeIndex;
    }

    if (writeIndex != times.size()) {

        this->times_ = AnimationUtils::arraySlice(times, 0, writeIndex);
        this->values_ = AnimationUtils::arraySlice(values, 0, writeIndex * stride);

    } else {

        this->times_ = times;
        this->values_ = values;
    }

    return *this;
}

std::unique_ptr<Interpolant> KeyframeTrack::createInterpolant(std::vector<float>* result) {

    switch (interpolation_) {
        case Interpolation::Linear:
            return std::make_unique<LinearInterpolant>(times_, values_, getValueSize(), result);
        case Interpolation::Smooth:
            return std::make_unique<CubicInterpolant>(times_, values_, getValueSize(), result);
        case Interpolation::Discrete:
            return std::make_unique<DiscreteInterpolant>(times_, values_, getValueSize(), result);
    }

    return nullptr;
}

void KeyframeTrack::setInterpolation(Interpolation interpolation) {

}
