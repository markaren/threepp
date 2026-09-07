
#ifndef THREEPP_CUBICINTERPOLANT_HPP
#define THREEPP_CUBICINTERPOLANT_HPP

#include "threepp/math/Interpolant.hpp"

namespace threepp {

    class CubicInterpolant: public Interpolant {

    public:
        template<typename... Args>
        explicit CubicInterpolant(Args&&... args): Interpolant(std::forward<Args>(args)...) {

            DefaultSettings_ = InterpolantSettings{
                    Ending::ZeroCurvature,
                    Ending::ZeroCurvature};
        }

        void intervalChanged_(size_t i1, float t0, float t1) override {
            // Reference, not a copy: this ran on every interval change and was
            // duplicating the whole key-time array each time.
            const auto& pp = this->parameterPositions;
            const size_t n = pp.size();

            // three.js finds the ends of the curve by reading pp[i1-2] and
            // pp[i1+1] and testing the result for `undefined`, which JS returns
            // for an out-of-range index. Ported literally, those are OUT-OF-BOUNDS
            // vector reads: i1-2 underflows size_t to SIZE_MAX on the first
            // interval, and i1+1 runs one past the end on the last. The values
            // they produced were essentially never NaN, so the boundary branches
            // below never fired and the end intervals computed their weights from
            // adjacent heap memory — giving values in the thousands, and NaN when
            // the bogus tPrev happened to equal t0. Detect the ends by index
            // instead; the NaN checks were only ever standing in for that.
            const bool hasPrev = i1 >= 2;
            const bool hasNext = i1 + 1 < n;

            size_t iPrev = hasPrev ? i1 - 2 : i1;
            size_t iNext = hasNext ? i1 + 1 : i1;

            float tPrev = hasPrev ? pp[iPrev] : 0.f;
            float tNext = hasNext ? pp[iNext] : 0.f;

            if (!hasPrev) {

                switch (this->getSettings_()->endingStart) {

                    case Ending::ZeroSlope:

                        // f'(t0) = 0
                        iPrev = i1;
                        tPrev = 2 * t0 - t1;

                        break;

                    case Ending::WrapAround:

                        // use the other end of the curve
                        iPrev = pp.size() - 2;
                        tPrev = t0 + pp[iPrev] - pp[iPrev + 1];

                        break;

                    default:// ZeroCurvatureEnding

                        // f''(t0) = 0 a.k.a. Natural Spline
                        iPrev = i1;
                        tPrev = t1;
                }
            }

            if (!hasNext) {

                switch (this->getSettings_()->endingEnd) {

                    case Ending::ZeroSlope:

                        // f'(tN) = 0
                        iNext = i1;
                        tNext = 2 * t1 - t0;

                        break;

                    case Ending::WrapAround:

                        // use the other end of the curve
                        iNext = 1;
                        tNext = t1 + pp[1] - pp[0];

                        break;

                    default:// ZeroCurvatureEnding

                        // f''(tN) = 0, a.k.a. Natural Spline
                        iNext = i1 - 1;
                        tNext = t0;
                }
            }

            const auto halfDt = (t1 - t0) * 0.5f;
            const auto stride = this->valueSize;

            this->_weightPrev = halfDt / (t0 - tPrev);
            this->_weightNext = halfDt / (tNext - t1);
            this->_offsetPrev = static_cast<float>(iPrev * stride);
            this->_offsetNext = static_cast<float>(iNext * stride);
        }

        Sample interpolate_(size_t i1, float t0, float t, float t1) override {
            const auto result = this->resultBuffer;
            const auto values = this->sampleValues;
            const auto stride = this->valueSize;

            const auto o1 = i1 * stride, o0 = o1 - stride;
            const auto oP = this->_offsetPrev, oN = this->_offsetNext;
            const auto wP = this->_weightPrev, wN = this->_weightNext;

            const auto p = (t - t0) / (t1 - t0),
                       pp = p * p,
                       ppp = pp * p;

            // evaluate polynomials

            const auto sP = -wP * ppp + 2 * wP * pp - wP * p;
            const auto s0 = (1 + wP) * ppp + (-1.5 - 2 * wP) * pp + (-0.5 + wP) * p + 1;
            const auto s1 = (-1 - wN) * ppp + (1.5 + wN) * pp + 0.5 * p;
            const auto sN = wN * ppp - wN * pp;

            // combine data linearly

            for (auto i = 0; i != stride; ++i) {

                result->at(i) =
                        sP * values[oP + i] +
                        s0 * values[o0 + i] +
                        s1 * values[o1 + i] +
                        sN * values[oN + i];
            }

            return *result;
        }

    private:
        float _weightPrev = -0;
        float _offsetPrev = -0;
        float _weightNext = -0;
        float _offsetNext = -0;

        // NB: no DefaultSettings_ member here. There used to be one, which
        // SHADOWED Interpolant::DefaultSettings_ — so the constructor's
        // ZeroCurvature assignment landed on the derived copy while
        // getSettings_() kept reading the base's still-empty optional.
    };

}// namespace threepp

#endif//THREEPP_CUBICINTERPOLANT_HPP
