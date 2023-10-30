
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
            const auto pp = this->parameterPositions;
            auto iPrev = i1 - 2;
            auto iNext = i1 + 1;

            auto tPrev = pp[iPrev],
                 tNext = pp[iNext];

            if (std::isnan(tPrev)) {

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

            if (std::isnan(tNext)) {

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

        InterpolantSettings DefaultSettings_;
    };

}// namespace threepp

#endif//THREEPP_CUBICINTERPOLANT_HPP
