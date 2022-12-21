

#ifndef THREEPP_IKSOLVER_HPP
#define THREEPP_IKSOLVER_HPP

#include <array>
#include <optional>

#include "../Kine.hpp"

#include <threepp/math/Vector3.hpp>

namespace kine {

    template <size_t numDof>
    class IKSolver {

        virtual std::array<float, numDof> solveIK(const Kine<numDof>& kine, const threepp::Vector3& target, const std::array<float, numDof>& startValues) = 0;

        void setEPS(float eps) {
            eps_ = eps;
        }

    protected:
        float eps_{0.001f};

    };

}

#endif//THREEPP_IKSOLVER_HPP
