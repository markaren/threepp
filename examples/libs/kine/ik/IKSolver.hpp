

#ifndef THREEPP_IKSOLVER_HPP
#define THREEPP_IKSOLVER_HPP

#include <array>
#include <optional>

#include "../Kine.hpp"

#include <threepp/math/Vector3.hpp>

namespace kine {

    class IKSolver {

        virtual std::vector<float> solveIK(const Kine& kine, const threepp::Vector3& target, const std::vector<float>& startValues) = 0;

        void setEPS(float eps) {
            eps_ = eps;
        }

    protected:
        float eps_{0.001f};
    };

}// namespace kine

#endif//THREEPP_IKSOLVER_HPP
