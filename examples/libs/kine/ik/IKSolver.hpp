

#ifndef THREEPP_IKSOLVER_HPP
#define THREEPP_IKSOLVER_HPP

#include "../Kine.hpp"

namespace kine {

    class IKSolver {

    public:

        virtual std::vector<float> solveIK(const Kine& kine, const Vector3& target, const std::vector<float>& startValues) = 0;

        void setEPS(float eps) {
            eps_ = eps;
        }

        virtual ~IKSolver() = default;

    protected:
        float eps_{0.001f};
    };

}// namespace kine

#endif//THREEPP_IKSOLVER_HPP
