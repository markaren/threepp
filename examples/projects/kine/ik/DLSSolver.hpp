
#ifndef THREEPP_DLSSOLVER_HPP
#define THREEPP_DLSSOLVER_HPP

#include "IKSolver.hpp"

namespace kine {

    template <size_t numDof>
    class DLSSolver: public IKSolver<numDof> {

    public:
        explicit DLSSolver(float lambda = 0.5f)
            : lambdaSq_(lambda*lambda) {}

        std::array<float, numDof> solveIK(const Kine<numDof>& kine, const threepp::Vector3& target, std::optional<std::array<float, numDof>> values) override {

            auto vals = values ? *values : std::array<float, numDof>{};

            threepp::Vector3 tmp;
            for (int i = 0; i < 100; ++i) {

                auto j = kine.computeJacobian(vals);
                auto m = kine.calculateEndEffectorTransformation(vals);
                auto& actual = tmp.setFromMatrixPosition(m);

                if (actual.distanceTo(target) < this->eps_) break;

                auto delta = linalg::vec<float, 3>{target.x - actual.x, target.y - actual.y, target.z - actual.z};
                linalg::mat<float, 3, numDof> inv = DLS(j);

                linalg::vec<float, numDof> theta_dot = linalg::mul(inv, delta);

                for (int k = 0; k < numDof; ++k) {

                    vals[k] += theta_dot[k];
                    kine.joints()[k]->limit.clamp(vals[k]);
                }
            }

            return vals;

        }

    private:
        float lambdaSq_;

        linalg::mat<float, 3, numDof> DLS(const linalg::mat<float, 3, 3> &j) {

            linalg::mat<float, 3, numDof> I = linalg::identity;

            auto jt = linalg::transpose(j);
            auto jjt = linalg::mul(j, jt);
            auto plus = linalg::cmul(I, lambdaSq_);

            return linalg::mul(jt, linalg::inverse(jjt + plus));
        }

    };

}

#endif//THREEPP_DLSSOLVER_HPP
