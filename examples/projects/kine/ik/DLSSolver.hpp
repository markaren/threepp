
#ifndef THREEPP_DLSSOLVER_HPP
#define THREEPP_DLSSOLVER_HPP

#include "IKSolver.hpp"

namespace kine {

    class DLSSolver: public IKSolver {

    public:
        explicit DLSSolver(float lambda = 0.5f)
            : lambdaSq_(lambda*lambda) {}

        std::vector<float> solveIK(const Kine& kine, const threepp::Vector3& target, const std::vector<float>& startValues) override {

            auto vals = startValues;

            threepp::Vector3 tmp;
            for (int i = 0; i < 100; ++i) {

                auto j = kine.computeJacobian(vals);
                auto m = kine.calculateEndEffectorTransformation(vals);
                auto& actual = tmp.setFromMatrixPosition(m);

                if (actual.distanceTo(target) < this->eps_) break;

                Eigen::Vector3<float> delta{target.x - actual.x, target.y - actual.y, target.z - actual.z};

                auto inv = DLS(j);
                auto theta_dot = inv * delta;

                for (int k = 0; k < kine.numDof(); ++k) {

                    vals[k] += theta_dot[k];
                    kine.joints()[k]->limit.clampWithinLimit(vals[k]);
                }
            }

            return vals;

        }

    private:
        float lambdaSq_;

        Eigen::MatrixX<float> DLS(const Eigen::MatrixX<float> &j) {

            Eigen::MatrixX<float> I(j.rows(), j.cols());
            I.setIdentity();

            auto jt = j.transpose();
            auto jjt = j * jt;
            auto plus = I * lambdaSq_;

            return jt * (jjt + plus).inverse();
        }

    };

}

#endif//THREEPP_DLSSOLVER_HPP
