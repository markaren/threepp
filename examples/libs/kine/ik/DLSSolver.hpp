
#ifndef THREEPP_DLSSOLVER_HPP
#define THREEPP_DLSSOLVER_HPP

#include "IKSolver.hpp"

#include "Eigen/Dense"

namespace kine {

    class DLSSolver: public IKSolver {

    public:
        explicit DLSSolver(double lambda = 0.5)
            : lambdaSq_(lambda * lambda) {}

        std::vector<float> solveIK(const Kine& kine, const threepp::Vector3& target, const std::vector<float>& startValues) override {

            auto vals = startValues;

            float error;
            threepp::Vector3 vec;
            Eigen::MatrixX<double> mat;
            for (int i = 0; i < 100; ++i) {

                auto j = computeJacobian(kine, vals);

                auto m = kine.calculateEndEffectorTransformation(vals);
                auto& actual = vec.setFromMatrixPosition(m);

                error = actual.distanceTo(target);
                if (error < this->eps_) break;

                Eigen::Vector3<double> delta{target.x - actual.x, target.y - actual.y, target.z - actual.z};

                auto inv = DLS(j);
                auto theta_dot = inv * delta;

                for (int k = 0; k < kine.numDof(); ++k) {

                    vals[k] += static_cast<float>(theta_dot[k]);
                    kine.joints()[k]->limit.clampWithinLimit(vals[k]);
                }
            }

            return vals;
        }

    private:
        double lambdaSq_;

        Eigen::MatrixX<double> DLS(const Eigen::MatrixX<double>& j) {

            Eigen::MatrixX<double> I(j.rows(), j.rows());
            I.setIdentity();

            auto jt = j.transpose();
            auto jjt = j * jt;
            auto plus = I * lambdaSq_;

            return jt * ((jjt + plus).inverse());
        }

        [[nodiscard]] Eigen::MatrixX<double> computeJacobian(const Kine& kine, const std::vector<float>& values) const {

            constexpr double h = 0.0001;// some low value

            Eigen::MatrixX<double> jacobian(3, kine.numDof());
            auto d1 = kine.calculateEndEffectorTransformation(values);

            for (int i = 0; i < 3; ++i) {
                auto vals = values;// copy
                vals[i] += h;
                auto d2 = kine.calculateEndEffectorTransformation(vals);

                jacobian(0, i) = (d2[12] - d1[12]) / h;
                jacobian(1, i) = (d2[13] - d1[13]) / h;
                jacobian(2, i) = (d2[14] - d1[14]) / h;
            }
            return jacobian;
        }
    };

}// namespace kine

#endif//THREEPP_DLSSOLVER_HPP
