

#ifndef THREEPP_CCDSOLVER_HPP
#define THREEPP_CCDSOLVER_HPP

#include "IKSolver.hpp"

namespace kine {

    class CCDSolver: public IKSolver {

    public:
        explicit CCDSolver(unsigned int maxTries = 10, float stepSize = 0.005f, float eps = 0.00001f)
            : maxTries(maxTries),
              stepSize(stepSize),
              eps(eps) {}

        std::vector<float> solveIK(const Kine& kine, const Vector3& target, const std::vector<float>& startValues) override {

            Vector3 endPos;
            endPos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(startValues));

            float error = endPos.distanceTo(target);
            if (error < eps) return startValues;

            unsigned int tries = 0;
            std::vector<float> newValues = kine.normalizeValues(startValues);
            while (true) {

                for (unsigned i = 0; i < kine.numDof(); ++i) {
                    float closest = std::numeric_limits<float>::max();
                    float k = 0;
                    float val = 0;

                    do {
                        newValues[i] = k;
                        endPos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(kine.denormalizeValues(newValues)));

                        error = endPos.distanceTo(target);
                        if (error < closest) {
                            closest = error;
                            val = k;
                        }
                    } while ((k += stepSize) <= 1);

                    newValues[i] = val;
                }
                if (++tries >= maxTries || error < eps) break;
            }

            return kine.denormalizeValues(newValues);
        }

    private:
        unsigned int maxTries;
        float stepSize;
        float eps;
    };

}// namespace kine

#endif//THREEPP_CCDSOLVER_HPP
