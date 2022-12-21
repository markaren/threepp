

#ifndef THREEPP_CCDSOLVER_HPP
#define THREEPP_CCDSOLVER_HPP

#include "IKSolver.hpp"

namespace kine {

    template<size_t numDof>
    class CCDSolver : public IKSolver<numDof> {

    public:
        explicit CCDSolver(int maxTries = 10, float stepSize = 0.005f)
            : maxTries(maxTries),
              stepSize(stepSize) {}

        std::array<float, numDof> solveIK(const Kine<numDof> &kine, const threepp::Vector3 &target, const std::array<float, numDof>& startValues) {

            threepp::Vector3 endPos;
            endPos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(startValues));

            if (endPos.distanceTo(target) < 0.001f) return startValues;

            int tries = 0;
            std::array<float, numDof> newValues = kine.normalizeValues(startValues);
            while(true) {

                for (unsigned i = 0; i < numDof; ++i) {
                    float closest = std::numeric_limits<float>::max();
                    float k = 0;
                    float val = 0;

                    do {
                        newValues[i] = k;
                        endPos.setFromMatrixPosition(kine.calculateEndEffectorTransformation(kine.denormalizeValues(newValues)));

                        float error = endPos.distanceTo(target);
                        if (error < closest) {
                            closest = error;
                            val = k;
                        }
                    } while ((k += stepSize) <= 1);

                    newValues[i] = val;

                }
                if (++tries >= maxTries) break ;
            }

            return kine.denormalizeValues(newValues);
        }

    private:
        int maxTries;
        float stepSize;
    };

}// namespace kine

#endif//THREEPP_CCDSOLVER_HPP
