
#ifndef THREEPP_KINE_HPP
#define THREEPP_KINE_HPP

#include <vector>

#include "KineComponent.hpp"
#include "KineLink.hpp"

#include "joints/KineJoint.hpp"
#include "joints/PrismaticJoint.hpp"
#include "joints/RevoluteJoint.hpp"

#include "../linalg.h"

namespace kine {

    class Kine {

    public:
        Kine &addComponent(std::unique_ptr<KineComponent> c) {

            auto j = dynamic_cast<KineJoint *>(c.get());
            if (j) joints_.emplace_back(dynamic_cast<KineJoint *>(c.get()));

            components_.emplace_back(std::move(c));

            return *this;
        }

        [[nodiscard]] size_t numDof() const {

            return joints_.size();
        }

        template<size_t numDof>
        [[nodiscard]] threepp::Matrix4 calculateEndEffectorTransformation(const std::array<float, numDof> &values) const {

            if (numDof != this->numDof()) throw std::runtime_error("Wrong template size");

            threepp::Matrix4 result;
            for (unsigned i = 0, j = 0; i < components_.size(); ++i) {
                auto& c = components_[i];
                if (dynamic_cast<KineJoint*>(c.get())) {
                    result.multiply(joints_[j]->getTransformation(values[j++]));
                } else {
                    result.multiply(c->getTransformation());
                }

            }
            return result;
        }

        template<size_t numDof>
        linalg::mat<float, 3, numDof> computeJacobian(const std::array<float, numDof> &values) const {

            if (numDof != this->numDof()) throw std::runtime_error("Wrong template size");

            constexpr float h = 0.0001f;// some low value

            linalg::mat<float, 3, numDof> jacobian{};
            auto d1 = calculateEndEffectorTransformation(values);

            for (int i = 0; i < 3; ++i) {
                auto vals = values;// copy
                vals[i] += h;
                auto d2 = calculateEndEffectorTransformation(vals);


                jacobian.x[i] = (d2[12] - d1[12]) / h;
                jacobian.y[i] = (d2[13] - d1[13]) / h;
                jacobian.z[i] = (d2[14] - d1[14]) / h;
            }
            return jacobian;
        }

        [[nodiscard]] const std::vector<KineJoint *> &joints() const {
            return joints_;
        }

        [[nodiscard]] std::vector<KineLimit> limits() const {
            std::vector<KineLimit> limits;
            for (const auto j : joints_) {
                limits.emplace_back(j->limit);
            }
            return limits;
        }

        [[nodiscard]] const std::vector<std::unique_ptr<KineComponent>> &components() const {
            return components_;
        }


    private:
        std::vector<KineJoint *> joints_;
        std::vector<std::unique_ptr<KineComponent>> components_;
    };

}// namespace kine

#endif//THREEPP_KINE_HPP
