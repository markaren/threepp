
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

    template <size_t numDOF>
    class Kine {

    public:
        Kine(std::vector<std::shared_ptr<KineComponent>> components)
            : components_(std::move(components)) {

            for (unsigned i = 0, j = 0; i < components_.size(); ++i) {
                auto c = components_[i];
                auto joint = dynamic_cast<KineJoint *>(c.get());
                if (joint) {
                    joints_[j++] = joint;
                }
            }

        }

        [[nodiscard]] size_t numDof() const {

            return numDOF;
        }

        [[nodiscard]] threepp::Matrix4 calculateEndEffectorTransformation(const std::array<float, numDOF> &values) const {

            threepp::Matrix4 result;
            for (unsigned i = 0, j = 0; i < components_.size(); ++i) {
                auto &c = components_[i];
                if (dynamic_cast<KineJoint *>(c.get())) {
                    result.multiply(joints_[j]->getTransformation(values[j++]));
                } else {
                    result.multiply(c->getTransformation());
                }
            }
            return result;
        }

        linalg::mat<float, 3, numDOF> computeJacobian(const std::array<float, numDOF> &values) const {

            constexpr float h = 0.0001f;// some low value

            linalg::mat<float, 3, numDOF> jacobian{};
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

        [[nodiscard]] const std::array<KineJoint *, numDOF> &joints() const {
            return joints_;
        }

        [[nodiscard]] std::array<KineLimit, numDOF> limits() const {
            std::array<KineLimit, numDOF> limits;
            for (unsigned i = 0; i < numDOF; i++) {
                limits[i] = joints_[i]->limit;
            }
            return limits;
        }

        [[nodiscard]] std::array<float, numDOF> meanAngles() const {
            auto lim = limits();
            std::array<float, numDOF> res{};
            for (unsigned i = 0; i < numDOF; ++i) {
                res[i] = lim[i].mean();
            }
            return res;
        }

    private:
        std::array<KineJoint *, numDOF> joints_;
        std::vector<std::shared_ptr<KineComponent>> components_;
    };

    class KineBuilder {

    public:
        KineBuilder &addRevoluteJoint(const threepp::Vector3 &axis, const KineLimit &limit) {
            components_.emplace_back(std::make_unique<RevoluteJoint>(axis, limit));
            ++nDOF;

            return *this;
        }

        KineBuilder &addPrismaticJoint(const threepp::Vector3 &axis, const KineLimit &limit) {
            components_.emplace_back(std::make_unique<PrismaticJoint>(axis, limit));
            ++nDOF;

            return *this;
        }

        KineBuilder &addLink(const threepp::Vector3 &axis) {
            components_.emplace_back(std::make_unique<KineLink>(axis));

            return *this;
        }

        template<size_t nDOF>
        Kine<nDOF> build() {

            if (nDOF != this->nDOF) throw std::runtime_error("");

            return Kine<nDOF>(components_);
        }

    private:
        size_t nDOF{};
        std::vector<std::shared_ptr<KineComponent>> components_;
    };

}// namespace kine

#endif//THREEPP_KINE_HPP
