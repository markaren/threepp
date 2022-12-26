
#ifndef THREEPP_KINE_HPP
#define THREEPP_KINE_HPP

#include <vector>

#include "KineComponent.hpp"
#include "KineLink.hpp"

#include "joints/KineJoint.hpp"
#include "joints/PrismaticJoint.hpp"
#include "joints/RevoluteJoint.hpp"

#include "Eigen/Dense"

namespace kine {

    class Kine {

    public:
        Kine(std::vector<std::shared_ptr<KineComponent>> components)
            : components_(std::move(components)) {

            for (const auto& c : components_) {
                auto joint = dynamic_cast<KineJoint *>(c.get());
                if (joint) {
                    joints_.emplace_back(joint);
                }
            }

        }

        [[nodiscard]] size_t numDof() const {

            return joints_.size();
        }

        [[nodiscard]] threepp::Matrix4 calculateEndEffectorTransformation(const std::vector<float> &values) const {

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

        [[nodiscard]] Eigen::MatrixX<double> computeJacobian(const std::vector<float> &values) const {

            constexpr double h = 0.0001;// some low value

            Eigen::MatrixX<double> jacobian(3, numDof());
            auto d1 = calculateEndEffectorTransformation(values);

            for (int i = 0; i < 3; ++i) {
                auto vals = values;// copy
                vals[i] += h;
                auto d2 = calculateEndEffectorTransformation(vals);

                jacobian(0, i) = (d2[12] - d1[12]) / h;
                jacobian(1, i) = (d2[13] - d1[13]) / h;
                jacobian(2, i) = (d2[14] - d1[14]) / h;
            }
            return jacobian;
        }

        [[nodiscard]] const std::vector<KineJoint *> &joints() const {
            return joints_;
        }

        [[nodiscard]] std::vector<KineLimit> limits() const {
            std::vector<KineLimit> limits;
            for (unsigned i = 0; i < numDof(); i++) {
                limits.emplace_back(joints_[i]->limit);
            }
            return limits;
        }

        [[nodiscard]] std::vector<float> meanAngles() const {
            auto lim = limits();
            std::vector<float> res(numDof());
            for (unsigned i = 0; i < numDof(); ++i) {
                res[i] = lim[i].mean();
            }
            return res;
        }

        [[nodiscard]] std::vector<float> normalizeValues(const std::vector<float>& values) const {
            std::vector<float> res(numDof());
            for (unsigned i = 0; i < numDof(); ++i) {
                res[i] = joints_[i]->limit.normalize(values[i]);
            }
            return res;
        }

        [[nodiscard]] std::vector<float> denormalizeValues(const std::vector<float>& values) const {
            std::vector<float> res(numDof());
            for (unsigned i = 0; i < numDof(); ++i) {
                res[i] = joints_[i]->limit.denormalize(values[i]);
            }
            return res;
        }

    private:
        std::vector<KineJoint *> joints_;
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

        Kine build() {

            return components_;
        }

    private:
        size_t nDOF{};
        std::vector<std::shared_ptr<KineComponent>> components_;
    };

}// namespace kine

#endif//THREEPP_KINE_HPP
