
#ifndef THREEPP_KINE_HPP
#define THREEPP_KINE_HPP

#include <vector>

#include "KineComponent.hpp"
#include "KineLink.hpp"

#include "joints/KineJoint.hpp"
#include "joints/PrismaticJoint.hpp"
#include "joints/RevoluteJoint.hpp"

namespace kine {

    class Kine {

    public:
        explicit Kine(std::vector<std::unique_ptr<KineComponent>> components)
            : components_(std::move(components)) {

            for (const auto& c : components_) {
                auto joint = dynamic_cast<KineJoint*>(c.get());
                if (joint) {
                    joints_.emplace_back(joint);
                }
            }
        }

        [[nodiscard]] size_t numDof() const {

            return joints_.size();
        }

        [[nodiscard]] threepp::Matrix4 calculateEndEffectorTransformation(const std::vector<float>& values) const {

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

        [[nodiscard]] const std::vector<KineJoint*>& joints() const {
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
        std::vector<KineJoint*> joints_;
        std::vector<std::unique_ptr<KineComponent>> components_;
    };

    class KineBuilder {

    public:
        KineBuilder& addRevoluteJoint(const threepp::Vector3& axis, const KineLimit& limit) {
            components_.emplace_back(std::make_unique<RevoluteJoint>(axis, limit));

            return *this;
        }

        KineBuilder& addPrismaticJoint(const threepp::Vector3& axis, const KineLimit& limit) {
            components_.emplace_back(std::make_unique<PrismaticJoint>(axis, limit));

            return *this;
        }

        KineBuilder& addLink(const threepp::Vector3& axis) {
            components_.emplace_back(std::make_unique<KineLink>(axis));

            return *this;
        }

        Kine build() {

            return Kine{std::move(components_)};
        }

    private:
        std::vector<std::unique_ptr<KineComponent>> components_;
    };

}// namespace kine

#endif//THREEPP_KINE_HPP
