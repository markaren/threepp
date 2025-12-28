
#ifndef THREEPP_ROBOT_HPP
#define THREEPP_ROBOT_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>

namespace threepp {

    class Robot: public Object3D {

    public:
        struct JointRange {
            float min;
            float max;

            [[nodiscard]] float mid() const {
                return (min + max) / 2;
            }

            [[nodiscard]] float clamp(float value) const {
                return std::clamp(value, min, max);
            }
        };

        enum class JointType {
            Revolute,
            Prismatic,
            Fixed
        };

        struct JointInfo {
            Vector3 axis;
            JointType type;
            std::string name;
            std::optional<JointRange> range;

            std::string parent;
            std::string child;
        };

        Robot() = default;

        void showColliders(bool flag) {
            for (auto& c : links_) {
                c->traverse([&](auto& obj) {
                    if (obj.userData.contains("collider")) {
                        obj.visible = flag;
                    }
                });
            }
        }

        void addLink(const std::shared_ptr<Object3D>& link) {
            links_.emplace_back(link);
        }

        void addJoint(const std::shared_ptr<Object3D>& joint, const JointInfo& info) {
            joints_.emplace_back(joint);
            jointInfos_.emplace_back(info);
            origPose_.emplace(std::make_pair(joint.get(), std::make_pair(joint->position.clone(), joint->quaternion.clone())));
            if (info.type != JointType::Fixed) {
                articulatedJoints_.emplace_back(joint.get(), info);
            }
        }

        [[nodiscard]] Matrix4 getEndEffectorTransform() const {
            auto end = joints_.back();
            end->updateMatrixWorld(true);
            return *end->matrixWorld;
        }

        [[nodiscard]] Matrix4 computeEndEffectorTransform(const std::vector<float>& values, bool deg = false, bool enforceLimits = true) const {
            Matrix4 result;

            for (unsigned i = 0, j = 0; i < joints_.size(); ++i) {

                const auto& joint = joints_.at(i);
                const auto& info = jointInfos_.at(i);

                auto jointTransform = Matrix4()
                                              .makeRotationFromQuaternion(origPose_.at(joint.get()).second)
                                              .setPosition(origPose_.at(joint.get()).first);

                switch (info.type) {

                    case JointType::Fixed: {
                        result.multiply(jointTransform);
                        break;
                    }
                    case JointType::Revolute: {
                        auto value = values[j++];
                        value = deg ? math::degToRad(value) : value;
                        if (enforceLimits && info.range.has_value()) {
                            value = info.range->clamp(value);
                        }

                        result.multiply(jointTransform.multiply(Matrix4().makeRotationAxis(info.axis, value)));
                        break;
                    }
                    case JointType::Prismatic: {
                        auto value = values[j++];
                        if (enforceLimits && info.range.has_value()) {
                            value = info.range->clamp(value);
                        }
                        result.multiply(jointTransform.multiply(Matrix4().makeTranslation(info.axis.clone().multiplyScalar(value))));
                        break;
                    }
                }
            }

            return result.premultiply(*matrix);
        }

        void finalize() {
            for (unsigned i = 0; i < joints_.size(); i++) {
                const auto& info = jointInfos_[i];
                const auto& joint = joints_[i];

                auto parent = std::ranges::find_if(links_, [&](const auto& link) {
                    return link->name == info.parent;
                });

                auto child = std::ranges::find_if(links_, [&](const auto& link) {
                    return link->name == info.child;
                });

                if (parent != links_.end() && child != links_.end()) {
                    (*parent)->add(*joint);
                    (joint)->add(**child);
                }
            }

            add(links_.front());

            jointValues_.resize(numDOF());
        }

        void setJointValues(const std::vector<float>& values, float deg = false) {
            for (size_t i = 0; i < values.size(); ++i) {
                setJointValue(i, values[i], deg);
            }
        }

        void setJointValue(size_t index, float value, bool deg = false) {

            const auto& joint = articulatedJoints_.at(index).first;
            const auto& info = articulatedJoints_.at(index).second;

            const auto& origPos = origPose_.at(joint).first;
            const auto& origQuat = origPose_.at(joint).second;

            switch (info.type) {
                case JointType::Revolute: {
                    value = deg ? math::degToRad(value) : value;
                    if (info.range.has_value()) {
                        value = info.range->clamp(value);
                    }
                    joint->quaternion.setFromAxisAngle(info.axis, value);
                    joint->quaternion.premultiply(origQuat);
                    jointValues_[index] = value;
                    break;
                }
                case JointType::Prismatic: {
                    static Vector3 tempAxis;
                    tempAxis.copy(info.axis).applyEuler(rotation);
                    joint->position.copy(origPos).addScaledVector(tempAxis, value);
                    jointValues_[index] = value;
                    break;
                }
                default:
                    break;
            }
        }

        [[nodiscard]] size_t numDOF() const {

            return articulatedJoints_.size();
        }

        [[nodiscard]] std::vector<float> jointValues(bool convertToDeg = false) const {

            if (!convertToDeg) {
                return jointValues_;
            }
            std::vector<float> values = jointValues_;
            for (unsigned i = 0; i < numDOF(); i++) {
                const auto type = articulatedJoints_.at(i).second.type;
                if (type == JointType::Revolute) {
                    values[i] = math::radToDeg(jointValues_[i]);
                }
            }

            return values;
        }

        [[nodiscard]] JointRange getJointRange(size_t index, bool deg = false) const {
            const auto& info = articulatedJoints_.at(index).second;
            const auto type = articulatedJoints_.at(index).second.type;
            if (!info.range) {
                return {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
            }

            float min = info.range->min;
            float max = info.range->max;

            if (deg && type == JointType::Revolute) {
                min = math::radToDeg(min);
                max = math::radToDeg(max);
            }

            return {min, max};
        }

        [[nodiscard]] std::vector<JointRange> getJointRanges(bool deg = false) const {
            std::vector<JointRange> ranges(numDOF());
            for (unsigned i = 0; i < numDOF(); i++) {
                ranges[i] = getJointRange(i, deg);
            }
            return ranges;
        }

        [[nodiscard]] std::vector<JointInfo> getArticulatedJointInfo() const {
            std::vector<JointInfo> info(numDOF());
            for (unsigned i = 0; i < numDOF(); i++) {
                info[i] = articulatedJoints_.at(i).second;
            }
            return info;
        }

        [[nodiscard]] float getJointValue(size_t index, bool deg = false) const {
            const auto& info = articulatedJoints_.at(index).second;
            if (deg && info.type == JointType::Revolute) {
                return math::radToDeg(jointValues_[index]);
            }
            return jointValues_[index];
        }

    private:
        std::vector<float> jointValues_;

        std::vector<JointInfo> jointInfos_;
        std::vector<std::shared_ptr<Object3D>> links_;
        std::vector<std::shared_ptr<Object3D>> colliders_;
        std::vector<std::shared_ptr<Object3D>> joints_;
        std::vector<std::pair<Object3D*, JointInfo>> articulatedJoints_;
        std::unordered_map<Object3D* , std::pair<Vector3, Quaternion>> origPose_;
    };

}// namespace threepp

#endif//THREEPP_ROBOT_HPP
