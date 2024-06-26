
#ifndef THREEPP_ROBOT_HPP
#define THREEPP_ROBOT_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>

namespace threepp {

    struct JointRange {
        float min;
        float max;

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
        std::optional<JointRange> range;

        std::string parent;
        std::string child;
    };

    class Robot: public Object3D {

    public:
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
            if (info.type != JointType::Fixed) {
                size_t idx = joints_.size() -1;
                articulatedJoints_.emplace(idx, std::make_pair(joint.get(), info));
                origPose_.emplace(idx, std::make_pair(joint->position.clone(), joint->quaternion.clone()));
            }
        }

        Matrix4 computeEndEffectorTransform(const std::vector<float>& values, bool deg = false) {
            Matrix4 result;

            for (auto i = 0, j = 0; i < joints_.size(); ++i) {

                const auto& joint = joints_.at(i);
                const auto& info = jointInfos_.at(i);

                switch (info.type) {

                    case JointType::Fixed: {
                        result.multiply(*joint->matrix);
                        break;
                    }
                    case JointType::Revolute: {
                        auto jointTransform = Matrix4()
                                                      .makeRotationFromQuaternion(origPose_.at(j).second)
                                                      .setPosition(origPose_.at(j).first);
                        auto value = values[j++];
                        value = deg ? math::degToRad(value) : value;
                        if (info.range.has_value()) {
                            value = info.range->clamp(value);
                        }

                        result.multiply(jointTransform.multiply(Matrix4().makeRotationAxis(info.axis, value)));
                        break;
                    }
                    case JointType::Prismatic: {
                        auto jointTransform = Matrix4()
                                                      .makeRotationFromQuaternion(origPose_.at(j).second)
                                                      .setPosition(origPose_.at(j).first);
                        auto value = values[j++];
                        result.multiply(jointTransform.multiply(Matrix4().makeTranslation(info.axis.clone().multiplyScalar(value))));
                        break;
                    }
                }
            }

            return result.premultiply(*matrix);
        }

        void finalize() {
            for (auto i = 0; i < joints_.size(); i++) {
                const auto info = jointInfos_[i];
                const auto joint = joints_[i];

                auto parent = std::ranges::find_if(links_, [&](auto link) {
                    return link->name == info.parent;
                });

                auto child = std::ranges::find_if(links_, [&](auto link) {
                    return link->name == info.child;
                });

                if (parent != links_.end() && child != links_.end()) {
                    (*parent)->add(*joint);
                    (joint)->add(**child);
                }
            }

            for (const auto& l : links_) {

                if (!l->parent) add(*l);
            }

            jointValues_.resize(numDOF());
        }

        void setJointValues(const std::vector<float>& values) {
            for (auto i = 0; i < values.size(); ++i) {
                setJointValue(i, values[i]);
            }
        }

        void setJointValue(int index, float value, bool deg = false) {

            const auto& joint = articulatedJoints_.at(index).first;
            const auto& info = articulatedJoints_.at(index).second;

            const auto& origPos = origPose_.at(index).first;
            const auto& origQuat = origPose_.at(index).second;

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
            }
        }

        [[nodiscard]] size_t numDOF() const {

            return articulatedJoints_.size();
        }

        [[nodiscard]] const std::vector<float>& jointValues() const {

            return jointValues_;
        }

        [[nodiscard]] std::vector<float> jointValuesWithConversionFromRadiansToDeg() const {

            std::vector<float> values = jointValues_;
            for (auto i = 0; i < numDOF(); i++) {
                const auto type = articulatedJoints_.at(i).second.type;
                if (type == JointType::Revolute) {
                    values[i] = math::radToDeg(jointValues_[i]);
                }
            }

            return values;
        }

        [[nodiscard]] std::pair<float, float> getJointRange(int index, bool deg = false) const {
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

        [[nodiscard]] std::vector<JointInfo> getArticulatedJointInfo() const {
            std::vector<JointInfo> info(numDOF());
            for (auto i = 0; i < numDOF(); i++) {
                info[i] = articulatedJoints_.at(i).second;
            }
            return info;
        }

        [[nodiscard]] float getJointValue(int index, bool deg = false) const {
            const auto& info = articulatedJoints_.at(index).second;
            if (deg && info.type == JointType::Revolute) {
                return math::radToDeg(jointValues_[index]);
            }
            return jointValues_[index];
        }

    private:
        std::vector<JointInfo> jointInfos_;
        std::vector<std::shared_ptr<Object3D>> links_;
        std::vector<std::shared_ptr<Object3D>> colliders_;
        std::vector<std::shared_ptr<Object3D>> joints_;
        std::unordered_map<size_t, std::pair<Object3D*, JointInfo>> articulatedJoints_;
        std::unordered_map<size_t, std::pair<Vector3, Quaternion>> origPose_;

        std::vector<float> jointValues_;
    };

}// namespace threepp

#endif//THREEPP_ROBOT_HPP
