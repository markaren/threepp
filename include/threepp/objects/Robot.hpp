
#ifndef THREEPP_ROBOT_HPP
#define THREEPP_ROBOT_HPP

#include "threepp/core/Object3D.hpp"

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
        Continuous,
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
                articulatedJoints_.emplace(joints_.size() - 1, std::make_pair(joint.get(), info));
            }
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
            minRange_.resize(numDOF());
            maxRange_.resize(numDOF());

            for (auto i = 0; i < numDOF(); i++) {
                const auto info = articulatedJoints_[i].second;
                if (info.range) {
                    minRange_[i] = info.range->min;
                    maxRange_[i] = info.range->max;
                } else {
                    minRange_[i] = std::numeric_limits<float>::infinity();
                    maxRange_[i] = std::numeric_limits<float>::infinity();
                }
            }
        }

        void setJointValues(std::vector<float> value) {
            for (auto i = 0; i < value.size(); ++i) {
                setJointValue(i, value[i]);
            }
        }


        void setJointValue(int index, float value) {

            auto joint = articulatedJoints_[index].first;
            auto info = articulatedJoints_[index].second;

            switch (info.type) {
                case JointType::Continuous:
                case JointType::Revolute: {
                    if (info.range.has_value()) {
                        value = info.range->clamp(value);
                    }
                    joint->quaternion.setFromAxisAngle(info.axis, value);
                    break;
                }
                case JointType::Prismatic: {
                    joint->position = info.axis * value;
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

        [[nodiscard]] const std::vector<float>& minJointValues() const {

            return minRange_;
        }

        [[nodiscard]] const std::vector<float>& maxJointValues() const {

            return maxRange_;
        }

    private:
        std::vector<JointInfo> jointInfos_;
        std::vector<std::shared_ptr<Object3D>> links_;
        std::vector<std::shared_ptr<Object3D>> colliders_;
        std::vector<std::shared_ptr<Object3D>> joints_;
        std::unordered_map<size_t, std::pair<Object3D*, JointInfo>> articulatedJoints_;

        std::vector<float> jointValues_;
        std::vector<float> minRange_;
        std::vector<float> maxRange_;
    };

}// namespace threepp

#endif//THREEPP_ROBOT_HPP
