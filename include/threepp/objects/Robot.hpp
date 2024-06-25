
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
            for (int i = 0; i < joints_.size(); i++) {
                auto info = jointInfos_[i];
                auto joint = joints_[i];

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

    private:
        std::vector<JointInfo> jointInfos_;
        std::vector<std::shared_ptr<Object3D>> links_;
        std::vector<std::shared_ptr<Object3D>> joints_;
        std::unordered_map<int, std::pair<Object3D*, JointInfo>> articulatedJoints_;
    };

}// namespace threepp

#endif//THREEPP_ROBOT_HPP
