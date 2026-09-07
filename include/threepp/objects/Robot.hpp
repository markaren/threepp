
#ifndef THREEPP_ROBOT_HPP
#define THREEPP_ROBOT_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

        static std::shared_ptr<Robot> create() {

            return std::make_shared<Robot>();
        }

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
            addJoint(joint, info, joint->position, joint->quaternion);
        }

        // The same, with the joint's REST pose given rather than read off the
        // node. A document stores a robot in whatever pose it was saved in, so
        // a reader has the joint node's driven transform and not the one it has
        // at zero — and reading the driven pose as the rest pose bakes the saved
        // angles into the joint's own frame, doubling them the moment anything
        // drives it.
        void addJoint(const std::shared_ptr<Object3D>& joint, const JointInfo& info,
                      const Vector3& restPosition, const Quaternion& restRotation) {
            joints_.emplace_back(joint);
            jointInfos_.emplace_back(info);
            // clone(), not a plain copy: copying a Quaternion copies its
            // onChange callback too, which would leave the stored rest pose
            // holding a capture of a node it must not notify (and may outlive).
            origPose_.emplace(std::make_pair(joint.get(),
                                             std::make_pair(restPosition.clone(), restRotation.clone())));
            if (info.type != JointType::Fixed) {
                jointDof_.emplace_back(static_cast<int>(articulatedJoints_.size()));
                articulatedJoints_.emplace_back(joint.get(), info);
            } else {
                jointDof_.emplace_back(-1);
            }
        }

        // Designate which link's inbound joint frame is the end effector.
        //
        // A robot is a TREE, not a chain: an arm with a two-finger gripper
        // branches at the palm. Without this, "the end effector" can only mean
        // the last joint that happened to be added, and the FK below can only
        // mean the product of every joint in the file — which for a branched
        // robot multiplies both fingers into the tool pose.
        //
        // Pass an empty name to restore the default (the last joint added),
        // which is what a plain serial arm wants and what every caller got
        // before this existed.
        void setEndEffector(const std::string& linkName) {
            eeLink_ = linkName;
            rebuildChain();
        }

        [[nodiscard]] const std::string& endEffectorLink() const {
            return eeLink_;
        }

        // The DOF indices on the path from the root to the end effector, in
        // root-to-tip order. For a serial arm this is every DOF; for an arm
        // with a gripper it is the arm's joints only — precisely the set an IK
        // solver is allowed to move, since closing the fingers must not be a
        // way to reach further.
        [[nodiscard]] const std::vector<size_t>& chainDofs() const {
            return chainDofs_;
        }

        [[nodiscard]] Matrix4 getEndEffectorTransform() const {
            auto end = chain_.empty() ? joints_.back() : joints_.at(chain_.back());
            end->updateMatrixWorld(true);
            return *end->matrixWorld;
        }

        // Analytic FK to the end effector, without touching the scene graph.
        //
        // `values` is indexed by GLOBAL dof index — the same order as
        // jointValues() and numDOF() — not by position along the chain. That
        // matters for a branched robot: the fingers keep their slots in the
        // vector, they simply do not contribute to the tool pose. Indices past
        // the end of `values` read as zero, so passing just the leading arm
        // DOFs of an arm-plus-gripper robot does the sensible thing.
        [[nodiscard]] Matrix4 computeEndEffectorTransform(const std::vector<float>& values, bool deg = false, bool enforceLimits = true) const {
            Matrix4 result;

            // Walk the root-to-tip path. Before finalize() there is no path yet,
            // so fall back to every joint in insertion order.
            const bool walkAll = chain_.empty();
            const size_t count = walkAll ? joints_.size() : chain_.size();

            for (size_t k = 0; k < count; ++k) {

                const size_t i = walkAll ? k : chain_.at(k);

                const auto& joint = joints_.at(i);
                const auto& info = jointInfos_.at(i);

                auto jointTransform = Matrix4()
                                              .makeRotationFromQuaternion(origPose_.at(joint.get()).second)
                                              .setPosition(origPose_.at(joint.get()).first);

                if (info.type == JointType::Fixed) {
                    result.multiply(jointTransform);
                    continue;
                }

                const int dof = jointDof_.at(i);
                const auto slot = static_cast<size_t>(dof);
                auto value = slot < values.size() ? values[slot] : 0.f;

                switch (info.type) {

                    case JointType::Revolute: {
                        value = deg ? math::degToRad(value) : value;
                        if (enforceLimits && info.range.has_value()) {
                            value = info.range->clamp(value);
                        }

                        result.multiply(jointTransform.multiply(Matrix4().makeRotationAxis(info.axis, value)));
                        break;
                    }
                    case JointType::Prismatic: {
                        if (enforceLimits && info.range.has_value()) {
                            value = info.range->clamp(value);
                        }
                        result.multiply(jointTransform.multiply(Matrix4().makeTranslation(info.axis.clone().multiplyScalar(value))));
                        break;
                    }
                    case JointType::Fixed:
                        break;// handled above
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
                    (*parent)->addRef(*joint);
                    (joint)->addRef(**child);
                }
            }

            // Attach the ROOT link — the one no joint names as a child.
            //
            // This used to be links_.front(), i.e. whichever <link> came first
            // in the file. That holds for hand-written URDFs and fails for
            // xacro-expanded ones, which emit links in macro-expansion order.
            // When the first link is not the root it has already been parented
            // under its own inbound joint by the loop above, and Object3D::add()
            // unlinks a node from its current parent — so the call tore that
            // link back out and silently orphaned everything below it.
            if (!links_.empty()) {
                std::unordered_set<std::string> childLinks;
                for (const auto& info : jointInfos_) {
                    childLinks.insert(info.child);
                }

                auto root = std::ranges::find_if(links_, [&](const auto& link) {
                    return !childLinks.contains(link->name);
                });

                add(root != links_.end() ? *root : links_.front());
            }

            jointValues_.resize(numDOF());

            rebuildChain();
        }

        // finalize() for a robot whose hierarchy ALREADY exists — the one a
        // deserialiser needs.
        //
        // finalize() parents each link under its inbound joint and the root link
        // under the robot, which is right when the loader built the nodes loose
        // and wrong when they were read back as a tree: the hierarchy is already
        // the document's, and Object3D::add() unlinks and re-appends, so
        // re-parenting an existing child moves it to the end of its parent's
        // list — reordering nodes that an asset override table identifies by
        // position, and shuffling the viewport's own hierarchy for no reason.
        // Only the derived tables are rebuilt here.
        void finalizeInPlace() {
            jointValues_.resize(numDOF());

            rebuildChain();
        }

        // --- what a document needs to write this robot down -------------------
        // The joint table is not derivable from the transforms: axes, limits,
        // types and rest poses exist only here. Without them a saved robot can
        // only come back frozen, and the only way to re-articulate it is to
        // re-import the source file — which rebuilds the subtree and discards
        // whatever was authored into it.

        [[nodiscard]] const std::vector<std::shared_ptr<Object3D>>& links() const {
            return links_;
        }

        // Every joint, fixed ones included, parallel to jointInfos(). Not to be
        // confused with jointValues(), which is indexed by DOF.
        [[nodiscard]] const std::vector<std::shared_ptr<Object3D>>& jointNodes() const {
            return joints_;
        }

        [[nodiscard]] const std::vector<JointInfo>& jointInfos() const {
            return jointInfos_;
        }

        // The joint node's local transform with this joint at zero.
        [[nodiscard]] std::pair<Vector3, Quaternion> jointRestPose(size_t index) const {
            return origPose_.at(joints_.at(index).get());
        }

        // The DOF slot joint `index` drives, or -1 when it is fixed.
        [[nodiscard]] int jointDof(size_t index) const {
            return jointDof_.at(index);
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
                    if (info.range.has_value()) {
                        value = info.range->clamp(value);
                    }
                    // The axis is given in the JOINT's own frame while
                    // joint->position lives in the PARENT's, so the slide has to
                    // be rotated by the joint's original orientation.
                    //
                    // This used to be applyEuler(rotation) — the ROBOT's own
                    // rotation, which has nothing to do with this joint. With the
                    // robot at identity that is a no-op, so the axis stayed in the
                    // joint frame and any joint with an rpy origin slid along the
                    // wrong world direction; rotating the robot then steered the
                    // slide direction as well. computeEndEffectorTransform() has
                    // always applied origQuat here, so the two disagreed.
                    Vector3 axis = info.axis;
                    axis.applyQuaternion(origQuat);
                    joint->position.copy(origPos).addScaledVector(axis, value);
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
        // The default end effector: the far end of the LONGEST joint chain.
        //
        // The obvious default — the last joint declared — is wrong on real
        // URDFs. The KUKA iiwa declares a base-plate branch AFTER its tool
        // joint, so "last" names the robot's own pedestal, and FK to it walks a
        // single fixed joint and reports the base as the end effector. (Today's
        // flat product gets the right answer there only by luck: that branch
        // joint's origin happens to be identity, so multiplying it in changes
        // nothing.) Depth picks the tool on every serial arm and on the iiwa,
        // and ties break on declaration order so the result is deterministic
        // regardless of hash ordering.
        [[nodiscard]] std::string deepestLeaf(const std::unordered_map<std::string, size_t>& inbound,
                                              const std::unordered_set<std::string>& parents) const {
            std::string best;
            size_t bestDepth = 0;
            size_t bestJoint = std::numeric_limits<size_t>::max();

            for (const auto& [child, jointIndex] : inbound) {
                if (parents.contains(child)) continue;// has children of its own, so not a leaf

                size_t depth = 0;
                std::string link = child;
                std::unordered_set<std::string> seen;
                while (true) {
                    auto it = inbound.find(link);
                    if (it == inbound.end()) break;
                    if (!seen.insert(link).second) break;// cycle: malformed tree
                    ++depth;
                    link = jointInfos_[it->second].parent;
                }

                if (depth > bestDepth || (depth == bestDepth && jointIndex < bestJoint)) {
                    best = child;
                    bestDepth = depth;
                    bestJoint = jointIndex;
                }
            }

            return best.empty() ? jointInfos_.back().child : best;
        }

        // Resolve the root-to-tip joint path, walking the child->parent link
        // relations backwards from the end-effector link. Cheap, and only ever
        // runs on finalize() or an explicit setEndEffector().
        void rebuildChain() {
            chain_.clear();
            chainDofs_.clear();
            if (joints_.empty()) return;

            // child link name -> the joint that drives it. A link has exactly
            // one inbound joint in a well-formed URDF, so this is a function.
            std::unordered_map<std::string, size_t> inbound;
            std::unordered_set<std::string> parents;
            for (size_t i = 0; i < jointInfos_.size(); ++i) {
                inbound.emplace(jointInfos_[i].child, i);
                parents.insert(jointInfos_[i].parent);
            }

            if (eeLink_.empty()) {
                eeLink_ = deepestLeaf(inbound, parents);
            }

            std::string link = eeLink_;
            std::unordered_set<std::string> visited;
            while (true) {
                auto it = inbound.find(link);
                if (it == inbound.end()) break;// reached the root link
                if (!visited.insert(link).second) break;// cycle: malformed tree
                chain_.push_back(it->second);
                link = jointInfos_[it->second].parent;
            }

            std::ranges::reverse(chain_);

            for (const size_t i : chain_) {
                const int dof = jointDof_.at(i);
                if (dof >= 0) chainDofs_.push_back(static_cast<size_t>(dof));
            }
        }

        std::vector<float> jointValues_;

        std::vector<JointInfo> jointInfos_;
        std::vector<std::shared_ptr<Object3D>> links_;
        std::vector<std::shared_ptr<Object3D>> colliders_;
        std::vector<std::shared_ptr<Object3D>> joints_;
        std::vector<int> jointDof_;// parallel to joints_; dof index, or -1 if fixed
        std::vector<std::pair<Object3D*, JointInfo>> articulatedJoints_;
        std::unordered_map<Object3D* , std::pair<Vector3, Quaternion>> origPose_;

        std::string eeLink_;
        std::vector<size_t> chain_;    // joint indices, root -> tip
        std::vector<size_t> chainDofs_;// dof indices along that path
    };

}// namespace threepp

#endif//THREEPP_ROBOT_HPP
