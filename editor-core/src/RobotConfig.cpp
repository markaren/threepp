
#include "threepp/extras/editor/RobotConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/objects/Robot.hpp"

#include <any>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    bool readBool(const Object3D& object, const char* key, bool fallback) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return fallback;
        if (it->second.type() != typeid(bool)) return fallback;
        return std::any_cast<bool>(it->second);
    }

    // Every argument value key on the object, whether or not the name list still
    // mentions it. Collected first and erased afterwards, because erasing while
    // walking userData would invalidate the walk.
    std::vector<std::string> argValueKeys(const Object3D& object) {

        const std::string prefix = xacro::argValueUserDataPrefix;

        std::vector<std::string> keys;
        for (const auto& [key, _] : object.userData) {
            if (key.size() > prefix.size() && key.compare(0, prefix.size(), prefix) == 0) {
                keys.push_back(key);
            }
        }
        return keys;
    }

}// namespace


// Same contract as the other editor configs: locale-independent, trailing zeros
// trimmed, byte-identical for an unchanged value so saved documents stay
// diff-clean.
std::string RobotConfig::encodeJoints(const std::vector<float>& values) {

    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += number(values[i]);
    }
    return out;
}

std::vector<float> RobotConfig::decodeJoints(const std::string& text) {

    std::vector<float> values;
    if (text.empty()) return values;

    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(',', start);
        const auto token = std::string_view(text).substr(
                start, (end == std::string::npos ? text.size() : end) - start);
        try {
            values.push_back(std::stof(std::string(token)));
        } catch (...) {
            // A malformed entry reads as zero rather than shifting every
            // joint after it onto the wrong axis.
            values.push_back(0.f);
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return values;
}

std::map<std::string, std::string> RobotConfig::argMap() const {

    std::map<std::string, std::string> args;
    for (const auto& [name, value] : xacroArgs) args[name] = value;
    return args;
}

std::optional<RobotConfig> RobotConfig::read(const Object3D& object) {

    const auto urdf = readString(object, urdfKey);
    if (urdf.empty()) return std::nullopt;

    RobotConfig config;
    config.urdf = urdf;
    config.joints = decodeJoints(readString(object, jointsKey));
    config.showColliders = readBool(object, collidersKey, false);

    config.xacroArgs = xacro::readArgsUserData(object);

    return config;
}

void RobotConfig::write(Object3D& object) const {

    if (urdf.empty()) {
        erase(object);
        return;
    }
    object.userData[urdfKey] = urdf;
    object.userData[jointsKey] = encodeJoints(joints);
    // Only written when on: hidden is the default, and a default should leave
    // no trace in the document.
    if (showColliders) {
        object.userData[collidersKey] = true;
    } else {
        object.userData.erase(collidersKey);
    }

    // Every value key goes first. Writing over the survivors is cheap, and it is
    // the only way an argument that was removed does not resurrect itself the
    // next time this is read: the name list would no longer mention it, but the
    // orphaned value would still be sitting in the document waiting for a name
    // list that does.
    for (const auto& key : argValueKeys(object)) object.userData.erase(key);

    if (xacroArgs.empty()) {
        object.userData.erase(xacro::argsUserDataKey);
        return;
    }

    std::string names;
    for (const auto& [name, value] : xacroArgs) {
        if (!names.empty()) names += ',';
        names += name;
        object.userData[xacro::argValueUserDataPrefix + name] = value;
    }
    object.userData[xacro::argsUserDataKey] = names;
}

void RobotConfig::erase(Object3D& object) {

    object.userData.erase(urdfKey);
    object.userData.erase(jointsKey);
    object.userData.erase(collidersKey);
    object.userData.erase(xacro::argsUserDataKey);
    for (const auto& key : argValueKeys(object)) object.userData.erase(key);
}

std::shared_ptr<Robot> threepp::editor::transplantRobot(Object3D& placeholder,
                                                        const std::shared_ptr<Robot>& donor,
                                                        const std::function<void(const std::string&)>& log) {

    if (!donor) return nullptr;
    auto* parent = placeholder.parent;
    if (!parent) return nullptr;

    const auto config = RobotConfig::read(placeholder).value_or(RobotConfig{});

    // First node with each name in the DOCUMENT's subtree. URDF link and joint
    // names are unique within a file — the loader's own finalize() resolves the
    // hierarchy by them — which is what makes this a lookup rather than a guess.
    // The root is excluded: it can be neither a link nor a joint node, and a
    // robot renamed to match one of its links would otherwise shadow that link
    // (traverse visits the root first, and the first name wins).
    std::unordered_map<std::string, Object3D*> byName;
    placeholder.traverse([&](Object3D& node) {
        if (&node == &placeholder || node.name.empty()) return;
        byName.emplace(node.name, &node);
    });

    const auto lookup = [&](const std::string& name) -> Object3D* {
        if (name.empty()) return nullptr;
        const auto it = byName.find(name);
        return it == byName.end() ? nullptr : it->second;
    };

    // The document's node for each of the donor's joints.
    //
    // Not by the joint node's own name: URDF link and joint names live in
    // separate namespaces and a file is free to reuse one. The joint node is
    // whatever sits between a link and its child link, so its child link — a
    // name that IS unique — identifies it without depending on the loader having
    // named the joint node at all. The name is the fallback for a document whose
    // hierarchy has been rearranged.
    const auto& donorJoints = donor->jointNodes();
    const auto& donorInfos = donor->jointInfos();

    std::vector<Object3D*> jointNodes(donorInfos.size(), nullptr);
    std::size_t resolved = 0;

    for (std::size_t i = 0; i < donorInfos.size(); ++i) {

        if (auto* childLink = lookup(donorInfos[i].child)) {
            // Not the placeholder itself: the ROOT link hangs directly off the
            // robot, and no joint drives it.
            if (childLink->parent && childLink->parent != &placeholder) {
                jointNodes[i] = childLink->parent;
            }
        }
        if (!jointNodes[i] && i < donorJoints.size() && donorJoints[i]) {
            jointNodes[i] = lookup(donorJoints[i]->name);
        }
        if (jointNodes[i]) ++resolved;
    }

    // Nothing to adopt, or nothing matched: an empty placeholder, or a file
    // rebuilt from scratch. Plant the donor rather than hand back a robot with no
    // joints — the annotations are worth less than the articulation — and say so.
    // A jointless URDF (one rigid link) has nothing to resolve and is adopted on
    // the strength of its subtree alone.
    const bool adopt = !placeholder.children.empty() && (resolved > 0 || donorInfos.empty());

    if (!adopt && log && !placeholder.children.empty()) {
        log("robot \"" + placeholder.name + "\": none of the " +
            std::to_string(donorInfos.size()) +
            " joints matched a node in the saved subtree, so it was rebuilt from the file - "
            "anything authored below the robot is gone (the file has likely changed since "
            "the document was saved)");
    }

    auto live = Robot::create();

    // The root's identity, placement and userData become the live robot's,
    // whichever subtree it ends up wearing.
    live->name = placeholder.name;
    live->uuid = placeholder.uuid;
    live->position.copy(placeholder.position);
    live->quaternion.copy(placeholder.quaternion);
    live->scale.copy(placeholder.scale);
    live->matrixAutoUpdate = placeholder.matrixAutoUpdate;
    live->visible = placeholder.visible;
    live->castShadow = placeholder.castShadow;
    live->receiveShadow = placeholder.receiveShadow;
    live->frustumCulled = placeholder.frustumCulled;
    live->renderOrder = placeholder.renderOrder;
    live->layers.disableAll();
    for (unsigned int channel = 0; channel < 32; ++channel) {
        if (placeholder.layers.mask() & (1u << channel)) live->layers.enable(channel);
    }
    live->animations = placeholder.animations;
    live->userData = placeholder.userData;

    // The saved joint values are indexed by the DONOR's DOF layout. When a joint
    // fails to resolve the live table compacts around it, so each live DOF
    // remembers the donor slot its value comes from — without this, every joint
    // after a gap would inherit its neighbour's saved value.
    std::vector<int> valueSlot;

    if (adopt) {

        // Move the document's subtree over wholesale, in order. Nothing is
        // rebuilt, so nothing below the root can be lost.
        std::vector<Object3D*> children = placeholder.children;
        for (auto* child : children) {
            if (!child) continue;
            if (auto owned = child->removeFromParent()) {
                live->add(owned);
            } else {
                // Attached by reference: the parent never owned it, so neither
                // does the new one.
                live->addRef(*child);
            }
        }

        // The table itself, read off the donor and applied to the document's
        // nodes. A co-owner, not a borrower: Robot keeps RAW pointers to its
        // joint nodes in origPose_/articulatedJoints_, and it is the table's own
        // shared_ptr that keeps those valid if a link is later deleted out of the
        // hierarchy.
        const auto refer = [](Object3D* node) {
            if (node->weak_from_this().expired()) {
                return std::shared_ptr<Object3D>(node, [](Object3D*) {});
            }
            return node->shared_from_this();
        };

        std::vector<std::string> unresolved;

        for (const auto& link : donor->links()) {
            if (!link) continue;
            if (auto* node = lookup(link->name)) {
                live->addLink(refer(node));
            } else {
                unresolved.push_back("link \"" + link->name + "\"");
            }
        }

        for (std::size_t i = 0; i < donorInfos.size(); ++i) {
            if (!jointNodes[i]) {
                unresolved.push_back("joint \"" + donorInfos[i].name + "\"");
                continue;
            }
            const auto [restPosition, restRotation] = donor->jointRestPose(i);
            live->addJoint(refer(jointNodes[i]), donorInfos[i], restPosition, restRotation);
            if (donorInfos[i].type != Robot::JointType::Fixed) {
                valueSlot.push_back(donor->jointDof(i));
            }
        }

        live->finalizeInPlace();
        if (const auto& ee = donor->endEffectorLink(); !ee.empty()) live->setEndEffector(ee);

        if (!unresolved.empty() && log) {
            std::string names;
            for (std::size_t i = 0; i < unresolved.size() && i < 4; ++i) {
                names += (i ? ", " : "") + unresolved[i];
            }
            if (unresolved.size() > 4) names += " and " + std::to_string(unresolved.size() - 4) + " more";
            log("robot \"" + live->name + "\": " +
                std::filesystem::path(config.urdf).filename().string() + " names " + names +
                " but the saved subtree has no matching node - those parts are not articulated");
        }

    } else {

        // The donor's own hierarchy, which it already owns and has a table for.
        // Re-hang it under the live root rather than returning the donor itself,
        // so both branches hand back the same thing.
        std::vector<Object3D*> children = donor->children;
        for (auto* child : children) {
            if (!child) continue;
            if (auto owned = child->removeFromParent()) {
                live->add(owned);
            } else {
                live->addRef(*child);
            }
        }
        for (const auto& link : donor->links()) {
            if (link) live->addLink(link);
        }
        for (std::size_t i = 0; i < donorInfos.size(); ++i) {
            if (i >= donorJoints.size() || !donorJoints[i]) continue;
            const auto [restPosition, restRotation] = donor->jointRestPose(i);
            live->addJoint(donorJoints[i], donorInfos[i], restPosition, restRotation);
            if (donorInfos[i].type != Robot::JointType::Fixed) {
                valueSlot.push_back(donor->jointDof(i));
            }
        }
        live->finalizeInPlace();
        if (const auto& ee = donor->endEffectorLink(); !ee.empty()) live->setEndEffector(ee);
    }

    for (std::size_t i = 0; i < valueSlot.size() && i < live->numDOF(); ++i) {
        const int slot = valueSlot[i];
        if (slot >= 0 && static_cast<std::size_t>(slot) < config.joints.size()) {
            live->setJointValue(i, config.joints[slot]);
        }
    }
    live->showColliders(config.showColliders);

    const auto index = childIndex(*parent, placeholder);
    placeholder.removeFromParent();
    insertChildAt(*parent, live, index);

    return live;
}
