
#include "threepp/extras/editor/RobotConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/objects/Robot.hpp"

#include <any>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    // Same contract as the other editor configs: locale-independent, trailing
    // zeros trimmed, byte-identical for an unchanged value so saved documents
    // stay diff-clean.
    // Pre-order walk of everything below `root`, root excluded. Both sides of a
    // transplant are built from the SAME URDF by the same loader, so a node's
    // position in this walk identifies it — which is what makes carrying an
    // authored entry across possible at all for the nodes that have no name.
    // The same walk (and the same reasoning) is ObjectLoader::applyAssetOverrides'.
    std::vector<Object3D*> flattenDescendants(Object3D& root) {

        std::vector<Object3D*> flat;

        std::function<void(Object3D&)> collect = [&](Object3D& node) {
            for (auto* child : node.children) {
                if (!child) continue;
                flat.push_back(child);
                collect(*child);
            }
        };
        collect(root);

        return flat;
    }

    bool readBool(const Object3D& object, const char* key, bool fallback) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return fallback;
        if (it->second.type() != typeid(bool)) return fallback;
        return std::any_cast<bool>(it->second);
    }

    std::vector<std::string> splitNames(const std::string& list) {

        std::vector<std::string> names;
        for (std::size_t start = 0; start <= list.size();) {
            const auto end = list.find(',', start);
            auto name = list.substr(start, (end == std::string::npos ? list.size() : end) - start);
            if (!name.empty()) names.push_back(std::move(name));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return names;
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

    for (const auto& name : splitNames(readString(object, xacro::argsUserDataKey))) {
        const auto key = xacro::argValueUserDataPrefix + name;
        // A name whose value key is gone is dropped, not read as empty: an empty
        // override is a real override and would suppress the file's default.
        if (object.userData.count(key)) config.xacroArgs.emplace_back(name, readString(object, key.c_str()));
    }

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

void threepp::editor::transplantRobot(Object3D& placeholder, const std::shared_ptr<Robot>& robot,
                                      const std::function<void(const std::string&)>& log) {

    if (!robot) return;
    auto* parent = placeholder.parent;
    if (!parent) return;

    const auto config = RobotConfig::read(placeholder).value_or(RobotConfig{});

    // Collect each placeholder DESCENDANT's non-empty userData before the swap,
    // keyed by its POSITION in the pre-order walk. Only descendants: the root's
    // userData is copied wholesale just below, and a sensor authored on the root
    // rides with it.
    //
    // Position, not name. A URDF's visual and collision groups are unnamed, and
    // so is every mesh the geometry loader hands back — and those are exactly the
    // nodes a viewport click drills down to, so a name-keyed carry-over dropped a
    // sensor authored by clicking the robot. The name is kept alongside as the
    // check that the two trees still line up (see below).
    struct Carried {
        std::size_t index;
        std::string name;
        std::unordered_map<std::string, std::any> data;
    };

    std::vector<Carried> descendantData;
    {
        const auto flat = flattenDescendants(placeholder);
        for (std::size_t i = 0; i < flat.size(); ++i) {
            if (flat[i]->userData.empty()) continue;
            descendantData.push_back({i, flat[i]->name, flat[i]->userData});
        }
    }

    // The root's identity, placement and userData move to the fresh robot.
    robot->name = placeholder.name;
    robot->uuid = placeholder.uuid;
    robot->position.copy(placeholder.position);
    robot->quaternion.copy(placeholder.quaternion);
    robot->scale.copy(placeholder.scale);
    robot->visible = placeholder.visible;
    robot->userData = placeholder.userData;

    // Re-apply descendant userData onto the fresh subtree. Same position AND the
    // same name is the same node; when the name disagrees the file has changed
    // underneath and the position no longer identifies anything, so fall back to
    // a name lookup before giving up. What cannot be placed is reported, not lost
    // silently — for an unnamed node that is all that can be said about it.
    const auto fresh = flattenDescendants(*robot);

    std::unordered_map<std::string, Object3D*> byName;
    for (auto* node : fresh) {
        if (node->name.empty()) continue;
        byName.emplace(node->name, node);// first match wins
    }

    for (auto& entry : descendantData) {

        Object3D* target = nullptr;
        if (entry.index < fresh.size() && fresh[entry.index]->name == entry.name) {
            target = fresh[entry.index];
        } else if (!entry.name.empty()) {
            if (const auto it = byName.find(entry.name); it != byName.end()) target = it->second;
        }

        if (!target) {
            if (log) {
                log("robot \"" + robot->name + "\": userData on " +
                    (entry.name.empty() ? "node #" + std::to_string(entry.index)
                                        : "\"" + entry.name + "\"") +
                    " did not resolve after re-articulation (node gone from the URDF)");
            }
            continue;
        }
        for (auto& [key, value] : entry.data) target->userData[key] = value;
    }

    for (std::size_t i = 0; i < config.joints.size() && i < robot->numDOF(); ++i) {
        robot->setJointValue(i, config.joints[i]);
    }
    robot->showColliders(config.showColliders);

    const auto index = childIndex(*parent, placeholder);
    placeholder.removeFromParent();
    insertChildAt(*parent, robot, index);
}
