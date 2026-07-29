
#include "threepp/extras/editor/RobotConfig.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"
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

namespace {

    // Same contract as the other editor configs: locale-independent, trailing
    // zeros trimmed, byte-identical for an unchanged value so saved documents
    // stay diff-clean.
    std::string number(float value) {

        char buffer[32];
        const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
        std::string out(buffer, buffer + (n > 0 ? n : 0));
        if (out.find('.') == std::string::npos) return out;
        while (!out.empty() && out.back() == '0') out.pop_back();
        if (!out.empty() && out.back() == '.') out.pop_back();
        return out.empty() ? "0" : out;
    }

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

    std::string readString(const Object3D& object, const char* key) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return {};
        if (it->second.type() != typeid(std::string)) return {};
        return std::any_cast<const std::string&>(it->second);
    }

    bool readBool(const Object3D& object, const char* key, bool fallback) {

        const auto it = object.userData.find(key);
        if (it == object.userData.end()) return fallback;
        if (it->second.type() != typeid(bool)) return fallback;
        return std::any_cast<bool>(it->second);
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

std::optional<RobotConfig> RobotConfig::read(const Object3D& object) {

    const auto urdf = readString(object, urdfKey);
    if (urdf.empty()) return std::nullopt;

    RobotConfig config;
    config.urdf = urdf;
    config.joints = decodeJoints(readString(object, jointsKey));
    config.showColliders = readBool(object, collidersKey, false);
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
}

void RobotConfig::erase(Object3D& object) {

    object.userData.erase(urdfKey);
    object.userData.erase(jointsKey);
    object.userData.erase(collidersKey);
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
