
#include "threepp/extras/editor/RobotConfig.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/objects/Robot.hpp"

#include <any>
#include <cstdio>
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

    // Collect each placeholder DESCENDANT's non-empty userData by node name,
    // before the swap. Only descendants: the root's userData is copied wholesale
    // just below, and a sensor authored on the root rides with it. Skip nodes
    // with no name (nothing to match them to) and empty userData (nothing to
    // carry). First name wins on both sides, which is the same rule the transform
    // override table uses for a URDF with duplicate link names.
    std::vector<std::pair<std::string, std::unordered_map<std::string, std::any>>> descendantData;
    placeholder.traverse([&](Object3D& node) {
        if (&node == &placeholder) return;
        if (node.name.empty() || node.userData.empty()) return;
        descendantData.emplace_back(node.name, node.userData);
    });

    // The root's identity, placement and userData move to the fresh robot.
    robot->name = placeholder.name;
    robot->uuid = placeholder.uuid;
    robot->position.copy(placeholder.position);
    robot->quaternion.copy(placeholder.quaternion);
    robot->scale.copy(placeholder.scale);
    robot->visible = placeholder.visible;
    robot->userData = placeholder.userData;

    // Re-apply descendant userData onto same-named nodes in the fresh subtree.
    // Build a name -> first node map once, then assign; a name that no longer
    // exists (the URDF changed under the document) is reported, not lost silently.
    std::unordered_map<std::string, Object3D*> byName;
    robot->traverse([&](Object3D& node) {
        if (&node == robot.get()) return;
        if (node.name.empty()) return;
        byName.emplace(node.name, &node);// first match wins
    });
    for (auto& [name, data] : descendantData) {
        const auto it = byName.find(name);
        if (it == byName.end()) {
            if (log) log("robot \"" + robot->name + "\": userData on \"" + name +
                         "\" did not resolve after re-articulation (node gone from the URDF)");
            continue;
        }
        for (auto& [key, value] : data) it->second->userData[key] = value;
    }

    for (std::size_t i = 0; i < config.joints.size() && i < robot->numDOF(); ++i) {
        robot->setJointValue(i, config.joints[i]);
    }
    robot->showColliders(config.showColliders);

    const auto index = childIndex(*parent, placeholder);
    placeholder.removeFromParent();
    insertChildAt(*parent, robot, index);
}
