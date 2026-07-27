
#include "threepp/extras/editor/RobotConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <any>
#include <cstdio>
#include <string_view>

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
    return config;
}

void RobotConfig::write(Object3D& object) const {

    if (urdf.empty()) {
        erase(object);
        return;
    }
    object.userData[urdfKey] = urdf;
    object.userData[jointsKey] = encodeJoints(joints);
}

void RobotConfig::erase(Object3D& object) {

    object.userData.erase(urdfKey);
    object.userData.erase(jointsKey);
}
