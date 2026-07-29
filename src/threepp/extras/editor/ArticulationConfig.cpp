
#include "threepp/extras/editor/ArticulationConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <cstdio>
#include <string>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Fixed 6 decimals with trailing zeros trimmed: stable across platforms
    // (unlike ostream defaults with a locale) and byte-identical for an
    // unchanged value, which keeps saved documents diff-clean. Same routine
    // PhysicsConfig/SensorConfig use, for the same reason.
    std::string number(float value) {

        char buffer[32];
        const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
        std::string out(buffer, buffer + (n > 0 ? n : 0));
        if (out.find('.') == std::string::npos) return out;
        while (!out.empty() && out.back() == '0') out.pop_back();
        if (!out.empty() && out.back() == '.') out.pop_back();
        return out.empty() ? "0" : out;
    }

    float toFloat(std::string_view text, float fallback) {

        try {
            return std::stof(std::string(text));
        } catch (...) {
            return fallback;
        }
    }

    int toInt(std::string_view text, int fallback) {

        try {
            return std::stoi(std::string(text));
        } catch (...) {
            return fallback;
        }
    }

}// namespace


std::string ArticulationConfig::encode() const {

    // Every key, every time — the PhysicsConfig contract. A field the user is not
    // currently looking at is still their setting; never let the written key set
    // depend on anything but "this is an articulation".
    std::string out;
    out += "fixedbase=";
    out += (fixedBase ? "1" : "0");
    out += ";stiffness=";
    out += number(stiffness);
    out += ";damping=";
    out += number(damping);
    out += ";maxforce=";
    out += number(maxForce);
    out += ";selfcollision=";
    out += (selfCollision ? "1" : "0");
    out += ";iterations=";
    out += std::to_string(iterations);
    out += ";density=";
    out += number(density);
    return out;
}

std::optional<ArticulationConfig> ArticulationConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    ArticulationConfig config;
    config.enabled = true;

    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(';', start);
        const auto token = std::string_view(text).substr(
                start, (end == std::string::npos ? text.size() : end) - start);
        const auto eq = token.find('=');
        if (eq != std::string_view::npos) {
            const auto key = token.substr(0, eq);
            const auto value = token.substr(eq + 1);
            if (key == "fixedbase") {
                config.fixedBase = toInt(value, config.fixedBase ? 1 : 0) != 0;
            } else if (key == "stiffness") {
                config.stiffness = toFloat(value, config.stiffness);
            } else if (key == "damping") {
                config.damping = toFloat(value, config.damping);
            } else if (key == "maxforce") {
                config.maxForce = toFloat(value, config.maxForce);
            } else if (key == "selfcollision") {
                config.selfCollision = toInt(value, config.selfCollision ? 1 : 0) != 0;
            } else if (key == "iterations") {
                config.iterations = toInt(value, config.iterations);
            } else if (key == "density") {
                config.density = toFloat(value, config.density);
            }
            // Unknown keys ignored on purpose: a document written by a newer
            // editor still loads here.
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }

    return config;
}

std::optional<ArticulationConfig> ArticulationConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    return decode(std::any_cast<const std::string&>(it->second));
}

void ArticulationConfig::write(Object3D& object) const {

    if (!enabled) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void ArticulationConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
