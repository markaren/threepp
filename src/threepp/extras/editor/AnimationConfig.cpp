
#include "threepp/extras/editor/AnimationConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Same formatting contract as PhysicsConfig: locale-independent, trailing
    // zeros trimmed, byte-identical for an unchanged value.
    std::string number(float value) {

        char buffer[32];
        const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
        std::string out(buffer, buffer + (n > 0 ? n : 0));
        const auto dot = out.find('.');
        if (dot == std::string::npos) return out;
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

    std::string sanitized(std::string name) {

        name.erase(std::remove_if(name.begin(), name.end(),
                                  [](char c) { return c == ';' || c == '='; }),
                   name.end());
        return name;
    }

}// namespace


bool AnimationConfig::isDefault() const {

    return *this == AnimationConfig{};
}

std::string AnimationConfig::encode() const {

    std::string out;
    out += "autoplay=";
    out += autoplay ? "1" : "0";
    out += ";loop=";
    out += loop ? "1" : "0";
    out += ";speed=";
    out += number(speed);
    if (!clip.empty()) {
        out += ";clip=";
        out += sanitized(clip);
    }
    return out;
}

std::optional<AnimationConfig> AnimationConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    AnimationConfig config;

    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(';', start);
        const auto token = std::string_view(text).substr(
                start, (end == std::string::npos ? text.size() : end) - start);
        const auto eq = token.find('=');
        if (eq != std::string_view::npos) {
            const auto key = token.substr(0, eq);
            const auto value = token.substr(eq + 1);
            if (key == "autoplay") {
                config.autoplay = value != "0";
            } else if (key == "loop") {
                config.loop = value != "0";
            } else if (key == "speed") {
                config.speed = toFloat(value, config.speed);
            } else if (key == "clip") {
                config.clip = std::string(value);
            }
            // Unknown keys ignored on purpose: a document written by a newer
            // editor still loads here.
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }

    return config;
}

std::optional<AnimationConfig> AnimationConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    return decode(std::any_cast<const std::string&>(it->second));
}

void AnimationConfig::write(Object3D& object) const {

    if (isDefault()) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void AnimationConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
