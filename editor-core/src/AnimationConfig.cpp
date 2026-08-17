
#include "threepp/extras/editor/AnimationConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    // Same formatting contract as PhysicsConfig: locale-independent, trailing
    // zeros trimmed, byte-identical for an unchanged value.
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

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
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
    });

    return config;
}

std::optional<AnimationConfig> AnimationConfig::read(const Object3D& object) {

    return readEntry<AnimationConfig>(object, userDataKey);
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
