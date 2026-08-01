
#include "threepp/extras/editor/ArticulationConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <cstdio>
#include <string>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

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

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
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
    });

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
