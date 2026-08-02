
#include "threepp/extras/editor/JointConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    JointConfig::Type typeFrom(std::string_view text, JointConfig::Type fallback) {

        if (text == "fixed") return JointConfig::Type::Fixed;
        if (text == "revolute") return JointConfig::Type::Revolute;
        if (text == "prismatic") return JointConfig::Type::Prismatic;
        if (text == "spherical") return JointConfig::Type::Spherical;
        if (text == "distance") return JointConfig::Type::Distance;
        return fallback;
    }

    const char* typeToken(JointConfig::Type type) {

        switch (type) {
            case JointConfig::Type::Fixed: return "fixed";
            case JointConfig::Type::Revolute: return "revolute";
            case JointConfig::Type::Prismatic: return "prismatic";
            case JointConfig::Type::Spherical: return "spherical";
            case JointConfig::Type::Distance: return "distance";
        }
        return "revolute";
    }

}// namespace


const char* JointConfig::label(Type type) {

    switch (type) {
        case Type::Fixed: return "Fixed";
        case Type::Revolute: return "Revolute";
        case Type::Prismatic: return "Prismatic";
        case Type::Spherical: return "Spherical";
        case Type::Distance: return "Distance";
    }
    return "Revolute";
}

std::string JointConfig::encode() const {

    std::string out;
    out += "type=";
    out += typeToken(type);
    // Every field rides along whatever the type is, so switching type and
    // back does not quietly reset the ones the other type hides — the same
    // rule PhysicsConfig's soft-body and V-HACD parameters follow.
    out += ";limited=";
    out += (limited ? "1" : "0");
    out += ";lower=";
    out += number(lower);
    out += ";upper=";
    out += number(upper);
    out += ";coney=";
    out += number(coneY);
    out += ";conez=";
    out += number(coneZ);
    out += ";stiffness=";
    out += number(stiffness);
    out += ";damping=";
    out += number(damping);
    out += ";maxforce=";
    out += number(maxForce);
    out += ";target=";
    out += number(target);
    out += ";velocity=";
    out += number(velocity);
    out += ";breakforce=";
    out += number(breakForce);
    out += ";breaktorque=";
    out += number(breakTorque);
    out += ";collide=";
    out += (collide ? "1" : "0");
    return out;
}

std::optional<JointConfig> JointConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    JointConfig config;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "type") {
            config.type = typeFrom(value, config.type);
        } else if (key == "limited") {
            config.limited = toInt(value, config.limited ? 1 : 0) != 0;
        } else if (key == "lower") {
            config.lower = toFloat(value, config.lower);
        } else if (key == "upper") {
            config.upper = toFloat(value, config.upper);
        } else if (key == "coney") {
            config.coneY = toFloat(value, config.coneY);
        } else if (key == "conez") {
            config.coneZ = toFloat(value, config.coneZ);
        } else if (key == "stiffness") {
            config.stiffness = toFloat(value, config.stiffness);
        } else if (key == "damping") {
            config.damping = toFloat(value, config.damping);
        } else if (key == "maxforce") {
            config.maxForce = toFloat(value, config.maxForce);
        } else if (key == "target") {
            config.target = toFloat(value, config.target);
        } else if (key == "velocity") {
            config.velocity = toFloat(value, config.velocity);
        } else if (key == "breakforce") {
            config.breakForce = toFloat(value, config.breakForce);
        } else if (key == "breaktorque") {
            config.breakTorque = toFloat(value, config.breakTorque);
        } else if (key == "collide") {
            config.collide = toInt(value, config.collide ? 1 : 0) != 0;
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    return config;
}

std::optional<JointConfig> JointConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    auto config = decode(std::any_cast<const std::string&>(it->second));
    if (config) config->body = readString(object, bodyKey);
    return config;
}

bool JointConfig::isJoint(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    return it != object.userData.end() && it->second.type() == typeid(std::string);
}

void JointConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
    if (body.empty()) {
        object.userData.erase(bodyKey);
    } else {
        object.userData[bodyKey] = body;
    }
}

void JointConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
    object.userData.erase(bodyKey);
}
