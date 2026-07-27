
#include "threepp/extras/editor/PhysicsConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <charconv>
#include <sstream>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Fixed 6 decimals with trailing zeros trimmed: stable across platforms
    // (unlike ostream defaults with a locale) and byte-identical for an
    // unchanged value, which keeps saved documents diff-clean.
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

    PhysicsConfig::Body bodyFrom(std::string_view text, PhysicsConfig::Body fallback) {

        if (text == "static") return PhysicsConfig::Body::Static;
        if (text == "dynamic") return PhysicsConfig::Body::Dynamic;
        if (text == "kinematic") return PhysicsConfig::Body::Kinematic;
        return fallback;
    }

    PhysicsConfig::Shape shapeFrom(std::string_view text, PhysicsConfig::Shape fallback) {

        if (text == "auto") return PhysicsConfig::Shape::Auto;
        if (text == "box") return PhysicsConfig::Shape::Box;
        if (text == "sphere") return PhysicsConfig::Shape::Sphere;
        if (text == "capsule") return PhysicsConfig::Shape::Capsule;
        if (text == "convex") return PhysicsConfig::Shape::Convex;
        if (text == "trimesh") return PhysicsConfig::Shape::TriMesh;
        return fallback;
    }

    const char* bodyToken(PhysicsConfig::Body body) {

        switch (body) {
            case PhysicsConfig::Body::Static: return "static";
            case PhysicsConfig::Body::Dynamic: return "dynamic";
            case PhysicsConfig::Body::Kinematic: return "kinematic";
        }
        return "dynamic";
    }

    const char* shapeToken(PhysicsConfig::Shape shape) {

        switch (shape) {
            case PhysicsConfig::Shape::Auto: return "auto";
            case PhysicsConfig::Shape::Box: return "box";
            case PhysicsConfig::Shape::Sphere: return "sphere";
            case PhysicsConfig::Shape::Capsule: return "capsule";
            case PhysicsConfig::Shape::Convex: return "convex";
            case PhysicsConfig::Shape::TriMesh: return "trimesh";
        }
        return "auto";
    }

}// namespace


const char* PhysicsConfig::label(Body body) {

    switch (body) {
        case Body::Static: return "Static";
        case Body::Dynamic: return "Dynamic";
        case Body::Kinematic: return "Kinematic";
    }
    return "Dynamic";
}

const char* PhysicsConfig::label(Shape shape) {

    switch (shape) {
        case Shape::Auto: return "Auto";
        case Shape::Box: return "Box";
        case Shape::Sphere: return "Sphere";
        case Shape::Capsule: return "Capsule";
        case Shape::Convex: return "Convex";
        case Shape::TriMesh: return "TriMesh";
    }
    return "Auto";
}

std::string PhysicsConfig::encode() const {

    std::string out;
    out += "body=";
    out += bodyToken(body);
    out += ";shape=";
    out += shapeToken(shape);
    out += ";mass=";
    out += number(mass);
    out += ";friction=";
    out += number(friction);
    out += ";restitution=";
    out += number(restitution);
    return out;
}

std::optional<PhysicsConfig> PhysicsConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    PhysicsConfig config;
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
            if (key == "body") {
                config.body = bodyFrom(value, config.body);
            } else if (key == "shape") {
                config.shape = shapeFrom(value, config.shape);
            } else if (key == "mass") {
                config.mass = toFloat(value, config.mass);
            } else if (key == "friction") {
                config.friction = toFloat(value, config.friction);
            } else if (key == "restitution") {
                config.restitution = toFloat(value, config.restitution);
            }
            // Unknown keys ignored on purpose: a document written by a newer
            // editor still loads here.
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }

    return config;
}

std::optional<PhysicsConfig> PhysicsConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    return decode(std::any_cast<const std::string&>(it->second));
}

void PhysicsConfig::write(Object3D& object) const {

    if (!enabled) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void PhysicsConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
