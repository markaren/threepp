
#include "threepp/extras/editor/GranularConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    GranularConfig::Visual visualFrom(std::string_view text, GranularConfig::Visual fallback) {

        if (text == "auto") return GranularConfig::Visual::Auto;
        if (text == "instanced") return GranularConfig::Visual::Instanced;
        if (text == "field") return GranularConfig::Visual::Field;
        return fallback;
    }

    const char* visualToken(GranularConfig::Visual visual) {

        switch (visual) {
            case GranularConfig::Visual::Auto: return "auto";
            case GranularConfig::Visual::Instanced: return "instanced";
            case GranularConfig::Visual::Field: return "field";
        }
        return "auto";
    }

}// namespace


std::string GranularConfig::encode() const {

    std::string out;
    out += "spacing=";
    out += number(spacing);
    out += ";iterations=";
    out += std::to_string(iterations);
    out += ";capacity=";
    out += std::to_string(capacity);
    out += ";maxvel=";
    out += number(maxVelocity);

    out += ";friction=";
    out += number(friction);
    out += ";damping=";
    out += number(damping);
    out += ";adhesion=";
    out += number(adhesion);
    out += ";cohesion=";
    out += number(cohesion);
    out += ";viscosity=";
    out += number(viscosity);
    out += ";gravityscale=";
    out += number(gravityScale);

    out += ";emitextx=";
    out += number(emitExtentX);
    out += ";emitextz=";
    out += number(emitExtentZ);
    out += ";rate=";
    out += number(rate);
    out += ";emitvelx=";
    out += number(emitVelocity.x);
    out += ";emitvely=";
    out += number(emitVelocity.y);
    out += ";emitvelz=";
    out += number(emitVelocity.z);
    out += ";mass=";
    out += number(mass);
    out += ";emitfor=";
    out += number(emitFor);
    out += ";jitter=";
    out += number(jitter);

    out += ";visual=";
    out += visualToken(visual);
    out += ";colr=";
    out += number(color.r);
    out += ";colg=";
    out += number(color.g);
    out += ";colb=";
    out += number(color.b);
    out += ";roughness=";
    out += number(roughness);

    return out;
}

std::optional<GranularConfig> GranularConfig::decode(const std::string& text) {

    GranularConfig config;

    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "spacing") {
            config.spacing = toFloat(value, config.spacing);
        } else if (key == "iterations") {
            config.iterations = toInt(value, config.iterations);
        } else if (key == "capacity") {
            config.capacity = toInt(value, config.capacity);
        } else if (key == "maxvel") {
            config.maxVelocity = toFloat(value, config.maxVelocity);
        } else if (key == "friction") {
            config.friction = toFloat(value, config.friction);
        } else if (key == "damping") {
            config.damping = toFloat(value, config.damping);
        } else if (key == "adhesion") {
            config.adhesion = toFloat(value, config.adhesion);
        } else if (key == "cohesion") {
            config.cohesion = toFloat(value, config.cohesion);
        } else if (key == "viscosity") {
            config.viscosity = toFloat(value, config.viscosity);
        } else if (key == "gravityscale") {
            config.gravityScale = toFloat(value, config.gravityScale);
        } else if (key == "emitextx") {
            config.emitExtentX = toFloat(value, config.emitExtentX);
        } else if (key == "emitextz") {
            config.emitExtentZ = toFloat(value, config.emitExtentZ);
        } else if (key == "rate") {
            config.rate = toFloat(value, config.rate);
        } else if (key == "emitvelx") {
            config.emitVelocity.x = toFloat(value, config.emitVelocity.x);
        } else if (key == "emitvely") {
            config.emitVelocity.y = toFloat(value, config.emitVelocity.y);
        } else if (key == "emitvelz") {
            config.emitVelocity.z = toFloat(value, config.emitVelocity.z);
        } else if (key == "mass") {
            config.mass = toFloat(value, config.mass);
        } else if (key == "emitfor") {
            config.emitFor = toFloat(value, config.emitFor);
        } else if (key == "jitter") {
            config.jitter = toFloat(value, config.jitter);
        } else if (key == "visual") {
            config.visual = visualFrom(value, config.visual);
        } else if (key == "colr") {
            config.color.r = toFloat(value, config.color.r);
        } else if (key == "colg") {
            config.color.g = toFloat(value, config.color.g);
        } else if (key == "colb") {
            config.color.b = toFloat(value, config.color.b);
        } else if (key == "roughness") {
            config.roughness = toFloat(value, config.roughness);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    // spacing is a divisor in the lattice emitter and the source of every
    // derived offset, so zero is not a degenerate case but a crash.
    config.spacing = std::max(config.spacing, 1e-4f);
    // 32 is where more iterations stop buying pile stability and start buying
    // frame time; 1 is the floor PhysX itself takes.
    config.iterations = std::max(1, std::min(32, config.iterations));
    config.capacity = std::max(1, config.capacity);
    config.maxVelocity = std::max(config.maxVelocity, 0.f);
    config.rate = std::max(config.rate, 0.f);
    config.jitter = std::max(0.f, std::min(1.f, config.jitter));
    config.emitExtentX = std::max(config.emitExtentX, 0.f);
    config.emitExtentZ = std::max(config.emitExtentZ, 0.f);
    config.mass = std::max(config.mass, 0.f);
    config.emitFor = std::max(config.emitFor, 0.f);

    return config;
}

std::optional<GranularConfig> GranularConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;
    return decode(std::any_cast<const std::string&>(it->second));
}

void GranularConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void GranularConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool GranularConfig::isGranular(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    return it != object.userData.end() && it->second.type() == typeid(std::string);
}

const char* GranularConfig::label(Visual visual) {

    switch (visual) {
        case Visual::Auto: return "Auto";
        case Visual::Instanced: return "Instanced";
        case Visual::Field: return "Particle field";
    }
    return "Auto";
}
