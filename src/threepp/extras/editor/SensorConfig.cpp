
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <cstdio>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Fixed 6 decimals with trailing zeros trimmed: stable across platforms
    // (unlike ostream defaults with a locale) and byte-identical for an
    // unchanged value, which keeps saved documents diff-clean. Same routine
    // PhysicsConfig uses, for the same reason.
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

    SensorConfig::Type typeFrom(std::string_view text, SensorConfig::Type fallback) {

        if (text == "imu") return SensorConfig::Type::Imu;
        if (text == "depth") return SensorConfig::Type::Depth;
        if (text == "lidar") return SensorConfig::Type::Lidar;
        if (text == "encoder") return SensorConfig::Type::Encoder;
        if (text == "contact") return SensorConfig::Type::Contact;
        if (text == "forcetorque") return SensorConfig::Type::ForceTorque;
        return fallback;
    }

    const char* typeToken(SensorConfig::Type type) {

        switch (type) {
            case SensorConfig::Type::Imu: return "imu";
            case SensorConfig::Type::Depth: return "depth";
            case SensorConfig::Type::Lidar: return "lidar";
            case SensorConfig::Type::Encoder: return "encoder";
            case SensorConfig::Type::Contact: return "contact";
            case SensorConfig::Type::ForceTorque: return "forcetorque";
        }
        return "imu";
    }

    SensorConfig::Beams beamsFrom(std::string_view text, SensorConfig::Beams fallback) {

        if (text == "dense") return SensorConfig::Beams::Dense;
        if (text == "vlp16") return SensorConfig::Beams::VLP16;
        if (text == "hdl32e") return SensorConfig::Beams::HDL32E;
        if (text == "os1_64") return SensorConfig::Beams::OS1_64;
        if (text == "os0_128") return SensorConfig::Beams::OS0_128;
        return fallback;
    }

    const char* beamsToken(SensorConfig::Beams beams) {

        switch (beams) {
            case SensorConfig::Beams::Dense: return "dense";
            case SensorConfig::Beams::VLP16: return "vlp16";
            case SensorConfig::Beams::HDL32E: return "hdl32e";
            case SensorConfig::Beams::OS1_64: return "os1_64";
            case SensorConfig::Beams::OS0_128: return "os0_128";
        }
        return "vlp16";
    }

}// namespace


const char* SensorConfig::label(Type type) {

    switch (type) {
        case Type::Imu: return "IMU";
        case Type::Depth: return "Depth Camera";
        case Type::Lidar: return "LIDAR";
        case Type::Encoder: return "Joint Encoder";
        case Type::Contact: return "Contact";
        case Type::ForceTorque: return "Force/Torque";
    }
    return "IMU";
}

const char* SensorConfig::label(Beams beams) {

    switch (beams) {
        case Beams::Dense: return "Dense Grid";
        case Beams::VLP16: return "VLP-16";
        case Beams::HDL32E: return "HDL-32E";
        case Beams::OS1_64: return "OS1-64";
        case Beams::OS0_128: return "OS0-128";
    }
    return "VLP-16";
}

bool SensorConfig::isProprioceptive(Type type) {

    return !isVision(type);
}

bool SensorConfig::isVision(Type type) {

    return type == Type::Depth || type == Type::Lidar;
}

std::uint64_t SensorConfig::streamSeed(int index) const {

    // SplitMix64's own mixing step: decorrelates the sub-streams of one authored
    // seed without needing a second PRNG, and index 0 is not the raw seed, so a
    // seed of 0 is still a usable stream rather than a degenerate one.
    std::uint64_t z = static_cast<std::uint64_t>(static_cast<std::int64_t>(seed)) +
                      0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(index + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::string SensorConfig::encode() const {

    // Every key, every time — see the header note. A type flip must not be a
    // data loss, and the only way to guarantee that is to never make the
    // written key set depend on the type.
    std::string out;
    out += "type=";
    out += typeToken(type);
    out += ";rate=";
    out += number(rateHz);
    out += ";seed=";
    out += std::to_string(seed);

    out += ";gyrodensity=";
    out += number(gyroNoiseDensity);
    out += ";gyrowalk=";
    out += number(gyroRandomWalk);
    out += ";acceldensity=";
    out += number(accelNoiseDensity);
    out += ";accelwalk=";
    out += number(accelRandomWalk);

    out += ";near=";
    out += number(nearPlane);
    out += ";far=";
    out += number(farPlane);
    out += ";rangestddev=";
    out += number(rangeStddev);
    out += ";rangepermetre=";
    out += number(rangeStddevPerMetre);
    out += ";rangebias=";
    out += number(rangeBias);

    out += ";fov=";
    out += number(fovY);
    out += ";width=";
    out += std::to_string(width);
    out += ";height=";
    out += std::to_string(height);

    out += ";beams=";
    out += beamsToken(beams);
    out += ";facesize=";
    out += std::to_string(faceSize);

    out += ";joint=";
    out += joint;// a URDF joint name; never carries ';' or '=', so it rides verbatim
    out += ";encoderres=";
    out += number(encoderResolution);
    out += ";contactthreshold=";
    out += number(contactForceThreshold);
    return out;
}

std::optional<SensorConfig> SensorConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    SensorConfig config;
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
            if (key == "type") {
                config.type = typeFrom(value, config.type);
            } else if (key == "rate") {
                config.rateHz = toFloat(value, config.rateHz);
            } else if (key == "seed") {
                config.seed = toInt(value, config.seed);
            } else if (key == "gyrodensity") {
                config.gyroNoiseDensity = toFloat(value, config.gyroNoiseDensity);
            } else if (key == "gyrowalk") {
                config.gyroRandomWalk = toFloat(value, config.gyroRandomWalk);
            } else if (key == "acceldensity") {
                config.accelNoiseDensity = toFloat(value, config.accelNoiseDensity);
            } else if (key == "accelwalk") {
                config.accelRandomWalk = toFloat(value, config.accelRandomWalk);
            } else if (key == "near") {
                config.nearPlane = toFloat(value, config.nearPlane);
            } else if (key == "far") {
                config.farPlane = toFloat(value, config.farPlane);
            } else if (key == "rangestddev") {
                config.rangeStddev = toFloat(value, config.rangeStddev);
            } else if (key == "rangepermetre") {
                config.rangeStddevPerMetre = toFloat(value, config.rangeStddevPerMetre);
            } else if (key == "rangebias") {
                config.rangeBias = toFloat(value, config.rangeBias);
            } else if (key == "fov") {
                config.fovY = toFloat(value, config.fovY);
            } else if (key == "width") {
                config.width = toInt(value, config.width);
            } else if (key == "height") {
                config.height = toInt(value, config.height);
            } else if (key == "beams") {
                config.beams = beamsFrom(value, config.beams);
            } else if (key == "facesize") {
                config.faceSize = toInt(value, config.faceSize);
            } else if (key == "joint") {
                config.joint = std::string(value);
            } else if (key == "encoderres") {
                config.encoderResolution = toFloat(value, config.encoderResolution);
            } else if (key == "contactthreshold") {
                config.contactForceThreshold = toFloat(value, config.contactForceThreshold);
            }
            // Unknown keys ignored on purpose: a document written by a newer
            // editor still loads here.
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }

    return config;
}

std::optional<SensorConfig> SensorConfig::read(const Object3D& object) {

    const auto it = object.userData.find(userDataKey);
    if (it == object.userData.end()) return std::nullopt;
    if (it->second.type() != typeid(std::string)) return std::nullopt;

    return decode(std::any_cast<const std::string&>(it->second));
}

void SensorConfig::write(Object3D& object) const {

    if (!enabled) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void SensorConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
