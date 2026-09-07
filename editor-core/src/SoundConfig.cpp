#include "threepp/extras/editor/SoundConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;

namespace {

    SoundConfig::DistanceModel modelFrom(std::string_view text, SoundConfig::DistanceModel fallback) {

        if (text == "none") return SoundConfig::DistanceModel::None;
        if (text == "inverse") return SoundConfig::DistanceModel::Inverse;
        if (text == "linear") return SoundConfig::DistanceModel::Linear;
        if (text == "exponential") return SoundConfig::DistanceModel::Exponential;
        return fallback;
    }

    const char* modelToken(SoundConfig::DistanceModel model) {

        switch (model) {
            case SoundConfig::DistanceModel::None: return "none";
            case SoundConfig::DistanceModel::Inverse: return "inverse";
            case SoundConfig::DistanceModel::Linear: return "linear";
            case SoundConfig::DistanceModel::Exponential: return "exponential";
        }
        return "inverse";
    }

}// namespace


std::string SoundConfig::encode() const {

    std::string out;
    out += "positional=";
    out += positional ? "1" : "0";
    out += ";autoplay=";
    out += autoplay ? "1" : "0";
    out += ";loop=";
    out += loop ? "1" : "0";
    out += ";volume=";
    out += number(volume);
    out += ";rate=";
    out += number(rate);
    // Written whether or not the sound is positional: switching spatialization
    // off and on again must not silently reset the distance curve.
    out += ";minDistance=";
    out += number(minDistance);
    out += ";maxDistance=";
    out += number(maxDistance);
    out += ";rolloff=";
    out += number(rolloff);
    out += ";model=";
    out += modelToken(model);
    return out;
}

std::optional<SoundConfig> SoundConfig::decode(const std::string& text) {

    if (text.empty()) return std::nullopt;

    SoundConfig config;

    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "positional") {
            config.positional = toBool(value, config.positional);
        } else if (key == "autoplay") {
            config.autoplay = toBool(value, config.autoplay);
        } else if (key == "loop") {
            config.loop = toBool(value, config.loop);
        } else if (key == "volume") {
            config.volume = toFloat(value, config.volume);
        } else if (key == "rate") {
            config.rate = toFloat(value, config.rate);
        } else if (key == "minDistance") {
            config.minDistance = toFloat(value, config.minDistance);
        } else if (key == "maxDistance") {
            config.maxDistance = toFloat(value, config.maxDistance);
        } else if (key == "rolloff") {
            config.rolloff = toFloat(value, config.rolloff);
        } else if (key == "model") {
            config.model = modelFrom(value, config.model);
        }
        // Unknown keys ignored on purpose: a document written by a newer
        // editor still loads here.
    });

    return config;
}

std::optional<SoundConfig> SoundConfig::read(const Object3D& object) {

    const auto text = readString(object, userDataKey);
    if (text.empty()) {
        // A file with no parameter entry is a sound at its defaults — which is
        // exactly what write() leaves behind for one nobody has tuned.
        if (readString(object, fileKey).empty()) return std::nullopt;
        return SoundConfig{};
    }
    return decode(text);
}

bool SoundConfig::isSound(const Object3D& object) {

    return read(object).has_value();
}

std::string SoundConfig::file(const Object3D& object) {

    return readString(object, fileKey);
}

void SoundConfig::setFile(Object3D& object, const std::string& path) {

    if (path.empty()) {
        object.userData.erase(fileKey);
        return;
    }
    object.userData[fileKey] = path;
}

void SoundConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void SoundConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
    object.userData.erase(fileKey);
}

std::filesystem::path SoundConfig::resolveFile(const std::string& stored,
                                               const std::filesystem::path& documentDir) {

    if (stored.empty()) return {};

    std::filesystem::path path{stored};
    if (path.is_relative() && !documentDir.empty()) path = documentDir / path;
    return path;
}

SoundAuthoring SoundAuthoring::read(const Object3D& object) {

    return {SoundConfig::read(object), SoundConfig::file(object)};
}

void SoundAuthoring::write(Object3D& object) const {

    if (!config) {
        SoundConfig::erase(object);
        return;
    }
    config->write(object);
    SoundConfig::setFile(object, file);
}

const char* SoundConfig::label(DistanceModel model) {

    switch (model) {
        case DistanceModel::None: return "None";
        case DistanceModel::Inverse: return "Inverse";
        case DistanceModel::Linear: return "Linear";
        case DistanceModel::Exponential: return "Exponential";
    }
    return "Inverse";
}
