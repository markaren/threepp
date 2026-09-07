
#include "threepp/extras/editor/AcousticSurfaceConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;


std::string AcousticSurfaceConfig::encode() const {

    std::string out;
    out += "enabled=";
    out += enabled ? "1" : "0";
    out += ";transmission=";
    out += number(transmission);
    out += ";absorption=";
    out += number(absorption);
    return out;
}

std::optional<AcousticSurfaceConfig> AcousticSurfaceConfig::decode(const std::string& text) {

    AcousticSurfaceConfig config;

    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "enabled") {
            config.enabled = toBool(value, config.enabled);
        } else if (key == "transmission") {
            config.transmission = toFloat(value, config.transmission);
        } else if (key == "absorption") {
            config.absorption = toFloat(value, config.absorption);
        }
    });

    config.transmission = std::clamp(config.transmission, 0.f, 1.f);
    config.absorption = std::clamp(config.absorption, 0.f, 1.f);

    return config;
}

std::optional<AcousticSurfaceConfig> AcousticSurfaceConfig::read(const Object3D& object) {

    return readEntry<AcousticSurfaceConfig>(object, userDataKey);
}

void AcousticSurfaceConfig::write(Object3D& object) const {

    if (!enabled) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void AcousticSurfaceConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
