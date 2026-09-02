
#include "threepp/extras/editor/SplatImportConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <string>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;


std::optional<SplatImportConfig> SplatImportConfig::read(const Object3D& object) {

    const auto source = readString(object, sourceKey);
    if (source.empty()) return std::nullopt;

    SplatImportConfig config;
    config.source = source;

    parsePairs(readString(object, opsKey), [&](std::string_view key, std::string_view value) {
        if (key == "cull") config.culled = toBool(value, false);
        else if (key == "removed") config.removed = static_cast<std::size_t>(std::max(0, toInt(value, 0)));
        else if (key == "flipX") config.flippedX = toBool(value, false);
        else if (key == "lod") config.lod = toInt(value, -1);
        else if (key == "points") config.pointCloud = toBool(value, false);
    });

    return config;
}

void SplatImportConfig::write(Object3D& object) const {

    if (source.empty()) {
        erase(object);
        return;
    }
    object.userData[sourceKey] = source;

    // Written in full every time, defaults included. Unlike the other configs
    // this is a record of what already happened rather than a setting, and an
    // absent key would read as "not culled" where the truth is "unknown".
    std::string ops;
    ops += "cull=";
    ops += culled ? "1" : "0";
    ops += ";removed=" + std::to_string(removed);
    ops += ";flipX=";
    ops += flippedX ? "1" : "0";
    ops += ";lod=" + std::to_string(lod);
    ops += ";points=";
    ops += pointCloud ? "1" : "0";
    object.userData[opsKey] = ops;
}

void SplatImportConfig::erase(Object3D& object) {

    object.userData.erase(sourceKey);
    object.userData.erase(opsKey);
}
