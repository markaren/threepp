
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <any>
#include <string_view>

using namespace threepp;
using namespace threepp::editor;
using namespace threepp::editor::codec;


std::string SplatSurfaceConfig::encode() const {

    std::string out;
    out += "enabled=";
    out += enabled ? "1" : "0";
    out += ";voxel=";
    out += number(voxelSize);
    out += ";island=";
    out += std::to_string(minComponentVoxels);
    out += ";poses=";
    out += std::to_string(poseCount);
    out += ";interior=";
    out += interior ? "1" : "0";
    out += ";method=";
    out += std::to_string(method);
    out += ";body=";
    out += std::to_string(body);
    out += ";mass=";
    out += number(mass);
    out += ";hulls=";
    out += std::to_string(hulls);
    return out;
}

std::optional<SplatSurfaceConfig> SplatSurfaceConfig::decode(const std::string& text) {

    SplatSurfaceConfig config;

    parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "enabled") {
            config.enabled = toBool(value, config.enabled);
        } else if (key == "voxel") {
            config.voxelSize = toFloat(value, config.voxelSize);
        } else if (key == "island") {
            config.minComponentVoxels = toInt(value, config.minComponentVoxels);
        } else if (key == "poses") {
            config.poseCount = toInt(value, config.poseCount);
        } else if (key == "interior") {
            config.interior = toBool(value, config.interior);
        } else if (key == "method") {
            config.method = toInt(value, config.method);
        } else if (key == "body") {
            config.body = toInt(value, config.body);
        } else if (key == "mass") {
            config.mass = toFloat(value, config.mass);
        } else if (key == "hulls") {
            config.hulls = toInt(value, config.hulls);
        }
    });
    config.method = std::clamp(config.method, static_cast<int>(SplatSurfaceConfig::Auto),
                               static_cast<int>(SplatSurfaceConfig::Points));
    config.body = std::clamp(config.body, static_cast<int>(SplatSurfaceConfig::Static),
                             static_cast<int>(SplatSurfaceConfig::Kinematic));
    config.mass = std::max(config.mass, 1e-3f);
    config.hulls = std::clamp(config.hulls, 1, 256);

    // 0 means "let the bake decide" for both derived knobs, so the floor is 0
    // and not a minimum sane value.
    config.voxelSize = std::max(config.voxelSize, 0.f);
    config.minComponentVoxels = std::max(config.minComponentVoxels, 0);
    config.poseCount = std::max(config.poseCount, 0);

    return config;
}

std::optional<SplatSurfaceConfig> SplatSurfaceConfig::read(const Object3D& object) {

    return readEntry<SplatSurfaceConfig>(object, userDataKey);
}

void SplatSurfaceConfig::write(Object3D& object) const {

    if (!enabled) {
        erase(object);
        return;
    }
    object.userData[userDataKey] = encode();
}

void SplatSurfaceConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}
