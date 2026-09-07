
#include "threepp/extras/editor/FlockConfig.hpp"

#include "threepp/extras/editor/detail/ConfigCodec.hpp"

#include "threepp/core/Object3D.hpp"

#include <sstream>

using namespace threepp;
using namespace threepp::editor;


std::string FlockConfig::encode() const {

    std::ostringstream out;
    out << "seed=" << seed
        << ";birdCount=" << birdCount
        << ";massKg=" << codec::number(massKg)
        << ";roamRadius=" << codec::number(roamRadius)
        << ";cruiseAltitude=" << codec::number(cruiseAltitude)
        << ";altitudeSpread=" << codec::number(altitudeSpread)
        << ";cruiseSpeed=" << codec::number(cruiseSpeed)
        << ";perching=" << (perching ? 1 : 0)
        << ";maxPerchedFraction=" << codec::number(maxPerchedFraction)
        << ";castShadow=" << (castShadow ? 1 : 0)
        << ";windX=" << codec::number(windX)
        << ";windZ=" << codec::number(windZ);
    return out.str();
}

std::optional<FlockConfig> FlockConfig::decode(const std::string& text) {

    FlockConfig config;
    codec::parsePairs(text, [&](std::string_view key, std::string_view value) {
        if (key == "seed") config.seed = codec::toInt(value, config.seed);
        else if (key == "birdCount") config.birdCount = codec::toInt(value, config.birdCount);
        else if (key == "massKg") config.massKg = codec::toFloat(value, config.massKg);
        else if (key == "roamRadius") config.roamRadius = codec::toFloat(value, config.roamRadius);
        else if (key == "cruiseAltitude") config.cruiseAltitude = codec::toFloat(value, config.cruiseAltitude);
        else if (key == "altitudeSpread") config.altitudeSpread = codec::toFloat(value, config.altitudeSpread);
        else if (key == "cruiseSpeed") config.cruiseSpeed = codec::toFloat(value, config.cruiseSpeed);
        else if (key == "perching") config.perching = codec::toBool(value, config.perching);
        else if (key == "maxPerchedFraction") config.maxPerchedFraction = codec::toFloat(value, config.maxPerchedFraction);
        else if (key == "castShadow") config.castShadow = codec::toBool(value, config.castShadow);
        else if (key == "windX") config.windX = codec::toFloat(value, config.windX);
        else if (key == "windZ") config.windZ = codec::toFloat(value, config.windZ);
        // Unknown keys are ignored — a newer editor's document still loads.
    });
    return config;
}

std::optional<FlockConfig> FlockConfig::read(const Object3D& object) {

    return codec::readEntry<FlockConfig>(object, userDataKey);
}

void FlockConfig::write(Object3D& object) const {

    object.userData[userDataKey] = encode();
}

void FlockConfig::erase(Object3D& object) {

    object.userData.erase(userDataKey);
}

bool FlockConfig::isFlock(const Object3D& object) {

    return codec::hasEntry(object, userDataKey);
}

Flock::Params FlockConfig::makeParams(const Vector3& home) const {

    Flock::Params p;
    p.seed = static_cast<unsigned int>(seed < 0 ? 0 : seed);
    p.birdCount = birdCount;
    p.massKg = massKg;
    p.home = home;
    p.roamRadius = roamRadius;
    p.cruiseAltitude = cruiseAltitude;
    p.altitudeSpread = altitudeSpread;
    p.cruiseSpeed = cruiseSpeed;
    p.perching = perching;
    p.maxPerchedFraction = maxPerchedFraction;
    p.birdsCastShadow = castShadow;
    p.wind.set(windX, windZ);
    return p;
}
