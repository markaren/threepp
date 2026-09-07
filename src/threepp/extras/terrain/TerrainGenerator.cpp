// Config (de)serialisation for the procedural terrain generator. The noise /
// erosion / bake logic is header-only (TerrainGenerator.hpp); only the JSON
// round-trip lives here so the public header doesn't pull in nlohmann (which is
// a PRIVATE threepp dependency) — consumers just see the declared free functions.

#include "threepp/extras/terrain/TerrainGenerator.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace threepp::terrain {

    void applyPreset(int preset, TerrainParams& p) {
        switch (preset) {
            case 0:// Alpine — ridged peaks
                p.noiseType = NoiseType::Ridged;
                p.worldSize = 1200.f; p.featureScale = 520.f; p.octaves = 6;
                p.lacunarity = 2.0f; p.gain = 0.5f; p.amplitude = 480.f;
                p.warp = 0.45f; p.ridgeSharpness = 0.5f; p.heightExponent = 1.25f;
                p.terraces = 0; p.falloff = Falloff::Radial; p.falloffStart = 0.62f;
                p.erosion = ErosionType::Both;
                p.droplets = 110000; p.erodeSpeed = 0.4f; p.erosionRadius = 3;
                p.talusAngle = 42.f; p.thermalIterations = 22;
                p.rockColor = {0.39f,0.36f,0.33f}; p.grassColor = {0.29f,0.33f,0.19f};
                p.screeColor = {0.49f,0.46f,0.42f}; p.snowColor = {0.90f,0.91f,0.93f};
                p.snowLine = 0.5f; p.slopeGrassMax = 0.28f; p.slopeRockMin = 0.55f;
                break;
            case 1:// Rolling Hills — soft fBm
                p.noiseType = NoiseType::fBm;
                p.worldSize = 1600.f; p.featureScale = 430.f; p.octaves = 6;
                p.lacunarity = 2.0f; p.gain = 0.5f; p.amplitude = 300.f;
                p.warp = 0.3f; p.ridgeSharpness = 0.0f; p.heightExponent = 1.0f;
                p.terraces = 0; p.falloff = Falloff::Radial; p.falloffStart = 0.8f;
                p.erosion = ErosionType::Thermal;
                p.talusAngle = 32.f; p.thermalIterations = 40;
                p.snowLine = 0.9f; p.slopeGrassMax = 0.4f;
                p.rockColor = {0.39f,0.36f,0.33f}; p.grassColor = {0.30f,0.40f,0.18f};
                p.screeColor = {0.49f,0.46f,0.42f}; p.snowColor = {0.90f,0.91f,0.93f};
                p.slopeRockMin = 0.55f;
                break;
            case 2:// Desert Mesa — terraced strata
                p.noiseType = NoiseType::Hybrid;
                p.worldSize = 1400.f; p.featureScale = 520.f; p.octaves = 6;
                p.lacunarity = 2.0f; p.gain = 0.45f; p.amplitude = 320.f;
                p.warp = 0.25f; p.ridgeSharpness = 0.2f; p.heightExponent = 1.0f;
                p.terraces = 10; p.falloff = Falloff::Radial; p.falloffStart = 0.72f;
                p.erosion = ErosionType::Hydraulic;
                p.droplets = 60000; p.erodeSpeed = 0.3f; p.depositSpeed = 0.4f;
                p.snowLine = 1.1f;
                p.rockColor = {0.46f,0.31f,0.22f}; p.grassColor = {0.60f,0.49f,0.33f};
                p.screeColor = {0.66f,0.52f,0.38f}; p.snowColor = {0.90f,0.91f,0.93f};
                p.slopeGrassMax = 0.28f; p.slopeRockMin = 0.55f;
                break;
            case 3:// Volcanic — single radial cone
                p.noiseType = NoiseType::Ridged;
                p.worldSize = 1200.f; p.featureScale = 360.f; p.octaves = 6;
                p.lacunarity = 2.2f; p.gain = 0.55f; p.amplitude = 600.f;
                p.warp = 0.35f; p.ridgeSharpness = 0.45f; p.heightExponent = 1.4f;
                p.terraces = 0; p.falloff = Falloff::Radial; p.falloffStart = 0.5f;
                p.erosion = ErosionType::Both;
                p.droplets = 100000; p.erodeSpeed = 0.45f; p.erosionRadius = 2;
                p.talusAngle = 40.f; p.thermalIterations = 22;
                p.snowLine = 1.1f;
                p.rockColor = {0.20f,0.18f,0.17f}; p.screeColor = {0.30f,0.27f,0.25f};
                p.grassColor = {0.24f,0.23f,0.21f}; p.snowColor = {0.90f,0.91f,0.93f};
                p.slopeGrassMax = 0.28f; p.slopeRockMin = 0.55f;
                break;
            default: break;
        }
    }

    std::string toJson(const TerrainParams& p) {
        nlohmann::json j;
        j["format"] = "threepp-terrain-1";
        j["seed"] = p.seed;
        j["worldSize"] = p.worldSize;
        j["resolution"] = p.resolution;
        j["noiseType"] = static_cast<int>(p.noiseType);
        j["featureScale"] = p.featureScale;
        j["octaves"] = p.octaves;
        j["lacunarity"] = p.lacunarity;
        j["gain"] = p.gain;
        j["amplitude"] = p.amplitude;
        j["warp"] = p.warp;
        j["ridgeSharpness"] = p.ridgeSharpness;
        j["heightExponent"] = p.heightExponent;
        j["terraces"] = p.terraces;
        j["falloff"] = static_cast<int>(p.falloff);
        j["falloffStart"] = p.falloffStart;
        j["erosion"] = static_cast<int>(p.erosion);
        j["droplets"] = p.droplets;
        j["dropletLifetime"] = p.dropletLifetime;
        j["inertia"] = p.inertia;
        j["sedimentCapacity"] = p.sedimentCapacity;
        j["minSlope"] = p.minSlope;
        j["erodeSpeed"] = p.erodeSpeed;
        j["depositSpeed"] = p.depositSpeed;
        j["evaporation"] = p.evaporation;
        j["gravity"] = p.gravity;
        j["erosionRadius"] = p.erosionRadius;
        j["talusAngle"] = p.talusAngle;
        j["thermalIterations"] = p.thermalIterations;
        j["thermalRate"] = p.thermalRate;
        j["snowLine"] = p.snowLine;
        j["snowNoiseAmp"] = p.snowNoiseAmp;
        j["snowSlopeMax"] = p.snowSlopeMax;
        j["slopeGrassMax"] = p.slopeGrassMax;
        j["slopeRockMin"] = p.slopeRockMin;
        j["bandEdge"] = p.bandEdge;
        j["rockColor"] = p.rockColor;
        j["grassColor"] = p.grassColor;
        j["screeColor"] = p.screeColor;
        j["snowColor"] = p.snowColor;
        return j.dump(2);
    }

    bool fromJson(const std::string& json, TerrainParams& p) {
        const auto j = nlohmann::json::parse(json, nullptr, false);
        if (j.is_discarded() || !j.is_object()) return false;

        // value(key, default) keeps the current field if the key is absent, so
        // older/partial configs load cleanly.
        p.seed = j.value("seed", p.seed);
        p.worldSize = j.value("worldSize", p.worldSize);
        p.resolution = j.value("resolution", p.resolution);
        p.noiseType = static_cast<NoiseType>(j.value("noiseType", static_cast<int>(p.noiseType)));
        p.featureScale = j.value("featureScale", p.featureScale);
        p.octaves = j.value("octaves", p.octaves);
        p.lacunarity = j.value("lacunarity", p.lacunarity);
        p.gain = j.value("gain", p.gain);
        p.amplitude = j.value("amplitude", p.amplitude);
        p.warp = j.value("warp", p.warp);
        p.ridgeSharpness = j.value("ridgeSharpness", p.ridgeSharpness);
        p.heightExponent = j.value("heightExponent", p.heightExponent);
        p.terraces = j.value("terraces", p.terraces);
        p.falloff = static_cast<Falloff>(j.value("falloff", static_cast<int>(p.falloff)));
        p.falloffStart = j.value("falloffStart", p.falloffStart);
        p.erosion = static_cast<ErosionType>(j.value("erosion", static_cast<int>(p.erosion)));
        p.droplets = j.value("droplets", p.droplets);
        p.dropletLifetime = j.value("dropletLifetime", p.dropletLifetime);
        p.inertia = j.value("inertia", p.inertia);
        p.sedimentCapacity = j.value("sedimentCapacity", p.sedimentCapacity);
        p.minSlope = j.value("minSlope", p.minSlope);
        p.erodeSpeed = j.value("erodeSpeed", p.erodeSpeed);
        p.depositSpeed = j.value("depositSpeed", p.depositSpeed);
        p.evaporation = j.value("evaporation", p.evaporation);
        p.gravity = j.value("gravity", p.gravity);
        p.erosionRadius = j.value("erosionRadius", p.erosionRadius);
        p.talusAngle = j.value("talusAngle", p.talusAngle);
        p.thermalIterations = j.value("thermalIterations", p.thermalIterations);
        p.thermalRate = j.value("thermalRate", p.thermalRate);
        p.snowLine = j.value("snowLine", p.snowLine);
        p.snowNoiseAmp = j.value("snowNoiseAmp", p.snowNoiseAmp);
        p.snowSlopeMax = j.value("snowSlopeMax", p.snowSlopeMax);
        p.slopeGrassMax = j.value("slopeGrassMax", p.slopeGrassMax);
        p.slopeRockMin = j.value("slopeRockMin", p.slopeRockMin);
        p.bandEdge = j.value("bandEdge", p.bandEdge);
        p.rockColor = j.value("rockColor", p.rockColor);
        p.grassColor = j.value("grassColor", p.grassColor);
        p.screeColor = j.value("screeColor", p.screeColor);
        p.snowColor = j.value("snowColor", p.snowColor);
        return true;
    }

    bool saveConfig(const std::string& filePath, const TerrainParams& p) {
        const std::filesystem::path path(filePath);
        if (path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
        }
        std::ofstream f(path);
        if (!f) return false;
        f << toJson(p);
        return f.good();
    }

    bool loadConfig(const std::string& filePath, TerrainParams& p) {
        std::ifstream f(filePath);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        return fromJson(ss.str(), p);
    }

}// namespace threepp::terrain
