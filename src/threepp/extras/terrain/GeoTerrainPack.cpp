// Region-pack loader implementation. The JSON round-trip lives here (nlohmann is
// a PRIVATE threepp dependency) so GeoTerrainPack.hpp stays dependency-free —
// mirrors TerrainGenerator.cpp's split.

#include "threepp/extras/terrain/GeoTerrainPack.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace threepp::terrain {

    namespace {

        std::string readTextFile(const std::filesystem::path& p) {
            std::ifstream f(p, std::ios::binary);
            if (!f) throw std::runtime_error("GeoTerrainPack: cannot open " + p.string());
            std::ostringstream ss;
            ss << f.rdbuf();
            return ss.str();
        }

        nlohmann::json parseJsonFile(const std::filesystem::path& p) {
            const std::string text = readTextFile(p);
            const auto j = nlohmann::json::parse(text, nullptr, false);
            if (j.is_discarded())
                throw std::runtime_error("GeoTerrainPack: malformed JSON in " + p.string());
            return j;
        }

    }// namespace

    GeoTerrainPack GeoTerrainPack::load(const std::string& path) {
        namespace fs = std::filesystem;
        const fs::path packDir(path);
        if (!fs::is_directory(packDir))
            throw std::runtime_error("GeoTerrainPack: not a directory: " + packDir.string());

        // ── region.json ──────────────────────────────────────────────────────
        const fs::path regionPath = packDir / "region.json";
        const nlohmann::json rj = parseJsonFile(regionPath);
        if (!rj.is_object())
            throw std::runtime_error("GeoTerrainPack: region.json is not an object");

        GeoTerrainPack pack;
        GeoRegion& r = pack.region;
        r.version = rj.value("version", 1);
        r.name = rj.value("name", std::string{});
        r.crs = rj.value("crs", std::string{});
        r.originEasting = rj.value("originEasting", 0.0);
        r.originNorthing = rj.value("originNorthing", 0.0);
        r.worldSize = rj.value("worldSize", 0.f);
        r.dim = rj.value("dim", 0);
        r.heightMin = rj.value("heightMin", 0.f);
        r.heightMax = rj.value("heightMax", 0.f);
        r.seaLevel = rj.value("seaLevel", 0.f);
        r.attribution = rj.value("attribution", std::string{});

        if (r.dim < 4)
            throw std::runtime_error("GeoTerrainPack: region.json dim < 4 (" + std::to_string(r.dim) + ")");
        if (!(r.worldSize > 0.f))
            throw std::runtime_error("GeoTerrainPack: region.json worldSize must be > 0");

        // ── heights.f32 ──────────────────────────────────────────────────────
        const std::string heightsName = rj.value("heights", std::string("heights.f32"));
        const fs::path heightsPath = packDir / heightsName;
        std::error_code ec;
        const auto fileSize = fs::file_size(heightsPath, ec);
        if (ec)
            throw std::runtime_error("GeoTerrainPack: cannot stat " + heightsPath.string());

        const std::uint64_t expected =
                static_cast<std::uint64_t>(r.dim) * static_cast<std::uint64_t>(r.dim) * 4ull;
        if (fileSize != expected)
            throw std::runtime_error("GeoTerrainPack: " + heightsPath.string() + " is " +
                                     std::to_string(fileSize) + " bytes, expected " +
                                     std::to_string(expected) + " (dim*dim*4, dim=" +
                                     std::to_string(r.dim) + ")");

        std::vector<float> heights(static_cast<size_t>(r.dim) * static_cast<size_t>(r.dim));
        {
            std::ifstream hf(heightsPath, std::ios::binary);
            if (!hf) throw std::runtime_error("GeoTerrainPack: cannot open " + heightsPath.string());
            hf.read(reinterpret_cast<char*>(heights.data()),
                    static_cast<std::streamsize>(expected));
            if (hf.gcount() != static_cast<std::streamsize>(expected))
                throw std::runtime_error("GeoTerrainPack: short read on " + heightsPath.string());
        }
        // Row-major [iz*dim+ix], spanning [-worldSize/2,+worldSize/2] around the
        // origin — the pack's stated layout. Drops straight into HeightGrid.
        pack.grid = HeightGrid(std::move(heights), r.dim, r.worldSize);

        // ── roads.json ───────────────────────────────────────────────────────
        const std::string roadsName = rj.value("roads", std::string("roads.json"));
        const fs::path roadsPath = packDir / roadsName;
        if (fs::exists(roadsPath)) {
            const nlohmann::json roadDoc = parseJsonFile(roadsPath);
            const auto itRoads = roadDoc.find("roads");
            if (itRoads != roadDoc.end() && itRoads->is_array()) {
                // The producer may emit id / typeVeg as either a string or a
                // number (e.g. NVDB id as an integer, typeVeg as a label), so
                // coerce both to string rather than assuming a JSON type.
                const auto toStr = [](const nlohmann::json& v) -> std::string {
                    if (v.is_string()) return v.get<std::string>();
                    if (v.is_number_integer()) return std::to_string(v.get<long long>());
                    if (v.is_number_unsigned()) return std::to_string(v.get<unsigned long long>());
                    if (v.is_number_float()) return std::to_string(v.get<double>());
                    return {};
                };
                for (const auto& rd : *itRoads) {
                    if (!rd.is_object()) continue;
                    GeoRoad road;
                    if (auto it = rd.find("id"); it != rd.end()) road.id = toStr(*it);
                    road.category = rd.value("category", std::string{});
                    if (auto it = rd.find("typeVeg"); it != rd.end()) road.typeVeg = toStr(*it);
                    road.width = rd.value("width", 6.f);

                    const auto itPts = rd.find("points");
                    if (itPts == rd.end() || !itPts->is_array()) continue;
                    road.points.reserve(itPts->size());
                    for (const auto& p : *itPts) {
                        if (!p.is_array() || p.size() < 3) continue;
                        road.points.emplace_back(
                                p[0].get<float>(), p[1].get<float>(), p[2].get<float>());
                    }
                    // A polyline needs at least two points to build a ribbon.
                    if (road.points.size() >= 2) pack.roads.push_back(std::move(road));
                }
            }
        }

        // ── buildings.json (optional) ────────────────────────────────────────
        const std::string bldName = rj.value("buildings", std::string("buildings.json"));
        const fs::path bldPath = packDir / bldName;
        if (fs::exists(bldPath)) {
            const nlohmann::json bldDoc = parseJsonFile(bldPath);
            const auto itBlds = bldDoc.find("buildings");
            if (itBlds != bldDoc.end() && itBlds->is_array()) {
                const auto parseRing = [](const nlohmann::json& arr) {
                    std::vector<Vector2> ring;
                    ring.reserve(arr.size());
                    for (const auto& p : arr) {
                        if (!p.is_array() || p.size() < 2) continue;
                        ring.emplace_back(p[0].get<float>(), p[1].get<float>());
                    }
                    return ring;
                };
                for (const auto& bd : *itBlds) {
                    if (!bd.is_object()) continue;
                    GeoBuilding b;
                    b.id = bd.value("id", std::string{});
                    b.type = bd.value("type", std::string{});
                    b.height = bd.value("height", 6.f);
                    b.groundMin = bd.value("groundMin", 0.f);
                    b.groundMax = bd.value("groundMax", 0.f);
                    b.levels = bd.value("levels", 0.f);
                    b.heightSource = bd.value("heightSource", std::string{});
                    b.colour = bd.value("colour", std::string{});
                    b.roofColour = bd.value("roofColour", std::string{});
                    b.roofShape = bd.value("roofShape", std::string{});
                    const auto itOuter = bd.find("outer");
                    if (itOuter == bd.end() || !itOuter->is_array()) continue;
                    b.outer = parseRing(*itOuter);
                    if (b.outer.size() < 3) continue;// a footprint needs a polygon
                    if (auto itHoles = bd.find("holes"); itHoles != bd.end() && itHoles->is_array()) {
                        for (const auto& h : *itHoles) {
                            if (!h.is_array()) continue;
                            auto ring = parseRing(h);
                            if (ring.size() >= 3) b.holes.push_back(std::move(ring));
                        }
                    }
                    pack.buildings.push_back(std::move(b));
                }
            }
        }

        return pack;
    }

}// namespace threepp::terrain
