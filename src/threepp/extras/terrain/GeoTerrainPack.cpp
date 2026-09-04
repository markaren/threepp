// Region-pack loader implementation. The JSON round-trip lives here (nlohmann is
// a PRIVATE threepp dependency) so GeoTerrainPack.hpp stays dependency-free —
// mirrors TerrainGenerator.cpp's split.

#include "threepp/extras/terrain/GeoTerrainPack.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
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

    HeightGrid computeFlowAccumulation(const HeightGrid& dem, float targetCell) {
        if (!dem.valid()) return {};

        const int dim = dem.dim();
        const float world = dem.worldSize();
        const float demStep = world / static_cast<float>(dim - 1);
        // Routing grid is capped at 2048 per side: the sort is O(N log N) over
        // every cell, and an uncapped 8 km / 2 m pack (4001²  = 16 M cells)
        // adds seconds to load for a field that is only ever sampled at metre
        // scale. The cap costs the 1 m pack nothing (it already wants stride 2).
        constexpr int kMaxFlowDim = 2048;
        const int strideCap = (dim - 2) / (kMaxFlowDim - 1) + 1;
        const int stride = std::max({1, strideCap, static_cast<int>(std::lround(targetCell / demStep))});
        const int fd = (dim - 1) / stride + 1;
        if (fd < 4) return {};
        const float cell = world / static_cast<float>(fd - 1);

        // ── routing DEM: box mean over each stride×stride window ─────────────
        // The mean (not a point sample) is what removes the single-cell lidar
        // pits that would otherwise terminate a fall line halfway down a wall.
        const std::vector<float>& src = dem.data();
        std::vector<float> h(static_cast<size_t>(fd) * fd, 0.f);
        for (int jz = 0; jz < fd; ++jz) {
            const int z0 = jz * stride, z1 = std::min(z0 + stride, dim);
            for (int jx = 0; jx < fd; ++jx) {
                const int x0 = jx * stride, x1 = std::min(x0 + stride, dim);
                double s = 0.0;
                int n = 0;
                for (int z = z0; z < z1; ++z)
                    for (int x = x0; x < x1; ++x) {
                        s += src[static_cast<size_t>(z) * dim + x];
                        ++n;
                    }
                h[static_cast<size_t>(jz) * fd + jx] = n ? static_cast<float>(s / n) : 0.f;
            }
        }

        // ── D8: one pass, cells visited high → low ──────────────────────────
        // Processing in descending height order guarantees a cell's own inflow
        // is complete before it donates downstream, so a single linear sweep
        // gives exact accumulation — no iteration to convergence.
        const size_t n = h.size();
        std::vector<std::uint32_t> order(n);
        for (std::uint32_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&h](std::uint32_t a, std::uint32_t b) { return h[a] > h[b]; });

        std::vector<float> acc(n, 1.f);// every cell drains at least itself
        static constexpr int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        static constexpr int DZ[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        for (const std::uint32_t idx : order) {
            const int jx = static_cast<int>(idx % fd);
            const int jz = static_cast<int>(idx / fd);
            const float hc = h[idx];
            int best = -1;
            float bestGrad = 0.f;
            for (int k = 0; k < 8; ++k) {
                const int nx = jx + DX[k], nz = jz + DZ[k];
                if (nx < 0 || nz < 0 || nx >= fd || nz >= fd) continue;
                const size_t ni = static_cast<size_t>(nz) * fd + nx;
                // Steepest DESCENT per unit distance — the diagonal step is
                // √2 longer, so an unweighted drop comparison would bias the
                // routing into diagonal staircases.
                const float drop = hc - h[ni];
                if (drop <= 0.f) continue;
                const float dist = (DX[k] && DZ[k]) ? 1.41421356f : 1.f;
                const float g = drop / dist;
                if (g > bestGrad) {
                    bestGrad = g;
                    best = static_cast<int>(ni);
                }
            }
            if (best >= 0) acc[best] += acc[idx];// no lower neighbour ⇒ sink, stops
        }

        // ── log-normalise to 0..1 ───────────────────────────────────────────
        const float cellArea = cell * cell;
        const float denom = std::log1p(static_cast<float>(n) * cellArea);
        for (size_t i = 0; i < n; ++i)
            acc[i] = std::clamp(std::log1p(acc[i] * cellArea) / denom, 0.f, 1.f);

        return HeightGrid(std::move(acc), fd, world);
    }

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

        // ── canopy.u8 (optional) ─────────────────────────────────────────────
        // Same node layout as heights.f32, one unsigned byte per node holding
        // canopy height in units of `canopyScale` metres (0.25 by default, so
        // 0..63.75 m). Missing file = pack predates the CHM: not an error, the
        // grid simply stays invalid and every consumer falls back.
        if (auto itCanopy = rj.find("canopy"); itCanopy != rj.end() && itCanopy->is_string()) {
            const fs::path canopyPath = packDir / itCanopy->get<std::string>();
            const float scale = rj.value("canopyScale", 0.25f);
            const std::uint64_t expectC =
                    static_cast<std::uint64_t>(r.dim) * static_cast<std::uint64_t>(r.dim);
            std::error_code ecc;
            if (fs::exists(canopyPath) && fs::file_size(canopyPath, ecc) == expectC && !ecc) {
                std::vector<unsigned char> raw(static_cast<size_t>(expectC));
                std::ifstream cf(canopyPath, std::ios::binary);
                if (cf) {
                    cf.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(expectC));
                    if (cf.gcount() == static_cast<std::streamsize>(expectC)) {
                        std::vector<float> canopy(raw.size());
                        for (size_t i = 0; i < raw.size(); ++i)
                            canopy[i] = static_cast<float>(raw[i]) * scale;
                        pack.canopy = HeightGrid(std::move(canopy), r.dim, r.worldSize);
                    }
                }
            }
        }

        // ── flow accumulation (derived, always) ──────────────────────────────
        pack.flow = computeFlowAccumulation(pack.grid);

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
