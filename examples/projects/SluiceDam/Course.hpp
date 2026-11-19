// The sluice dam example's ground: a tile of forest floor with a stream on it, upstream to downstream (the ground
// falls toward +z): a brook from the tile's edge into a mill pond, a log crib dam with a sluice gate, a gorge below
// it, a boulder rapid, a rock ledge the stream falls over, and a small lake at the bottom of the tile. All of it is
// pure functions of (x, z), so the terrain mesh and the water's bed are the same surface.

#ifndef THREEPP_EXAMPLE_SLUICE_COURSE_HPP
#define THREEPP_EXAMPLE_SLUICE_COURSE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace sluice {

    // the tile and the grid both the terrain mesh and the water use (cell centres)
    constexpr float tileX0 = -12, tileX1 = 12, tileZ0 = -24, tileZ1 = 24;
    constexpr float cellSize = 0.15f;
    constexpr int columns = 160, rows = 320;

    // the dam: its sill's centre at (damX, sill, damZ), +z downstream
    constexpr float damX = 0, damZ = -8;
    constexpr float crest = 1.5f;        // the crib's top: the pond's spill level
    constexpr float sill = crest - 1.15f;// the sluice's sill
    constexpr float cribHalf = 0.45f;    // half the crib's thickness (along z)
    constexpr float cribWidth = 3.6f;    // half its length (along x)
    constexpr float gateHalf = 0.42f;    // half the sluice's opening
    constexpr float gatePlane = -0.35f;  // the gate's plane from the crib's middle (z)
    constexpr float gateMax = 1.1f;      // the gate's travel
    constexpr float gateStart = 0.06f;   // its opening at the start: the pond a few cm under its crest
    constexpr float gateLeak = 0.008f;   // shut, old planks still let this much through
    constexpr float winch = 0.22f;       // m/s: how fast the windlass moves the gate

    constexpr float inflow = 0.12f;// m^3/s: the brook

    // the pond (an ellipse; the dam cuts its downstream end), the lake at the tile's bottom
    constexpr float pondX = -0.4f, pondZ = -12.8f, pondRX = 4.2f, pondRZ = 5.6f;
    constexpr float lakeX = 0.5f, lakeZ = 16.6f, lakeRX = 6.6f, lakeRZ = 4.6f;
    constexpr float lakeLevel = -1.75f;

    inline float smoothstep(float a, float b, float v) {
        const float t = std::clamp((v - a) / (b - a), 0.f, 1.f);
        return t * t * (3 - 2 * t);
    }

    // polynomial smooth minimum: rounds the crease where a cut meets the ground
    inline float smin(float a, float b, float k) {
        const float h = std::clamp(0.5f + 0.5f * (b - a) / k, 0.f, 1.f);
        return b * (1 - h) + a * h - k * h * (1 - h);
    }

    inline float interpolate(const std::vector<std::pair<float, float>>& table, float x) {
        if (x <= table.front().first) return table.front().second;
        for (size_t k = 1; k < table.size(); ++k) {
            if (x <= table[k].first) {
                const auto& [x0, y0] = table[k - 1];
                const auto& [x1, y1] = table[k];
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            }
        }
        return table.back().second;
    }

    struct Rock {
        float x, y, z;// centre
        float radius, squash, yaw;
        bool inWater;
    };

    class Course {

    public:
        Course() {
            planRocks();
        }

        // ---- the ground ------------------------------------------------------------------------------------------
        // the forest floor before the stream: falling toward +z (gently above the dam, faster below), low hummocks,
        // rising toward the tile's sides
        [[nodiscard]] static float base(float x, float z) {
            float y = z < damZ ? 1.6f + 0.05f * (damZ - z) : 1.6f - 0.11f * (z - damZ);
            y += 0.16f * std::sin(0.7f * x + 0.3f) * std::sin(0.5f * z + 1.1f);
            y += 0.08f * std::sin(1.7f * x + 2.0f * z) + 0.05f * std::sin(3.1f * x - 2.3f * z + 0.7f);
            const float side = std::max(std::abs(x) - 5.f, 0.f);
            return y + 0.035f * side * side;
        }

        // the stream's centre line below the dam (x of z): a gentle meander, straight into the lake
        [[nodiscard]] static float centreX(float z) {
            const float meander = 0.55f * std::sin(0.32f * (z - damZ));
            return damX + meander * (1 - smoothstep(7.f, 10.f, z)) + (lakeX - damX) * smoothstep(9.f, 13.f, z);
        }

        // the height of the ground at (x, z): the pond's basin and bank, the lake's, the brook and the gorge cut,
        // the bowl at the dam's foot
        [[nodiscard]] float height(float x, float z) const {
            float h = base(x, z);
            h = basin(x, z, h, pondSd(x, z), pondDepth(z), crest);
            h = basin(x, z, h, lakeSd(x, z), 1.0f, lakeLevel + 0.04f);
            if (z >= brookZ0 - 0.4f && z <= brookZ1) h = std::min(h, smin(h, brookCut(x, z), 0.15f));
            if (z >= damZ && z <= valleyEnd) h = std::min(h, smin(h, valley(x, z), 0.3f));
            // the bowl at the crib's foot across nearly its whole width (the whole log face shows from downstream)
            const float lx = x - damX, lz = z - damZ;
            if (lz > 0 && lz < 6 && std::abs(lx) < 7) {
                const float ax = std::abs(lx), out = std::max(ax - tailHalf, 0.f);
                const float bowl = sill - 0.45f + 0.3f * std::pow(std::clamp(ax / tailHalf, 0.f, 1.f), 2.f) + 1.1f * out +
                                   0.4f * out * out + 0.9f * std::pow(std::max(lz - 1.6f, 0.f), 1.5f);
                h = std::min(h, smin(h, bowl, 0.25f));
            }
            // inside the crib's footprint the ground stays under its top
            if (std::abs(lx) < cribWidth && std::abs(lz) < cribHalf + 0.05f) h = std::min(h, crest - 0.08f);
            return h;
        }

        // ---- the water's places ----------------------------------------------------------------------------------
        // about the signed distance to the pond's shore (negative in the water); past the crib's middle plane the
        // pond does nothing (the step from its basin to the gorge lies inside the crib)
        [[nodiscard]] static float pondSd(float x, float z) {
            if (z > damZ) return 9;
            return ellipseSd(x - pondX, z - pondZ, pondRX, pondRZ, 0.4f);
        }

        [[nodiscard]] static float lakeSd(float x, float z) {
            return ellipseSd(x - lakeX, z - lakeZ, lakeRX, lakeRZ, 2.3f);
        }

        // signed distance across the stream below the dam (+ to its right looking downstream), with its floor's
        // height, half width and bank slope there
        void stream(float x, float z, float& across, float& floor, float& halfWidth, float& bank) const {
            const float zc = std::clamp(z, damZ, valleyEnd);
            const float h = 0.05f;
            const float slope = (centreX(zc + h) - centreX(zc - h)) / (2 * h);
            across = (centreX(zc) - x) / std::sqrt(1 + slope * slope);
            floor = interpolate(thalweg_, z);
            halfWidth = interpolate(width_, z);
            bank = interpolate(bankSlope_, z);
        }

        // where the water may go: the pond, the lake, the brook, the stream (with a margin)
        [[nodiscard]] bool wet(float x, float z, float margin) const {
            if (pondSd(x, z) < margin || lakeSd(x, z) < margin) return true;
            if (z >= brookZ0 - 0.4f && z <= brookZ1 && std::abs(x - brookX(z)) < brookHalf + 0.5f + margin) return true;
            if (z >= damZ && z <= valleyEnd) {
                float d, floor, w, k;
                stream(x, z, d, floor, w, k);
                if (std::abs(d) < w + 1.0f + margin) return true;
            }
            return false;
        }

        // where nothing grows: the water and the stream's banks
        [[nodiscard]] bool clear(float x, float z, float margin) const {
            if (wet(x, z, margin + 0.5f)) return true;
            const float lx = x - damX, lz = z - damZ;
            if (std::abs(lx) < cribWidth + 1.5f + margin && lz > -1.5f - margin && lz < 4.0f + margin) return true;
            if (z >= damZ && z <= valleyEnd) {
                float d, floor, w, k;
                stream(x, z, d, floor, w, k);
                const float top = w + std::max(base(x, z) - floor, 0.f) / std::max(k, 0.3f);
                if (std::abs(d) < top + 0.8f + margin) return true;
            }
            return false;
        }

        // the brook's head, where the inflow comes in (cells of the brook within this z range)
        [[nodiscard]] static bool spring(float x, float z) {
            return z > tileZ0 && z < tileZ0 + 0.6f && std::abs(x - brookX(z)) < brookHalf;
        }

        // the lake's outlet: cells held at its level (a hidden drain on its west shore)
        [[nodiscard]] static bool drain(float x, float z) {
            return std::hypot(x - (lakeX - 5.0f), z - (lakeZ + 0.6f)) < 1.0f && lakeSd(x, z) < -0.3f;
        }

        [[nodiscard]] const std::vector<Rock>& rocks() const { return rocks_; }

    private:
        static constexpr float brookZ0 = tileZ0, brookZ1 = -16.5f, brookHalf = 0.32f;
        static constexpr float valleyEnd = 13.f;
        static constexpr float tailHalf = 2.9f;

        // the thalweg below the dam (z, height): the plunge pool and its lip, the rapid, the ledge, the reach into
        // the lake; the floor's half width and the banks' slope along it
        std::vector<std::pair<float, float>> thalweg_{{-8.0f, sill - 0.55f}, {-7.0f, sill - 0.65f}, {-5.6f, 0.05f},
                                                      {-4.0f, -0.05f},       {0.0f, -0.3f},          {4.0f, -0.55f},
                                                      {8.5f, -0.85f},        {8.75f, -0.9f},         {9.15f, -2.25f},
                                                      {10.5f, -2.4f},        {13.0f, -2.6f}};
        std::vector<std::pair<float, float>> width_{{-8.0f, 1.1f}, {-6.8f, 1.3f}, {-5.6f, 0.9f}, {0.0f, 0.85f},
                                                    {8.0f, 0.8f},  {9.2f, 1.2f},  {13.0f, 1.6f}};
        std::vector<std::pair<float, float>> bankSlope_{{-8.0f, 0.8f}, {-5.6f, 1.0f}, {0.0f, 1.0f}, {8.0f, 1.3f},
                                                        {9.2f, 1.5f},  {13.0f, 0.6f}};
        std::vector<Rock> rocks_;

        static float ellipseSd(float dx, float dz, float rx, float rz, float phase) {
            const float th = std::atan2(dz / rz, dx / rx);
            const float rho = std::hypot(dx / rx, dz / rz);
            const float r = 1 + 0.07f * std::sin(2 * th + phase) + 0.05f * std::sin(3 * th + 1.9f + phase) +
                            0.03f * std::sin(5 * th + 3.3f);
            return (rho - r) * std::hypot(rx * std::cos(th), rz * std::sin(th));
        }

        // the pond's depth at the dam and toward its head
        static float pondDepth(float z) {
            return 0.45f + 0.65f * smoothstep(pondZ - pondRZ, damZ, z);
        }

        // a basin under `level` inside a shore (signed distance s): the floor falls from the waterline at a slope,
        // the bank rises from it at a slope to 18 cm over the level and eases back into the ground over 3.5 m (low
        // ground raised to it, high ground cut back)
        static float basin(float, float, float h, float s, float depth, float level) {
            if (s > 5) return h;
            const float waterline = level - 0.04f, rim = level + 0.18f;
            if (s < 0) {
                const float u = std::clamp(-s / 2.2f, 0.f, 1.f);
                return std::min(h, waterline - depth * (1 - (1 - u) * (1 - u)));
            }
            const float up = 1 - std::pow(1 - std::min(s / 0.8f, 1.f), 2.f);
            const float bank = s < 0.8f ? waterline + (rim - waterline) * up : rim - (rim - h) * smoothstep(0.8f, 3.5f, s);
            return std::max(bank, std::min(h, waterline + 0.9f * s));
        }

        static float brookX(float z) {
            return 1.5f + (pondX + 0.6f - 1.5f) * smoothstep(tileZ0, brookZ1, z);
        }

        // the brook: a small channel under the ground, its bed never rising downstream, into the pond's head
        static float brookCut(float x, float z) {
            float bed = 1e9f;
            for (float t = tileZ0; t <= z; t += 0.5f) bed = std::min(bed, base(brookX(t), t) - 0.25f);
            bed = std::min(bed, base(brookX(z), z) - 0.25f);
            const float a = std::abs(x - brookX(z)), out = std::max(a - brookHalf, 0.f);
            return bed + 0.08f * std::min(a / brookHalf, 1.f) + 0.9f * out + 0.3f * out * out;
        }

        [[nodiscard]] float valley(float x, float z) const {
            float d, floor, w, k;
            stream(x, z, d, floor, w, k);
            const float a = std::abs(d), out = std::max(a - w, 0.f);
            floor += 0.04f * std::sin(2.1f * x + 1.3f * z) + 0.03f * std::sin(4.3f * z - 1.7f * x);
            return floor + 0.12f * std::pow(std::min(a / w, 1.f), 2.f) + out * k + 0.35f * k * out * out;
        }

        void planRocks() {
            std::mt19937 rng(5531);
            std::uniform_real_distribution<float> u01(0, 1);
            // the rapid's boulders: two thirds in the channel, the rest on its banks
            for (int tries = 0; rocks_.size() < 13 && tries < 2000; ++tries) {
                const float z = -4.6f + 12.f * u01(rng);
                float d, floor, w, k;
                stream(0, z, d, floor, w, k);
                const bool inWater = u01(rng) < 0.65f;
                const float off = inWater ? (u01(rng) * 1.5f - 0.75f) * w : (u01(rng) < 0.5f ? -1.f : 1.f) * (w + 0.5f * u01(rng));
                const float x = centreX(z) - off;
                const float r = (0.2f + 0.22f * u01(rng)) * (inWater ? 0.85f : 1.f);
                bool apart = true;
                for (const auto& o : rocks_) apart &= std::hypot(x - o.x, z - o.z) > (r + o.radius) * 1.15f + 0.3f;
                if (!apart) continue;
                rocks_.push_back({x, height(x, z) - 0.3f * r, z, r, 0.75f, 6.28f * u01(rng), inWater});
            }
            // the ledge: a row of rocks across the gorge, their tops at the lip
            const float lz = 8.95f;
            for (float off : {-1.45f, -0.72f, 0.0f, 0.74f, 1.48f}) {
                const float r = 0.55f + 0.12f * u01(rng);
                rocks_.push_back({centreX(lz) + off, -1.62f, lz + 0.08f * (u01(rng) - 0.5f), r, 1.25f, 6.28f * u01(rng), true});
            }
        }
    };

}// namespace sluice

#endif//THREEPP_EXAMPLE_SLUICE_COURSE_HPP
