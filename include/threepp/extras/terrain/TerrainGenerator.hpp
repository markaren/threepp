// Procedural terrain heightfield generator (CPU).
//
// Pipeline:
//   1. buildField()  — evaluate a multifractal noise stack (fBm / ridged /
//                      hybrid + domain warp + radial falloff) into a dense
//                      [0,1] heightfield grid, one cell per mesh vertex.
//   2. erode()       — optional droplet-hydraulic + thermal/talus erosion that
//                      carves drainage networks, V-valleys and scree slopes
//                      (the single biggest realism lever). Deterministic for a
//                      fixed seed.
//   3. makeGeometry()/displaceTo() — bake the field into a horizontal
//                      PlaneGeometry (Y = field*amplitude − rim sink) and
//                      recompute normals.
//
// Header-only, dependency-free beyond threepp core, so it can be reused from
// examples and later as the CPU reference / fallback for a GPU compute port.
//
// Coordinate convention: the geometry lies in the XZ plane (rotated flat, +Y
// up), centred on the origin, spanning [-worldSize/2, +worldSize/2] per axis.
// Heights are eroded in normalised [0,1] space with unit cell spacing (so the
// erosion is scale-invariant) and de-normalised by `amplitude` at bake time.

#ifndef THREEPP_EXTRAS_TERRAIN_TERRAINGENERATOR_HPP
#define THREEPP_EXTRAS_TERRAIN_TERRAINGENERATOR_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <execution>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace threepp::terrain {

    enum class NoiseType { fBm = 0,
                           Ridged = 1,
                           Hybrid = 2 };

    enum class Falloff { None = 0,
                         Radial = 1 };// radial fade to base: grounds the massif on a plain
                                      // (high falloffStart = gentle round base; low = tight cone)

    enum class ErosionType { None = 0,
                             Hydraulic = 1,// droplet simulation — drainage networks, V-valleys
                             Thermal = 2,  // talus / angle-of-repose slumping
                             Both = 3 };

    // Configurator-facing knobs. Mapped 1:1 onto ImGui controls in the example.
    struct TerrainParams {
        unsigned int seed = 1337;

        float worldSize = 1200.f;// metres — full mesh extent on X and Z
        int resolution = 256;    // grid segments per side (verts = resolution+1 squared)

        NoiseType noiseType = NoiseType::Ridged;
        float featureScale = 420.f;// metres per base-noise period (macro wavelength)
        int octaves = 7;
        float lacunarity = 2.0f;// frequency multiplier per octave
        float gain = 0.5f;      // amplitude persistence per octave
        float amplitude = 350.f;// metres — peak elevation
        float warp = 0.4f;      // 0..1 domain-warp strength (breaks grid-aligned ridges)

        float ridgeSharpness = 0.7f;// 0..1 — crest contrast for Ridged/Hybrid
        float heightExponent = 1.3f;// >1 deepens valleys, <1 lifts plateaus
        int terraces = 0;           // 0 = off; >1 = strata steps (mesa look)

        Falloff falloff = Falloff::None;
        float falloffStart = 0.55f;// radial: full height inside this normalised radius

        // ── Erosion ──────────────────────────────────────────────────────────
        ErosionType erosion = ErosionType::None;
        // Hydraulic (droplet) parameters — Beyer/Lague model. Heights are in
        // normalised [0,1] units with unit cell spacing, so these are tuned
        // once and are independent of worldSize / amplitude.
        int droplets = 80000;       // total droplets (≈ 0.3 / cell at 512²)
        int dropletLifetime = 30;   // max steps before a droplet evaporates
        float inertia = 0.05f;      // 0 = follow gradient exactly, 1 = keep direction
        float sedimentCapacity = 4.0f;// carry-capacity factor
        float minSlope = 0.01f;     // floors capacity on flats so droplets still carry
        float erodeSpeed = 0.3f;    // fraction of (capacity − load) dissolved per step
        float depositSpeed = 0.3f;  // fraction of excess sediment dropped per step
        float evaporation = 0.01f;  // water lost per step
        float gravity = 4.0f;       // speed gain factor with descent
        int erosionRadius = 3;      // erosion brush radius in cells (spreads cuts → no 1px ravines)
        // Thermal (talus) parameters. Kept light by default so the talus passes
        // relax over-steep noise spikes without rounding ridgelines into clay.
        float talusAngle = 38.f;    // degrees — slopes steeper than this slump
        int thermalIterations = 30; // slump sweeps
        float thermalRate = 0.5f;   // fraction of the excess moved per sweep

        // ── Texturing (slope / altitude splat baked into an albedo map) ───────
        float snowLine = 0.5f;      // altitude fraction (0..1) where snow begins
        float snowNoiseAmp = 0.08f; // snowline wiggle (breaks the iso-contour)
        float snowSlopeMax = 0.72f; // slope (0 flat..1 vertical) above which snow sheds (cliffs go bare)
        float slopeGrassMax = 0.28f;// grass → scree slope
        float slopeRockMin = 0.55f; // scree → rock slope
        float bandEdge = 0.07f;     // transition softness
        std::array<float, 3> rockColor = {0.39f, 0.36f, 0.33f};// neutral grey-brown
        std::array<float, 3> grassColor = {0.29f, 0.33f, 0.19f};// muted olive ("low+flat" band; sand/ash for other presets)
        std::array<float, 3> screeColor = {0.49f, 0.46f, 0.42f};
        std::array<float, 3> snowColor = {0.90f, 0.91f, 0.93f};// faintly off-white (pure white reads plastic)
    };

    class TerrainGenerator {

    public:
        explicit TerrainGenerator(unsigned int seed = 1337) { reseed(seed); }

        void reseed(unsigned int seed) {
            std::array<int, 256> p{};
            std::iota(p.begin(), p.end(), 0);
            std::mt19937 rng(seed ? seed : 1u);
            for (int i = 255; i > 0; --i) {
                std::uniform_int_distribution<int> d(0, i);
                std::swap(p[i], p[d(rng)]);
            }
            for (int i = 0; i < 512; ++i) perm_[i] = p[i & 255];
            seed_ = seed;
        }

        [[nodiscard]] unsigned int seed() const { return seed_; }
        [[nodiscard]] int dim() const { return dim_; }

        // ── Step 1: evaluate the noise stack into the dense field grid ───────
        // One cell per mesh vertex (dim = resolution+1), in PlaneGeometry vertex
        // order, storing the faded [0,1] base height (no amplitude / rim sink).
        void buildField(const TerrainParams& tp) {
            dim_ = std::max(tp.resolution, 1) + 1;
            field_.assign(static_cast<size_t>(dim_) * dim_, 0.f);
            const float half = tp.worldSize * 0.5f;
            const float step = tp.worldSize / static_cast<float>(dim_ - 1);

            std::vector<int> rows(static_cast<size_t>(dim_));
            std::iota(rows.begin(), rows.end(), 0);
            std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int iz) {
                const float z = -half + static_cast<float>(iz) * step;
                for (int ix = 0; ix < dim_; ++ix) {
                    const float x = -half + static_cast<float>(ix) * step;
                    field_[static_cast<size_t>(iz) * dim_ + ix] = shape01(x, z, tp) * falloffMul(x, z, tp);
                }
            });
        }

        // ── Step 2: erode the current field in place ─────────────────────────
        void erode(const TerrainParams& tp) {
            if (dim_ < 4) return;
            if (tp.erosion == ErosionType::Hydraulic || tp.erosion == ErosionType::Both)
                erodeHydraulic(tp);
            if (tp.erosion == ErosionType::Thermal || tp.erosion == ErosionType::Both)
                erodeThermal(tp);
        }

        // ── Step 3: bake the field into geometry ─────────────────────────────
        [[nodiscard]] std::shared_ptr<BufferGeometry> makeGeometry(const TerrainParams& tp) const {
            auto geo = PlaneGeometry::create(tp.worldSize, tp.worldSize,
                                             static_cast<unsigned int>(std::max(tp.resolution, 1)),
                                             static_cast<unsigned int>(std::max(tp.resolution, 1)));
            geo->rotateX(-kHalfPi);// lie flat, +Y up
            displaceTo(*geo, tp);
            return geo;
        }

        // Displace an existing (matching-resolution) geometry from the current
        // field. Bumps position/normal versions so the renderer rebuilds the
        // BLAS (the plain dynamic-geometry path — no special mesh type).
        void displaceTo(BufferGeometry& geo, const TerrainParams& tp) const {
            auto* pos = geo.getAttribute<float>("position");
            if (!pos) return;
            auto& a = pos->array();
            const int vcount = pos->count();
            const bool haveField = (static_cast<size_t>(vcount) == field_.size());

            std::vector<int> idx(static_cast<size_t>(vcount));
            std::iota(idx.begin(), idx.end(), 0);
            std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
                const float x = a[i * 3 + 0];
                const float z = a[i * 3 + 2];
                // field cell index == vertex index (built in the same order).
                const float base01 = haveField ? field_[static_cast<size_t>(i)]
                                                : shape01(x, z, tp) * falloffMul(x, z, tp);
                a[i * 3 + 1] = base01 * tp.amplitude - baseSink(x, z, tp);
            });
            pos->needsUpdate();

            geo.computeVertexNormals();
            if (auto* nrm = geo.getAttribute<float>("normal")) nrm->needsUpdate();
            geo.computeBoundingBox();
            geo.computeBoundingSphere();
        }

        // Convenience: noise (+ optional erosion) → fresh geometry.
        [[nodiscard]] std::shared_ptr<BufferGeometry> createGeometry(const TerrainParams& tp, bool withErosion) {
            buildField(tp);
            if (withErosion) erode(tp);
            return makeGeometry(tp);
        }

        // Single-sample procedural elevation in metres (no erosion). Useful for
        // physics/placement queries; the mesh path uses the eroded field.
        [[nodiscard]] float heightAt(float wx, float wz, const TerrainParams& tp) const {
            return shape01(wx, wz, tp) * falloffMul(wx, wz, tp) * tp.amplitude - baseSink(wx, wz, tp);
        }

        // Bake a slope/altitude/snow splat into an sRGB RGBA8 albedo image
        // (dim×dim, row-major, one texel per mesh vertex — the PlaneGeometry UVs
        // map it 1:1). Reads the CURRENT (eroded) field, so call after buildField
        // /erode. Blends four bands — grass (low+flat), scree (mid slope), rock
        // (steep), snow (high, slope-shed, noise-perturbed snowline). Returns the
        // byte buffer; the caller wraps it in a DataTexture (no texture
        // dependency here keeps the generator header lean).
        [[nodiscard]] std::vector<unsigned char> bakeSplatColors(const TerrainParams& tp) const {
            const int dim = dim_;
            std::vector<unsigned char> out(static_cast<size_t>(std::max(dim, 1)) * std::max(dim, 1) * 4, 255u);
            if (dim < 2 || static_cast<size_t>(dim) * dim != field_.size()) return out;

            const float cellWorld = tp.worldSize / static_cast<float>(dim - 1);
            const float e = std::max(tp.bandEdge, 1e-3f);
            const auto at = [dim](int x, int y) { return static_cast<size_t>(y) * dim + x; };
            const auto band = [](float x, float lo, float hi, float ee) {
                return smoothstep(lo - ee, lo + ee, x) * (1.f - smoothstep(hi - ee, hi + ee, x));
            };

            std::vector<int> rows(static_cast<size_t>(dim));
            std::iota(rows.begin(), rows.end(), 0);
            std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int z) {
                for (int x = 0; x < dim; ++x) {
                    const int xm = std::max(x - 1, 0), xp = std::min(x + 1, dim - 1);
                    const int zm = std::max(z - 1, 0), zp = std::min(z + 1, dim - 1);
                    const float hC = field_[at(x, z)];
                    const float dHdx = (field_[at(xp, z)] - field_[at(xm, z)]) * tp.amplitude / (2.f * cellWorld);
                    const float dHdz = (field_[at(x, zp)] - field_[at(x, zm)]) * tp.amplitude / (2.f * cellWorld);
                    const float ny = 1.f / std::sqrt(dHdx * dHdx + dHdz * dHdz + 1.f);
                    const float slope = 1.f - ny;             // 0 flat .. 1 vertical
                    const float alt = std::clamp(hC, 0.f, 1.f);
                    const float wig = noise2(static_cast<float>(x) * 0.06f, static_cast<float>(z) * 0.06f);

                    float wGrass = band(slope, 0.f, tp.slopeGrassMax, e) * band(alt, 0.f, tp.snowLine, e);
                    float wScree = band(slope, tp.slopeGrassMax, tp.slopeRockMin, e);
                    float wRock = smoothstep(tp.slopeRockMin - 0.05f, tp.slopeRockMin + 0.2f, slope);
                    float wSnow = smoothstep(tp.snowLine - 0.06f, tp.snowLine + 0.06f, alt + wig * tp.snowNoiseAmp) *
                                  (1.f - smoothstep(tp.snowSlopeMax - 0.1f, tp.snowSlopeMax + 0.1f, slope));

                    float total = wGrass + wScree + wRock + wSnow;
                    if (total < 1e-4f) {
                        wRock = 1.f;
                        total = 1.f;
                    }
                    // De-plastic: break flat band fills with multi-scale albedo
                    // variation and darken concave creases (cheap baked AO). A
                    // pure per-band flat colour reads as painted plastic; real
                    // ground has grain and occlusion in the folds.
                    const float n1 = fbm(static_cast<float>(x) * 0.16f, static_cast<float>(z) * 0.16f, 4, 2.f, 0.5f);
                    const float n2 = noise2(static_cast<float>(x) * 0.8f, static_cast<float>(z) * 0.8f);
                    const float varia = std::clamp(1.f + 0.15f * n1 + 0.08f * n2, 0.65f, 1.25f);
                    const float curv = field_[at(xm, z)] + field_[at(xp, z)] + field_[at(x, zm)] + field_[at(x, zp)] - 4.f * hC;
                    const float ao = 1.f - std::clamp(-curv * 45.f, 0.f, 0.35f);// valleys/creases darker

                    const float inv = (varia * ao) / total;
                    const float r = (tp.grassColor[0] * wGrass + tp.screeColor[0] * wScree + tp.rockColor[0] * wRock + tp.snowColor[0] * wSnow) * inv;
                    const float g = (tp.grassColor[1] * wGrass + tp.screeColor[1] * wScree + tp.rockColor[1] * wRock + tp.snowColor[1] * wSnow) * inv;
                    const float b = (tp.grassColor[2] * wGrass + tp.screeColor[2] * wScree + tp.rockColor[2] * wRock + tp.snowColor[2] * wSnow) * inv;

                    // Z-flip: the albedo map's rows run opposite to world Z vs the
                    // PlaneGeometry UV v-axis, so store this cell at the mirrored row.
                    const size_t o = at(x, dim - 1 - z) * 4;
                    out[o + 0] = static_cast<unsigned char>(std::clamp(r, 0.f, 1.f) * 255.f + 0.5f);
                    out[o + 1] = static_cast<unsigned char>(std::clamp(g, 0.f, 1.f) * 255.f + 0.5f);
                    out[o + 2] = static_cast<unsigned char>(std::clamp(b, 0.f, 1.f) * 255.f + 0.5f);
                    out[o + 3] = 255u;
                }
            });
            return out;
        }

    private:
        static constexpr float kHalfPi = 1.57079632679489661923f;
        static constexpr float kRimSink = 0.12f;// fraction of amplitude the faded rim tucks below 0

        // ── shaping ──────────────────────────────────────────────────────────
        // [0,1] noise shape (no falloff, no amplitude).
        [[nodiscard]] float shape01(float wx, float wz, const TerrainParams& tp) const {
            float nx = wx / tp.featureScale;
            float nz = wz / tp.featureScale;
            if (tp.warp > 0.f) {
                const float w = tp.warp * 1.5f;
                const float qx = fbm(nx, nz, 4, 2.f, 0.5f);
                const float qz = fbm(nx + 5.2f, nz + 1.3f, 4, 2.f, 0.5f);
                nx += w * qx;
                nz += w * qz;
            }
            float h;
            switch (tp.noiseType) {
                case NoiseType::Ridged: h = ridged(nx, nz, tp.octaves, tp.lacunarity, tp.gain, tp.ridgeSharpness); break;
                case NoiseType::Hybrid: h = hybrid(nx, nz, tp.octaves, tp.lacunarity, tp.gain, tp.ridgeSharpness); break;
                case NoiseType::fBm:
                default: h = fbm(nx, nz, tp.octaves, tp.lacunarity, tp.gain) * 0.5f + 0.5f; break;
            }
            h = std::clamp(h, 0.f, 1.f);
            if (tp.terraces > 1) {
                const float steps = static_cast<float>(tp.terraces);
                h = std::round(h * steps) / steps;
            }
            if (tp.heightExponent != 1.f) h = std::pow(h, tp.heightExponent);
            return h;
        }

        // Radial fade multiplier (1 = full height, 0 = base plane). 1 if None.
        [[nodiscard]] static float falloffMul(float wx, float wz, const TerrainParams& tp) {
            if (tp.falloff != Falloff::Radial) return 1.f;
            const float half = tp.worldSize * 0.5f;
            const float r = std::sqrt(wx * wx + wz * wz) / std::max(half, 1e-3f);
            return smoothstep(1.0f, tp.falloffStart, r);
        }

        // Rim sink: drops the faded border below the surrounding ground plane so
        // the finite square patch tucks under it instead of forming a lip.
        [[nodiscard]] static float baseSink(float wx, float wz, const TerrainParams& tp) {
            if (tp.falloff != Falloff::Radial) return 0.f;
            return kRimSink * tp.amplitude * (1.f - falloffMul(wx, wz, tp));
        }

        // ── hydraulic (droplet) erosion ──────────────────────────────────────
        void erodeHydraulic(const TerrainParams& tp) {
            const int dim = dim_;
            const int radius = std::clamp(tp.erosionRadius, 1, 8);

            // Precompute the erosion brush (relative offsets + normalised weights):
            // eroding over a disc instead of a point prevents 1-pixel ravines.
            std::vector<int> brushDX, brushDY;
            std::vector<float> brushW;
            float wsum = 0.f;
            for (int by = -radius; by <= radius; ++by)
                for (int bx = -radius; bx <= radius; ++bx) {
                    const float dist = std::sqrt(static_cast<float>(bx * bx + by * by));
                    if (dist <= static_cast<float>(radius)) {
                        const float w = 1.f - dist / static_cast<float>(radius);
                        brushDX.push_back(bx);
                        brushDY.push_back(by);
                        brushW.push_back(w);
                        wsum += w;
                    }
                }
            for (auto& w : brushW) w /= wsum;

            std::mt19937 rng(seed_ ^ 0x9e3779b9u);
            std::uniform_real_distribution<float> spawn(0.f, static_cast<float>(dim - 1));
            const auto at = [dim](int x, int y) { return static_cast<size_t>(y) * dim + x; };

            const float inertia = tp.inertia, capF = tp.sedimentCapacity, minSlope = tp.minSlope;
            const float erodeSpeed = tp.erodeSpeed, depositSpeed = tp.depositSpeed;
            const float evap = tp.evaporation, gravity = tp.gravity;
            const int maxLife = std::max(tp.dropletLifetime, 1);
            const int n = std::max(tp.droplets, 0);

            for (int d = 0; d < n; ++d) {
                float posX = spawn(rng), posY = spawn(rng);
                float dirX = 0.f, dirY = 0.f, speed = 1.f, water = 1.f, sediment = 0.f;

                for (int life = 0; life < maxLife; ++life) {
                    const int nodeX = static_cast<int>(posX);
                    const int nodeY = static_cast<int>(posY);
                    if (nodeX < 0 || nodeX >= dim - 1 || nodeY < 0 || nodeY >= dim - 1) break;
                    const float fx = posX - static_cast<float>(nodeX);
                    const float fy = posY - static_cast<float>(nodeY);

                    const float hNW = field_[at(nodeX, nodeY)];
                    const float hNE = field_[at(nodeX + 1, nodeY)];
                    const float hSW = field_[at(nodeX, nodeY + 1)];
                    const float hSE = field_[at(nodeX + 1, nodeY + 1)];

                    // Bilinear gradient + height at the droplet.
                    const float gradX = (hNE - hNW) * (1.f - fy) + (hSE - hSW) * fy;
                    const float gradY = (hSW - hNW) * (1.f - fx) + (hSE - hNE) * fx;
                    const float oldH = hNW * (1 - fx) * (1 - fy) + hNE * fx * (1 - fy) +
                                       hSW * (1 - fx) * fy + hSE * fx * fy;

                    // Steer: blend momentum with the downhill gradient.
                    dirX = dirX * inertia - gradX * (1.f - inertia);
                    dirY = dirY * inertia - gradY * (1.f - inertia);
                    const float len = std::sqrt(dirX * dirX + dirY * dirY);
                    if (len < 1e-6f) break;// settled in a pit
                    dirX /= len;
                    dirY /= len;
                    posX += dirX;
                    posY += dirY;

                    const int nX = static_cast<int>(posX);
                    const int nY = static_cast<int>(posY);
                    if (nX < 0 || nX >= dim - 1 || nY < 0 || nY >= dim - 1) break;
                    const float nfx = posX - static_cast<float>(nX);
                    const float nfy = posY - static_cast<float>(nY);
                    const float newH = field_[at(nX, nY)] * (1 - nfx) * (1 - nfy) +
                                       field_[at(nX + 1, nY)] * nfx * (1 - nfy) +
                                       field_[at(nX, nY + 1)] * (1 - nfx) * nfy +
                                       field_[at(nX + 1, nY + 1)] * nfx * nfy;
                    const float deltaH = newH - oldH;

                    const float capacity = std::max(-deltaH, minSlope) * speed * water * capF;

                    if (sediment > capacity || deltaH > 0.f) {
                        // Deposit (bilinear, at the OLD position — fills pits and
                        // builds sediment fans). Uphill: drop just enough to fill.
                        const float deposit = (deltaH > 0.f) ? std::min(deltaH, sediment)
                                                             : (sediment - capacity) * depositSpeed;
                        sediment -= deposit;
                        field_[at(nodeX, nodeY)] += deposit * (1 - fx) * (1 - fy);
                        field_[at(nodeX + 1, nodeY)] += deposit * fx * (1 - fy);
                        field_[at(nodeX, nodeY + 1)] += deposit * (1 - fx) * fy;
                        field_[at(nodeX + 1, nodeY + 1)] += deposit * fx * fy;
                    } else {
                        // Erode (over the brush, capped at the local relief so we
                        // never dig below the cell we are flowing toward).
                        const float erode = std::min((capacity - sediment) * erodeSpeed, -deltaH);
                        for (size_t b = 0; b < brushW.size(); ++b) {
                            const int cx = nodeX + brushDX[b];
                            const int cy = nodeY + brushDY[b];
                            if (cx < 0 || cx >= dim || cy < 0 || cy >= dim) continue;
                            const float we = erode * brushW[b];
                            field_[at(cx, cy)] -= we;
                            sediment += we;
                        }
                    }

                    speed = std::sqrt(std::max(0.f, speed * speed + deltaH * gravity));
                    water *= (1.f - evap);
                    if (water < 1e-4f) break;
                }
            }
        }

        // ── thermal (talus) erosion ──────────────────────────────────────────
        // Material on slopes steeper than the angle of repose slumps to lower
        // neighbours until the slope relaxes. Carves scree benches and softens
        // the over-steep ridges raw noise produces.
        void erodeThermal(const TerrainParams& tp) {
            const int dim = dim_;
            const float cellWorld = tp.worldSize / static_cast<float>(dim - 1);
            // Height step (in [0,1] field units) that corresponds to the repose
            // angle over one cell: tan(angle)·cellWorld / amplitude.
            const float talus = std::tan(tp.talusAngle * (kHalfPi / 90.f)) *
                                cellWorld / std::max(tp.amplitude, 1e-3f);
            const float rate = std::clamp(tp.thermalRate, 0.f, 1.f);
            const auto at = [dim](int x, int y) { return static_cast<size_t>(y) * dim + x; };

            std::vector<float> delta(field_.size(), 0.f);

            static constexpr int dx8[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
            static constexpr int dy8[8] = {0, 0, -1, 1, -1, 1, -1, 1};

            for (int it = 0; it < std::max(tp.thermalIterations, 0); ++it) {
                std::fill(delta.begin(), delta.end(), 0.f);
                // Accumulate slumps (read field, write delta). Serial: each cell
                // writes delta into its neighbours (across row boundaries), so a
                // row-parallel split would race on the shared border rows.
                for (int y = 0; y < dim; ++y) {
                    for (int x = 0; x < dim; ++x) {
                        const float h = field_[at(x, y)];
                        float dTotal = 0.f, dMax = 0.f;
                        float diff[8];
                        for (int k = 0; k < 8; ++k) {
                            const int nx = x + dx8[k], ny = y + dy8[k];
                            float dh = 0.f;
                            if (nx >= 0 && nx < dim && ny >= 0 && ny < dim) {
                                dh = h - field_[at(nx, ny)];
                                if (dh > talus) {
                                    dTotal += dh;
                                    dMax = std::max(dMax, dh);
                                } else
                                    dh = 0.f;
                            }
                            diff[k] = std::max(dh, 0.f);
                        }
                        if (dTotal <= 0.f) continue;
                        // Move a fraction of the worst over-steepness, split among
                        // the lower neighbours in proportion to their drop.
                        const float move = rate * 0.5f * (dMax - talus);
                        for (int k = 0; k < 8; ++k) {
                            if (diff[k] <= 0.f) continue;
                            const int nx = x + dx8[k], ny = y + dy8[k];
                            delta[at(nx, ny)] += move * (diff[k] / dTotal);
                        }
                        delta[at(x, y)] -= move;
                    }
                }
                for (size_t i = 0; i < field_.size(); ++i) field_[i] += delta[i];
            }
        }

        // ── noise primitives ─────────────────────────────────────────────────
        static float fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
        static float lerp(float a, float b, float t) { return a + t * (b - a); }
        static float smoothstep(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }
        static float grad2(int h, float x, float y) {
            switch (h & 7) {
                case 0: return x + y;
                case 1: return -x + y;
                case 2: return x - y;
                case 3: return -x - y;
                case 4: return x;
                case 5: return -x;
                case 6: return y;
                default: return -y;
            }
        }

        // 2D Perlin gradient noise, output ~[-1, 1].
        float noise2(float x, float y) const {
            const int X = static_cast<int>(std::floor(x)) & 255;
            const int Y = static_cast<int>(std::floor(y)) & 255;
            x -= std::floor(x);
            y -= std::floor(y);
            const float u = fade(x), v = fade(y);
            const int aa = perm_[perm_[X] + Y];
            const int ab = perm_[perm_[X] + Y + 1];
            const int ba = perm_[perm_[X + 1] + Y];
            const int bb = perm_[perm_[X + 1] + Y + 1];
            const float x1 = lerp(grad2(aa, x, y), grad2(ba, x - 1.f, y), u);
            const float x2 = lerp(grad2(ab, x, y - 1.f), grad2(bb, x - 1.f, y - 1.f), u);
            return lerp(x1, x2, v);
        }

        float fbm(float x, float y, int oct, float lac, float gain) const {
            float f = 1.f, a = 1.f, sum = 0.f, norm = 0.f;
            for (int i = 0; i < oct; ++i) {
                sum += a * noise2(x * f, y * f);
                norm += a;
                f *= lac;
                a *= gain;
            }
            return norm > 0.f ? sum / norm : 0.f;
        }

        float ridged(float x, float y, int oct, float lac, float gain, float sharp) const {
            const float offset = 1.0f;
            const float exp = 1.f + sharp * 2.f;
            float freq = 1.f, amp = 1.f, sum = 0.f, norm = 0.f, prev = 1.f;
            for (int i = 0; i < oct; ++i) {
                const float n = noise2(x * freq, y * freq);
                float signal = offset - std::abs(n);
                signal = std::pow(std::clamp(signal, 0.f, 1.f), exp);
                signal *= std::clamp(prev * 2.f, 0.f, 1.f);
                sum += amp * signal;
                norm += amp;
                prev = signal;
                freq *= lac;
                amp *= gain;
            }
            return norm > 0.f ? std::clamp(sum / norm, 0.f, 1.f) : 0.f;
        }

        float hybrid(float x, float y, int oct, float lac, float gain, float sharp) const {
            const float fb = fbm(x, y, oct, lac, gain) * 0.5f + 0.5f;
            const float rg = ridged(x, y, oct, lac, gain, sharp);
            const float t = std::clamp(fb, 0.f, 1.f);
            return std::clamp(lerp(fb, rg, 0.5f * t + 0.25f * sharp), 0.f, 1.f);
        }

        std::array<int, 512> perm_{};
        unsigned int seed_ = 0;
        std::vector<float> field_;// [0,1] faded base height per vertex (eroded in place)
        int dim_ = 0;
    };

    // ── Config (de)serialisation (implemented in TerrainGenerator.cpp) ────────
    // Round-trip the full TerrainParams to/from JSON. Generation is deterministic,
    // so a loaded config reproduces the EXACT terrain (erosion + texturing
    // included). Unknown/missing keys keep their current value, so a config stays
    // forward/backward compatible as params are added.
    [[nodiscard]] std::string toJson(const TerrainParams& params);
    [[nodiscard]] bool fromJson(const std::string& json, TerrainParams& out);

    // File convenience wrappers. saveConfig creates missing parent directories.
    // Both return false on I/O (or, for load, JSON parse) failure.
    [[nodiscard]] bool saveConfig(const std::string& filePath, const TerrainParams& params);
    [[nodiscard]] bool loadConfig(const std::string& filePath, TerrainParams& out);

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_TERRAINGENERATOR_HPP
