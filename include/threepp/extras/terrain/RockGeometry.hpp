// Low-poly faceted boulder: a sphere displaced by a few smooth lumps.
// Pair with a flat-shaded material for crisp facets.
//
// One geometry per seed; instance it for a field of rocks (see the boulder
// scatter blocks in forest_demo and vulkan_fjord). Deterministic: the same
// seed always yields the same boulder.

#ifndef THREEPP_EXTRAS_TERRAIN_ROCKGEOMETRY_HPP
#define THREEPP_EXTRAS_TERRAIN_ROCKGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace threepp::terrain {

    inline std::shared_ptr<BufferGeometry> makeRockGeometry(unsigned int seed,
                                                            int latSegs = 5, int lonSegs = 7) {
        constexpr float PI = 3.14159265358979f;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-PI, PI);
        const float p1 = u(rng), p2 = u(rng), p3 = u(rng);

        std::vector<float> pos, nrm, uv;
        std::vector<unsigned int> idx;
        for (int lat = 0; lat <= latSegs; ++lat) {
            const float theta = static_cast<float>(lat) / static_cast<float>(latSegs) * PI;
            const float sinT = std::sin(theta), cosT = std::cos(theta);
            for (int lon = 0; lon <= lonSegs; ++lon) {
                const float phi = static_cast<float>(lon) / static_cast<float>(lonSegs) * 2.f * PI;
                const float nx = sinT * std::cos(phi);
                const float ny = cosT;
                const float nz = sinT * std::sin(phi);
                // Every phi-dependent term is gated by sinT so it vanishes at the
                // poles (theta = 0, PI). Ungated cos(3phi)/cos(5phi) give each pole
                // vertex a DIFFERENT radius, smearing the single pole point into a
                // fan of long, thin sliver triangles stacked along Y — whose sub-
                // pixel coverage toggles with the per-frame TAA jitter and flickers
                // uniformly every frame (the 0-2 faces facing the camera). Gated,
                // the pole vertices coincide → zero-area triangles that rasterize to
                // nothing, leaving a clean pole fan. Only the phi-independent
                // sin(3*theta) term survives at the poles (a uniform radius there).
                float disp = 1.f + sinT * (0.30f * std::sin(2.f * phi + p1) +
                                           0.24f * std::cos(3.f * phi + p2) +
                                           0.14f * std::cos(5.f * phi + 4.f * theta + p1)) +
                             0.22f * std::sin(3.f * theta + p3);
                disp = std::clamp(disp, 0.6f, 1.5f);
                pos.insert(pos.end(), {nx * disp, ny * disp, nz * disp});
                nrm.insert(nrm.end(), {nx, ny, nz});
                uv.insert(uv.end(), {static_cast<float>(lon) / static_cast<float>(lonSegs),
                                     static_cast<float>(lat) / static_cast<float>(latSegs)});
            }
        }
        const int rowVerts = lonSegs + 1;
        for (int lat = 0; lat < latSegs; ++lat) {
            for (int lon = 0; lon < lonSegs; ++lon) {
                const auto a = static_cast<unsigned int>(lat * rowVerts + lon);
                const auto b = static_cast<unsigned int>(a + rowVerts);
                // CCW from outside (outward-facing normals for flat shading).
                idx.insert(idx.end(), {a, a + 1, b, a + 1, b + 1, b});
            }
        }
        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        return geo;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_ROCKGEOMETRY_HPP
