#include "threepp/objects/DisplacedMesh.hpp"

#include <cmath>

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    std::string DisplacedMesh::type() const {
        return "DisplacedMesh";
    }

    float DisplacedMesh::sampleHeight(float worldX, float worldZ) const {
        if (heightField.empty() || heightFieldDim == 0 || heightFieldTileSize <= 0.f) {
            return 0.f;
        }
        const uint32_t dim = heightFieldDim;
        const float invTile = 1.f / heightFieldTileSize;

        // UV in cells. The water_displace.comp shader samples the height
        // texture at the SAME mesh world position used to derive the rest
        // vertex (after the rotateX(-π/2) on the plane geometry, the V axis
        // is mirrored). We mirror the Z here so this CPU sample lines up
        // with the GPU-displaced surface.
        const float u = worldX * invTile;
        const float v = -worldZ * invTile;

        // Wrap to [0, dim). std::fmod handles negatives by returning a
        // matching-sign remainder, so the +dim+wrap-fmod normalises back into
        // the positive range.
        auto wrap = [dim](float x) {
            const float fdim = float(dim);
            float w = std::fmod(x, fdim);
            if (w < 0.f) w += fdim;
            return w;
        };
        const float fx = wrap(u * float(dim));
        const float fz = wrap(v * float(dim));

        const uint32_t x0 = uint32_t(std::floor(fx)) % dim;
        const uint32_t z0 = uint32_t(std::floor(fz)) % dim;
        const uint32_t x1 = (x0 + 1) % dim;
        const uint32_t z1 = (z0 + 1) % dim;
        const float tx = fx - std::floor(fx);
        const float tz = fz - std::floor(fz);

        // Stride-2 indexing: heightField is RG32F packed (R = height,
        // G = velocity/dht — ignored here).
        auto h = [&](uint32_t x, uint32_t z) {
            return heightField[(z * dim + x) * 2u + 0u];
        };
        const float h00 = h(x0, z0);
        const float h10 = h(x1, z0);
        const float h01 = h(x0, z1);
        const float h11 = h(x1, z1);
        const float a = h00 * (1.f - tx) + h10 * tx;
        const float b = h01 * (1.f - tx) + h11 * tx;
        const float bil = a * (1.f - tz) + b * tz;

        // Match the displace shader's amplitude normalization exactly:
        //   GPU: y = texture(height, uv).r * (1/tileSize) * waveScale
        // so we mirror the same factors here. Without the waveScale factor,
        // setting params.waveScale = 0 flattens the rendered water (GPU)
        // but sampleHeight keeps returning non-zero height — anything
        // following the surface (boat hydrodynamics) would bob on a flat
        // surface like a trampoline.
        return bil * invTile * params.waveScale;
    }

}// namespace threepp
