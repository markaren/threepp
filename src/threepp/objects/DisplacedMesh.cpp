#include "threepp/objects/DisplacedMesh.hpp"

#include <cmath>

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    std::string DisplacedMesh::type() const {
        return "DisplacedMesh";
    }

    namespace {

        float sampleCascade(const DisplacedMesh::CascadeField& cf,
                            float worldX, float worldZ) {
            if (cf.data.empty() || cf.dim == 0 || cf.tileSize <= 0.f)
                return 0.f;

            const uint32_t dim = cf.dim;
            const float invTile = 1.f / cf.tileSize;
            const float u = worldX * invTile;
            const float v = -worldZ * invTile;

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

            auto h = [&](uint32_t x, uint32_t z) {
                return cf.data[(z * dim + x) * 2u + 0u];
            };
            const float h00 = h(x0, z0);
            const float h10 = h(x1, z0);
            const float h01 = h(x0, z1);
            const float h11 = h(x1, z1);
            const float a = h00 * (1.f - tx) + h10 * tx;
            const float b = h01 * (1.f - tx) + h11 * tx;

            return (a * (1.f - tz) + b * tz) * invTile;
        }

    }// namespace

    float DisplacedMesh::sampleHeight(float worldX, float worldZ) const {
        float total = 0.f;
        for (const auto& cf : heightFields)
            total += sampleCascade(cf, worldX, worldZ);
        return total * params.waveScale;
    }

}// namespace threepp
