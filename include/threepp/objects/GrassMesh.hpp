// A mesh whose vertex positions are GPU-displaced every frame by a wind
// compute shader. Designed for the Vulkan path tracer: the renderer
// recognises the type, runs the grass-wind compute pass (grass_wind.comp),
// and refits the BLAS in place every frame — so an entire grass field is a
// single TLAS instance and the per-frame cost is one cheap BLAS/TLAS refit
// rather than the O(all-instances) TLAS rebuild that an animated InstancedMesh
// would trigger.
//
// The geometry is a SINGLE merged BufferGeometry containing every blade baked
// into its vertices (NOT an InstancedMesh). Besides the usual position
// (rest pose), normal, uv and (optional) color attributes, it must carry a
// custom per-vertex float attribute "heightFrac" — 0 at a blade's base, 1 at
// its tip — which the wind shader uses to weight the sway (base planted, tip
// sways). The renderer keeps an immutable copy of the rest positions, so the
// shader can read the rest pose while writing the displaced pose into the same
// BLAS vertex buffer.
//
// On the GL / WebGPU raster backends this type renders as a plain static Mesh
// (no wind); those backends animate grass via their own cheaper paths.

#ifndef THREEPP_GRASSMESH_HPP
#define THREEPP_GRASSMESH_HPP

#include "threepp/math/Vector2.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cstdint>

namespace threepp {

    class GrassMesh : public Mesh {

    public:
        struct Params {
            float windStrength = 0.18f;     // sway amplitude
            Vector2 windDir{0.8f, 0.6f};    // horizontal wind direction (world XZ)
            float time = 0.f;               // animation clock (seconds); set per frame
        };
        Params params;

        GrassMesh(const std::shared_ptr<BufferGeometry>& geometry,
                  const std::shared_ptr<Material>& material);

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<GrassMesh> create(
                const std::shared_ptr<BufferGeometry>& geometry,
                const std::shared_ptr<Material>& material) {
            return std::make_shared<GrassMesh>(geometry, material);
        }

        // Bumped each render() so the renderer's dirty-detect always treats this
        // mesh as moved (it is — the wind advances every frame). Internal.
        uint64_t frameTick = 0;
    };

}// namespace threepp

#endif//THREEPP_GRASSMESH_HPP
