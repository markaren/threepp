// Geometry buffer management (interleaved vertex buffers, index buffers,
// wireframe indices) with version-based update tracking.

#ifndef THREEPP_DAWNGEOMETRIES_HPP
#define THREEPP_DAWNGEOMETRIES_HPP

#include "DawnState.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <webgpu/webgpu.h>

namespace threepp {
    class BufferGeometry;
}

namespace threepp::dawn {

    // pos(12) + normal(12) + uv(8) + color(12) = 44 bytes per vertex
    constexpr uint32_t VERTEX_STRIDE = 44;

    struct GeometryBuffers {
        WGPUBuffer vertexBuffer = nullptr;
        WGPUBuffer indexBuffer = nullptr;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        unsigned int positionVersion = 0;
        unsigned int normalVersion = 0;
        unsigned int uvVersion = 0;
        unsigned int colorVersion = 0;
        unsigned int lineDistanceVersion = 0;
        unsigned int indexVersion = 0;
    };

    struct WireframeBuffers {
        WGPUBuffer indexBuffer = nullptr;
        uint32_t indexCount = 0;
    };

    class DawnGeometries {
    public:
        explicit DawnGeometries(DawnState& state);

        GeometryBuffers& getOrCreateGeometryBuffers(BufferGeometry* geometry);

        WireframeBuffers& getOrCreateWireframeBuffers(BufferGeometry* geometry);

        void dispose();

        [[nodiscard]] size_t count() const { return geometryCache_.size(); }

    private:
        DawnState& state_;
        std::unordered_map<unsigned int, GeometryBuffers> geometryCache_;
        std::unordered_map<unsigned int, WireframeBuffers> wireframeCache_;

        static std::vector<float> buildInterleavedVertexData(BufferGeometry* geometry, uint32_t count);
        static void storeAttributeVersions(BufferGeometry* geometry, GeometryBuffers& gb);
        static bool geometryNeedsUpdate(BufferGeometry* geometry, const GeometryBuffers& gb);
        static bool indexNeedsUpdate(BufferGeometry* geometry, const GeometryBuffers& gb);
    };

}// namespace threepp::dawn

#endif//THREEPP_DAWNGEOMETRIES_HPP
