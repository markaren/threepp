// The clustered-light grid as the shaders see it: the ClusterLight record
// (GpuClusterLight in VulkanGpuLayouts.hpp, scalar layout, 64 bytes), the
// grid's shape and its exponential Z ladder, and the cell lookup for a
// screen uv at a view distance. One declaration for cluster_build (which
// fills the grid), the deferred shade, froxel_inject and particle_light
// (which read it); until 2026-10-10 each carried its own copy under a
// "KEEP IN SYNC" plea. The two buffers (bindings 48, 49) stay with each
// shader: the builder writes the grid, the readers declare it readonly.
//
// The Z slices are a FIXED exponential ladder over view-space distance
// [kClusterZMin, kClusterZMax] (camera-independent, no near/far plumbing):
// slice 0 extends to the camera, the last slice absorbs everything beyond
// kClusterZMax. deferred_shade_20_restir_di.glsl has the pixel-indexed
// form of clusterCellBase; this one takes a uv.

#ifndef CLUSTER_LIGHTS_GLSL
#define CLUSTER_LIGHTS_GLSL

struct ClusterLight {
    vec3  position;   float range;         // range 0 = infinite (three.js)
    vec3  color;      float decay;         // color premultiplied by intensity
    vec3  direction;  float cosAngleOuter; // spot cone; points carry the -1.1/-1.05 sentinel (cone test → 1)
    float cosAngleInner;
    float radius;                          // physical source radius (soft shadows)
    float cullRadius;                      // conservative influence radius (range, or the atten<eps solve)
    float type;                            // 0 = point, 1 = spot
};

const uint  kClusterX = 16u, kClusterY = 8u, kClusterZ = 24u;
const uint  kClusterMaxPerCell = 24u;
const float kClusterZMin = 0.1, kClusterZMax = 1024.0;

uint clusterCellBase(vec2 uv, float viewDist) {
    const uint cx = min(uint(uv.x * float(kClusterX)), kClusterX - 1u);
    const uint cy = min(uint(uv.y * float(kClusterY)), kClusterY - 1u);
    const float t = log(max(viewDist, kClusterZMin) / kClusterZMin)
                  / log(kClusterZMax / kClusterZMin);
    const uint cz = min(uint(max(t, 0.0) * float(kClusterZ)), kClusterZ - 1u);
    return (cx + cy * kClusterX + cz * kClusterX * kClusterY) * (kClusterMaxPerCell + 1u);
}

#endif// CLUSTER_LIGHTS_GLSL
