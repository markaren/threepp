// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/utils/BufferGeometryUtils.js

#ifndef THREEPP_BUFFERGEOMETRYUTILS_HPP
#define THREEPP_BUFFERGEOMETRYUTILS_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <vector>

namespace threepp {

    class Object3D;

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<BufferGeometry*>& geometries, bool useGroups = false);

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<std::shared_ptr<BufferGeometry>>& geometries, bool useGroups = false);

    std::shared_ptr<BufferGeometry> mergeVertices(const BufferGeometry& geometry, float tolerance = 1e-4f);

    std::shared_ptr<BufferGeometry> simplifyGeometry(const BufferGeometry& geometry, float ratio, float error = 1e-2f);

    // Which attributes compressAttributes() should narrow. Each is only rewritten
    // when it is currently float and its values fit the target range; anything
    // that would lose data is left alone.
    struct AttributeCompression {

        bool normal{true}; // float32x3 -> int16 snorm   (12 -> 6 bytes/vertex)
        bool tangent{true};// float32x4 -> int16 snorm   (16 -> 8 bytes/vertex)
        bool uv{true};     // float32x2 -> uint16 unorm  ( 8 -> 4 bytes/vertex)
        bool color{true};  // float32xN -> uint8 unorm   (12 -> 3 bytes/vertex)
    };

    // Narrow a geometry's vertex attributes in place, returning the number of
    // host bytes reclaimed.
    //
    // Positions and indices are never touched: positions need float's dynamic
    // range unless the mesh is quantised against a known AABB, and index
    // narrowing is a separate change.
    //
    // Precision: int16 snorm resolves a normal to ~0.003 degrees and uint16 unorm
    // a UV to 1/65535 of the [0,1] range, both well below what an 8k texture or a
    // shading normal can distinguish. uint8 colour is the same precision the
    // source PNG had. UVs are skipped when any coordinate falls outside [0,1],
    // since tiled/atlas UVs would clamp; tangents are skipped when the w
    // handedness component is not +/-1.
    //
    // Backend support: the GL renderer consumes these natively — GLAttributes
    // uploads the narrow type and GLBindingStates forwards the normalized flag,
    // so the saving carries through to VRAM with no visible difference. The
    // Vulkan backend widens narrow attributes once at upload (FloatAttributeView
    // in buildBlasFor), so the saving there is host-side only, with identical
    // output. Mesh::raycast reads uv/uv2 through views as well.
    //
    // Deforming geometry is the exception: the Vulkan skinned / softbody /
    // morph / displaced paths rewrite their float device buffers every frame
    // and require float attributes — do NOT compress skinned meshes (the
    // renderer warns and skips them). mergeVertices() on a compressed geometry
    // widens the narrow attributes back to float in its output.
    //
    // Use FloatAttributeView to read a possibly-narrowed attribute as float.
    size_t compressAttributes(BufferGeometry& geometry, const AttributeCompression& what = {});

    // Compress every *safe* geometry under `root`: plain static meshes only.
    // Skinned meshes, DisplacedMesh, morph-target geometry and softbody (tet)
    // geometry are skipped automatically — those deform per frame and both
    // renderers require them to stay float. Shared geometry is naturally
    // handled (compressAttributes is idempotent, the second visit is a no-op).
    // Returns total host bytes reclaimed. Call once, after the scene is built;
    // geometry added later is not affected.
    size_t compressSceneAttributes(Object3D& root, const AttributeCompression& what = {});

}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRYUTILS_HPP
