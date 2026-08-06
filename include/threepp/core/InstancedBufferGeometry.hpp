// https://github.com/mrdoob/three.js/blob/r129/src/core/InstancedBufferGeometry.js

#ifndef THREEPP_INSTANCEDBUFFERGEOMETRY_HPP
#define THREEPP_INSTANCEDBUFFERGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace threepp {

    // A geometry whose vertices are drawn `instanceCount` times, with the
    // per-instance data supplied as InstancedBufferAttributes in its own
    // attribute map.
    //
    // This is the OTHER way to be instanced, and the difference from
    // InstancedMesh is where the per-instance data lives and who decides what it
    // is. InstancedMesh is an object with two fixed attributes baked in — a
    // 16-float instanceMatrix and an optional instanceColor — and every mesh
    // that uses it pays for the matrix whether or not it wants a per-instance
    // transform. A splat cloud does not: its placement is a mean and a
    // covariance fetched from a texture, so a million identity matrices were a
    // million identity matrices (64 bytes each, host and VRAM both). Here the
    // geometry carries exactly the attributes the shader declares and nothing
    // else.
    //
    // Neither replaces the other. InstancedMesh stays the right answer whenever
    // instances really are the same mesh at different transforms — it is a
    // scene-graph object with matrices you can set, raycast against, and pick an
    // instance out of. Reach for this one when the per-instance data is not a
    // transform.
    class InstancedBufferGeometry: public BufferGeometry {

    public:
        // How many times the vertices are drawn. Kept public and mutable: a
        // cloud that hides or streams its instances changes this per frame, and
        // there is nothing to invalidate when it does — the draw call reads it
        // directly.
        size_t instanceCount = 0;

        explicit InstancedBufferGeometry(size_t instanceCount = 0)
            : instanceCount(instanceCount) {}

        [[nodiscard]] std::string type() const override {

            return "InstancedBufferGeometry";
        }

        static std::shared_ptr<InstancedBufferGeometry> create(size_t instanceCount = 0) {

            return std::make_shared<InstancedBufferGeometry>(instanceCount);
        }
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDBUFFERGEOMETRY_HPP
