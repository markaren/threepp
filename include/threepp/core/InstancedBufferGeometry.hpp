// https://github.com/mrdoob/three.js/blob/r129/src/core/InstancedBufferGeometry.js

#ifndef THREEPP_INSTANCEDBUFFERGEOMETRY_HPP
#define THREEPP_INSTANCEDBUFFERGEOMETRY_HPP

#include "BufferGeometry.hpp"

#include "threepp//math/infinity.hpp"

namespace threepp {

    class InstancedBufferGeometry: public BufferGeometry {

    public:
        int instanceCount = Infinity<int>;
        int _maxInstanceCount;

        static std::shared_ptr<InstancedBufferGeometry> create() {

            return std::shared_ptr<InstancedBufferGeometry>(new InstancedBufferGeometry());
        }

    protected:
        InstancedBufferGeometry(): BufferGeometry() {}
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDBUFFERGEOMETRY_HPP
