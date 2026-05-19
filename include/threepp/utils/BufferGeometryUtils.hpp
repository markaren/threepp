// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/utils/BufferGeometryUtils.js

#ifndef THREEPP_BUFFERGEOMETRYUTILS_HPP
#define THREEPP_BUFFERGEOMETRYUTILS_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <vector>

namespace threepp {

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<BufferGeometry*>& geometries, bool useGroups = false);

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<std::shared_ptr<BufferGeometry>>& geometries, bool useGroups = false);

    std::shared_ptr<BufferGeometry> mergeVertices(const BufferGeometry& geometry, float tolerance = 1e-4f);

#ifdef THREEPP_WITH_MESHOPT
    std::shared_ptr<BufferGeometry> simplifyGeometry(const BufferGeometry& geometry, float ratio, float error = 1e-2f);
#endif

}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRYUTILS_HPP
