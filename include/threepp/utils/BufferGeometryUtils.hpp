// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/utils/BufferGeometryUtils.js

#ifndef THREEPP_BUFFERGEOMETRYUTILS_HPP
#define THREEPP_BUFFERGEOMETRYUTILS_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <vector>

namespace threepp {

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<BufferGeometry*>& geometries, bool useGroups = false);

    std::shared_ptr<BufferGeometry> mergeBufferGeometries(const std::vector<std::shared_ptr<BufferGeometry>>& geometries, bool useGroups = false);


}// namespace threepp

#endif//THREEPP_BUFFERGEOMETRYUTILS_HPP
