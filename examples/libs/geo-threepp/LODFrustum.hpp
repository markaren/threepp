
#ifndef THREEPP_LODFRUSTUM_HPP
#define THREEPP_LODFRUSTUM_HPP

#include "threepp/math/Frustum.hpp"
#include "threepp/math/Matrix4.hpp"

namespace threepp {

    class LODFrustum {

    private:
        Matrix4 projection;
        Vector3 pov;
        Frustum frustum;
        Vector3 position;
    };

}// namespace threepp

#endif//THREEPP_LODFRUSTUM_HPP
