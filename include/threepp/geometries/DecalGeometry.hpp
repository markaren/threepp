

#ifndef THREEPP_DECALGEOMETRY_HPP
#define THREEPP_DECALGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class DecalGeometry: public BufferGeometry {

    public:

    static std::shared_ptr<DecalGeometry> create(const Mesh& mesh, const Vector3& position, const Euler& orientation, const Vector3& size) {

        return std::shared_ptr<DecalGeometry>(new DecalGeometry(mesh, position, orientation, size));
    }


    private:
        DecalGeometry(const Mesh& mesh, const Vector3& position, const Euler& orientation, const Vector3& size);
    };

}

#endif//THREEPP_DECALGEOMETRY_HPP
