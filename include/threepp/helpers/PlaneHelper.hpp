// https://github.com/mrdoob/three.js/blob/r129/src/helpers/PlaneHelper.js

#ifndef THREEPP_PLANEHELPER_HPP
#define THREEPP_PLANEHELPER_HPP

#include "threepp/objects/Line.hpp"

namespace threepp {

    class Mesh;

    class PlaneHelper: public Line {

    public:
        Plane plane;
        float size;

        void updateMatrixWorld(bool force) override;

        static std::shared_ptr<PlaneHelper> create(const Plane& plane, float size = 1, const Color& color = 0xffff00);

    protected:
        PlaneHelper(const Plane& plane, float size, const Color& color);

    private:
        std::shared_ptr<Mesh> mesh_;
    };

}// namespace threepp

#endif//THREEPP_PLANEHELPER_HPP
