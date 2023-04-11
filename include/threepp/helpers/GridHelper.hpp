// https://github.com/mrdoob/three.js/blob/r129/src/helpers/GridHelper.js

#ifndef THREEPP_GRIDHELPER_HPP
#define THREEPP_GRIDHELPER_HPP

#include "threepp/objects/LineSegments.hpp"

#include <memory>

namespace threepp {

    class GridHelper: public LineSegments {

    public:
        ~GridHelper() override;

        static std::shared_ptr<GridHelper> create(
                unsigned int size = 10,
                unsigned int divisions = 10,
                const Color& color1 = 0x444444,
                const Color& color2 = 0x888888);

    protected:
        GridHelper(unsigned int size, unsigned int divisions, const Color& color1, const Color& color2);
    };

}// namespace threepp

#endif//THREEPP_GRIDHELPER_HPP
