
#ifndef THREEPP_MAPPROVIDER_HPP
#define THREEPP_MAPPROVIDER_HPP

#include <string>
#include <optional>

#include "threepp/textures/Image.hpp"

namespace threepp {

    class MapProvider {

    public:

        float minZoom = 0;
        float maxZoom = 20;

        virtual std::string name() const = 0;

        virtual Image fetchTile(float zoom, float x, float y) = 0;

    };

}

#endif//THREEPP_MAPPROVIDER_HPP
