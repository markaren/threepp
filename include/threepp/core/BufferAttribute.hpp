// https://github.com/mrdoob/three.js/blob/r129/src/core/BufferAttribute.js

#ifndef THREEPP_BUFFER_ATTRIBUTE_HPP
#define THREEPP_BUFFER_ATTRIBUTE_HPP

#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

namespace threepp {

    class buffer_attribute {

    private:

        static vector3 _vector;
        static vector2 _vector2;

    };

    vector3 buffer_attribute::_vector = vector3();
    vector2 buffer_attribute::_vector2 = vector2();

}

#endif //THREEPP_BUFFER_ATTRIBUTE_HPP
