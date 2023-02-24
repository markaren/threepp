
#ifndef THREEPP_INFINITY_HPP
#define THREEPP_INFINITY_HPP

#include <limits>

namespace threepp {

    template<typename T>
    constexpr T Infinity = std::numeric_limits<T>::infinity();

}

#endif//THREEPP_INFINITY_HPP
