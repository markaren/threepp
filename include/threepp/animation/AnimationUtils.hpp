
#ifndef THREEPP_ANIMATIONUTILS_HPP
#define THREEPP_ANIMATIONUTILS_HPP

#include <vector>

namespace threepp::AnimationUtils {

    inline std::vector<float> arraySlice(const std::vector<float>& array, int from = 0, int to = 0) {

        if (from == 0 && to == 0) return array;

        return {array.begin() + from, array.begin() + to};
    }

}// namespace threepp::AnimationUtils

#endif//THREEPP_ANIMATIONUTILS_HPP
