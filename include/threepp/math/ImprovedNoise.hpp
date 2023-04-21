//

#ifndef THREEPP_IMPROVEDNOISE_HPP
#define THREEPP_IMPROVEDNOISE_HPP

namespace threepp::math {

    class ImprovedNoise {

    public:
        [[nodiscard]] float noise(float x, float y, float z) const;
    };

}// namespace threepp::math

#endif//THREEPP_IMPROVEDNOISE_HPP
