
#ifndef THREEPP_WINDOWSIZE_HPP
#define THREEPP_WINDOWSIZE_HPP

namespace threepp {

    struct WindowSize {
        int width;
        int height;

        [[nodiscard]] float getAspect() const {

            return static_cast<float>(width) / static_cast<float>(height);
        }
    };

}

#endif//THREEPP_WINDOWSIZE_HPP