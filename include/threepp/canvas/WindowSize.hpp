
#ifndef THREEPP_WINDOWSIZE_HPP
#define THREEPP_WINDOWSIZE_HPP

namespace threepp {

    struct WindowSize {

        int width{};
        int height{};

        WindowSize() = default;

        WindowSize(int width, int height): width(width), height(height) {}

        [[nodiscard]] float aspect() const {

            return static_cast<float>(width) / static_cast<float>(height);
        }
    };

}// namespace threepp

#endif//THREEPP_WINDOWSIZE_HPP
