
#ifndef THREEPP_WINDOWSIZE_HPP
#define THREEPP_WINDOWSIZE_HPP

#include <utility>
#include <ostream>

namespace threepp {

    struct WindowSize: std::pair<int, int> {

        WindowSize() = default;

        WindowSize(int width, int height): pair(width, height) {}

        WindowSize(std::pair<int, int> size): pair(size) {}

        [[nodiscard]] int width() const { return first; }

        [[nodiscard]] int height() const { return second; }

        [[nodiscard]] float aspect() const {

            return static_cast<float>(first) / static_cast<float>(second);
        }

        friend std::ostream& operator<<(std::ostream& os, const WindowSize& size) {

            os << "WindowSize(" << size.first << ", " << size.second << ")";
            return os;
        }
    };

}// namespace threepp

#endif//THREEPP_WINDOWSIZE_HPP
